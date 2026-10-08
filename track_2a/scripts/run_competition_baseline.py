"""Run a resumable, source-frozen pipeline on official mixed JSONL inputs.

Inference reads only cases.jsonl, never expected labels. Optional experimental
features are recorded in the run identity; the baseline remains selectable.
Output token caps are omitted, as explicitly requested by the user.
All API attempts are recorded; a failed case stays invalid during offline scoring.
"""

import time
INVOCATION_START = time.perf_counter()
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--env-file', type=Path, action='append', default=[])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--retry-errors', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error('workers and limit must be positive')
    from dotenv import load_dotenv
    endpoint_environment = {k: os.environ[k] for k in ('BASE_URL', 'LLM_BASE_URL', 'API_KEY', 'LLM_API_KEY') if k in os.environ}
    for path in args.env_file:
        load_dotenv(path, override=False)
    for official, legacy in (('BASE_URL', 'LLM_BASE_URL'), ('API_KEY', 'LLM_API_KEY')):
        if official in endpoint_environment or legacy in endpoint_environment:
            os.environ[official] = endpoint_environment.get(official, endpoint_environment.get(legacy))
    from src import config
    from src.apertus_client import ApertusClient
    from src.inference import ClaimVerificationEngine
    from src.measurement import RequestRecorder
    # Prevent SDK/server exception text from accidentally including credentials.
    logging.getLogger('src.apertus_client').setLevel(logging.CRITICAL)
    api_key = os.environ.get('API_KEY') or config.LLM_API_KEY
    endpoint = os.environ.get('BASE_URL') or config.LLM_BASE_URL
    if not api_key or 'Apertus-v1.5' not in config.LLM_NAME:
        raise ValueError('Apertus v1.5 API credentials and model are required; mock is not permitted')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    prediction_dir = output / 'predictions'
    prediction_dir.mkdir(exist_ok=True)
    raw = args.cases.read_bytes()
    cases = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    if args.limit:
        cases = cases[:args.limit]
    if not cases or len({r['id'] for r in cases}) != len(cases):
        raise ValueError('Input IDs must be unique and cases nonempty')
    for case in cases:
        if ('booklet' in case) == ('reference' in case) or any(
                key in case for key in ('label', 'entailment_label', 'true_label', 'expected')):
            raise ValueError('Only official Task A/B inference inputs are permitted')
    options = {name: getattr(config, name) for name in (
        'LLM_NAME', 'DEFAULT_STRATEGY', 'DEFAULT_TOP_K', 'PROMPT_MODE', 'PAGE_MAX_CHARS',
        'PASSAGE_CHARS', 'SPEAKER_BOOST', 'SPEAKER_HINT', 'SPEAKER_AWARE', 'FEW_SHOT',
        'THINKING', 'THINKING_BUDGET', 'IDS_REASON', 'TRANSLATE_CLAIM', 'NUMERIC_OVERRIDE',
        'LLM_MAX_RETRIES', 'LLM_RETRY_BUDGET_S', 'EVIDENCE_POLICY', 'CACHE_SINGLE_FLIGHT',
        'LLM_STREAMING', 'LLM_REQUEST_TIMEOUT_S', 'LLM_JSON_REPAIR', 'SOURCE_PAGES_LAZY', 'RETRIEVAL_QUERY_MODE')}
    options['output_token_cap'] = None
    options['parsed_cache_directory'] = str(config.BOOKLET_CACHE_DIR.resolve())
    options['query_translation_cache_directory'] = str(config.QUERY_TRANSLATION_CACHE_DIR.resolve())
    sources = {p.name: digest(p.read_bytes()) for p in (ROOT / 'src').glob('*.py')}
    booklet_paths = {case['booklet']['path'] for case in cases if 'booklet' in case}
    booklet_hashes = {path: digest((args.cases.parent / path).read_bytes())
                      for path in sorted(booklet_paths)}
    identity = {'source_hashes': sources, 'cases_sha256': digest(raw), 'case_count': len(cases),
                'options': options, 'workers': args.workers, 'booklet_sha256': booklet_hashes,
                'runner_sha256': digest(Path(__file__).read_bytes())}
    signature = digest(json.dumps(identity, sort_keys=True).encode())
    frozen = output / 'run.json'
    if frozen.exists():
        if json.loads(frozen.read_text())['signature'] != signature:
            raise ValueError('Run settings/sources/cases changed; use a new output directory')
    else:
        versions = {name: importlib.metadata.version(name) for name in (
            'openai', 'pydantic', 'pypdf', 'rank-bm25', 'scikit-learn', 'rapidfuzz', 'pymupdf')}
        snapshot = output / 'frozen_sources'
        snapshot.mkdir(exist_ok=True)
        for source in (ROOT / 'src').glob('*.py'):
            (snapshot / source.name).write_bytes(source.read_bytes())
        (snapshot / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        atomic_json(frozen, {**identity, 'signature': signature, 'commit': commit,
                            'python': platform.python_version(), 'dependencies': versions,
                            'mode': 'configured_pipeline_uncapped_transport',
                            'gold_read_during_inference': False})
    recorder = RequestRecorder(output / 'requests.api.jsonl', uuid4().hex)
    client = ApertusClient(base_url=endpoint, api_key=api_key, mock=False)
    if client.mock:
        raise ValueError('Real Apertus client initialization failed')
    # The transport boundary includes complete streaming consumption, so tail
    # usage and the full in-flight interval are recorded for every retry.
    client._request_completion = recorder.wrap(client._request_completion)
    engine = ClaimVerificationEngine(apertus_client=client)
    cpu_start = time.process_time()

    def path_for(cid):
        return prediction_dir / (digest(cid.encode()) + '.json')

    def predict(case):
        cid = case['id']
        start = time.perf_counter()
        with recorder.case(cid):
            try:
                if 'booklet' in case:
                    booklet = (args.cases.parent / case['booklet']['path']).resolve()
                    result = engine.verify_claim(claim=case['claim']['text'], booklet_pdf=booklet,
                                                 claim_language=case['claim']['language'],
                                                 booklet_language=case['booklet'].get('language'),
                                                 vote=case.get('vote'), case_id=cid)
                else:
                    result = engine.verify_premise(claim=case['claim']['text'],
                                                   reference=case['reference']['text'],
                                                   claim_language=case['claim']['language'], case_id=cid)
                prediction = result.to_official_dict(case_id=cid)
                raw_result = result.model_dump()
                error = bool(result.error)
                if error:
                    prediction.update(label=-1, label_name='invalid', evidence=[])
            except Exception as exception:
                error = True
                raw_result = {'error_type': type(exception).__name__}
                prediction = {'id': cid, 'label': -1, 'label_name': 'invalid', 'evidence': [],
                              'metrics': {'input_tokens': 0, 'output_tokens': 0, 'inference_time_ms': 0}}
        record = {'id': cid, 'prediction': prediction, 'raw_result': raw_result,
                  'operational_error': error, 'start': start, 'end': time.perf_counter(),
                  'session': recorder.session}
        atomic_json(path_for(cid), record)
        return record

    pending = []
    for case in cases:
        path = path_for(case['id'])
        if not path.exists() or (args.retry_errors and json.loads(path.read_text())['operational_error']):
            pending.append(case)
    print(f"Baseline: {len(cases)} cases; {len(pending)} pending; {args.workers} workers", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(predict, case): case['id'] for case in pending}
        for i, future in enumerate(as_completed(futures), 1):
            record = future.result()
            print(f"{i}/{len(pending)} {record['id']} {'ERROR' if record['operational_error'] else 'OK'}", flush=True)
    records = [json.loads(path_for(case['id']).read_text()) for case in cases]
    journal = output / 'requests.api.jsonl'
    events = [json.loads(line) for line in (journal.read_text() if journal.exists() else '').splitlines()]
    for record in records:
        own = [e for e in events if e['case_id'] == record['id']]
        metrics = record['prediction']['metrics']
        for key in ('input_tokens', 'output_tokens'):
            metrics[key] = sum(e[key] for e in own if type(e.get(key)) is int)
    (output / 'predictions.jsonl').write_text(''.join(
        json.dumps(record['prediction'], ensure_ascii=False) + '\n' for record in records))
    end = time.perf_counter()
    session = {'session': recorder.session, 'cases_executed': len(pending),
               'wall_seconds': end - INVOCATION_START,
               'non_llm_seconds': recorder.processing_seconds(INVOCATION_START, end),
               'cpu_seconds': time.process_time() - cpu_start,
               'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                                 (1 if sys.platform == 'darwin' else 1024),
               'timing_scope': 'local SDK intervals; includes startup and cold preprocessing'}
    with (output / 'sessions.jsonl').open('a') as out:
        out.write(json.dumps(session) + '\n')
    sessions = [json.loads(line) for line in (output / 'sessions.jsonl').read_text().splitlines()]
    summary = {'cases': len(cases), 'operational_errors': sum(r['operational_error'] for r in records),
               'api_attempts': len(events), 'usage_unknown_attempts': sum(not e['usage_known'] for e in events),
               'input_tokens_known': sum(e['input_tokens'] for e in events if type(e['input_tokens']) is int),
               'output_tokens_known': sum(e['output_tokens'] for e in events if type(e['output_tokens']) is int),
               'non_llm_seconds_all_sessions': sum(s['non_llm_seconds'] for s in sessions),
               'sessions': sessions, 'gold_read_during_inference': False}
    atomic_json(output / 'measurement.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'sessions'}), flush=True)


if __name__ == '__main__':
    main()
