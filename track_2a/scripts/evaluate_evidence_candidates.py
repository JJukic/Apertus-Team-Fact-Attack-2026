"""Offline evidence ablation on saved predictions; never change labels.

Candidate construction reads only the prediction, claim/vote and cited PDF pages.
Gold references are passed only to the official evaluator after outputs exist.
This experiment diagnoses source extraction independently of NLI quality.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.official_evaluation import normalize


from src.evidence import SourcePages, construct_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases-dir', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lazy-pages', action='store_true', help='Extract/cache only requested PDF pages')
    args = parser.parse_args()
    cases_file = args.cases_dir / 'cases.jsonl'
    cases = {r['id']: r for r in map(json.loads, cases_file.read_text().splitlines())}
    predictions = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    if len({r['id'] for r in predictions}) != len(predictions) or {r['id'] for r in predictions} != set(cases):
        raise ValueError('Case/prediction populations must match exactly')
    args.output.mkdir(parents=True, exist_ok=False)
    source_files = [Path(__file__), ROOT / 'src/evidence.py', ROOT / 'src/text_utils.py']
    frozen = args.output / 'frozen_sources'
    frozen.mkdir()
    source_hashes = {}
    for source in source_files:
        raw = source.read_bytes()
        (frozen / source.name).write_bytes(raw)
        source_hashes[str(source.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    report = {'measurement_type': 'offline_evidence_ablation_fixed_predictions',
              'input_sha256': hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
              'cases_sha256': hashlib.sha256(cases_file.read_bytes()).hexdigest(),
              'label_changes': 0, 'remote_api_calls': 0,
              'lazy_pages': args.lazy_pages,
              'construction_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'source_hashes': source_hashes,
              'official_evaluator': json.loads((ROOT / 'vendor/hackapertus_starter/provenance.json').read_text()),
              'dataset_manifest': json.loads((args.cases_dir / 'manifest.json').read_text()),
              'variants': {}}
    for policy in ('original', 'raw_pages', 'raw_pages_and_blocks'):
        pages = SourcePages(args.cases_dir, lazy=args.lazy_pages)
        output, timings = [], []
        for prediction in predictions:
            started = time.perf_counter()
            candidate = (prediction if policy == 'original' else
                         construct_evidence(prediction, cases[prediction['id']], pages, policy))
            timings.append((time.perf_counter() - started) * 1000)
            if candidate['label'] != prediction['label'] or candidate['id'] != prediction['id']:
                raise AssertionError('Evidence ablation changed case identity or NLI label')
            output.append(candidate)
        output_file = args.output / f'{policy}.predictions.jsonl'
        output_file.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in output))
        score_file = args.output / f'{policy}.score.json'
        completed = subprocess.run([
            sys.executable, '-m', 'vendor.hackapertus_starter.evaluate',
            '--predictions', str(output_file.resolve()),
            '--expected', str((args.cases_dir / 'expected-labels.jsonl').resolve()),
            '--cases', str(cases_file.resolve()), '--json', str(score_file.resolve()),
        ], cwd=ROOT, check=True, capture_output=True, text=True)
        (args.output / f'{policy}.score.log').write_text(completed.stdout)
        score = json.loads(score_file.read_text())
        missing = sum('booklet' in cases[r['id']] and r['label'] in (0, 2) and not r['evidence'] for r in output)
        report['variants'][policy] = {
            'tasks': score['tasks'], 'issues': score['issues'],
            'missing_required_evidence': missing,
            'evidence_component_ms_mean': statistics.mean(timings),
            'evidence_component_ms_median': statistics.median(timings),
            'evidence_component_ms_p95': sorted(timings)[round(0.95 * (len(timings) - 1))],
            'timing_scope': 'offline evidence construction only, includes lazy PDF extraction',
        }
        print(policy, 'A F1', score['tasks']['A']['macro_f1'],
              'Hit@5', score['tasks']['A'].get('evidence_score'), 'missing', missing, flush=True)
    report['extractor'] = {'name': 'pymupdf', 'version': __import__('pymupdf').VersionBind}
    (args.output / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
