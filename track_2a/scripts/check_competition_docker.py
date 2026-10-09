"""Check an existing CPU image using source-only mixed cases and runtime keys.

Choose the small contract fixture before evaluating labels. No gold files or
credentials are copied into /data or the image. Real calls are optional and all
their attempts are journalled. Never builds, pushes or publishes an image.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.validate_submission_sources import validate


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True, help='Small mixed source-only JSONL fixture')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--image', default='fact-attack:test')
    parser.add_argument('--env-file', type=Path, help='Host-only credentials file; values never enter command arguments')
    parser.add_argument('--api-mode', action='append', default=[], choices=['original', 'translated', 'union'])
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    if not cases or not any('booklet' in case for case in cases) or not any('reference' in case for case in cases):
        raise ValueError('Contract fixture must contain both tasks')
    if any(set(case) & {'label', 'gold_label', 'entailment_label', 'expected_label'} for case in cases):
        raise ValueError('Contract fixture must contain no gold labels')
    image = json.loads(subprocess.check_output(['docker', 'image', 'inspect', args.image], text=True))[0]
    if (image['Os'], image['Architecture']) != ('linux', 'amd64'):
        raise ValueError('Image must be CPU linux/amd64')
    # A tag can move during a build. Keep every check on the inspected image.
    args.image = image['Id']
    sensitive_env = {'BASE_URL', 'API_KEY', 'LLM_BASE_URL', 'LLM_API_KEY'}
    if any(key in sensitive_env and value for key, value in
           (item.split('=', 1) for item in image['Config'].get('Env', []))):
        raise ValueError('Image contains a runtime credential or endpoint')
    inspection = """import hashlib,json
from pathlib import Path
root=Path('/app')
for p in root.rglob('*'):
 if p.is_file() and (p.name.startswith('.env') or p.name.startswith('expected-labels') or p.suffix in ('.safetensors','.pt')):
  raise ValueError('Forbidden image artifact: '+str(p))
demo=[json.loads(line) for line in (root/'data/demo_dataset.jsonl').read_text().splitlines() if line.strip()]
assert not any(set(row)&{'entailment_label','gold_label','expected_label','label'} for row in demo)
print(json.dumps({'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'src').glob('*.py')},'demo_rows_without_gold':len(demo),'forbidden_artifacts':0}))
"""
    image_check = json.loads(subprocess.check_output(
        ['docker', 'run', '--rm', '--platform', 'linux/amd64', '--network', 'none', '--read-only',
         '--tmpfs', '/tmp', args.image, 'python', '-c', inspection], text=True))
    for filename, sha in image_check['source_sha256'].items():
        if digest(ROOT / filename) != sha:
            raise ValueError('Image source differs from current checked source: ' + filename)
    data, output = args.output / 'data', args.output / 'output'
    data.mkdir(); output.mkdir()
    for case in cases:
        if 'booklet' in case:
            relative = Path(case['booklet']['path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Fixture booklet paths must be relative')
            target = data / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.cases.parent / relative, target)
    for name, rows in [('cases.jsonl', cases), ('cases.reverse.jsonl', list(reversed(cases)))]:
        (data / name).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    hashes = {str(path.relative_to(data)): digest(path) for path in data.rglob('*') if path.is_file()}
    environment = dict(os.environ)
    if args.env_file:
        from dotenv import dotenv_values
        values = dotenv_values(args.env_file)
        for official, legacy in [('BASE_URL', 'LLM_BASE_URL'), ('API_KEY', 'LLM_API_KEY')]:
            environment[official] = values.get(official) or values.get(legacy) or environment.get(official, '')
    if args.api_mode and not all(environment.get(key) for key in ('BASE_URL', 'API_KEY')):
        raise ValueError('Real checks require host BASE_URL and API_KEY')
    flags = {'NLI_STRATEGY': 'hybrid', 'NLI_TOP_K': '12', 'PROMPT_MODE': 'ids', 'PASSAGE_CHARS': '0',
             'PAGE_MAX_CHARS': '3000', 'SPEAKER_BOOST': '2', 'LLM_STREAMING': 'true', 'LLM_JSON_REPAIR': 'true',
             'EVIDENCE_POLICY': 'raw_pages_and_blocks', 'CACHE_SINGLE_FLIGHT': 'true', 'SOURCE_PAGES_LAZY': 'true',
             'THINKING': 'false', 'IDS_REASON': 'false', 'FEW_SHOT': 'false', 'SPEAKER_HINT': 'false',
             'SPEAKER_AWARE': 'false', 'NUMERIC_OVERRIDE': 'false', 'TRANSLATE_CLAIM': 'false',
             'REQUEST_JOURNAL_PATH': '/output/requests.api.jsonl'}
    runs = [('offline-forward', 'cases.jsonl', 'original', True),
            ('offline-reverse', 'cases.reverse.jsonl', 'original', True)]
    runs += [('real-' + mode, 'cases.jsonl', mode, False) for mode in dict.fromkeys(args.api_mode)]
    report = {'image': image['Id'], 'platform': 'linux/amd64', 'image_check': image_check,
              'cases': len(cases), 'gold_read': False, 'runs': {}, 'input_sha256': hashes}
    predictions = {}
    for name, filename, mode, mock in runs:
        command = ['docker', 'run', '--rm', '--platform', 'linux/amd64', '--read-only', '--tmpfs', '/tmp',
                   '-v', str(data) + ':/data:ro', '-v', str(output) + ':/output']
        for key, value in {**flags, 'RETRIEVAL_QUERY_MODE': mode, 'MOCK_APERTUS': str(mock).lower()}.items():
            command += ['-e', key + '=' + value]
        command += ['--network', 'none'] if mock else ['-e', 'BASE_URL', '-e', 'API_KEY']
        command += [args.image, '--input', '/data/' + filename, '--output', '/output/' + name + '.data']
        if mock:
            command += ['--mock']
        started = time.perf_counter()
        with (args.output / (name + '.log')).open('w') as log:
            result = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(name + ' failed; preserve its log and request journal')
        source_check = validate(data / filename, output / (name + '.data'))
        diagnostics = json.loads((output / (name + '.data.diagnostics.json')).read_text())
        if diagnostics['failures']:
            raise ValueError('CLI reported operational failures: ' + name)
        report['runs'][name] = {'exit_code': result.returncode, 'host_wall_seconds': time.perf_counter() - started,
                                'source_validation': source_check, 'diagnostics': diagnostics}
        rows = [json.loads(line) for line in (output / (name + '.data')).read_text().splitlines()]
        predictions[name] = {row['id']: {key: row[key] for key in ('label', 'label_name', 'evidence')} for row in rows}
    if predictions['offline-forward'] != predictions['offline-reverse']:
        raise ValueError('Deterministic inference changed with input order')
    if hashes != {str(path.relative_to(data)): digest(path) for path in data.rglob('*') if path.is_file()}:
        raise ValueError('Input files changed')
    journal = output / 'requests.api.jsonl'
    events = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    if any(event['output_token_cap'] is not None or 'Apertus-v1.5' not in event['model'] for event in events):
        raise ValueError('Requests violated model or uncapped response contract')
    report.update(read_only_inputs_unchanged=True, input_order_independent_with_deterministic_mock=True,
                  real_api_order_repeatability_not_assumed=True, api_attempts=len(events),
                  usage_unknown_attempts=sum(not event['usage_known'] for event in events),
                  input_tokens_known=sum(event['input_tokens'] for event in events if type(event['input_tokens']) is int),
                  output_tokens_known=sum(event['output_tokens'] for event in events if type(event['output_tokens']) is int),
                  request_journal_sha256=digest(journal) if journal.exists() else None,
                  validator_sha256=digest(Path(__file__)))
    (args.output / 'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print({key: report[key] for key in ('image', 'cases', 'api_attempts', 'usage_unknown_attempts', 'gold_read')})


if __name__ == '__main__':
    main()
