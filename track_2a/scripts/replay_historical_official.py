"""Export saved historical predictions and score them with the exact official evaluator.

No inference calls. Use only pinned v1.0 historical-test cases; verify the
population/claims/languages/gold labels before using historical row IDs.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference import EvidenceSource, PredictionResult
from scripts.audit_competition_baseline import FILES


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.cases_dir / 'manifest.json').read_text())
    if manifest['dataset'] != 'v1.0' or manifest['scope'] != 'historical-test':
        raise ValueError('Historical exports require the matching v1.0 test population')
    cases = {r['id']: r for r in read_jsonl(args.cases_dir / 'cases.jsonl')}
    expected = {r['id']: r for r in read_jsonl(args.cases_dir / 'expected-labels.jsonl')}
    exported, provenance = [], {}
    for task, filename in FILES.items():
        path = ROOT / 'results' / filename
        raw = path.read_bytes()
        archive = json.loads(raw)
        provenance[task] = {'file': filename, 'sha256': hashlib.sha256(raw).hexdigest(),
                            'metadata': archive['meta']}
        for row in archive['results']:
            cid = f"v1.0-row-{int(row['id'].split('-')[1])}-{task}"
            case = cases[cid]
            source = case.get('reference', case.get('booklet'))
            if (row['claim'] != case['claim']['text'] or row['vote'] != case['vote']
                    or row['pair'] != f"{case['claim']['language']}->{source['language']}"
                    or row['true_label'] != expected[cid]['label']):
                raise ValueError(f'Historical case identity mismatch: {cid}')
            if len(row['evidence']) != len(row['evidence_pages']):
                raise ValueError(f'Historical evidence metadata mismatch: {cid}')
            result = PredictionResult(
                id=cid, claim=row['claim'], label=row['pred_label'],
                label_name=['Entailment', 'Neutral', 'Contradiction'][row['pred_label']],
                reasoning=row['reasoning'], evidence=row['evidence'],
                evidence_sources=[EvidenceSource(quote=q, page_number=p)
                                  for q, p in zip(row['evidence'], row['evidence_pages'])],
                strategy=archive['meta']['strategy'], booklet_path='historical_archive',
                tokens_prompt=row['tokens_prompt'], tokens_completion=row['tokens_completion'],
                tokens_total=row['tokens_prompt'] + row['tokens_completion'], latency_ms=row['latency_ms'],
                error=row.get('error'),
            )
            exported.append(result.to_official_dict())
    if {r['id'] for r in exported} != set(cases) or len(exported) != len(cases):
        raise ValueError('Historical prediction and case populations differ')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions = args.cases_dir / 'historical-predictions.jsonl'
    predictions.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in exported))
    subprocess.run([sys.executable, '-m', 'vendor.hackapertus_starter.evaluate',
                    '--predictions', str(predictions.resolve()),
                    '--expected', str((args.cases_dir / 'expected-labels.jsonl').resolve()),
                    '--cases', str((args.cases_dir / 'cases.jsonl').resolve()),
                    '--json', str(args.output.resolve())], cwd=ROOT, check=True)
    report = json.loads(args.output.read_text())
    report['provenance'] = {'measurement_type': 'historical_official_export_replay',
                            'fresh_api_run': False, 'archives': provenance,
                            'dataset': manifest,
                            'official_evaluator': json.loads((ROOT / 'vendor' / 'hackapertus_starter' / 'provenance.json').read_text())}
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
