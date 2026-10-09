"""Prepare pinned public cases, separate gold labels and date-grouped folds offline.

This script is development tooling, never called by the prediction entrypoint.
Source labels/reference gold are not included in Task A inference inputs.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.hf_dataset import split_dates

SOURCES = {
    "v1.0": {
        "revision": "adca9a5ee87e9813cef80b305c3b12e3e8a05959",
        "sha256": "f13c50392bd36d744a304fdacb1e61d53ab5eab5c9062f2a5182a175814fb88c",
        "rows": 1495,
    },
    "v1.1": {
        "revision": "9ff08597fb79dc68cbb3af9eb1388f34d21223e6",
        "sha256": "87ee4afd9c0ee8861106c50f05884481402ca668aecf927df68fc52d915b02fd",
        "rows": 1488,
    },
}


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def download(url):
    request = urllib.request.Request(url, headers={"User-Agent": "FactAttack-reproducible-preparation/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def build_folds(rows, n_folds=5, seed=42):
    dates = sorted({str(row['booklet_publish_date'])[:10] for row in rows})
    if not 2 <= n_folds <= len(dates):
        raise ValueError('Fold count must be between two and the number of voting dates')
    random.Random(seed).shuffle(dates)
    return {date: i % n_folds for i, date in enumerate(dates)}


def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=SOURCES, default='v1.1')
    parser.add_argument('--scope', choices=['all', 'historical-test'], default='all')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--download-booklets', action='store_true')
    parser.add_argument('--reuse-booklets', type=Path)
    parser.add_argument('--folds', type=int, default=5)
    args = parser.parse_args()
    source = SOURCES[args.version]
    url = (f'https://huggingface.co/datasets/OSTswiss/MNLIoverSwissVotingBooklets/resolve/'
           f"{source['revision']}/{args.version}.jsonl")
    cache = ROOT / '.cache' / 'competition' / f'ost_{args.version}.jsonl'
    if not cache.exists():
        raw = download(url)
        if digest(raw) != source['sha256']:
            raise ValueError('Downloaded dataset does not match the pinned source')
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(raw)
    raw = cache.read_bytes()
    if digest(raw) != source['sha256']:
        raise ValueError('Cached dataset does not match the pinned source')
    rows = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
    if len(rows) != source['rows']:
        raise ValueError('Incorrect pinned dataset population')
    fold_by_date = build_folds(rows, args.folds)
    selected_dates = set(split_dates([row['booklet_publish_date'] for row in rows]))
    selected = [(i, row) for i, row in enumerate(rows)
                if args.scope == 'all' or row['booklet_publish_date'] in selected_dates]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases, expected, assignments = [], [], []
    pdfs = {}
    for i, row in selected:
        date, lang = row['booklet_publish_date'], row['reference_language']
        filename = f'{date}_{lang}.pdf'
        pdfs[filename] = row['booklet_url']
        for task in ('A', 'B'):
            cid = f'{args.version}-row-{i}-{task}'
            case = {'id': cid, 'vote': row['vote'],
                    'claim': {'text': row['claim'], 'language': row['claim_language']}}
            if task == 'A':
                case['booklet'] = {'path': f'booklets/{filename}', 'language': lang}
            else:
                case['reference'] = {'text': row['reference_string'], 'language': lang}
            gold = {'id': cid, 'label': row['entailment_label']}
            if row['entailment_label'] != 1:
                gold['reference'] = row['reference_string']
            cases.append(case)
            expected.append(gold)
            assignments.append({'id': cid, 'row': i, 'date': date, 'fold': fold_by_date[date],
                                'vote': row['vote'], 'source_language': lang,
                                'claim_language': row['claim_language'], 'task': task})
    booklet_hashes = {}
    if args.download_booklets:
        (output / 'booklets').mkdir(exist_ok=True)
        for filename, pdf_url in sorted(pdfs.items()):
            dest = output / 'booklets' / filename
            existing = args.reuse_booklets / filename if args.reuse_booklets else None
            pdf = dest.read_bytes() if dest.exists() else (
                existing.read_bytes() if existing and existing.exists() else download(pdf_url))
            if not pdf.startswith(b'%PDF'):
                raise ValueError(f'Invalid source PDF: {filename}')
            if not dest.exists():
                dest.write_bytes(pdf)
            booklet_hashes[filename] = {'sha256': digest(pdf), 'url': pdf_url}
            print(f'PDF ready: {filename}', flush=True)
    write_jsonl(output / 'cases.jsonl', cases)
    write_jsonl(output / 'expected-labels.jsonl', expected)
    write_jsonl(output / 'folds.jsonl', assignments)
    manifest = {'dataset': args.version, **source, 'url': url, 'scope': args.scope,
                'case_count': len(cases), 'rows_selected': len(selected), 'fold_count': args.folds,
                'fold_by_date': fold_by_date, 'booklets': booklet_hashes,
                'historical_test_is_observed': True, 'untouched_holdout_available': False,
                'file_hashes': {name: digest((output / name).read_bytes())
                                for name in ['cases.jsonl', 'expected-labels.jsonl', 'folds.jsonl']}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Prepared {len(selected)} rows / {len(cases)} mixed cases at {output}')


if __name__ == '__main__':
    main()
