"""Offline reference-text availability proxy, not semantic evidence recall or NLI F1."""
import hashlib
import json
import re
import sys
from functools import lru_cache
from importlib.metadata import version
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.inference import ClaimVerificationEngine, attributed_section, opposing_sections
from src.apertus_client import ApertusClient
from src.text_utils import clip_to_query

BASE = config.BASE_DIR
NGRAM = 12


@lru_cache(maxsize=512)
def anchors(text):
    text = re.sub(r'([A-Za-zÀ-ÿ])\s*-\s+([A-Za-zÀ-ÿ])', r'\1\2', text.replace('\u00ad', ''))
    words = re.findall(r'\w+', text.casefold())
    return {tuple(words[i:i + NGRAM]) for i in range(len(words) - NGRAM + 1)}


def main():
    path = config.DATA_DIR / 'hf/dev.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    engine = ClaimVerificationEngine(apertus_client=ApertusClient(mock=True))
    variants = [(k, b) for k in (5, 12, 20) for b in (0, 2)]
    counts = {v: defaultdict(float) for v in variants}
    pairs = {v: defaultdict(lambda: defaultdict(int)) for v in variants}
    cases, missing, unmatched = [], [], []
    booklets = {}
    for number, row in enumerate(rows, 1):
        pdf = config.BOOKLETS_DIR / row['booklet_file']
        if not pdf.exists():
            missing.append(row['id'])
            continue
        if pdf.name not in booklets:
            data = engine._get_booklet_data(pdf)
            page_anchors = [anchors(p['text']) for p in data['pages']]
            booklets[pdf.name] = (data, set().union(*page_anchors))
        data, all_anchors = booklets[pdf.name]
        reference = anchors(row['reference_string'])
        located = reference & all_anchors
        if not located:
            unmatched.append(row['id'])
            continue
        case = {'id': row['id'], 'pair': f"{row['claim_language']}->{row['reference_language']}", 'variants': {}}
        side = attributed_section(row['claim'])
        for k, boost in variants:
            passages = data['retriever'].retrieve_hybrid(row['claim'], top_k=k, target_vote=row['vote'],
                exclude_sections=opposing_sections(row['claim']) if config.SPEAKER_AWARE else None,
                ensure_sections={side: boost} if boost and side else None)
            before = set().union(*(anchors(p['text']) for p in passages))
            query = row['claim'] + ' ' + (row['vote'] or '')
            texts = [clip_to_query(p['text'], query, config.PAGE_MAX_CHARS) if config.PAGE_MAX_CHARS else p['text'] for p in passages]
            after = set().union(*(anchors(text) for text in texts))
            hit_before, hit_after = bool(located & before), bool(located & after)
            c = counts[(k, boost)]
            c['eligible'] += 1
            c['before'] += hit_before
            c['after'] += hit_after
            c['coverage'] += len(located & after) / len(located)
            c['lost_to_clip'] += hit_before and not hit_after
            pair = pairs[(k, boost)][case['pair']]
            pair['eligible'] += 1
            pair['hit'] += hit_after
            case['variants'][f'k{k}_boost{boost}'] = {'pages': [p['page_number'] for p in passages], 'hit_after_clip': hit_after}
        cases.append(case)
        if number % 100 == 0:
            print(f'{number}/{len(rows)} processed', flush=True)
    summaries = []
    for (k, boost), c in counts.items():
        n = c['eligible']
        summaries.append({'k': k, 'speaker_boost': boost, 'eligible': int(n),
            'hit_before_clip': c['before']/n if n else None, 'hit_after_clip': c['after']/n if n else None,
            'mean_located_anchor_coverage': c['coverage']/n if n else None,
            'lost_to_clip': int(c['lost_to_clip']), 'by_pair': dict(pairs[(k, boost)])})
    result = {'runtime': {'pypdf': version('pypdf'), 'fonttools': version('fonttools')}, 'dataset_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'dataset_rows': len(rows), 'source_parquet_sha256': hashlib.sha256((path.parent/'train.parquet').read_bytes()).hexdigest(),
        'page_max_chars': config.PAGE_MAX_CHARS, 'speaker_aware': config.SPEAKER_AWARE,
        'ngram_words': NGRAM, 'missing_pdf_ids': missing, 'unmatched_reference_ids': unmatched,
        'booklet_sha256': {name: hashlib.sha256((config.BOOKLETS_DIR/name).read_bytes()).hexdigest() for name in booklets},
        'summaries': summaries, 'cases': cases}
    out = BASE / 'docs/retrieval_offline_measurement.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    lines = ['# Offline retrieval measurement', '',
        f'Dev cases: {len(rows)}. Missing PDFs: {len(missing)}. Reference not located: {len(unmatched)}.',
        f'Dataset SHA256: `{result["dataset_sha256"]}`.', '',
        'This is a lexical reference-availability proxy. A hit means at least one normalized, consecutive 12-word reference fragment occurs in selected text. It does not establish semantic support, full reference coverage, or correct classification. Boilerplate/repeated text can inflate hits. Unlocated references are excluded from rates and listed in JSON.',
        'Whitespace, case, soft hyphens and letter-to-letter PDF hyphenation are normalized. No model calls. Selected context is clipped with the same query and limit as the pipeline.', '',
        '| k | Speaker boost | Eligible | Hit before clipping | Hit after clipping | Mean anchor coverage | Cases lost to clipping |',
        '|---|---|---|---|---|---|---|']
    for s in summaries:
        lines.append(f'| {s["k"]} | {s["speaker_boost"]} | {s["eligible"]} | {s["hit_before_clip"]:.1%} | {s["hit_after_clip"]:.1%} | {s["mean_located_anchor_coverage"]:.1%} | {s["lost_to_clip"]} |')
    lines += ['', 'The public dataset now contains 1,488 total rows (dev 1,090/test 398), whereas historical benchmark documentation describes 1,495 (test 402). Do not compare these rates directly with historical F1. Case ids derived from row indices must be interpreted with this dataset hash.', '',
        'These methods share retrieval before extraction: this measurement does not distinguish baseline NLI from the two-stage judge. It tests how much source reference text is available upstream. A filter cannot recover omitted text.', '',
        'Reproduce from repository root: `.venv/bin/python track_2a/scripts/measure_retrieval.py`. Downloaded dev PDFs and splits are local, ignored data. JSON includes file hashes and per-case selected pages.', '']
    (BASE/'docs/retrieval_offline_measurement.md').write_text('\n'.join(lines))
    print('\n'.join(lines[-12:]), flush=True)

if __name__ == '__main__':
    main()
