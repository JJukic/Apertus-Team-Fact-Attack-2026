"""Run local E5/BM25 comparison on all dev rows. No Apertus/API inference."""
import hashlib
import json
import sys
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config
from src.apertus_client import ApertusClient
from src.inference import ClaimVerificationEngine, attributed_section, opposing_sections
from src.local_dense import LocalE5, MODEL, REVISION, WINDOW, STRIDE, page_scores, ranking, minmax, fuse, select
from src.text_utils import clip_to_query
from measure_retrieval import anchors


def main():
    if config.PASSAGE_CHARS:
        raise ValueError('This page benchmark requires PASSAGE_CHARS=0')
    started = time.perf_counter()
    path = config.DATA_DIR / 'hf/dev.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    data_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    cache = config.BASE_DIR / '.cache/e5'
    t = time.perf_counter()
    model = LocalE5(cache)
    load_s = time.perf_counter() - t
    print(f'Model loaded on {model.device} in {load_s:.1f}s', flush=True)
    queries = list(dict.fromkeys(text for r in rows for text in (r['claim'], r['vote'] or '') if text))
    query_sha = hashlib.sha256(json.dumps([REVISION, queries], ensure_ascii=False).encode()).hexdigest()
    query_cache = cache / f'queries_{query_sha}.npz'
    t = time.perf_counter()
    if query_cache.exists():
        with np.load(query_cache, allow_pickle=False) as saved:
            query_vectors = saved['vectors']
        query_status = 'disk'
    else:
        query_vectors = model.encode(queries, 'query')
        np.savez_compressed(query_cache, vectors=query_vectors)
        query_status = 'built'
    query_s = time.perf_counter() - t
    print(f'{len(queries)} query vectors ready in {query_s:.1f}s ({query_status})', flush=True)
    query_by_text = {text: vec for text, vec in zip(queries, query_vectors)}
    engine = ClaimVerificationEngine(apertus_client=ApertusClient(mock=True))
    documents, hashes, cases, excluded = {}, {}, [], []
    variants = ['bm25_title', 'dense_claim', 'dense_title', 'rrf']
    ks = [5, 12, 20]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['booklet_file']].append(row)
    document_stats = []
    for booklet, booklet_rows in sorted(grouped.items()):
        pdf = config.BOOKLETS_DIR / booklet
        if not pdf.exists():
            raise FileNotFoundError(pdf)
        hashes[booklet] = hashlib.sha256(pdf.read_bytes()).hexdigest()
        data = engine._get_booklet_data(pdf)
        paras = data['paragraphs']
        if len({p['page_number'] for p in paras}) != len(paras):
            raise ValueError('Expected one passage per page')
        t = time.perf_counter()
        vectors, owners, status = model.document(paras)
        index_s = time.perf_counter() - t
        all_anchors = set().union(*(anchors(p['text']) for p in data['pages']))
        number_to_index = {p['page_number']: i for i, p in enumerate(paras)}
        document_stats.append({'file': booklet, 'chunks': len(owners), 'pages': len(paras), 'index_s': index_s, 'cache': status})
        for row in booklet_rows:
            located = anchors(row['reference_string']) & all_anchors
            if not located:
                excluded.append(row['id'])
                continue
            lexical = data['retriever'].retrieve_hybrid(row['claim'], top_k=len(paras), target_vote=row['vote'])
            lexical_order = [number_to_index[p['page_number']] for p in lexical]
            claim_scores = page_scores(vectors @ query_by_text[row['claim']], owners, len(paras))
            title_scores = page_scores(vectors @ query_by_text[row['vote']], owners, len(paras)) if row['vote'] else np.zeros(len(paras))
            dense_order = ranking(claim_scores)
            orders = {'bm25_title': lexical_order, 'dense_claim': dense_order,
                      'dense_title': ranking(minmax(claim_scores) + 2 * minmax(title_scores)),
                      'rrf': fuse([lexical_order, dense_order])}
            side = attributed_section(row['claim'])
            excluded_sections = opposing_sections(row['claim']) if config.SPEAKER_AWARE else None
            case = {'id': row['id'], 'pair': f"{row['claim_language']}->{row['reference_language']}", 'results': {}}
            for method in variants:
                for k in ks:
                    selected = select(orders[method], paras, k, side, config.SPEAKER_BOOST, excluded_sections)
                    query = row['claim'] + ' ' + (row['vote'] or '')
                    texts = [clip_to_query(p['text'], query, config.PAGE_MAX_CHARS) if config.PAGE_MAX_CHARS else p['text'] for p in selected]
                    found = located & set().union(*(anchors(text) for text in texts))
                    case['results'][f'{method}_k{k}'] = {'hit': bool(found), 'coverage': len(found)/len(located), 'pages': [p['page_number'] for p in selected]}
            cases.append(case)
        print(f'{booklet}: {len(paras)} pages, {len(owners)} chunks, index={index_s:.1f}s ({status}); {len(cases)}/{len(rows)} cases', flush=True)
    summaries = []
    for k in ks:
        for method in variants:
            key, baseline = f'{method}_k{k}', f'bm25_title_k{k}'
            by_pair = {}
            for pair in sorted({c['pair'] for c in cases}):
                group = [c for c in cases if c['pair'] == pair]
                by_pair[pair] = {'n': len(group), 'hit': sum(c['results'][key]['hit'] for c in group)/len(group), 'coverage': float(np.mean([c['results'][key]['coverage'] for c in group]))}
            summaries.append({'method': method, 'k': k, 'n': len(cases), 'hit': sum(c['results'][key]['hit'] for c in cases)/len(cases),
                              'coverage': float(np.mean([c['results'][key]['coverage'] for c in cases])),
                              'fixed_hits': sum(c['results'][key]['hit'] and not c['results'][baseline]['hit'] for c in cases),
                              'lost_hits': sum(c['results'][baseline]['hit'] and not c['results'][key]['hit'] for c in cases), 'by_pair': by_pair})
    report = {'model': MODEL, 'revision': REVISION, 'device': model.device, 'dataset_sha256': data_sha,
              'dataset_n': len(rows), 'excluded_ids': excluded, 'booklet_sha256': hashes,
              'window': WINDOW, 'stride': STRIDE, 'speaker_boost': config.SPEAKER_BOOST,
              'speaker_aware': config.SPEAKER_AWARE, 'page_max_chars': config.PAGE_MAX_CHARS,
              'ngram_words': 12, 'rrf_constant': 60, 'vote_weight': 2,
              'runtime': {p: version(p) for p in ['torch', 'sentence-transformers', 'transformers', 'pypdf', 'fonttools']},
              'timing': {'model_load_s': load_s, 'all_query_embedding_s': query_s, 'query_cache': query_status, 'wall_s': time.perf_counter()-started},
              'documents': document_stats, 'summaries': summaries, 'cases': cases}
    output = config.BASE_DIR / 'docs/local_dense_comparison.json'
    header = json.dumps({key: value for key, value in report.items() if key != 'cases'}, ensure_ascii=False, indent=2)
    output.write_text(header[:-2] + ',\n  \"cases\": [\n' + ',\n'.join('    ' + json.dumps(case, ensure_ascii=False, separators=(',', ':')) for case in cases) + '\n  ]\n}\n')
    lines = ['# Local multilingual retrieval comparison', '',
             f'Model: [{MODEL}](https://huggingface.co/{MODEL}), immutable revision `{REVISION}`, device `{model.device}`.',
             f'Dev rows: {len(rows)}; eligible: {len(cases)}; excluded unlocated references: {len(excluded)}.',
             f'Dataset SHA256: `{data_sha}`.', '',
             'All methods use identical parsed pages, page clipping and speaker boost. bm25_title is the current BM25(claim) + 2 × BM25(vote title) baseline. dense_claim uses only semantic claim similarity. dense_title combines independently min-max-normalized semantic claim/title page scores with title weight 2. RRF combines complete BM25/title and dense-claim rankings with constant 60. None uses gold text for ranking.',
             'E5 receives query:/passage: prefixes. Long pages use overlapping 384-token windows, stride 256, within the 512-token model limit. Page score is maximum chunk cosine similarity; more chunks can still create a multiple-comparisons advantage. Queries are truncated at the model limit if exceptionally long.', '',
             'Hit = at least one shared normalized 12-word reference fragment after clipping. Coverage = fraction of reference fragments located in the full booklet that are also selected. This lexical proxy does not measure semantic evidence validity or NLI accuracy; repeated boilerplate can inflate it.', '',
             '| Method | k | Reference-fragment hit | Mean anchor coverage | Hits gained vs BM25 | Hits lost vs BM25 |',
             '|---|---|---|---|---|---|']
    for s in summaries:
        lines.append(f'| {s["method"]} | {s["k"]} | {s["hit"]:.1%} | {s["coverage"]:.1%} | {s["fixed_hits"]} | {s["lost_hits"]} |')
    lines += ['', 'The current public dataset differs from the historical 1,495-case dataset. No comparison to historical F1 is valid. This is not BGE-M3 or a replay of the separate dense branch: only general max-per-page and RRF ideas are shared; the model, chunking and baseline differ.',
              f'This run: model load {load_s:.1f}s; all query embeddings {query_s:.1f}s ({query_status}); wall {report["timing"]["wall_s"]:.1f}s. Document indexing times/cache states are in JSON. Batched timings are not per-request app latency.', '',
              'Reproduce: install `track_2a/requirements-dense-experiment.txt`, then run `.venv/bin/python track_2a/scripts/compare_local_dense.py`. Requires downloaded dev data/PDFs, no CSCS key. Heavy weights/vectors are ignored under `.cache/e5`. Production retrieval remains unchanged.', '']
    (config.BASE_DIR/'docs/local_dense_comparison.md').write_text('\n'.join(lines))
    print('\n'.join(lines[12:]), flush=True)

if __name__ == '__main__':
    main()
