"""Source-only CPU profiling; no model calls and no expected-label input.

These isolated component times are not the proxy's wall-minus-request-union
score. Cold extraction, cached parsing, ranking and evidence are measured
separately to identify costs hidden by concurrent remote requests.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import config
from src.competition_selection import distribution
from src.evidence import SourcePages, construct_evidence
from src.inference import attributed_section, load_parsed_booklet
from src.pdf_parser import PDFParser
from src.retriever import PassageRetriever
from src.text_utils import clip_to_query


def timed(operation):
    started = time.perf_counter()
    result = operation()
    return result, (time.perf_counter() - started) * 1000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    predictions = [json.loads(line) for line in args.predictions.read_text().splitlines() if line.strip()]
    if (len({r['id'] for r in cases}) != len(cases)
            or len({r['id'] for r in predictions}) != len(predictions)
            or {r['id'] for r in cases} != {r['id'] for r in predictions}):
        raise ValueError('Profiling populations must match exactly')
    args.output.mkdir(parents=True, exist_ok=False)
    config.BOOKLET_CACHE_DIR = args.output / 'source-cache'
    config.PASSAGE_CHARS = 0
    pdf_parser = PDFParser()
    retrievers, pdf_timings = {}, []
    start = time.perf_counter()
    for filename in sorted({case['booklet']['path'] for case in cases if 'booklet' in case}):
        path = args.cases.parent / filename
        pages, extraction = timed(lambda: pdf_parser.extract_pages(path))
        paragraphs, segmentation = timed(lambda: pdf_parser.extract_paragraphs(path, pages=pages, passage_chars=None))
        retriever, indexing = timed(lambda: PassageRetriever(paragraphs))
        cold, cold_cache = timed(lambda: load_parsed_booklet(path, pdf_parser))
        warm, warm_cache = timed(lambda: load_parsed_booklet(path, pdf_parser))
        if cold != warm or cold['pages'] != pages or cold['paragraphs'] != paragraphs:
            raise AssertionError('Cold/cached source parsing differs: ' + filename)
        retrievers[filename] = retriever
        pdf_timings.append({'path': filename, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'pages': len(pages), 'extraction_ms': extraction,
                            'segmentation_ms': segmentation, 'bm25_indexing_ms': indexing,
                            'cold_parse_and_cache_write_ms': cold_cache,
                            'cached_parse_load_ms': warm_cache})
    ranking, clipping = [], []
    for case in cases:
        if 'booklet' not in case:
            continue
        side = attributed_section(case['claim']['text'])
        passages, rank = timed(lambda: retrievers[case['booklet']['path']].retrieve_hybrid(
            case['claim']['text'], top_k=12, target_vote=case.get('vote'),
            ensure_sections={side: 2} if side else None))
        _, clip = timed(lambda: [clip_to_query(p['text'], case['claim']['text'] + ' ' + case.get('vote', ''), 3000)
                                for p in passages])
        ranking.append(rank)
        clipping.append(clip)
    del retrievers
    by_id = {r['id']: r for r in cases}
    evidence, reference_output_hash = {}, None
    for lazy in (False, True):
        pages = SourcePages(args.cases.parent, lazy=lazy)
        times = []
        digest = hashlib.sha256()
        for prediction in predictions:
            candidate, milliseconds = timed(lambda: construct_evidence(
                prediction, by_id[prediction['id']], pages, 'raw_pages_and_blocks'))
            times.append(milliseconds)
            digest.update(json.dumps(candidate, sort_keys=True, ensure_ascii=False).encode())
            digest.update(b'\n')
        output_hash = digest.hexdigest()
        if reference_output_hash is not None and reference_output_hash != output_hash:
            raise AssertionError('Lazy extraction changes submitted predictions')
        reference_output_hash = output_hash
        evidence['lazy' if lazy else 'eager'] = {**distribution(times), 'prediction_sha256': output_hash}
        del pages
    columns = ('extraction_ms', 'segmentation_ms', 'bm25_indexing_ms',
               'cold_parse_and_cache_write_ms', 'cached_parse_load_ms')
    report = {'measurement_scope': 'isolated source-only CPU components, sequential, local; not official proxy efficiency',
              'remote_api_calls': 0, 'gold_read': False, 'python': platform.python_version(),
              'cases': len(cases), 'advanced_cases': len(ranking), 'booklets': len(pdf_timings),
              'cases_sha256': hashlib.sha256(args.cases.read_bytes()).hexdigest(),
              'predictions_sha256': hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'src').glob('*.py')},
              'pdf_components_ms': {name: distribution([row[name] for row in pdf_timings]) for name in columns},
              'ranking_ms_per_advanced_case': distribution(ranking),
              'clipping_ms_per_advanced_case': distribution(clipping),
              'evidence_ms_per_mixed_case': evidence,
              'pdfs': pdf_timings, 'wall_seconds': time.perf_counter() - start,
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024)}
    (args.output / 'profile.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print({'booklets': len(pdf_timings), 'cases': len(cases), 'wall_seconds': report['wall_seconds'],
           'byte_equivalent_evidence': True, 'peak_rss_bytes': report['peak_rss_bytes']}, flush=True)


if __name__ == '__main__':
    main()
