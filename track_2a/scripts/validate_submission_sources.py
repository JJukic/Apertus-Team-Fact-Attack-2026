"""Validate saved JSONL against physical source PDFs, without loading gold.

This checks source provenance and the output contract, not whether a quotation
semantically supports the model's NLI decision. It makes no model calls.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evidence import SourcePages
from src.text_utils import official_normalize


def validate(cases_path, predictions_path):
    cases = [json.loads(line) for line in cases_path.read_text().splitlines() if line.strip()]
    predictions = [json.loads(line) for line in predictions_path.read_text().splitlines() if line.strip()]
    ids = [case['id'] for case in cases]
    if len(ids) != len(set(ids)) or [row['id'] for row in predictions] != ids:
        raise ValueError('Output IDs must match unique input IDs in order')
    if any(set(case) & {'label', 'entailment_label', 'gold_label', 'expected_label'} for case in cases):
        raise ValueError('Source validation input must contain no gold labels')
    pages = SourcePages(cases_path.parent, lazy=True)
    quote_count, advanced_required = 0, 0
    source_hashes = {}
    for case, prediction in zip(cases, predictions):
        cid = case['id']
        label = prediction['label']
        if type(label) is not int or label not in (0, 1, 2):
            raise ValueError('Invalid label: ' + str(cid))
        if prediction['label_name'] != ('entailment', 'neutral', 'contradiction')[label]:
            raise ValueError('Inconsistent label name: ' + str(cid))
        metrics = prediction['metrics']
        if set(metrics) != {'input_tokens', 'output_tokens', 'inference_time_ms'} or any(
                type(value) is not int or value < 0 for value in metrics.values()):
            raise ValueError('Invalid metric fields: ' + str(cid))
        evidence = prediction['evidence']
        if not isinstance(evidence, list) or len(evidence) > 5 or (label == 1 and evidence):
            raise ValueError('Invalid evidence list: ' + str(cid))
        advanced = 'booklet' in case
        if advanced:
            filename = case['booklet']['path']
            if filename not in source_hashes:
                source_hashes[filename] = hashlib.sha256((cases_path.parent / filename).read_bytes()).hexdigest()
            if label != 1:
                advanced_required += 1
                if not evidence:
                    raise ValueError('Missing required Advanced evidence: ' + str(cid))
        seen = set()
        for item in evidence:
            if set(item) != {'page', 'text'} or not isinstance(item['text'], str):
                raise ValueError('Invalid evidence fields: ' + str(cid))
            quote, page = item['text'], item['page']
            normalized = official_normalize(quote)
            if not normalized or len(normalized) > 5000 or (page, normalized) in seen:
                raise ValueError('Empty, oversized or duplicate quotation: ' + str(cid))
            seen.add((page, normalized))
            if advanced:
                source = pages.get(case, page) if type(page) is int and page > 0 else None
                if source is None or not (quote.strip() == source['text'].strip() or
                                          any(quote.strip() == block.strip() for block in source['blocks'])):
                    raise ValueError('Quote is not the original physical page or source block: ' + str(cid))
                quote_count += 1
            elif page is not None:
                raise ValueError('Beginner evidence must not invent PDF pages: ' + str(cid))
    return {'cases': len(cases), 'source_quotes_verified': quote_count,
            'advanced_non_neutral_cases_with_source_evidence': advanced_required,
            'invalid_or_missing_outputs': 0, 'gold_read': False,
            'case_order_preserved': True, 'source_pdf_sha256': source_hashes,
            'validation_scope': 'Every Advanced quote equals its physical PDF page or complete source block; all output IDs, labels and metrics are valid. Semantic entailment is not proved.',
            'predictions_sha256': hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
            'cases_sha256': hashlib.sha256(cases_path.read_bytes()).hexdigest(),
            'validator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior validation reports: ' + str(args.output))
    report = validate(args.cases, args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print({key: report[key] for key in ('cases', 'source_quotes_verified', 'invalid_or_missing_outputs', 'gold_read')})


if __name__ == '__main__':
    main()
