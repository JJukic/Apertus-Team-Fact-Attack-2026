"""Offline, source-backed error audit; gold is never imported by inference.

Label confusions and evidence-matching failures are exact observations.
Retrieval recall is a fuzzy passage-coverage diagnostic, not semantic proof.
Speaker/numerical/conditional causes require a documented manual annotation.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evidence import SourcePages
from src.inference import attributed_section, load_parsed_booklet
from src.official_evaluation import evidence_hit5, label_of, macro_f1, quote_matches
from src.pdf_parser import PDFParser
from src.retriever import PassageRetriever
from src.text_utils import clip_to_query


CATEGORIES = {
    'A': 'Entailment predicted as Contradiction',
    'B': 'Contradiction predicted as Entailment',
    'C': 'Neutral incorrectly predicted as non-neutral',
    'D': 'Non-neutral incorrectly predicted as Neutral',
    'E': 'Incorrect speaker attribution',
    'F': 'Missing relevant evidence',
    'G': 'Correct evidence but incorrect NLI interpretation',
    'H': 'Cross-lingual retrieval failure',
    'I': 'Numerical reasoning error',
    'J': 'Legal or conditional reasoning error',
    'K': 'Incorrect evidence page',
    'L': 'Evidence mismatch despite correct label',
}


def load(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate IDs')
    return {row['id']: row for row in rows}


def validate_annotations(annotations, cases, directory):
    """Verify manual observations against physical PDFs, not their plausibility."""
    pages = SourcePages(directory, lazy=True)
    for cid, annotation in annotations.items():
        if cid not in cases:
            raise ValueError('Annotation ID is outside the audit population: ' + cid)
        if not annotation.get('source_reason'):
            raise ValueError('Manual causes need a source-backed rationale: ' + cid)
        if set(annotation.get('categories', [])) - set(CATEGORIES):
            raise ValueError('Unknown error category: ' + cid)
        if annotation.get('review_status') not in ('source_reviewed', 'scope_adjudication_pending'):
            raise ValueError('Manual review status must be explicit: ' + cid)
        proof = annotation.get('source_proof', {})
        source_case = cases.get(proof.get('audit_source_case_id'))
        case = cases[cid]
        if (source_case is None or 'booklet' not in source_case
                or source_case.get('claim') != case.get('claim')
                or source_case.get('vote') != case.get('vote')):
            raise ValueError('PDF proof must belong to the reviewed claim/vote: ' + cid)
        if 'booklet' in case and source_case['id'] != cid:
            raise ValueError('Advanced proof must use its actual input booklet: ' + cid)
        filename = source_case['booklet']['path']
        if proof.get('pdf_path') != filename:
            raise ValueError('Source proof filename differs from input: ' + cid)
        if proof.get('pdf_sha256') != hashlib.sha256((directory / filename).read_bytes()).hexdigest():
            raise ValueError('Source PDF changed since review: ' + cid)
        page = pages.get(source_case, proof.get('page'))
        if page is None or proof.get('verbatim_text_block') not in page['blocks']:
            raise ValueError('Reviewed quotation is not an original PDF block: ' + cid)
        if 'reference' in case:
            digest = hashlib.sha256(case['reference']['text'].encode()).hexdigest()
            if annotation.get('reference_sha256') != digest:
                raise ValueError('Beginner input reference changed since review: ' + cid)


def audit(cases, gold, predictions, directory, records=None, annotations=None):
    if set(cases) != set(gold) or set(cases) != set(predictions):
        raise ValueError('Audit populations must be identical')
    annotations = annotations or {}
    validate_annotations(annotations, cases, directory)
    pages, booklets, rows = SourcePages(directory), {}, []
    parser = PDFParser()
    for cid, case in cases.items():
        label = label_of(predictions[cid])[0]
        truth = gold[cid]['label']
        wrong = label != truth
        task = 'A' if 'booklet' in case else 'B'
        source_language = case.get('booklet', case.get('reference', {})).get('language')
        claim_language = case['claim'].get('language')
        cross = source_language != claim_language
        hit = evidence_hit5(predictions[cid], gold[cid]['reference']) if task == 'A' and truth != 1 else None
        categories, review_flags = [], []
        if (truth, label) == (0, 2):
            categories.append('A')
        elif (truth, label) == (2, 0):
            categories.append('B')
        elif truth == 1 and label in (0, 2):
            categories.append('C')
        elif truth in (0, 2) and label == 1:
            categories.append('D')
        if task == 'A' and truth != 1 and not wrong and not hit:
            categories.append('L')
        if not wrong and not categories:
            continue
        context = []
        retrieval_hit = None
        side = attributed_section(case['claim']['text'])
        raw = records.get(cid, {}).get('raw_result', {}) if records else {}
        context_provenance = 'direct_reference' if task == 'B' else 'baseline_reconstruction'
        if task == 'A':
            filename = case['booklet']['path']
            if filename not in booklets:
                parsed = load_parsed_booklet(directory / filename, parser)
                booklets[filename] = PassageRetriever(parsed['paragraphs'])
            mode = raw.get('retrieval_query_metadata', {}).get('mode', 'original')
            if mode in ('translated', 'union'):
                # Never substitute R0 retrieval for a translated-query run.
                # These frozen page-level runs clip using the ORIGINAL claim,
                # so source text can be reconstructed without another API call.
                recorded = raw.get('context_pages')
                source_passages = booklets[filename].paragraphs
                by_page = {p['page_number']: p for p in source_passages}
                if (not isinstance(recorded, list) or not recorded or len(recorded) > 12
                        or len(recorded) != len(set(recorded))
                        or any(type(page) is not int or page not in by_page for page in recorded)
                        or len(by_page) != len(source_passages)):
                    raise ValueError('Recorded translated context needs unique source pages: ' + cid)
                context = [by_page[page] for page in recorded]
                context_provenance = 'recorded_' + mode + '_pages_original_claim_clipping'
            elif mode == 'original':
                context = booklets[filename].retrieve_hybrid(
                    case['claim']['text'], top_k=12, target_vote=case.get('vote'),
                    ensure_sections={side: 2} if side else None)
            else:
                raise ValueError('Unknown recorded retrieval mode: ' + cid)
            context = [{**p, 'text': clip_to_query(p['text'], case['claim']['text'] + ' ' + case.get('vote', ''), 3000)} for p in context]
            reconstructed = list(dict.fromkeys(p['page_number'] for p in context))
            if raw.get('context_pages') is not None and raw['context_pages'] != reconstructed:
                raise ValueError('Reconstructed context differs from recorded pages: ' + cid)
            if truth != 1:
                retrieval_hit = any(quote_matches(p['text'], gold[cid]['reference'], 90, 5000) for p in context)
                if wrong:
                    review_flags.append('G' if retrieval_hit else 'F')
                    if cross and not retrieval_hit:
                        review_flags.append('H')
            for evidence in predictions[cid].get('evidence', []):
                if not isinstance(evidence, dict) or pages.get(case, evidence.get('page')) is None:
                    categories.append('K')
                    break
        if wrong:
            if side:
                review_flags.append('E')
            if re.search(r'\d|Prozent|pour cent|percent|million|milliard', case['claim']['text'], re.I):
                review_flags.append('I')
            if re.search(r'\b(wenn|sofern|falls|nur|Gesetz|Verfassung|si|se|sauf|legge|loi|oblig|droit)\b', case['claim']['text'], re.I):
                review_flags.append('J')
        annotation = annotations.get(cid)
        if annotation:
            binding = annotation.get('applies_to')
            if binding is not None:
                current = {'gold_label': truth, 'predicted_label': label,
                           'context_pages': list(dict.fromkeys(p['page_number'] for p in context)),
                           'evidence_pages': sorted(set(e.get('page') for e in predictions[cid].get('evidence', [])
                                                        if isinstance(e, dict) and type(e.get('page')) is int))}
                if (not {'gold_label', 'predicted_label', 'context_pages'} <= set(binding)
                        or any(key not in current or current[key] != value for key, value in binding.items())):
                    raise ValueError('Source review no longer matches labels or supplied context: ' + cid)
            if annotation['review_status'] == 'source_reviewed':
                categories.extend(annotation.get('categories', []))
        rows.append({'id': cid, 'task': task, 'gold_label': truth, 'predicted_label': label,
                     'wrong_label': wrong, 'categories': sorted(set(categories)),
                     'review_flags_not_confirmed_causes': sorted(set(review_flags)),
                     'language_pair': f'{source_language}->{claim_language}',
                     'cross_language': cross, 'speaker_attribution': side or 'unattributed_or_ambiguous',
                     'retrieval_fuzzy_passage_hit': retrieval_hit, 'evidence_hit5': hit,
                     'context_provenance': context_provenance,
                     'context_pages': list(dict.fromkeys(p['page_number'] for p in context)),
                     'evidence_pages': [e.get('page') for e in predictions[cid].get('evidence', []) if isinstance(e, dict)],
                     'claim': case['claim']['text'], 'gold_reference': gold[cid].get('reference'),
                     'raw_reasoning': raw.get('reasoning'), 'manual_annotation': annotation})
    baseline = {}
    for task in ('A', 'B'):
        selected = [cid for cid in cases if ('A' if 'booklet' in cases[cid] else 'B') == task]
        baseline[task] = macro_f1([(gold[cid]['label'], label_of(predictions[cid])[0]) for cid in selected])[0]
    categories = {}
    for code, description in CATEGORIES.items():
        selected = [r for r in rows if code in r['categories']]
        upper_bound = 0.0
        for task, weight in (('A', 0.30), ('B', 0.28)):
            corrected = {r['id'] for r in selected if r['task'] == task and r['wrong_label']}
            ids = [cid for cid in cases if ('A' if 'booklet' in cases[cid] else 'B') == task]
            after = macro_f1([(gold[cid]['label'], gold[cid]['label'] if cid in corrected else label_of(predictions[cid])[0]) for cid in ids])[0]
            upper_bound += weight * (after - baseline[task])
        if code == 'L':
            denominator = sum('booklet' in case and gold[cid]['label'] != 1 for cid, case in cases.items())
            upper_bound += 0.12 * len(selected) / denominator if denominator else 0
        categories[code] = {'description': description, 'confirmed_count': len(selected),
                            'share_of_audited_error_cases': len(selected) / len(rows) if rows else 0,
                            'review_candidate_count': sum(code in r['review_flags_not_confirmed_causes'] for r in rows),
                            'affected_tasks': dict(Counter(r['task'] for r in selected)),
                            'language_pairs': dict(Counter(r['language_pair'] for r in selected)),
                            'retrieval_fuzzy_passage_hits': sum(r['retrieval_fuzzy_passage_hit'] is True for r in selected),
                            'speaker_attribution': dict(Counter(r['speaker_attribution'] for r in selected)),
                            'evidence_hits': sum(r['evidence_hit5'] is True for r in selected),
                            'counterfactual_quality_score_upper_bound': upper_bound,
                            'impact_assumption': 'Fix every confirmed case, no new errors, constant token/time efficiency; user-supplied weights.'}
    label_errors = [r for r in rows if r['wrong_label']]
    unreviewed = [r['id'] for r in label_errors if not r['manual_annotation']]
    unresolved = [r['id'] for r in label_errors if r['manual_annotation']
                  and r['manual_annotation']['review_status'] != 'source_reviewed']
    return {'status': ('observed_confusions_complete_semantic_cause_review_pending'
                       if unreviewed or unresolved else 'observed_confusions_and_source_review_complete'),
            'semantic_review': {'reviewed_label_errors': len(label_errors) - len(unreviewed),
                                'unreviewed_ids': unreviewed,
                                'scope_adjudication_pending_ids': unresolved,
                                'review_is_independent_human_adjudication': False},
            'prediction_cases': len(cases), 'audited_error_cases': len(rows),
            'label_errors': sum(r['wrong_label'] for r in rows),
            'categories_can_overlap': True, 'retrieval_is_fuzzy_proxy_not_semantic_proof': True,
            'categories': categories,
            'priority_by_confirmed_quality_impact': sorted(categories, key=lambda key: categories[key]['counterfactual_quality_score_upper_bound'], reverse=True),
            'cases': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases-dir', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--records-dir', type=Path)
    parser.add_argument('--annotations', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    records = None
    if args.records_dir:
        records = {row['id']: row for row in (json.loads(p.read_text()) for p in args.records_dir.glob('*.json'))}
    annotations = json.loads(args.annotations.read_text()) if args.annotations else None
    paths = [args.cases_dir / 'cases.jsonl', args.cases_dir / 'expected-labels.jsonl', args.predictions]
    if args.annotations:
        paths.append(args.annotations)
    report = audit(load(paths[0]), load(paths[1]), load(paths[2]), args.cases_dir, records, annotations)
    report['provenance'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    report['audit_code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'error_analysis.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    fields = ['id', 'task', 'gold_label', 'predicted_label', 'language_pair', 'categories',
              'review_flags_not_confirmed_causes', 'retrieval_fuzzy_passage_hit', 'speaker_attribution',
              'evidence_hit5', 'claim', 'gold_reference', 'raw_reasoning']
    with (args.output / 'manual_review.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in report['cases'] if row['wrong_label'])
    print({'cases': report['prediction_cases'], 'label_errors': report['label_errors'],
           'priority': report['priority_by_confirmed_quality_impact']})


if __name__ == '__main__':
    main()
