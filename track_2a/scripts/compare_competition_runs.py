"""Compare predictions on identical date-grouped folds using official scoring.

This is offline evaluation tooling. The prediction process never imports this
module or reads its gold-label input. Simulations use the user-supplied weights,
not unknown private competition efficiency references.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.official_evaluation import evidence_hit5, label_of, macro_f1
from src.competition_selection import distribution, quality_component, simulated_score, known_component_pareto, rank_by_known_quality


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError(f'Duplicate IDs: {path}')
    return {r['id']: r for r in rows}


def metrics(ids, predictions, gold, hits=None, score_evidence=True):
    pairs = [(gold[cid]['label'], label_of(predictions[cid])[0]) for cid in ids]
    score, classes = macro_f1(pairs)
    required = [cid for cid in ids if gold[cid]['label'] != 1] if score_evidence else []
    missing = sum('reference' not in gold[cid] for cid in required)
    found = sum(hits[cid] if hits is not None else evidence_hit5(predictions[cid], gold[cid]['reference'])
                for cid in required if 'reference' in gold[cid])
    return {'cases': len(ids), 'macro_f1': score, 'per_class': classes,
            'evidence_hit5': found / len(required) if required and not missing else None,
            'gold_evidence_cases': len(required), 'evidence_found': found,
            'missing_gold_reference': missing,
            'invalid_predictions': sum(predicted == -1 for _, predicted in pairs)}


def summarize(predictions, cases, gold, folds):
    result = {'tasks': {}, 'by_language': {}, 'folds': {}}
    hits = {cid: evidence_hit5(predictions[cid], row['reference'])
            for cid, row in gold.items()
            if 'booklet' in cases[cid] and row['label'] != 1 and 'reference' in row}
    result['case_evidence_hit5'] = hits
    for task in ('A', 'B'):
        ids = [cid for cid, case in cases.items() if ('A' if 'booklet' in case else 'B') == task]
        if not ids:
            continue
        result['tasks'][task] = metrics(ids, predictions, gold, hits, task == 'A')
        if task == 'B':
            result['tasks'][task]['evidence_hit5'] = None
        result['tasks'][task]['passes_acceptance_threshold'] = result['tasks'][task]['macro_f1'] >= (
            0.60 if task == 'A' else 0.75)
        cross = [cid for cid in ids if cases[cid]['claim']['language'] !=
                 cases[cid].get('booklet', cases[cid].get('reference'))['language']]
        result['tasks'][task]['cross_language_macro_f1'] = metrics(cross, predictions, gold, hits, task == 'A')['macro_f1'] if cross else None
        fold_scores = []
        for fold in sorted({folds[cid]['fold'] for cid in ids}):
            selected = [cid for cid in ids if folds[cid]['fold'] == fold]
            data = metrics(selected, predictions, gold, hits, task == 'A')
            if task == 'B':
                data['evidence_hit5'] = None
            result['folds'][f'{task}:{fold}'] = data
            fold_scores.append(data['macro_f1'])
        result['tasks'][task]['fold_f1_mean'] = statistics.mean(fold_scores)
        result['tasks'][task]['fold_f1_std'] = statistics.pstdev(fold_scores)
        result['tasks'][task]['nonempty_fold_count'] = len(fold_scores)
        pairs = {}
        for cid in ids:
            case = cases[cid]
            source = case.get('booklet', case.get('reference'))['language']
            pair = f"{source}->{case['claim']['language']}"
            pairs.setdefault(pair, []).append(cid)
        for pair, selected in pairs.items():
            result['by_language'][f'{task}:{pair}'] = metrics(selected, predictions, gold, hits, task == 'A')
        result['tasks'][task]['input_tokens_reported'] = sum(predictions[cid].get('metrics', {}).get('input_tokens', 0) for cid in ids)
        result['tasks'][task]['output_tokens_reported'] = sum(predictions[cid].get('metrics', {}).get('output_tokens', 0) for cid in ids)
        result['tasks'][task]['input_tokens_per_case_reported'] = distribution([
            predictions[cid].get('metrics', {}).get('input_tokens', 0) for cid in ids])
        result['tasks'][task]['output_tokens_per_case_reported'] = distribution([
            predictions[cid].get('metrics', {}).get('output_tokens', 0) for cid in ids])
    result['missing_required_evidence'] = sum('booklet' in cases[cid] and label_of(prediction)[0] in (0, 2)
                                              and not prediction.get('evidence') for cid, prediction in predictions.items())
    result['qualifies'] = (all(task['passes_acceptance_threshold'] and not task['invalid_predictions']
                             for task in result['tasks'].values()) and not result['missing_required_evidence'])
    return result


def paired_changes(baseline, candidate, gold, cases, before_hits=None, after_hits=None):
    fixed, broken, evidence_fixed, evidence_broken = [], [], [], []
    for cid, expected in gold.items():
        before = label_of(baseline[cid])[0] == expected['label']
        after = label_of(candidate[cid])[0] == expected['label']
        if after and not before:
            fixed.append(cid)
        if before and not after:
            broken.append(cid)
        if 'booklet' in cases[cid] and expected['label'] != 1 and 'reference' in expected:
            before_hit = before_hits[cid] if before_hits is not None else evidence_hit5(baseline[cid], expected['reference'])
            after_hit = after_hits[cid] if after_hits is not None else evidence_hit5(candidate[cid], expected['reference'])
            if after_hit and not before_hit:
                evidence_fixed.append(cid)
            if before_hit and not after_hit:
                evidence_broken.append(cid)
    return {'fixed': fixed, 'broken': broken, 'evidence_fixed': evidence_fixed, 'evidence_broken': evidence_broken}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases-dir', type=Path, required=True)
    parser.add_argument('--run', action='append', required=True, help='NAME=predictions.jsonl')
    parser.add_argument('--baseline', default='B0')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--measurement', action='append', default=[], help='NAME=measurement.json (local SDK timing)')
    parser.add_argument('--time-factor', action='append', default=[], help='NAME=A_FACTOR,B_FACTOR, explicit simulation assumptions')
    parser.add_argument('--diagnostic', action='append', default=[], help='NAME is a replay/assembled diagnostic, excluded from final selection')
    args = parser.parse_args()
    cases = load(args.cases_dir / 'cases.jsonl')
    gold = load(args.cases_dir / 'expected-labels.jsonl')
    folds = load(args.cases_dir / 'folds.jsonl')
    if set(cases) != set(gold) or set(cases) != set(folds):
        raise ValueError('Cases, labels and fold assignments differ')
    date_folds = {}
    for row in folds.values():
        previous = date_folds.setdefault(row['date'], row['fold'])
        if previous != row['fold']:
            raise ValueError('Date/booklet leakage across folds')
    runs, provenance = {}, {}
    for argument in args.run:
        name, path = argument.split('=', 1)
        if name in runs:
            raise ValueError(f'Duplicate run name: {name}')
        runs[name] = load(path)
        if set(runs[name]) != set(cases):
            raise ValueError(f'Run {name} has a different case population')
        provenance[name] = {'file': path, 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}
    if args.baseline not in runs:
        raise ValueError('Baseline run is required')
    summaries = {name: summarize(predictions, cases, gold, folds) for name, predictions in runs.items()}
    if set(args.diagnostic) - set(summaries):
        raise ValueError('Unknown diagnostic run')
    for name, result in summaries.items():
        result['selection_eligible'] = (result['qualifies'] and name not in args.diagnostic and
                                       all(task['nonempty_fold_count'] == 5 for task in result['tasks'].values()))
        result['diagnostic_replay'] = name in args.diagnostic
    baseline = summaries[args.baseline]
    measured = {}
    for argument in args.measurement:
        name, path = argument.split('=', 1)
        if name not in summaries or name in measured:
            raise ValueError('Unknown or duplicate measurement name')
        measurement = json.loads(Path(path).read_text())
        measured[name] = {key: measurement.get(key) for key in (
            'cases', 'operational_errors', 'api_attempts', 'usage_unknown_attempts',
            'input_tokens_known', 'output_tokens_known', 'non_llm_seconds_all_sessions')}
        measured[name]['peak_rss_bytes'] = max((s['peak_rss_bytes'] for s in measurement.get('sessions', [])), default=None)
        measured[name]['source_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        measured[name]['timing_scope'] = 'local SDK union over mixed-task sessions, not official task A/B efficiencies'
        measured[name]['token_totals_are_lower_bounds'] = bool(measured[name]['usage_unknown_attempts'])
        summaries[name]['measurement'] = measured[name]
    time_factors = {args.baseline: (1.0, 1.0)}
    for argument in args.time_factor:
        name, factors = argument.split('=', 1)
        values = tuple(float(value) for value in factors.split(','))
        if name not in summaries or len(values) != 2 or min(values) <= 0:
            raise ValueError('time-factor must name a run and positive A,B factors')
        time_factors[name] = values
    rows = []
    for name, result in summaries.items():
        changes = paired_changes(runs[args.baseline], runs[name], gold, cases,
                                 baseline['case_evidence_hit5'], result['case_evidence_hit5'])
        result['paired_changes'] = changes
        for task, data in result['tasks'].items():
            original = baseline['tasks'][task]
            data['delta_macro_f1'] = data['macro_f1'] - original['macro_f1']
            data['delta_evidence_hit5'] = (data['evidence_hit5'] - original['evidence_hit5']
                                         if data['evidence_hit5'] is not None and original['evidence_hit5'] is not None else None)
            for field in ('input_tokens_reported', 'output_tokens_reported'):
                data['delta_' + field] = data[field] - original[field]
            original_tokens = original['input_tokens_reported'] + original['output_tokens_reported']
            own_tokens = data['input_tokens_reported'] + data['output_tokens_reported']
            data['token_reduction_fraction_reported'] = 1 - own_tokens / original_tokens if original_tokens else None
        result['quality_component_user_weights'] = quality_component(result)
        if name in time_factors:
            result['simulated_total_scores_not_official'] = {
                str(fraction): simulated_score(result, baseline, fraction, time_factors[name])
                for fraction in (0.1, 0.25, 0.5, 1.0)}
            result['assumed_A_B_processing_time_factors'] = time_factors[name]
        else:
            result['simulated_total_scores_not_official'] = None
        if name in measured and args.baseline in measured:
            original_time = measured[args.baseline]['non_llm_seconds_all_sessions']
            own_time = measured[name]['non_llm_seconds_all_sessions']
            result['local_mixed_processing_time_reduction_fraction'] = (1 - own_time / original_time
                                                                        if original_time else None)
        else:
            result['local_mixed_processing_time_reduction_fraction'] = None
        result['language_regressions'] = {pair: value['macro_f1'] - baseline['by_language'][pair]['macro_f1']
                                          for pair, value in result['by_language'].items()
                                          if value['macro_f1'] < baseline['by_language'][pair]['macro_f1']}
        result['class_regressions'] = {f'{task}:{label}': cls['f1'] - baseline['tasks'][task]['per_class'][label]['f1']
                                       for task, data in result['tasks'].items()
                                       for label, cls in data['per_class'].items()
                                       if cls['f1'] < baseline['tasks'][task]['per_class'][label]['f1']}
        row = {'run': name, 'qualifies': result['qualifies'], 'fixed': len(changes['fixed']), 'broken': len(changes['broken']),
               'evidence_fixed': len(changes['evidence_fixed']), 'evidence_broken': len(changes['evidence_broken'])}
        row['selection_eligible'] = result['selection_eligible']
        for task, data in result['tasks'].items():
            for key in ('macro_f1', 'evidence_hit5', 'cross_language_macro_f1', 'fold_f1_mean', 'fold_f1_std',
                        'input_tokens_reported', 'output_tokens_reported', 'delta_macro_f1', 'delta_evidence_hit5'):
                row[f'{task}_{key}'] = data[key]
            row[f'{task}_token_reduction_fraction_reported'] = data['token_reduction_fraction_reported']
        row['quality_component_user_weights'] = result['quality_component_user_weights']
        row['local_mixed_processing_time_reduction_fraction'] = result['local_mixed_processing_time_reduction_fraction']
        rows.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {'status': 'component_comparison_no_official_total_score',
              'baseline': args.baseline, 'runs': summaries, 'provenance': provenance,
              'language_direction': 'source->claim', 'selection_population': 'observed grouped validation',
              'untouched_holdout_available': False,
              'known_quality_ranking': rank_by_known_quality(summaries),
              'known_quality_token_pareto': known_component_pareto(summaries),
              'true_efficiency_frontier_unknown': True,
              'scenario_limitations': ['Simulations use user-supplied weights, known token lower bounds and explicitly assumed task processing-time factors.',
                                       'No mixed-task concurrent timer is assigned to separate A/B efficiency scores.',
                                       'Actual competitor/proxy costs are unknown; there is no official total score.']}
    (args.output / 'experiments.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    with (args.output / 'comparison.csv').open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row['run'], 'A F1', row.get('A_macro_f1'), 'A Hit@5', row.get('A_evidence_hit5'),
              'B F1', row.get('B_macro_f1'), 'qualifies', row['qualifies'])


if __name__ == '__main__':
    main()
