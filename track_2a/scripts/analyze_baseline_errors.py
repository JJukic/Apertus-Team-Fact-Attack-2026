"""Summarise historical errors; no model requests or causal claims."""
import json
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / 'results/20261006T140501_advanced_hybrid_ids_Apertus-v1.5-70B-thinking_final_docker.json'
OUT = BASE / 'docs/offline_baseline_analysis.md'
CASES = BASE / 'data/evaluation/two_stage_diagnostic_cases.json'


def main():
    report = json.loads(SOURCE.read_text())
    rows = report['results']
    wrong = [r for r in rows if r['true_label'] != r['pred_label'] and not r.get('error')]
    pairs = Counter(r['pair'] for r in rows)
    errors = Counter(r['pair'] for r in wrong)
    confusion = Counter((r['true_label'], r['pred_label']) for r in wrong)
    groups = defaultdict(list)
    for r in wrong:
        groups[(r['claim'], r.get('vote'), r['pair'])].append(r['id'])
    fallback = [r for r in wrong if 'evidence fallback' in (r.get('decision_rule') or '')]
    overall_fallback = sum('evidence fallback' in (r.get('decision_rule') or '') for r in rows)
    cross = [r for r in rows if len(set(r['pair'].split('->'))) > 1]
    cross_errors = sum(r['true_label'] != r['pred_label'] for r in cross)
    lines = ['# Offline analysis of baseline errors', '',
             f'Source: `{SOURCE.name}`, commit `{report["meta"]["git_commit"]}`. Existing live results only; no new model calls.', '',
             f'- {len(rows)} cases, {len(wrong)} misclassifications, {report["errors"]["count"]} technical errors.',
             f'- Saved macro-F1: {report["macro_f1"]:.4f}.',
             f'- Cross-lingual: {cross_errors}/{len(cross)} errors; same-language: {len(wrong)-cross_errors}/{len(rows)-len(cross)}.',
             f'- {len(groups)} distinct claim/vote/language-pair groups among the {len(wrong)} errors.',
             f'- {len(fallback)} errors mention automatic top-passage fallback; {overall_fallback} cases overall use it.', '',
             '## Wrong-label transitions', '', '| Expected | Predicted | Count |', '|---|---|---|']
    lines += [f'| {a} | {b} | {n} |' for (a, b), n in sorted(confusion.items())]
    lines += ['', '## Language pairs (include denominators)', '', '| Pair | Errors / cases | Error rate |', '|---|---|---|']
    lines += [f'| {pair} | {errors[pair]}/{n} | {errors[pair]/n:.1%} |' for pair, n in sorted(pairs.items())]
    lines += ['', '## Repeated claims', '']
    lines += [f'- `{", ".join(ids)}`: {key[0]}' for key, ids in groups.items() if len(ids)>1]
    lines += ['', '## Interpretation and next measurement', '',
              'Observed topics include hydropower reserves, pesticide imports, VAT changes and actor-attributed claims. These are review candidates, not established failure causes.',
              'A recorded decision rule does not establish that the rule changed a correct raw label. Original raw responses and selected context are not fully present in this report.',
              'Neutral outputs have no evidence by design: absence of evidence alone is not proof of a retrieval failure.',
              'We cannot decide whether errors come from missing context, language interpretation, source attribution, judgment rules or gold labels without inspecting original context and references.',
              'The diagnostic JSON retains all 24 rows, including repeated claims, with source provenance. It is NOT an independent evaluation set: these are exposed test cases selected because baseline failed. Do not tune on these and then claim held-out improvement.',
              'Use a fixed dev sample with both errors and correct controls for the actual comparison. Score the two methods on identical data and report technical failures separately. The saved baseline predates our stricter response/evidence validation, so rerun the current ids baseline alongside two_stage.', '',
              '## Offline pipeline coverage', '',
              'Automated tests exercise a real PDF, retrieval, simulated extraction and judgment responses, source-id mapping, official export, token accounting and stage failures. No network requests are made. These tests establish pipeline behavior, not model accuracy or CSCS latency.', '']
    OUT.write_text('\n'.join(lines))
    CASES.parent.mkdir(parents=True, exist_ok=True)
    CASES.write_text(json.dumps({'purpose': 'diagnostic_only_exposed_test_errors', 'source': SOURCE.name,
                               'source_commit': report['meta']['git_commit'], 'cases': wrong}, ensure_ascii=False, indent=2))
    print('\n'.join(lines[:11]))

if __name__ == '__main__':
    main()
