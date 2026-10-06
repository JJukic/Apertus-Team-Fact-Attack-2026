"""Paired diagnostics for local dense retrieval; no tuning or new inference."""
import json
from collections import defaultdict
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]

def main():
    report = json.loads((BASE / 'docs/local_dense_comparison.json').read_text())
    rows = {r['id']: r for r in (json.loads(line) for line in (BASE/'data/hf/dev.jsonl').read_text().splitlines())}
    for k in (5, 12):
        cases = report['cases']
        groups = defaultdict(list)
        for case in cases:
            groups[case['pair']].append(case)
        lines = [f'# Paired E5/title vs BM25/title diagnostics (k={k})', '',
                 'Same frozen dev rows as the local comparison. Changes below concern lexical reference-fragment coverage, not model decisions.', '',
                 '| Language pair | Cases | Coverage improved | Coverage reduced | Mean change (percentage points) |',
                 '|---|---|---|---|---|']
        deltas = {}
        for pair, group in sorted(groups.items()):
            changes = [c['results'][f'dense_title_k{k}']['coverage'] - c['results'][f'bm25_title_k{k}']['coverage'] for c in group]
            deltas.update({c['id']: d for c, d in zip(group, changes)})
            lines.append(f'| {pair} | {len(group)} | {sum(d>1e-9 for d in changes)} | {sum(d < -1e-9 for d in changes)} | {100*sum(changes)/len(changes):+.2f} |')
        for title, keys in [('Largest coverage gains', sorted(deltas, key=lambda i: -deltas[i])[:5]), ('Largest coverage losses', sorted(deltas, key=deltas.get)[:5])]:
            lines += ['', f'## {title}', '']
            for cid in keys:
                case = next(c for c in cases if c['id'] == cid)
                lines += [f"- `{cid}` ({case['pair']}, change {100*deltas[cid]:+.1f} points): {rows[cid]['claim']}",
                          f"  BM25 pages: {case['results'][f'bm25_title_k{k}']['pages']}; E5/title pages: {case['results'][f'dense_title_k{k}']['pages']}."]
        lines += ['', 'These examples were selected after inspecting the measured dev results and are diagnostic, not independent validation. A reduction may reflect omitted relevant text or the removal of irrelevant parts of a long reference section; a gain may include repeated boilerplate. Inspect source text before attributing causes.', '']
        (BASE/f'docs/dense_case_diagnostics_k{k}.md').write_text('\n'.join(lines))
        print(f'k={k}: coverage improves in {sum(d>1e-9 for d in deltas.values())}, decreases in {sum(d < -1e-9 for d in deltas.values())} of {len(deltas)} cases')

if __name__ == '__main__': main()
