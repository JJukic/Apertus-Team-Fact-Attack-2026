# D1 — task A context rebuild and the voting-text budget (E23, offline part)

See `docs/independent_experiments.md`, section 19. Run from `track_2a` after `scripts/prepare_competition_cases.py`
(`.cache/gen/data`) and `scripts/build_holdout.py` (`.cache/holdout_2026-09-27`). No inference call: Apertus is
mocked and the query translations come from the E12 runs' caches (`.cache/gen/queries/E12_section_route`,
`HO_E12_section_route`).

- `retrieval_miss.py . gen|ho <out.jsonl>` — rebuilds every task A context of E12 (part selection, else the hybrid
  retrieval), maps the gold passage to booklet pages and records whether the gold page is in the context and how much
  of its text survived the budget → `rows_E12_dataset.jsonl`, `rows_E12_holdout.jsonl`
- `summarize.py <rows.jsonl> [predictions.jsonl]` — error categories (gold page missing / cut / in context), Hit@5
  misses, rebuild fidelity (cited pages outside the rebuilt context)
- `part_variants.py . gen|ho <rows.jsonl> <out.json>` — alternative selections inside an over-budget part (vote-title
  weight, ~1,000-character pieces, larger voting-text budget) → `part_variants_dataset.json`, `part_variants_holdout.json`
- `build_subset.py` — the cases whose voting-text part is over 6,000 characters (92 dataset, 8 holdout), for the paired
  runs `E23_legal_base` / `E23_legal9000` and `HO_E23_legal_base` / `HO_E23_legal9000`
  (`scripts/run_decision.py --data .cache/gen/legal_gen --task A --set PART_CONTEXT_CHARS_LEGAL=0|9000 --query-cache-dir …`;
  the base runs were made before the default changed)
- `compare_e23.py .` — paired comparison of those runs, with the saved E12 run as a noise reference

## E24 — stress test of the part selection with rephrased claim openings (offline)

- `stress_routing.py . gen|ho <rows.jsonl> <out.json> [part_context.py copy]` — replaces the opening that names the
  source by other wordings (SYN: the wordings later added to the cue list; SYN_B: other wordings, used for evaluation
  only; LATE: the source at the end; NONE: no source), routes every variant again and records same part / fallback
  retrieval / other part and whether the gold page is in the resulting context (fallback rebuilt without the claim's
  translation, so conservative). The optional last argument compares another cue list, e.g. the one before E24:
  `git show 8541cb4:track_2a/src/part_context.py > part_context_E23.py`
- `stress_dataset_old.json` / `stress_holdout_old.json`: cue list of E23; `stress_dataset_new.json` /
  `stress_holdout_new.json`: E24
