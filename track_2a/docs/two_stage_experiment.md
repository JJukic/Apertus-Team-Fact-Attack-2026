# Extract-then-verify experiment

Branch: `experiment/two-stage-fact-check`. Model quality has not been measured yet.

Existing retrieval supplies numbered original passages. An additional Apertus call
extracts at most eight relevant exact quotes with source ids. Invalid quotes or ids
stop the check with an explicit error. Quotes are checked against source text without metadata prefixes. Speaker hints are rebuilt after passage selection; usage is retained even when judgment parsing fails. A second call evaluates the original claim
against the complete source passages selected by extraction. Its evidence ids are
mapped back to original retrieval ids. Tokens for both calls and overall model
latency are recorded. Reports include `extracted_statements`.

An empty extraction yields Neutral; extraction misses can therefore introduce false
Neutral decisions. Existing retrieval and decision rules remain for comparison.
This version extracts document statements; it does not decompose compound claims.
Offline mock runs only check wiring and do not simulate extraction.

Choose the experiment under **Prüfmethode** in the app, or use `--prompt-mode two_stage`
for CLI benchmarks (or set `PROMPT_MODE=two_stage` for predictions). Cached baseline demo fallback is disabled for this method.

Compare the same dev sample first (requires API key and downloaded dataset):

```sh
cd track_2a
../.venv/bin/python -m src benchmark -d data/hf/dev.jsonl -n 150 --seed 42 -p ids --tag baseline_comparison
../.venv/bin/python -m src benchmark -d data/hf/dev.jsonl -n 150 --seed 42 -p two_stage --tag extract_verify
```

Compare macro-F1, errors, evidence grounding, tokens and latency. Inspect extraction
misses separately from incorrect judgments. Only then evaluate the held-out test set.
The official export format remains unchanged.
