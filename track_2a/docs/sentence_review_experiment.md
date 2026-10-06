# Sentence-ID review with full retrieved context

Experimental prompt mode: `sentence_review`. No live accuracy measurement yet.

Retrieved passages are split at punctuation followed by whitespace. These are
approximate sentence spans; abbreviations and decimals can create imperfect boundaries.
Each span preserves exact source text, original passage id, page and speaker metadata.
Global sentence IDs stay fixed across both calls, not across different retrieval runs.

1. A selector suggests at most 12 sentence IDs without classifying the claim.
2. A judge receives ALL sentences from the same retrieved, clipped context, the claim
   and optional suggestions. It may cite sentences absent from the suggestions.
3. Evidence is extracted from original source spans, not generated quotes.

Empty or invalid selection does not short-circuit to Neutral; the full-context judge
still runs. Selection failures appear as `stage_warnings` in the app and saved reports.
This is not a baseline fallback and is not guaranteed to preserve baseline accuracy.
Responses with usage count even on parse failures. Failed API requests with no usage
cannot have their token cost measured exactly.

Baseline calibrated decision rules still apply. Retrieval, clipping and optional
translation remain. Speaker attribution uses source section metadata and the claim;
no passage hint requiring renumbering is appended. Mock mode checks wiring only.

Select **Experiment: Satz-IDs mit vollständigem Kontext** in the app, or benchmark:

```sh
cd track_2a
../.venv/bin/python -m src benchmark -d data/hf/dev.jsonl -n 150 --seed 42 -p sentence_review --tag full_context_sentences
```

Compare `ids`, `two_stage` and `sentence_review` on identical frozen cases and settings.
This variant changes evidence granularity and prompts too: it is not a pure context
filter ablation. Measure F1, evidence validity, stage warnings, tokens, latency,
fixed and newly broken cases. Offline simulated judgments do not prove model quality.
