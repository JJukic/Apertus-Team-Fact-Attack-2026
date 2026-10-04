# Experiment Log — OST benchmark (Hugging Face, 1,495 pairs)

All numbers below come from `python -m src benchmark` runs saved in [`results/`](../results/)
(each file contains model, strategy, prompt mode, passage size, git commit and per-sample predictions).
`python -m src.results_summary` collects them into `results/summary.json`, which the app reads.

## Setup

- **Data:** `OSTswiss/MNLIoverSwissVotingBooklets` (1,495 human-annotated pairs, 60 booklets 2020–2026,
  ~66 % cross-lingual). Downloaded with `python -m src.hf_dataset`.
- **Split by voting date** (all three languages of a booklet stay together), seed 42:
  - **dev** 1,093 pairs — used for every design decision
  - **test** 402 pairs from 5 unseen voting dates (2021-06-13, 2021-09-26, 2022-09-25, 2024-06-09, 2024-09-22) —
    only used for before/after measurements
  - 2026-06-14 and 2024-11-24 are pinned to dev because the earlier pipeline was tuned on them.
- **Samples:** label-balanced; 150 pairs for quick comparisons, 450 for close calls, the full test split for final numbers.
- **Model:** `swiss-ai/Apertus-v1.5-70B-thinking` on CSCS (the only model our key is authorised for).
- Latency is measured client-side with 3–4 parallel requests; runs that overlapped with other runs are slower.

## Final result: full test split (402 pairs, 5 unseen voting dates)

| Task | Macro-F1 | Mono-lingual | Cross-lingual | Input tokens | Output tokens | Latency mean / p95 |
|---|---:|---:|---:|---:|---:|---:|
| **Advanced** (booklet PDF + claim + vote) | **0.920** | 0.951 | 0.906 | 5,737 | 59 | 2.1 s / 4.0 s |
| **Beginner** (reference string + claim) | **0.975** | 0.984 | 0.971 | 2,523 | 55 | 1.1 s / 1.7 s |

Configuration: hybrid retrieval (BM25 claim + 2 × BM25 vote title, booklet-wide), top 10 pages,
section labels, `ids` prompt mode.

Advanced macro-F1 by language pair (claim → booklet):

| | → de | → fr | → it |
|---|---:|---:|---:|
| **de** | 0.951 | 0.933 | 0.922 |
| **fr** | 0.914 | 0.956 | 1.000 |
| **it** | 0.788 | 0.874 | 0.934 |

## Before / after on the test split (same 150-pair sample)

| Task | System | Macro-F1 | Cross-lingual F1 | Input tokens | Output tokens | Latency mean / p95 |
|---|---|---:|---:|---:|---:|---:|
| Advanced | Full booklet (baseline) | 0.730 | 0.756 | 59,517 | 202 | 21.9 s / 43.7 s* |
| Advanced | Old pipeline (proposal filter, k=5, JSON) | 0.795 | 0.806 | 3,408 | 157 | 16.2 s / 36.2 s* |
| Advanced | Proposal filter, k=10, `ids` | 0.880 | 0.853 | 6,502 | 54 | 2.6 s / 4.1 s |
| Advanced | **Hybrid, k=10, `ids`** | **0.926** | **0.910** | 6,023 | 55 | 2.3 s / 3.8 s |
| Beginner | Old pipeline (JSON, numeric override) | 0.953 | 0.980 | 2,248 | 149 | 3.9 s / 12.0 s* |
| Beginner | **`ids`** | **0.974** | **0.971** | 2,472 | 54 | 1.2 s / 1.6 s |

\* measured while three runs shared the endpoint; see the dev comparison below for like-for-like latency.

**Selected context beats the full booklet** on quality (+0.19 F1) with ~10× fewer input tokens: with the
whole booklet (up to 70k tokens) the relevant passage gets lost.

## Dev-split ablations

### Prompt mode (beginner task, n = 150)

| Prompt mode | Macro-F1 | Output tokens | Notes |
|---|---:|---:|---|
| `json` (reasoning + verbatim quotes) | 0.973 | 152 | evidence = whole reference |
| `compact` (label digit + ids, logprobs) | 0.724 | 8 | Neutral recall 0.34: predicts Contradiction for unaddressed claims |
| **`ids`** (confidences + passage ids) | **0.980** | 55 | evidence = specific passages |
| `ids` + viewpoint/recommendation rules | 0.993 | 55 | |

Label-only answering loses the "estimate p_entail / p_neutral / p_contra first" step, which acts as a
short reasoning step; recalibrating the logprobs only recovered 0.82. `ids` keeps that step and drops the
free-text reasoning and quotes.

### Prompt mode and retrieval (advanced task, n = 150)

| Retrieval | Prompt | Macro-F1 | Input tokens | Output tokens | Latency mean / p95 |
|---|---|---:|---:|---:|---:|
| Proposal filter, k=5 | `json` | 0.786 | 2,961 | 164 | 3.3 s / 5.5 s |
| Proposal filter, k=5 | `ids` | 0.797 | 3,025 | 54 | 1.4 s / 1.8 s |
| Hybrid, k=10 | `ids` | 0.862 | 4,780 | 56 | 2.1 s / 2.9 s |
| Hybrid, k=12 | `ids` | 0.861 | 5,734 | 56 | 2.1 s / 2.9 s |
| Hybrid, k=10 | `ids` + viewpoint rules | 0.870 | 4,917 | 56 | 2.2 s / 3.2 s |

### Passage size (advanced task, hybrid, `ids`)

| Passages | k | Section labels | Reason sentence | n | Macro-F1 | Evidence in gold section | Input tokens |
|---|---:|:---:|:---:|---:|---:|---:|---:|
| 600 chars | 15 | – | – | 150 | 0.799 | 0.89 | 2,689 |
| 600 chars | 20 | – | – | 150 | 0.843 | 0.92 | 3,379 |
| 600 chars | 30 | – | – | 150 | 0.833 | 0.91 | 4,745 |
| 900 chars | 15 | – | – | 150 | 0.837 | 0.90 | 3,398 |
| 600 chars | 20 | ✓ | – | 150 | 0.858 | 0.89 | 3,469 |
| 600 chars | 20 | ✓ | ✓ | 150 | 0.859 | 0.91 | 3,491 |
| whole page | 10 | ✓ | – | 150 | 0.862 | 0.79 | 4,957 |
| 600 chars | 20 | ✓ | – | **450** | 0.831 | 0.91 | 3,443 |
| **whole page** | 10 | ✓ | – | **450** | **0.908** | 0.86 | 4,842 |

On n = 150 the variants were within noise; the paired n = 450 comparison settled it: whole pages are right
on 44 pairs where 600-char passages are wrong, and the reverse on only 10. Smaller passages give more precise
evidence and use fewer tokens, but lose the surrounding context the model needs, so pages stay the default
(`PASSAGE_CHARS=0`). A short reason sentence before the confidences fixed 8 of 23 errors in a targeted
check but gave no gain on the full run, so it stays off (`IDS_REASON=false`).

### Retrieval recall (no API calls, dev)

Share of dev pairs where at least one retrieved passage lies in the gold `reference_string`
(strict: ≥ 80 % of the passage's words appear in the gold section).

| Retrieval | Passages | Strict recall | ~Input tokens |
|---|---|---:|---:|
| Proposal filter, k=5 | page | 0.744 | 2,536 |
| Proposal filter, k=8 | page | 0.792 | 4,026 |
| Hybrid (claim + 2×vote), k=8 | page | 0.866 | 3,361 |
| Hybrid (claim + 2×vote), k=10 | page | 0.932 | 4,231 |
| Hybrid (claim + 2×vote), k=12 | page | 0.955 | 5,185 |
| Hybrid, k=20 | 600 chars | 0.981 | 2,507 |

Why the proposal filter loses recall: in ~11 % of pairs the gold text sits on the per-proposal
overview pages at the front of the booklet (proposal id 0), which the hard filter drops, and in ~10 %
the vote→proposal mapping picks the wrong proposal. The vote title is written in the booklet's
language, so BM25 on it is a reliable anchor even for cross-lingual claims. Higher recall with small
passages did not translate into higher F1 (see above).

## Findings that changed the design

1. **The hard numerical override hurt.** On the beginner baseline it fired 5 times and was wrong 5 times
   (claims citing a year the booklet never mentions are Neutral, not Contradiction). It is now opt-in
   (`NUMERIC_OVERRIDE=true`); the detector still reports conflicts in the output.
2. **Viewpoints.** Booklets contain both the Federal Council's and the committee's arguments. Claims like
   "the committee argues X" were judged against the opposing side and labelled Contradiction. A prompt rule
   tells the model to check only the attributed actor's text, and each page is labelled with its section
   ("Arguments of the initiative/referendum committee", "Arguments of the Federal Council and Parliament", …).
3. **Evidence as passage ids** guarantees verbatim booklet text with correct page numbers and cuts output
   tokens by ~65 %.
4. **The full booklet is the worst option** on both quality and cost.

## Known gaps / next steps

- Italian claims against German booklets are the weakest pair (0.79 F1 on the test split).
- Most remaining advanced errors are Entailment → Contradiction (16 of 32 on the test split).
- Evidence precision with whole pages: ~75 % of cited pages lie in the gold section.
- Section headings are text-based; older booklets (≈2021) set them as graphics, so labels are partial there.
- Proposal boundary detection misses proposals in 2021-06-13, 2022-09-25 and 2024-03-03 (IT); hybrid
  retrieval does not depend on it, but the legacy `retrieval` strategy does.
- Only `70B-thinking` is accessible with our key; 8B (faster) could not be evaluated.
