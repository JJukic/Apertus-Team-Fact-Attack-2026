# Competition optimization — audit, 8 October 2026

## 1. Executive summary

Work is isolated on `experiment/competition-score-optimization`, starting from
latest `origin/main`, commit `a0e9e772dd388bdff4b8635ef0e09872ef705af6`.
The adaptive branch and its uncommitted work remain separate.

The strongest documented production baseline has Advanced Macro-F1 **0.940236**
and Beginner **0.974948** on 402 historical cases per task. Recalculation from the
saved case predictions confirms these numbers; **this is not a fresh API run**.
No competition improvement or submission readiness has been established yet.

## 2. Official evaluation criteria

The reference implementation is [the official starter](https://gitlab.com/ifsoftware/hackapertus-starter),
inspected at commit `5e6957729b77a126006049305ad1ef241211a5b4`.
Its `evaluate.py` counts invalid/missing predictions as wrong, and scores Task A
evidence over **gold** labels 0/2. Only the first five submitted items count.
Normalization is NFKC, soft-hyphen removal, line-break dehyphenation, whitespace
collapse and lowercase. The 5,000-character cap applies **after normalization**;
`rapidfuzz.fuzz.partial_ratio >= 90` determines a match.

The inspected starter uses qualifying thresholds A=0.60, B=0.70. The user's goal
requires B=0.75; retain the stricter threshold for acceptance. The supplied goal
weights A/B at 60/40, with A components 50/20/15/15 and B components 70/15/15.
The efficiency references from other teams are unknown; no official total score
can be derived from local results. These weights still require verification
against the full competition instructions.

## 3. Starting repository architecture

PDF parsing → normalized BM25(claim) + 2× normalized BM25(vote title) → Top-12
pages → guarantee two pages from the named speaker's argument section → clip
long pages to a query-relevant 3,000-character window → Apertus v1.5 70B →
confidence/label decision rules → evidence passage IDs and page attribution.
Default prompt mode is `ids`; Thinking, translation, few-shot examples and
speaker filtering are disabled. Beginner uses the supplied reference directly.

## 4. Baseline results

| Historical task | Cases | Macro-F1 | Cross-language F1 | Reported errors |
|---|---:|---:|---:|---:|
| Advanced | 402 | 0.940236 | 0.934396 | 0 |
| Beginner | 402 | 0.974948 | 0.970607 | 0 |

Exact recomputed values, per-class/language metrics, token distributions,
archive hashes and historical settings are in
[`baseline.json`](../results/competition_optimization/baseline.json).
Fresh baseline inference, exact official evidence scoring and processing-time
measurement are pending. These historical test dates were subsequently used
in configuration decisions; they are observed data, not an untouched holdout.

## 5. Error analysis

The historical Advanced run has 24 incorrect labels; Beginner has 10.
The adaptive error review also identifies absence misread as contradiction,
swapped numerical assignments and a correct prediction lost by calibration.
Systematic categories A–L with evidence and competition impact remain pending.
Do not relabel gold cases simply to increase F1.

## 6. Evidence Hit@5 analysis

The existing evaluator measures word overlap with a denominator based on
**predicted** non-neutral labels. This does not reproduce official Hit@5.
`PredictionResult.to_official_dict()` substitutes page 1 for missing page
metadata and strips characters at quotation boundaries; audit and fix source
provenance before recommending an evidence change. No Hit@5 result is claimed.

## 7. Speaker attribution experiments

Preserve the existing two-page speaker boost. Historical speaker filtering,
speaker hints, few-shot prompts and a focused second judge produced regressions.
Do not repeat them without a distinct hypothesis supported by new error analysis.

## 8. Retrieval experiments

The adaptive validation experiment on 108 November 2024 cases has:

| Adaptive profile | Macro-F1 | Valid answers |
|---|---:|---:|
| E1, simplified original-query BM25 | 0.4666 | 108/108 |
| E6, multilingual candidate union + NLI reranking + neighbors | 0.7397 | 108/108 |
| E7, E6 + validation-fitted neutral calibration | 0.8517 | 108/108 |
| E8, binary NLI | 0.8831 | 56/108 |

E8's score covers successful responses only and cannot select a winner.
E7 leads the complete adaptive variants; its threshold was fitted on these same
cases. This adaptive E1 is **not** the production BM25 baseline. Different case
populations prevent a direct comparison with production F1 0.940236.

## 9. NLI experiments

No new prompt has been selected. First reproduce the production prompt on the
same cases as the candidate. Then use date-grouped validation folds for selection
and freeze every configuration before any further test evaluation.

## 10. Token optimization

Historical average prompt/completion counts are about 5,576/58 for Advanced and
2,523/55 for Beginner. These reports do not prove accounting for all unsuccessful
requests. New measurements must include retries, translation and auxiliary calls.
The historical client has response token caps; future API execution must respect
the user's instruction to omit output-token limits and document this difference.

## 11. Processing-time optimization

The evaluator pre-parses PDFs before starting its wall timer. Its model latency
is not official non-LLM processing time. Measure complete wall time minus the
**union** of in-flight request intervals, including cold source preprocessing.
Assess source-only PDF/index caching and use `/tmp` for runtime artifacts.

## 12. Ablation results

No new controlled ablation has run on this branch. Reproduce B0 first, then test
evidence fixes separately from retrieval, prompts and efficiency changes. Keep
historical prediction archives unchanged.

## 13. Per-language results

Historical language-pair and class metrics are recalculated in `baseline.json`.
Pair direction is claim→reference, whereas the official starter reports
source→claim. State the direction whenever presenting results.

## 14. Official score component comparison

Only historical label F1 and self-reported tokens have been verified. Exact
Hit@5, complete request accounting, non-LLM time and fresh fold measurements are
still missing. No validated replacement for the production baseline exists.

## 15. Estimated competition-score scenarios

Pending measured evidence/time components and verified competition rules.
Any future efficiency scenarios must be explicitly labeled simulations.

## 16. Recommended configuration

Retain production weighted BM25 Top-12 + speaker boost + `ids` NLI as the
provisional baseline. Adaptive E7 is an experimental candidate awaiting comparison
on identical cases, broader validation, and CPU/container measurements.

## 17. Reproduction instructions

From `track_2a`, using a Python environment with `requirements.txt` installed:

```bash
python -m unittest discover -s tests -v
python scripts/audit_competition_baseline.py \
  --output results/competition_optimization/baseline.json
```

The audit does not call the API. This session used Python 3.12.6, openai 3.24.0,
pydantic 2.13.5, pypdf 6.19.0, scikit-learn 1.9.1, rank-bm25 0.2.2,
python-dotenv 1.2.4 and typer 0.27.2. The 70 existing tests passed.

## 18. Docker compliance results

Static/local contract checks found that bare `--input/--output` fails because the
CLI currently requires the `run` subcommand. Only `LLM_BASE_URL/LLM_API_KEY` are
read; official `BASE_URL/API_KEY` are ignored. URL-based booklet resolution can
download at runtime. Full linux/amd64, read-only `/data`, offline runtime,
case-order and secret-exclusion checks are pending. No image has been published.

## 19. Known limitations

The pinned public JSONL revision `9ff08597fb79dc68cbb3af9eb1388f34d21223e6`
contains 1,488 rows; historical runs used 1,495. Matching by claim, language pair
and vote identifies 398/402 historical test rows unambiguously, without label
disagreements. Four rows are absent (`hf-0129`, `hf-0414`, `hf-1070`, `hf-1158`).
Do not silently join by row number or exclude missing gold evidence from the
official denominator. Resolve dataset provenance before claiming reproduction.
No private competition score or genuinely untouched holdout is available.

## 20. Final recommendation

The best complete adaptive validation profile is E7; the stronger historical
production baseline remains the recommended starting architecture. Next verify
dataset versions, reproduce both tasks with the official contract, measure exact
evidence and cost components, then compare isolated candidates on identical
date-grouped folds. Optimization and submission acceptance remain in progress.
