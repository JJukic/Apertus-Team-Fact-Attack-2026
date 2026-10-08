# Competition optimization — audit, 8 October 2026

## 1. Executive summary

Work is isolated on `experiment/competition-score-optimization`, starting from
latest `origin/main`, commit `a0e9e772dd388bdff4b8635ef0e09872ef705af6`.
The adaptive branch and its uncommitted work remain separate.

The strongest documented production baseline has Advanced Macro-F1 **0.940236**
and Beginner **0.974948** on 402 historical cases per task. Recalculation from the
saved case predictions confirms these numbers; **this is not a fresh API run**.
A fresh reproduction now covers all 804 historical task cases: Advanced **0.945152**,
Beginner **0.974948**, baseline official Hit@5 **0.361940**. Source-only evidence
postprocessing raises Hit@5 to **0.720149** on those same predictions without
changing labels or adding model calls. The independently executed C1 candidate
now covers all 2,976 mixed v1.1 cases: Advanced **0.944034**, Beginner **0.982497**,
Hit@5 **0.755556**, zero invalid answers. Compared with the broader B0 run,
Advanced F1 decreases by 0.003665 while Hit@5 increases by 0.287879. This is a
competition-relevant trade-off, not an F1 improvement. R1/R2 retrieval experiments
and final selection remain in progress; submission readiness is not established.

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
Fresh 804-case inference completed with no remaining operational errors.
The uncapped transport reproduction has Advanced 0.945152 and Beginner 0.974948.
This small F1 difference is not attributed to an architectural improvement:
response limits were removed at the user's request and the endpoint is not fully
repeatable. The measured 823 attempts include 19 failures with unknown usage;
known totals 3,255,939 input / 53,663 output tokens are lower bounds. Summed local
SDK-based non-LLM time across original run and recovery sessions is 19.0545 s.
This is neither a single-run processing improvement nor official proxy timing. These historical test dates were subsequently used
in configuration decisions; they are observed data, not an untouched holdout.

## 5. Error analysis

The historical Advanced run has 24 incorrect labels; Beginner has 10.
The adaptive error review also identifies absence misread as contradiction,
swapped numerical assignments and a correct prediction lost by calibration.
The fresh 804-case audit finds 32 label errors: A=9, B=4, C=6, D=13.
Category L has 153 correctly labeled Advanced cases with unmatched evidence.
All 32 label errors were inspected against the physical PDF and actual context.
Source-backed overlapping causes are E=2, F=8, G=18, H=6, I=2; J remains
unconfirmed. The resulting priority starts L/G/D/A/F/H. Eight failures omit
decisive source pages, six across languages. Eighteen have relevant information
available but incorrect interpretation. The two speaker errors use government
arguments for committee claims. Keyword flags and fuzzy coverage are not proven
causes; these counts overlap and must not be added together.
[`competition_error_source_review_v1.json`](competition_error_source_review_v1.json)
preserves exact original text blocks, physical pages and PDF/reference hashes.
The audit verifies those proofs. One neutral pension-reform claim has a wording
and scope ambiguity requiring independent adjudication; its gold is unchanged
and no semantic cause is asserted. This review concerns the historical 804-case
reproduction; broader v1.1 source-cause review remains open.
See `results/competition_optimization/error_analysis.json` and the reproducible
`analyze_competition_errors.py` utility.
Do not relabel gold cases simply to increase F1.

## 6. Evidence Hit@5 analysis

The existing evaluator measures word overlap with a denominator based on
**predicted** non-neutral labels. This does not reproduce official Hit@5.
The original exporter substituted page 1 for missing metadata and stripped
quotation boundaries. The experimental exporter now preserves digits, quote
marks, Unicode and whitespace, canonicalizes labels, retains explicit empty IDs,
deduplicates the first five valid items, and never invents page 1. Beginner pages
are null. Strict official mode rejects missing Advanced non-neutral evidence.

| Fixed fresh predictions, 402 Advanced cases | Macro-F1 | Hit@5 | Evidence fixed / broken |
|---|---:|---:|---:|
| B0 original output | 0.945152 | 0.361940 | 0 / 0 |
| E1 complete original pages | 0.945152 | 0.582090 | 59 / 0 |
| E2 original pages + ranked complete text blocks | 0.945152 | 0.720149 | 97 / 1 |

Candidate construction uses attributed source pages, including the baseline's
explicit top-ranked fallback when no passage IDs were returned. It never reads
gold references. The official evaluator receives gold only after predictions
exist. Source construction and its normalizer are frozen and hashed per ablation.
E2's one lost hit is the 2021 terrorism case: retaining three whole pages plus two
blocks displaced the fourth cited detailed argument page. This is a documented
ordering regression; it is not corrected by a case-ID-specific rule.
For the full independent C1 run, all **4,600 submitted Advanced quotations**
were separately checked against their actual physical PDF page or complete
original text block. Every output ID, label name, first-five limit and required
non-neutral evidence passes; Beginner evidence pages are null. This source
contract check reads no expected labels and does not imply the quotes support
every predicted NLI label semantically.

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

The distinct production retrieval experiment preserves the original hypothesis
for NLI. R1 uses a source-language translation only for claim BM25; R2 combines
original/translated normalized claim scores by maximum before the unchanged
2× vote-title score and speaker boost. Same-language claims skip translation;
changed numeric values or negation presence fall back to the original query.
The guard is conservative, not proof of semantic equivalence. All translation
attempts, rejected translations and cache acquisition tokens are journaled.
Both fresh full-population runs use separate cold source/translation caches on
the same five date-grouped folds. R1 is running; R2 follows sequentially.

## 9. NLI experiments

No semantic prompt has been selected. N1 is a separate optional reliability
experiment: if a JSON NLI request exhausts transport retries, one supported
`response_format={"type":"json_object"}` generation is counted separately.
Plain-text translation is excluded. The persistent v1.1 row 289 Advanced failure
had identical primary request hashes to the frozen baseline; two normal attempts
failed at about 61 s, then JSON-constrained generation completed in 1.15 s with
51 output tokens. Its valid label is not asserted to be correct. An assembled
full-population replay is diagnostic only; C1 executed independently with full
coverage. This N1 name denotes transport reliability, not structured speaker
attribution. First reproduce the production prompt on the
same cases as the candidate. Then use date-grouped validation folds for selection
and freeze every configuration before any further test evaluation.

## 10. Token optimization

Historical average prompt/completion counts are about 5,576/58 for Advanced and
2,523/55 for Beginner. These reports do not prove accounting for all unsuccessful
requests. New measurements must include retries, translation and auxiliary calls.
The client now omits both output-token cap fields for every remote call,
including translations and Thinking. Old hard Thinking budgets no longer force
a second answer. Prompt/context/decision changes are independent of this user
instruction. Every remote attempt is journaled, including retries and JSON repair;
unknown failed-request usage remains unknown rather than zero.

## 11. Processing-time optimization

The evaluator pre-parses PDFs before starting its wall timer. Its model latency
is not official non-LLM processing time. Measure complete wall time minus the
**union** of in-flight request intervals, including cold source preprocessing.
Source-only PDF/index caching defaults to `/tmp`. Optional single-flight caching
prevents concurrent duplicate parsing and invalidates changed files. Optional
lazy evidence extraction reads only requested physical pages; it must preserve
the eager extractor's outputs exactly. Tests verify both properties. Whole-PDF
E2 evidence construction took 20.838 ms/case mean, 0.0096 median, 1.5439 p95 on
the mixed 804-case offline replay; cold parsing dominates the mean. Those are
component measurements, not official non-LLM time. The lazy replay produces
byte-identical predictions for all 804 cases and all three policies. For E2 it
takes 7.731 ms/case mean, 0.0105 median, 44.661 p95: mean falls by 62.9%, while
p95 increases because eager extraction concentrates all work into fewer cold
booklet loads. This is a component trade-off, not an official efficiency ratio.
C1's full cold mixed run records 3.0134 s local wall-minus-SDK-union time and
228,245,504 bytes peak RSS. B0's 21.0656 s sums multiple recovery invocations;
the comparison does not isolate a pure caching effect. Complete source-only
profiling now covers all 60 PDFs and 1,488 Advanced rankings. Its per-PDF
cold parse/cache-write mean is 1,147.36 ms versus 4.74 ms for cached loading.
Ranking per Advanced case has mean/median/p95 2.825/0.701/8.677 ms; clipping
1.747/0.484/6.277 ms. Sequential eager/lazy evidence reconstruction from saved
C1 prediction pages has equal output hashes and means 18.426/10.123 ms per
mixed case. Those stages, distributions, source hashes and peak profiler RSS
155,435,008 bytes are in `processing_profile.json`. They were measured on a
shared local host and are not separate-task proxy efficiency scores.

## 12. Ablation results

B0/E1/E2 fixed-prediction ablations are complete on the historical 804 mixed cases.
Machine-readable paired changes, classes and language metrics are in
`experiments.json` / `comparison.csv`. The full v1.1 baseline executed all 2,976
mixed cases from 20 dates; one persistent gateway failure remains in B0 and is
counted as invalid, not dropped. The independently executed C1 combination uses
E2, source-only caching and conditional JSON repair on those identical cases.
Five date-grouped validation folds keep all languages of each date together.
An assembled N1 recovery replay is explicitly not a fresh full candidate run. Keep
historical prediction archives unchanged.

| Full v1.1 run, 1,488 cases per task | Advanced F1 | Beginner F1 | Hit@5 | Invalid |
|---|---:|---:|---:|---:|
| B0, original baseline | 0.947699 | 0.981830 | 0.467677 | 1 |
| N1 + E2, assembled diagnostic | 0.947389 | 0.981830 | 0.758586 | 0 |
| C1, independent fresh execution | 0.944034 | 0.982497 | 0.755556 | 0 |

The diagnostic cannot select a final configuration. C1 fixes three labels and
breaks seven relative to B0; evidence fixes/breaks are 292/7. Its five-fold
Advanced mean/std are 0.944348/0.030060; Beginner 0.981952/0.004897. Hit@5 improves
in the broader population without changing the NLI prompt, but generation is
not fully repeatable and the small F1 regressions are explicitly retained.

## 13. Per-language results

Historical language-pair and class metrics are recalculated in `baseline.json`.
Pair direction is claim→reference, whereas the official starter reports
source→claim. State the direction whenever presenting results.

## 14. Official score component comparison

Historical and fresh label F1, exact official Hit@5, paired evidence differences,
request attempts and local union-of-interval timing are now measured. Failed
requests lack proxy usage, so complete billable token totals remain unknown.
C1's full grouped comparison is complete; final selection waits for R1/R2 and
the submission checks. Advanced source→claim regressions are de→de −0.006689,
it→de −0.006273, fr→fr −0.012409 and de→fr −0.006671. Beginner de→de decreases
by 0.006710. Class-level changes, paired IDs and all folds are in `experiments.json`.
The C1 journal counts 2,995 attempts, 19 with unknown usage; known input/output
totals are 11,732,096/191,322. These are lower bounds, not complete proxy costs.

## 15. Estimated competition-score scenarios

Under user-supplied weights, the historical E2 evidence gain alone contributes
`0.60 * 0.20 * (0.720149 - 0.361940) = 0.042985` to the simulated total at unchanged
other components. Actual time-efficiency costs must still be measured. This is
an isolated quality contribution, not an official final competition score.
For the independently executed C1 comparison, the measured quality-component
gain is `0.30*delta_F1_A + 0.12*delta_Hit5 + 0.28*delta_F1_B = 0.033633`.
The scenario utility uses explicitly assumed separate-task time factors; it
never assigns the mixed concurrent timer to Task A/B official efficiency.

## 16. Recommended configuration

Retain production weighted BM25 Top-12 + speaker boost + `ids` NLI as the
provisional architecture. C1 is the strongest complete fresh competition-quality
candidate so far; it remains provisional until R1/R2 and final submission checks.
Adaptive E7 is not promoted from its single-date, validation-calibrated measurement.

## 17. Reproduction instructions

From `track_2a`, using a Python environment with `requirements.txt` installed:

```bash
python -m unittest discover -s tests -v
python scripts/audit_competition_baseline.py \
  --output results/competition_optimization/baseline.json
```

The audit does not call the API. This session used Python 3.12.6, openai 3.24.0,
pydantic 2.13.5, pypdf 6.19.0, scikit-learn 1.9.1, rank-bm25 0.2.2,
python-dotenv 1.2.4 and typer 0.27.2. The initial 70 existing tests passed. The evidence/contract implementation
passed 95 tests and six subtests; 98 passed after the first transport tests.
The current source suite passes **111 tests and six subtests**, including
translation invariants and rejection of invented/changed manual source proofs.
Fresh runs use openai 3.26.1, typer 0.27.3 and PyMuPDF 1.28.2; initial audit
versions above describe the earlier environment. Exact fresh dependencies are
in `requirements-lock.txt`. Final image verification remains pending.

## 18. Docker compliance results

Bare `--input/--output` now routes to official JSONL mode regardless of output
extension. Official `BASE_URL/API_KEY` and legacy aliases work; runtime values
outrank dotenv values even across aliases. Inference source resolution never
downloads missing PDFs or falls back from a missing booklet to a reference.
Root and track Dockerfiles build CPU `linux/amd64`, copy only explicit code/demo
assets, omit experiment gold/private journals/secrets, and use `/tmp` caches.

The first image build passed; 95 tests / six subtests passed inside a read-only,
network-disabled container. Mixed offline inference passed with `/data:ro` and
`/output` writable. Two genuine Apertus calls through runtime `BASE_URL/API_KEY`
passed the same bare-argument contract with zero failures, and hashes prove input
files were unchanged. The image predates the JSON repair/lazy additions, so final
container verification remains required. No image has been published.

## 19. Known limitations

The pinned public JSONL revision `9ff08597fb79dc68cbb3af9eb1388f34d21223e6`
contains 1,488 rows; historical runs used 1,495. Matching by claim, language pair
and vote identifies 398/402 historical test rows unambiguously, without label
disagreements. Four rows are absent (`hf-0129`, `hf-0414`, `hf-1070`, `hf-1158`).
The pinned v1.0 revision `adca9a5ee87e9813cef80b305c3b12e3e8a05959` restores all
402 historical rows and agrees with every claim, vote, language and label.
Historical reproduction therefore uses v1.0, while broader validation uses
v1.1 with all 1,488 rows and 60 source PDFs. SHA256s and manifests are retained.
Never join across these versions by row number or exclude missing evidence from
the official denominator.
No private competition score or genuinely untouched holdout is available.

## 20. Final recommendation

The best complete adaptive validation profile is E7; the stronger historical
production baseline remains the recommended starting architecture. Next verify
dataset versions, reproduce both tasks with the official contract, measure exact
evidence and cost components, then compare isolated candidates on identical
date-grouped folds. Optimization and submission acceptance remain in progress.
