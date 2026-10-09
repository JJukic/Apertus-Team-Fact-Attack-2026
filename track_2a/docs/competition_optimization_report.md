# Competition optimization — final experiment report, 9 October 2026

## 1. Executive summary

Work is isolated on `experiment/competition-score-optimization`, from main
`a0e9e772dd388bdff4b8635ef0e09872ef705af6`. Adaptive WIP remains untouched.
Historical 402-case archives verify Advanced/Beginner F1 0.940236/0.974948;
fresh historical reproduction gives 0.945152/0.974948. They are observed data,
not private competition results or untouched holdouts.

Full identical-population validation now covers 1,488 cases per task, five folds
grouped by 20 voting dates. R2 leads measured quality: **Advanced F1 0.956237,
Beginner F1 0.982497, Hit@5 0.773737**. It preserves weighted BM25 Top-12,
speaker boost and the original NLI claim/prompt while adding translated-query
ranking union and source-grounded evidence. The initial R2 execution required
23 technical recoveries; every previous request remains in its costs.

R2 increases reported Advanced tokens 1.39%, regresses on two Advanced folds and
three language pairs, and has 148 attempts with unknown billed usage. Local mixed
timing does not establish separate official task efficiency. All source quotes
and CPU Docker contracts pass. All 91 R2 label discrepancies have source reviews;
two retain an explicit annotation/scope ambiguity without changing gold. R2 is
the recommended measured configuration, with the efficiency limitations below.
No official total score is claimed. The experiment is closed; further model
optimization requires a separate experiment rather than extending this run.

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
can be derived from local results. Projections use the user's supplied weights;
the pinned starter proves the metric/CLI contract, not the relative team-cost
references or an independently measured final leaderboard score.

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
reproduction. Full v1.1 confusions/evidence failures are also classified, with
source proofs are revalidated for the comparison candidates. All 91 R2 label
discrepancies are inspected: confirmed overlapping causes F=9, G=70, H=7,
I=15 and J=8; no E or K cause is confirmed. There are 185 category-L cases.
Counts are observations under unchanged gold and must not be added together.
Two reviews remain ambiguous: `v1.1-row-235-A` attributes a parliamentary
counterproposal argument to the committee, and `v1.1-row-757-A` has a pension
reform scope ambiguity. No gold labels are changed to improve reported F1.
Source review is Codex inspection, not independent human adjudication.
Remaining unreviewed C1/R1 semantic causes are explicitly recorded rather than
treated as absent. Every candidate's observed A–D/K/L confusions is classified.
Comparator source inspection covers **67/109 C1** and **66/100 R1** label
discrepancies, with 42/34 remaining cause reviews retained as future diagnostic
work. Closing this experiment does not turn those unknown causes into confirmed
retrieval or speaker failures. R2's confirmed quality-impact priority is
L/G/A/D/B/I/C/F/J/H; the reported counterfactual bounds assume no new errors.
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
the same five date-grouped folds. Both complete independently, with R2 recovery counted in cumulative costs. R2 improves overall Advanced F1 and evidence most, but regresses in two folds and some language pairs.

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
attribution. The full candidate runs preserve the production semantic prompt.
Additional prompt refinements and heavyweight NLI reranking are deferred: the
current improvement is independently measured, earlier related changes regressed,
and no new controlled benefit is established. They are not required to operate R2.

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
| R1, translated-query ranking | 0.950844 | 0.981830 | 0.765657 | 0 |
| R2, original/translated ranking union | **0.956237** | **0.982497** | **0.773737** | 0 |

The diagnostic cannot select a final configuration. C1 fixes three labels and
breaks seven relative to B0; evidence fixes/breaks are 292/7. Its five-fold
Advanced mean/std are 0.944348/0.030060; Beginner 0.981952/0.004897. Hit@5 improves
in the broader population without changing the NLI prompt, but generation is
not fully repeatable and the small F1 regressions are explicitly retained.

## 13. Per-language results

Pair direction below is **source→claim**. Each task has 1,488 cases.

| Advanced source→claim | B0 F1 | R2 F1 | Difference |
|---|---:|---:|---:|
| fr->de | 0.962095 | 0.967547 | +0.005452 |
| de->de | 0.947035 | 0.966544 | +0.019509 |
| it->de | 0.946790 | 0.987128 | +0.040338 |
| it->fr | 0.986111 | 0.986111 | +0.000000 |
| fr->fr | 0.924969 | 0.917468 | -0.007500 |
| de->fr | 0.886882 | 0.947210 | +0.060328 |
| de->it | 0.931615 | 0.924721 | -0.006894 |
| fr->it | 0.962824 | 0.941651 | -0.021173 |
| it->it | 0.973648 | 0.973648 | +0.000000 |

R2 Advanced fold mean/std is **0.955664/0.033242**; cross-language F1 is
**0.959217**. F1 regresses in folds 1 and 3, while Hit@5 improves in all five.
The largest Advanced language-pair decline is fr→it, **−0.021173**. Beginner
de→de decreases 0.006710. Beginner prompts/rules are unchanged; small Beginner
score changes are not credited to retrieval. All class/pair/fold metrics and
paired case IDs are retained in `experiments.json`.

## 14. Official score component comparison

All candidates use identical inputs and date-grouped folds. B0 retains its one
invalid output and is excluded by the complete-output acceptance gate.

| Configuration | Advanced F1 | Beginner F1 | Hit@5 | Invalid outputs |
|---|---:|---:|---:|---:|
| B0 | 0.947699 | 0.981830 | 0.467677 | 1 |
| C1 | 0.944034 | 0.982497 | 0.755556 | 0 |
| R1 | 0.950844 | 0.981830 | 0.765657 | 0 |
| R2 | 0.956237 | 0.982497 | 0.773737 | 0 |

| Configuration | API attempts | Unknown usage | Known input tokens | Known output tokens | Local mixed non-LLM seconds |
|---|---:|---:|---:|---:|---:|
| B0 | 3,001 | 26 | 11,726,698 | 195,209 | 21.065644 |
| C1 | 2,995 | 19 | 11,732,096 | 191,322 | 3.013437 |
| R1 | 3,749 | 24 | 11,788,990 | 228,397 | 4.365191 |
| R2 | 3,873 | 148 | 11,818,740 | 214,805 | 15.250753 |

Token totals are lower bounds. R2 initially has 23 failures, preserved before
repeating only those cases under identical frozen settings/sources. All previous
failed/rejected translation and NLI attempts remain counted. R2 fixes/breaks
30/16 labels and 349/46 evidence hits versus B0. Full label errors are B0 105,
C1 109, R1 100 and R2 91 (invalid B0 included).
The runs have different recovery, cache and host conditions. Mixed timers are
not official separate-task efficiencies or a controlled processing-time comparison.

## 15. Estimated competition-score scenarios

The user-supplied quality weights yield contributions of 0.615343 for B0,
0.648976 for C1, 0.652044 for R1 and **0.654819 for R2**. R2 increases this
quality contribution by **0.039475** versus B0, before unknown efficiency.

At equal assumed task processing times and competitor cost references of
0.1/0.25/0.5/1.0 times B0, simulated R2 totals are
0.684698/0.729516/0.804214/0.953585. These are **not official scores**, use
known-token lower bounds and do not assign mixed timing to separate tasks.
C1/R1/R2 all remain on the known quality/token Pareto frontier. Unknown failed
usage and actual proxy timing can change the overall ranking. A measured quality
leader is not proof of the private competition total-score winner.

## 16. Recommended configuration

Recommend R2 as the measured quality leader. Keep production weighted BM25 Top-12,
two-page speaker boost, original NLI claim and `ids` prompt. Its flags are:

```text
RETRIEVAL_QUERY_MODE=union
EVIDENCE_POLICY=raw_pages_and_blocks
CACHE_SINGLE_FLIGHT=true
SOURCE_PAGES_LAZY=true
LLM_STREAMING=true
LLM_JSON_REPAIR=true
NLI_BATCH_WORKERS=4
```

The original baseline remains selectable with original queries, legacy evidence
and the added reliability/cache flags disabled. Thinking, forced second answers,
semantic prompt changes, speaker filtering and calibration are not promoted.
C1 is the simpler alternative if proxy-measured translation costs outweigh R2's
quality gain. The default remains the selectable baseline; apply the flags above
explicitly for R2. Four CLI workers preserve input order and use the same
concurrency as the native full evaluation. They do not change the NLI prompt.

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
The final source suite passes **115 tests**, including
translation invariants and rejection of invented/changed manual source proofs.
Fresh runs use openai 3.26.1, typer 0.27.3 and PyMuPDF 1.28.2; initial audit
versions above describe the earlier environment. Exact fresh dependencies are
in `requirements-lock.txt`. The source-only CPU image passes all 115 tests.

For a new full reproduction, set `BASE_URL` and `API_KEY` securely in the host
environment. Commands below run from `track_2a`; use fresh output directories
and install `requirements-lock.txt`. The preparation step downloads pinned
sources **outside** prediction; inference itself never downloads them.

```bash
REPRO_DIR=.cache/competition/reproduction-2026-10-09
test ! -e "$REPRO_DIR" || exit 1
python scripts/prepare_competition_cases.py --version v1.1 --scope all \
  --download-booklets --output "$REPRO_DIR/data"

export NLI_STRATEGY=hybrid NLI_TOP_K=12 PROMPT_MODE=ids
export PASSAGE_CHARS=0 PAGE_MAX_CHARS=3000 SPEAKER_BOOST=2
export THINKING=false THINKING_BUDGET=0 IDS_REASON=false TRANSLATE_CLAIM=false
export SPEAKER_AWARE=false SPEAKER_HINT=false FEW_SHOT=false NUMERIC_OVERRIDE=false
export LLM_NAME=swiss-ai/Apertus-v1.5-70B-thinking

EVIDENCE_POLICY=legacy CACHE_SINGLE_FLIGHT=false SOURCE_PAGES_LAZY=false \
LLM_STREAMING=false LLM_JSON_REPAIR=false RETRIEVAL_QUERY_MODE=original \
BOOKLET_CACHE_DIR="$REPRO_DIR/B0/parsed" QUERY_TRANSLATION_CACHE_DIR="$REPRO_DIR/B0/queries" \
python scripts/run_competition_baseline.py --cases "$REPRO_DIR/data/cases.jsonl" \
  --output "$REPRO_DIR/B0" --workers 4

export EVIDENCE_POLICY=raw_pages_and_blocks CACHE_SINGLE_FLIGHT=true SOURCE_PAGES_LAZY=true
export LLM_STREAMING=true LLM_JSON_REPAIR=true
export NLI_BATCH_WORKERS=4
for candidate in C1 R1 R2; do
  case "$candidate" in
    C1) query_mode=original ;;
    R1) query_mode=translated ;;
    R2) query_mode=union ;;
  esac
  RETRIEVAL_QUERY_MODE="$query_mode" BOOKLET_CACHE_DIR="$REPRO_DIR/$candidate/parsed" \
  QUERY_TRANSLATION_CACHE_DIR="$REPRO_DIR/$candidate/queries" \
  python scripts/run_competition_baseline.py --cases "$REPRO_DIR/data/cases.jsonl" \
    --output "$REPRO_DIR/$candidate" --workers 4
done

python scripts/compare_competition_runs.py --cases-dir "$REPRO_DIR/data" \
  --run B0="$REPRO_DIR/B0/predictions.jsonl" --measurement B0="$REPRO_DIR/B0/measurement.json" \
  --run C1="$REPRO_DIR/C1/predictions.jsonl" --measurement C1="$REPRO_DIR/C1/measurement.json" \
  --run R1="$REPRO_DIR/R1/predictions.jsonl" --measurement R1="$REPRO_DIR/R1/measurement.json" \
  --run R2="$REPRO_DIR/R2/predictions.jsonl" --measurement R2="$REPRO_DIR/R2/measurement.json" \
  --time-factor C1=1,1 --time-factor R1=1,1 --time-factor R2=1,1 \
  --output "$REPRO_DIR/comparison"

python scripts/validate_submission_sources.py --cases "$REPRO_DIR/data/cases.jsonl" \
  --predictions "$REPRO_DIR/R2/predictions.jsonl" --output "$REPRO_DIR/R2/source_validation.json"
python scripts/analyze_competition_errors.py --cases-dir "$REPRO_DIR/data" \
  --predictions "$REPRO_DIR/R2/predictions.jsonl" --records-dir "$REPRO_DIR/R2/predictions" \
  --output "$REPRO_DIR/R2/error_audit"
```

Different remote answers can produce different F1 even with unchanged requests.
If retrying technical errors, first preserve the terminal run/measurement/journal
and failed records, then rerun the **identical** command with `--retry-errors`.
Keep all sessions and unknown usage. A run's signature must match; editing
sources/settings requires a new run, not bypassing identity checks. The B0
command reproduces the selectable baseline configuration; exact historical
sources, response hashes, recovery overlays and counters remain in the recorded
original run manifests. No new run is claimed to reproduce gateway failures or
remote text byte for byte. Scenario time factors above are assumptions.
Saved source annotations are bound to the reviewed gold/predicted labels and
context pages; do not transfer them blindly to a new API run. Use `--annotations`
only after verifying those conditions or conducting a fresh source review.

For the small Docker contract check, copy only the committed source-only fixture
next to the prepared booklet directory. The helper copies only case inputs and
their PDFs into `/data`; it never mounts `expected-labels.jsonl`.

```bash
docker build --platform linux/amd64 -t fact-attack:test ..
docker run --rm --platform linux/amd64 --network none --read-only --tmpfs /tmp fact-attack:test test
cp tests/fixtures/competition_contract_cases.jsonl "$REPRO_DIR/data/contract_fixture.jsonl"
python scripts/check_competition_docker.py --cases "$REPRO_DIR/data/contract_fixture.jsonl" \
  --output "$REPRO_DIR/docker-contract" --image fact-attack:test \
  --api-mode translated --api-mode union
```

Omit both `--api-mode` flags for the offline container contract check. Do not
copy host credentials, development labels or experiment journals into the image.

## 18. Docker compliance results

The pinned CPU linux/amd64 image
`sha256:87f8111a6ac4ed0a8ae980005749b533e93c8bcef7835ea7dec2d96837b300be`
passes **115 tests** in a read-only, network-disabled container. A 12-case fixture
covers both tasks, six language directions, an empty ID and Unicode IDs. Offline
forward/reverse predictions agree, excluding timing fields. Both R1/R2 real-API
modes pass bare `--input/--output`, `/data:ro`, writable `/output`, `/tmp` caches,
physical source quotation checks and complete valid JSONL regardless of suffix.
The earlier R1/R2 check journals **30 successful API attempts**, zero unknown usage.
The final image is checked directly in R2 mode with **four CLI workers**:
**15 successful requests**, zero unknown usage, 44,201 input / 839 output tokens.
Only `src/cli.py` and `src/config.py` differ from the earlier image's inference
sources, adding configurable batch concurrency and preserving the baseline's
single-worker default. Retrieval, prompt, evidence and client sources are unchanged.
The small mixed fixture measures local wall/SDK-union/non-LLM times of
30.398/19.385/11.012 seconds. These are contract-check timings, not full-dataset
processing efficiency or evidence of a separate-task speedup.
API repeatability itself is not assumed. Input hashes are unchanged.

All calls use runtime `BASE_URL/API_KEY`, Apertus v1.5 and no client output cap.
Image inspection verifies inference-source hashes, no `.env`, expected-label
files, local model weights or embedded runtime endpoint/key. Source-only demos
replace labeled development demos only in Docker COPY. Original development
data remain unchanged. All container inference checks load no gold.
`docker_validation.json` retains the full report; earlier failed verifier/test
attempts are preserved separately. No image is published.

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

R2 currently has the strongest measured quality: Advanced F1 increases
**0.008538**, Beginner **0.000667**, Hit@5 **0.306061** versus full B0.
Reported Advanced tokens increase 1.39%; neither complete billed-token reduction
nor controlled official processing-time reduction is established. C1 supplies
most evidence improvement without translation; R1/R2 add multilingual ranking
at extra API cost. Historical compact contexts, speaker filtering and adaptive
calibration are not selected from incomparable or regressing results.

Recommend **R2 with the explicit flags in section 16** for the evaluated CPU/CLI
submission contract. All outputs and physical source quotes are validated;
115 host tests and 115 CPU-container tests pass. All **91 R2 label discrepancies**
have source reviews. Two gold/speaker-scope ambiguities remain flagged without
relabeling; the review is not independent human adjudication. Comparator semantic
review coverage and remaining IDs are retained in `error_analysis.json`.

The experiment closes here. In a new experiment, the highest source-backed
counterfactual opportunities are category L (evidence mismatch with correct label,
quality contribution at most +0.022424 if every case is fixed without regressions)
and G (available evidence misinterpreted, at most +0.013865). These bounds overlap
with other categories and are not measured improvements. Further prompt/reranker
work is deferred; no speculative change is included in the recommendation.

The solution passes the tested local submission contract. Private-set quality,
the true overall competition winner and official proxy efficiency remain
unmeasured external outcomes. Baseline and C1 remain selectable alternatives.
The branch can be uploaded to GitHub as requested; main is not merged and the
container is not published.


The closure checks are traceable to these artifacts:

| Acceptance item | Verified evidence |
|---|---|
| Baseline reproduction and unchanged historical archives | `baseline.json`, pinned dataset/commit/source hashes |
| Identical populations and five grouped folds | `experiments.json`, `comparison.csv`; all four runs recomputed from saved predictions |
| Official Macro-F1 and exact Hit@5 rules | Vendored official evaluator plus normalization/first-five/denominator tests |
| Source quotations, IDs, labels and pages | Full candidate source-contract reports embedded in `evidence_analysis.json` |
| CPU processing profile and honest request accounting | `processing_profile.json`, cumulative attempt/unknown-usage measurements |
| Configurable selection and documented tradeoffs | `best_config.json`, sections 12–16; R2 selected, baseline retained |
| Automated tests and mixed official container contract | `docker_validation.json`: 115 host + 115 CPU tests, source-only fixture, four workers |
| Error categories and remaining diagnostic limits | `error_analysis.json`, source-review JSONs; no gold relabeling |
| Reproduction and deployment instructions | Section 17 and both READMEs; runtime keys, existing PDFs, `/data:ro`, `/tmp` |
| Isolated branch and GitHub upload | Commit/push on `experiment/competition-score-optimization`; no merge or image publication |
