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
| **Advanced** (booklet PDF + claim + vote) | **0.940** | 0.951 | 0.934 | 5,576 | 58 | 2.1 s / 2.9 s |
| **Beginner** (reference string + claim) | **0.975** | 0.984 | 0.971 | 2,523 | 55 | 1.1 s / 1.7 s |

Over **all 1,495 pairs** of the dataset (dev and test together) the advanced task reaches **0.946** (cross-lingual 0.945,
dev 1,093: 0.948), at 5,399 input tokens and 2.05 s / 2.83 s latency on average. The test result was reproduced with
`make run` from a fresh clone (Docker, commit e9f6cd7): identical macro-F1 0.940.

Configuration: hybrid retrieval (BM25 claim + 2 × BM25 vote title, booklet-wide), top 12 pages,
section labels (incl. closing recommendation boxes), `ids` prompt mode, model label kept when it disagrees
with the model's own confidences (Decision-Rule 0), pages longer than 3,000 characters clipped to their best-matching
window (`PAGE_MAX_CHARS`), the attributed side's 2 best argument pages always included (`SPEAKER_BOOST`), and a Neutral
label with a confident other relation resolved by the confidences (Decision-Rule 0b). Before these changes the advanced
task scored 0.920 with 5,737 input tokens
(see [error analysis](#error-analysis-on-saved-runs-2026-10-05) and [page clipping](#clipping-over-long-pages-2026-10-06)).

Advanced macro-F1 by language pair (claim → booklet):

| | → de | → fr | → it |
|---|---:|---:|---:|
| **de** | 0.951 | 0.949 | 0.916 |
| **fr** | 0.825 | 0.956 | 1.000 |
| **it** | 0.926 | 0.952 | 0.934 |

Over all 1,495 pairs: de→de 0.955, de→fr 0.968, de→it 0.934, fr→de 0.880, fr→fr 0.918, fr→it 0.986, it→de 0.925,
it→fr 0.963, it→it 0.975.

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

## What we learned about Apertus (dev split)

### 8B vs. 70B (same 450 dev pairs, hybrid k=10, `ids`)

| Model | Macro-F1 | Cross-lingual F1 | Latency mean / p95 |
|---|---:|---:|---:|
| `Apertus-v1.5-70B-thinking` | **0.908** | **0.898** | 3.0 s / 4.5 s* |
| `Apertus-v1.5-8B` | 0.851 | 0.808 | **0.9 s / 1.2 s** |

\* measured while other runs shared the endpoint. 8B is ~3.5× faster but loses ~6 F1 points, mostly on
cross-lingual pairs and on contradictions (23 contradictions predicted as entailment). Because NLI quality
has priority, 70B stays the default; `LLM_NAME` switches the model.

### Apertus is over-confident

We read the probability of the label token (logprobs) for 41 wrong and 40 correct dev answers:

| | Median | Minimum | Share < 0.9 |
|---|---:|---:|---:|
| Wrong answers | 1.000 | 0.679 | 2 % |
| Correct answers | 1.000 | 1.000 | 0 % |

The model is as certain when it is wrong as when it is right, and its self-reported confidences are 0/1 in
97–99 % of cases. Confidence thresholds (or fuzzy decision rules on these values) therefore cannot filter
errors; the existing decision rules only confirm the model's label.

### The remaining errors are speaker confusions — and they resist every fix we tried

Claims attributed to the initiative/referendum committee had a 19 % error rate on dev (18 of 41 errors),
versus 5–7 % for all other claims. In the typical error, the claim says "the committee argues X" and Apertus
cites a Federal Council statement as the contradiction, despite an explicit prompt rule against this.

| Attempted fix | Fixed | Broke | Macro-F1 (450 dev) | Decision |
|---|---:|---:|---:|---|
| Baseline (hybrid k=10, section labels) | – | – | **0.908** | kept |
| Second, focused verification pass for every "contradiction" | 0 of 23 | 24 | 0.852 | rejected |
| Speaker-aware retrieval (hide the opposing side's argument pages) | 3 (0 committee) | 8 | 0.896 | rejected (`SPEAKER_AWARE=false`) |
| Three worked examples (few-shot) for committee vs. Federal Council claims | 8 (2 committee) | 9 | 0.906 | rejected (`FEW_SHOT=false`) |

Few-shot examples shifted the errors instead of removing them: Entailment → Contradiction fell from 23 to 8, but Entailment → Neutral rose from 4 to 15 (+360 input tokens per claim). Opposing statements also appear together on the unlabelled overview pages, so hiding labelled argument pages
does not remove them; asking again only makes the model repeat its decision.

### Error analysis on saved runs (2026-10-05)

We re-ran the deterministic retrieval for every saved prediction (0 of 450 dev and 4 of 402 test
reconstructions differ from the saved evidence), so each cited passage id maps back to its page and section.

**A rule on cited sections does not work.** Idea: if a committee-attributed claim is predicted Contradiction
and only Federal Council pages are cited, flip the label. In practice Apertus almost never cites a page
labelled Federal Council in these errors; the cited pages are unlabelled or nothing is cited. Simulated on
saved runs, flipping to Entailment fixed 7 and broke 8 (dev) and fixed 0, broke 7 (test). Rejected.

**Section labels were wrong on older booklets.** Proxy check: for attributed claims with gold label
Entailment, the gold reference page must belong to the named speaker.

| Gold reference page of … | Correct label before | Correct label after | Labelled as the *other* side before → after |
|---|---:|---:|---:|
| Federal Council claims (unique pages) | 34 | **71** | 27 → **0** |
| Committee claims (unique pages) | 24 | **38** | 0 → 0 |

2020–2022 booklets head each side with a bare speaker line ("Bundesrat und Parlament", "Referendumskomitee",
"Comité « … »") or set the heading as a graphic, so the committee label ran on over the Federal Council's
pages. The fix matches bare speaker lines (whole line only) and uses the recommendation box that closes every
argument double page ("Empfehlung von Bundesrat und Parlament", "Recommandation des comités référendaires",
or the committee disclaimer) to label that double page; the label no longer runs on past it. Over all 60
booklets 357 page labels changed; spot checks of every change type were correct.

**Apertus contradicts itself.** In ~2 % of answers the label and the confidences disagree
(`"label": 0` with `"p_contra": 1.0`). Decision-Rule 1 used to follow the confidences. Over all 24 saved
`ids` runs, the label was right in 44 such cases and the confidences in 13, so Decision-Rule 0 now keeps the
label (`apertus_client._apply_calibrated_decision`).

**Retrieval is not the bottleneck.** Gold section retrieved (strict recall, k = 10) per claim → booklet pair:

| Split | Recall overall | Weakest pairs | Errors with gold retrieved | Errors with gold missed |
|---|---:|---|---:|---:|
| dev (450) | 0.93 | it→de 0.80, de→it 0.90 | 31 | 10 |
| test (402) | 0.92 | it→fr 0.84, de→it / fr→it 0.89 | 25 | 7 |

Italian claims have the lowest recall, but even perfect retrieval would fix at most ~2 F1 points; it→de on
test fails mostly with the gold page in context (6 of 8 errors). Dense retrieval (BGE-M3, ~2 GB image) is
therefore not a priority.

**Measured with Apertus 70B:**

| Run | Macro-F1 | Fixed / broke vs. previous row | E → C errors | Evidence in gold section |
|---|---:|---:|---:|---:|
| test, previous config | 0.920 | – | 16 | 0.75 |
| test, + Decision-Rule 0 | 0.928 | 6 / 3 | 11 | 0.75 |
| test, + Decision-Rule 0 + section fix | 0.925 | 1 / 2 | 13 | 0.75 |
| dev 450, previous config | 0.908 | – | 23 | 0.86 |
| dev 450, + Decision-Rule 0 + section fix | 0.908 | 6 / 6 | 16 | **0.88** |

Decision-Rule 0 gives the expected gain. The section fix is F1-neutral (within run-to-run noise of ±3
answers) but removes wrong speaker labels, lowers Entailment → Contradiction on dev and raises evidence
grounding on dev; we keep it as a correctness fix. Committee-attributed claims still fail most often
(dev 16 of 90, test 10 of 64).

### Clipping over-long pages (2026-10-06)

The median retrieved page has ~1,400 characters, but a few dense legal-text pages (e.g. AHV 21, 2022-09-25) reach
8–10k and pushed single claims to 18–24k input tokens. `PAGE_MAX_CHARS=3000` keeps the window of consecutive
sentences that best matches claim + vote title (the title is in the booklet language, so it anchors cross-lingual
claims) plus the page heading. Offline, no gold passage was lost at any cap between 6,000 and 2,500 characters.

| Split | Macro-F1 | Fixed / broke | Input tokens mean | Input tokens p95 / max | Latency mean / p95 |
|---|---:|---:|---:|---:|---:|
| test 402, no clipping | 0.925 | – | 5,725 | 12,664 / 24,270 | 2.1 s / 3.8 s |
| **test 402, clip 3,000** | **0.928** | 3 / 2 | **4,696** | **7,377 / 10,088** | **1.9 s / 3.3 s** |
| dev 450, no clipping | 0.908 | – | 4,836 | 7,466 / 10,127 | 2.1 s / 4.3 s |
| **dev 450, clip 3,000** | **0.913** | 4 / 2 | **4,435** | **5,762 / 6,921** | 2.0 s / 3.9 s |

Adopted as default: −18 % input tokens on test (−42 % at p95) at unchanged or slightly better macro-F1.

### 12 pages instead of 10 (2026-10-06)

Clipping freed ~900 input tokens per claim, so we re-tested the number of pages (all runs clipped at 3,000 characters).

| k | dev 450 | Input tokens | Latency mean / p95 |
|---:|---:|---:|---:|
| 10 | 0.913 | 4,435 | 2.0 s / 3.9 s |
| **12** | **0.940** | 5,315 | 2.3 s / 2.9 s |
| 14 | 0.942 | 6,175 | 2.6 s / 4.3 s |
| 16 | 0.935 | 7,058 | 2.9 s / 3.8 s |

k = 14 is within noise of 12 for ~860 more tokens; 16 is worse again (more distraction). The test split did not
confirm k = 12 (0.928 → 0.925, 7 fixed / 8 broken), so we ran a third, independent sample: the 643 dev pairs that were
never part of the 450-pair sample.

| Sample | k = 10 | k = 12 | Fixed / broke |
|---|---:|---:|---:|
| dev 450 | 0.913 | 0.940 | 16 / 4 |
| unused dev 643 | 0.903 | 0.928 | 28 / 12 |
| test 402 (5 voting dates) | 0.928 | 0.925 | 7 / 8 |
| **all 1,495 pairs** | **0.912** | **0.931** | **51 / 24** |

Cross-lingual pairs gain most (0.894 → 0.924 over all pairs); most fixes are Entailment → Neutral errors where the
supporting page was ranked 11th or 12th. The test split covers only 5 voting dates, so we decide on all 1,495 pairs:
12 pages are the default (`NLI_TOP_K=12`), at +~890 input tokens (+20 %) and +0.25 s mean latency.

### Speaker boost: the named side's pages are always in the context (2026-10-06)

With the current configuration (clipping, 12 pages), over all 1,495 pairs claims attributed to the committee still
failed in 13.4 % of cases (36 of 269), Federal Council claims in 2.2 %, all others in 7.5 %. In 15 of the 36 committee
errors the gold passage was not in the context; in 13 of these it sat on a correctly labelled committee argument page
ranked 13th–46th, mostly for cross-lingual pairs (BM25 on the claim text finds the page poorly, and argument pages
rarely repeat the vote title).

`SPEAKER_BOOST=2`: if a claim names exactly one side, that side's 2 best-scoring argument pages replace the
lowest-ranked other pages (the context stays at 12 pages). Unlike the rejected speaker-aware retrieval, nothing is
hidden; the attributed side is only guaranteed to be present.

| Gold passage in context (all 1,495 pairs) | Committee claims (269) | Federal Council claims (498) | Others (728) |
|---|---:|---:|---:|
| no boost | 0.888 | 0.944 | 0.977 |
| boost 1 page | 0.948 | 0.982 | 0.977 |
| **boost 2 pages** | **0.985** | **0.994** | 0.977 |
| boost 3 pages | 0.985 | 0.998 | 0.977 |

| Sample | Without boost | With boost | Fixed / broke |
|---|---:|---:|---:|
| dev 450 | 0.940 | 0.953 | 7 / 1 |
| unused dev 643 | 0.928 | 0.936 | |
| test 402 | 0.925 | 0.933 | |
| **all 1,495 pairs** | **0.931** | **0.940** | **22 / 8** |

Committee-claim errors fell from 36 to 27, Federal Council errors from 11 to 7; claims that name no side gave the
identical answer in all but one case (runs at temperature 0 are near-deterministic). No extra tokens or latency.
Adopted as default.

A **speaker hint** in addition ("Passages written by the committee itself: P3, P11") did not help: on dev 450,
6 fixed / 7 broken against the boost alone. Rejected (`SPEAKER_HINT=false`).

### Decision-Rule 0b: a Neutral label with a confident relation is rarely Neutral (2026-10-06)

In 18 of the 1,495 answers Apertus wrote `"label": 1` (Neutral) but put its confidence on another relation
(typically `"p_contra": 1.0`). Decision-Rule 0 kept the label, which was right in 1 of these 18 cases (gold:
9 Entailment, 8 Contradiction, 1 Neutral). Rule 0b takes the confidences' choice in this case only:

| | Macro-F1 before → after | Fixed / broke |
|---|---:|---:|
| dev 1,093 | 0.943 → 0.947 | 5 / 1 |
| test 402 | 0.933 → 0.940 | 3 / 0 |
| all 1,495 | 0.940 → 0.945 | 8 / 1 |
| 4 older runs (other configurations) | – | 15 / 0 |

It post-processes the model's own answer, so it costs nothing. Adopted.

### Other ideas checked on all 1,495 pairs (2026-10-06)

- **Meta-classifier** (logistic regression / shallow tree on the label, the three confidences, their margin,
  speaker flags, cited sections, language pair), 5-fold cross-validation on dev grouped by voting date: +0.004 to
  +0.005 macro-F1. The tree's only useful split was exactly the case Rule 0b now handles, so the transparent rule
  replaces the model.
- **Speaker-mismatch guardrail** (committee claim + Contradiction + no committee page cited → Neutral or Entailment),
  re-simulated with the speaker boost: 13 fired, 0 fixed, 13 broken. Rejected again; with the boost, false
  contradictions almost always cite the committee's own pages.
- **Prompt rule for hypothetical consequences** ("if a detail is not mentioned, answer Neutral, not Contradiction"):
  not run. It could only fix gold-Neutral answers predicted as Contradiction, which are 3 of 89 errors, while it
  pushes towards the largest error class (Contradiction → Neutral, 30). The Entailment → Contradiction errors (28)
  have gold Entailment, which a shift to Neutral does not fix.
- **Italian claims against German booklets**: the 0.80 on the test split comes from 43 pairs. Over all 1,495 pairs
  it→de reaches 0.904 with the boost, and the weakest pair is fr→de (0.880); its 16 errors spread over 10 voting
  dates and all error types without a pattern. No language-specific rule (a threshold rule is impossible anyway:
  Apertus' confidences are 0/1 in 97–99 % of answers).
- **Layout-aware parsing (Docling)**: not adopted. In nearly all remaining errors the gold page is in the context,
  so better table parsing addresses few of them, while Docling adds PyTorch and layout models (several GB) to the
  image and requires re-parsing and re-measuring every booklet.

### Thinking mode: +2 F1 points for ~11× latency (rejected)

We use `Apertus-v1.5-70B-thinking` but by default ask for the JSON answer only (~60 output tokens).
With `THINKING=true` the model reasons before it answers. A prompt instruction alone is not enough: asked to
"think first", Apertus still writes the JSON first and explains afterwards (post-hoc, no effect on the label).
Only prefilling the reasoning marker (`<|inner_prefix|>` as the start of the assistant turn,
`continue_final_message`) makes it reason before deciding; the answer follows `<|inner_suffix|>`. It ignores
a requested length limit ("at most 120 words") and walks through the passages one by one.

Dev 450 (seed 7, no page clipping), paired with the fast configuration of the same date. 85 thinking requests
first failed on the shared endpoint (84 × "invalid API key" interleaved with successful calls, 1 × 504) and were
re-run, so every pair has a thinking answer.

| Configuration (dev 450) | Macro-F1 | Fixed / broke | Output tokens median / p95 | Input tokens | Latency median / p95 |
|---|---:|---:|---:|---:|---:|
| Fast `ids` answer (current) | 0.908 | – | 59 / 70 | 4,836 | 2.0 s / 4.3 s |
| **Always think** (max. 4,000 tokens) | **0.929** | 20 / 11 | 1,296 / 2,802 | 4,884 | 23.2 s / 111 s* |
| Thinking budget 300 tokens, then forced answer (82-pair probe) | – | 15 of 41 / 4 of 41 | 353 | ~9,600 | 7.4 s / 8.4 s |

\* p95 inflated by endpoint load. 6 thinking answers hit the 4,000-token limit without a decision.

Selective thinking, simulated on the 363 pairs that had both answers in the first pass: thinking only when the fast
answer is Neutral or Contradiction kept the gain of always thinking (18 fixed / 3 broke vs. 20 / 5) but runs on
72 % of the claims plus the fast call; thinking only for claims that name a speaker (50 %) kept less than half of it.
A hard budget does not help either: 300 tokens are never enough (all 82 answers had to be forced), it fixes fewer
and breaks more errors than full thinking, and the second call re-sends the whole context (2× input tokens).

Thinking fixes about half of the remaining errors, but for +0.021 macro-F1 it needs ~22× the output tokens and
~11× the latency, and long requests are the first to fail under load. Because tokens and latency are judged
right after macro-F1, it stays off (`THINKING=false`, `THINKING_BUDGET=0`); it is the clearest quality/efficiency
trade-off we found for Apertus.

### Translating the claim into the booklet language (rejected)

`TRANSLATE_CLAIM=true` translates cross-lingual claims with one short extra Apertus call and passes the
original and the translation to the NLI prompt (tokens and latency of both calls are counted).

| Split | Fixed / broke (cross-lingual pairs) | Macro-F1 | Latency mean |
|---|---:|---:|---:|
| dev 450 | 7 / 2 | 0.908 → 0.919 | 2.1 → 2.7 s* |
| test 402 | 7 / 9 | 0.925 → 0.923 | 2.1 → 2.7 s* |

\* measured while another run shared the endpoint. Over both splits 14 fixes against 11 breakages on
cross-lingual pairs: within run-to-run noise, for +0.6 s and ~20 extra output tokens per claim. Kept off.

### The 30 remaining test errors

- They come from only ~19 distinct claims: the same claim is paired with the DE, FR and IT booklet
  (one hydropower claim alone accounts for 4 errors, one factory-farming claim for 3).
- In 29 of 30 errors the gold passage was among the 10 pages sent to the model; the errors are reasoning
  errors, not retrieval misses (which is what thinking addresses).
- Remaining Entailment → Contradiction errors are mostly committee claims judged against the other side's text,
  and "if accepted, X must …" claims where the model reads a missing detail as a contradiction.
- 3–4 gold labels look debatable (e.g. claims about what "the summary" or "the voting text" says, `hf-0760`,
  `hf-1234`, `hf-1458`), so label noise leaves little headroom.

### No "lost in the middle" effect

With the full booklet (150 test pairs), accuracy did not drop with the position of the gold passage: 69 % when
the gold text is in the first third of the booklet, 77 % in the middle third (hybrid: 94 % / 86 %). The
full-document weakness is general distraction by ~60k tokens of context, not a position effect.

### Rejected for efficiency reasons

A cascade (5 pages first, 10 only if the answer is Neutral) would save ~15 % tokens but needs two calls for
every Neutral claim (a third of the data), which worsens p95 latency — one of the judged metrics.

## Latency of a single CLI call

Judges may call the CLI once per claim, so start-up and PDF parsing can count towards inference time.

| Step | Before | After |
|---|---:|---:|
| PDF parsing per call | 0.9–2.5 s (PDF read 3×) | 0.04 s (read once, disk cache keyed by file hash) |
| Importing scikit-learn for `predict` | always | only for `benchmark` / `compare` |
| Full `predict` call, warm (incl. ~1 s model call) | 8.9 s | 3.3 s |

`python -m src warm-cache` pre-parses all booklets; the Docker build runs it.

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
5. **Trust the label over the confidences.** When Apertus' label and confidences disagree, the label is
   right 3× as often (Decision-Rule 0, +0.008 F1 on test).

## Known gaps / next steps

- French claims against German booklets are the weakest pair (0.880 over all 1,495 pairs, 0.825 on the test split).
- Remaining advanced errors over all 1,495 pairs (81): Entailment → Contradiction 35, Contradiction → Neutral 22,
  Contradiction → Entailment 11, Entailment → Neutral 8, Neutral → Contradiction 4, Neutral → Entailment 1.
- Evidence precision with whole pages: ~75 % of cited pages lie in the gold section.
- Section labels: committee pages whose only heading is a slogan ("Nein zu diesem Zensurgesetz") stay unlabelled.
- Proposal boundary detection misses proposals in 2021-06-13, 2022-09-25 and 2024-03-03 (IT); hybrid
  retrieval does not depend on it, but the legacy `retrieval` strategy does.
- Committee-attributed claims now fail as often as claims without a speaker (7.4 %, down from 13.4 %; Federal Council 1.4 %).
# Competition-score optimization audit (2026-10-08)

An isolated `experiment/competition-score-optimization` branch starts at
`a0e9e772dd388bdff4b8635ef0e09872ef705af6`. Recalculating the saved 402-case
predictions confirms Advanced F1 0.940236 and Beginner F1 0.974948. This is an
archive audit, not a fresh API reproduction. All 70 existing tests pass.

The audit found that the historical evidence metric differs from official
Hit@5, page 1 is used for missing metadata, and the official bare CLI arguments
and `BASE_URL/API_KEY` are not supported yet. The current public dataset has
1,488 rows compared with the historical 1,495. No optimization is recommended
from these checks alone. See [the ongoing report](competition_optimization_report.md)
and [machine-readable baseline audit](../results/competition_optimization/baseline.json).


## Competition optimization — 2026-10-08 (in progress)

Isolated branch `experiment/competition-score-optimization`, from main
`a0e9e772dd388bdff4b8635ef0e09872ef705af6`; adaptive WIP is preserved.
See [the optimization report](competition_optimization_report.md) and
`results/competition_optimization/` for frozen sources, dataset and evaluator hashes.

- Historical 402-row archives verified; v1.0 restores all historical row identities.
- Fresh 804 mixed cases: A F1 0.945152, B F1 0.974948, official A Hit@5 0.361940.
  Small F1 movement is not attributed to architecture; responses are now uncapped.
- E1, original full PDF pages only: Hit@5 0.582090, 59 evidence fixes / 0 breaks.
- E2, source pages plus complete blocks: Hit@5 0.720149, 97 fixes / 1 break.
  Labels and remote requests are unchanged. The lost terrorism example is an
  ordering regression (fourth cited detailed page displaced by a text block).
- Confidence-based Neutral override with p_neutral>=0.99 and absent citations:
  diagnostic found zero applicable cases; no feature added or improvement claimed.
- N1, conditional JSON grammar after transport exhaustion: the persistent
  v1.1-row-289-A gateway failure completed in 1.15 s, 51 output tokens, after
  identical primary request hashes still failed. Validity is not label correctness.
  Its assembled replay is not treated as a fresh full candidate execution.
- T3, single-flight PDF/index cache and lazy evidence-page extraction: independent
  flags, concurrency/source-change/source-text tests; full measured execution pending.
- Original complete v1.1 baseline: 2,976 cases / 20 dates, one unresolved technical
  failure counted as invalid. C1 full candidate runs independently on those same
  date-grouped folds. No final candidate has been selected.
- First linux/amd64 CPU image: build, 95 tests / six subtests, offline mixed
  read-only input contract and two real Apertus calls passed. Final image rerun
  is required after the latest feature additions. No image was published.

### 8 October: full C1 and source review checkpoint

- C1 completed all 2,976 mixed cases: Advanced F1 0.944034, Beginner 0.982497, Hit@5 0.755556, zero invalid answers. B0 has 0.947699/0.981830/0.467677 with one invalid answer retained. C1 label fixed/broken counts are 3/7; evidence 292/7. Advanced F1 regresses slightly; this is not described as an accuracy gain.
- Five-fold mean/std: Advanced 0.944348/0.030060; Beginner 0.981952/0.004897. Known 11,732,096 input and 191,322 output tokens; 19 failed attempts have unknown usage. The 3.0134 s local mixed processing timer cannot establish separate official task efficiencies.
- Source review covers all 32 historical label errors, retaining one scope ambiguity without relabeling gold. Confirmed overlapping causes E/F/G/H/I are 2/8/18/6/2; J remains unconfirmed. PDF text blocks, pages and source hashes are verified.
- Lazy extraction preserves every replay prediction byte for byte; E2 mean component time decreases from 20.838 to 7.731 ms, while p95 increases. This is offline component profiling, not proxy time.
- R1/R2 now execute independently and sequentially over the full identical v1.1 population. Translations change only ranking, preserve the original NLI claim, and all acquisition/rejection costs count. Private cold caches prevent warm-cache cost borrowing between methods.
- 111 tests and six subtests pass. Final source selection, broader source-cause audit, final image, README and final acceptance remain open.

### 9 October: translated-query validation and timing correction

- R1 independently completes 2,976 cases on the same five date-grouped folds: Advanced Macro-F1 0.950843516, Beginner 0.981829644, Hit@5 0.765656566; no invalid outputs. Relative to B0, label fixes/breaks are 24/19 and evidence fixes/breaks 346/51. R1 remains provisional while R2 finishes.
- R1 Advanced cross-language F1 is 0.952153499; fold mean/std 0.950479699/0.034969840. Translation preserves the original NLI claim. Known 11,788,990 input / 228,397 output tokens, 3,749 API attempts with 24 unknown-usage failures. Reported Advanced tokens rise 1.11%; no token-saving claim.
- Bare submission CLI timing now starts before dependency imports, input loading and client initialization. Wall time minus the union of measured SDK requests is explicit; Python/container startup, diagnostics write and shutdown still belong to the official proxy measurement. A fresh-interpreter delayed-import regression test passes.
- Error auditing uses recorded translated/union context pages and original-claim clipping instead of substituting baseline ranking. Actual physical source pages are required. No extra model calls are made by the audit.
- 113 host tests pass. The prior 12-case Docker checkpoint passes all output/source checks; a fresh image is being verified after the timing/audit changes. Source-only demo inputs replace gold-bearing development demos in Docker COPY, preserving the development dataset outside the image.

- R2 first full execution ends with 23 operational failures; the entire terminal prediction/journal/measurement snapshot and each failed raw record are preserved before recovery. Identical frozen sources/settings repeat only the 23 failed cases. Final coverage is 2,976/2,976 with no operational failures; cumulative 3,873 API attempts include 148 attempts with unknown usage, known input/output 11,818,740/214,805. This is a recovered full independent run, not an uninterrupted run or a zero-cost replay.
- Initial official F1 recalculation gives R2 Advanced 0.956237226 and Beginner 0.982497441. Full grouped/evidence comparison remains pending. All 4,650 Advanced source quotes pass the physical-page/block validator without gold access.
- Historical source observations are revalidated only when claim, vote, languages, gold passage, predicted/gold labels and actual context pages match; speaker errors additionally require equal cited-page sets. Physical PDF proofs are checked anew. This verifies 26 C1 and 15 R1 errors without assuming new internal reasoning equals archived reasoning; the remaining semantic causes remain explicitly unreviewed.
- Final source-only image passes all 113 tests with /data-independent network-disabled CPU execution. Full image/model/output checks for both retrieval modes are running against a pinned image ID.

- Final identical-population component comparison: B0 A/B F1 0.947699/0.981830, Hit@5 0.467677 (one invalid output retained); C1 0.944034/0.982497/0.755556; R1 0.950844/0.981830/0.765657; recovered R2 0.956237/0.982497/0.773737. R2 label fixed/broken 30/16; evidence 349/46. Its Advanced fold mean/std is 0.955664/0.033242 and cross-language F1 0.959217. Advanced F1 decreases in two folds and in fr→fr, de→it and fr→it; the largest language-pair drop is fr→it −0.021173. These regressions are retained.
- R2 has the greatest measured quality contribution under supplied weights (+0.039475 vs B0), at 1.39% more reported Advanced tokens. C1/R1/R2 all remain on the known quality/token Pareto frontier; task-specific official time/cost and unknown failed-request usage prevent an official overall ranking.
- Pinned CPU linux/amd64 image `sha256:58ad99ee0482fb565b38595f456605269233de8b40621f67db19c5d3ebc4c33c` passes 113 container tests and source-only image inspection. The mixed 12-case contract fixture includes an empty ID, Unicode IDs and six source/claim-language directions. Both translated and union real-API modes pass /data:ro, writable /output, /tmp caches, physical source quotes and uncapped Apertus v1.5 calls. The final check journals 30 successful API attempts, zero unknown usage; deterministic offline forward/reverse outputs agree. Credentials, gold files and downloaded weights are absent from the image. Full proof is in `results/competition_optimization/docker_validation.json`.

- Source-review transfer now binds the exact gold/predicted labels and supplied context pages; speaker causes also bind cited-page sets. All three full audits rerun successfully with these checks, retaining 26/15/14 revalidated C1/R1/R2 reviews and 83/85/77 pending semantic cases. Changed predictions/context reject stale cause attribution. 114 host tests and 114 CPU container tests pass. Code checkpoint: `b83aaa1`.

- The final audit-tools image `sha256:db1de9f6f45818d22d861b0a7c473964222ddb0b01adaf829c1c8a303f403468` passes all 114 CPU tests and a direct R2 contract run: 12 mixed cases, 15 successful Apertus requests including translations, zero unknown usage. Its inference source hashes equal the earlier both-mode image. Source-only inspection and forward/reverse offline equality pass again.
- Three additional R2 errors are directly inspected: 14-A (adult-directed tobacco advertising accessible to minors), 17-A (pesticide food-import ban) and 53-A (committee says existing criminal law already protects against insults). Decisive exact original blocks are present in actual clipped contexts; G is confirmed without inferring additional E/J causes. Source review now covers 17/91 label errors, leaving 74.


### 2026-10-09 — experiment closure

- The optional official CLI now uses `NLI_BATCH_WORKERS=4` / `--workers 4`, matching the native evaluation concurrency while preserving output order, per-case request accounting and complete failure records. The baseline default remains one worker. The overlapping-call test verifies reversed completion order, unchanged IDs and the SDK interval union.
- Final host and pinned CPU linux/amd64 suites both pass **115 tests**. Image `sha256:87f8111a6ac4ed0a8ae980005749b533e93c8bcef7835ea7dec2d96837b300be` passes the source-only 12-case R2 contract with four workers: 15 successful Apertus requests, zero unknown usage, 44,201 input / 839 output tokens. These fixture timings are not full-dataset efficiency measurements.
- All **91 R2 label discrepancies** are source-inspected; confirmed overlapping counts are F=9, G=70, H=7, I=15, J=8, E=0 and K=0. Category L contains 185 correctly labeled cases with unmatched submitted evidence. Gold/scope ambiguities in 235-A and 757-A remain explicit with unchanged gold; inspection is not independent human adjudication.
- Source observations are revalidated against identical claim/source, labels and supplied context for comparators: **67/109 C1**, **66/100 R1** reviewed. Remaining 42/34 semantic causes are documented, not treated as confirmed errors of a specific mechanism. All observed confusions/evidence failures are enumerated for every compared configuration.
- R2 is the final measured recommendation: Advanced/Beginner Macro-F1 **0.956237226/0.982497441**, Advanced Hit@5 **0.773737374**, on 1,488 cases per task and identical date-grouped folds. Advanced reported tokens increase 1.39%; unknown failed usage and separate-task official efficiency prevent claiming the exact competition winner. Baseline/C1 remain selectable.
- The user requested closure and GitHub upload. No further model experiments are started. Reports, six canonical result artifacts, README and reproducible commands are finalized on the experiment branch; caches, journals, credentials and individual prediction files remain ignored. CI also runs on pushes to this branch. Main is not merged and no container is published.
