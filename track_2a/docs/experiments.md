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
| **Advanced** (booklet PDF + claim + vote) | **0.925** | 0.941 | 0.917 | 5,725 | 59 | 2.1 s / 3.8 s |
| **Beginner** (reference string + claim) | **0.975** | 0.984 | 0.971 | 2,523 | 55 | 1.1 s / 1.7 s |

Configuration: hybrid retrieval (BM25 claim + 2 × BM25 vote title, booklet-wide), top 10 pages,
section labels (incl. closing recommendation boxes), `ids` prompt mode, model label kept when it disagrees
with the model's own confidences (Decision-Rule 0). Before these last two changes the advanced task scored 0.920
(see [error analysis](#error-analysis-on-saved-runs-2026-10-05)).

Advanced macro-F1 by language pair (claim → booklet):

| | → de | → fr | → it |
|---|---:|---:|---:|
| **de** | 1.000 | 0.933 | 0.922 |
| **fr** | 0.914 | 0.940 | 1.000 |
| **it** | 0.816 | 0.907 | 0.875 |

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

### Thinking mode: +~3 F1 points for ~10× latency (rejected)

We use `Apertus-v1.5-70B-thinking` but by default ask for the JSON answer only (~60 output tokens).
With `THINKING=true` the model reasons before it answers. A prompt instruction alone is not enough: asked to
"think first", Apertus still writes the JSON first and explains afterwards (post-hoc, no effect on the label).
Only prefilling the reasoning marker (`<|inner_prefix|>` as the start of the assistant turn,
`continue_final_message`) makes it reason before deciding; the answer follows `<|inner_suffix|>`. It ignores
a requested length limit ("at most 120 words") and walks through the passages one by one.

Dev 450 (seed 7), paired with the fast configuration. 85 of the 450 thinking requests failed on the shared
endpoint (84 × "invalid API key" interleaved with successful calls, 1 × 504) and are excluded here;
the table covers the 363 pairs with a complete answer from both runs (this subset contains 40 of the 41 errors
of the fast run, so its F1 is lower than the full-sample 0.908).

| Policy (363 pairs) | Macro-F1 | Fixed / broke | Thinking on | Output tokens (median) | Latency (median) |
|---|---:|---:|---:|---:|---:|
| Fast `ids` answer (current) | 0.889 | – | 0 % | 59 | ~2 s |
| **Always think** | **0.931** | 20 / 5 | 100 % | ~1,200 | ~22 s |
| Think if the fast answer is Neutral or Contradiction | 0.931 | 18 / 3 | 72 % | | |
| Think if the claim names a speaker | 0.912 | 9 / 1 | 50 % | | |
| Think if the fast answer is Contradiction | 0.909 | 7 / 0 | 32 % | | |

Extrapolated to all 450 pairs, always thinking would raise macro-F1 from 0.908 to roughly 0.94 (preliminary:
the 85 failed requests still have to be re-run). Answers are long (p95 ~2,500 output tokens); 2 answers
hit the 4,000-token limit without a decision, and long requests are the first to fail when the endpoint is
under load. Selective thinking keeps the gain only when it runs on ~3/4 of the claims plus the fast call, so
it saves little. Because tokens and latency are judged right after macro-F1, thinking stays off
(`THINKING=false`); it is the clearest quality/efficiency trade-off we found for Apertus.

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

- Italian claims against German booklets are the weakest pair (0.82 F1 on the test split), mostly with the gold page retrieved.
- Most remaining advanced errors are Entailment → Contradiction (13 of 30 on the test split).
- Evidence precision with whole pages: ~75 % of cited pages lie in the gold section.
- Section labels: committee pages whose only heading is a slogan ("Nein zu diesem Zensurgesetz") stay unlabelled.
- Proposal boundary detection misses proposals in 2021-06-13, 2022-09-25 and 2024-03-03 (IT); hybrid
  retrieval does not depend on it, but the legacy `retrieval` strategy does.
- Committee-attributed claims remain the main error source (see above).
