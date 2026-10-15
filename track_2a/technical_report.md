# Technical Report — Fact Attack: Multilingual NLI over Swiss Voting Booklets

- **Track:** Track 2A — OST: Multilingual Natural Language Inference over Swiss Official Voting Booklets
- **Event:** Hack Apertus Online (October 1–16, 2026)
- **Team:** Team Fact Attack 2026 — Josip Jukic, Felipe Wüthrich, Sheyla Sampietro, Andrea Petretta
- **Repository:** [GitHub](https://github.com/JJukic/Apertus-Team-Fact-Attack-2026) · **Experiment log:** [docs/experiments.md](docs/experiments.md)
- **Demo:** Streamlit app (`make web`: claim check with highlighted evidence pages, benchmark dashboard) and the
  entailment CLI (`python -m src --input cases.jsonl --output predictions.jsonl`, the Docker image's entrypoint, see 2.5)

---

## 1. Summary

Swiss voters receive an official booklet for every federal vote, and campaigns make claims about it that are
hard to check. We built a multilingual claim-verification system on **Apertus v1.5**: given a booklet (German,
French or Italian) and a claim (any of the three languages), it classifies the claim as **Entailment (0)**,
**Neutral (1)** or **Contradiction (2)** and returns the booklet pages that justify the decision, together with
token usage and inference time.

On **booklets outside the dataset** (our holdout: 90 claims on the 27 September 2026 vote, labels checked by a
second person), it reaches **macro-F1 0.989** and **evidence Hit@5 0.983** on the advanced task (booklet PDF; 0.956 /
0.933 before E23, which two holdout errors suggested, so that gain is not confirmed independently) and **0.955** on
the beginner task (reference text; 4 of 90 labels wrong). On the whole official
dataset (v1.1, 1,488 pairs, 20 voting dates in 5 date-grouped folds) the same configuration scores
**0.977 / Hit@5 0.980** and **0.982**, at 1,757 / 865 input tokens and 59 / 21 output tokens per case, with
processing time outside the LLM calls in the millisecond range. The beginner task trades 0.005 macro-F1 (0.987 at
1,640 input tokens before) for half the tokens (section 2.4). All scores come from the organisers' `evaluate.py` on the official CLI's output.

The official score also weighs evidence, processing time and tokens (relative to the best team), so after the
organisers' Q&A we re-tuned for it (2.6), each choice decided by paired runs on the same cases (5.3). The largest
step is **contents-guided part selection** (2.2): every claim names its source in its opening, and the booklet's
contents page locates that part of the vote, which becomes the whole context (macro-F1 0.949 → 0.968, Hit@5
0.881 → 0.975, −55 % input tokens); a 9,000-character budget for the voting text (0.974) and a prompt without its
role sentence (0.977, −34 tokens) followed. Before it: evidence from PyMuPDF pages and blocks, retrieval with the
claim and its translation, four compact label rules, a start-up path that keeps processing time in milliseconds,
and for the beginner task cut references and a short first call (−48 % input tokens, −0.005 F1; 2.4).

The central finding for the challenge's research question: **less, but the right context wins** — the whole booklet
(~60k tokens) reaches 0.730 macro-F1, eight retrieved pages (~3,900 tokens) 0.949, and the one part the claim refers
to (~1,800 tokens) 0.977.

---

## 2. Architecture

```
 Booklet PDF (DE/FR/IT)                       Claim (DE/FR/IT)  +  vote title
        │                                              │
        ▼                                              │
 pypdf → pages; each page labelled with its section    │
 (committee arguments / Federal Council arguments /    │
  voting text / explanation / overview)                │
        │                                              │
        └────► Contents-guided part selection ◄────────┘
       claim opening → part (summary / detail / committee / Federal Council / voting text)
       contents page → pages of that part of the vote → context (≤ 6,000 / voting text 9,000 chars), [P1] …
                                  │   (no part found: hybrid retrieval, top 8 pages, 2.2)
                                  │
                                  ▼
                 Apertus v1.5 on CSCS — NLI prompt ('ids' mode)
                 → {"p_entail", "p_neutral", "p_contra", "label", "evidence_ids"}
                                  │
                                  ▼
       label + evidence: cited pages and their best text blocks (PyMuPDF), verbatim, with page
```

### 2.1 Document processing
`pdf_parser.py` extracts the text of every page with pypdf and detects section headings in all three languages
("Argumente Initiativkomitee", "Arguments du Conseil fédéral et du Parlement", "Il testo in votazione", …).
A heading applies to its page and the following pages until the next heading, and table-of-contents entries
(heading followed by a page number) are ignored. Section labels matter because booklets deliberately contain
opposing viewpoints: the model has to know whether a page states the committee's or the Federal Council's position.
The same pass also reads every page with PyMuPDF (text and text blocks) for the evidence quotes (2.3). Booklets that
are not in the image are parsed by a process pool at start-up, one booklet after the other in the order of their cases;
the parse of the 60 dataset booklets (only the PDFs' text and blocks: no claims, labels or scores) is in the image.

### 2.2 Context selection: contents-guided part selection
The dataset's claims name their source in their opening ("Laut der Zusammenfassung …", "Le comité estime …",
"Se il voto viene accettato …"), and every booklet lists, per vote, the pages of its parts on its contents page
("In Kürze 4 – 5 | Im Detail 14 | Argumente 20 | Abstimmungstext 24", likewise in French and Italian).
`part_context.py` combines both:

- **Claim → part:** the earliest cue that starts within the claim's first 30 characters (E24; every dataset and
  holdout source starts by character 20) selects summary, committee, Federal Council, voting text or, for the
  consequences of a yes, the detailed explanation.
- **Contents → pages:** the contents page (also over two pages) gives each part's start page, checked against the
  heading printed on that page; navigation bars on the summary pages are not mistaken for contents. The arguments
  are split into committee and Federal Council pages with the parser's speaker labels (2.1).
- **Vote:** the case's vote title is matched by its words to the contents titles and the first lines of each vote's
  summary and detail pages (contents titles are often abbreviated).
- **Context:** the part's pages in reading order, labelled with the part, at most 6,000 characters (voting text
  9,000, E23); a longer part keeps its best pages by BM25, and only then is a cross-lingual claim translated.
- Coverage: contents read in 60/60 dataset booklets and both holdout booklets; a part found for 1,476 of 1,488 claims.
  The other claims, and any booklet without a readable contents page, use the hybrid retrieval below.

**Fallback: hybrid retrieval** (the submission before E12). `retriever.py` scores every page with normalised BM25
against the claim and, for a cross-lingual claim, its translation by Apertus (48 fixed / 25 broken labels over three
paired comparisons), plus twice BM25 against the vote title, which is written in the booklet's language and anchors
the proposal; it keeps the top 8 pages, clipped to 3,000 characters. If the claim names a side, that side's two best
argument pages are always included (speaker boost).

### 2.3 Apertus as the NLI model
The selected pages are numbered `[P1]`, `[P2]`, …. The system prompt (without a role sentence, E25) defines the three labels, tells the model to
judge only against the passages (not world knowledge), to answer Neutral for topics or details the passages do
not address, to check claims attributed to an actor only against that actor's text, and to read "recommends
rejecting X" as supported when the Federal Council and Parliament recommend "No". For the advanced task, four
compact label rules follow (~100 tokens): compare actor, action, object and decisive conditions; Entailment also for
paraphrases, summaries and other languages, but never on unstated information; missing support is not a
contradiction. Apertus answers with a small JSON object: three confidences, the label, and the ids of the pages that
justify it.

- **Evidence by id** guarantees verbatim booklet text with the correct page number and avoids paraphrased quotes.
- **Short answers**: ~55 output tokens instead of ~150 for a JSON answer with reasoning and quotes, which more
  than halves inference time (p95 5.5 s → 1.8 s on the dev split) at equal or better F1.
- **Evidence items:** up to three cited pages as PyMuPDF reads them, then the two text blocks of the cited pages that
  best match claim and vote (at most five items). Hit@5 counts an item only if it lies inside the gold passage or
  contains it; the gold passages follow PyMuPDF's text, so the same blocks built from pypdf or pypdfium2 text reach
  only 0.80 / 0.73 instead of 0.87 on dev. The PyMuPDF pages are stored with the parse, so no PDF is opened while
  cases are answered.
- For **Neutral**, the evidence list is empty. If the model predicts 0 or 2 without citing a page, the top-ranked page
  is used.
- **Decision rules:** in ~2 % of answers the label and the stated confidences disagree. The label is right 3× as
  often, so it is kept (Rule 0) — except a Neutral label next to a confident other relation, which was right in only
  1 of 18 cases and follows the confidences (Rule 0b). Otherwise label and confidences agree and are used as given.

### 2.4 Beginner task
The reference string (up to 25k characters) is split into passages of ≤ 700 characters at sentence boundaries,
numbered and sent with the same prompt, so evidence is a specific passage rather than the whole reference. A
reference longer than 6,000 characters is cut to its opening passage plus the passages that best match the claim
(BM25 with the claim and, for cross-lingual pairs, its translation), in document order, up to 6,000 characters.
On all 1,488 beginner cases this saves 30 % of the input tokens at equal macro-F1 (0.981 → 0.984; 8 fixed / 3 broken
claims, against 2 changed labels when the unchanged system runs twice); on the holdout 0.955 → 0.966 at −13 %. The
length of a reference is never used as a hint for its label.

**Tokens (E17–E19).** The beginner prompt costs ~620 fixed tokens and its answer carries confidences that task B does
not score, while tokens are scored relative to the other teams. (1) References up to 9,000 characters keep 3,000
characters, longer ones 6,000: macro-F1 0.9845 → 0.9872, 1,732 → 1,640 input tokens. (2) A short ~190-token prompt that
fixes the decision order (is the subject dealt with, then support, contradiction, else neutral) and answers with the
digit alone loses 0.02–0.04 F1 by itself. (3) **Cascade (submission):** that short call sees at most 2,000 characters;
when its label probability is below 0.98 (~10 % of the cases) the case is asked again with the full prompt of (1), and
both calls' tokens count. Most of the short call's errors lie in that unsure part. All pairs: macro-F1 0.982,
865 input and 21 output tokens (−48 %). On the holdout (two votes, ~30 distinct claims) it saves 21–22 % at 4–6
instead of 3 wrong labels (two runs). The token score cannot be computed locally, so the net effect is estimated (about +0.06 on the
beginner score under an assumed best-team value of ~820 tokens). `B_PROMPT=` restores (1).

### 2.5 CLI and output format
`python -m src --input cases.jsonl --output predictions.jsonl` (the Docker entrypoint) reads the official case format (JSON or JSONL; a case has
`booklet.path` + `vote` for the advanced task or `reference.text` for the beginner task) as well as rows of the OST
dataset on Hugging Face (`claim`, `reference_string`, `booklet_url`, `vote`; also Parquet or CSV), and writes one
record per case:

```json
{"id": "case-0042", "label": 2, "label_name": "contradiction",
 "evidence": [{"page": 7, "text": "<page 7 as PyMuPDF reads it>"}, {"page": 7, "text": "<best text block>"}, ...],
 "metrics": {"input_tokens": 3983, "output_tokens": 75, "inference_time_ms": 2053}}
```

Evidence is empty for Neutral, and its page is `null` for a reference text.

### 2.6 Evaluation contract and efficiency
The organisers run the image as `--input /data/cases.jsonl --output /output/predictions.jsonl` with `BASE_URL` (a
token-counting proxy) and `API_KEY` injected, `/data` read-only. The score is `0.6 · S_A + 0.4 · S_B`, where
S_A = 50 % macro-F1 + 20 % evidence Hit@5 + 15 % processing time + 15 % tokens and S_B = 70 % macro-F1 + 15 % time +
15 % tokens; time and tokens count as best ÷ own (best of the teams with macro-F1 ≥ 0.65), and processing time is the wall clock minus the time
with an Apertus request in flight.

- **Contract:** `BASE_URL` / `API_KEY` take precedence over our older variable names; the parse cache of the 60 dataset
  booklets is baked into the image read-only and new entries go to `/tmp`; a CI step runs the image exactly this way.
- **Tokens:** the claim's part instead of 8 pages (−55 %), after 8 instead of 12 pages (−31 %); task B: cut
  references and the cascade (2.4); shorter prompts and smaller budgets cost F1 (5.3).
- **Processing time:** a standard-library HTTP client (~1 s less start-up), four cases in flight, uncached booklets
  parsed by a process pool started first; until a new booklet is parsed, a case reads only its part's pages
  (identical context and evidence in 1,566/1,566 cases). Holdout, new booklets: ~70 ms (A) / ~30 ms (B) per case.
- **Robustness:** a broken case line, an unreadable page, an API error or a stuck parser costs at most its own case,
  returned as a valid Neutral record; the exit code stays 0. If the endpoint does not serve our model, the client
  switches once to the best Apertus v1.5 model it lists instead of failing every case.

---

## 3. Use of Apertus

- **Model:** `swiss-ai/Apertus-v1.5-70B-thinking` (the model our CSCS key is authorised for; the thinking phase is
  not triggered by our prompt, answers start directly with the JSON object).
- **Where it runs:** CSCS inference endpoint (`https://api.inference.cscs.ch/v1`), OpenAI-compatible chat
  completions, temperature 0, retries with exponential backoff.
- **Role:** Apertus is the only model in the pipeline. It translates cross-lingual claims for ranking (only when a part exceeds its
  budget or the fallback retrieval runs), performs the NLI classification and selects the evidence pages. Locally:
  pypdf and PyMuPDF (text, blocks), pypdfium2 (fallback), the contents parser and BM25; no OCR, no embeddings. Apertus 8B v1.5 was used only for a comparison
  run (5.3); no other model was used for development or evaluation.

---

## 4. Data and evaluation protocol

- **Dataset:** `OSTswiss/MNLIoverSwissVotingBooklets` — 1,495 human-annotated claim/reference pairs from
  60 booklets (2020–2026), balanced labels, ~66 % cross-lingual pairs.
- **Booklets:** downloaded from `bk.admin.ch` via the `booklet_url` field (`python -m src.hf_dataset`).
- **Split by voting date**, so all three language versions of a booklet stay on the same side: **dev** (1,093 pairs,
  15 dates) and **test** (402 pairs, 5 dates; 398 in v1.1). Test served several before/after decisions, so it is
  development data by now, not a holdout.
- **Official scorer:** cases from the starter repo's `prepare_cases.py` (dataset v1.1), scored with its
  `evaluate.py`. Final measurements run the official CLI on **all 1,488 pairs per task in five folds grouped by
  voting date** (`scripts/run_decision.py`, which forces the submission settings of `.env.example`), and report
  per fold, language pair and class, macro-F1 on de-duplicated pairs (38 % of the rows are exact duplicates) and a
  bootstrap over voting dates. Running the unchanged system twice changes 3 (advanced) / 2 (beginner) of 1,488
  labels: the endpoint is nearly deterministic, so paired label changes measure the effect of a change.
- **Holdout on unseen booklets:** the jury's set may include booklets outside the dataset. We wrote 90 claims (30 per
  label, 10 per language pair, claim types in the dataset's proportions) on the 27 September 2026 booklets before the
  system ever ran on them; a second team member checked every label (90/90 agree). It confirmed E2 and served as a
  regression check for E8, E12, E19c and E25. Two of its errors suggested E23, so for E23 only the dataset counts,
  and the holdout's 0.989 is optimistic (0.956 before E23).
  (An earlier set of 36 claims on the same booklets was used for two decisions and later lost.)

---

## 5. Results

### 5.1 Final configuration — official CLI and scorer

| Evaluation set | Task | Macro-F1 | Hit@5 | Input / output tokens | Apertus time (mean / p95) |
|---|---|---:|---:|---:|---:|
| **Holdout** (90 pairs) | Advanced | **0.989** | **0.983** | 1,791 / 57 | – |
| **Holdout** (90 pairs) | Beginner | **0.955** | – | 1,390 / 40 | – |
| **All pairs** (1,488) | Advanced | **0.977** | **0.980** | 1,757 / 59 | 1.1 s / 1.4 s |
| **All pairs** (1,488) | Beginner | **0.982** | – | 865 / 21 | 0.25 s / 1.2 s |

Holdout = unseen booklets; all pairs = 5 date-grouped folds, macro-F1 per fold 0.971–0.980 (advanced) and
0.969–0.991 (beginner). Processing time outside the model calls: ~2 ms per case (CLI estimate, cached booklets). Before part selection
(8 retrieved pages): 0.949 / Hit@5 0.881 at 3,915 tokens on all pairs, 0.933 / 0.850 on the holdout. Macro-F1 and
Hit@5 by language pair (booklet → claim), all 1,488 pairs per task:

| Pair | de→de | de→fr | de→it | fr→de | fr→fr | fr→it | it→de | it→fr | it→it |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Advanced F1 | 0.947 | 0.947 | 0.986 | 0.995 | 0.959 | 0.990 | 0.987 | 0.980 | 1.000 |
| Advanced Hit@5 | 0.960 | 0.943 | 0.980 | 1.000 | 0.977 | 0.992 | 0.981 | 0.979 | 1.000 |
| Beginner F1 | 0.980 | 0.960 | 0.958 | 0.989 | 0.979 | 0.990 | 0.994 | 0.986 | 0.994 |

### 5.2 Full document vs. selected context (advanced, same 150 test pairs)

| Context | Prompt | Macro-F1 | Input tokens | Output tokens | Latency p95 |
|---|---|---:|---:|---:|---:|
| Full booklet | JSON | 0.730 | 59,517 | 202 | 43.7 s* |
| Proposal filter, top 5 pages | JSON | 0.795 | 3,408 | 157 | 36.2 s* |
| Hybrid, top 10 pages | ids | 0.926 | 6,023 | 55 | 3.8 s |
| 12 pages, clipping, speaker boost, decision rules | ids | 0.946 | 5,713 | 55 | 2.8 s |
| **Final: the claim's part (2.2), all 1,488 pairs** | **ids** | **0.977** | 1,757 | 59 | 1.4 s |

\* measured while three runs shared the endpoint. The full booklet is the most expensive and least accurate option.

### 5.3 Ablations (dev and test split, then all pairs)

| Change | Effect | Decision |
|---|---|---|
| `ids` answer instead of JSON with reasoning + quotes | beginner 0.973 → 0.980, advanced 0.786 → 0.797, p95 5.5 s → 1.8 s | adopted |
| Hybrid retrieval instead of proposal filter | advanced 0.797 → 0.862 | adopted |
| Viewpoint and recommendation rules in the prompt | advanced 0.862 → 0.870, beginner 0.980 → 0.993 | adopted |
| Decision-Rule 0: keep the label when it disagrees with the confidences | 44 fixed / 13 broken over all saved runs; test 0.920 → 0.928 | adopted |
| Clip pages > 3,000 characters to the best-matching window | test 0.925 → 0.928, input tokens 5,725 → 4,696 | adopted |
| 12 instead of 10 pages (after clipping) | all 1,495 pairs 0.912 → 0.931 (51 fixed / 24 broken); +~890 input tokens | adopted |
| Speaker boost (2.2) | all pairs 0.931 → 0.940 (22 fixed / 8 broken), committee-claim errors 36 → 27, no extra tokens | adopted |
| Decision-Rule 0b (2.3) | all pairs 0.940 → 0.945 (8 fixed / 1 broken) | adopted |
| Thinking before answering (prefilled reasoning marker) | dev 450: 0.908 → 0.929 (20 fixed / 11 broken); ~1,300 output tokens and ~23 s per claim instead of 59 and ~2 s; a 300-token budget is worse | rejected (efficiency) |
| Other context and answer variants: 600-character passages, claim translated for the prompt, hiding the other side, verification pass, few-shot, speaker hint, label-only with logprobs, reasoning sentence, confidence thresholds, JSON schema (`response_format`), dropping low-value pages | each broke at least as many answers as it fixed (e.g. passages 0.831 vs. 0.908, JSON schema 0.933 → 0.833) | rejected |
| 8 instead of 12 pages (dev 300, official scorer) | F1 0.937 → 0.934, Hit@5 0.70 → 0.76, input tokens 5,370 → 3,721; k=7 / 6: F1 0.924 / 0.904; test F1 0.945 → 0.928 | adopted (tokens are scored) |
| Condensed system prompt (`PROMPT_SHORT`) | dev 300 B 0.980 → 0.850, A 0.937 → 0.781, 6–13 % fewer input tokens | rejected |
| Evidence: PyMuPDF pages + 2 best text blocks (same predictions) | dev 300 0.792 → 0.872, test 0.727 → 0.813 | adopted |
| Retrieval with claim + translation (`union`, paired real runs) | dev 300 0.934 → 0.944, test 0.930 → 0.945, all 1,488 0.948 → 0.956; ~3 % more tokens | adopted |
| *Paired runs on all 1,488 pairs per task (5 date folds; unchanged system run twice: 3 / 2 labels change):* | | |
| E1 / E3 / E11 / E4–E6: fewer legal-text pages, narrower windows, shorter lower-ranked pages; other evidence rankings | A 0.949 → 0.943 at −3 … −9 % tokens; Hit@5 within noise | rejected |
| E2: beginner references > 6,000 chars cut to the matching passages | B 0.981 → 0.984, −30 % input tokens; holdout 0.955 → 0.966 | adopted |
| E7/E8: four compact label rules in the advanced prompt | A 0.942 → 0.949 (22 fixed / 11 broken), Hit@5 0.867 → 0.881, +2.5 % tokens; holdout unchanged (2 / 2) | adopted |
| **E12: contents-guided part selection (2.2)** | A 0.949 → 0.968 (56 / 29; date bootstrap +0.005 … +0.035), Hit@5 0.881 → 0.975, −55 % tokens; holdout 0.933 → 0.956 | **adopted** |
| E13 / E14: the prompt's rules in fewer words (−210 tokens) / part budget 4,000 instead of 6,000 chars | A 0.968 → 0.952 (B → ~0.94; errors turn to contradiction) / A → 0.948, Hit@5 → 0.940 | rejected |
| E15 / E16: task B references cut at 3,000 chars / label-only answer inside the full prompt | B 0.984 → 0.980 (−21 % tokens) / 0.871 (no confidences: N→C) | rejected |
| E17a: task B references ≤ 9,000 chars cut to 3,000 | B 0.9845 → 0.9872, −5 % tokens; holdout equal | adopted |
| E18: short label-only prompts (four variants, 190 tokens) | B 0.987 → 0.948–0.966, −30 % tokens | rejected alone |
| **E19c: cascade, short call (≤ 2,000 chars), full prompt below p = 0.98** | B 0.987 → 0.982, −48 % tokens; holdout 3 → 4–6 wrong of 90 (4 in 5.1), −21 … −22 % | **adopted** |
| E21b: cascade at p = 0.999, 2,500 chars | B 0.987 → 0.987, −26 % tokens; holdout +18 % tokens | settings only |
| **E23: voting text 9,000 chars (2.2)** | A 0.968 → 0.974, Hit@5 → 0.983, +2.3 % tokens | **adopted** |
| E24: cue within 30 chars, more wordings | known claims unchanged; rephrased: wrong part 38 → 6 of 1,400 | adopted |
| **E25** / E27: no role sentence / short prompt | A 0.977, −34 tokens / 0.941, holdout 9 wrong | **adopted** / no |

Full tables: [docs/experiments.md](docs/experiments.md) and [docs/independent_experiments.md](docs/independent_experiments.md).
Runs on 150 pairs carry about ±3 F1 points of noise; close decisions were re-run on 450, the last on all pairs.

### 5.4 What we learned about Apertus

- **Selected context beats more context.** With the whole booklet (~60k tokens) Apertus 70B reaches 0.73 macro-F1,
  with 10 selected pages 0.93, with only the part the claim refers to 0.97. There is no position effect ("lost in the
  middle"); the degradation is general distraction.
- **Apertus is over-confident.** Its self-reported confidences are 0 or 1 in 97–99 % of cases, and the probability of
  the label token is ~1.0 for wrong answers as well as for correct ones (41 wrong vs. 40 correct dev answers):
  thresholds cannot filter the errors of the full prompt. The short task B call is the exception: 84 % of its
  answers are saturated and its errors lie in the rest, which the cascade (2.4) re-asks.
- **Distraction, not missing text, caused most errors.** With 8 retrieved pages the gold page was in the context in
  97.9 % of the Entailment/Contradiction cases and in 69 of 85 label errors; removing the other parts and votes
  (E12) fixed 56 answers. Trimming the part itself (E14) or the prompt's wording (E13, E25, E27) costs accuracy at once; only the role
  sentence could go (E25).
- **Speaker attribution was a context problem.** Committee claims failed in 13.4 % of cases with retrieved pages;
  with the committee's own part as the context, 3 of 275 fail.
- **Thinking helps, at a high price.** Letting the thinking model reason before its answer fixes half of the remaining
  dev errors (+0.021 F1 on dev 450), but needs ~22× the output tokens and ~11× the latency; the reasoning marker has
  to be prefilled, and selective thinking or a short budget saves little. **Apertus 8B** is ~3.5× faster but 6 F1
  points weaker than 70B on the same contexts.

---

## 6. Limitations

- **Remaining errors** (advanced, 34 of 1,488 pairs): Contradiction → Neutral 10, → Entailment 10, Entailment →
  Contradiction 6, Entailment → Neutral 5, Neutral → Contradiction 3. Voting-text claims: 10 of 235 wrong (24
  before E23).
- **Part selection relies on the claims naming their source** and on a readable contents page (2020–2026, 62
  booklets); otherwise it falls back to the retrieval of 8 pages (0.949 / Hit@5 0.881). Rephrased openings (E24,
  offline stress test) reach that fallback, not a wrong part (38 → 6 of 1,400).
- **Unseen booklets:** the holdout (0.989 advanced, 0.956 before E23; 0.955 beginner) covers two votes with 90 claims written by us
  (± ~0.05). The jury's set may differ in layout and claim style.
- **Development data:** all 20 dates of the dataset have been used for decisions; the holdout confirmed E2, E8,
  E12, E19c and E25 (on two votes) and suggested E23. Our time and token numbers are estimates (client-side counts, in-flight clock); booklets
  outside the 60 dataset booklets are parsed at run time.
- **Not political advice:** outputs only describe the relationship between a claim and the booklet.

---

## 7. Reproducibility

```bash
docker build --platform linux/amd64 -t fact-attack .          # pre-parses the 60 dataset booklets
docker run --rm -e BASE_URL -e API_KEY -v "$PWD/cases:/data:ro" -v "$PWD/output:/output" \
  fact-attack --input /data/cases.jsonl --output /output/predictions.jsonl
make test                                                      # unit tests, no API calls
```

Cases come from the starter repo's `prepare_cases.py` (or `scripts/prepare_competition_cases.py`, `build_holdout.py`)
and are scored with its `evaluate.py`; `scripts/run_decision.py` runs the submission configuration through the
official CLI and keeps predictions, scores and settings in `results/generalization/`.

Settings: `.env.example` lists every setting with its default (a test keeps it equal to `src/config.py`), among them
`NLI_TOP_K` (`8`), `PROMPT_MODE` (`ids`), `RETRIEVAL_QUERY_MODE` (`union`), `PART_CONTEXT` (`true`), `PART_CONTEXT_CHARS_LEGAL` (`9000`), `A_PROMPT_DROP` (`role`),
`REFERENCE_MID_CHARS` (`3000`), `B_PROMPT` (`topic`), `B_ESCALATE_BELOW` (`0.98`), `B_SHORT_CHARS` (`2000`).

---

## 8. Next steps

1. Claims without a recognisable opening: choose the part from the claim's content.
2. The remaining Contradiction → Neutral errors; a larger holdout over several new votes.

**License:** code Apache-2.0 ([LICENSE](../LICENSE)); report CC-BY-4.0. With PyMuPDF (AGPL-3.0) the image is AGPL-3.0.
