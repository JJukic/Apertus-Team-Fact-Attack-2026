# 🇨🇭 Fact Attack 2026 — Hack Apertus Track 2A (OST)
## Multilingual Natural Language Inference over Swiss Official Voting Booklets

[![CI](https://github.com/JJukic/Apertus-Team-Fact-Attack-2026/actions/workflows/ci.yml/badge.svg)](https://github.com/JJukic/Apertus-Team-Fact-Attack-2026/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12.6-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-Apertus%20v1.5--70B-orange.svg)](https://huggingface.co/swiss-ai)
[![Dataset](https://img.shields.io/badge/HuggingFace-OSTswiss%2FMNLIoverSwissVotingBooklets-yellow.svg)](https://huggingface.co/datasets/OSTswiss/MNLIoverSwissVotingBooklets)
[![License](https://img.shields.io/badge/License-Apache--2.0%20(code)%20%2F%20CC--BY--4.0%20(docs)-green.svg)](LICENSE)

An **Apertus-powered, document-grounded claim-verification system** that decides whether an official Swiss
voting booklet (*Abstimmungsbüchlein*) **entails (0)**, is **neutral to (1)** or **contradicts (2)** a political
claim, and cites the booklet pages that justify the decision. Claims and booklets can be in German, French or
Italian, in any combination.

- **Challenge:** [Hack Apertus Track 2A (OST)](https://hackapertus.ch/)
- **Team:** Josip Jukic, Felipe Wüthrich, Sheyla Sampietro, Andrea Petretta
- **Technical Report:** [track_2a/technical_report.md](track_2a/technical_report.md) ([PDF](track_2a/docs/FactAttack_Technical_Report.pdf)) · **Experiment log:** [track_2a/docs/experiments.md](track_2a/docs/experiments.md)

> Predictions describe the relationship between a claim and the official booklet. They are not political advice.

---

## 📊 Results

Model: `swiss-ai/Apertus-v1.5-70B-thinking` on CSCS. Every number below comes from the submission configuration run
through the official CLI (`--input/--output`) and scored with the organisers' `evaluate.py`; predictions and reports
are saved in [`track_2a/results/generalization/`](track_2a/results/generalization/).

| Evaluation set | Task | Macro-F1 | Hit@5 | Input tokens | Output tokens |
| :--- | :---: | ---: | ---: | ---: | ---: |
| **Holdout** (unseen booklets, 90 claims) | A | **0.989** | **0.983** | 1,825 | 57 |
| **Holdout** (unseen booklets, 90 claims) | B | **0.955** | – | 1,390 | 40 |
| **Dataset** (1,488 pairs) | A | **0.974** | **0.983** | 1,791 | 59 |
| **Dataset** (1,488 pairs) | B | **0.982** | – | 865 | 21 |

Tokens are means per case; Hit@5 is the share of gold entailment/contradiction cases where one of the first five evidence items overlaps the gold passage (task A only).

- **Tasks:** A (advanced) = booklet PDF + claim + vote; B (beginner) = reference text + claim. Mean Apertus time per
  case 1.1 s (A) / 0.25 s (B); processing time outside the model calls stays in the millisecond range per case.
- **Sets:** the holdout is our own, on the 27 September 2026 vote; the dataset is v1.1 with 20 voting dates in 5
  date-grouped folds (macro-F1 per fold: A 0.965–0.980, B 0.969–0.991).
- **What the holdout is:** 90 claims (30 per label, 10 per language pair) on the booklets of the 27 September 2026
  vote, which are not part of the dataset. We wrote them before the system ever ran on them, and a second team member
  checked every label (90/90 agree). We used the holdout only at the end: to confirm the task B changes (reference cut,
  tiered budget, cascade) and to check the compact label rules, the part selection and the voting-text budget for
  regressions (6 → 6 → 4 → 1 wrong advanced labels; two of the cases behind the last step had pointed us to it, so
  for that step the dataset is the measure). It is small (± ~0.05 macro-F1) and, for task B, covers only two votes in three languages
  (about 30 distinct claims).
- **Whole dataset:** all of it has been used for design decisions by now, including the earlier 5-date test split
  (where the submission before the compact label rules scored A 0.942 / Hit@5 0.833, B 0.972); it shows how stable
  the results are across voting dates, not how they generalise. Running the same configuration twice changes 3 of
  1,488 labels.

---

## ⚙️ How it works

```
 Booklet PDF (DE/FR/IT)                       Claim (DE/FR/IT)  +  vote title
        │                                              │
        ▼                                              │
 pypdf → pages, each labelled with its section         │
 ("Arguments of the committee", "Arguments of the      │
  Federal Council", "Voting text", …); PyMuPDF text    │
  and text blocks of every page for the evidence       │
        │                                              │
        └──────► Contents-guided part selection ◄──────┘
        claim opening → part (summary / detail / committee / Federal Council / voting text)
        contents page → pages of that part of the case's vote → context (≤ 6,000 chars, voting text ≤ 9,000), [P1] …
                                  │
                                  │   fallback, no part found: hybrid retrieval
                                  │   BM25(claim + its translation) + 2 × BM25(vote title) over all pages
                                  │   → top 8 pages (long ones clipped to 3,000 chars), [P1] … [P8]
                                  ▼
                 Apertus v1.5 (CSCS) — NLI prompt ('ids' mode)
                 → {"p_entail", "p_neutral", "p_contra", "label", "evidence_ids": [3, 7]}
                                  │
                                  ▼
     label 0/1/2 + evidence: cited pages and their best text blocks (PyMuPDF), verbatim, with page
```

1. **Contents-guided part selection (primary path).** The claims name their source in their opening ("Laut der
   Zusammenfassung …", "Le comité estime …", "Se il voto viene accettato …"), and every booklet lists, per vote, the
   pages of its parts on its contents page. The earliest cue in the claim's opening picks the part; the contents
   page, checked against the headings printed on those pages, gives its pages; the case's vote title picks the vote.
   That part (at most 6,000 characters; the voting text, often two dense legal pages, 9,000) becomes the whole
   context. A part is found for 1,476 of the 1,488 dataset claims.
2. **Fallback: hybrid retrieval that works across languages.** Claims without a recognisable part, and booklets
   without a readable contents page, get the 8 best pages by BM25 with the claim, its translation into the booklet
   language and the vote title. The vote title is written in the booklet's language, so it anchors the right
   proposal even when the claim is in another language; gold-section recall rose from 0.74 (hard proposal filter)
   to 0.93. If the claim names a side ("the committee argues …"), that side's two best argument pages are always in
   the context (speaker boost: committee-claim errors 13 % → 7 %).
3. **Evidence as passage ids.** Apertus cites the numbers of the pages it relies on instead of writing quotes,
   so evidence is always verbatim booklet text with a correct page number, and answers need ~55 output tokens
   instead of ~150.
4. **Booklet-aware prompt.** Booklets deliberately contain opposing viewpoints: a claim such as "the committee
   argues X" is checked against the committee's text only, and section labels tell the model who is speaking. For
   task A, four compact label rules follow (compare actor, action, object and conditions; a paraphrase can entail;
   missing support is not a contradiction).
5. **Decision rules.** A Neutral label next to a confident other relation follows the confidences; otherwise
   Apertus' own label is kept.
6. **Task B gets only the relevant part of a long reference, and a cheap first look.** The reference is split into
   passages; its opening passage and the passages that best match the claim (BM25, with the claim's translation for
   cross-lingual pairs) are kept in document order, up to 3,000 characters (6,000 above 9,000 characters). A short
   label-only call sees at most 2,000 of them; when its label probability is below 0.98, the full prompt is asked.
7. **Measured, not assumed.** Every design choice was tested on paired runs; ideas that did not help (a hard
   numerical override, label-only answers, small passages, full-booklet prompting, fewer or narrower pages) were
   dropped. See the [experiment log](track_2a/docs/experiments.md) and the
   [independent experiments](track_2a/docs/independent_experiments.md).

**Evidence:** every Entailment / Contradiction prediction returns up to five verbatim quotes with their 1-based page:
up to three cited pages as PyMuPDF reads them, plus the two text blocks of the cited pages that best match claim and
vote. Neutral predictions cite nothing.

---

## 🔬 Design decisions

The official score weights macro-F1, evidence, processing time (wall clock minus time waiting for Apertus) and tokens
(both relative to the best team). We chose every part of the submission by paired runs on the same cases; the tags
in brackets point to the [experiment log](track_2a/docs/experiments.md) and the
[independent experiments](track_2a/docs/independent_experiments.md):

- **Contents-guided part selection** [E12]: instead of 8 retrieved pages, the part of the vote the claim names.
  All 1,488 pairs: macro-F1 0.949 → 0.968 (date bootstrap +0.005 … +0.035), Hit@5 0.881 → 0.975, input tokens
  3,915 → 1,750 (−55 %); holdout 0.933 → 0.956. Rejected after measuring: a shorter system prompt [E13] (A −0.016,
  B ~−0.04) and a smaller part budget [E14] (−0.020).
- **A larger budget for the voting text** [E23]: an offline rebuild of every context showed that the gold page was
  missing in only 1 of 48 errors (so no embedding retrieval), but in 14 it was cut: a voting text over two dense
  pages lost the named article to the 6,000-character budget. 9,000 characters for that part only: all 1,488 pairs
  0.968 → 0.974 (date bootstrap +0.002 … +0.012, 5 of 5 folds equal or better), Hit@5 0.975 → 0.983, voting-text
  claims 24 → 13 wrong of 235, input tokens +2.3 %; holdout 0.956 → 0.989.
- **Robust to other claim wordings** [E24]: in an offline stress test with rephrased openings, claims that name
  their source differently fall back to the retrieval instead of a wrong part (wrong part on unseen wordings 38 → 6
  of 1,400): the source must start within the first 30 characters, and the cue list has other wordings in DE/FR/IT.
  All 3,173 known claims keep exactly their part, so every score above is unchanged.
- **Task B: tiered reference budget and cascade** [E17a, E19c]: references up to 9,000 characters keep 3,000
  (0.984 → 0.987, 1,732 → 1,640 input tokens). On top, a short label-only prompt answers first on at most 2,000
  reference characters; below a label probability of 0.98 (~10 % of the cases) we ask the full prompt again. All
  pairs: macro-F1 0.987 → 0.982, input tokens 1,640 → 865 (−48 %); on the holdout (two votes) 3 → 4–6 wrong labels of
  90 (4 in the run reported above, 6 in the first run; runs differ by about two labels) at −21 … −22 % tokens. The
  token score is relative to the other teams, so the net effect is estimated, not measured (technical report 2.4);
  `B_PROMPT=` restores the state before the cascade.
- **Task B: long references cut to the passages that match the claim** [E2]: −30 % input tokens on the whole dataset
  (macro-F1 0.981 → 0.984), −13 % on the holdout (0.955 → 0.966).
- **Compact label rules in the task A prompt** [E7/E8]: macro-F1 0.942 → 0.949, Hit@5 0.867 → 0.881 on all 1,488
  pairs (22 fixed / 11 broken) for +2.5 % tokens; neutral on the holdout.
- **Retrieval with the claim and its translation into the booklet language** [R2]: 48 fixed / 25 broken labels over
  three paired comparisons, macro-F1 0.930 → 0.945 on test, for ~3 % more tokens.
- **Evidence from PyMuPDF pages and text blocks**: Hit@5 0.727 → 0.813 on the same predictions.
- **8 instead of 12 retrieved pages**: 31 % fewer input tokens for −0.003 macro-F1 (dev 300).
- **Fast start-up and parsing**: a standard-library HTTP client, booklets parsed in a process pool and cases
  processed booklet by booklet: 104 → 8 ms per case on new booklets.
- **Measured and dropped:** fewer legal-text pages [E1] (−0.005 F1), narrower page windows [E3] (−9 % tokens but
  −0.008 F1), evidence blocks ranked with the translated claim (+0.004 Hit@5, inconsistent), section labels for
  navigation-bar pages [E5] (noise), evidence blocks only [E6] (gains only via the vote title in the gold passages),
  smaller windows for lower-ranked pages [E11] (−0.006 F1).

### Earlier comparison of retrieval and evidence variants

All 1,488 cases per task of the pinned v1.1 dataset, k = 12 retrieved pages, five folds grouped by voting date,
measured before the switch to the official CLI contract:

| Configuration | Advanced Macro-F1 | Beginner Macro-F1 | Advanced Hit@5 |
|---|---:|---:|---:|
| B0 baseline | 0.947699 | 0.981830 | 0.467677 |
| C1: grounded evidence and caching | 0.944034 | 0.982497 | 0.755556 |
| R1: translated BM25 query | 0.950844 | 0.981830 | 0.765657 |
| **R2: original/translated BM25 union** | **0.956237** | **0.982497** | **0.773737** |

See the [optimization report](track_2a/docs/competition_optimization_report.md),
[comparison](track_2a/results/competition_optimization/comparison.csv) and
[configuration](track_2a/results/competition_optimization/best_config.json).

### Full booklet vs. selected context (advanced task, same 150 test pairs)

| Context supplied to Apertus | Macro-F1 | Ø input tokens | Latency p95 |
| :--- | :---: | :---: | :---: |
| Full booklet (baseline) | 0.730 | 59,517 | 43.7 s* |
| Our first pipeline (proposal filter + BM25, verbose JSON) | 0.795 | 3,408 | 36.2 s* |
| Hybrid retrieval, 10 pages + `ids` prompt | 0.926 | 6,023 | 3.8 s |
| 12 pages (long ones clipped), speaker boost, decision rules | **0.946** | 5,713 | 2.8 s |

\* measured while several runs shared the endpoint. **Selecting a dozen pages beats sending the whole booklet by
0.22 F1 with ~10× fewer tokens**: with up to 70k tokens of context the relevant passage gets lost.
Every number comes from a saved run in [`track_2a/results/`](track_2a/results/).

### What we learned about Apertus

- **More context hurts:** the whole booklet (~60k tokens) scores 0.73, ten selected pages 0.93.
- **Over-confident:** the label probability is ~1.0 for wrong answers too, so confidence thresholds cannot catch the
  errors of the full prompt. The short task B call is the exception (84 % of its answers are saturated, its errors lie
  in the rest), which is what the cascade uses.
- **Sometimes self-contradictory:** in ~2 % of answers the label and the stated confidences disagree; the label is right
  3× as often, so we keep it (+0.008 F1 on test).
- **Speaker attribution was mostly a retrieval problem:** committee claims failed 13 % of the time because the committee's
  own page was often ranked just outside the context; always including the named side's best pages brought this to 7 %,
  the same as claims without a speaker.
- **Thinking pays in quality, not in efficiency:** reasoning before the answer adds ~2 F1 points (dev) but costs ~22× the output
  tokens and ~11× the latency, so it is off by default (`THINKING=true` to reproduce).
- **8B vs. 70B:** 8B is ~3.5× faster (p95 1.2 s) but 6 F1 points weaker (0.851 vs. 0.908 on 450 dev pairs).

Details and every rejected idea: [experiment log](track_2a/docs/experiments.md).

---

## 🚀 Quick start

### Submission container (official contract)

The organisers build or pull the image and call it with the case file only; `BASE_URL` and `API_KEY` are injected at
run time (no key in the image), `/data` is read-only and caches go to `/tmp`:

```bash
docker build --platform linux/amd64 -t fact-attack:dev .
docker run --rm --platform linux/amd64   -e BASE_URL=https://api.inference.cscs.ch/v1 -e API_KEY="$CSCS_API_KEY"   -v "$PWD/data/cases.jsonl:/data/cases.jsonl:ro" -v "$PWD/data/booklets:/data/booklets:ro"   -v "$PWD/output:/output"   fact-attack:dev --input /data/cases.jsonl --output /output/predictions.jsonl
```

Cases are processed 4 at a time (`NLI_WORKERS`); the CLI prints its own estimate of the scored processing time
(`[timing] … non-LLM … ms/case` on stderr). Local scoring: generate cases with the starter repo's `prepare_cases.py`
and score them with its `evaluate.py`.

**What the image contains.** Code, pinned dependencies, the 60 booklet PDFs of the public dataset and their parse
cache. The cache holds one JSON file per booklet, keyed by a SHA-1 hash of the PDF's bytes, with only text extracted
from that PDF: the pypdf text of every page, its passages with the parser's section labels, and the PyMuPDF text and
text blocks of every page. It holds nothing from the dataset rows (no claims, reference strings or labels) and no
queries, translations, retrieval scores or model outputs; the dataset file is downloaded and deleted in the same
build step, so it never reaches an image layer. The BM25 index, the contents/part map and all predictions are computed at
run time. A booklet outside the cache, or one whose bytes differ, is parsed the same way at run time; the cache only
saves parsing time.

### Development

```bash
export API_KEY="your_api_key_here"   # LLM_API_KEY works too
make run          # Docker: benchmark on the 398-pair test split, ~4 min (report + JSON in track_2a/results/)
make test         # unit tests (no API calls)
make web          # Streamlit app at http://localhost:8501
```

`make run` runs our own benchmark (task A) on the 398-pair test split (the 5 test voting dates in dataset v1.1, which
the benchmark downloads; 402 pairs in v1.0, as in the older entries of the experiment log; macro-F1 0.972, 1,800 input
tokens per case, checked on a fresh clone; temperature 0, so runs differ by at most a few pairs). The dataset is
downloaded when the run starts (the image holds no gold labels), so `make run` needs network access.

The main defaults (override via environment variables or `.env`; [`.env.example`](.env.example) lists every setting
with its default, and a test keeps that file equal to the code):

```bash
LLM_NAME=swiss-ai/Apertus-v1.5-70B-thinking
BASE_URL=https://api.inference.cscs.ch/v1   # LLM_BASE_URL works too
NLI_STRATEGY=hybrid                     # 'hybrid' | 'retrieval' | 'full'
NLI_TOP_K=8
PROMPT_MODE=ids                         # 'ids' | 'json' | 'compact'
RETRIEVAL_QUERY_MODE=union              # BM25 with the claim and its translation into the booklet language
EVIDENCE_POLICY=raw_pages_and_blocks    # cited PyMuPDF pages + best-matching text blocks
REFERENCE_MAX_CHARS=6000                # task B: longer references cut to the passages that match the claim
REFERENCE_MID_CHARS=3000                # task B: references up to 9,000 characters keep this many
B_PROMPT=topic                          # task B: short label-only first call, full prompt when unsure ("" = off)
B_ESCALATE_BELOW=0.98                   # task B: label probability below which the full prompt is asked again
B_SHORT_CHARS=2000                      # task B: reference characters the first call sees
NLI_WORKERS=4                           # cases processed in parallel by the batch CLI
```

Keep only `BASE_URL` / `API_KEY` in a local `.env`: any other value there changes what a local run measures.
Decision runs that must match the submission exactly: `python scripts/run_decision.py --data <cases> --name <run>`
(forces the `.env.example` settings, calls the official CLI, scores with `evaluate.py`, saves everything under
`track_2a/results/generalization/`); compare two runs with `scripts/report_decision.py`.

### CLI

```bash
cd track_2a

# Advanced task: booklet + claim (+ vote title)
python -m src predict --booklet data/booklets/2026-06-14_fr.pdf \
  --claim "Der Bundesrat empfiehlt, die Initiative abzulehnen." \
  --vote "Initiative populaire « Pas de Suisse à 10 millions ! (initiative pour la durabilité) »" --json

# Beginner task: reference text + claim
python -m src predict --reference "…" --claim "…" --json

# Batch file (JSON/JSONL) in the official OST format -> one prediction per case (the `run` command is optional)
python -m src --input cases.jsonl --output predictions.jsonl

# Reproduce the evaluation
python -m src.hf_dataset                                   # dataset, 60 booklets, dev/test split
python -m src benchmark -d data/hf/test.jsonl -t advanced  # saves a report to results/
python -m src benchmark -d data/hf/test.jsonl -t beginner
python -m src.results_summary                              # results/summary.json for the app
```

### Official input / output format

```jsonl
{"id":"case-0042","booklet":{"path":"booklets/2024-11-24_de.pdf","language":"de"},"vote":"Ausbauschritt 2023 für die Nationalstrassen","claim":{"text":"La proposition entraînera une augmentation de la TVA.","language":"fr"}}
{"id":"case-0043","reference":{"text":"Der Bundesrat empfiehlt, die Initiative abzulehnen.","language":"de"},"vote":"Beispielvorlage","claim":{"text":"Le Conseil fédéral recommande d'accepter l'initiative.","language":"fr"}}
```

`run` also reads rows as published in the OST dataset on Hugging Face (`claim`, `reference_string`, `booklet_url`,
`booklet_publish_date`, `vote`, optionally `claim_language`), as JSONL, JSON, Parquet or CSV. `--task auto` (default)
uses the booklet when a case names one and the reference text otherwise; `--task advanced` / `beginner` forces one.
In official mode, booklet paths must name existing input PDFs, resolved relative
to the input file. Provide them in the read-only input mount. Missing PDFs are
reported as failures; prediction does not download a booklet or substitute gold references.

Each case yields `{"id", "label", "label_name", "evidence": [{"page", "text"}], "metrics": {"input_tokens",
"output_tokens", "inference_time_ms"}}`; evidence is empty for neutral, and its page is `null` for a reference text. The claim language is detected
automatically, booklet paths are resolved robustly, and a failing case is reported on stderr and returned as a
valid neutral record so IDs remain complete; failing cases do not change the exit code (a transient endpoint problem
must not invalidate the whole run). Only an unreadable input (missing file, unsupported format) exits nonzero. For
task B the first, short answer is kept when the second call fails.
Source-derived caches are created in `/tmp`; the input mount remains unchanged.

---

## 📁 Repository structure

```
Dockerfile, .env.example      # Submission image (built from the repository root) and every setting with its default
track_2a/
├── app.py                    # Streamlit app (claim check, benchmark dashboard from results/)
├── technical_report.md       # Architecture, evaluation, limitations
├── docs/experiments.md       # Every experiment with its numbers
├── docs/independent_experiments.md  # Grouped validation, noise floor, holdout, experiments E1–E24
├── docs/FactAttack_Technical_Report.pdf  # Submission PDF of the report (docs/build_report_pdf.py)
├── results/                  # Saved benchmark runs (+ summary.json); generalization/ = decision runs
├── data/                     # Demo dataset, 2026-06-14 booklets, holdout claims; hf/ and other booklets are downloaded
├── scripts/                  # Decision runs (run_decision / report_decision), holdout builder, audit tools
├── vendor/                   # Organisers' starter scorer (evaluate.py)
├── tests/                    # Unit tests (no API calls)
└── src/
    ├── cli.py                # predict / run / benchmark / compare / warm-cache / download / web
    ├── config.py             # Every setting with its default (kept equal to .env.example by a test)
    ├── inference.py          # ClaimVerificationEngine: part or retrieval → Apertus → evidence; booklet cache
    ├── part_context.py       # Contents-guided part selection: claim opening → part, contents page → pages
    ├── pdf_parser.py         # Pages, section labels, proposal boundaries
    ├── quick_booklet.py      # Page-by-page reading of a booklet that is not parsed yet
    ├── page_pool.py          # Process pool that parses new booklets at start-up
    ├── retriever.py          # Hybrid (booklet-wide) and proposal-filtered BM25 retrieval (fallback)
    ├── query_translation.py  # Claim translated into the booklet language for ranking (guarded)
    ├── apertus_client.py     # CSCS client, prompts (ids / json / compact, task B cascade), retries
    ├── http_chat.py          # Standard-library HTTP client for the chat endpoint
    ├── evidence.py           # Evidence quotes: cited PyMuPDF pages and their best text blocks
    ├── text_utils.py         # De-hyphenation, chunking, snippet selection, language detection
    ├── numerical_checker.py  # Swiss number normalisation; reports numerical conflicts
    ├── evaluator.py          # Macro-F1, language pairs, evidence grounding, p95 latency
    ├── official_evaluation.py  # Offline scoring with the organisers' pinned evaluate.py
    ├── hf_dataset.py         # OST dataset + booklet download, dev/test split by voting date
    └── results_summary.py    # Collects saved runs for the app and README
```

The root `Dockerfile`, `.dockerignore` and `.env.example` define the submission image (the organisers and our
publish workflow build from the repository root); the root `Makefile` forwards every target to `track_2a/Makefile`.
`track_2a/` carries its own `Dockerfile` and `.dockerignore` for building from that folder (`make run`). A test keeps
both `.env.example` files byte-identical.

---

## 📜 License

Code: [Apache License 2.0](LICENSE). Documentation (READMEs, technical report, experiment log): CC-BY-4.0.

Third-party: the submission image includes [PyMuPDF](https://github.com/pymupdf/PyMuPDF) (AGPL-3.0), used to quote
evidence pages and text blocks from the booklets. Apache-2.0 code may be combined with AGPL-3.0 code; the image as a
whole is therefore distributed under the terms of the AGPL-3.0, and its complete source is this public repository.
