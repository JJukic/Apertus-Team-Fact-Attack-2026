# Hack Apertus — Track 2A: OST Challenge
## Multilingual Natural Language Inference over Swiss Official Voting Booklets

[![CI](https://github.com/JJukic/Apertus-Team-Fact-Attack-2026/actions/workflows/ci.yml/badge.svg)](https://github.com/JJukic/Apertus-Team-Fact-Attack-2026/actions/workflows/ci.yml)
[![Model](https://img.shields.io/badge/Model-Apertus%20v1.5-orange.svg)](https://huggingface.co/swiss-ai)
[![Dataset](https://img.shields.io/badge/HuggingFace-OSTswiss%2FMNLIoverSwissVotingBooklets-yellow.svg)](https://huggingface.co/datasets/OSTswiss/MNLIoverSwissVotingBooklets)

An Apertus-powered system that decides whether an official Swiss voting booklet **entails (0)**, is **neutral to (1)**
or **contradicts (2)** a claim, and cites the booklet pages that justify the decision — for every combination of
German, French and Italian.

- **Results** (submission configuration, official CLI, organisers' `evaluate.py`):
  - **unseen booklets** (own holdout, 27 Sep 2026 vote, 90 claims): advanced **0.989** macro-F1,
    Hit@5 **0.983**, 1,825 input tokens; beginner **0.955**, 1,390 input tokens
  - **whole dataset** (v1.1, 1,488 pairs, 20 voting dates in 5 grouped folds): advanced **0.974**, Hit@5 **0.983**,
    1,791 input tokens; beginner **0.982**, 865 input tokens (before the task B cascade: 0.987, 1,640)
- **Overview:** [../README.md](../README.md) · **Technical report:** [technical_report.md](technical_report.md) ([PDF](docs/FactAttack_Technical_Report.pdf)) ·
  **Experiment logs:** [docs/experiments.md](docs/experiments.md), [docs/independent_experiments.md](docs/independent_experiments.md)

> Predictions describe the relationship between a claim and the official booklet. They are not political advice.

The earlier comparison of retrieval and evidence variants (R2 and others, all 1,488 cases, k=12) is in the
[optimization report](docs/competition_optimization_report.md). The measurements since then (reproducible decision
runs, a repeated baseline as noise floor, the holdout, and the experiments E1–E24) are in
[independent_experiments.md](docs/independent_experiments.md) with their predictions in
[results/generalization/](results/generalization/).

---

## 🚀 Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # then set API_KEY
```

Configuration (environment variables or `.env`). [`.env.example`](.env.example) lists every setting with its
default (what the submission image runs; a test keeps it equal to `src/config.py`). The main ones:

```ini
LLM_NAME=swiss-ai/Apertus-v1.5-70B-thinking
BASE_URL=https://api.inference.cscs.ch/v1   # official name; LLM_BASE_URL works too
API_KEY=your_api_key_here                   # official name; LLM_API_KEY works too
NLI_STRATEGY=hybrid                         # 'hybrid' (default) | 'retrieval' | 'full'
NLI_TOP_K=8
PROMPT_MODE=ids                             # 'ids' (default) | 'json' | 'compact'
PAGE_MAX_CHARS=3000                         # clip over-long pages (0 = off)
SPEAKER_BOOST=2                             # always include the named side's 2 best argument pages (0 = off)
RETRIEVAL_QUERY_MODE=union                  # BM25 with the claim and its translation into the booklet language
EVIDENCE_POLICY=raw_pages_and_blocks        # cited PyMuPDF pages + best-matching text blocks
REFERENCE_MAX_CHARS=6000                    # task B: longer references cut to the passages that match the claim
REFERENCE_MID_CHARS=3000                    # task B: references up to REFERENCE_LONG_CHARS (9000) keep this many
B_PROMPT=topic                              # task B: short label-only first call ("" = the full prompt only)
B_ESCALATE_BELOW=0.98                       # task B: full prompt again when the first answer is less sure
B_SHORT_CHARS=2000                          # task B: reference characters the first call sees
A_COMPACT_PROMPT=true                       # task A: compact label rules (E7, on in the submission; false = previous baseline)
PART_CONTEXT=true                           # task A: context = the booklet part the claim names, via the contents page (E12)
PART_CONTEXT_CHARS=6000                     # at most this many characters of that part
PART_CONTEXT_CHARS_LEGAL=9000               # the voting-text part may keep this many (E23; 0 = PART_CONTEXT_CHARS)
NLI_WORKERS=4                               # cases processed in parallel by the batch CLI
```

`A_COMPACT_PROMPT` (on by default, so active in the submission image, which sets no pipeline variable) appends four
label rules to the task A prompt only (task B is never affected). Evaluated in
E7 on 1,488 cases against a fresh baseline: macro-F1 0.9419 → 0.9493 (95 % date-bootstrap interval −0.002 … +0.018,
so not significant), Hit@5 858 → 872 of 990, +2.5 % input tokens; see `docs/independent_experiments.md`, section 7.
With `A_COMPACT_PROMPT=false` every request is byte-identical to the previous submission baseline.

`B_PROMPT=topic` (on by default, E18/E19c) is the task B cascade: a ~190-token label-only prompt that sees at most
`B_SHORT_CHARS` characters of the reference answers first; when its label probability is below `B_ESCALATE_BELOW` the
case is asked again with the full prompt (references up to 9,000 characters keep `REFERENCE_MID_CHARS`, E17a). All
1,488 task B pairs: macro-F1 0.987 → 0.982, input tokens 1,640 → 865; holdout (two votes) 3 → 4–6 wrong labels of 90
(4 in the run reported above, 6 in the first run).
`B_PROMPT=` (empty) restores the state before the cascade; see `docs/independent_experiments.md`, sections 17–18.

`PART_CONTEXT` (on by default, E12) replaces the 8 retrieved pages by the part of the booklet the claim's opening names,
located with the contents page (`src/part_context.py`); claims without such an opening, or booklets without a readable
contents page, keep the retrieval. All 1,488 task A pairs: macro-F1 0.949 → 0.968, Hit@5 0.881 → 0.975, input tokens
−55 %; see `docs/independent_experiments.md`, sections 11–13.

`PART_CONTEXT_CHARS_LEGAL` (9000 by default, E23) gives the voting-text part its own budget: a voting text often spans
two dense legal pages, and with 6,000 characters the article a claim names was cut off behind the first one. All
1,488 task A pairs: macro-F1 0.968 → 0.974, Hit@5 0.975 → 0.983, input tokens +2.3 %; holdout 0.956 → 0.989.
`PART_CONTEXT_CHARS_LEGAL=0` restores E12; see `docs/independent_experiments.md`, section 19.

A value set in a local `.env` overrides these for local runs; keep only endpoint and key there.
`python scripts/run_decision.py --data <prepared cases> --name <run>` runs exactly the submission configuration
(plus `--set KEY=VALUE` overrides) through the official CLI and saves predictions and scores.

Without an API key the CLI stops with an error; `--mock` (or `MOCK_APERTUS=true`) runs an offline heuristic mock
mode for tests only.

---

## 📋 Requirements

| | |
|---|---|
| **Runtime** | Pinned CPU `linux/amd64` Docker image using Python 3.12.6; install `requirements-lock.txt` for the measured local environment |
| **Hardware** | Any CPU machine, no GPU: ~300 MB RAM (peak 253 MB measured), ~1.4 GB disk for the image. The model runs remotely |
| **API keys** | `API_KEY` (or `LLM_API_KEY`) for the CSCS inference service (Apertus). `LLM_NAME` and `BASE_URL` default to `swiss-ai/Apertus-v1.5-70B-thinking` and `https://api.inference.cscs.ch/v1` |
| **Model weights** | None to download: Apertus v1.5 70B is served by CSCS. Apertus is the only model in the pipeline; Apertus 8B v1.5 was used once for a comparison run, no other model for development or evaluation |
| **Network** | Build downloads pinned Python dependencies; inference calls only the configured Apertus endpoint. Prepare datasets/PDFs outside prediction |

---

## 💻 CLI

```bash
# Advanced task: booklet + claim (+ vote title)
python -m src predict -b data/booklets/2026-06-14_fr.pdf \
  -c "Der Bundesrat empfiehlt, die Initiative abzulehnen." \
  -v "Initiative populaire « Pas de Suisse à 10 millions ! (initiative pour la durabilité) »" --json

# Beginner task: reference text + claim
python -m src predict -r "Der Bundesrat lehnt die Initiative ab." -c "Le Conseil fédéral recommande d'accepter l'initiative." --json

# Batch file in the official OST format or as Hugging Face dataset rows (JSON, JSONL, Parquet, CSV)
python -m src --input cases.jsonl --output predictions.jsonl   # official call; `run` may be given too

# Evaluation on the OST dataset
python -m src.hf_dataset                                    # dataset, 60 booklets, dev/test split
python -m src benchmark -d data/hf/test.jsonl -t advanced   # report saved to results/
python -m src benchmark -d data/hf/test.jsonl -t beginner
python -m src.results_summary                               # results/summary.json for the app

# Utilities
python -m src warm-cache                                    # pre-parse booklets into the disk cache
python -m src web                                           # Streamlit demo at http://localhost:8501
```

The input/output format is documented in the [overview README](../README.md#official-input--output-format).

---

## 🐳 Docker submission

```bash
export API_KEY="your_api_key_here"
make run      # builds the image (downloads + pre-parses the booklets), benchmarks the 398-pair test split (~4 min)
make test     # unit tests, no API calls
```

`NLI_WORKERS` sets the parallel requests (default 4); the benchmark downloads the OST dataset when the run starts.
The report is printed to the console; the run's JSON (metrics, run settings incl. the image's git commit, and
per-sample predictions) is written to `track_2a/results/` on the host.

Other commands run inside the container the same way, e.g. `docker run --rm -e API_KEY -v $PWD/cases:/cases
hackapertus-track2a run -i /cases/cases.jsonl -o /cases/predictions.jsonl`. Booklet paths in the cases are resolved
relative to the input file (so PDFs can sit next to it in the mounted folder); the 60 dataset booklets are already
in the image, and a `booklet_url` that is not is downloaded.

---

## 📁 Project structure

```
track_2a/
├── app.py                    # Streamlit demo (fact check, benchmark dashboard, architecture)
├── technical_report.md       # Architecture, evaluation, findings, limitations
├── .env.example              # Every setting with its default (= what the submission image runs)
├── docs/experiments.md       # Every experiment with its numbers
├── docs/independent_experiments.md  # Grouped validation, noise floor, holdout, experiments E1–E24
├── docs/FactAttack_Technical_Report.pdf  # Submission PDF of the report (docs/build_report_pdf.py)
├── results/                  # Saved benchmark runs + summary.json
│   └── generalization/       # Decision runs: predictions, official scores, fold/language/class reports
├── scripts/                  # run_decision / report_decision, holdout builder, competition audit tools
├── data/
│   ├── holdout_2026-09-27/   # Untouched holdout: 90 claims on the 27 Sep 2026 booklets + label review
│   ├── booklets/             # 2026-06-14 booklets (others are downloaded)
│   ├── demo_dataset.jsonl    # Early 28-pair demo set
│   ├── demo_cases.json       # Verified demo examples for the app (with cached results)
│   └── demo_votes.json       # Vote titles per demo booklet
├── tests/                    # Unit tests
└── src/
    ├── cli.py                # predict / run / benchmark / compare / warm-cache / download / web
    ├── inference.py          # ClaimVerificationEngine: part or retrieval -> Apertus -> evidence; booklet cache
    ├── part_context.py       # Contents-guided part selection: claim opening -> part, contents page -> pages
    ├── retriever.py          # Hybrid booklet-wide BM25 (claim + vote title) and proposal-filtered retrieval (fallback)
    ├── query_translation.py  # Claim translated into the booklet language for ranking (guarded)
    ├── pdf_parser.py         # Pages, section labels, proposal boundaries
    ├── quick_booklet.py      # Page-by-page reading of a booklet that is not parsed yet
    ├── page_pool.py          # Process pool that parses new booklets at start-up
    ├── apertus_client.py     # CSCS client, prompts (ids / json / compact, task B cascade), retries
    ├── http_chat.py          # Standard-library HTTP client for the chat endpoint
    ├── evidence.py           # Evidence quotes: cited PyMuPDF pages and their best text blocks
    ├── text_utils.py         # De-hyphenation, chunking, snippet selection, language detection
    ├── numerical_checker.py  # Swiss number normalisation; reports numerical conflicts
    ├── evaluator.py          # Macro-F1, language pairs, evidence grounding, p95 latency
    ├── official_evaluation.py  # Offline scoring with the organisers' pinned evaluate.py
    ├── hf_dataset.py         # OST dataset + booklet download, dev/test split by voting date
    ├── results_summary.py    # Collects saved runs for the app and README
    ├── download_data.py      # Demo dataset + 2026/2024 booklets
    └── config.py             # Environment configuration
```
