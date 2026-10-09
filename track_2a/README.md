# Hack Apertus — Track 2A: OST Challenge
## Multilingual Natural Language Inference over Swiss Official Voting Booklets

[![CI](https://github.com/JJukic/Apertus-Team-Fact-Attack-2026/actions/workflows/ci.yml/badge.svg)](https://github.com/JJukic/Apertus-Team-Fact-Attack-2026/actions/workflows/ci.yml)
[![Model](https://img.shields.io/badge/Model-Apertus%20v1.5-orange.svg)](https://huggingface.co/swiss-ai)
[![Dataset](https://img.shields.io/badge/HuggingFace-OSTswiss%2FMNLIoverSwissVotingBooklets-yellow.svg)](https://huggingface.co/datasets/OSTswiss/MNLIoverSwissVotingBooklets)

An Apertus-powered system that decides whether an official Swiss voting booklet **entails (0)**, is **neutral to (1)**
or **contradicts (2)** a claim, and cites the booklet pages that justify the decision — for every combination of
German, French and Italian.

- **Historical results** (402 pairs, initially held out by date and subsequently observed): Advanced **0.940**, Beginner **0.975** Macro-F1.
- **Closed v1.1 comparison** (1,488 cases per task, five date-grouped folds): recommended **R2 BM25 query union**, Advanced **0.956237**, Beginner **0.982497**, official Advanced Hit@5 **0.773737**.
- **Overview:** [../README.md](../README.md) · **Technical report:** [technical_report.md](technical_report.md) ([PDF](docs/FactAttack_Technical_Report.pdf)) ·
  **Experiment log:** [docs/experiments.md](docs/experiments.md)

> Predictions describe the relationship between a claim and the official booklet. They are not political advice.

R2 combines the original/translated BM25 queries with existing speaker boost and
source-grounded page/block evidence. It preserves the original claim and NLI
prompt. Reported Advanced tokens increase 1.39%; official separate-task efficiency
and private competition scores are unavailable. **115 local and 115 CPU-container
tests pass.** All 91 R2 label discrepancies have source reviews; two annotation/scope
ambiguities remain flagged without changing gold. Full comparison and reproduction:
[final report](docs/competition_optimization_report.md),
[comparison.csv](results/competition_optimization/comparison.csv),
[best_config.json](results/competition_optimization/best_config.json).

---

## 🚀 Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # then set LLM_API_KEY
```

Configuration (environment variables or `.env`):

```ini
LLM_NAME=swiss-ai/Apertus-v1.5-70B-thinking
LLM_BASE_URL=https://api.inference.cscs.ch/v1
LLM_API_KEY=your_api_key_here
NLI_STRATEGY=hybrid      # 'hybrid' (default) | 'retrieval' | 'full'
NLI_TOP_K=12
PROMPT_MODE=ids          # 'ids' (default) | 'json' | 'compact'
PAGE_MAX_CHARS=3000      # clip over-long pages (0 = off)
SPEAKER_BOOST=2          # always include the named side's 2 best argument pages (0 = off)
```

Without `LLM_API_KEY` the CLI stops with an error; `--mock` (or `MOCK_APERTUS=true`) runs an offline heuristic mock
mode for tests only.

---

## 📋 Requirements

| | |
|---|---|
| **Runtime** | Pinned CPU `linux/amd64` Docker image using Python 3.12.6; install `requirements-lock.txt` for the measured local environment |
| **Hardware** | CPU only; Apertus runs remotely. Recorded processing/memory measurements are in the final report |
| **API keys** | `LLM_API_KEY` for the CSCS inference service (Apertus). `LLM_NAME` and `LLM_BASE_URL` default to `swiss-ai/Apertus-v1.5-70B-thinking` and `https://api.inference.cscs.ch/v1` |
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
python -m src run -i cases.jsonl -o predictions.jsonl

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
cd ..  # repository root
docker build --platform linux/amd64 -t fact-attack:test .
# Set BASE_URL and API_KEY securely in the host environment.
# Place source-only cases.jsonl and its PDFs in ./cases.
mkdir -p output
docker run --rm --platform linux/amd64 --read-only --tmpfs /tmp \
  -e BASE_URL -e API_KEY -e NLI_STRATEGY=hybrid \
  -e RETRIEVAL_QUERY_MODE=union -e EVIDENCE_POLICY=raw_pages_and_blocks \
  -e CACHE_SINGLE_FLIGHT=true -e SOURCE_PAGES_LAZY=true \
  -e LLM_STREAMING=true -e LLM_JSON_REPAIR=true -e NLI_BATCH_WORKERS=4 \
  -v "$PWD/cases:/data:ro" -v "$PWD/output:/output" \
  fact-attack:test --input /data/cases.jsonl --output /output/predictions.jsonl
docker run --rm --network none --read-only --tmpfs /tmp fact-attack:test test
```

`NLI_BATCH_WORKERS=4` enables parallel official CLI cases while preserving output
order; its default is 1 for the baseline. Evaluation preparation and host-side gold
scoring are documented in the final report. The image includes source-only demo
inputs and demo PDFs. Prediction uses existing mounted PDFs and `/tmp` caches;
it does not download missing PDFs. Operational failures preserve IDs but produce
a nonzero exit code and a diagnostics sidecar. Runtime environment variables
`BASE_URL`/`API_KEY` take precedence over `.env` aliases. Baseline queries/evidence
remain selectable by disabling the experimental flags.

---

## 📁 Project structure

```
track_2a/
├── app.py                    # Streamlit demo (fact check, benchmark dashboard, architecture)
├── technical_report.md       # Architecture, evaluation, findings, limitations
├── docs/experiments.md       # Every experiment with its numbers
├── docs/FactAttack_Technical_Report.pdf  # Submission PDF of the report (docs/build_report_pdf.py)
├── results/                  # Saved benchmark runs + summary.json
├── data/
│   ├── booklets/             # 2026-06-14 booklets (others are downloaded)
│   ├── demo_dataset.jsonl    # Early 28-pair demo set
│   ├── demo_cases.json       # Verified demo examples for the app (with cached results)
│   └── demo_votes.json       # Vote titles per demo booklet
├── tests/                    # Unit tests
└── src/
    ├── cli.py                # predict / run / benchmark / compare / warm-cache / download / web
    ├── inference.py          # ClaimVerificationEngine: retrieval -> Apertus -> evidence pages; booklet cache
    ├── retriever.py          # Hybrid booklet-wide BM25 (claim + vote title) and proposal-filtered retrieval
    ├── pdf_parser.py         # Pages, section labels, proposal boundaries
    ├── apertus_client.py     # CSCS client, prompts (ids / json / compact), retries
    ├── text_utils.py         # De-hyphenation, chunking, snippet selection, language detection
    ├── numerical_checker.py  # Swiss number normalisation; reports numerical conflicts
    ├── evaluator.py          # Macro-F1, language pairs, evidence grounding, p95 latency
    ├── hf_dataset.py         # OST dataset + booklet download, dev/test split by voting date
    ├── results_summary.py    # Collects saved runs for the app and README
    ├── download_data.py      # Demo dataset + 2026/2024 booklets
    └── config.py             # Environment configuration
```
