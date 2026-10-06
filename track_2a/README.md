# Hack Apertus — Track 2A: OST Challenge
## Multilingual Natural Language Inference over Swiss Official Voting Booklets

[![CI](https://github.com/JJukic/Apertus-Team-Fact-Attack-2026/actions/workflows/ci.yml/badge.svg)](https://github.com/JJukic/Apertus-Team-Fact-Attack-2026/actions/workflows/ci.yml)
[![Model](https://img.shields.io/badge/Model-Apertus%20v1.5-orange.svg)](https://huggingface.co/swiss-ai)
[![Dataset](https://img.shields.io/badge/HuggingFace-OSTswiss%2FMNLIoverSwissVotingBooklets-yellow.svg)](https://huggingface.co/datasets/OSTswiss/MNLIoverSwissVotingBooklets)

An Apertus-powered system that decides whether an official Swiss voting booklet **entails (0)**, is **neutral to (1)**
or **contradicts (2)** a claim, and cites the booklet pages that justify the decision — for every combination of
German, French and Italian.

- **Results** (test split of 5 unseen voting dates, 402 pairs): advanced task **0.928** macro-F1, beginner task **0.975**
- **Overview:** [../README.md](../README.md) · **Technical report:** [technical_report.md](technical_report.md) ·
  **Experiment log:** [docs/experiments.md](docs/experiments.md)

> Predictions describe the relationship between a claim and the official booklet. They are not political advice.

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
NLI_TOP_K=10
PROMPT_MODE=ids          # 'ids' (default) | 'json' | 'compact'
PAGE_MAX_CHARS=3000      # clip over-long pages (0 = off)
```

Without `LLM_API_KEY` the system runs in an offline heuristic mock mode (for tests only).

---

## 📋 Requirements

| | |
|---|---|
| **Runtime** | Docker (`make run` builds and runs everything in a `python:3.11-slim` container); locally Python 3.9+ with `requirements.txt` |
| **Hardware** | Any CPU machine, no GPU: ~300 MB RAM (peak 253 MB measured), ~1.4 GB disk for the image. The model runs remotely |
| **API keys** | `LLM_API_KEY` for the CSCS inference service (Apertus). `LLM_NAME` and `LLM_BASE_URL` default to `swiss-ai/Apertus-v1.5-70B-thinking` and `https://api.inference.cscs.ch/v1` |
| **Model weights** | None to download: Apertus v1.5 70B is served by CSCS. No other model is used, neither in the pipeline nor for evaluation |
| **Network** | At build time: Hugging Face (OST dataset) and admin.ch (booklet PDFs). At run time: the CSCS endpoint |

---

## 💻 CLI

```bash
# Advanced task: booklet + claim (+ vote title)
python -m src predict -b data/booklets/2026-06-14_fr.pdf \
  -c "Der Bundesrat empfiehlt, die Initiative abzulehnen." \
  -v "Initiative populaire « Pas de Suisse à 10 millions ! (initiative pour la durabilité) »" --json

# Beginner task: reference text + claim
python -m src predict -r "Der Bundesrat lehnt die Initiative ab." -c "Le Conseil fédéral recommande d'accepter l'initiative." --json

# Batch file in the official OST format (JSON or JSONL) -> one prediction per case
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

## 🐳 Docker (`make run`)

```bash
export LLM_API_KEY="your_api_key_here"
make run      # builds the image (downloads + pre-parses the booklets) and runs the benchmark
make test     # unit tests, no API calls
```

Other commands run inside the container the same way, e.g. `docker run --rm -e LLM_API_KEY -v $PWD/cases:/cases
hackapertus-track2a run -i /cases/cases.jsonl -o /cases/predictions.jsonl`.

---

## 📁 Project structure

```
track_2a/
├── app.py                    # Streamlit demo (fact check, benchmark dashboard, architecture)
├── technical_report.md       # Architecture, evaluation, findings, limitations
├── docs/experiments.md       # Every experiment with its numbers
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
