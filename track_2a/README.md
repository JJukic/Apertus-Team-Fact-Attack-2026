# Hack Apertus — Track 2A: OST Challenge
## Multilingual Natural Language Inference over Swiss Official Voting Booklets

[![CI](https://github.com/JJukic/Apertus-Team-Fact-Attack-2026/actions/workflows/ci.yml/badge.svg)](https://github.com/JJukic/Apertus-Team-Fact-Attack-2026/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-Apertus%20v1.5-orange.svg)](https://huggingface.co/swiss-ai)
[![Dataset](https://img.shields.io/badge/HuggingFace-OSTswiss%2FMNLIoverSwissVotingBooklets-yellow.svg)](https://huggingface.co/datasets/OSTswiss/MNLIoverSwissVotingBooklets)
[![License](https://img.shields.io/badge/License-MIT%20%2F%20CC--BY--4.0-green.svg)](LICENSE)

An Apertus-powered multilingual claim-verification system that checks political claims against official Swiss voting booklets (*Abstimmungsbüchlein*).

---

## 🎯 The Challenge & Task

Given an official Swiss voting booklet (PDF in German, French, or Italian) and a natural-language claim:
1. **Classify the claim-document relationship:**
   - **`0` — Entailment:** The booklet strictly supports the claim.
   - **`1` — Neutral:** The booklet does not provide enough information either way.
   - **`2` — Contradiction:** The booklet contradicts the claim.
2. **Extract transparent evidence:** Returns verbatim passages from the booklet that justify the classification.
3. **Report efficiency:** Measures prompt tokens, completion tokens, total tokens, and latency.
4. **Compare architectures:** Evaluates **Full Document Context** vs. **Passage Retrieval (BM25/chunking)**.

---

## 🚀 Quickstart

### 1. Requirements & Setup

Create a virtual environment and install dependencies:

```bash
# Using uv (fast) or standard pip
uv venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Configure your CSCS Apertus API key:
```ini
LLM_NAME=swiss-ai/Apertus-v1.5-8B
LLM_BASE_URL=https://api.inference.cscs.ch/v1
LLM_API_KEY=your_api_key_here
NLI_STRATEGY=retrieval
MOCK_APERTUS=false
```

*(Note: If `LLM_API_KEY` is empty or `MOCK_APERTUS=true`, the system runs in offline mock mode so you can test the pipeline immediately.)*

---

## 💻 CLI Usage

The system provides a full CLI via `python -m src`:

### 1. Verify a Single Claim (`predict`)
```bash
python -m src predict --claim "Die Initiative verlangt, die Wohnbevölkerung zu begrenzen." --lang de
```

Options:
- `--claim`, `-c`: The claim text to verify.
- `--booklet`, `-b`: Path to booklet PDF (default: `data/booklets/2026-06-14_de.pdf`).
- `--lang`, `-l`: Claim language (`de`, `fr`, `it`).
- `--strategy`, `-s`: `retrieval` (default, top-k passages) or `full` (entire booklet).
- `--top-k`, `-k`: Number of passages to retrieve (default: 5).

### 2. Run Benchmark Evaluation (`benchmark`)
Evaluates the official held-out benchmark and outputs Macro-F1 across 0, 1, 2, plus per-language metrics:
```bash
python -m src benchmark --strategy retrieval
```

### 3. Compare Strategies (`compare`)
Runs side-by-side comparison between **Passage Retrieval** and **Full Document**:
```bash
python -m src compare --limit 5
```

### 4. Download / Refresh Data (`download`)
```bash
python -m src download
```

### 5. Launch Interactive Web Demo (`app.py`)
Launch the interactive voting booklet verification demo in your browser:
```bash
streamlit run app.py
```

---

## 🐳 Docker & Submission (`make run`)

Judges run `make run` from the project root:

```bash
make run
```

This builds the Docker image and executes the benchmark inside the container using the environment variables passed:
```bash
export LLM_NAME="swiss-ai/Apertus-v1.5-8B"
export LLM_BASE_URL="https://api.inference.cscs.ch/v1"
export LLM_API_KEY="your_api_key_here"
make run
```

To run test suites:
```bash
make test
```

---

## 📁 Project Structure

```
track_2a/
├── Dockerfile                  # Container definition for reproducible evaluation
├── Makefile                    # Target `make run` for judges
├── requirements.txt            # Python dependencies
├── .env.example                # Template for Apertus CSCS credentials
├── README.md                   # This file
├── technical_report.md         # Deep-dive report & benchmark numbers
├── data/
│   ├── demo_dataset.jsonl      # Official benchmark dataset from Hugging Face
│   └── booklets/               # Official Swiss voting booklets (DE, FR, IT)
└── src/
    ├── __init__.py
    ├── __main__.py             # Entry point
    ├── config.py               # Env vars and label definitions
    ├── pdf_parser.py           # Extracts pages and clean paragraphs from PDF
    ├── retriever.py            # BM25 passage retrieval for claims
    ├── apertus_client.py       # Apertus API client, metrics & mock fallback
    ├── inference.py            # Claim verification engine (Full vs. Retrieval)
    ├── evaluator.py            # Computes Macro-F1 & efficiency metrics
    ├── download_data.py        # Automated data downloader
    └── cli.py                  # Typer & Rich CLI
```
