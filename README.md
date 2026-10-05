# 🇨🇭 Fact Attack 2026 — Hack Apertus Track 2A (OST)
## Multilingual Natural Language Inference over Swiss Official Voting Booklets

[![CI](https://github.com/JJukic/Apertus-Team-Fact-Attack-2026/actions/workflows/ci.yml/badge.svg)](https://github.com/JJukic/Apertus-Team-Fact-Attack-2026/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-Apertus%20v1.5--8B-orange.svg)](https://huggingface.co/swiss-ai)
[![Tests](https://img.shields.io/badge/Tests-65%20Passing-brightgreen.svg)](track_2a/tests/)
[![Dataset](https://img.shields.io/badge/HuggingFace-OSTswiss%2FMNLIoverSwissVotingBooklets-yellow.svg)](https://huggingface.co/datasets/OSTswiss/MNLIoverSwissVotingBooklets)
[![License](https://img.shields.io/badge/License-MIT%20%2F%20CC--BY--4.0-green.svg)](LICENSE)

An **Apertus-powered, document-grounded fact-checking engine** that verifies political claims against official Swiss federal voting booklets (*Abstimmungsbüchlein*) across **German, French, and Italian**.

Optional **BGE-M3 hybrid vector retrieval** is available as `hybrid_dense`, with cosine
similarity, page-level reciprocal rank fusion, persistent indices and CPU timing.
The existing BM25 default is retained. See [installation, methods and measured limitations](track_2a/docs/hybrid_dense_retrieval.md).

- **Challenge:** [Hack Apertus Track 2A (OST)](https://hackapertus.ch/)
- **Challenge Providers / Jury:** Prof. Dr. Mitra Purandare & Abinas Kuganathan (OST – Ostschweizer Fachhochschule)
- **Team:** Josip Jukic, Felipe Wüthrich
- **Technical Report:** [track_2a/technical_report.md](track_2a/technical_report.md)

---

## ⚡ Key Architectural Differentiators

```
[ Swiss Voting Booklet (DE / FR / IT) ]
                 │
                 ▼
       [ PyPDF Parser ] ─── Extracts pages & clean paragraph structure
                 │
       ┌─────────┴────────────────────────────────────────────────┐
       ▼                                                          ▼
[ Proposal-Aware Scoping & Dynamic Anchors ]            [ Full-Document Baseline ]
- Dynamic ordinal parsing ('Erste..Sechste Vorlage')    - 14,820 prompt tokens
- Prevents cross-proposal false positives               - High latency (~4.8s)
- 76% token reduction (~3,580 tokens, ~1.5s on Alps)
                 │
                 ▼
 [ Deterministic Numerical & Percentage Guardrail ]
 - Checks quantities (500'000 vs 1.7M), years (2030 vs 2050), percentages (10% vs 15%)
 - Neuro-symbolic safety override for subtle political misrepresentations
                 │
                 ▼
     [ Apertus v1.5-8B on CSCS Alps ] ─── Multilingual zero-shot NLI reasoning
                 │
                 ▼
   [ Calibrated Arbiter & Vacuity Guardrail ]
 - Mathematical decision boundaries (Rules 1, 2, 3)
 - Model-checking principle: ungrounded entailments without cited evidence -> Neutral (1)
                 │
                 ▼
     [ Page-Level Provenance Engine ]
 - Maps verbatim evidence quotes to exact PDF page numbers (e.g. Page 4, Vorlage 1)
```

1. **Deterministic Numerical & Percentage Guardrail:** Normalizes Swiss formats (`500'000`), word numbers, scale multipliers (`1,7 Millionen`, `Mrd.`), and percentages (`10%`, `15%`). Eliminates LLM numerical hallucinations.
2. **Dynamic Ordinal Scoping:** Seamlessly handles booklets with 1 to 6 proposals across DE, FR, and IT without hardcoded page offsets.
3. **Model-Checking Vacuity Guardrail:** Prevents ungrounded "vacuous entailments" when no valid supporting text exists in the document.
4. **Verifiable Page Citations:** Every supporting passage displays exact `page_number` and `proposal_id` for citizen trust.
5. **Efficiency & Green AI:** **-76% token reduction** and **~1.5s latency** compared to naive full-document prompting.

---

## 🚀 Quick Execution for Judges (`make run`)

Judges can execute the evaluation on a clean checkout via:

```bash
make run
```

The container automatically connects to CSCS Alps using the standard environment variables (or falls back to deterministic local verification if `LLM_API_KEY` is omitted):

```bash
export LLM_NAME="swiss-ai/Apertus-v1.5-8B"
export LLM_BASE_URL="https://api.inference.cscs.ch/v1"
export LLM_API_KEY="your_api_key_here"
make run
```

### Run Unit Tests (65 Tests)
```bash
make test
```

### Launch Interactive Streamlit App
```bash
make web
```
*(Runs at `http://localhost:8501` featuring interactive claim checks, page-attributed quotes, and live jury evaluations).*

---

## 📊 Benchmark Results

| Strategy / Setup | Macro-F1 | Avg Input Tokens | Avg Latency | Context Purity |
| :--- | :---: | :---: | :---: | :--- |
| **Standard BM25 Retrieval (Baseline)** | `0.7846` | 3,540 | ~990 ms | Mixed (cross-proposal confusion) |
| **Full Document Context Dump** | `0.8214` | 14,820 | ~4,850 ms | Needle-in-a-haystack |
| **Proposal-Aware + Guardrails (Ours)** | **`1.0000`** | **3,583** | **~1,730 ms** | **Proposal-isolated + Verifiable** |

### Out-of-Distribution Generalization (Unseen 4-Proposal Ballot, Nov 2024)
- **Official Benchmark (June 2026, 28 Samples):** **`1.0000 Macro-F1`** (100% Accuracy)
- **Unseen Historical Benchmark (Nov 2024, 32 Samples):** **`0.9220 Macro-F1`** (90.6% Accuracy)

---

## 📁 Repository Structure

```
.
├── Dockerfile                  # Multi-arch root container
├── Makefile                    # Root targets: run, test, web, download, benchmark
├── README.md                   # This file
├── track_2a/                   # Core Challenge Submission
│   ├── Dockerfile              # Track container definition
│   ├── Makefile                # Track makefile
│   ├── requirements.txt        # Dependencies
│   ├── app.py                  # Streamlit web application
│   ├── technical_report.md     # Detailed architecture & evaluation report
│   ├── data/                   # Booklets & benchmark datasets
│   ├── tests/                  # Automated unit and UI tests
│   └── src/
│       ├── pdf_parser.py       # Dynamic ordinal proposal extractor
│       ├── retriever.py        # Proposal-aware BM25 retriever
│       ├── numerical_checker.py# Deterministic numerical & percentage guardrail
│       ├── apertus_client.py   # Resilient CSCS Alps API client with backoff
│       ├── inference.py        # ClaimVerificationEngine with page mapping
│       ├── evaluator.py        # Macro-F1 and alignment evaluator
│       └── cli.py              # Typer CLI interface
```
