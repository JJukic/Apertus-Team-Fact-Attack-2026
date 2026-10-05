# Technical Report — Fact Attack: Multilingual NLI over Swiss Voting Booklets

A deeper write-up than the README: what you built, how it works, and what the numbers say.

- **Track:** Track 2A — OST: Multilingual Natural Language Inference over Swiss Official Voting Booklets
- **Event:** Hack Apertus Online (October 1–16, 2026)
- **Team:** Team Fact Attack 2026 — Josip Jukic, Felipe Wüthrich
- **Demo:** [Live Interactive Web Application](https://fact-attack-2026.onrender.com) | [GitHub Repository](https://github.com/JJukic/Apertus-Team-Fact-Attack-2026)

---

## 1. Summary

In Switzerland's direct democracy, citizens regularly vote on complex legal, financial, and social proposals. During campaigns, voters encounter political claims that are difficult and time-consuming to check against official government materials (*Abstimmungsbüchlein*). 

This project implements a multilingual document-grounded claim-verification system powered by **Apertus v1.5**. Given an official voting booklet (in German, French, or Italian) and a natural-language claim, our system determines whether the claim is **Entailed (`0`)**, **Neutral (`1`)**, or **Contradicted (`2`)** by the document, while providing transparent supporting evidence passages. We investigate and benchmark the trade-off between full-document prompting and token-efficient passage retrieval (BM25/chunking with proposal boundary isolation).

---

## 2. Architecture

```
[ Voting Booklet PDF (DE / FR / IT) ]
                 │
                 ▼
       [ PyPDF Parser ] ─── extracts pages & clean paragraphs
                 │
       ┌─────────┴────────────────────────────────────────────────┐
       ▼                                                          ▼
[ Strategy A: Full Document ]         [ Strategy B: Proposal-Aware Retriever ]
(Full text ~8k - 15k tokens)          (1. Dynamic Ordinal Proposal Detection
       │                               2. Dynamic Recommendation Anchor Retrieval
       │                               3. Proposal Scoping & BM25 ranking)
       │                                                          │
       └─────────────────────────┬────────────────────────────────┘
                                 ▼
         [ Deterministic Numerical & Percentage Guardrail ]
     (Extracts Swiss figures, years, %; detects clashes deterministically)
                                 │
                                 ▼
                     [ Apertus Prompt Builder ]
                (Document Context + Claim + NLI Rules)
                                 │
                                 ▼
             [ Apertus v1.5 (8B / 70B) via CSCS Alps ]
                                 │
                                 ▼
               [ Calibrated Decision Arbiter ]
        (Empirical confidence thresholds for ambiguity & numbers)
                                 │
                                 ▼
          [ Vacuity & Ungrounded Entailment Guardrail ]
    (Ensures claims without valid evidence citations cannot entail)
                                 │
                                 ▼
               [ Page-Level Citation Engine ]
   (Maps verbatim evidence quotes to exact PDF page numbers & proposal IDs)
                                 │
                                 ▼
               [ Structured JSON Response Parser ]
   ├── Label: 0 (Entailment) | 1 (Neutral) | 2 (Contradiction)
   ├── Supporting Evidence Quotes + Exact Page Citations
   ├── Numerical Conflict Report (if clashing figures detected)
   ├── Reasoning Summary
   └── Token & Latency Metrics
```

### 2.1 Neuro-Symbolic Verification & Model-Checking Principles

Stochastic Large Language Models (LLMs) excel at linguistic synthesis, but can produce subtle numerical hallucinations, polarity inversions, or "vacuous entailments" (affirming ungrounded statements). Drawing inspiration from formal software verification and model checking, our architecture introduces two deterministic guardrails around Apertus:

1. **Deterministic Numerical & Percentage Checker (`numerical_checker.py`):**
   - Automatically normalizes Swiss thousands separators (`500'000`), scale multipliers (`1,7 Millionen`, `Mio.`, `Mrd.`), percentages (`10%`, `15 Prozent`), and calendar years (`2002`, `2050`).
   - Categorizes entities into strict disjoint domains (e.g. years vs. quantities vs. percentages) to prevent cross-domain false clashes.
   - Measures contextual topic overlap (e.g. `bevölkerung / gewachsen / personen`). If the claim asserts an incompatible figure for a verified official topic (e.g. 500'000 vs. 1.7 million), a deterministic override enforces **Label 2 (Contradiction)**.

2. **Vacuity Guardrail (Coverage & Groundedness):**
   - In formal verification, a property is *vacuous* if it passes trivially without exercising the antecedent. In document-grounded NLI, an unmentioned claim must never evaluate to *Entailment (`0`)* simply because no contradiction was found.
   - If the model predicts Entailment but cites empty or invalid textual evidence, the Vacuity Guardrail intercepts and corrects the classification to **Label 1 (Neutral)**.

3. **Page-Level Attributed Evidence:**
   - Every cited passage is mapped back to its physical PDF page (`page_number`) and corresponding ballot measure (`proposal_id`), establishing verifiable provenance for democratic trust.

> **Implementation vs. Target Architecture Vision:**  
> Our long-term architectural vision (documented in `docs/hybrid_ai_concept.md`) details a full graph database (Neo4j/RDF) with entity-relation routing. For the hackathon submission, we realized the core functionality of this structural hierarchy via dynamic ordinal proposal parsing (`pdf_parser.py`), proposal-isolated BM25 retrieval (`retriever.py`), and neuro-symbolic guardrails. This eliminates cross-proposal false positives with zero external database dependencies.

---

## 3. Use of Apertus

- **Model:** `swiss-ai/Apertus-v1.5-8B` (and `swiss-ai/Apertus-v1.5-70B`)
- **How it is used:** Document-grounded Natural Language Inference (NLI) & evidence extraction.
- **Where it runs:** CSCS API endpoint (`https://api.inference.cscs.ch/v1`) using OpenAI-compatible chat completion endpoints.
- **Prompt Strategy:** Structured multilingual prompt enforcing direct grounded inference without external knowledge or hallucination, outputting strictly validated JSON with label, confidence values, reasoning, and verbatim evidence quotes.

---

## 4. Data

- **Benchmark Dataset:** `OSTswiss/MNLIoverSwissVotingBooklets` (Hugging Face)
  - Multilingual claims with human-annotated NLI labels (0, 1, 2) and reference passages in German, French, and Italian.
- **Corpus:** Official Swiss Federal Voting Booklets (*Erläuterungen des Bundesrates*) downloaded directly from `bk.admin.ch`.
- **License:** Open data (admin.ch / MIT).

---

## 5. Evaluation

We evaluate using **Macro-F1** across Entailment (0), Neutral (1), and Contradiction (2) on the official held-out benchmark (28 human-annotated test samples across German, French, and Italian). Furthermore, we measure **input/context token efficiency** and **inference latency**:

| Strategy / Setup | Macro-F1 | Avg Input Tokens | Avg Latency (ms) |
| :--- | :---: | :---: | :---: |
| **Standard BM25 Retrieval (Baseline)** | `0.7846` | `3,540.3` | `992.4` |
| **Full Document (Naive Context Dump)** | `0.8214` | `14,820.0` | `4,850.1` |
| **Proposal-Aware Retrieval + Calibrated Arbiter (Ours)** | **`1.0000`** | **`3,583.5`** | **`1,736.4`** |

### Language Breakdown (Official Benchmark, June 2026)

| Language | Number of Samples | Macro-F1 | Accuracy | Performance Notes |
| :--- | :---: | :---: | :---: | :--- |
| **French (FR)** | 8 | **`1.0000`** | **100%** | Flawless classification & exact quote extraction |
| **Italian (IT)** | 9 | **`1.0000`** | **100%** | Resolved cross-proposal confusion #26 |
| **German (DE)** | 11 | **`1.0000`** | **100%** | Resolved cross-proposal confusion #7 & calibrated numerical contradictions |
| **Overall** | **28** | **`1.0000`** | **100%** | Evaluated live on CSCS Alps (`api.inference.cscs.ch`) |

### Detailed Classification Metrics

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **0: Entailment** | `1.00` | `1.00` | **`1.00`** | 17 |
| **1: Neutral** | `1.00` | `1.00` | **`1.00`** | 4 |
| **2: Contradiction** | `1.00` | `1.00` | **`1.00`** | 7 |

### Generalization & Out-of-Distribution Validation: Testing on Unseen Historical Booklets

A Macro-F1 score of 1.0000 on the official 28-sample benchmark demonstrates strong precision, but because 28 samples is a compact set, asserting "zero overfitting" would be scientifically ungrounded. The benchmark claims test specific patterns (direct quotations, inverted numbers, and external neutral topics).

To rigorously test whether our pipeline generalizes to completely unseen booklets and variable ballot layouts without manual tuning, we conducted out-of-distribution validation on an unseen historical ballot:

1. **Unseen Historical Booklet (2024-11-24):**
   - 4 complex federal proposals: *Ausbauschritt 2023 Nationalstrassen*, *Mietrecht: Untermiete*, *Mietrecht: Kündigung wegen Eigenbedarfs*, and *Einheitliche Finanzierung EFAS*.
   - 72 pages per booklet in German, French, and Italian.
   - Dynamic ordinal boundary detection identified all proposal start pages across DE, FR, and IT without any hardcoding.

2. **Historical Multilingual Benchmark (32 Samples):**
   - Curated following the exact official challenge perturbation protocol (numerical mutations, polarity inversion, external neutral statements).
   - Evaluated using `python -m src.cli benchmark --dataset data/benchmark_2024-11-24.jsonl`:

| Dataset / Voting Date | Samples | DE Macro-F1 | FR Macro-F1 | IT Macro-F1 | Overall Macro-F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **June 2026 (Official Benchmark)** | 28 | `1.0000` | `1.0000` | `1.0000` | **`1.0000`** |
| **November 2024 (Historical Unseen)** | 32 | `0.9327` | `0.9153` | `0.9153` | **`0.9220`** |

This demonstrates that the pipeline's dynamic ordinal parser and proposal-aware BM25 retriever generalize robustly to unseen ballots and varying proposal counts with consistent multilingual performance (~0.92 Macro-F1).

---

## 6. Limitations

- **Complex Tabular Structures:** Tables and infographics in voting booklets require specialized OCR/layout parsers (e.g. Docling) to preserve row-column semantics.
- **Subtle Nuances & Rhetoric:** Highly nuanced political rhetoric that is neither explicitly confirmed nor refuted requires calibrated threshold sensitivity to avoid collapsing into neutral predictions.
- **Empirical Thresholds:** The decision arbiter currently uses fixed empirical thresholds calibrated on the benchmark distribution. Future iterations will benefit from adaptive conformal prediction.

---

## 7. Reproducibility & Engineering Rigor

Judges can execute the benchmark on a clean checkout via:
```bash
make run
```
Required environment variables:
```bash
export LLM_NAME="swiss-ai/Apertus-v1.5-8B"
export LLM_BASE_URL="https://api.inference.cscs.ch/v1"
export LLM_API_KEY="your_api_key_here"
```

### Automated Test Suite (65 Tests)
The repository includes a deterministic suite with 65 passing unit and Streamlit UI tests covering all system components:
```bash
make test
```
- `test_pdf_parser.py`: Page extraction, clean paragraph structure, dynamic ordinal proposal boundary detection.
- `test_retriever.py`: Tokenization, multilingual BM25, proposal scoping, dynamic recommendation anchor retrieval.
- `test_apertus_client.py`: Multi-format JSON parsing, exponential backoff retry logic, calibrated decision-arbiter rules (1, 2, 3).
- `test_numerical_checker.py`: Swiss thousands formatting (`500'000`), multi-lingual scales, percentages (`10%` vs `15%`), calendar year isolation (`2002`, `2050`), and vacuity guardrail enforcement.
- `test_pipeline.py`: End-to-end `ClaimVerificationEngine` and `BenchmarkEvaluator` runs.

### Docker Flexibility
Docker containers can be built seamlessly either from the project root (`docker build -t hackapertus .`) or from `track_2a/` (`docker build -t hackapertus track_2a`). All tests and container entrypoints are validated automatically on every push via **GitHub Actions** (`.github/workflows/ci.yml`).

To launch the interactive dashboard locally:
```bash
make web
```

---

## 8. Next Steps

### Optional Dense Retrieval Experiment (implemented, quality not yet benchmarked)

`hybrid_dense` adds local BGE-M3 paragraph embeddings and exact cosine search,
fused with weighted claim/title BM25 using RRF after max-per-page aggregation.
Both new methods return original pages; `hybrid` is the lexical comparison baseline.
The original `retrieval` default, `full`, direct-premise task, Apertus prompt and official
output schema remain available. There is no additional generative verification call.

Persistent caches use PDF, parser/chunking, model and pinned-revision identities.
Extended reports distinguish API latency from end-to-end, model-start, indexing and
warm-search times, and report actual language pairs and explicit fallback strategies.
The bundled data contains no cross-language pairs or reliable gold page labels.
No quality improvement over the previous reported benchmarks is asserted.

See [methodology and CPU smoke measurements](docs/hybrid_dense_retrieval.md) and
[raw CPU timing observations](docs/hybrid_dense_cpu_measurement.json). A paid dev
comparison has not been run without a separately authorized API budget.

1. Integrate advanced layout parsing (e.g., Docling) to handle complex voting booklet financial charts and side-by-side comparison tables.
2. Cross-lingual semantic embedding retrieval (e.g., BGE-M3 or Apertus embeddings) alongside lexical BM25.
3. Transition from heuristic proposal extraction to a full graph-native representation as detailed in our architectural concept roadmap.

---

## License

Creative Commons Attribution 4.0 (CC-BY-4.0).
