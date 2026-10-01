# Technical Report — Fact Attack: Multilingual NLI over Swiss Voting Booklets

A deeper write-up than the README: what you built, how it works, and what the numbers say.

- **Track:** Track 2A — OST: Multilingual Natural Language Inference over Swiss Official Voting Booklets
- **Event:** Hack Apertus Online (October 1–16, 2026)
- **Team:** Team Fact Attack 2026 — `[member1, member2, ...]`
- **Demo:** `[link to video / notebook / deployment]`

---

## 1. Summary

In Switzerland's direct democracy, citizens regularly vote on complex legal, financial, and social proposals. During campaigns, voters encounter political claims that are difficult and time-consuming to check against official government materials (*Abstimmungsbüchlein*). 

This project implements a multilingual document-grounded claim-verification system powered by **Apertus v1.5**. Given an official voting booklet (in German, French, or Italian) and a natural-language claim, our system determines whether the claim is **Entailed (`0`)**, **Neutral (`1`)**, or **Contradicted (`2`)** by the document, while providing transparent supporting evidence passages. We investigate and benchmark the trade-off between full-document prompting and token-efficient passage retrieval (BM25/chunking).

---

## 2. Architecture

```
[ Voting Booklet PDF (DE / FR / IT) ]
                 │
                 ▼
       [ PyPDF Parser ] ─── extracts pages & clean paragraphs
                 │
       ┌─────────┴─────────────────────────────────┐
       ▼                                           ▼
[ Strategy A: Full Document ]         [ Strategy B: Passage Retriever ]
(Full text ~8k - 15k tokens)          (BM25 ranking -> Top-k passages)
       │                                           │
       └─────────────────┬─────────────────────────┘
                         ▼
             [ Apertus Prompt Builder ]
        (Document Context + Claim + NLI Rules)
                         │
                         ▼
        [ Apertus-8B / 70B via CSCS API ]
                         │
                         ▼
       [ Structured JSON Response Parser ]
   ├── Label: 0 (Entailment) | 1 (Neutral) | 2 (Contradiction)
   ├── Supporting Evidence Quotes
   ├── Reasoning Summary
   └── Token & Latency Metrics
```

---

## 3. Use of Apertus

- **Model:** `swiss-ai/Apertus-8B-Instruct` (and `swiss-ai/Apertus-70B-Instruct`)
- **How it is used:** Document-grounded Natural Language Inference (NLI) & evidence extraction.
- **Where it runs:** CSCS API endpoint (`https://api.cscs.ch/v1`) using OpenAI-compatible chat completion endpoints.
- **Prompt Strategy:** Structured multilingual prompt enforcing direct grounded inference without external knowledge or hallucination, outputting strictly validated JSON with label, reasoning, and verbatim evidence quotes.

---

## 4. Data

- **Benchmark Dataset:** `OSTswiss/MNLIoverSwissVotingBooklets` (Hugging Face)
  - Multilingual claims with human-annotated NLI labels (0, 1, 2) and reference passages in German, French, and Italian.
- **Corpus:** Official Swiss Federal Voting Booklets (*Erläuterungen des Bundesrates*) downloaded directly from `bk.admin.ch`.
- **License:** Open data (admin.ch / MIT).

---

## 5. Evaluation

We evaluate using **Macro-F1** across Entailment (0), Neutral (1), and Contradiction (2) on the official held-out benchmark (28 human-annotated test samples across German, French, and Italian). Furthermore, we measure **input/context token efficiency** and **inference latency**:

| Strategy / Setup | Macro-F1 | Precision (0 / 1 / 2) | Recall (0 / 1 / 2) | Avg Input Tokens | Avg Total Tokens | Avg Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Standard Retrieval (Baseline)** | `0.7846` | 0.80 / 0.80 / 1.00 | 0.94 / 1.00 / 0.43 | `3,540.3` | `3,686.3` | `992.4` |
| **Hybrid + Fuzzy Logic (Ours)** | **`0.9177`** | **0.94 / 0.80 / 1.00** | **0.94 / 1.00 / 0.86** | `3,681.3` | `3,874.4` | `1,278.3` |

### Language Breakdown (Hybrid + Fuzzy Logic)

| Language | Number of Samples | Macro-F1 | Performance Notes |
| :--- | :---: | :---: | :--- |
| **French (FR)** | 8 | **`1.0000`** | Flawless 100% classification & verbatim evidence |
| **Italian (IT)** | 9 | **`0.9030`** | Robust cross-lingual and monolingual grounding |
| **German (DE)** | 11 | **`0.8632`** | Drastic jump from 0.6556 due to numerical conflict detection |
| **Overall** | **28** | **`0.9177`** | Evaluated live on CSCS Alps (`api.inference.cscs.ch`) |

---

## 6. Limitations

- Complex tabular structures or graphical infographics in booklets require advanced layout parsing.
- Subtle political framing or ambiguous claims require careful prompt calibration to avoid bias toward neutral predictions.

---

## 7. Reproducibility

Judges can execute the benchmark on a clean checkout via:
```bash
make run
```
Environment variables:
```bash
export LLM_NAME="swiss-ai/Apertus-8B-Instruct"
export LLM_BASE_URL="https://api.cscs.ch/v1"
export LLM_API_KEY="your_cscs_api_key"
```

---

## 8. Next Steps

1. Integrate advanced layout parsing (e.g., Docling) to handle voting booklet charts and multi-column comparison tables.
2. Cross-lingual semantic embedding retrieval (e.g., BGE-M3 or Apertus embeddings) alongside lexical BM25.
3. Develop an interactive voting assistant web UI for Swiss citizens.

---

## License

Creative Commons Attribution 4.0 (CC-BY-4.0).
