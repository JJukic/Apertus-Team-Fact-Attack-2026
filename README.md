# 🇨🇭 Fact Attack 2026 — Hack Apertus Track 2A (OST)
## Multilingual Natural Language Inference over Swiss Official Voting Booklets

[![CI](https://github.com/JJukic/Apertus-Team-Fact-Attack-2026/actions/workflows/ci.yml/badge.svg)](https://github.com/JJukic/Apertus-Team-Fact-Attack-2026/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Model](https://img.shields.io/badge/Model-Apertus%20v1.5--70B-orange.svg)](https://huggingface.co/swiss-ai)
[![Dataset](https://img.shields.io/badge/HuggingFace-OSTswiss%2FMNLIoverSwissVotingBooklets-yellow.svg)](https://huggingface.co/datasets/OSTswiss/MNLIoverSwissVotingBooklets)
[![License](https://img.shields.io/badge/License-MIT%20%2F%20CC--BY--4.0-green.svg)](LICENSE)

An **Apertus-powered, document-grounded claim-verification system** that decides whether an official Swiss
voting booklet (*Abstimmungsbüchlein*) **entails (0)**, is **neutral to (1)** or **contradicts (2)** a political
claim, and cites the booklet pages that justify the decision. Claims and booklets can be in German, French or
Italian, in any combination.

- **Challenge:** [Hack Apertus Track 2A (OST)](https://hackapertus.ch/)
- **Team:** Josip Jukic, Felipe Wüthrich
- **Technical Report:** [track_2a/technical_report.md](track_2a/technical_report.md) · **Experiment log:** [track_2a/docs/experiments.md](track_2a/docs/experiments.md)

> Predictions describe the relationship between a claim and the official booklet. They are not political advice.

---

## 📊 Results

Evaluated on the official OST dataset (1,495 human-annotated pairs, 60 booklets, ~66 % cross-lingual). The
**test split holds 5 voting dates that were never used during development** (402 pairs), as a stand-in for the
held-out benchmark. Model: `swiss-ai/Apertus-v1.5-70B-thinking` on CSCS.

| Task | Macro-F1 | Cross-lingual F1 | Ø input tokens | Ø output tokens | Latency mean / p95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Advanced** — booklet PDF + claim + vote | **0.928** | 0.913 | 4,696 | 59 | 1.9 s / 3.3 s |
| **Beginner** — reference text + claim | **0.975** | 0.971 | 2,523 | 55 | 1.1 s / 1.7 s |

### Full booklet vs. selected context (advanced task, same 150 test pairs)

| Context supplied to Apertus | Macro-F1 | Ø input tokens | Latency p95 |
| :--- | :---: | :---: | :---: |
| Full booklet (baseline) | 0.730 | 59,517 | 43.7 s* |
| Our first pipeline (proposal filter + BM25, verbose JSON) | 0.795 | 3,408 | 36.2 s* |
| **Hybrid retrieval + `ids` prompt (current)** | **0.926** | 6,023 | 3.8 s |

\* measured while several runs shared the endpoint. **Selecting ~10 pages beats sending the whole booklet by
0.19 F1 with ~10× fewer tokens**: with up to 70k tokens of context the relevant passage gets lost.
Every number comes from a saved run in [`track_2a/results/`](track_2a/results/).

### What we learned about Apertus

- **More context hurts:** the whole booklet (~60k tokens) scores 0.73, ten selected pages 0.93.
- **Over-confident:** the label probability is ~1.0 for wrong answers too, so confidence thresholds cannot catch errors.
- **Sometimes self-contradictory:** in ~2 % of answers the label and the stated confidences disagree; the label is right
  3× as often, so we keep it (+0.008 F1 on test).
- **Speaker attribution is the main weakness:** claims attributed to the initiative committee fail 16–18 % of the time
  (2–8 % otherwise), even with every booklet page labelled by who is speaking.
- **Thinking pays in quality, not in efficiency:** reasoning before the answer adds ~3 F1 points but costs ~20× the output
  tokens and ~10× the latency, so it is off by default (`THINKING=true` to reproduce).
- **8B vs. 70B:** 8B is ~3.5× faster (p95 1.2 s) but 6 F1 points weaker (0.851 vs. 0.908 on 450 dev pairs).

Details and every rejected idea: [experiment log](track_2a/docs/experiments.md).

---

## ⚙️ How it works

```
 Booklet PDF (DE/FR/IT)                       Claim (DE/FR/IT)  +  vote title
        │                                              │
        ▼                                              │
 pypdf → pages, each labelled with its section         │
 ("Arguments of the committee", "Arguments of the      │
  Federal Council", "Voting text", …)                  │
        │                                              │
        └──────────────► Hybrid retrieval ◄────────────┘
                 BM25(claim) + 2 × BM25(vote title) over all pages
                 → top 10 pages, numbered [P1] … [P10]
                                  │
                                  ▼
                 Apertus v1.5 (CSCS) — NLI prompt
                 → {"p_entail", "p_neutral", "p_contra", "label", "evidence_ids": [3, 7]}
                                  │
                                  ▼
          label 0/1/2 + verbatim evidence pages + tokens + inference time
```

1. **Hybrid retrieval that works across languages.** The vote title is written in the booklet's language, so it
   anchors the right proposal even when the claim is in another language. Unlike a hard proposal filter it keeps
   the overview pages at the front of the booklet. Gold-section recall rose from 0.74 to 0.93.
2. **Evidence as passage ids.** Apertus cites the numbers of the pages it relies on instead of writing quotes,
   so evidence is always verbatim booklet text with a correct page number, and answers need ~55 output tokens
   instead of ~150.
3. **Booklet-aware prompt.** Booklets deliberately contain opposing viewpoints: a claim such as "the committee
   argues X" is checked against the committee's text only, and section labels tell the model who is speaking.
4. **Measured, not assumed.** Every design choice was tested on a dev split; ideas that did not help (a hard
   numerical override, label-only answers, small passages, full-booklet prompting) were dropped. See the
   [experiment log](track_2a/docs/experiments.md).

---

## 🚀 Quick start

```bash
export LLM_API_KEY="your_api_key_here"
make run          # builds the Docker image and runs the benchmark
make test         # unit tests (no API calls)
make web          # Streamlit app at http://localhost:8501
```

Defaults (override via environment variables or `.env`):

```bash
LLM_NAME=swiss-ai/Apertus-v1.5-70B-thinking
LLM_BASE_URL=https://api.inference.cscs.ch/v1
NLI_STRATEGY=hybrid      # 'hybrid' | 'retrieval' | 'full'
NLI_TOP_K=10
PROMPT_MODE=ids          # 'ids' | 'json' | 'compact'
```

### CLI

```bash
cd track_2a

# Advanced task: booklet + claim (+ vote title)
python -m src predict --booklet data/booklets/2026-06-14_fr.pdf \
  --claim "Der Bundesrat empfiehlt, die Initiative abzulehnen." \
  --vote "Initiative populaire « Pas de Suisse à 10 millions ! (initiative pour la durabilité) »" --json

# Beginner task: reference text + claim
python -m src predict --reference "…" --claim "…" --json

# Batch file (JSON/JSONL) in the official OST format -> one prediction per case
python -m src run --input cases.jsonl --output predictions.jsonl

# Reproduce the evaluation
python -m src.hf_dataset                                   # dataset, 60 booklets, dev/test split
python -m src benchmark -d data/hf/test.jsonl -t advanced  # saves a report to results/
python -m src benchmark -d data/hf/test.jsonl -t beginner
python -m src.results_summary                              # results/summary.json for the app
```

### Official input / output format

```jsonc
// advanced task                                   // beginner task
{"id": "case-0042",                                {"id": "case-0043",
 "booklet": {"path": "booklets/2024_11_24_de.pdf"},  "reference": {"text": "Der Bundesrat ... lehnen die Volksinitiative ab."},
 "vote": "Étape d'aménagement 2023 des routes nationales",
 "claim": {"text": "La proposition entraînera une augmentation de la TVA."}}
                                                    "claim": {"text": "Le Conseil fédéral recommande d'accepter l'initiative."}}
```

Each case yields `{"id", "label", "label_name", "evidence": [{"page", "text"}], "metrics": {"input_tokens",
"output_tokens", "inference_time_ms"}}`; evidence is empty for neutral. The claim language is detected
automatically, booklet paths are resolved robustly, and a failing case is reported on stderr and returned as a
valid neutral record instead of aborting the batch. The Docker build downloads and pre-parses all 60 known
booklets (disk cache keyed by file hash), so PDF parsing does not slow down the first claim per booklet.

---

## 📁 Repository structure

```
track_2a/
├── app.py                    # Streamlit app (claim check, benchmark dashboard from results/)
├── technical_report.md       # Architecture, evaluation, limitations
├── docs/experiments.md       # Every experiment with its numbers
├── results/                  # Saved benchmark runs (+ summary.json)
├── data/                     # Demo dataset, 2026-06-14 booklets; hf/ and other booklets are downloaded
├── tests/                    # Unit tests
└── src/
    ├── hf_dataset.py         # OST dataset + booklet download, dev/test split by voting date
    ├── pdf_parser.py         # Pages, section labels, proposal boundaries
    ├── retriever.py          # Hybrid (booklet-wide) and proposal-filtered BM25 retrieval
    ├── apertus_client.py     # CSCS client, prompts (ids / json / compact), retries
    ├── inference.py          # ClaimVerificationEngine: retrieval → Apertus → evidence pages
    ├── numerical_checker.py  # Swiss number normalisation; reports numerical conflicts
    ├── evaluator.py          # Macro-F1, language pairs, evidence grounding, p95 latency
    ├── results_summary.py    # Collects saved runs for the app and README
    └── cli.py                # predict / run / benchmark / download / web
```
