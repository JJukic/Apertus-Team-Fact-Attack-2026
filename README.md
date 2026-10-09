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
- **Team:** Josip Jukic, Felipe Wüthrich
- **Technical Report:** [track_2a/technical_report.md](track_2a/technical_report.md) ([PDF](track_2a/docs/FactAttack_Technical_Report.pdf)) · **Experiment log:** [track_2a/docs/experiments.md](track_2a/docs/experiments.md)

> Predictions describe the relationship between a claim and the official booklet. They are not political advice.

---

## 📊 Results

The historical evaluation used 1,495 OST pairs and 60 booklets. Its 402-pair split
was initially held out by voting date, then observed during subsequent experiments;
it is no longer an untouched holdout. Historical results below are retained.
Model: `swiss-ai/Apertus-v1.5-70B-thinking` on CSCS.

The closed `experiment/competition-score-optimization` comparison uses the pinned
v1.1 dataset: **1,488 cases per task**, with five folds grouped by 20 voting dates.

| Configuration | Advanced Macro-F1 | Beginner Macro-F1 | Advanced Hit@5 |
|---|---:|---:|---:|
| B0 baseline | 0.947699 | 0.981830 | 0.467677 |
| C1: grounded evidence and caching | 0.944034 | 0.982497 | 0.755556 |
| R1: translated BM25 query | 0.950844 | 0.981830 | 0.765657 |
| **R2: original/translated BM25 union** | **0.956237** | **0.982497** | **0.773737** |

**Recommend R2 for the evaluated submission contract.** It retains Top-12 BM25,
speaker boost and the original NLI claim/prompt. Reported Advanced tokens rise
1.39%; failed requests include unknown usage, and official separate-task processing
efficiency is unmeasured. These are public validation results, not private competition
scores. B0 includes one invalid output counted as wrong; R2 recovers 23 technical
failures and retains every attempt in its measurements. Baseline defaults remain available.
All 115 local and 115 CPU-container tests pass. See the
[final report](track_2a/docs/competition_optimization_report.md),
[comparison](track_2a/results/competition_optimization/comparison.csv) and
[recommended configuration](track_2a/results/competition_optimization/best_config.json).

Historical results:

| Task | Macro-F1 | Cross-lingual F1 | Ø input tokens | Ø output tokens | Latency mean / p95 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Advanced** — booklet PDF + claim + vote | **0.940** | 0.934 | 5,576 | 58 | 2.1 s / 2.9 s |
| **Beginner** — reference text + claim | **0.975** | 0.971 | 2,523 | 55 | 1.1 s / 1.7 s |

**Evidence:** every Entailment / Contradiction prediction cites at least one passage, verbatim with its page
number. In the advanced task 78 % of the cited pages lie inside the human-annotated reference section (beginner: 100 %);
Neutral predictions cite nothing, as the output format requires.

### Full booklet vs. selected context (advanced task, same 150 test pairs)

| Context supplied to Apertus | Macro-F1 | Ø input tokens | Latency p95 |
| :--- | :---: | :---: | :---: |
| Full booklet (baseline) | 0.730 | 59,517 | 43.7 s* |
| Our first pipeline (proposal filter + BM25, verbose JSON) | 0.795 | 3,408 | 36.2 s* |
| Hybrid retrieval, 10 pages + `ids` prompt | 0.926 | 6,023 | 3.8 s |
| **Final: 12 pages (long ones clipped), speaker boost, decision rules** | **0.946** | 5,713 | 2.8 s |

\* measured while several runs shared the endpoint. **Selecting a dozen pages beats sending the whole booklet by
0.22 F1 with ~10× fewer tokens**: with up to 70k tokens of context the relevant passage gets lost.
Every number comes from a saved run in [`track_2a/results/`](track_2a/results/).

### What we learned about Apertus

- **More context hurts:** the whole booklet (~60k tokens) scores 0.73, ten selected pages 0.93.
- **Over-confident:** the label probability is ~1.0 for wrong answers too, so confidence thresholds cannot catch errors.
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
                 → top 12 pages (long ones clipped to 3,000 chars), numbered [P1] … [P12]
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
4. **Speaker boost and decision rules.** If the claim names a side ("the committee argues …"), that side's two best
   argument pages are always in the context; this cut the error rate on committee claims from 13 % to 7 %. A Neutral label next to a
   confident other relation follows the confidences, otherwise Apertus' own label is kept.
5. **Measured, not assumed.** Every design choice was tested on a dev split; ideas that did not help (a hard
   numerical override, label-only answers, small passages, full-booklet prompting) were dropped. See the
   [experiment log](track_2a/docs/experiments.md).

---

## 🚀 Quick start

```bash
docker build --platform linux/amd64 -t fact-attack:test .
# Set BASE_URL and API_KEY securely in your terminal environment first.
# ./cases contains source-only cases.jsonl and its referenced PDFs.
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

Download evaluation datasets/booklets outside prediction using the preparation
commands in the final report. The image includes demo PDFs; evaluation inputs and
gold files are not downloaded during build or prediction. Remote responses may
vary even at temperature zero. Gold labels belong only in the host-side evaluator.

Defaults (override via environment variables or `.env`):

```bash
LLM_NAME=swiss-ai/Apertus-v1.5-70B-thinking
LLM_BASE_URL=https://api.inference.cscs.ch/v1
NLI_STRATEGY=hybrid      # 'hybrid' | 'retrieval' | 'full'
NLI_TOP_K=12
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
"output_tokens", "inference_time_ms"}}`; evidence is empty for neutral. The claim language is detected
automatically, booklet paths are resolved robustly, and a failing case is reported on stderr and returned as a
valid neutral record so IDs remain complete. A batch with an operational failure
exits nonzero and writes diagnostics; neutral fallback is not a successful classification.
Source-derived caches are created in `/tmp`; the input mount remains unchanged.

---

## 📁 Repository structure

```
track_2a/
├── app.py                    # Streamlit app (claim check, benchmark dashboard from results/)
├── technical_report.md       # Architecture, evaluation, limitations
├── docs/experiments.md       # Every experiment with its numbers
├── docs/FactAttack_Technical_Report.pdf  # Submission PDF of the report (docs/build_report_pdf.py)
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

---

## 📜 License

Code: [Apache License 2.0](LICENSE). Documentation (READMEs, technical report, experiment log): CC-BY-4.0.
