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

Evaluated on the official OST dataset (v1.1, 60 booklets, ~66 % cross-lingual) with the organisers' `evaluate.py`.
The **test split holds 5 voting dates that were never used during development** (401 pairs), as a stand-in for the
held-out benchmark. Model: `swiss-ai/Apertus-v1.5-70B-thinking` on CSCS.

The closed `experiment/competition-score-optimization` comparison uses the pinned
v1.1 dataset: **1,488 cases per task**, with five folds grouped by 20 voting dates.

| Configuration | Advanced Macro-F1 | Beginner Macro-F1 | Advanced Hit@5 |
|---|---:|---:|---:|
| B0 baseline | 0.947699 | 0.981830 | 0.467677 |
| C1: grounded evidence and caching | 0.944034 | 0.982497 | 0.755556 |
| R1: translated BM25 query | 0.950844 | 0.981830 | 0.765657 |
| **R2: original/translated BM25 union** | **0.956237** | **0.982497** | **0.773737** |

Josip's comparison on all cases (k=12, before the merge with the official contract): It retains Top-12 BM25,
speaker boost and the original NLI claim/prompt. Reported Advanced tokens rise
1.39%; failed requests include unknown usage, and official separate-task processing
efficiency is unmeasured. These are public validation results, not private competition
scores. B0 includes one invalid output counted as wrong; R2 recovers 23 technical
failures and retains every attempt in its measurements. Baseline defaults remain available.
All 115 local and 115 CPU-container tests pass. See the
[final report](track_2a/docs/competition_optimization_report.md),
[comparison](track_2a/results/competition_optimization/comparison.csv) and
[recommended configuration](track_2a/results/competition_optimization/best_config.json).


| Task | Macro-F1 | Evidence Hit@5 | Ø input tokens | Ø output tokens | Non-LLM time / case |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A · Advanced** — booklet PDF + claim + vote | **0.930** | **0.727** | 3,877 | 56 | ~5 ms |
| **B · Beginner** — reference text + claim | **0.973** | – | 2,532 | 56 | ~3 ms |

The official score weights macro-F1, evidence, processing time (wall clock minus time waiting for Apertus) and tokens
(both relative to the best team). After the organisers' Q&A (2026-10-08) we traded a little F1 for efficiency and
evidence: 8 instead of 12 pages cut the input tokens by 31 % (test F1 0.945 → 0.930, dev 0.937 → 0.934), and
evidence made of page pieces raised Hit@5 from 0.34 to 0.73, see the [experiment log](track_2a/docs/experiments.md#official-evaluation-contract-and-scoring-ost-qa-2026-10-08).

**Evidence:** every Entailment / Contradiction prediction returns five verbatim excerpts with their 1-based page: the
two cited pages in halves, then the best-matching third of further pages (other cited pages, then retrieved ones). An
item counts if it lies inside the gold passage or contains it; whole pages usually do neither. Neutral predictions
cite nothing.

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
                 → top 8 pages (long ones clipped to 3,000 chars), numbered [P1] … [P8]
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

### Development

```bash
export API_KEY="your_api_key_here"   # LLM_API_KEY works too
make run          # Docker: benchmark on the 402-pair test split, ~4 min (report + JSON in track_2a/results/)
make test         # unit tests (no API calls)
make web          # Streamlit app at http://localhost:8501
```

`make run` runs our own benchmark on the test split (macro-F1 ≈ 0.93 with 8 pages; temperature 0, so runs differ by
at most a few pairs). `NLI_DATASET=data/demo_dataset.jsonl make run` runs the 28-pair demo set instead (~40 s, ≈ 0.91).

Defaults (override via environment variables or `.env`):

```bash
LLM_NAME=swiss-ai/Apertus-v1.5-70B-thinking
BASE_URL=https://api.inference.cscs.ch/v1   # LLM_BASE_URL works too
NLI_STRATEGY=hybrid      # 'hybrid' | 'retrieval' | 'full'
NLI_TOP_K=8
PROMPT_MODE=ids          # 'ids' | 'json' | 'compact'
EVIDENCE_SPLIT=2,2       # first two cited pages in halves as evidence
EVIDENCE_FILL=true       # free evidence slots: best third of further pages
NLI_WORKERS=4            # cases processed in parallel by the batch CLI
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

Third-party: the submission image includes [PyMuPDF](https://github.com/pymupdf/PyMuPDF) (AGPL-3.0), used to quote
evidence pages and text blocks from the booklets. Apache-2.0 code may be combined with AGPL-3.0 code; the image as a
whole is therefore distributed under the terms of the AGPL-3.0, and its complete source is this public repository.
