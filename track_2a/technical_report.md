# Technical Report — Fact Attack: Multilingual NLI over Swiss Voting Booklets

- **Track:** Track 2A — OST: Multilingual Natural Language Inference over Swiss Official Voting Booklets
- **Event:** Hack Apertus Online (October 1–16, 2026)
- **Team:** Team Fact Attack 2026 — Josip Jukic, Felipe Wüthrich
- **Repository:** [GitHub](https://github.com/JJukic/Apertus-Team-Fact-Attack-2026) · **Experiment log:** [docs/experiments.md](docs/experiments.md)

---

## 1. Summary

Swiss voters receive an official booklet for every federal vote, and campaigns make claims about it that are
hard to check. We built a multilingual claim-verification system on **Apertus v1.5**: given a booklet (German,
French or Italian) and a claim (any of the three languages), it classifies the claim as **Entailment (0)**,
**Neutral (1)** or **Contradiction (2)** and returns the booklet pages that justify the decision, together with
token usage and inference time.

On a test split of **5 voting dates never used during development** (402 pairs from the official dataset), the
system reaches **macro-F1 0.920** on the advanced task (booklet PDF) and **0.975** on the beginner task (reference
text), at ~5,700 / ~2,500 input tokens, ~57 output tokens and a p95 inference time of 4.0 s / 1.7 s.

The central finding for the challenge's research question: **sending ~10 well-chosen pages beats sending the
whole booklet** — 0.926 vs. 0.730 macro-F1 with ~10× fewer input tokens.

---

## 2. Architecture

```
 Booklet PDF (DE/FR/IT)                       Claim (DE/FR/IT)  +  vote title
        │                                              │
        ▼                                              │
 pypdf → pages; each page labelled with its section    │
 (committee arguments / Federal Council arguments /    │
  voting text / explanation / overview)                │
        │                                              │
        └──────────────► Hybrid retrieval ◄────────────┘
                 score = BM25(claim) + 2 × BM25(vote title), over all pages
                 → top 10 pages, numbered [P1] … [P10]
                                  │
                                  ▼
                 Apertus v1.5 on CSCS — NLI prompt ('ids' mode)
                 → {"p_entail", "p_neutral", "p_contra", "label", "evidence_ids"}
                                  │
                                  ▼
       label + verbatim evidence pages (page number) + tokens + inference time
```

### 2.1 Document processing
`pdf_parser.py` extracts the text of every page with pypdf and detects section headings in all three languages
("Argumente Initiativkomitee", "Arguments du Conseil fédéral et du Parlement", "Il testo in votazione", …).
A heading applies to its page and the following pages until the next heading, and table-of-contents entries
(heading followed by a page number) are ignored. Section labels matter because booklets deliberately contain
opposing viewpoints: the model has to know whether a page states the committee's or the Federal Council's position.

### 2.2 Context selection: hybrid retrieval
`retriever.py` scores every page of the booklet with normalised BM25 against the claim plus twice the
normalised BM25 against the vote title, and keeps the top 10 pages.

- The **vote title is written in the booklet's language**, so it anchors the right proposal even when the claim
  is in another language; lexical matching of the claim alone fails for cross-lingual pairs.
- Our first version filtered hard to one detected proposal. That dropped the per-proposal overview pages at the
  front of the booklet, where the gold passage sits for ~11 % of pairs, and chose the wrong proposal for another
  ~10 %. Booklet-wide scoring raised gold-section recall on the dev split from 0.74 to 0.93.

### 2.3 Apertus as the NLI model
The selected pages are numbered `[P1]…[P10]`. The system prompt defines the three labels, tells the model to
judge only against the passages (not world knowledge), to answer Neutral for topics or details the passages do
not address, to check claims attributed to an actor only against that actor's text, and to read "recommends
rejecting X" as supported when the Federal Council and Parliament recommend "No". Apertus answers with a small
JSON object: three confidences, the label, and the ids of the pages that justify it.

- **Evidence by id** guarantees verbatim booklet text with the correct page number and avoids paraphrased quotes.
- **Short answers**: ~55 output tokens instead of ~150 for a JSON answer with reasoning and quotes, which more
  than halves inference time (p95 5.5 s → 1.8 s on the dev split) at equal or better F1.
- For **Neutral**, the evidence list is empty, as required by the output format. If the model predicts 0 or 2
  without citing a page, the top-ranked page is returned as evidence.

### 2.4 Beginner task
The reference string (up to 25k characters) is split into passages of ≤ 700 characters at sentence boundaries,
numbered and sent with the same prompt, so evidence is a specific passage rather than the whole reference.

---

## 3. Use of Apertus

- **Model:** `swiss-ai/Apertus-v1.5-70B-thinking` (the model our CSCS key is authorised for; the thinking phase is
  not triggered by our prompt, answers start directly with the JSON object).
- **Where it runs:** CSCS inference endpoint (`https://api.inference.cscs.ch/v1`), OpenAI-compatible chat
  completions, temperature 0, retries with exponential backoff.
- **Role:** Apertus is the only model in the pipeline. It performs the NLI classification and selects the
  evidence pages; retrieval is lexical (BM25) and runs locally.

---

## 4. Data and evaluation protocol

- **Dataset:** `OSTswiss/MNLIoverSwissVotingBooklets` — 1,495 human-annotated claim/reference pairs from
  60 booklets (2020–2026), balanced labels, ~66 % cross-lingual pairs.
- **Booklets:** downloaded from `bk.admin.ch` via the `booklet_url` field (`python -m src.hf_dataset`).
- **Split by voting date**, so all three language versions of a booklet stay on the same side:
  - **dev** (1,093 pairs) for all design decisions;
  - **test** (402 pairs, 5 voting dates: 2021-06-13, 2021-09-26, 2022-09-25, 2024-06-09, 2024-09-22), used only
    for before/after measurements. The two booklets our first version was tuned on are pinned to dev.
- **Metrics** (mirroring the judging criteria): macro-F1 over the three labels (primary), macro-F1 per
  claim→booklet language pair, share of cited evidence inside the gold reference section, input/output tokens,
  mean and p95 inference time. Every run is saved with its configuration and git commit in `results/`.

---

## 5. Results

### 5.1 Final configuration — full test split (402 pairs)

| Task | Macro-F1 | Mono-lingual | Cross-lingual | Input tokens | Output tokens | Latency mean / p95 |
|---|---:|---:|---:|---:|---:|---:|
| Advanced | **0.920** | 0.951 | 0.906 | 5,737 | 59 | 2.1 s / 4.0 s |
| Beginner | **0.975** | 0.984 | 0.971 | 2,523 | 55 | 1.1 s / 1.7 s |

Advanced task by language pair (claim → booklet):

| | → de | → fr | → it |
|---|---:|---:|---:|
| **de** | 0.951 | 0.933 | 0.922 |
| **fr** | 0.914 | 0.956 | 1.000 |
| **it** | 0.788 | 0.874 | 0.934 |

### 5.2 Full document vs. selected context (advanced, same 150 test pairs)

| Context | Prompt | Macro-F1 | Input tokens | Output tokens | Latency p95 |
|---|---|---:|---:|---:|---:|
| Full booklet | JSON | 0.730 | 59,517 | 202 | 43.7 s* |
| Proposal filter, top 5 pages | JSON | 0.795 | 3,408 | 157 | 36.2 s* |
| Proposal filter, top 10 pages | ids | 0.880 | 6,502 | 54 | 4.1 s |
| **Hybrid, top 10 pages** | **ids** | **0.926** | 6,023 | 55 | 3.8 s |

\* measured while three runs shared the endpoint.

The full booklet is both the most expensive and the least accurate option: booklets reach 250k characters
(~70k tokens), and the relevant passage gets lost in the middle. Selecting context is therefore not only an
efficiency measure but also improves NLI quality.

### 5.3 Ablations on the dev split

| Change | Effect | Decision |
|---|---|---|
| Hard numerical override (force Contradiction on number mismatch) | fired 5×, wrong 5× (unmentioned years are Neutral) | off by default |
| Label-only answer with logprob confidences (8 output tokens) | beginner F1 0.724 (Neutral recall 0.34) | rejected |
| `ids` answer instead of JSON with reasoning + quotes | beginner 0.973 → 0.980, advanced 0.786 → 0.797, p95 5.5 s → 1.8 s | adopted |
| Hybrid retrieval instead of proposal filter | advanced 0.797 → 0.862 | adopted |
| Viewpoint and recommendation rules in the prompt | advanced 0.862 → 0.870, beginner 0.980 → 0.993 | adopted |
| 600-character passages instead of pages (n = 450, paired) | 0.831 vs. 0.908; better evidence precision, fewer tokens | rejected |
| One-sentence reasoning before the confidences | no gain on a full run, +40 output tokens | rejected |

Full tables: [docs/experiments.md](docs/experiments.md). Runs on 150 pairs carry about ±3 F1 points of sampling
noise; close decisions were re-run on 450 pairs.

---

## 6. Limitations

- **Italian claims against German booklets** are the weakest pair (0.79 F1). Lexical retrieval relies on the vote
  title for cross-lingual pairs; multilingual dense retrieval would likely help.
- **Entailment → Contradiction** accounts for 16 of the 32 remaining advanced errors on the test split, often
  where the booklet contains both sides of an argument.
- **Evidence precision:** with whole pages as passages, ~75 % of cited pages lie inside the gold reference section.
  Smaller passages raise this to ~90 % but cost F1 (5.3).
- **Layout:** section headings are detected from text; older booklets set some headings as graphics. Tables and
  charts are read as plain text.
- **Model coverage:** only `Apertus-v1.5-70B-thinking` was accessible with our key, so the faster 8B model could
  not be evaluated. Latency was measured client-side on a shared endpoint.
- **Not political advice:** outputs describe the relationship between a claim and the official booklet only.

---

## 7. Reproducibility

```bash
export LLM_API_KEY="your_api_key_here"
make run                    # Docker build + benchmark on the demo set
make test                   # 41 unit tests, no API calls

cd track_2a
python -m src.hf_dataset                                   # dataset, booklets, dev/test split
python -m src benchmark -d data/hf/test.jsonl -t advanced  # writes results/<timestamp>_....json
python -m src benchmark -d data/hf/test.jsonl -t beginner
python -m src.results_summary                              # results/summary.json (used by the app)
```

Configuration is read from environment variables: `LLM_NAME`, `LLM_BASE_URL`, `LLM_API_KEY`,
`NLI_STRATEGY` (`hybrid`), `NLI_TOP_K` (`10`), `PROMPT_MODE` (`ids`), `PASSAGE_CHARS` (`0` = pages).

---

## 8. Next steps

1. Multilingual dense retrieval (e.g. BGE-M3) combined with BM25, aimed at the weakest language pairs.
2. Error analysis of Entailment → Contradiction cases on booklets with opposing viewpoints.
3. Return a precise sentence within each cited page as evidence, keeping whole pages as model context.
4. Layout-aware parsing (e.g. Docling) for tables and graphical headings.
5. Evaluate Apertus 8B against 70B for the speed/quality trade-off.

---

## License

Creative Commons Attribution 4.0 (CC-BY-4.0).
