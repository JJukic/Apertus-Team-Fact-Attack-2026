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
system reaches **macro-F1 0.928** on the advanced task (booklet PDF) and **0.975** on the beginner task (reference
text), at ~4,700 / ~2,500 input tokens, ~57 output tokens and a p95 inference time of 3.3 s / 1.7 s.

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
| Advanced | **0.928** | 0.960 | 0.913 | 4,696 | 59 | 1.9 s / 3.3 s |
| Beginner | **0.975** | 0.984 | 0.971 | 2,523 | 55 | 1.1 s / 1.7 s |

Advanced task by language pair (claim → booklet):

| | → de | → fr | → it |
|---|---:|---:|---:|
| **de** | 1.000 | 0.936 | 0.884 |
| **fr** | 0.914 | 0.944 | 1.000 |
| **it** | 0.798 | 0.922 | 0.940 |

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
| Second, focused verification pass for every "contradiction" | fixed 0 of 23 errors, broke 24 correct answers (0.908 → 0.852) | rejected |
| Speaker-aware retrieval (hide the opposing side's argument pages) | 0.908 → 0.896 | rejected |
| Few-shot examples for committee vs. Federal Council claims | 0.908 → 0.906; errors shift from E→C to E→N | rejected |
| Confidence threshold on label logprobs | wrong answers are as confident as correct ones | rejected |
| Keep the model's label when it disagrees with its own confidences (Decision-Rule 0) | 44 fixed / 13 broken over all saved runs; test 0.920 → 0.928 | adopted |
| Section labels from bare speaker lines and closing recommendation boxes (older booklets) | Federal Council pages labelled as committee 27 → 0; F1-neutral (test 0.928 → 0.925, dev 0.908 → 0.908), dev E→C 23 → 16, evidence grounding 0.86 → 0.88 | adopted (correctness) |
| Flip committee Contradictions that cite only Federal Council pages | dev 7 fixed / 8 broken, test 0 / 7 (simulated) | rejected |
| Clip pages > 3,000 characters to the window best matching claim + vote title | test 0.925 → 0.928, input tokens 5,725 → 4,696 (p95 12.7k → 7.4k); dev 0.908 → 0.913 | adopted |
| Thinking before answering (prefilled reasoning marker) | ~+3 F1 (0.889 → 0.931 on 363 paired dev pairs, 20 fixed / 5 broken); ~1,200 output tokens and ~22 s per claim instead of 59 and ~2 s | rejected (efficiency), `THINKING=false` |
| Translate cross-lingual claims into the booklet language first | dev 0.908 → 0.919, test 0.925 → 0.923 (14 fixed / 11 broken overall); +0.6 s | rejected (noise) |
| Apertus 8B instead of 70B (450 dev pairs) | 0.851 vs. 0.908 F1, p95 1.2 s vs. 4.5 s | 70B kept (quality first) |

Full tables: [docs/experiments.md](docs/experiments.md). Runs on 150 pairs carry about ±3 F1 points of sampling
noise; close decisions were re-run on 450 pairs.

### 5.4 What we learned about Apertus

- **Selected context beats more context.** With the whole booklet (~60k tokens) Apertus 70B reaches 0.73 macro-F1,
  with 10 selected pages 0.93. We found no position effect ("lost in the middle"): the full-document runs fail even
  when the answer is in the first third of the booklet, so the degradation is general distraction.
- **Apertus is over-confident.** Its self-reported confidences are 0 or 1 in 97–99 % of cases, and the probability of
  the label token is ~1.0 for wrong answers as well as for correct ones (41 wrong vs. 40 correct dev answers).
  Confidence thresholds cannot filter its errors.
- **Speaker attribution is its main weakness.** Claims attributed to the initiative/referendum committee have a 16–18 %
  error rate (2–8 % for all others), even when instructed not to judge them against the other side, when asked to
  verify, with worked examples, when the opposing argument pages are removed, and after fixing section labels that had
  marked Federal Council pages as committee pages in older booklets.
- **Thinking helps, at a high price.** Letting the thinking model reason before its answer fixes half of the remaining
  dev errors (~+3 F1), but needs ~20× the output tokens and ~10× the latency. A prompt instruction alone does not make it
  think first; the reasoning marker has to be prefilled. Selective thinking saves little (it is needed on ~3/4 of claims).
- **Self-contradictory answers.** In ~2 % of answers the label and the stated confidences disagree; the label is right
  3× as often, so the decision rules now keep it.
- **Cross-lingual asymmetry.** Italian claims against German booklets are the hardest pair (0.80); French claims
  against Italian booklets reach 1.00.
- **8B vs. 70B.** Apertus 8B is ~3.5× faster (p95 1.2 s) but 6 F1 points weaker, mostly on cross-lingual pairs and
  contradictions.

---

## 6. Limitations

- **Italian claims against German booklets** are the weakest pair (0.80 F1). Most of these errors happen with the gold page
  in context, so better retrieval alone would not fix them (gold-section recall is 0.92–0.93 overall).
- **Remaining errors** on the test split (29): Entailment → Neutral 10, Contradiction → Neutral 8, Entailment →
  Contradiction 6 — mostly claims attributed to the committee (see 5.4) and 'if accepted, X must …' claims; prompt rules,
  a verification pass and speaker-aware retrieval did not fix them, thinking fixes about half at ~10× the latency.
- **Evidence precision:** with whole pages as passages, ~75 % of cited pages lie inside the gold reference section.
  Smaller passages raise this to ~90 % but cost F1 (5.3).
- **Layout:** section headings are detected from text lines and recommendation boxes; committee pages headed only by a
  slogan stay unlabelled. Tables and charts are read as plain text.
- **Latency** was measured client-side on a shared endpoint; the organisers measure it themselves.
- **Not political advice:** outputs describe the relationship between a claim and the official booklet only.

---

## 7. Reproducibility

```bash
export LLM_API_KEY="your_api_key_here"
make run                    # Docker build + benchmark on the demo set
make test                   # unit tests, no API calls

cd track_2a
python -m src.hf_dataset                                   # dataset, booklets, dev/test split
python -m src benchmark -d data/hf/test.jsonl -t advanced  # writes results/<timestamp>_....json
python -m src benchmark -d data/hf/test.jsonl -t beginner
python -m src.results_summary                              # results/summary.json (used by the app)
python -m src warm-cache                                   # pre-parse booklets (disk cache keyed by file hash)
```

Configuration is read from environment variables: `LLM_NAME`, `LLM_BASE_URL`, `LLM_API_KEY`,
`NLI_STRATEGY` (`hybrid`), `NLI_TOP_K` (`10`), `PROMPT_MODE` (`ids`), `PASSAGE_CHARS` (`0` = pages).

---

## 8. Next steps

1. Error analysis of the remaining Entailment → Contradiction cases on Italian claims (it→de), where the gold page is retrieved.
2. Multilingual dense retrieval (e.g. BGE-M3) — low priority: perfect retrieval would fix at most ~2 F1 points.
3. Return a precise sentence within each cited page as evidence, keeping whole pages as model context.
4. Layout-aware parsing (e.g. Docling) for tables and graphical headings.
5. Fine-tuning or a stronger model for speaker attribution, the error class that resisted prompting and retrieval changes.
6. A cheaper form of reasoning (a short, length-limited thinking budget enforced by `max_tokens` with a forced answer) to
   keep part of the thinking gain at a fraction of its latency.

---

## License

Creative Commons Attribution 4.0 (CC-BY-4.0).
