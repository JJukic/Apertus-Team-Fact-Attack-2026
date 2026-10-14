# Independent experiments — generalisation to unseen voting dates (from 2026-10-11)

Goal: better expected performance on **unseen voting dates, claims and language pairs** under the full
competition score (macro-F1, evidence Hit@5, tokens, non-LLM time), not a higher score on a split we already know.
Every change is our own hypothesis, implemented in our code and judged with our own runs; no external code, prompt
or pipeline is used.

## 1. Data status and protocol

- **Population:** OST dataset v1.1 (pinned revision `9ff08597`, 1,488 rows → 1,488 task A + 1,488 task B cases),
  20 voting dates, prepared by `scripts/prepare_competition_cases.py` into 5 folds grouped by voting date (seed 42,
  all languages of a date in one fold).
- **Everything here is development data.** All 20 dates have been used for earlier design decisions; the former
  test split (5 dates, 401/402 pairs) served several before/after decisions and is **not an independent holdout**.
  Grouped folds measure how stable a change is across voting dates; they do not prove generalisation to unseen
  dates.
- **Duplicates:** 575 of 1,495 rows (236 groups) are exact duplicates of (claim, booklet) with identical labels;
  12 claim texts occur at more than one date, 3 in both former splits. Reports therefore add macro-F1 on
  de-duplicated groups and a bootstrap over voting dates rather than over rows.
- **Only near-unseen check:** our 36 claims on the 27 September 2026 booklet (not in the repository). Task A there
  is 0.77–0.80 — a warning sign, not a precise estimate (n = 36, own claims, already looked at for two decisions).
  **The set itself is lost:** it was never committed and is in no branch, remote, earlier scratchpad or user folder
  on this machine (searched 2026-10-11); only the scores in `docs/experiments.md` remain.
- **New untouched holdout (built 2026-10-11):** 90 own claims × task A/B on the booklets of the
  **27 September 2026** vote (Neutralitätsinitiative, Ernährungsinitiative; downloaded from admin.ch, not part of
  the dataset). 30 entailment / 30 neutral / 30 contradiction, 10 claims per source→claim language pair, claim
  types (Federal Council, summary, committee, legal text, consequences of a yes) in roughly the dataset's
  proportions per label. Each claim names its basis page; contradictions change exactly one fact; neutral claims
  concern a topic the booklet does not address. Gold references follow the dataset (vote title + PyMuPDF text of
  the part the claim refers to; the detailed explanation for consequence and neutral claims).
  `data/holdout_2026-09-27/claims.jsonl` (sha256 `66528247…70bf54`) and `manifest.json` freeze it;
  `scripts/build_holdout.py` rebuilds the cases from the official PDFs and checks their hashes.
  **Rules:** written before any system ran on these booklets; never used to choose or tune anything; run
  **once**, at the end, on the final candidate and on the baseline. Limitations: written by the same person/agent
  that develops the changes (an independent label check, e.g. by Josip, is advisable), own claims may differ in
  style from the organisers', two proposals only, n = 90 per task (macro-F1 ± ~0.05).
- **Decision rule:** a change is adopted only if its benefit clears the run-to-run noise of the unchanged system,
  is not concentrated in single folds or language pairs, and costs no relevant quality, evidence or efficiency.

## 2. Reproducibility of the submission configuration (E0)

- The Docker image sets none of the pipeline variables, so the **submission runs on the defaults of
  `src/config.py`**. `.env.example` (repository root and `track_2a/`, kept identical) now lists every setting
  with that default; `tests/test_submission_config.py` checks file and code agree, also inside the image.
- **Local discrepancy found:** the developer `.env` files (not in git, excluded from the image) set
  `NLI_STRATEGY=retrieval` (the old proposal filter) and, in `track_2a/.env`, `LLM_NAME=swiss-ai/Apertus-v1.5-8B`.
  Any local run from `track_2a/` that relied on them measured a different system than the submission
  (one unit test also failed because of it). The image is not affected (`.dockerignore` excludes `.env`).
- **Decision runs** use `scripts/run_decision.py`: it forces the settings of `.env.example` (+ explicit overrides),
  checks the effective configuration, calls the official CLI exactly as the jury does
  (`python -m src --input … --output …`), scores with the vendored official `evaluate.py`, and saves predictions,
  official score, per-fold/language/class metrics, the run manifest (commit, source hashes, settings, CLI timing,
  cases answered through the CLI failure path) and, for uncommitted sources, the exact diff under
  `results/generalization/<run>/`. Before/after reports: `scripts/report_decision.py`.
- Documented final numbers before this work (test 401: A 0.945 / Hit@5 0.839, B 0.975) cannot be checked from the
  repository: their prediction files were never committed.

## 3. Experiments

| ID | Date | Hypothesis | Status |
|---|---|---|---|
| E0 | 2026-10-11 | Baseline of the exact submission configuration on all 2,976 cases | done (reference) |
| E1 | 2026-10-11 | Fewer legal-text pages let the explanation pages into the context (task A) | **rejected** |
| E2 | 2026-10-11 | Task B: a long reference can be cut to its relevant chunks without losing quality | **adopted** (default since the holdout check) |
| E3 | 2026-10-11 | Task A: tighter page windows (2,000 instead of 3,000 characters) save tokens at equal quality | **rejected** |
| E4 | 2026-10-11 | Task A evidence: rank text blocks with the claim and its translation | **not adopted** (offline replay) |
| E5 | 2026-10-12 | Task A: navigation bars give their pages a section label (summary / in detail) instead of none | **rejected** (section 6) |
| E6 | 2026-10-12 | Task A evidence: the five best text blocks of the cited pages instead of whole pages first | **rejected** (offline, section 6) |

### E0 — baseline (`results/generalization/B0_submission`)

Commit `3bf73bd`, unchanged sources, submission defaults (then written out in `configs/submission.env`, now in
`.env.example` with the same values), official CLI, 4 workers, booklet parses cached
(as in the image), translations acquired cold. 2,976 cases, 0 failed cases, 0 API failures.

| Task | Macro-F1 | Fold mean ± sd (min–max) | Cross-language F1 | Hit@5 | Input / output tokens per case |
|---|---:|---:|---:|---:|---:|
| A | 0.943 | 0.943 ± 0.024 (0.919–0.977) | 0.943 | 0.869 | 3,866 / 76 |
| B | 0.981 | 0.981 ± 0.004 (0.976–0.985) | 0.979 | – | 2,482 / 56 |

On the 5 dates of the former test split (398 v1.1 pairs) the run gives A 0.942 / Hit@5 0.833 and B 0.972, which
matches the documented 0.945 / 0.839 / 0.975 within run-to-run variation: **the documented numbers reproduce**.
Task A varies across date folds by ±0.024, i.e. a single split can be off by several points. CLI non-LLM time
0.3 ms per case in this mixed, concurrent run (local estimate, not the proxy's measurement).

**Errors of E0** (task A 85, task B 28 of 1,488 each; gold → predicted):

| Task | E→C | E→N | C→N | C→E | N→C | N→E |
|---|---:|---:|---:|---:|---:|---:|
| A | 34 | 20 | 19 | 8 | 3 | 1 |
| B | 5 | 0 | 11 | 6 | 6 | 0 |

- Missing information is rarely read as a contradiction (N→C: 3 in task A); the largest confusion is a supported
  claim read as contradicted (E→C).
- Task A error rate by the part of the booklet a claim names: consequences of a yes ("Wird die Vorlage
  angenommen …") 15.5 % (30/194), committee 7.6 %, legal text 6.3 %, summary 5.0 %, Federal Council 0.2 %.
  The same consequence claims fail in only 3.6 % of task B cases, where the right passage is given: their
  task A errors come mostly from page selection, not from reading — the bottleneck E1 addresses.

**Run-to-run noise** (`results/generalization/B0b_repeat`: the unchanged configuration run a second time, with the
E1/E2 code present but switched off): task A 0.945 / Hit@5 0.870, task B 0.982. Compared with E0 only **3 labels in
task A and 2 in task B** of 1,488 changed (evidence 4 / 3); no fold got worse. The endpoint answers almost
deterministically, so the label changes of an experiment are effects of the change, not noise. It also shows that the
refactoring for E1/E2 does not change the behaviour while their flags are off.

### E1 — page selection in task A: cap the legal-text pages

- **Bottleneck (own analysis, no API):** the gold page is among the 8 selected pages in 0.928 of the
  de-duplicated gold-0/2 cases, but only 0.903 for claims about the summary and 0.761 for claims about the
  consequences of a yes ("Wird die Vorlage angenommen …"); per voting date the recall@8 ranges 0.755–1.000.
  In the misses the gold page sits just below the cut (rank 9–12) while legal-text pages hold ~40 % of the
  top-8 places: the legal text repeats the vote title, which the ranking weights twice.
- **Hypothesis:** at most 2 legal-text pages in the context (the next pages by rank take the freed places, the
  context size stays 8) bring the explanation pages in without hurting claims about the legal text.
- **Own implementation:** `section_caps` in `PassageRetriever.retrieve_hybrid`, switched by
  `LEGAL_TEXT_MAX_PAGES` (default 0 = off). It relies only on our existing legal-text section label, not on how a
  claim is phrased.
- **Offline selection of the variant** (original query only, a lower bound for the production `union` ranking):

  | Variant | recall@8 | worst date @8 | dates better / worse | legal-text claims @8 |
  |---|---:|---:|---:|---:|
  | production | 0.928 | 0.755 | – | 0.984 |
  | **cap 2, every claim** | **0.943** | **0.837** | **8 / 0** | 0.984 |
  | cap 1, every claim | 0.933 | 0.857 | 8 / 6 | 0.855 |
  | cap 1 unless the claim names the legal text | 0.953 | 0.867 | 11 / 0 | 0.984 |

  The cap-1 variant that exempts legal-text claims is better offline but depends on recognising the claim's
  wording; the plain cap of 2 is chosen because it does not, which we expect to transfer better to unseen claims.
  The implementation selects the same pages as the simulation in 402/402 test cases.
- **Nested check of the choice** (does picking a variant on some dates transfer to dates not used for picking?).
  Rule fixed before looking: among variants worse on no selection date, the highest per-date mean recall@8.

  | Variant | select on folds 0–2 → held-out folds 3–4 | select on folds 3–4 → held-out folds 0–2 |
  |---|---:|---:|
  | production | 0.912 | 0.946 |
  | cap 2, every claim (tested in E1) | 0.934 (3 dates better / 0 worse) | 0.957 (5 / 0) |
  | cap 1 unless the claim names the legal text | 0.944 (4 / 0) | 0.966 (7 / 0) |

  The effect holds on the held-out dates in both directions. The rule picks the claim-dependent cap-1 variant both
  times; it stays a candidate (E1b) with the risk that a claim about the legal text which we do not recognise is
  treated like any other claim (cap 1 for every claim drops legal-text claims from 0.984 to 0.855).
- **Related earlier result:** on 2026-10-08 a context filter `low+legal1` (at most one legal-text page, other
  low-value pages dropped, **freed places not refilled**) was rejected: dev 300 macro-F1 0.927 → 0.910, Sep 2026
  set 0.804 → 0.779. E1 differs in refilling the freed places (the context keeps 8 pages) and in allowing two
  legal-text pages; whether that difference matters is exactly what the API run has to show.
- **Result** (`results/generalization/E1_legal_cap2`, `LEGAL_TEXT_MAX_PAGES=2`, task A, 1,488 cases, same CLI
  path, 0 failed cases; full report in `report/report.md`):

  | | Baseline E0 | E1 | Change |
  |---|---:|---:|---:|
  | Macro-F1 | 0.943 | 0.937 | −0.005 (95 % date bootstrap −0.017 … +0.005) |
  | Macro-F1, de-duplicated | 0.944 | 0.943 | −0.002 |
  | Evidence Hit@5 | 0.869 | 0.866 | −0.003 |
  | Input tokens / case | 3,866 | 3,752 | −2.9 % |
  | Folds / language pairs worse | – | 3 of 5 / 4 of 9 | |

  Labels fixed / broken: 11 / 19. By the part a claim names: consequence claims 4 / 1 and summary claims 3 / 6,
  but **legal-text claims 0 / 10** — although their gold page stayed in the context (offline recall unchanged at
  0.984). Questions about the legal text need the other legal-text pages as context, not only the one page that
  holds the gold passage; recall@k of the gold page does not capture that.
- **Decision: rejected.** No gain in macro-F1 or Hit@5, worse in most folds; the token saving (−2.9 %) does not
  make up for it. The 30 changed labels are far above the 3 changes of the repeated baseline: the harm is real. E1b (exempt claims that name the legal text) would avoid the ten breaks, but what remains
  (non-legal-text claims 11 fixed / 9 broken) is within run-to-run noise; not pursued before the deadline.
- **Lesson for the method:** the offline proxy (gold page among the selected pages) overstated the benefit; for
  context changes it must be complemented by an API run before any claim of improvement.
- **Origin:** we broke our recall down by the part a claim names; the bottleneck (legal text crowding out
  explanation pages) and the fix are our own findings and design.

### E2 — task B: relevant chunks of long references

- **Bottleneck (own analysis):** task B sends the whole reference (mean 7,048 characters, ~2,530 input tokens per
  case). All 498 neutral references are longer than 8,000 characters (median 14,568), entailment/contradiction
  references only in 34 of 997 cases. This length pattern is a property of the development data; the method must
  not and does not use length as a signal for the label.
- **Hypothesis:** a reference longer than `REFERENCE_MAX_CHARS` can be sent as its first chunk plus the chunks
  that best match the claim, in document order, without a relevant loss in macro-F1.
- **Own implementation:** `ClaimVerificationEngine._focus_reference`, using our existing reference chunking and
  BM25 ranking; for a claim in another language it adds the claim's translation to the query (our existing
  retrieval translation, its tokens counted). Default 0 = off.
- **Budget chosen up front: 6,000 characters** (mean reference characters sent −42 %; 91 of 997
  entailment/contradiction references would be cut, each only moderately). The choice used the length
  distribution of the development data and is documented as such.
- **Design:** chunking, BM25 + translation, the opening chunk and the character budget are our own and are tested
  against our full-text baseline on our folds.
- **Result** (`results/generalization/E2_reference_6000`, `REFERENCE_MAX_CHARS=6000`, task B, 1,488 cases,
  0 failed cases; full report in `report/report.md`):

  | | Baseline E0 | E2 | Change |
  |---|---:|---:|---:|
  | Macro-F1 | 0.981 | 0.984 | +0.003 (95 % date bootstrap −0.002 … +0.008) |
  | Macro-F1, de-duplicated | 0.979 | 0.983 | +0.004 |
  | Macro-F1, cross-language | 0.979 | 0.984 | +0.005 |
  | Input / output tokens per case | 2,482 / 56 | 1,732 / 62 | **−30.2 %** / +10 % (translation answers) |
  | Folds / language pairs worse | – | 1 of 5 (−0.007) / 1 of 9 (it→fr −0.014) | |

  589 of 1,488 references were cut; among them, the 91 entailment/contradiction cases are correct 79 times before
  and after (no net loss where information could be cut away). Labels fixed / broken 9 / 4 (unique claims 8 / 3:
  two broken rows are the same claim). All per-class F1 within ±0.007.
- **Decision: adopt**, subject to the one final run on the untouched 27 September 2026 holdout. (a) Noise check
  done: 13 changed labels against 2 for the repeated baseline, so the changes are caused by E2, with a net gain of 5. The token saving is large and
  stable; quality does not drop in any class and in at most one fold / pair by less than the bootstrap width.

## 4. External material

Only the public dataset (v1.1), the organisers' starter scorer (`evaluate.py`) and the Apertus endpoint are used.

### E3 — task A: tighter page windows

- **Bottleneck:** task A sends ~3,866 input tokens per case, mostly the 8 selected pages (~11,300 characters after
  the existing 3,000-character window per page); tokens are scored relative to the most frugal team.
- **Hypothesis:** a 2,000-character window per page (the window that best matches claim and vote title, as today)
  keeps the information the model needs while sending fewer tokens. E1 showed that removing whole pages removes
  needed context, so this experiment keeps all 8 pages and only narrows each one. One variable: `PAGE_MAX_CHARS`.
- **Offline estimate (written down before the API run):** context −11 % (≈ −8 % input tokens with the prompt),
  pages narrowed 9.6 % → 23.8 %, share of gold-passage words still visible on gold pages 0.997 → 0.968
  (16 of 734 cases lose more than 20 %). 1,500 characters would save −26 % but lose more than 20 % of the gold
  words in 282 of 734 cases and is not tested. Evidence is built from the full page and is not narrowed.
- **Provenance:** own analysis; no external source.
- **Result** (`results/generalization/E3_page_clip2000`, `PAGE_MAX_CHARS=2000`, task A, 1,488 cases, 0 failed
  cases; full report in `report/report.md`):

  | | Baseline E0 | E3 | Change |
  |---|---:|---:|---:|
  | Macro-F1 | 0.943 | 0.934 | −0.008 (95 % date bootstrap −0.014 … −0.002) |
  | Macro-F1, cross-language | 0.943 | 0.933 | −0.010 |
  | Evidence Hit@5 | 0.869 | 0.864 | −0.005 |
  | Input tokens / case | 3,866 | 3,514 | −9.1 % |
  | Folds / language pairs worse | – | 3 of 5 / 5 of 9 | |

  Labels fixed / broken 6 / 18 against 3 / 0 for the repeated baseline; all three classes lose F1.
- **Decision: rejected.** The whole interval of the F1 change is negative, and the loss is spread over folds,
  language pairs and classes. As in E1, the offline proxy (gold words still visible: 0.968) did not predict the
  loss: the model also uses the text around the gold passage. Further cuts of the task A context are not pursued.

### E4 — evidence blocks ranked in the booklet language (offline replay, no API)

- **Bottleneck:** Hit@5 misses of E0 (130 of 990 gold entailment/contradiction cases): 55 cite pages other than the
  gold pages, 39 are predicted neutral (no evidence), 36 cite a gold page but no matching item is among the first
  five. Block ranking uses only the original claim's terms, which barely match for cross-language claims.
- **Hypothesis:** adding the terms of the claim's translation (already computed for retrieval, no extra tokens)
  ranks the right blocks of the cited pages higher.
- **Method:** the evidence of E0 was rebuilt from its cited pages; the control (current ranking) reproduces the
  submitted evidence exactly in 951/951 cases, translations come from E0's cache (no API call possible).
- **Result:** Hit@5 0.869 → 0.873 (9 hits gained / 5 lost), fold 1 −0.010, it→de −0.037, cross-language
  0.856 → 0.863.
- **Decision: not adopted.** +4 cases of 990 and not consistent across folds and language pairs; at 20 % weight
  in task A the expected effect is negligible. The remaining evidence headroom lies in which page the model cites
  and in neutral predictions, not in block ranking.
- **Provenance:** own analysis; no external source.

### Label check of the holdout (2026-10-11)

- Felipe Wüthrich checked the 90 holdout claims on a review page: German translation of every French/Italian
  claim, the booklet part the claim names with a link to the page of the German booklet, and an option to hide the
  author's label and basis until his own judgement.
- **90 of 90 agree** (entailment 30, neutral 30, contradiction 30; Cohen's kappa 1.0); the judgement on claim 8
  (neutral) was not saved on the page and was confirmed in chat. Three notes, none asking for a change. No label
  was changed.
- Limits: the reviewer had the claims file with the labels open before the review, so the check may not be
  fully blind; translations are reading aids, the claims are judged in their original language.
  Judgements: `data/holdout_2026-09-27/label_review.jsonl`.

## 4. Final check on the untouched holdout (2026-10-11, run once)

Both configurations on the 27 September 2026 holdout (90 claims × task A/B, labels checked 90/90), each run once
with `scripts/run_decision.py` (`results/generalization/HO_baseline`, `HO_E2_reference_6000`; reports in
`HO_E2_reference_6000/report_A|B`). 0 failed cases.

| | Baseline | Baseline + E2 |
|---|---:|---:|
| Task A macro-F1 (identical configuration in both runs) | 0.933 | 0.944 |
| Task A evidence Hit@5 | 0.833 | 0.850 |
| Task A input tokens / case | 3,883 | 3,863 |
| **Task B macro-F1** | 0.955 | **0.966** |
| Task B cross-language macro-F1 | 0.967 | 0.983 |
| **Task B input / output tokens per case** | 2,027 / 55 | **1,758 / 60** (−13 % input) |

- **E2 confirmed:** task B 1 label fixed, 0 broken, no language pair worse; the saving is smaller than on the
  dataset (−13 % vs −30 %) because the holdout's references are shorter. **E2 is now the default**
  (`REFERENCE_MAX_CHARS=6000` in `src/config.py` and `.env.example`).
- Task A ran with the same settings twice; its one changed label (claim 73) is run-to-run variation, consistent
  with the noise floor measured on the dataset.
- **Generalisation estimate on unseen booklets:** task A ≈ 0.93–0.94 (Hit@5 0.83–0.85), task B ≈ 0.96 — close to
  the development population (0.943 / 0.869 / 0.981–0.984), with ± ~0.05 at n = 90. The earlier 0.77–0.80 on the
  lost September set is not reproduced on this one. Weakest on the holdout: cross-language task A pairs
  (de→fr 0.60–0.85, fr→de 0.79); all errors stay within a few cases per pair.
- From now on this holdout is spent: it is reported, not used for further decisions.

## 5. Final image check with the real API (2026-10-11)

Image built from commit `59b55cd` (`docker build --platform linux/amd64`, 60/60 dataset booklets pre-parsed).
- Test suite in the container, `--network none --read-only --tmpfs /tmp`: 143 tests pass.
- Jury-style run against the real Apertus endpoint (`--input /data/cases.jsonl --output /output/predictions.jsonl`,
  `/data` read-only, `BASE_URL` / `API_KEY` injected): 27 mixed cases — 9 advanced cases on booklets in the image
  (all 9 language pairs), 6 on the new 27 September 2026 booklets (parsed at run time), 12 beginner cases (7 with
  references over 6,000 characters). Exit code 0, 27/27 answered in input order, no warnings, valid labels, evidence
  pages 1-based for advanced and `null` for beginner, token metrics present; CLI estimate 15 ms non-LLM time per
  case including start-up and the three new booklets. E2 is active in the image: long references cost
  2,100–2,500 input tokens per case. Official `evaluate.py`: advanced 15/15 correct, Hit@5 6/7; beginner 11/12
  (macro-F1 0.896 on 12 cases); both above the minimum thresholds. This is a contract check, not a quality estimate.

## 6. Task A error diagnosis, E5 and E6 (2026-10-12)

Everything here is development data (all 20 voting dates); the holdout was not touched. Artefacts:
`results/generalization/D0_error_diagnosis` (diagnosis), `E5_navigation_sections`, `E6_evidence_blocks_offline`.

### D0 — where task A loses points (offline, B0 predictions, no API)

The context of every task A case of B0 was rebuilt with the production retrieval (`diag_contexts.py`; the cited pages
of B0 lie outside the rebuilt context in 2 of 1,488 cases). Gold passages were mapped to booklet pages with the
official matcher on the passage **without its leading vote title**: every one of the 990 gold references starts
with the vote title, and any block that only repeats the title would otherwise count as a gold page.

| B0, task A | Label errors (85) | Hit@5 misses (130) |
|---|---:|---:|
| Gold page not in the 8-page context | 12 | 12 (+9 predicted neutral) |
| Gold page in context, decisive text clipped away | 0 | – |
| Gold page in context, label wrong | 69 | – |
| Gold page in context, predicted neutral (no evidence) | – | 30 |
| Gold page in context, not cited | – | 57 (30 summary claims; 51 cite three other pages) |
| Gold page cited, none of the five items matches | – | 22 |
| API / format failures | 0 | 0 |

- Retrieval is not the bottleneck (gold page in context in 97.9 % of gold-0/2 cases); errors are reading and
  citation errors. 82 of 85 label errors are identical in B0 and B0b, and they come from 64 claim texts: 16 claims
  (the same claim against the DE/FR/IT booklets) account for 37 errors.
- Largest groups: consequence claims E→C 18 / E→N 11, committee claims E→C 15. Example: "les droits de douane sur
  tous les principaux produits suisses exportés vers l'Indonésie seront éliminés" is answered Contradiction against
  the German booklet four times, although the context contains "auf allen wichtigen Schweizer Exporten nach
  Indonesien [werden] die Zölle wegfallen".
- Rejected without an API run: choosing the 3,000-character page window with the claim's translation as well
  (`clip_to_query` uses claim + vote only). Only 109 gold pages in 82 cases are clipped at all, 92 % of their gold
  sentences stay in the window with or without the translation (0 better, 0 worse), and none of the 85 label errors
  involves a clipped gold page.

### E5 — section labels for navigation-bar pages (`results/generalization/E5_navigation_sections`)

- **Bottleneck:** every part of a proposal opens with a navigation bar listing the parts that follow with their page
  numbers ("Zweite Vorlage: Jagdgesetz | Vorlage im Detail 26 | Argumente 34"). `detect_section_heading` read these
  as a table of contents and reset the section, so 434 pages (overview, in detail) reached the model without a label,
  while the prompt asks it to check "according to the summary …" claims against the summary.
- **Hypothesis:** a navigation-bar page belongs to the part before the first listed one; with the label, summary
  claims cite the summary page (Hit@5 up) at unchanged tokens.
- **Implementation** (branch `exp/e5-navigation-sections`, commit `edbd8f7`, not merged; diff in `src.diff`):
  `navigation_section()` in `pdf_parser.py`, parse cache version 8. Page text identical on all 3,241 pages; exactly
  434 labels change, all from none (191 summary, 223 in detail, 20 Federal Council arguments).
- **Label check** (stratified reading of the relabelled pages): in detail 8/8 and summary ~7/8 correct, but the 20
  "Federal Council" labels are all wrong (detail pages): plural headings ("Arguments des comités référendaires") are
  not matched and the Federal Council heading wraps before its page number, so only "voting text" was found. Through
  the speaker boost these labels moved 26 of 1,488 contexts.
- **Result** (task A, 1,488 cases, 0 API failures, 0 failed cases; the scoring step of `run_decision.py` crashed on
  Windows after inference and was redone on the unchanged predictions, see `run.json` and the UTF-8 fix below):

  | | B0 | B0b (repeat) | E5 |
  |---|---:|---:|---:|
  | Macro-F1 | 0.9425 | 0.9446 | 0.9446 |
  | Evidence Hit@5 | 0.8687 (860) | 0.8697 (861) | 0.8737 (865) |
  | Input tokens / case | 3,866 | 3,867 | 3,868 |
  | Labels fixed / broken vs B0 · vs B0b | – | – | 8 / 5 · 6 / 6 |
  | Hit@5 fixed / broken vs B0 · vs B0b | – | 7 cases differ from B0 | 21 / 16 · 20 / 16 |

  Macro-F1 change vs B0 −0.002 … +0.007, vs B0b −0.005 … +0.005 (95 % date bootstrap). F1 worse in 3 of 5 folds;
  Hit@5 worse in folds 0 (−3) and 4 (−5) and for de→fr (0.81 → 0.73). Summary claims gain Hit@5 (183 → 193), but
  they do not cite the summary page more often (93 → 92 cases), so the predicted mechanism is not what moved them.
  Federal Council claims 1 → 2 label errors, legal-text claims 14 → 16.
- **Decision: rejected.** The Hit@5 gain (+5 vs B0, +4 vs B0b) does not clear the predeclared bar (> 5 hits) nor
  the run-to-run noise (7 cases), F1 is unchanged, and the change brings 20 wrong labels. A corrected variant without
  the Federal Council labels was estimated offline: only 2 of the 16 lost hits and 1–2 label changes involve them,
  so it would stay within noise as well; not run.

### E6 — five best text blocks instead of whole pages first (offline, `results/generalization/E6_evidence_blocks_offline`)

- **Hypothesis:** of the five evidence items (three whole cited pages + two best blocks), more blocks hit the gold
  passage more often. Labels, prompt and tokens unchanged; replayed on the fixed citations of B0 and B0b (the replay
  of the production composition reproduces 860/990 exactly). Variant chosen from four (3/2, 2/3, 1/4, 0/5 pages /
  blocks); a nested fold check picked 0/5 on either half and it gained on the other (+7, +8 hits).
- **What Hit@5 measures** (official `evaluate.py`): for the 990 cases with gold entailment / contradiction, whether
  one of the first five *submitted evidence texts* (each ≤ 5,000 normalised characters) fuzzy-matches the gold
  passage (`partial_ratio` ≥ 90). It is not retrieval recall; the page number is not used; a neutral answer is a miss.
- **Result:**

  | Fixed B0 citations (B0b alike) | Production | E6 |
  |---|---:|---:|
  | Hit@5, official | 0.8687 | 0.8838 (+27 / −12) |
  | Hit@5 counting only matches of the passage body (without the vote title) | **0.8434** | **0.7657** |
  | Hits that match only the vote title | 25 | 117 |

  12 of the 27 gained hits match only the vote title that opens every gold reference (gained blocks: median 130
  characters); all 12 lost hits are whole gold pages that contain the passage.
- **Decision: rejected.** The gain comes from the scoring definition, not from better evidence: it would exploit
  the title prefix while the real evidence gets clearly worse (0.843 → 0.766 on the passage body).

### Not changed by E5 / E6

The production configuration (`.env.example`, `src/config.py`) is untouched: E5 lives only on its unmerged branch,
E6 only as offline scripts. Fixed alongside: `scripts/compare_competition_runs.load()` (used by `run_decision.py`
and `report_decision.py`) now reads prediction files as UTF-8; before, the scoring step of a decision run crashed
on Windows (cp1252) after inference had finished. Inference, prompts and model settings are unchanged. On a Windows
console `report_decision.py` also needs `PYTHONIOENCODING=utf-8` to print its report (it contains "→").

## 7. E7 — compact task A prompt, paired full evaluation (2026-10-12)

Candidate (see *Integration* below); the submission default stays off. Branch `exp/task-a-prompt-screening`, artefacts in
`results/generalization/E7_*` (see `E7_analysis/README.md`).

- **Hypothesis:** four short label rules appended to the task A system prompt (`A_COMPACT_PROMPT`, ~+97 tokens) fix
  over-cautious neutral answers without the shift towards contradiction seen with the longer skill and the relation
  prompt (180-case screen: representative 7 → 5 errors, Hit@5 66 → 70 of 80; diagnostic E→C 20 → 18).
- **Protocol:** fresh baseline and candidate on the same 1,488 cases; NLI input hashes 1,488/1,488 identical to a mock
  dry run before inference; same translation cache (unchanged); only `A_COMPACT_PROMPT` differs. 0 final API
  failures (baseline: 2 transient errors, retried successfully), 0 cases through the failure path.

| Task A, 1,488 cases | Fresh baseline | Compact | Change |
|---|---:|---:|---:|
| Macro-F1 | 0.9419 | 0.9493 | +0.0075 (date bootstrap 95 %: −0.0022 … +0.0179; 7 % of resamples ≤ 0) |
| Macro-F1, de-duplicated | 0.943 | 0.953 | +0.010 |
| Accuracy | 0.9422 | 0.9496 | |
| Errors | 86 | 75 | 22 fixed / 11 broken / 8 wrong → other wrong |
| Evidence Hit@5 (990 gold E/C) | 858 (0.867) | 872 (0.881) | +23 / −9 (bootstrap 95 %: +0.002 … +0.025) |
| Input / output tokens per case | 3,818 / 57.9 | 3,915 / 60.2 | +2.5 % / +4 % |
| Non-LLM time per case (CLI estimate) | 1.7 ms | 2.5 ms | both negligible, not a measured difference |

Confusion (rows gold, columns prediction, order E, C, N): baseline `[[441,34,20],[9,467,19],[1,3,494]]`, compact
`[[448,33,14],[10,472,13],[1,4,493]]`. E→C 34 → 33, E→N 20 → 14, C→N 19 → 13, N→C 3 → 4, C→E 9 → 10.

- **Folds:** 1 (+0.021) and 2 (+0.017) better, 3 and 4 unchanged (identical accuracy), fold 0 worse (−0.011; 1 fixed,
  3 broken, 2 of them N→C on one salary-minimum claim). **Dates:** 6 of 20 net better, 4 worse, 10 unchanged;
  2020-09-27 contributes 6 of the 11 net fixes (several proposals), but leaving out any single date keeps ΔF1
  between +0.004 and +0.009. **Language pairs:** 7 of 9 equal or better, de→it −0.014.
- **Reading:** a consistent but small gain in macro-F1 and Hit@5 at +2.5 % tokens. The F1 interval still includes 0
  (20 voting dates); the Hit@5 interval does not. No statistical significance is claimed.

### E7 — integration as a candidate (2026-10-12, no new inference)

- **Switch:** `A_COMPACT_PROMPT` in `src/config.py` and both `.env.example` files, **default `false`**. The submission
  image runs the defaults, so the submission is unchanged until the default is changed by a separate decision.
- **Text:** `ApertusClient.A_COMPACT_PROMPT_TEXT`, unchanged since commit `5907dd3` and identical to the source of both
  E7 runs (`src_sha256` of `apertus_client.py` in their `run.json`); SHA-256 of the text
  `1d392f22b1eb19ee741f7a0fef1cd676c00ef05db480a811269abc8586d67adc`. It contains "…, but never on information they do
  not state." (added before the screen); a version without that clause was never evaluated.
- **Scope:** appended to the task A system prompt only; retrieval, translation, caching, labels, evidence formatting
  and model parameters are untouched.
- **Tests** (`tests/test_apertus_client.py`, `TestCompactPromptSwitch`): default off in a clean interpreter; with every
  switch off the task A and task B request fingerprints equal those of the submission baseline `7c17c9c`; task B
  requests unchanged even with all three task A switches on; switch on appends exactly the evaluated text (hash) and
  changes nothing else in the request; no other prompt variant is enabled. Full suite: 153 tests pass (1 skipped).
- **Evidence status:** measured on development data only (all 20 voting dates were used for earlier decisions); the
  macro-F1 interval includes 0, the Hit@5 interval does not; fold 0 (−0.011) and de→it (−0.014) are worse. There is no
  independent confirmation, and no change of the final competition score is claimed (token and time scores are relative
  to other teams).

## 8. E8 — compact task A prompt active in the submission (2026-10-12, no new inference)

- **Activation:** `A_COMPACT_PROMPT` default `true` in `src/config.py` and both `.env.example` files. The submission image
  is built from the repository-root `Dockerfile`, which sets no pipeline variable and copies no `.env`, so the image runs
  these code defaults (checked by `test_submission_dockerfile_sets_no_pipeline_variable`). No second switch or image-only
  setting was added. `A_COMPACT_PROMPT=false` restores the previous baseline (requests byte-identical to `7c17c9c`).
- **Text:** unchanged, the E7 text (SHA-256 `1d392f22b1eb19ee741f7a0fef1cd676c00ef05db480a811269abc8586d67adc`).
- **Effect on tools:** `scripts/run_decision.py` applies `.env.example`, so decision runs now use the compact rules unless
  `--set A_COMPACT_PROMPT=false` is given (the E7 runs set the switch explicitly and are unaffected).
- **Evidence status:** as in section 7 (development data only; macro-F1 interval includes 0, Hit@5 interval does not;
  fold 0 and de→it worse). No change of the final competition score is claimed.

## 9. E11 — smaller window for lower-ranked task A pages (2026-10-12, rejected)

- **Hypothesis:** the best gold page is among the first 4 context pages in 85 % of the cases (D0 reconstruction), so
  clipping only pages 5–8 to 1,500 characters (`PAGE_MAX_CHARS_TAIL=1500`, `PAGE_FULL_RANKS=4`) saves input tokens without
  losing the decisive text. Offline: context −11 %, 78 % of the gold sentences on the clipped pages kept.
- **Protocol:** one run on all 1,488 task A cases against `E7_compact` (the submission configuration). A mock dry run with
  the switch off reproduced the E7_compact NLI inputs in 1,488/1,488 cases; with the switch on, 1,363 cases change.
  Same translation cache (unchanged), 0 API failures. Artefacts: `results/generalization/E11_tail1500`, `E11_analysis`.

| Task A, 1,488 cases | E7_compact (submission) | E11 tail 1500 | Change |
|---|---:|---:|---:|
| Macro-F1 | 0.9493 | 0.9433 | −0.0061 (date bootstrap 95 %: −0.0159 … +0.0042) |
| Evidence Hit@5 (990) | 872 | 852 | −20 (95 %: −0.036 … −0.004) |
| Input tokens / case | 3,915 | 3,586 | −8.4 % |
| Errors | 75 | 84 | 17 fixed / 26 broken; E→C 33 → 35, C→N 13 → 17 |

- Folds: 0 better, 3 equal, 1, 2 and 4 worse (fold 2 −0.014); 8 of 20 voting dates net worse, 5 better. Hit@5 drops
  although evidence is built from whole pages: with less context the model cites other pages and changes labels.
- **Decision: rejected** against the predeclared criteria (F1 loss ≤ 0.003, Hit@5 and E→C not worse). The token saving
  (−8.4 %) does not make up for it (score scenarios negative for a token score ≤ 0.5). The switch stays in the code,
  off by default; the submission is unchanged. Like E1 and E3, it confirms that the model needs the full context of
  the lower-ranked pages as well.

## 10. Holdout check of the final configuration (2026-10-13, task A only)

- **Why:** E8 changed the task A prompt after the holdout run of section 4. The holdout was declared spent there, so
  this run is a **no-regression check**, not a decision: E8 had already been adopted on the dataset (section 8), and
  only a clear loss on the unseen booklets would have reverted it.
- **Protocol:** `scripts/run_decision.py --data .cache/holdout_2026-09-27 --name HO_E8_compact_prompt --task A` at commit
  `fd79d83` (no uncommitted source changes, `A_COMPACT_PROMPT=true` from `.env.example`). The rebuilt holdout cases,
  labels and folds have the same SHA-256 as in section 4. Task B is unchanged by E8 and was not re-run. 0 failed cases.
  Local paths of the run's worktree are replaced by `<worktree>` in `cli.log` and `run.json`.

| Holdout, task A (90 claims) | HO_baseline | HO_E2_reference_6000 | **HO_E8_compact_prompt** |
|---|---:|---:|---:|
| Macro-F1 | 0.933 | 0.944 | **0.933** |
| Evidence Hit@5 | 0.833 | 0.850 | **0.850** |
| Wrong labels | 6 | 5 | 6 |
| Labels fixed / broken by E8 | 2 / 2 | 1 / 2 | – |
| Input / output tokens per case | 3,883 / 72 | 3,863 / 67 | 3,980 / 73 |

- **Reading:** the compact prompt is neutral on the unseen booklets (within the one-label run-to-run variation of the
  two earlier runs with identical settings) at +3 % input tokens, as on the dataset (+2.5 %). **E8 stays active.**
  The configuration is frozen for the submission.

## 11. E12 — contents-guided part selection: the booklet part each claim names (2026-10-13)

- **Origin of the idea:** a competing team's public repository describes choosing the booklet part a claim's opening
  names. The implementation here is our own (`src/part_context.py`, switch `PART_CONTEXT`, first named
  `SECTION_ROUTE`, which the E12 run folders still record); no code was taken over.
- **Finding behind it:** D0 showed that the remaining task A errors are reading errors with the gold page in context;
  the 8 retrieved pages also carry text of other parts and votes. The dataset's claims name their source in the
  opening ("Laut der Zusammenfassung …", "Le comité estime …", "Se il voto viene accettato …"), and every booklet's
  contents page lists, per vote, the pages of summary, detail, arguments and voting text.
- **Part selection:** claim opening → part (summary, detail for consequences of a yes, committee, Federal Council, voting text;
  the earliest cue wins); contents page → page ranges per vote (navigation bars on the summary pages ignored,
  two-page contents merged, start pages checked against the page headings); vote matched by the words of its title,
  also against the first lines of its summary and detail pages; the arguments split by the parser's speaker labels.
  Context: the part's pages in reading order, at most 6,000 characters (best pages by BM25 first when the part is
  longer, the claim translated only then). Claims without a part keep the regular retrieval.
- **Offline (no API):** contents read in 60/60 booklets, a part found for 1,476 of 1,488 claims, the vote for all;
  gold-word coverage of the context 0.956 vs 0.894 for the 8 retrieved pages (cases below 0.8: 19 vs 206).
- **Predeclared bar** (against E7_compact, the submission): macro-F1 ≥ −0.005, Hit@5 ≥ 872, input tokens ≥ −40 %.

| Task A, 1,488 cases | E7_compact (submission) | **E12 part selection** |
|---|---:|---:|
| Macro-F1 | 0.949 | **0.968** (date bootstrap 95 %: +0.005 … +0.035) |
| Macro-F1 de-duplicated / cross-language | 0.953 / 0.949 | 0.965 / 0.971 |
| Evidence Hit@5 (990) | 872 | **965** (110 fixed / 17 broken) |
| Input / output tokens per case | 3,915 / 60 | **1,750 / 59** (−55 %) |
| Labels fixed / broken | – | 56 / 29 (noise floor of an unchanged rerun: 11 / 22) |
| Holdout (new booklets, 90; a part for all) | 0.933 / Hit@5 0.850 / 3,980 tokens | **0.956 / 0.933 / 1,751** (6 → 4 wrong) |

- Folds: 4 of 5 better (fold 3 +0.059, fold 4 +0.040; fold 2 −0.007). Language pairs: 8 of 9 better, de→de −0.021.
- **Decision: passes every criterion.** Selecting the part the claim names gives the model less distraction and the
  evidence the gold part, at less than half the tokens — the clearest single gain of the project.

## 12. E13 — the system prompt's rules in fewer words (2026-10-13, rejected)

- **Hypothesis:** with the part context of E12 the fixed prompt (~700 tokens) is the largest token item; the same rules,
  labels and JSON schema in 1,766 instead of 2,618 characters (`PROMPT_LEAN`) save ~210 tokens per case in both tasks.
- **Runs:** task A with the part context (E12 settings) against E12, task B against E2_reference_6000 (the submission), all 1,488
  cases each, 0 failed cases.

| | Macro-F1 | Hit@5 | Labels fixed / broken | Input tokens |
|---|---:|---:|---:|---:|
| Task A: E12 → E13 | 0.968 → 0.952 | 0.975 → 0.986 | 16 / 40 | 1,750 → 1,533 |
| Task B: E2 → E13 | 0.984 → ~0.94 (23 → 91 wrong) | – | 4 / 72 | 1,732 → 1,515 |

- The new errors turn towards contradiction (task A 0→2: 26, 1→2: 13; task B 0→2: 45, 1→2: 25), the same failure as
  the earlier `PROMPT_SHORT`. Apertus depends on the exact wording of the full checklist.
- **Decision: rejected** (bar: F1 ≥ −0.005). The full prompt stays; the switch stays off.

## 13. E14 — a smaller part budget (2026-10-13, rejected) and adoption of E12

- **Hypothesis:** with `PART_CONTEXT_CHARS=4000` instead of 6,000 the long parts (detail, voting text) keep their best
  pages only, for ~120 fewer input tokens per case (offline: gold-word coverage 0.910 → 0.873).
- **Result** (all 1,488 task A cases against E12): macro-F1 0.968 → 0.948, Hit@5 0.975 → 0.940, labels 7 fixed /
  37 broken, input tokens 1,750 → 1,636 (−6.5 %). **Rejected:** the model needs the whole part.
- **Adopted: E12** (`PART_CONTEXT=true`, `PART_CONTEXT_CHARS=6000`) is now the default in `src/config.py` and both
  `.env.example` files, so the submission image runs it. The full system prompt stays (E13).

## 14. Before / after in detail: submission of 13 October vs. part selection (2026-10-13)

Before = `main` at the merge of PR #10 (E7/E8, 8 retrieved pages; run `E7_compact`); after = E12 (`E12_part_context`).
Task B is unchanged (E2, run `E2_reference_6000`). All numbers from the official scorer on the official CLI's output.

| Task A, all 1,488 pairs | Before | After |
|---|---:|---:|
| Macro-F1 (de-duplicated / cross-language) | 0.949 (0.953 / 0.949) | **0.968** (0.965 / 0.971) |
| F1 entailment / neutral / contradiction | 0.939 / 0.969 / 0.940 | 0.975 / 0.972 / 0.956 |
| Evidence Hit@5 | 0.881 (872 / 990) | **0.975** (965 / 990) |
| Wrong labels | 75 | **48** |
| Input tokens mean / median / p95 | 3,915 / 3,848 / 5,223 | **1,750** / 1,580 / 2,531 |
| Output tokens | 60 | 59 |
| Apertus time mean / p95 | 1.6 s / 2.8 s | 1.0 s / 1.2 s |
| Time outside model calls (CLI estimate) | 2.5 ms | 1.7 ms |

| Wrong labels by claim type (task A) | Before | After |
|---|---:|---:|
| Consequences of a yes (240) | 31 | 12 |
| Committee (275) | 17 | 3 |
| Summary (299) | 12 | 10 |
| Federal Council (438) | 3 | 1 |
| Voting text (214) | 10 | **22** |

The voting-text claims are the new weak spot: long legal texts exceed the 6,000-character budget (E14 shows that a
smaller budget hurts everywhere, so a better selection inside long voting texts is the next step, not a cut).

| Task A by fold (F1 / Hit@5) | Before | After |
|---|---:|---:|
| 0 | 0.971 / 0.908 | 0.975 / 0.967 |
| 1 | 0.950 / 0.879 | 0.954 / 0.953 |
| 2 | 0.980 / 0.904 | 0.973 / 0.973 |
| 3 | 0.919 / 0.874 | 0.978 / 0.995 |
| 4 | 0.919 / 0.838 | 0.958 / 0.985 |

| Task B (unchanged), all 1,488 pairs | B0 (earlier submission) | **E2 (submission)** | E13 lean prompt (rejected) |
|---|---:|---:|---:|
| Macro-F1 (cross-language) | 0.981 (0.979) | **0.984** (0.984) | 0.940 (0.950) |
| F1 entailment / neutral / contradiction | 0.989 / 0.983 / 0.972 | 0.987 / 0.990 / 0.977 | 0.939 / 0.963 / 0.916 |
| Wrong labels (C→N, E→C, C→E, N→C) | 28 (11, 5, 6, 6) | 23 (9, 7, 6, 1) | 91 (8, 52, 3, 26) |
| Input tokens mean / p95 | 2,482 / 5,023 | 1,732 / 2,349 | 1,515 / 2,130 |
| Folds | 0.976–0.985 | 0.968–0.990 | 0.894–0.965 |
| Holdout | 0.955 | 0.966 | – |

**Image check** (`fact-attack:part`, built from this branch): 60/60 booklets pre-parsed, 160 tests pass in the
container with `--network none --read-only`; jury-style run on 27 mixed cases against the real endpoint: exit 0, 27/27
records, every task A label identical to the E12 runs (the sample happens to contain 3 of E12's 29 broken cases, so its
15-case F1 is 0.77; both minimum thresholds pass), 1,808 input tokens per case.

## 15. E15 / E16 — fewer task B tokens (2026-10-13, both rejected)

Tokens are scored relative to the best team, and task B's ~1,794 input tokens per case were the largest open lever. Bar: fewer tokens **and no loss of macro-F1** (all 1,488 task B cases, against E2).

| Task B, 1,488 cases | E2 (submission) | E15: references ≤ 3,000 chars | E16: label-only answer |
|---|---:|---:|---:|
| Macro-F1 | **0.9845** | 0.9805 | 0.8705 |
| Wrong labels (fixed / broken) | 23 | 29 (7 / 13) | 188 (12 / 177; 172 of them N→C) |
| Input / output tokens per case | 1,732 / 62 | 1,375 / 70 | 1,578 / 15 |

- E15 cuts long references to their best-matching passages at 3,000 instead of 6,000 characters: −21 % input tokens,
  but −0.004 macro-F1. Rejected (F1 loss).
- E16 (`B_LABEL_ONLY`, off) keeps the prompt and drops only the confidences and passage ids from the answer schema.
  Without having to state `p_neutral`, Apertus labels unrelated references as contradiction: the confidences act as a
  small reasoning scaffold, not just as output. Rejected.
- With E13 (shorter rules) this closes the task B token question: with Apertus 70B the full prompt, the confidences
  and 6,000-character references are all needed for the 0.984.

## 16. Processing time and robustness (2026-10-13)

- **Page-by-page reading of new booklets (`QUICK_PARTS=true`, `src/quick_booklet.py`).** A booklet that is not parsed
  yet costs 4.5–7.7 s to parse in full, and the first one is not hidden behind any request (local time is scored). With
  part selection a case needs only the contents (PyMuPDF text, fast), the headings and the 2–6 pages of its part (pypdf,
  the model's text). Those are read on their own while the full parse runs on in the background; later cases and
  claims without a part use the full parse. **Equality check** (no API) over all 1,488 dataset and 90 holdout task A
  cases, full parse against page-by-page reading: 1,566 identical in passages, page texts, part labels and evidence
  source pages, 0 different, 12 without a part on both paths. A booklet once opened serves further cases in 2 ms (median).
- **Lazy imports** in the CLI (`rich`, `hf_dataset`): not needed by the official batch run.
- **Model fallback.** The endpoint answers a model it does not serve with an error (CSCS: an empty HTTP 403, like a
  refused key); without a fallback every case would come back neutral and the submission would fall below the
  thresholds. Now the client reads `/models` once and switches to the best Apertus v1.5 model listed (70B-thinking,
  70B, 8B-thinking, 8B); a listed model or an unreadable list leaves the error as it is. `MODEL` is accepted as an alias
  of `LLM_NAME`. Checked live: a wrong model name switches to Apertus-v1.5-70B-thinking and answers normally.
- **Measured in the image** (holdout, 3 new booklets, real endpoint through a token-counting proxy, local time = wall
  clock of `docker run` minus time with a request in flight, two alternating runs each): task A 79 / 75 → 72 / 70 ms per
  case, task B 31 / 31 → 31 / 30 ms; F1, Hit@5 and tokens identical. The gain is small here because Docker Desktop
  gives the container 16 CPUs, so the parse pool reads the first booklet in ~3 s anyway (first request after 4.4 s, of
  which ~1.5 s container start); with fewer CPUs the page-by-page reading saves more.

## 17. E17 — a tiered reference budget for task B (2026-10-13)

- **Observation** (E2 against E15, per reference length): all of E15's F1 loss sits in references above 9,000
  characters (524 of 1,488 cases, 498 of them neutral): 7 → 17 wrong labels, 12 of them neutral → contradiction. Up to
  9,000 characters the 3,000-character cut was better than 6,000 (16 → 12 wrong). Computed offline from the two runs
  before any new request: 0.9872, 19 wrong, 1,640 tokens.
- **Switch:** `REFERENCE_MID_CHARS` (budget of references up to `REFERENCE_LONG_CHARS`, default 9,000; 0 = off) next to
  `REFERENCE_MAX_CHARS` (longer references). Off by default; the test is in `tests/test_reference_focus.py`.
- **Runs:** all 1,488 task B cases, official CLI and scorer, 0 failed cases, against E2 (`E2_reference_6000`).

| Task B, 1,488 cases | E2 (submission) | E17a: ≤ 9,000 → 3,000, longer 6,000 | E17b: ≤ 9,000 → 3,000, longer 4,500 |
|---|---:|---:|---:|
| Macro-F1 | 0.9845 | **0.9872** | 0.9818 |
| Wrong labels | 23 | **19** (5 fixed / 1 broken) | 27 (2 fixed / 10 broken vs E17a) |
| Input tokens per case | 1,732 | **1,640** (−5 %) | 1,507 (−13 %) |
| References > 9,000 characters (524): wrong / tokens | 7 / 2,228 | 7 / 2,227 | 15 / 1,848 |

- E17a reproduces the offline computation exactly (0.9872, 19, 1,640). The cut of the long references is what costs F1
  (E15 at 3,000, E17b at 4,500); the short and middle references do not need more than 3,000 characters.
- **Holdout** (27 Sep 2026 booklets, 90 task B claims, `HO_E17a_mid3000` against `HO_E2_reference_6000`): macro-F1 0.966 → 0.966,
  3 wrong labels both, input tokens 1,758 → 1,739.
- **Decision: E17a adopted** (`REFERENCE_MID_CHARS=3000` is the default in `src/config.py` and both `.env.example`
  files); E17b is rejected. The long references keep 6,000 characters. The gain is 92 tokens per case on the dataset,
  so most of the token lever stays open.
- `groups.json` of E17b is missing: the fold report crashed on a Windows paging-file error after scoring (the
  predictions and `score.official.json` are complete).

## 18. E18 / E19 — a short label-only prompt with a cascade for task B (2026-10-13)

Task B costs ~1,640 input tokens per case, and the score is
S_B = 70 % macro-F1 + 15 % time + 15 % tokens (best ÷ own). Our prompt is ~620 fixed tokens (regression of input
tokens on reference length: 621 + 0.26 · characters) and the answer carries confidences and passage ids, which task B
does not score. E13 (the same checklist in fewer words) and E16 (label only inside JSON) lost F1; this section tests a
different prompt structure (`B_PROMPTS` in `src/apertus_client.py`: decide in order, subject first, then relation;
about 190 tokens; the answer is the digit alone, passages unnumbered).

**E18 — the short prompt alone** (all 1,488 task B cases, E17a as the reference):

| Prompt | Macro-F1 | Wrong | Tokens in + out | Main error |
|---|---:|---:|---:|---|
| E17a (full prompt, 3,000/6,000 chars) | 0.9872 | 19 | 1,708 | – |
| `topic`: order subject → relation | 0.9629 | 55 | 1,186 | contradiction → neutral (36) |
| `topic2`: relation rule widened | 0.9538 | 68 | 1,207 | entailment ↔ contradiction |
| `topic3`: + "do not choose 1 for a negation" | 0.9658 | 51 | 1,232 | neutral / entailment → contradiction |
| `topic4`: 12-word reason, then the digit | 0.9477 | 78 | 1,114 | neutral → contradiction (59) |

The short prompts lose 0.02–0.04 F1 whichever way the rule is worded, so the F1 of E17a is not reachable with a
single short call. Biasing the label probabilities (grid on the log-probability of the digit) recovers at most
+0.008 in-sample. **But 84 % of the short answers are saturated (top probability ≥ 0.999) and 53 of the 57 errors of
`topic` sit in the unsaturated rest** (offline, from the journal of the label probabilities).

**E19 — cascade:** the short call first; when its top probability is below `B_ESCALATE_BELOW` (or unreadable), the
same case is asked again with the full prompt of E17a and that answer counts. Both calls' tokens are counted. The first
call may see fewer reference characters (`B_SHORT_CHARS`), because a less sure answer escalates anyway.

| Task B, 1,488 cases | E17a | E19a: < 0.98, same passages | E19c: < 0.98, first call ≤ 2,000 chars | E19d: first call ≤ 2,500 |
|---|---:|---:|---:|---:|
| Macro-F1 | 0.9872 | 0.9858 | **0.9832** | 0.9818 |
| Wrong labels | 19 | 21 | 25 | 27 |
| Tokens in + out per case | 1,708 | 1,386 | **888** (−48 %) | 1,024 |
| Escalated | – | 12.7 % | 9.8 % | – |
| Extra wrong labels per fold (5 folds) | – | 0 1 1 0 0 | 1 1 2 1 1 | 0 6 0 0 2 |

Holdout (27 Sep 2026 booklets, 90 balanced task B claims, new booklets):

| | E17a | E19a | E19c | E19d |
|---|---:|---:|---:|---:|
| Macro-F1 / wrong labels | 0.966 / 3 | 0.955 / 4 | 0.933 / 6 | 0.944 / 5 |
| Tokens in + out | 1,810 | 1,963 | 1,413 | 1,609 |

- On the unseen booklets the first call is less sure, so the cascade without a cut (E19a) escalates more and saves
  nothing there; with the 2,000-character cut (E19c) it saves 22 % at three more wrong labels (two of them claims
  attributed to the committee whose passage the cut dropped). 90 claims are few: three labels are 0.034 F1.
- A rule "escalate when the claim is attributed to a side and the reference was cut" would fix the three, but 83 % of
  the claims are attributed, so the saving would be gone (2,012 tokens on dev).
- **Estimated effect on S_B** assuming a best-team value of ~820 tokens (an assumption, the real value is unknown): dev +0.064
  (F1 −0.004 · 0.70, token score 0.48 → 0.93), holdout −0.004, pooled over both +0.06. With a lower best value the
  token gain shrinks proportionally but stays positive. The F1 risk is real: a loss like the holdout's on the private
  set would cost 0.024 of S_B, which the token gain more than covers.
- **Run to run:** the adopted defaults without any `--set` (`E20_default_cascade`, `HO_E20_default_cascade`) give
  macro-F1 0.9818 / 27 wrong at 886 tokens on the dataset and 4 wrong at 1,429 tokens on the holdout; E19c was
  0.9832 / 25 and 6. The same configuration differs by about two labels between runs.
- **Stricter threshold (E21, `B_ESCALATE_BELOW=0.999`):**

| | E17a | E19c (0.98, 2,000) | E21a (0.999, 2,000) | E21b (0.999, 2,500) |
|---|---:|---:|---:|---:|
| Dataset: macro-F1 / wrong | 0.9872 / 19 | 0.9832 / 25 | 0.9858 / 21 | **0.9872 / 19** (0 fixed, 0 broken) |
| Dataset: tokens | 1,708 | 888 | 1,126 | 1,270 (−26 %) |
| Holdout: wrong of 90 | 3 | 6 (4 in the default run) | 4 | 4 |
| Holdout: tokens | 1,810 | 1,413 | 1,901 (+5 %) | 2,139 (+18 %) |

  On the dataset E21b is the one configuration that keeps the labels of E17a exactly and still saves tokens. On the
  unseen booklets of the holdout the first call is less sure, so the stricter threshold escalates far more cases and
  the cascade costs more than E17a there. The cascade is therefore a trade of tokens against a small F1 risk, not a
  free gain: no configuration here is better on both sets. The token and time scores are relative to the other
  teams and cannot be computed locally (the starter scorer says so), so the effect on S_B is only estimated.
- **Escalation rules compared offline (E22):** the first call alone was run with the cut on all 1,488 cases
  (`E22_cheap_cut1500/2000/2500`, with the label probabilities), and the escalated cases take the label and tokens of
  E17a (a simulation, not a run). Searching a threshold per predicted label for the lowest tokens at the F1 of E17a:

| First call sees | Cheap only: F1 / tokens | Uniform threshold at F1 of E17a | Per-label threshold (0 / 1 / 2) at F1 of E17a |
|---|---:|---:|---:|
| 1,500 chars | 0.9537 / 624 | 0.9999 → 1,380 tokens | 0.999 / 0.999 / 0.9999 → **1,110 tokens (−35 %)** |
| 2,000 chars | 0.9650 / 732 | none (always escalates) | 0.95 / 0.999 / never → 1,386 tokens |
| 2,500 chars | 0.9530 / 832 | 0.999 → 1,274 tokens | 0.5 / 0.999 / 0.99 → 1,175 tokens |

  Chosen on four folds and applied to the fifth, the per-label thresholds of the 1,500 case held the F1 of E17a in
  four folds (−520 to −730 tokens per case); in the fifth the search selected different thresholds and lost 0.018. So
  the saving at the F1 of E17a is about 20–35 %, and a threshold tuned on this data is not a guarantee for new
  booklets. These per-label thresholds are not in the code (`B_ESCALATE_BELOW` is one number).
- **Decision: E19c adopted** (`B_PROMPT=topic`, `B_ESCALATE_BELOW=0.98`, `B_SHORT_CHARS=2000`, defaults in
  `src/config.py` and both `.env.example` files). An endpoint without logprobs makes every case escalate (the full
  prompt again, so the labels stay those of E17a at ~2.1× the tokens); a refused `logprobs` parameter is retried
  without it. `B_PROMPT=""` restores E17a exactly.

## 19. E23 — a larger budget for the voting-text part of task A (2026-10-14)

- **Error analysis first (offline, no API):** every task A context of E12 was rebuilt (part selection, else the
  hybrid retrieval; 6 of 1,488 cases cite a page outside the rebuilt context) and the gold passage mapped to booklet
  pages. The gold page was in the context in 985 of 990 entailment/contradiction cases and in all 60 on the holdout.
  Of the 48 dataset label errors only 1 had its gold page outside the context (25 had the gold text in full, 8 were
  neutral gold, 14 had the gold page in context but cut). So denser retrieval (e.g. an embedding model) could fix at
  most ~1 label and 5 Hit@5 misses, at the cost of image size and local CPU time: not pursued.
- **What the cut cases have in common:** 12 of the 14 are claims about the voting text, 10 of them cross-lingual. A
  voting text often spans two dense legal pages; the first one alone nearly fills the shared 6,000-character part
  budget, and the article the claim names, on the second page, is clipped away (holdout cases 058 and 064: pages
  30–31, only page 30 kept).
- **Offline comparison of selections inside an over-budget part** (260 dataset parts over 6,000 characters; gold
  words kept ≥ 0.6 counts as "in context"):

| Selection inside the part | Gold E/C in context (986) | Cut failures restored (18) | Newly lost | Mean context chars |
|---|---:|---:|---:|---:|
| E12: whole pages, BM25 + 2 × vote title, 6,000 | 954 | – | – | 3,613 |
| No vote-title weight inside the part | 958 | 0 | 1 | 3,612 |
| ~1,000-character pieces instead of pages | 956 | 2 | 4 | 3,602 |
| **Voting text 9,000 characters, other parts 6,000** | **964** | **8** | **0** | 3,766 |
| 9,000 + no vote-title weight | 968 | 8 | 1 | 3,767 |
| ~1,000-character pieces, 5,000 for all parts | 934 | 2 | 23 | 3,397 |

- **Paired real runs on the changed cases only** (voting-text part over 6,000 characters; same translation cache;
  `E23_legal_base` / `E23_legal9000`, `HO_E23_legal_base` / `HO_E23_legal9000`): dataset 92 cases 16 → 7 wrong
  (9 fixed / 0 broken; the base run repeats E12's labels exactly), holdout 8 cases 3 → 0 wrong (3 / 0).
- **Full runs with the new default** (`E23_legal_text`, `HO_E23_legal_text`, commit `115ca49`, no overrides, 0 failed
  cases, 0 API failures):

| Task A | E12 (submission before) | **E23** |
|---|---:|---:|
| Macro-F1, 1,488 cases | 0.968 | **0.974** (date bootstrap 95 %: +0.002 … +0.012) |
| Macro-F1 de-duplicated / cross-language | 0.965 / 0.971 | 0.968 / 0.981 |
| Evidence Hit@5 (990) | 0.975 | **0.983** (8 fixed / 0 broken) |
| Labels fixed / broken | – | 12 / 2 (see below) |
| Input / output tokens per case | 1,750 / 59 | 1,791 / 59 (+2.3 %; p95 2,531 → 3,038) |
| Apertus time mean / p95 | 1.0 s / 1.2 s | 1.1 s / 1.4 s |
| Claims about the voting text: wrong of 235 | 24 | **13** |
| Holdout (90): macro-F1 / Hit@5 / tokens | 0.956 / 0.933 / 1,751 | **0.989 / 0.983 / 1,825** (4 → 1 wrong) |

- Of the 14 label changes against E12, 10 are in the 92 changed cases (9 fixed, 1 broken: row 923, which the paired
  run had kept right) and 4 in cases the switch does not touch (3 fixed, 1 broken) — the run-to-run noise of an
  unchanged system (3 of 1,488). Folds: 5 of 5 better or equal (+0.000 … +0.011); language pairs: 8 of 9, it→it
  −0.012 (two labels).
- **The holdout is not an independent confirmation here:** its cases 058 and 064 were among those that suggested the
  idea. The dataset gain (+0.007, interval above zero) is the number to rely on.
- **Decision: adopted** (`PART_CONTEXT_CHARS_LEGAL=9000`, default in `src/config.py` and both `.env.example` files;
  `0` restores E12 exactly). Only the voting-text part changes; summary, detail, arguments and the fallback retrieval
  are untouched.

## 20. E24 — robustness of the part selection to other claim wordings (2026-10-14, offline)

- **Question:** the part selection (E12) recognises the source a claim names by a fixed list of words in its first
  80 characters, written for the dataset's templates ("Das Komitee vertritt die Auffassung, …", "Si le vote est
  approuvé, …", "Il comitato afferma che …"). How does it behave if the jury's claims are worded differently?
- **Stress test** (`results/generalization/D1_context_rebuild/stress_routing.py`, no API): in 1,400 dataset and 88
  holdout claims the opening that names the source was replaced — SYN: other words for the same source; SYN_B: a
  second set of other words, used only to measure, never to extend the list; LATE: the source at the end ("…, so das
  Komitee."); NONE: no source. Every variant was routed again; the fallback context was rebuilt without the claim's
  translation (conservative).
- **Finding:** with the cue list of E23 almost every rephrased claim falls back to the hybrid retrieval (gold page in
  context 84 % instead of 99 %, the level of the submission before E12), but ~3 % went to the **wrong part**, nearly
  always the Federal Council's, because the claim's content mentions it ("L'iniziativa è stata respinta dal Consiglio
  federale …"); there the gold page was present in 1 of 38 cases. In all 1,578 dataset and holdout claims the source
  starts by character 20.
- **Change:** a cue counts only if it starts within the first 30 characters (`_CUE_START`), and the list has
  unambiguous other wordings in DE / FR / IT (Befürworter / partisans / sostenitori, Landesregierung / gouvernement /
  governo, Kurzübersicht / aperçu / sintesi, Wortlaut / libellé / articolo costituzionale, "Bei einem Ja" / "en cas de
  oui" / "con un sì").
- **Known cases unchanged:** all 3,173 claims of the dataset (tasks A and B), the holdout, its edge cases, the contract
  and demo cases get exactly the same part as before, so every measured score stays as it is and no new run is needed.

| Dataset, 1,400 rephrased claims (wrong part / gold page in context, E/C 921) | E23 cue list | **E24** |
|---|---:|---:|
| SYN (these wordings are now in the list: coverage, not a test) | 35 / 777 | 0 / 916 |
| **SYN_B (wordings never used for the list)** | 38 / 775 | **6 / 798** |
| LATE (source at the end) | 44 / 833 | 34 / 831 |
| NONE (no source) | 44 / 778 | 34 / 781 |

- Holdout (88): wrong parts 0–1 → 0 in every variant. LATE claims whose source was within 80 characters (96) now
  use the fallback instead of the right part; their gold page stays in context (833 → 831). The remaining wrong parts
  under NONE are claims whose content starts with an actor ("Der Bundesrat und das Parlament lehnen … ab"), which no
  word list can tell apart.
- **Decision: adopted.** It cannot change a known case and turns most wrong-part failures on unfamiliar wordings into
  the fallback retrieval (~0.95 macro-F1 in E8). Not done: adding the
  1–2 best retrieved pages outside the part as a safety net (needs a paired run; E12 showed that extra pages distract).
