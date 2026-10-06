# Local semantic retrieval in the app

Choose **Erweitert: Kontext-Strategie & eigenes PDF → Semantische Suche + Vorlagentitel**.
Enter a claim, select its vote title, and click **Suchtreffer lokal ansehen**.
This performs PDF parsing and local E5 retrieval only, without an Apertus call or CSCS key.
The displayed pages are candidates, not verified support. Changing input invalidates the preview.

This is the measured `dense_title` recipe, immutable E5 revision, overlapping token
windows, maximum chunk similarity per passage, normalized claim + 2 × title scores,
current speaker boost and query-aware clipping. Without a vote title it reduces to
claim-only dense retrieval, which was weaker in our dev experiment.

For live classification, the same selected context is passed to whichever judgment
method is selected. Cached BM25 demo answers are never shown for semantic retrieval.
Local dependencies: `pip install -r requirements-dense-experiment.txt` (optional).
Weights/vectors are stored under `.cache/e5`. The local model is shared across app
sessions; access is serialized. Failures are explicit, with no silent BM25 fallback.
CLI predict/run/benchmark accept `--strategy dense_title`. Model/revision and local
retrieval times are recorded in benchmark reports; model latency stays separate.

The paired diagnostics at 12 pages show reference-text coverage increases in 312
cases and decreases in 256 of 1,090. Average changes differ across language pairs:
it→de +11.32 percentage points, fr→de +6.94, but fr→it −1.60 and it→it −0.17.
These are lexical reference-fragment measurements, not NLI accuracy or demonstrated
causes. Keep BM25 available and compare live judgments before replacing the default.
