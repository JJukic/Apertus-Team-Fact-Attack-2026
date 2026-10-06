# Local multilingual retrieval comparison

Model: [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small), immutable revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, device `mps`.
Dev rows: 1090; eligible: 1090; excluded unlocated references: 0.
Dataset SHA256: `46e4fcabdf6df77020f193cc29ba112329a647abac20898edc025ae692a19438`.

All methods use identical parsed pages, page clipping and speaker boost. bm25_title is the current BM25(claim) + 2 × BM25(vote title) baseline. dense_claim uses only semantic claim similarity. dense_title combines independently min-max-normalized semantic claim/title page scores with title weight 2. RRF combines complete BM25/title and dense-claim rankings with constant 60. None uses gold text for ranking.
E5 receives query:/passage: prefixes. Long pages use overlapping 384-token windows, stride 256, within the 512-token model limit. Page score is maximum chunk cosine similarity; more chunks can still create a multiple-comparisons advantage. Queries are truncated at the model limit if exceptionally long.

Hit = at least one shared normalized 12-word reference fragment after clipping. Coverage = fraction of reference fragments located in the full booklet that are also selected. This lexical proxy does not measure semantic evidence validity or NLI accuracy; repeated boilerplate can inflate it.

| Method | k | Reference-fragment hit | Mean anchor coverage | Hits gained vs BM25 | Hits lost vs BM25 |
|---|---|---|---|---|---|
| bm25_title | 5 | 95.4% | 52.6% | 0 | 0 |
| dense_claim | 5 | 83.1% | 49.0% | 34 | 168 |
| dense_title | 5 | 96.3% | 55.4% | 37 | 27 |
| rrf | 5 | 93.5% | 54.9% | 25 | 46 |
| bm25_title | 12 | 99.7% | 76.9% | 0 | 0 |
| dense_claim | 12 | 94.0% | 64.0% | 3 | 65 |
| dense_title | 12 | 100.0% | 80.2% | 3 | 0 |
| rrf | 12 | 99.4% | 72.5% | 3 | 7 |
| bm25_title | 20 | 99.7% | 89.7% | 0 | 0 |
| dense_claim | 20 | 97.6% | 74.7% | 3 | 26 |
| dense_title | 20 | 100.0% | 93.6% | 3 | 0 |
| rrf | 20 | 100.0% | 85.1% | 3 | 0 |

The current public dataset differs from the historical 1,495-case dataset. No comparison to historical F1 is valid. This is not BGE-M3 or a replay of the separate dense branch: only general max-per-page and RRF ideas are shared; the model, chunking and baseline differ.
This run: model load 5.7s; all query embeddings 2.2s (built); wall 53.3s. Document indexing times/cache states are in JSON. Batched timings are not per-request app latency.

Reproduce: install `track_2a/requirements-dense-experiment.txt`, then run `.venv/bin/python track_2a/scripts/compare_local_dense.py`. Requires downloaded dev data/PDFs, no CSCS key. Heavy weights/vectors are ignored under `.cache/e5`. Production retrieval remains unchanged.
