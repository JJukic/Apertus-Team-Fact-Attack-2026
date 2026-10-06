# Offline retrieval measurement

Dev cases: 1090. Missing PDFs: 0. Reference not located: 0.
Dataset SHA256: `46e4fcabdf6df77020f193cc29ba112329a647abac20898edc025ae692a19438`.

This is a lexical reference-availability proxy. A hit means at least one normalized, consecutive 12-word reference fragment occurs in selected text. It does not establish semantic support, full reference coverage, or correct classification. Boilerplate/repeated text can inflate hits. Unlocated references are excluded from rates and listed in JSON.
Whitespace, case, soft hyphens and letter-to-letter PDF hyphenation are normalized. No model calls. Selected context is clipped with the same query and limit as the pipeline.

| k | Speaker boost | Eligible | Hit before clipping | Hit after clipping | Mean anchor coverage | Cases lost to clipping |
|---|---|---|---|---|---|---|
| 5 | 0 | 1090 | 92.4% | 92.4% | 33.9% | 0 |
| 5 | 2 | 1090 | 95.4% | 95.4% | 52.6% | 0 |
| 12 | 0 | 1090 | 99.5% | 99.5% | 70.7% | 0 |
| 12 | 2 | 1090 | 99.7% | 99.7% | 76.9% | 0 |
| 20 | 0 | 1090 | 99.7% | 99.7% | 88.3% | 0 |
| 20 | 2 | 1090 | 99.7% | 99.7% | 89.7% | 0 |

The public dataset now contains 1,488 total rows (dev 1,090/test 398), whereas historical benchmark documentation describes 1,495 (test 402). Do not compare these rates directly with historical F1. Case ids derived from row indices must be interpreted with this dataset hash.

These methods share retrieval before extraction: this measurement does not distinguish baseline NLI from the two-stage judge. It tests how much source reference text is available upstream. A filter cannot recover omitted text.

Reproduce from repository root: `.venv/bin/python track_2a/scripts/measure_retrieval.py`. Downloaded dev PDFs and splits are local, ignored data. JSON includes file hashes and per-case selected pages.
