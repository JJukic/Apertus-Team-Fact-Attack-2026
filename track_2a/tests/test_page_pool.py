"""
Parallel page extraction (src/page_pool.py): the same text as the sequential parser, and a sequential fallback where
no pool is started. The pool needs fork (Linux, as in the container and in CI).
"""

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from src import page_pool
from src.pdf_parser import PDFParser

BOOKLET = _pkg_root / "data" / "booklets" / "2026-06-14_de.pdf"


class TestPagePool(unittest.TestCase):
    def tearDown(self):
        page_pool.stop()

    def test_not_queued_means_sequential(self):
        self.assertIsNone(page_pool.extract_pages(BOOKLET))

    def test_single_cpu_starts_no_pool(self):
        with mock.patch.dict(os.environ, {"PARSE_PROCESSES": "1"}):
            self.assertFalse(page_pool.start([BOOKLET]))

    @unittest.skipUnless(hasattr(os, "fork"), "the pool forks (Linux)")
    def test_same_pages_as_sequential(self):
        with mock.patch.dict(os.environ, {"PARSE_PROCESSES": "3"}), mock.patch.object(page_pool, "TASKS_PER_BOOKLET", 2):
            self.assertTrue(page_pool.start([BOOKLET, BOOKLET.with_name("2026-06-14_fr.pdf")]))
            parallel = [page_pool.extract_pages(BOOKLET), page_pool.extract_pages(BOOKLET.with_name("2026-06-14_fr.pdf"))]
        page_pool.stop()
        sequential = [PDFParser().extract_pages(BOOKLET), PDFParser().extract_pages(BOOKLET.with_name("2026-06-14_fr.pdf"))]
        # The same pypdf text; with the default evidence policy the same tasks also read the PyMuPDF evidence pages
        sources = [{p["page_number"]: p.pop("source", None) for p in pages} for pages in parallel]
        self.assertEqual(parallel, sequential)
        self.assertGreater(len(parallel[0]), 10)
        from src import config
        if config.EVIDENCE_POLICY != "legacy":
            from src.evidence import extract_source_pages

            self.assertEqual(sources[0], extract_source_pages(str(BOOKLET), list(sources[0])))


if __name__ == "__main__":
    unittest.main()
