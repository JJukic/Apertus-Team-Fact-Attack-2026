"""
Unit tests for PDFParser and Dynamic Proposal Boundary Detection.
"""

import unittest
from pathlib import Path
from src.pdf_parser import PDFParser
from src import config


class TestPDFParser(unittest.TestCase):
    def setUp(self):
        self.parser = PDFParser()
        self.booklet_de = config.BOOKLETS_DIR / "2026-06-14_de.pdf"

    def test_extract_pages(self):
        if not self.booklet_de.exists():
            self.skipTest(f"Booklet not found at {self.booklet_de}")

        pages = self.parser.extract_pages(self.booklet_de)
        self.assertGreater(len(pages), 0)
        self.assertIn("page_number", pages[0])
        self.assertIn("text", pages[0])

    def test_extract_full_text(self):
        if not self.booklet_de.exists():
            self.skipTest(f"Booklet not found at {self.booklet_de}")

        full_text = self.parser.extract_full_text(self.booklet_de)
        self.assertIsInstance(full_text, str)
        self.assertGreater(len(full_text), 500)

    def test_detect_proposal_boundaries(self):
        if not self.booklet_de.exists():
            self.skipTest(f"Booklet not found at {self.booklet_de}")

        pages = self.parser.extract_pages(self.booklet_de)
        page_to_prop, start_pages = self.parser._detect_proposal_boundaries(pages)

        # 2026-06-14 has 2 federal proposals
        self.assertGreaterEqual(len(start_pages), 2)
        # Verify page mapping is populated
        self.assertGreater(len(page_to_prop), 0)

    def test_extract_paragraphs_structure(self):
        if not self.booklet_de.exists():
            self.skipTest(f"Booklet not found at {self.booklet_de}")

        paras = self.parser.extract_paragraphs(self.booklet_de)
        self.assertGreater(len(paras), 0)
        first = paras[0]
        self.assertIn("text", first)
        self.assertIn("page_number", first)
        self.assertIn("proposal_id", first)


if __name__ == "__main__":
    unittest.main()
