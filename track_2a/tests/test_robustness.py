"""
Inputs the evaluation set may contain: a byte order mark, broken or non-object lines in the case file, and booklet
pages pypdf cannot read (some fonts of the 2018-03-04 booklets). None of them may cost other cases.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.cli import load_cases
from src.pdf_parser import PDFParser, extract_page_text

BOOKLET = Path(__file__).resolve().parent.parent / "data" / "booklets" / "2026-06-14_de.pdf"

try:
    import pypdfium2  # noqa: F401
    HAS_PDFIUM = True
except ImportError:
    HAS_PDFIUM = False


class TestLoadCases(unittest.TestCase):
    def _write(self, text: str) -> Path:
        path = Path(tempfile.mkdtemp()) / "cases.jsonl"
        path.write_text(text, encoding="utf-8")
        return path

    def test_bom_broken_and_non_object_lines(self):
        good = {"id": "a", "claim": {"text": "x"}, "reference": {"text": "y"}}
        lines = ["﻿" + json.dumps(good), "", '{"id": "b-broken", "claim": {"text": "x"', "[1, 2]",
                 '{"no id at all": ', json.dumps({**good, "id": "c"})]
        cases = load_cases(self._write("\n".join(lines)))
        self.assertEqual([c["id"] for c in cases], ["a", "b-broken", "c"])
        self.assertIn("_invalid", cases[1])  # answered as neutral, so the id is not missing in the output

    def test_json_list_with_bom(self):
        cases = load_cases(self._write("﻿" + json.dumps([{"id": "a"}, {"id": "b"}])))
        self.assertEqual([c["id"] for c in cases], ["a", "b"])


class TestPageFallback(unittest.TestCase):
    def _failing_reader(self, index):
        import pypdf

        reader = pypdf.PdfReader(str(BOOKLET))
        broken = reader.pages[index]
        broken.extract_text = mock.Mock(side_effect=pypdf.errors.PdfReadError("More than one /FontFile found"))
        return reader

    @unittest.skipUnless(HAS_PDFIUM, "pypdfium2 not installed")
    def test_failing_page_is_read_with_pdfium(self):
        expected = PDFParser().extract_pages(BOOKLET)[2]["text"]
        text = extract_page_text(self._failing_reader(2), 2, BOOKLET)
        self.assertGreater(len(text), 200)
        self.assertNotIn("\r", text)
        self.assertEqual(text.split()[:5], expected.split()[:5])

    def test_without_fallback_the_page_is_empty_not_the_booklet(self):
        with mock.patch.dict("sys.modules", {"pypdfium2": None}):
            self.assertEqual(extract_page_text(self._failing_reader(2), 2, BOOKLET), "")


if __name__ == "__main__":
    unittest.main()
