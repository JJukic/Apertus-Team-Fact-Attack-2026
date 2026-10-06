"""
Unit tests for PDFParser and Dynamic Proposal Boundary Detection.
"""

import sys
from pathlib import Path
import unittest

_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from src.pdf_parser import PDFParser, detect_closing_speaker, detect_section_heading
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


COMMITTEE = "Arguments of the initiative/referendum committee"
FEDERAL_COUNCIL = "Arguments of the Federal Council and Parliament"
BODY = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt."


class TestSectionDetection(unittest.TestCase):
    def test_bare_speaker_lines_are_headings(self):
        self.assertEqual(detect_section_heading(f"52 Vierte Vorlage: Medien\nReferendumskomitee\n{BODY}"), COMMITTEE)
        self.assertEqual(detect_section_heading(f"52 Troisième objet\nComité « Non à l’arnaque ! »\n{BODY}"), COMMITTEE)
        self.assertEqual(detect_section_heading(f"26 Secondo oggetto\nComitati referendari\n{BODY}"), COMMITTEE)
        self.assertEqual(detect_section_heading(f"54\nBundesrat und Parlament\n{BODY}"), FEDERAL_COUNCIL)
        self.assertEqual(detect_section_heading(f"28\nConsiglio federale e Parlamento\n{BODY}"), FEDERAL_COUNCIL)

    def test_speaker_in_running_text_is_not_a_heading(self):
        self.assertIsNone(detect_section_heading(f"16\nAnders als die Initiative wollen Bundesrat und Parlament dies\n{BODY}"))

    def test_closing_recommendation_names_the_speaker(self):
        self.assertEqual(detect_closing_speaker(f"{BODY}\nEmpfehlung von \nBundesrat und \nParlament"), FEDERAL_COUNCIL)
        self.assertEqual(detect_closing_speaker(f"{BODY}\nRecommandation \ndes comités \nréférendaires"), COMMITTEE)
        self.assertIsNone(detect_closing_speaker(f"Empfehlung von Bundesrat und Parlament\n{BODY}\n{BODY}\n{BODY}\n{BODY}"))

    def test_closing_box_relabels_its_double_page(self):
        # committee double page (disclaimer on its 2nd page), then a Federal Council double page without heading
        pages = [
            {"page_number": 62, "text": f"62 Vierte Vorlage: Vaterschaftsurlaub\n{BODY}"},
            {"page_number": 63, "text": f"63\n{BODY}\nDer Text auf dieser Doppelseite stammt vom Referendumskomitee."},
            {"page_number": 64, "text": f"64 Vierte Vorlage: Vaterschaftsurlaub\n{BODY}"},
            {"page_number": 65, "text": f"65\n{BODY}\nEmpfehlung von \nBundesrat und \nParlament"},
            {"page_number": 66, "text": f"66\nDie Bundesversammlung der Schweizerischen Eidgenossenschaft\n{BODY}"},
        ]
        paras = PDFParser().extract_paragraphs("unused.pdf", passage_chars=None, pages=pages)
        sections = {p["page_number"]: p["section"] for p in paras}
        self.assertEqual(sections, {62: COMMITTEE, 63: COMMITTEE, 64: FEDERAL_COUNCIL, 65: FEDERAL_COUNCIL, 66: None})


if __name__ == "__main__":
    unittest.main()
