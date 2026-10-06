"""Offline regressions for quote formatting and false acceptance."""
import unittest
import csv
import hashlib
import json
from pathlib import Path

from src.evidence_matcher import match_quote


class EvidenceMatcherTests(unittest.TestCase):
    def test_formatting_preserves_original_span(self):
        original = "Préambule: Les mar-\nchandises sont transportées – par le rail. Fin."
        result = match_quote('[Page 99] Les marchandises sont transportées \\— par le rail.',
                             [{"page_number": 22, "text": original}])
        self.assertEqual(result[0]["page_number"], 22)
        self.assertEqual(result[1], "Les mar-\nchandises sont transportées – par le rail.")

    def test_collapsed_pdf_break_and_soft_hyphen(self):
        for text in ("Le persone dovran- no rispettare le regole.",
                     "Le persone dovran -\nno rispettare le regole.",
                     "Le persone dovran\u00adno rispettare le regole."):
            self.assertIsNotNone(match_quote("Le persone dovranno rispettare le regole.",
                                            [{"page_number": 26, "text": text}]))

    def test_rejects_semantic_changes_and_composites(self):
        text = "Der Bund bezahlt nicht 10 Millionen. Die Kantone bezahlen den Rest."
        for quote in ("Der Bund bezahlt 10 Millionen.",
                      "Der Bund bezahlt nicht 11 Millionen.",
                      "Der Bund bezahlt nicht 10 Millionen. [...] den Rest.",
                      "Der Bund übernimmt keine 10 Millionen.", "", "Der Bund"):
            with self.subTest(quote=quote):
                self.assertIsNone(match_quote(quote, [{"page_number": 4, "text": text}]))

    def test_no_number_range_or_word_boundary_corruption(self):
        for text, quote in (("Die Kosten betragen 10- 20 Millionen.", "Die Kosten betragen 1020 Millionen."),
                            ("Die Kosten betragen 100 Millionen.", "Die Kosten betragen 10"),
                            ("Unverändert bleiben die Kosten.", "verändert bleiben die Kosten.")):
            self.assertIsNone(match_quote(quote, [{"page_number": 4, "text": text}]))

    def test_page_tag_is_only_a_tiebreaker(self):
        text = "Die Kosten bleiben unverändert."
        sources = [{"page_number": p, "text": text} for p in (4, 22)]
        self.assertEqual(match_quote("[Page 22] " + text, sources)[0]["page_number"], 22)
        self.assertEqual(match_quote("[Page 99] " + text, sources)[0]["page_number"], 4)
        self.assertIsNone(match_quote(text, []))

    def test_does_not_stitch_pages(self):
        sources = [{"page_number": 4, "text": "Die Kosten bleiben"},
                   {"page_number": 5, "text": "unverändert bei zehn Millionen."}]
        self.assertIsNone(match_quote("Die Kosten bleiben unverändert bei zehn Millionen.", sources))

    def test_reviewed_cases_against_original_pdfs(self):
        from src.pdf_parser import PDFParser

        root = Path(__file__).resolve().parents[2]
        evaluation = root / "track_2a/docs/evaluation"
        with (evaluation / "hybrid_dense_first_pass_manual_review.csv").open() as f:
            rows = list(csv.DictReader(f))
        hashes = json.loads((evaluation / "hybrid_quality_2026-10-05.json").read_text())["booklet_sha256"]
        pages = {}
        for row in rows:
            path = root / row["booklet_pdf"]
            if not path.exists():
                self.skipTest(f"Historical PDF not available: {path}")
            if path not in pages:
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), hashes[row["booklet_language"]])
                pages[path] = PDFParser().extract_pages(path)
        self.assertEqual(len(rows), 14)
        for row in rows:
            with self.subTest(case=row["case_id"]):
                selected = json.loads(row["selected_pages"])
                sources = [p for p in pages[root / row["booklet_pdf"]] if p["page_number"] in selected]
                matches = [m for ev in json.loads(row["raw_evidence"])
                           if (m := match_quote(ev, sources))]
                # This case contains two quotes: an authentic formatting-only
                # span on p48 and a paraphrased supporting statement. Matching
                # the first is NOT proof that it entails the claim.
                mixed = row["case_id"] == "historical-20"
                self.assertEqual(bool(matches), row["failure_cause"] == "quote_format" or mixed)
                if matches:
                    expected_page = 48 if mixed else int(row["verified_support_page"])
                    self.assertIn(expected_page, [m[0]["page_number"] for m in matches])
                    for source, span in matches:
                        self.assertIn(span, source["text"])
                if mixed:
                    self.assertIsNone(match_quote(json.loads(row["raw_evidence"])[1], sources))


if __name__ == "__main__":
    unittest.main()
