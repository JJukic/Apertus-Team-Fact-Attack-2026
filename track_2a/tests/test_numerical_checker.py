"""
Unit tests for deterministic numerical conflict detection and evidence source attribution.
"""

import unittest
from src.numerical_checker import extract_numbers, detect_numerical_conflict, NumberEntity
from src.inference import ClaimVerificationEngine, EvidenceSource, PredictionResult
from src.apertus_client import ApertusClient
from src import config


class TestNumericalChecker(unittest.TestCase):
    def test_extract_numbers_swiss_formats(self):
        text_de = "Seit 2002 wuchs die Bevölkerung um rund 500'000 Personen, bis 2050 maximal 10 Millionen."
        entities = extract_numbers(text_de)
        
        # Should extract 2002 (year), 500000 (quantity), 2050 (year), 10000000 (quantity)
        values = [e.value for e in entities]
        self.assertIn(2002, values)
        self.assertIn(500000.0, values)
        self.assertIn(2050, values)
        self.assertIn(10000000.0, values)

        # Check types
        year_entities = [e for e in entities if e.num_type == "year"]
        qty_entities = [e for e in entities if e.num_type == "quantity"]
        self.assertEqual(len(year_entities), 2)
        self.assertEqual(len(qty_entities), 2)

    def test_extract_numbers_multilingual(self):
        text_fr = "La population a augmenté de 1,7 million de personnes avant 2050."
        entities_fr = extract_numbers(text_fr)
        fr_vals = [e.value for e in entities_fr]
        self.assertIn(1700000.0, fr_vals)
        self.assertIn(2050, fr_vals)

        text_it = "Non superare i dieci milioni di abitanti prima del 2050."
        entities_it = extract_numbers(text_it)
        it_vals = [e.value for e in entities_it]
        self.assertIn(10000000.0, it_vals)
        self.assertIn(2050, it_vals)

    def test_detect_conflict_quantity_mismatch(self):
        claim = "Gemäss dem Abstimmungstext ist die Bevölkerung seit 2002 um rund 500'000 Personen gewachsen."
        context = "Seit 2002 ist die ständige Wohnbevölkerung der Schweiz um 1,7 Millionen Personen gewachsen."
        conflict = detect_numerical_conflict(claim, context)
        
        self.assertIsNotNone(conflict)
        self.assertIn("500'000", conflict)
        self.assertIn("1,7 Millionen", conflict)

    def test_detect_conflict_year_mismatch(self):
        claim = "Die Grenze von 10 Millionen darf bereits vor 2030 nicht überschritten werden."
        context = "Vor dem Jahr 2050 darf die ständige Wohnbevölkerung 10 Millionen Menschen nicht überschreiten."
        conflict = detect_numerical_conflict(claim, context)
        
        self.assertIsNotNone(conflict)
        self.assertIn("2030", conflict)
        self.assertIn("2050", conflict)

    def test_no_false_positive_when_numbers_agree(self):
        claim = "Vor dem Jahr 2050 darf die ständige Wohnbevölkerung 10 Millionen Menschen nicht überschreiten."
        context = "Die Initiative verlangt: Vor 2050 darf die ständige Wohnbevölkerung 10 Millionen Menschen nicht überschreiten."
        conflict = detect_numerical_conflict(claim, context)
        self.assertIsNone(conflict)

    def test_no_cross_type_false_positive(self):
        # A year (2050) should never clash with a quantity (10 Millionen) even if nearby
        claim = "Die Zahl beträgt 10 Millionen Menschen vor 2050."
        context = "Vor 2050 soll die Grenze von 10 Millionen gelten."
        conflict = detect_numerical_conflict(claim, context)
        self.assertIsNone(conflict)


class TestEvidenceAttribution(unittest.TestCase):
    def test_evidence_source_page_mapping(self):
        booklet = config.BOOKLETS_DIR / "2026-06-14_de.pdf"
        if not booklet.exists():
            self.skipTest(f"Booklet not found at {booklet}")

        engine = ClaimVerificationEngine(apertus_client=ApertusClient(mock=True))
        claim = "Gemäss dem Abstimmungstext ist die Bevölkerung seit 2002 um rund 500'000 Personen gewachsen."
        
        result = engine.verify_claim(
            claim=claim,
            booklet_pdf=booklet,
            claim_language="de",
            strategy="retrieval",
            top_k=3,
        )

        self.assertIsInstance(result, PredictionResult)
        self.assertTrue(hasattr(result, "evidence_sources"))
        self.assertIsInstance(result.evidence_sources, list)
        self.assertGreater(len(result.evidence_sources), 0)

        first_src = result.evidence_sources[0]
        self.assertIsInstance(first_src, EvidenceSource)
        self.assertIsNotNone(first_src.quote)
        # Verify page number is populated
        self.assertIsNotNone(first_src.page_number)
        self.assertGreater(first_src.page_number, 0)


if __name__ == "__main__":
    unittest.main()
