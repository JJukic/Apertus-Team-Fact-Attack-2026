"""
Unit tests for end-to-end ClaimVerificationEngine and BenchmarkEvaluator.
"""

import unittest
from src.inference import ClaimVerificationEngine, PredictionResult
from src.evaluator import BenchmarkEvaluator
from src.apertus_client import ApertusClient
from src import config


class TestPipeline(unittest.TestCase):
    def setUp(self):
        # Force mock mode for pipeline tests
        self.mock_client = ApertusClient(mock=True)
        self.engine = ClaimVerificationEngine(apertus_client=self.mock_client)
        self.evaluator = BenchmarkEvaluator(engine=self.engine)
        self.booklet = config.BOOKLETS_DIR / "2026-06-14_de.pdf"

    def test_verify_claim_retrieval(self):
        if not self.booklet.exists():
            self.skipTest(f"Booklet not found at {self.booklet}")

        claim = "Die Initiative verlangt, dass die ständige Wohnbevölkerung vor dem Jahr 2050 zehn Millionen Menschen nicht überschreiten darf."
        res = self.engine.verify_claim(
            claim=claim,
            booklet_pdf=self.booklet,
            claim_language="de",
            strategy="retrieval",
            top_k=3,
        )

        self.assertIsInstance(res, PredictionResult)
        self.assertIn(res.label, (0, 1, 2))
        self.assertEqual(res.strategy, "retrieval")
        self.assertGreater(res.tokens_total, 0)
        self.assertGreater(res.latency_ms, 0)

    def test_verify_claim_full_document(self):
        if not self.booklet.exists():
            self.skipTest(f"Booklet not found at {self.booklet}")

        claim = "Gemäss dem Abstimmungstext soll der Zivildienst angepasst werden."
        res = self.engine.verify_claim(
            claim=claim,
            booklet_pdf=self.booklet,
            claim_language="de",
            strategy="full",
        )

        self.assertIsInstance(res, PredictionResult)
        self.assertIn(res.label, (0, 1, 2))
        self.assertEqual(res.strategy, "full")

    def test_evaluator_subset(self):
        if not config.BENCHMARK_PATH.exists() or not self.booklet.exists():
            self.skipTest("Benchmark dataset or booklet missing")

        report = self.evaluator.evaluate(strategy="retrieval", limit=2)
        self.assertIn("macro_f1", report)
        self.assertIn("avg_latency_ms", report)
        self.assertIn("by_language", report)
        self.assertEqual(report["sample_count"], 2)


if __name__ == "__main__":
    unittest.main()
