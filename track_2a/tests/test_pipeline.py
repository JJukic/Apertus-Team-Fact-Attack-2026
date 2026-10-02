"""
Unit tests for end-to-end ClaimVerificationEngine and BenchmarkEvaluator.
"""

import sys
from pathlib import Path
import unittest

_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

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

    def test_verify_premise_beginner_task(self):
        claim = "Der Bundesrat empfiehlt die Ablehnung."
        reference = "Aus Sicht von Bundesrat und Parlament bringt die Initiative Unsicherheit. Empfehlung: Nein"
        res = self.engine.verify_premise(
            claim=claim,
            reference=reference,
            claim_language="de",
            case_id="case-0043",
        )

        self.assertIsInstance(res, PredictionResult)
        self.assertEqual(res.id, "case-0043")
        self.assertEqual(res.strategy, "direct_reference")
        self.assertIn(res.label, (0, 1, 2))

        official = res.to_official_dict()
        self.assertEqual(official["id"], "case-0043")
        self.assertIn("label", official)
        self.assertIn("label_name", official)
        self.assertIn("evidence", official)
        self.assertIn("metrics", official)
        self.assertIn("input_tokens", official["metrics"])
        self.assertIn("output_tokens", official["metrics"])
        self.assertIn("inference_time_ms", official["metrics"])

    def test_neutral_evidence_is_strictly_empty(self):
        # A claim unrelated to the booklet should yield neutral and have an empty evidence array
        res = PredictionResult(
            id="case-neutral-1",
            claim="Die Astronauten fliegen zum Mars.",
            label=1,
            label_name="neutral",
            reasoning="Not mentioned in the document",
            evidence=["Some non-empty evidence quote that must be cleared"],
            evidence_sources=[],
            strategy="retrieval",
            booklet_path="dummy.pdf",
            tokens_prompt=100,
            tokens_completion=20,
            tokens_total=120,
            latency_ms=45.0,
        )

        official = res.to_official_dict()
        self.assertEqual(official["label"], 1)
        self.assertEqual(official["label_name"], "neutral")
        self.assertEqual(official["evidence"], [], "Evidence list for Neutral (1) must be strictly []")

    def test_verify_claim_with_vote_target(self):
        if not self.booklet.exists():
            self.skipTest(f"Booklet not found at {self.booklet}")

        claim = "Die Vorlage begrenzt die Zuwanderung vor 2050."
        res = self.engine.verify_claim(
            claim=claim,
            booklet_pdf=self.booklet,
            vote="Nachhaltigkeitsinitiative",
            strategy="retrieval",
            top_k=3,
            case_id="case-0099",
        )

        self.assertEqual(res.id, "case-0099")
        official = res.to_official_dict()
        self.assertEqual(official["id"], "case-0099")
        self.assertIn(official["label"], (0, 1, 2))

    def test_cli_runner_batch_execution(self):
        import json
        import tempfile
        from typer.testing import CliRunner
        from src.cli import app as cli_app

        runner = CliRunner()
        
        # Test 1: Beginner Task input JSON
        sample_input = [
            {
                "id": "case-test-beginner",
                "reference": {"text": "Der Bundesrat empfiehlt Nein zur Initiative."},
                "claim": {"text": "Der Bundesrat empfiehlt die Ablehnung."}
            },
            {
                "id": "case-test-advanced",
                "booklet": {"path": str(self.booklet)},
                "vote": "Nachhaltigkeitsinitiative",
                "claim": {"text": "Die Initiative verlangt 10 Millionen als Obergrenze vor 2050."}
            }
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tf:
            json.dump(sample_input, tf)
            temp_path = tf.name

        try:
            result = runner.invoke(cli_app, ["run", "--input", temp_path, "--mock"])
            self.assertEqual(result.exit_code, 0)
            parsed_output = json.loads(result.output.strip())
            self.assertIsInstance(parsed_output, list)
            self.assertEqual(len(parsed_output), 2)
            self.assertEqual(parsed_output[0]["id"], "case-test-beginner")
            self.assertEqual(parsed_output[1]["id"], "case-test-advanced")
            for item in parsed_output:
                self.assertIn("label", item)
                self.assertIn("label_name", item)
                self.assertIn("evidence", item)
                self.assertIn("metrics", item)
        finally:
            Path(temp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()

