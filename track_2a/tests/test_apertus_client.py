"""
Unit tests for ApertusClient, JSON parsing, and Decision Arbiter.
"""

import sys
from pathlib import Path
import unittest
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from src.apertus_client import (ApertusClient, NLIOutput, ApertusAPIError,
                                ApertusConfigurationError, ApertusResponseError)


class TestApertusClient(unittest.TestCase):
    def setUp(self):
        # Force mock mode for fast and deterministic unit testing
        self.client = ApertusClient(mock=True)

    def test_calibrated_decision_arbiter_conflict(self):
        # Rule 1: Conflict dominant
        label, rule = ApertusClient._apply_calibrated_decision(
            p_entail=0.10, p_neutral=0.20, p_contra=0.70, raw_label=0
        )
        self.assertEqual(label, 2)
        self.assertIn("Conflict dominant", rule)

    def test_calibrated_decision_arbiter_support(self):
        # Rule 2: Direct support dominant
        label, rule = ApertusClient._apply_calibrated_decision(
            p_entail=0.85, p_neutral=0.10, p_contra=0.05, raw_label=1
        )
        self.assertEqual(label, 0)
        self.assertIn("Direct support dominant", rule)

    def test_calibrated_decision_arbiter_neutrality(self):
        # Rule 3: High neutrality
        label, rule = ApertusClient._apply_calibrated_decision(
            p_entail=0.20, p_neutral=0.65, p_contra=0.15, raw_label=0
        )
        self.assertEqual(label, 1)
        self.assertIn("High neutrality", rule)

    def test_calibrated_decision_arbiter_ambiguity(self):
        # Rule 3: Epistemic ambiguity zone (|p_entail - p_contra| < 0.15)
        label, rule = ApertusClient._apply_calibrated_decision(
            p_entail=0.48, p_neutral=0.06, p_contra=0.46, raw_label=0
        )
        self.assertEqual(label, 1)
        self.assertIn("Epistemic ambiguity zone", rule)

    def test_backward_compatible_fuzzy_alias(self):
        label, rule = ApertusClient._apply_fuzzy_decision(
            p_entail=0.85, p_neutral=0.10, p_contra=0.05, raw_label=1
        )
        self.assertEqual(label, 0)

    def test_parse_json_clean(self):
        raw = '{"label": 0, "reasoning": "Supports statement", "evidence": ["Text sample"], "p_entail": 0.9}'
        parsed = ApertusClient._parse_json(raw)
        self.assertEqual(parsed["label"], 0)
        self.assertEqual(parsed["p_entail"], 0.9)

    def test_parse_json_markdown_blocks(self):
        raw = '```json\n{"label": 2, "reasoning": "Contradicts", "evidence": [], "p_contra": 0.8}\n```'
        parsed = ApertusClient._parse_json(raw)
        self.assertEqual(parsed["label"], 2)
        self.assertEqual(parsed["p_contra"], 0.8)

    def test_parse_json_embedded_fallback(self):
        raw = 'Here is the analysis result: {"label": 1, "reasoning": "Neutral topic"} Hope this helps!'
        parsed = ApertusClient._parse_json(raw)
        self.assertEqual(parsed["label"], 1)

    def test_mock_infer(self):
        context = "Die Volksinitiative verlangt, dass die ständige Wohnbevölkerung vor 2050 10 Millionen nicht überschreitet."
        claim = "Die Bevölkerung darf gemäss Initiative 10 Millionen Menschen nicht überschreiten."
        out = self.client.infer(context, claim, claim_language="de")

        self.assertIsInstance(out, NLIOutput)
        self.assertIn(out.label, (0, 1, 2))
        self.assertGreater(out.tokens_total, 0)
        self.assertGreater(out.latency_ms, 0.0)


class TestOperationalFailures(unittest.TestCase):
    def setUp(self):
        self.client = ApertusClient(mock=True)
        self.client.mock = False
        self.create = MagicMock()
        self.client.client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=self.create)))
        self.valid = dict(label=1, p_entail=0.1, p_neutral=0.8, p_contra=0.1,
                          reasoning="Information fehlt", evidence=[])

    def response(self, content=None, finish_reason="stop", usage=True):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps(self.valid) if content is None else content),
                finish_reason=finish_reason)],
            usage=SimpleNamespace(prompt_tokens=13, completion_tokens=7, total_tokens=20)
            if usage else None)

    @patch("src.apertus_client.time.sleep")
    def test_exhausted_transport_is_not_neutral(self, sleep):
        self.create.side_effect = RuntimeError("SECRET API KEY")
        with self.assertLogs("src.apertus_client", level="WARNING") as logs:
            with self.assertRaises(ApertusAPIError) as caught:
                self.client.infer("context", "claim")
        self.assertEqual(self.create.call_count, 3)
        self.assertEqual(caught.exception.api_attempts, 3)
        self.assertIsNone(caught.exception.tokens_total)
        self.assertFalse(hasattr(caught.exception, "label"))
        self.assertNotIn("SECRET", str(caught.exception) + str(logs.output))

    @patch("src.apertus_client.time.sleep")
    def test_recovery_keeps_real_neutral_and_usage(self, sleep):
        self.create.side_effect = [TimeoutError(), self.response()]
        result = self.client.infer("context", "claim")
        self.assertEqual(result.label, 1)
        self.assertEqual(result.reasoning, self.valid["reasoning"])
        self.assertEqual(result.tokens_total, 20)
        self.assertEqual(self.create.call_count, 2)
        for parameter in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            self.assertNotIn(parameter, self.create.call_args.kwargs)

    def test_invalid_json_and_truncation_preserve_usage_without_label(self):
        for content, finish in [("not JSON", "stop"), ('{"label": 1', "stop"),
                                (json.dumps(self.valid), "length"),
                                (json.dumps(self.valid), "content_filter")]:
            with self.subTest(content=content, finish=finish):
                self.create.return_value = self.response(content, finish)
                with self.assertRaises(ApertusResponseError) as caught:
                    self.client.infer("context", "claim")
                self.assertEqual(caught.exception.tokens_total, 20)
                self.assertEqual(caught.exception.api_attempts, 1)
                self.assertFalse(hasattr(caught.exception, "label"))

    def test_invalid_response_fields_are_not_coerced_to_neutral(self):
        invalid = [("label", v) for v in (None, 9, True, 0.5, "1")]
        invalid += [("p_entail", v) for v in (-0.1, 1.1, float("nan"), float("inf"), True, "0.2")]
        invalid += [("evidence", [42]), ("reasoning", None)]
        for field, value in invalid:
            with self.subTest(field=field, value=value):
                self.create.return_value = self.response(json.dumps({**self.valid, field: value}))
                with self.assertRaises(ApertusResponseError):
                    self.client.infer("context", "claim")
        for field in self.valid:
            with self.subTest(missing=field):
                self.create.return_value = self.response(json.dumps(
                    {k: v for k, v in self.valid.items() if k != field}))
                with self.assertRaises(ApertusResponseError):
                    self.client.infer("context", "claim")

    def test_missing_usage_on_bad_response_is_unknown(self):
        self.create.return_value = self.response("invalid", usage=False)
        with self.assertRaises(ApertusResponseError) as caught:
            self.client.infer("context", "claim")
        self.assertIsNone(caught.exception.tokens_total)

    def test_client_initialization_failure_never_enables_mock(self):
        with patch("openai.OpenAI", side_effect=RuntimeError("SECRET")):
            with self.assertRaises(ApertusConfigurationError) as caught:
                ApertusClient(api_key="fake-test-key", mock=False)
        self.assertNotIn("SECRET", str(caught.exception))

    def test_explicit_live_without_key_fails_and_explicit_mock_works(self):
        with self.assertRaises(ApertusConfigurationError):
            ApertusClient(api_key="", mock=False)
        self.assertTrue(ApertusClient(api_key="", mock=True).mock)

    def test_sdk_retry_is_disabled_to_keep_attempts_countable(self):
        with patch("openai.OpenAI") as sdk:
            ApertusClient(api_key="fake-test-key", mock=False)
        self.assertEqual(sdk.call_args.kwargs["max_retries"], 0)

    def test_evaluation_stops_instead_of_scoring_error_as_neutral(self):
        import tempfile
        from src.evaluator import BenchmarkEvaluator
        engine = MagicMock()
        engine.verify_claim.side_effect = ApertusAPIError("transport failure")
        with tempfile.TemporaryDirectory() as folder:
            dataset = Path(folder) / "cases.jsonl"
            dataset.write_text(json.dumps(dict(claim="claim", entailment_label=1)) + "\n")
            evaluator = BenchmarkEvaluator(engine)
            with patch.object(evaluator, "print_report") as report:
                with self.assertRaises(ApertusAPIError):
                    evaluator.evaluate(dataset)
                report.assert_not_called()


if __name__ == "__main__":
    unittest.main()
