"""
Unit tests for ApertusClient, JSON parsing, and Decision Arbiter.
"""

import sys
from pathlib import Path
import unittest

_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from src.apertus_client import ApertusClient, NLIOutput


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

    def test_calibrated_decision_keeps_label_when_confidences_disagree(self):
        # Rule 0: Apertus writes "label": 0 with "p_contra": 1.0 -> trust the label
        label, rule = ApertusClient._apply_calibrated_decision(
            p_entail=0.0, p_neutral=0.0, p_contra=1.0, raw_label=0, label_given=True
        )
        self.assertEqual(label, 0)
        self.assertIn("Decision-Rule 0", rule)

    def test_calibrated_decision_consistent_label_still_uses_rules(self):
        label, rule = ApertusClient._apply_calibrated_decision(
            p_entail=0.48, p_neutral=0.06, p_contra=0.46, raw_label=0, label_given=True
        )
        self.assertEqual(label, 1)
        self.assertIn("Epistemic ambiguity zone", rule)

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

    def test_strip_thinking_keeps_only_the_final_answer(self):
        raw = '<|inner_prefix|>The committee says {"label": 2} is wrong.<|inner_suffix|>{"label": 0, "evidence_ids": [3]}'
        self.assertEqual(ApertusClient._parse_json(ApertusClient._strip_thinking(raw))["label"], 0)

    def test_strip_thinking_truncated_reasoning_uses_last_json(self):
        raw = '<|inner_prefix|>First guess {"label": 2}. On reflection {"label": 1, "p_neutral": 1.0}'
        self.assertEqual(ApertusClient._parse_json(ApertusClient._strip_thinking(raw))["label"], 1)
        self.assertEqual(ApertusClient._strip_thinking('{"label": 0}'), '{"label": 0}')

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


if __name__ == "__main__":
    unittest.main()
