import sys
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from src.grounded_apertus import (ApertusJSONTransport, GroundedApertusNLI, QueryPreparer,
                                  ApertusResponseError, RequestLimitReached)
from src.evidence_units import EvidenceUnit
from src.adaptive_calibration import search_neutral_policy


class GroundedApertusTests(unittest.TestCase):
    def source(self):
        return EvidenceUnit("a"*64, 26, "paragraph-1", "Consent", "it",
                            "Il consenso del proprietario è richiesto per subaffittare l'alloggio.", (0,0,1,1), 0)

    def transport(self, payloads):
        sdk = MagicMock()
        responses = []
        for payload in payloads:
            content = json.dumps(payload)
            responses.append(SimpleNamespace(model="test-model", usage=SimpleNamespace(
                prompt_tokens=13, completion_tokens=7, total_tokens=20),
                choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))],
                model_dump=lambda **kwargs: {}))
        sdk.chat.completions.create.side_effect = responses
        return ApertusJSONTransport(sdk=sdk, model="test-model", journal=MagicMock()), sdk

    def test_original_quotes_and_physical_pages_are_copied_by_id(self):
        transport, sdk = self.transport([dict(label=2, p_entail=0, p_neutral=0, p_contra=1,
            reasoning="Consenso richiesto", evidence_ids=["U0001"])])
        source = self.source()
        result = GroundedApertusNLI(transport).classify("Non serve un consenso.", [source], claim_language="it")
        self.assertEqual(result.evidence[0]["text"], source.text)
        self.assertEqual(result.evidence[0]["page"], 26)
        self.assertEqual(result.evidence_ids, [source.id])
        for cap in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
            self.assertNotIn(cap, sdk.chat.completions.create.call_args.kwargs)
        self.assertEqual(transport.requests, 1)
        self.assertEqual(transport.input_tokens, 13)

    def test_unknown_source_is_error_without_label_or_retry(self):
        transport, sdk = self.transport([dict(label=2, p_entail=0, p_neutral=0, p_contra=1,
            reasoning="reasoning", evidence_ids=["U9999"])])
        with self.assertRaises(ApertusResponseError) as caught:
            GroundedApertusNLI(transport).classify("claim", [self.source()], claim_language="it")
        self.assertFalse(hasattr(caught.exception, "label"))
        self.assertEqual(sdk.chat.completions.create.call_count, 1)

    def test_binary_no_no_yields_neutral_with_two_measured_calls(self):
        transport, _ = self.transport([dict(answer=False, confidence=.95, reasoning="no", evidence_ids=[])]*2)
        result = GroundedApertusNLI(transport).classify("claim", [self.source()], claim_language="it", mode="binary")
        self.assertEqual(result.label, 1)
        self.assertEqual(result.evidence, [])
        self.assertEqual(transport.requests, 2)
        self.assertEqual(len(result.calls), 2)

    def test_binary_yes_yes_uses_explicit_third_judge(self):
        yes = dict(answer=True, confidence=.9, reasoning="yes", evidence_ids=["U0001"])
        judge = dict(label=1, p_entail=.1, p_neutral=.8, p_contra=.1, reasoning="actor ambiguity", evidence_ids=[])
        transport, _ = self.transport([yes, yes, judge])
        result = GroundedApertusNLI(transport).classify("claim", [self.source()], claim_language="it", mode="binary")
        self.assertEqual(result.label, 1)
        self.assertEqual(transport.requests, 3)
        self.assertEqual(result.mode, "binary_conflict_judge")

    def test_budget_guard_does_not_make_a_call(self):
        transport, sdk = self.transport([{}])
        transport.maximum = 1
        transport.requests = 1
        with self.assertRaises(RequestLimitReached):
            transport.request([], stage="nli")
        sdk.chat.completions.create.assert_not_called()

    def test_retrieval_only_initialization_does_not_require_api_key(self):
        with patch("src.grounded_apertus.config.LLM_API_KEY", ""), patch("openai.OpenAI") as sdk:
            transport = ApertusJSONTransport()
        sdk.assert_not_called()
        self.assertIsNone(transport.sdk)

    def test_neutral_thresholds_are_validation_selected_and_reject_test_rows(self):
        rows = [{"split": "validation", "gold": gold, "label": label,
                 "probabilities": {"neutral": .1}, "signals": {"best_reranker_score": relevance,
                   "reranker_margin": .1, "retrieval_agreement": 2}}
                for gold,label,relevance in ((0,0,.95), (1,0,.1), (2,2,.9))]
        best, grid = search_neutral_policy(rows)
        self.assertEqual(best.signal, "best_reranker_score")
        self.assertEqual(best.validation_macro_f1, 1)
        self.assertGreater(best.validation_macro_f1, best.baseline_macro_f1)
        self.assertTrue(grid)
        with self.assertRaises(ValueError):
            search_neutral_policy([{**row, "split": "test"} for row in rows])


if __name__ == "__main__":
    unittest.main()
