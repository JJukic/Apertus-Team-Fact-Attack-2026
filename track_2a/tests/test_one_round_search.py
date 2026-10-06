import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from src.apertus_client import ApertusClient, ApertusResponseError
from src.inference import ClaimVerificationEngine
from one_round_search import ApertusSearchPlanner, OneRoundSearch
from run_hybrid_quality_comparison import BudgetedCreate, BudgetExhausted


class SearchTests(unittest.TestCase):
    quote = "Le Conseil fédéral recommande le rejet de cette initiative."

    def response(self, label=1, plan=False, finish="stop"):
        data = {"query": "Conseil fédéral recommandation"} if plan else {
            "label": label, "reasoning": "Information manque",
            "evidence": [] if label == 1 else [self.quote],
            "p_entail": 0.9 if label == 0 else 0.05,
            "p_neutral": 0.9 if label == 1 else 0.05,
            "p_contra": 0.9 if label == 2 else 0.05}
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish,
            message=SimpleNamespace(content=json.dumps(data)))],
            usage=SimpleNamespace(prompt_tokens=13, completion_tokens=7, total_tokens=20),
            model_dump=lambda **kwargs: {"choices": [{"message": {"content": json.dumps(data)},
                                                       "finish_reason": finish}],
                                          "usage": {"prompt_tokens": 13, "completion_tokens": 7,
                                                    "total_tokens": 20}})

    def setup_search(self, responses, maximum=7, duplicate=False):
        client = ApertusClient(mock=True)
        client.mock = False
        transport = MagicMock(side_effect=responses)
        client.budget = BudgetedCreate(transport, maximum, MagicMock())
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=client.budget)))
        engine = ClaimVerificationEngine(apertus_client=client)
        def context(number, text):
            return {"context": text, "sources": [{"page_number": number, "text": text}],
                    "booklet_data": {"paragraphs": []}, "metrics": {
                        "requested_strategy": "hybrid_dense", "actual_strategy": "hybrid_dense",
                        "selected_pages": [number]}}
        first = context(1, "Informations générales sur cette initiative.")
        engine.retrieve_context = MagicMock(side_effect=[first, first if duplicate else context(2, self.quote)])
        return OneRoundSearch(engine, ApertusSearchPlanner(client)), transport

    def verify(self, experiment, claim_language="de", booklet_language="fr"):
        return experiment.verify("Der Bundesrat empfiehlt die Ablehnung.", "unused.pdf",
                                 claim_language=claim_language, booklet_language=booklet_language)

    def test_recovery_counts_planning_and_both_classifications(self):
        experiment, transport = self.setup_search([
            self.response(), self.response(plan=True), self.response(0)])
        result = self.verify(experiment)
        self.assertEqual(result["initial_prediction"]["label"], 1)
        self.assertEqual(result["prediction"]["label"], 0)
        self.assertEqual(result["search_rounds"], 1)
        self.assertEqual(result["exact_selected_source_matches"], 1)
        self.assertEqual((result["api_requests"], result["input_tokens"], result["output_tokens"]), (3, 39, 21))
        self.assertEqual(result["selected_pages"], [1, 2])
        for call in transport.call_args_list:
            self.assertNotIn("max_tokens", call.kwargs)

    def test_non_neutral_skips_planning(self):
        experiment, transport = self.setup_search([self.response(2)])
        self.assertEqual(self.verify(experiment)["search_rounds"], 0)
        self.assertEqual(transport.call_count, 1)

    def test_duplicate_context_skips_final_inference(self):
        experiment, transport = self.setup_search([self.response(), self.response(plan=True)], duplicate=True)
        result = self.verify(experiment)
        self.assertEqual(result["search_rounds"], 1)
        self.assertEqual(transport.call_count, 2)

    def test_final_neutral_does_not_repeat_search(self):
        experiment, transport = self.setup_search([self.response(), self.response(plan=True), self.response()])
        self.assertEqual(self.verify(experiment)["prediction"]["label"], 1)
        self.assertEqual(transport.call_count, 3)
        self.assertEqual(experiment.engine.retrieve_context.call_count, 2)

    @patch("src.apertus_client.time.sleep")
    def test_budget_exhaustion_is_not_retried_or_scored(self, sleep):
        experiment, transport = self.setup_search([self.response(), self.response(plan=True)], maximum=2)
        with self.assertRaises(BudgetExhausted):
            self.verify(experiment)
        self.assertEqual(transport.call_count, 2)
        sleep.assert_not_called()

    def test_truncated_plan_is_error(self):
        experiment, transport = self.setup_search([self.response(), self.response(plan=True, finish="length")])
        with self.assertRaises(ApertusResponseError) as caught:
            self.verify(experiment)
        self.assertEqual(caught.exception.tokens_total, 20)
        self.assertEqual(transport.call_count, 2)

    def test_language_pairs_are_explicit(self):
        for a in ("de", "fr", "it"):
            for b in ("de", "fr", "it"):
                with self.subTest(pair=(a, b)):
                    experiment, transport = self.setup_search([self.response(), self.response(plan=True), self.response()])
                    self.verify(experiment, a, b)
                    content = transport.call_args_list[1].kwargs["messages"][1]["content"]
                    self.assertIn(f"Claim language: {a}", content)
                    self.assertIn(f"Booklet language: {b}", content)


if __name__ == "__main__":
    unittest.main()
