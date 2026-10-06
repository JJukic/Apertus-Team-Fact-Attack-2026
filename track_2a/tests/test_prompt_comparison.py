"""No-network checks for prompt isolation, hard budgets and frozen context."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
import run_prompt_comparison as comparison
from run_hybrid_quality_comparison import BudgetedCreate, BudgetExhausted


class PromptComparisonTests(unittest.TestCase):
    def test_candidate_changes_only_system_message_and_attempts_have_hard_cap(self):
        client = comparison.PromptClient.__new__(comparison.PromptClient)
        client.addendum = "Exact relevant quotes only."
        response = MagicMock(usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2))
        create = MagicMock(return_value=response)
        journal = MagicMock()
        client.budget = BudgetedCreate(create, 2, journal)
        messages = [{"role": "system", "content": "Original"},
                    {"role": "user", "content": "Source and claim"}]
        args = {"model": "model", "messages": messages, "temperature": 0}
        client.variant = "baseline"
        client.create_with_prompt(**args)
        self.assertEqual(create.call_args.kwargs, args)
        client.variant = "candidate"
        client.create_with_prompt(**args)
        actual = create.call_args.kwargs
        self.assertEqual(actual["messages"][0]["content"], "Original\n\nExact relevant quotes only.")
        self.assertEqual(actual["messages"][1], messages[1])
        self.assertEqual(messages[0]["content"], "Original")
        with self.assertRaises(BudgetExhausted):
            client.create_with_prompt(**args)
        self.assertEqual(create.call_count, 2)
        self.assertEqual(journal.call_count, 2)

    def test_failed_attempt_counts_toward_budget(self):
        create = MagicMock(side_effect=RuntimeError("Transport failed"))
        budget = BudgetedCreate(create, 1, MagicMock())
        with self.assertRaises(RuntimeError):
            budget(model="model")
        with self.assertRaises(BudgetExhausted):
            budget(model="model")
        self.assertEqual(create.call_count, 1)

    def test_frozen_context_cannot_be_mutated_between_variants(self):
        engine = comparison.FrozenEngine.__new__(comparison.FrozenEngine)
        engine.frozen = {"context": "Original", "sources": [{"text": "Quote"}]}
        first = engine.retrieve_context()
        first["sources"][0]["text"] = "Mutated"
        self.assertEqual(engine.retrieve_context()["sources"][0]["text"], "Quote")

    def test_prepare_serializes_context_without_gold_leaking_into_retrieval(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            pdf = root / "2024-09-22_de.pdf"
            pdf.write_bytes(b"Source PDF placeholder")
            prompt = root / "prompt.txt"
            prompt.write_text("Candidate")
            dataset, output = root / "dataset.jsonl", root / "plan.json"
            record = {"id": "new-1", "claim": "Claim text", "entailment_label": 2,
                      "reference_string": "SECRET GOLD REFERENCE", "split": "dev",
                      "booklet_date": "2024-09-22", "booklet_language": "de", "claim_language": "de"}
            dataset.write_text(json.dumps(record) + "\n")
            engine = MagicMock()
            engine.retrieve_context.return_value = {
                "context": "Source text", "sources": [{"page_number": 1, "text": "Source text"}],
                "metrics": {"actual_strategy": "retrieval", "requested_strategy": "retrieval"},
                "booklet_data": {"paragraphs": [], "retriever": object()}}
            with patch.object(comparison.config, "BOOKLETS_DIR", root), \
                    patch.object(comparison, "ClaimVerificationEngine", return_value=engine), \
                    patch.object(comparison, "PromptClient") as live:
                comparison.prepare(dataset, prompt, ["retrieval"], output)
                live.assert_not_called()
            plan = json.loads(output.read_text())
            self.assertEqual(plan["minimum_api_requests"], 2)
            self.assertNotIn("booklet_data", plan["snapshots"][0]["prepared"])
            engine.retrieve_context.assert_called_once_with("Claim text", pdf, strategy="retrieval", vote=None)
            comparison.validate_plan(plan)
            plan["snapshots"][0]["prepared"]["context"] = "Changed"
            with self.assertRaisesRegex(ValueError, "Frozen context changed"):
                comparison.validate_plan(plan)

    def test_three_class_metrics_and_label_changes(self):
        def result(label):
            return {"label": label, "source_quotes": 0, "exact_source_matches": 0}
        rows = [{"id": str(i), "gold": i, "language": "de", "baseline": result(1),
                 "candidate": result(i)} for i in (0, 1, 2)]
        report = comparison.summarize(rows)
        self.assertEqual(report["candidate"]["macro_f1"], 1)
        self.assertEqual(report["corrected"], 2)
        self.assertEqual(report["regressed"], 0)
        self.assertEqual(report["changed_cases"], ["0", "2"])
        self.assertEqual(report["candidate"]["semantic_evidence_validity"], "pending_manual_review")

    def test_execution_rejects_unapproved_plan_before_live_client(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = Path(folder) / "plan.json"
            plan.write_text("{}")
            with patch.object(comparison, "PromptClient") as live:
                with self.assertRaisesRegex(ValueError, "exact approved plan hash"):
                    comparison.execute(plan, Path(folder) / "result.json", 200, "wrong")
                live.assert_not_called()


if __name__ == "__main__":
    unittest.main()
