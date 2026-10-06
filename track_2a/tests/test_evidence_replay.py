"""Check the offline replay against the real evidence stage for all 200 answers."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
from build_hybrid_failure_audit import label_after_stages, read_jsonl
from historical_apertus_client import ApertusClient
from src.apertus_client import NLIOutput
from src.inference import ClaimVerificationEngine
from src.pdf_parser import PDFParser


class ReplayParityTests(unittest.TestCase):
    def test_all_recorded_answers_match_real_pipeline_evidence_stage(self):
        folder = BASE / "docs/evaluation"
        report = json.loads((folder / "evidence_replay_2026-10-06_v2.json").read_text())
        expected = {(r["strategy"], r["phase"], r["case_id"]): r for r in report["results"]}
        apis = {}
        for event in read_jsonl(folder / "hybrid_quality_2026-10-05.api.jsonl"):
            apis.setdefault((event["strategy"], event["phase"]), []).append(event)
        pages = {}
        cases = read_jsonl(folder / "hybrid_quality_2026-10-05.cases.jsonl")
        self.assertEqual(len(cases), 200)
        for case in cases:
            record, prediction = case["dataset_record"], case["prediction"]
            path = BASE / "data/booklets" / f'{record["booklet_date"]}_{record["booklet_language"]}.pdf'
            if not path.exists():
                self.skipTest(f"Historical PDF not available: {path}")
            if path not in pages:
                pages[path] = {p["page_number"]: p for p in PDFParser().extract_pages(path)}
            key = (case["strategy"], case["phase"])
            api = apis[key][case["case_index"]]
            parsed = ApertusClient._parse_json(api["response"]["choices"][0]["message"]["content"])
            _, _, before_evidence, _, evidence = label_after_stages(parsed, prediction)
            # Freeze every upstream stage at its recorded output. No API,
            # embedding model, retrieval or numerical recomputation is needed.
            engine = ClaimVerificationEngine.__new__(ClaimVerificationEngine)
            engine.client = MagicMock()
            engine.client.infer.return_value = NLIOutput(label=before_evidence, reasoning="Replay", evidence=evidence)
            engine.retrieve_context = MagicMock(return_value={
                "metrics": prediction["retrieval_metrics"], "context": "",
                "sources": [pages[path][p] for p in prediction["retrieval_metrics"]["selected_pages"]],
                "booklet_data": {"paragraphs": []}})
            with self.subTest(method=key, case=record["id"]):
                actual = engine.verify_claim(record["claim"], path, strategy=case["strategy"])
                target = expected[(*key, record["id"])]
                self.assertEqual(actual.label, target["new_label"])
                self.assertEqual([{"page": s.page_number, "quote": s.quote} for s in actual.evidence_sources],
                                 target["new_evidence"])
                if target["source_numerical_conflict"]:
                    self.assertEqual(actual.label, 2)
                    self.assertIn("100 Prozent", actual.numerical_conflict)
                    self.assertIn("55 Prozent", actual.numerical_conflict)
