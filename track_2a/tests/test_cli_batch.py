"""
Batch CLI in the official OST input/output format (info-session slides 24/25) and with Hugging Face dataset rows,
offline (mock model).
"""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from src import config
from src.cli import app, case_booklet, load_cases, resolve_booklet_path
from src.text_utils import guess_language

BOOKLET = config.BOOKLETS_DIR / "2026-06-14_de.pdf"


class TestBatchCli(unittest.TestCase):
    def setUp(self):
        if not BOOKLET.exists():
            self.skipTest(f"Booklet not found at {BOOKLET}")
        self.tmp = Path(tempfile.mkdtemp())

    def _run(self, cases):
        inp, out = self.tmp / "cases.jsonl", self.tmp / "pred.jsonl"
        inp.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases), encoding="utf-8")
        result = CliRunner().invoke(app, ["run", "-i", str(inp), "-o", str(out), "--mock"])
        self.assertEqual(result.exit_code, 0, result.output)
        return [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]

    def test_official_schema_and_broken_case_does_not_stop_the_batch(self):
        preds = self._run([
            {"id": "case-1", "booklet": {"path": "booklets/2026_06_14_de.pdf"}, "vote": "Änderung des Zivildienstgesetzes",
             "claim": {"text": "Le Conseil fédéral recommande de rejeter l'initiative."}},
            {"id": "case-2", "booklet": {"path": "booklets/missing.pdf"}, "vote": "x", "claim": {"text": "Test."}},
            {"id": "case-3", "reference": {"text": "Der Bundesrat lehnt die Initiative ab. Sie bringt neue Vorschriften."},
             "claim": {"text": "Il Consiglio federale raccomanda di accettare l'iniziativa."}},
        ])
        self.assertEqual([p["id"] for p in preds], ["case-1", "case-2", "case-3"])
        for p in preds:
            self.assertEqual(set(p), {"id", "label", "label_name", "evidence", "metrics"})
            self.assertEqual(set(p["metrics"]), {"input_tokens", "output_tokens", "inference_time_ms"})
            self.assertEqual(p["label_name"], {0: "entailment", 1: "neutral", 2: "contradiction"}[p["label"]])
            if p["label"] == 1:
                self.assertEqual(p["evidence"], [])
            for ev in p["evidence"]:
                self.assertEqual(set(ev), {"page", "text"})
        self.assertEqual(preds[1]["label"], 1)  # missing booklet -> valid neutral record

    def test_hugging_face_rows_use_the_booklet(self):
        # Rows as published on Hugging Face: plain claim, reference_string, booklet_url, no id; the publish date
        # (2026-05-28) differs from the booklet's vote date (2026-06-14), which only the URL carries
        row = json.loads((config.DATA_DIR / "demo_dataset.jsonl").read_text(encoding="utf-8").splitlines()[0])
        for f in ("entailment_label", "baseline_score"):
            row.pop(f, None)
        inp = self.tmp / "hf.csv"
        import pandas as pd

        pd.DataFrame([row, row]).to_csv(inp, index=False)
        for args, page_evidence in ((["--task", "auto"], True), (["--task", "beginner"], False)):
            result = CliRunner().invoke(app, ["run", "-i", str(inp), "--mock", *args])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertNotIn("warning", result.output)
            preds = json.loads(result.output[result.output.index("["):])
            self.assertEqual([p["id"] for p in preds], ["case-0001", "case-0002"])
            pages = {ev["page"] for p in preds for ev in p["evidence"]}
            if page_evidence and pages:
                self.assertTrue(pages - {1}, "advanced: evidence carries booklet page numbers")

    def test_advanced_task_without_booklet_is_a_reported_error(self):
        inp = self.tmp / "ref.jsonl"
        inp.write_text(json.dumps({"id": "r", "claim": "Test.", "reference_string": "Der Bundesrat lehnt ab."}), encoding="utf-8")
        result = CliRunner().invoke(app, ["run", "-i", str(inp), "--mock", "--task", "advanced"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("has no booklet", result.output)


class TestConcurrentBatch(unittest.TestCase):
    def test_overlapping_calls_preserve_ids_costs_and_a_complete_failed_case(self):
        barrier = threading.Barrier(2)
        fast_recorded = threading.Event()

        class Client:
            mock = False

            def _request_completion(self, **body):
                claim = body['messages'][0]['content']
                barrier.wait(timeout=5)
                if claim == 'slow':
                    if not fast_recorded.wait(timeout=5):
                        raise TimeoutError('Fast request was not recorded')
                else:
                    time.sleep(0.01)
                return SimpleNamespace(
                    usage=SimpleNamespace(prompt_tokens=3 if claim == 'slow' else 7, completion_tokens=2),
                    choices=[SimpleNamespace(message=SimpleNamespace(content='{}'), finish_reason='stop')])

        client = Client()

        class Engine:
            def __init__(self, **kwargs):
                self.client = client

            def verify_premise(self, **kwargs):
                self.client._request_completion(model='swiss-ai/Apertus-v1.5-70B-thinking',
                                                messages=[{'role': 'user', 'content': kwargs['claim']}])
                if kwargs['claim'] == 'fast':
                    fast_recorded.set()
                prediction = {'id': kwargs['case_id'], 'label': 0, 'label_name': 'entailment',
                              'evidence': [], 'metrics': {'input_tokens': 0, 'output_tokens': 0, 'inference_time_ms': 1}}
                return SimpleNamespace(error=None, to_official_dict=lambda **unused: prediction)

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            inp, out, journal = directory / 'cases.jsonl', directory / 'results.data', directory / 'requests.jsonl'
            cases = [{'id': '', 'claim': 'slow', 'reference': 'Source'},
                     {'id': 'fast-ä', 'claim': 'fast', 'reference': 'Source'},
                     {'id': 'missing', 'claim': 'Unknown', 'booklet': {'path': 'missing.pdf'}}]
            inp.write_text(''.join(json.dumps(case) + '\n' for case in cases))
            with patch('src.cli.ClaimVerificationEngine', Engine), patch.object(config, 'OFFICIAL_IO', True), \
                    patch.dict('os.environ', {'REQUEST_JOURNAL_PATH': str(journal)}):
                result = CliRunner().invoke(app, ['run', '--input', str(inp), '--output', str(out), '--workers', '2'])
            self.assertEqual(result.exit_code, 2)
            outputs = [json.loads(line) for line in out.read_text().splitlines()]
            self.assertEqual([row['id'] for row in outputs], ['', 'fast-ä', 'missing'])
            self.assertEqual([row['metrics']['input_tokens'] for row in outputs], [3, 7, 0])
            self.assertEqual(outputs[-1]['label'], 1)
            events = [json.loads(line) for line in journal.read_text().splitlines()]
            self.assertEqual([event['case_id'] for event in events], ['fast-ä', ''])
            self.assertEqual(sum(event['input_tokens'] for event in events), 10)
            diagnostics = json.loads(out.with_name(out.name + '.diagnostics.json').read_text())
            self.assertEqual(diagnostics['workers'], 2)
            self.assertEqual(diagnostics['api_attempts'], 2)
            self.assertEqual(diagnostics['failures'], [{'id': 'missing', 'error_type': 'FileNotFoundError'}])
            self.assertLess(diagnostics['llm_in_flight_union_seconds_local'],
                            sum(event['end'] - event['start'] for event in events))


class TestHelpers(unittest.TestCase):
    def test_case_booklet_from_hugging_face_fields(self):
        if not BOOKLET.exists():
            self.skipTest(f"Booklet not found at {BOOKLET}")
        url = "https://www.bk.admin.ch/dam/de/sd-web/WeUrKyC0FyPc/2026-06-14_erlaeuterungen.pdf"
        self.assertEqual(case_booklet({"booklet_url": url, "booklet_publish_date": "2026-05-28"}, Path(".")), BOOKLET)
        self.assertEqual(case_booklet({"booklet_file": "2026-06-14_de.pdf"}, Path(".")), BOOKLET)
        self.assertIsNone(case_booklet({"reference_string": "x"}, Path(".")))

    def test_load_cases_formats(self):
        tmp = Path(tempfile.mkdtemp())
        one, many = {"claim": "a"}, [{"claim": "a"}, {"claim": "b"}]
        (tmp / "one.json").write_text(json.dumps(one), encoding="utf-8")
        (tmp / "list.json").write_text(json.dumps(many), encoding="utf-8")
        (tmp / "rows.jsonl").write_text("\n".join(json.dumps(c) for c in many), encoding="utf-8")
        self.assertEqual(load_cases(tmp / "one.json"), [one])
        self.assertEqual(load_cases(tmp / "list.json"), many)
        self.assertEqual(load_cases(tmp / "rows.jsonl"), many)

    def test_resolve_booklet_path_accepts_underscore_dates(self):
        if not BOOKLET.exists():
            self.skipTest(f"Booklet not found at {BOOKLET}")
        self.assertEqual(resolve_booklet_path("booklets/2026_06_14_de.pdf", Path(".")).name, "2026-06-14_de.pdf")
        with self.assertRaises(FileNotFoundError):
            resolve_booklet_path("nowhere/none.pdf", Path("."))

    def test_guess_language(self):
        self.assertEqual(guess_language("La proposition entraînera une augmentation de la TVA."), "fr")
        self.assertEqual(guess_language("Il Consiglio federale consiglia di respingere l'iniziativa."), "it")
        self.assertEqual(guess_language("Der Bundesrat empfiehlt ein Nein."), "de")


if __name__ == "__main__":
    unittest.main()
