"""
Batch CLI in the official OST input/output format (info-session slides 24/25), offline (mock model).
"""

import json
import tempfile
import unittest
from pathlib import Path

from typer.testing import CliRunner

from src import config
from src.cli import app, resolve_booklet_path
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


class TestHelpers(unittest.TestCase):
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
