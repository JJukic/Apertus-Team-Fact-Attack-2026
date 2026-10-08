"""
Evaluation contract of the OST Q&A (2026-10-08): the container is called with `--input/--output` and no command,
BASE_URL / API_KEY are injected at run time, and task A evidence is scored as Hit@5 on items of at most ~one page.
"""

import importlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src import config
from src.inference import evidence_items
from src.text_utils import split_evenly

TRACK_DIR = Path(__file__).resolve().parent.parent


class TestEndpointVariables(unittest.TestCase):
    def _reload(self, env):
        with mock.patch.dict(os.environ, env, clear=False):
            return importlib.reload(config)

    def tearDown(self):
        importlib.reload(config)

    def test_official_names_take_precedence(self):
        cfg = self._reload({"BASE_URL": "http://proxy/v1", "API_KEY": "team", "LLM_BASE_URL": "http://old", "LLM_API_KEY": "old"})
        self.assertEqual(cfg.LLM_BASE_URL, "http://proxy/v1")
        self.assertEqual(cfg.LLM_API_KEY, "team")

    def test_legacy_names_still_work(self):
        with mock.patch.dict(os.environ, {"LLM_BASE_URL": "http://old", "LLM_API_KEY": "old"}, clear=False):
            os.environ.pop("BASE_URL", None)
            os.environ.pop("API_KEY", None)
            cfg = importlib.reload(config)
        self.assertEqual(cfg.LLM_BASE_URL, "http://old")
        self.assertEqual(cfg.LLM_API_KEY, "old")


class TestOfficialCall(unittest.TestCase):
    def test_input_output_without_command(self):
        tmp = Path(tempfile.mkdtemp())
        inp, out = tmp / "cases.jsonl", tmp / "out" / "predictions.jsonl"
        inp.write_text(json.dumps({
            "id": "v1.1-row-1-B",
            "reference": {"text": "Der Bundesrat empfiehlt, die Initiative abzulehnen.", "language": "de"},
            "vote": "Initiative",
            "claim": {"text": "Le Conseil fédéral recommande de rejeter l'initiative.", "language": "fr"},
        }, ensure_ascii=False), encoding="utf-8")
        env = {**os.environ, "MOCK_APERTUS": "true"}
        proc = subprocess.run([sys.executable, "-m", "src", "--input", str(inp), "--output", str(out)],
                              cwd=TRACK_DIR, env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["id"] for r in rows], ["v1.1-row-1-B"])
        self.assertEqual(rows[0]["label_name"], ["entailment", "neutral", "contradiction"][rows[0]["label"]])
        self.assertTrue(all(e["page"] is None for e in rows[0]["evidence"]))  # task B: no page


class TestEvidenceItems(unittest.TestCase):
    PAGE = " ".join(f"Satz Nummer {i} mit etwas Inhalt." for i in range(60))

    def test_split_evenly_keeps_text_and_order(self):
        pieces = split_evenly(self.PAGE, 3)
        self.assertEqual(len(pieces), 3)
        self.assertEqual(" ".join(pieces), self.PAGE)
        self.assertLess(max(map(len, pieces)) - min(map(len, pieces)), 120)

    def test_cited_pages_are_split_and_capped(self):
        passages = [{"text": "clipped", "page_text": self.PAGE, "page_number": n} for n in (4, 7, 9)]
        with mock.patch.object(config, "EVIDENCE_SPLIT", "3,2"):
            items = evidence_items(passages, cited=[2, 1, 3], query="Satz")
        self.assertEqual(len(items), 5)
        self.assertEqual([p["page_number"] for _, p in items], [7, 7, 7, 4, 4])
        self.assertTrue(all(len(q) <= config.EVIDENCE_MAX_CHARS for q, _ in items))
        self.assertNotIn("clipped", [q for q, _ in items])  # the unclipped page text is quoted

    def test_free_slots_take_further_pages(self):
        passages = [{"text": self.PAGE, "page_text": self.PAGE, "page_number": n} for n in (4, 7, 9, 11)]
        with mock.patch.object(config, "EVIDENCE_SPLIT", "2,2"), mock.patch.object(config, "EVIDENCE_FILL", True):
            one_cited = evidence_items(passages, cited=[2], query="Satz")
            three_cited = evidence_items(passages, cited=[2, 1, 4], query="Satz")
        # one cited page in halves, then the best third of the retrieved pages by rank
        self.assertEqual([p["page_number"] for _, p in one_cited], [7, 7, 4, 9, 11])
        self.assertLess(len(one_cited[2][0]), len(self.PAGE) / 2)
        self.assertTrue(all("…" not in q and q in self.PAGE for q, _ in one_cited))  # verbatim excerpts
        # further cited pages come before the other retrieved pages
        self.assertEqual([p["page_number"] for _, p in three_cited], [7, 7, 4, 4, 11])

    def test_reference_chunks_stay_whole(self):
        chunks = [{"text": "Erster Abschnitt.", "page_number": None}, {"text": "Zweiter Abschnitt.", "page_number": None}]
        self.assertEqual([q for q, _ in evidence_items(chunks, cited=[2], query="x")], ["Zweiter Abschnitt."])

    def test_overlong_page_is_clipped_verbatim(self):
        long_page = " ".join(f"Wort{i} steht in einem langen Satz." for i in range(400))
        with mock.patch.object(config, "EVIDENCE_SPLIT", "1"):
            (quote, _), = evidence_items([{"text": long_page, "page_number": 1}], cited=[1], query="Wort5")
        self.assertLessEqual(len(quote), config.EVIDENCE_MAX_CHARS)
        self.assertNotIn("…", quote)


if __name__ == "__main__":
    unittest.main()
