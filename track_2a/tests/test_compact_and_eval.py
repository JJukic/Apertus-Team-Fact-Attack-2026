"""
Unit tests for compact prompt mode, reference chunking, evaluator helpers and the HF split.
No network access: the OpenAI client is replaced by a stub.
"""

import unittest
from types import SimpleNamespace

from src.apertus_client import ApertusClient
from src import config
from src.evaluator import evidence_grounded, load_records, stratified_sample
from src.hf_dataset import DEV_PINNED_DATES, booklet_filename, split_dates
from src.inference import ClaimVerificationEngine, chunk_reference
from src.text_utils import best_snippet, split_passages


def _fake_response(content: str, top: dict):
    top_logprobs = [SimpleNamespace(token=t, logprob=lp) for t, lp in top.items()]
    choice = SimpleNamespace(
        message=SimpleNamespace(content=content),
        logprobs=SimpleNamespace(content=[SimpleNamespace(token=content[:1], logprob=0.0, top_logprobs=top_logprobs)]),
    )
    return SimpleNamespace(choices=[choice], usage=SimpleNamespace(prompt_tokens=100, completion_tokens=4))


class _StubCompletions:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _stub_client(content: str, top: dict) -> ApertusClient:
    client = ApertusClient(api_key="test", mock=True)
    client.mock = False
    completions = _StubCompletions(_fake_response(content, top))
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client


class TestCompactInference(unittest.TestCase):
    def test_label_ids_and_probs_are_parsed(self):
        client = _stub_client("2|P2,P3,P9", {"2": -0.1, "1": -2.5, "0": -6.0})
        out = client.infer_compact(["a", "b", "c"], "claim", claim_language="de")
        self.assertEqual(out.label, 2)
        self.assertEqual(out.evidence_ids, [2, 3])  # P9 is out of range and dropped
        self.assertEqual(out.evidence, ["b", "c"])
        self.assertAlmostEqual(out.p_entail + out.p_neutral + out.p_contra, 1.0, places=3)
        self.assertGreater(out.p_contra, out.p_neutral)
        self.assertEqual(out.tokens_prompt, 100)

    def test_neutral_returns_no_evidence(self):
        engine = ClaimVerificationEngine(apertus_client=_stub_client("1|", {"1": -0.05, "2": -3.0}), prompt_mode="compact")
        res = engine.verify_premise("Die Steuer steigt.", "Absatz eins ist lang genug.\n\nAbsatz zwei ist auch lang genug.")
        self.assertEqual(res.label, 1)
        self.assertEqual(res.evidence, [])
        self.assertEqual(res.to_official_dict()["evidence"], [])

    def test_missing_ids_fall_back_to_top_passage(self):
        engine = ClaimVerificationEngine(apertus_client=_stub_client("0|", {"0": -0.01}), prompt_mode="compact")
        res = engine.verify_premise("x", "Erster Absatz mit genug Text darin.\n\nZweiter Absatz mit genug Text darin.")
        self.assertEqual(res.label, 0)
        self.assertEqual(len(res.evidence), 1)
        self.assertIn("evidence fallback", res.decision_rule)


class TestChunking(unittest.TestCase):
    def test_long_paragraphs_are_split_and_short_ones_merged(self):
        text = "Kurz.\n\n" + " ".join(["Dies ist ein Satz mit etwas Inhalt."] * 60)
        chunks = chunk_reference(text, max_chars=300, min_chars=80)
        self.assertGreater(len(chunks), 3)
        self.assertTrue(all(len(c) <= 400 for c in chunks))
        self.assertTrue(chunks[0].startswith("Kurz."))  # merged into the following chunk

    def test_split_passages_dehyphenates_and_merges_short_tail(self):
        text = "Die Personenfreizü -\ngigkeit gilt seit 2002. " + "Ein langer Satz folgt hier. " * 30 + "Ende."
        chunks = split_passages(text, max_chars=200, min_chars=80)
        self.assertIn("Personenfreizügigkeit", chunks[0])
        self.assertTrue(all(len(c) >= 80 for c in chunks))
        self.assertEqual(split_passages("Kurzer Text mit zwei Sätzen. Und noch einer.", max_chars=30, min_chars=80),
                         ["Kurzer Text mit zwei Sätzen. Und noch einer."])

    def test_best_snippet_matches_across_languages(self):
        page = (
            "20 Texte soumis au vote Texte soumis au vote Texte soumis au vote Premier objet. "
            "Le Conseil fédéral a examiné l'initiative populaire déposée en 2024 avec attention. "
            "La population résidante permanente de la Suisse ne doit pas dépasser dix millions de personnes avant l'année 2050."
        )
        claim = "La popolazione residente permanente della Svizzera non potrà superare i 10 milioni prima del 2050."
        snippet = best_snippet(page, claim)
        self.assertTrue(snippet.startswith("La population résidante permanente"))

    def test_short_reference_is_single_chunk(self):
        self.assertEqual(chunk_reference("Ein einziger Absatz."), ["Ein einziger Absatz."])


class TestEvaluatorHelpers(unittest.TestCase):
    def test_stratified_sample_is_balanced_and_deterministic(self):
        records = [{"label": i % 3, "id": i} for i in range(300)]
        a = stratified_sample(records, 30, seed=1)
        b = stratified_sample(records, 30, seed=1)
        self.assertEqual([r["id"] for r in a], [r["id"] for r in b])
        self.assertEqual(sorted(r["label"] for r in a).count(0), 10)

    def test_evidence_grounded(self):
        ref = "Der Bundesrat empfiehlt, die Initiative abzulehnen. Die Kosten betragen 5 Milliarden Franken."
        self.assertTrue(evidence_grounded(["Die Kosten betragen 5 Milliarden Franken."], ref))
        self.assertFalse(evidence_grounded(["Le Conseil fédéral recommande de rejeter l'initiative."], ref))


    def test_demo_dataset_booklets_resolve(self):
        # `make run` benchmarks the demo set; its publish date (2026-05-28) differs from the booklet file date
        records = load_records(config.BENCHMARK_PATH)
        self.assertEqual(len(records), 28)
        self.assertTrue(all(r["booklet_pdf"].exists() for r in records), {r["booklet_pdf"].name for r in records})


class TestHFSplit(unittest.TestCase):
    def test_booklet_filename(self):
        url = "https://www.bk.admin.ch/dam/fr/sd-web/WeUrKyC0FyPc/2026-06-14_explications_du_conseil_federal.pdf"
        self.assertEqual(booklet_filename(url, "2026-06-14"), "2026-06-14_fr.pdf")

    def test_split_is_deterministic_and_excludes_tuned_booklets(self):
        dates = [f"20{y:02d}-0{m}-01" for y in range(20, 27) for m in (3, 6, 9)] + sorted(DEV_PINNED_DATES)
        test_a, test_b = split_dates(dates), split_dates(dates)
        self.assertEqual(test_a, test_b)
        self.assertFalse(set(test_a) & DEV_PINNED_DATES)


if __name__ == "__main__":
    unittest.main()
