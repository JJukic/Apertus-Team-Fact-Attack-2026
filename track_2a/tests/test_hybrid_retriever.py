"""Controlled vectors test mechanics only, never multilingual model quality."""

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.embeddings import DenseIndex, EmbeddingError, HybridSettings, normalize_embeddings, _load_model
from src.hybrid_retriever import HybridPageRetriever, page_context, reciprocal_rank_fusion
from src.inference import ClaimVerificationEngine, PredictionResult, _GLOBAL_BOOKLET_CACHE
from src.apertus_client import ApertusClient, NLIOutput
from src.evaluator import BenchmarkEvaluator
from src import config


class FakeEmbeddings:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return np.array([[2, 0] if "target" in t.lower() else [0, 3] for t in texts], dtype=float)


class HybridFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.settings = HybridSettings(model="test/fake", revision="fake-v1", cache_dir=Path(self.temp.name),
                                       final_pages=2, candidates=3)
        self.pages = [
            {"page_number": 7, "text": "Target evidence original\npage text preserved."},
            {"page_number": 2, "text": "Unrelated legislation and independent arguments."},
            {"page_number": 9, "text": "Another unrelated original page text."},
        ]
        self.paras = [
            {"id": 0, "page_number": 7, "text": "Target evidence original page text preserved.",
             "proposal_id": 2, "section_type": "arguments", "speaker": "committee"},
            {"id": 1, "page_number": 2, "text": self.pages[1]["text"], "proposal_id": 1, "section_type": "detail"},
            {"id": 2, "page_number": 9, "text": self.pages[2]["text"], "proposal_id": 3, "section_type": "legal_text"},
        ]
        self.backend = FakeEmbeddings()

    def retriever(self, **kwargs):
        return HybridPageRetriever(self.pages, self.paras, "pdf-content-hash", self.settings,
                                   kwargs.get("backend", self.backend))

    def index(self, **kwargs):
        return DenseIndex(kwargs.get("source_hash", "pdf-content-hash"), kwargs.get("paras", self.paras),
                          kwargs.get("parser_version", "parser-v1"), kwargs.get("settings", self.settings),
                          kwargs.get("backend", self.backend))


class TestHybridRanking(HybridFixture):
    def test_no_lexical_signal_does_not_inject_arbitrary_page_ranks(self):
        retriever = self.retriever()
        with patch.object(retriever.bm25, "get_scores", return_value=np.zeros(len(self.paras))):
            self.assertEqual(retriever.lexical_ranking("different script"), [])

    def test_rrf_is_deterministic_and_counts_page_once(self):
        rank, scores = reciprocal_rank_fusion([[7, 7, 2], [2, 7]], 60)
        self.assertEqual(rank, [2, 7])  # equal scores -> original page number
        self.assertAlmostEqual(scores[7], 1 / 61 + 1 / 62)
        self.assertEqual((rank, scores), reciprocal_rank_fusion([[7, 7, 2], [2, 7]], 60))

    def test_cosine_normalization_and_exact_search(self):
        matrix = normalize_embeddings([[3, 4], [0, 5]], 2)
        np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1)
        cosine, _ = self.index().search("target claim in another language")
        np.testing.assert_allclose(cosine, [1, 0, 0])

    def test_title_weight_is_two_and_not_a_filter(self):
        retriever = self.retriever()
        with patch.object(retriever.bm25, "get_scores", side_effect=[[3, 1, 0], [0, 2, 0]]):
            ranking = retriever.lexical_ranking("target", "other proposal")
        self.assertEqual(ranking, [2, 7, 9])
        self.assertIn(7, ranking)  # despite belonging to a different proposal

    def test_max_aggregation_does_not_sum_paragraph_votes(self):
        retriever = self.retriever()
        retriever.paragraphs += [self.paras[0].copy() for _ in range(10)]
        ranking = retriever._page_ranking([0.6, 0.9, 0.1] + [0.6] * 10)
        self.assertEqual(ranking, [2, 7, 9])

    def test_original_pages_metadata_and_source_identity(self):
        pages, metrics = self.retriever().retrieve("target claim")
        self.assertEqual(len({p["source_id"] for p in pages}), len(pages))
        page = next(p for p in pages if p["page_number"] == 7)
        self.assertEqual(page["text"], self.pages[0]["text"])
        self.assertEqual(page["proposal_id"], 2)
        self.assertEqual(page["speaker"], "committee")
        self.assertIn("[Page 7] [Section arguments] [Proposal 2] [Speaker committee]", page_context([page]))
        self.assertEqual(metrics["actual_strategy"], "hybrid_dense")

    def test_duplicate_page_objects_are_deduplicated(self):
        retriever = HybridPageRetriever(self.pages + [self.pages[0]], self.paras, "hash", self.settings, self.backend)
        self.assertEqual(len(retriever.pages), 3)

    def test_conflicting_source_page_is_rejected(self):
        with self.assertRaises(ValueError):
            HybridPageRetriever(self.pages + [{"page_number": 7, "text": "different"}], self.paras,
                                "hash", self.settings, self.backend)

    def test_lexical_strategy_never_encodes(self):
        backend = MagicMock()
        backend.encode.side_effect = AssertionError("must not load model")
        self.retriever(backend=backend).retrieve("target", dense=False)
        backend.encode.assert_not_called()

    def test_missing_embeddings_fail_explicitly(self):
        backend = MagicMock()
        backend.encode.return_value = None
        with self.assertRaises(EmbeddingError):
            self.retriever(backend=backend).retrieve("target")

    def test_fallback_reports_actual_strategy_and_logs(self):
        backend = MagicMock()
        backend.encode.side_effect = EmbeddingError("weights missing")
        retriever = HybridPageRetriever(self.pages, self.paras, "hash",
                                       replace(self.settings, fallback=True), backend)
        with self.assertLogs("src.hybrid_retriever", level="WARNING") as logs:
            _, metrics = retriever.retrieve("target")
        self.assertEqual(metrics["requested_strategy"], "hybrid_dense")
        self.assertEqual(metrics["actual_strategy"], "hybrid")
        self.assertIn("weights missing", metrics["fallback_reason"])
        self.assertIn("actual strategy=hybrid", logs.output[0])

    def test_empty_document_and_claim_are_actionable(self):
        retriever = HybridPageRetriever([], [], "hash", self.settings, self.backend)
        with self.assertRaisesRegex(EmbeddingError, "no extractable"):
            retriever.retrieve("target")
        with self.assertRaises(ValueError):
            self.retriever().retrieve(" ")
        self.assertEqual(self.backend.calls, [])


class TestDenseCache(HybridFixture):
    def test_real_backend_rejects_moving_model_revision_before_import_or_download(self):
        with self.assertRaisesRegex(EmbeddingError, "immutable"):
            _load_model("BAAI/bge-m3", "main", "cpu", 2048, True)

    def test_persistent_cache_and_warm_query_skip_document_encoding(self):
        index = self.index()
        self.assertEqual(index.ensure()["cache_status"], "miss")
        self.assertEqual(index.ensure()["cache_status"], "memory_hit")
        backend = FakeEmbeddings()
        second = self.index(backend=backend)
        _, stats = second.search("target")
        self.assertEqual(stats["cache_status"], "disk_hit")
        self.assertEqual(backend.calls, [["target"]])
        self.assertEqual(stats["document_embedding_ms"], 0)

    def test_cache_key_changes_for_pdf_parser_model_revision_and_chunking(self):
        key = self.index().key
        variants = [
            self.index(source_hash="new-pdf"), self.index(parser_version="new-parser"),
            self.index(settings=replace(self.settings, model="another/model")),
            self.index(settings=replace(self.settings, revision="fake-v2")),
            self.index(settings=replace(self.settings, max_length=1000)),
            self.index(paras=[dict(self.paras[0], text="Changed text")]),
        ]
        self.assertTrue(all(index.key != key for index in variants))

    def test_corrupt_cache_is_logged_and_rebuilt(self):
        index = self.index()
        index.path.write_bytes(b"corrupt cache")
        with self.assertLogs("src.embeddings", level="WARNING"):
            stats = index.ensure()
        self.assertEqual(stats["cache_status"], "rebuilt_corrupt")
        self.assertEqual(self.index().ensure()["cache_status"], "disk_hit")

    def test_cache_checksum_detects_changed_values(self):
        index = self.index()
        index.ensure()
        with np.load(index.path, allow_pickle=False) as cache:
            metadata, checksum = cache["metadata"], cache["checksum"]
        np.savez(index.path, metadata=metadata, checksum=checksum, vectors=np.ones((3, 2)))
        with self.assertLogs("src.embeddings", level="WARNING"):
            self.assertEqual(self.index().ensure()["cache_status"], "rebuilt_corrupt")

    def test_dimension_mismatch_is_not_silently_used(self):
        index = self.index()
        index.ensure()
        index.backend = MagicMock()
        index.backend.last_load_ms = 0
        index.backend.encode.return_value = [[1, 0, 0]]
        with self.assertRaisesRegex(EmbeddingError, "dimensions differ"):
            index.search("target")

    def test_invalid_vectors_rejected(self):
        for values in ([[0, 0]], [[float("nan"), 1]], [[float("inf"), 1]], None, []):
            with self.subTest(values=values), self.assertRaises(EmbeddingError):
                normalize_embeddings(values, 1)


class TestHybridPipeline(HybridFixture):
    def test_empty_pdf_stops_before_any_nli_call(self):
        from pypdf import PdfWriter
        path = Path(self.temp.name) / "empty.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.write(path)
        client = MagicMock()
        engine = ClaimVerificationEngine(apertus_client=client, hybrid_settings=self.settings,
                                         embedding_backend=self.backend)
        with self.assertRaisesRegex(EmbeddingError, "no extractable"):
            engine.verify_claim("target", path, strategy="hybrid_dense")
        client.infer.assert_not_called()
        self.assertEqual(self.backend.calls, [])

    def engine(self, evidence=None, backend=None, settings=None):
        client = MagicMock()
        client.mock = True
        client.infer.return_value = NLIOutput(
            label=0, reasoning="Controlled test response", evidence=evidence or ["[Page 99] Target evidence original page text preserved."],
            p_entail=0.9, latency_ms=10, tokens_prompt=100, tokens_completion=20, tokens_total=120,
        )
        engine = ClaimVerificationEngine(apertus_client=client, hybrid_settings=settings or self.settings,
                                         embedding_backend=backend or self.backend)
        engine._get_booklet_data = MagicMock(return_value={
            "pages": self.pages, "paragraphs": self.paras, "source_hash": "pdf-content-hash",
            "full_text": "original full context", "retriever": MagicMock(),
        })
        engine._get_booklet_data.return_value["retriever"].retrieve.return_value = [self.paras[0]]
        return engine

    def test_one_inference_call_and_correct_original_page_not_rank_or_model_tag(self):
        engine = self.engine()
        result = engine.verify_claim("target", "fake.pdf", strategy="hybrid_dense")
        engine.client.infer.assert_called_once()
        self.assertEqual(result.evidence_sources[0].page_number, 7)
        self.assertEqual(result.evidence_sources[0].proposal_id, 2)
        self.assertEqual(result.evidence_sources[0].section_type, "arguments")
        self.assertEqual(result.evidence_sources[0].speaker, "committee")
        official = result.to_official_dict()
        self.assertEqual(set(official), {"id", "label", "label_name", "evidence", "metrics"})
        self.assertEqual(set(official["metrics"]), {"input_tokens", "output_tokens", "inference_time_ms"})
        self.assertEqual(official["evidence"][0]["page"], 7)
        self.assertEqual(result.latency_ms, 10)  # original API metric meaning
        self.assertGreater(result.total_latency_ms, 0)

    def test_invalid_quote_cannot_create_a_page_one_citation(self):
        engine = self.engine(["This sentence is not in any selected original page."])
        result = engine.verify_claim("target", "fake.pdf", strategy="hybrid_dense")
        self.assertEqual(result.label, 1)
        self.assertEqual(result.to_official_dict()["evidence"], [])

    def test_existing_strategies_keep_context_and_never_load_embeddings(self):
        engine = self.engine()
        for strategy in ("retrieval", "full"):
            result = engine.verify_claim("target", "fake.pdf", strategy=strategy)
            self.assertEqual(result.strategy, strategy)
            context = engine.client.infer.call_args.kwargs["context"]
            self.assertEqual(context, "original full context" if strategy == "full" else
                             "[Page 7] Target evidence original page text preserved.")
        self.assertEqual(self.backend.calls, [])

    def test_beginner_task_unchanged_even_with_dense_strategy(self):
        engine = self.engine()
        engine.strategy = "hybrid_dense"
        result = engine.verify_premise("target", "Target evidence original page text preserved.")
        self.assertEqual(result.strategy, "direct_reference")
        engine._get_booklet_data.assert_not_called()
        self.assertEqual(self.backend.calls, [])

    def test_fallback_result_exposes_actual_strategy(self):
        backend = MagicMock()
        backend.encode.side_effect = EmbeddingError("offline")
        engine = self.engine(backend=backend, settings=replace(self.settings, fallback=True))
        with self.assertLogs("src.hybrid_retriever", level="WARNING"):
            result = engine.verify_claim("target", "fake.pdf", strategy="hybrid_dense")
        self.assertEqual(result.strategy, "hybrid")
        self.assertEqual(result.requested_strategy, "hybrid_dense")

    def test_source_cache_invalidates_replaced_pdf(self):
        pdf = Path(self.temp.name) / "booklet.pdf"
        pdf.write_bytes(b"first PDF content")
        engine = ClaimVerificationEngine(apertus_client=ApertusClient(mock=True))
        engine.pdf_parser = MagicMock(VERSION="fake-parser")
        engine.pdf_parser.extract_pages.return_value = self.pages
        engine.pdf_parser.extract_paragraphs.return_value = self.paras
        engine.pdf_parser.extract_full_text.return_value = "full"
        first = engine._get_booklet_data(pdf)
        pdf.write_bytes(b"second PDF content")
        second = engine._get_booklet_data(pdf)
        self.assertNotEqual(first["source_hash"], second["source_hash"])
        self.assertEqual(engine.pdf_parser.extract_pages.call_count, 2)
        _GLOBAL_BOOKLET_CACHE.clear()

    def test_bad_strategy_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown strategy"):
            self.engine().verify_claim("target", "fake.pdf", strategy="typo")


class TestExtendedEvaluation(HybridFixture):
    def test_gold_pages_are_not_compared_after_booklet_language_fallback(self):
        dataset = Path(self.temp.name) / "dev.jsonl"
        dataset.write_text(json.dumps({"claim": "target", "entailment_label": 0,
                                       "claim_language": "de", "booklet_language": "fr",
                                       "gold_evidence_pages": [7]}))
        with patch.object(config, "BOOKLETS_DIR", Path(self.temp.name)), \
             patch.object(BenchmarkEvaluator, "print_report"), \
             self.assertLogs("src.evaluator", level="WARNING"):
            report = BenchmarkEvaluator(self.engine_for_evaluation()).evaluate(dataset)
        self.assertIsNone(report["evidence_page_recall"])
        self.assertEqual(report["results"][0]["booklet_lang"], "de")
        self.assertEqual(report["results"][0]["requested_booklet_lang"], "fr")

    def test_language_pairs_and_gold_not_used_for_prediction(self):
        dataset = Path(self.temp.name) / "dev.jsonl"
        records = [
            {"claim": "target", "claim_language": "fr", "booklet_language": "de", "entailment_label": 0,
             "reference_string": "DO NOT USE THIS GOLD TEXT", "gold_evidence_pages": [7], "vote": "proposal title"},
            {"claim": "target", "claim_language": "de", "booklet_language": "de", "entailment_label": 2},
        ]
        dataset.write_text("\n".join(json.dumps(r) for r in records))
        engine = self.engine_for_evaluation()
        with patch.object(BenchmarkEvaluator, "print_report"):
            report = BenchmarkEvaluator(engine).evaluate(dataset, strategy="hybrid_dense")
        self.assertEqual(report["cross_language_sample_count"], 1)
        self.assertIn("fr->de", report["by_language_pair"])
        self.assertEqual(report["avg_latency_ms"], 10)
        self.assertEqual(report["avg_total_latency_ms"], 30)
        self.assertEqual(report["avg_output_tokens"], 20)
        self.assertEqual(report["evidence_page_recall"], 1)
        self.assertTrue(report["mock_mode"])
        for call in engine.verify_claim.call_args_list:
            self.assertNotIn("reference_string", call.kwargs)
            self.assertNotIn("entailment_label", call.kwargs)
            self.assertNotIn("gold_evidence_pages", call.kwargs)
        self.assertEqual(engine.verify_claim.call_args_list[0].kwargs["vote"], "proposal title")

    def engine_for_evaluation(self):
        from src.inference import EvidenceSource
        engine = MagicMock()
        engine.client.mock = True
        engine.verify_claim.return_value = PredictionResult(
            claim="target", label=0, label_name="entailment", reasoning="test", evidence=["target evidence"],
            evidence_sources=[EvidenceSource(quote="target evidence", page_number=7)],
            strategy="hybrid_dense", booklet_path="fake.pdf", tokens_prompt=100,
            tokens_completion=20, tokens_total=120, latency_ms=10, total_latency_ms=30,
            retrieval_metrics={"cache_status": "memory_hit", "retrieval_ms": 5},
        )
        return engine

    def test_no_cross_language_or_gold_pages_means_not_measured(self):
        dataset = Path(self.temp.name) / "dev.jsonl"
        dataset.write_text(json.dumps({"claim": "target", "entailment_label": 0}))
        with patch.object(BenchmarkEvaluator, "print_report"):
            report = BenchmarkEvaluator(self.engine_for_evaluation()).evaluate(dataset)
        self.assertIsNone(report["cross_language_macro_f1"])
        self.assertIsNone(report["evidence_page_recall"])


class TestHybridCLI(HybridFixture):
    def test_compare_requires_explicit_mode_and_rejects_test_split(self):
        from typer.testing import CliRunner
        from src.cli import app
        dataset = Path(self.temp.name) / "dev.jsonl"
        dataset.write_text(json.dumps({"claim": "target", "entailment_label": 0}))
        runner = CliRunner()
        with patch("src.apertus_client.ApertusClient.infer", side_effect=AssertionError("No paid call")):
            args = ["compare-hybrid", "--dataset", str(dataset)]
            self.assertNotEqual(runner.invoke(app, args + ["--split", "dev"]).exit_code, 0)
            self.assertNotEqual(runner.invoke(app, args + ["--split", "test", "--mock"]).exit_code, 0)

    def test_compare_schedules_identical_cases_and_saves_both_phases(self):
        from typer.testing import CliRunner
        from src.cli import app
        dataset = Path(self.temp.name) / "dev.jsonl"
        output = Path(self.temp.name) / "comparison.json"
        dataset.write_text(json.dumps({"claim": "target", "entailment_label": 0, "split": "dev"}))
        with patch.object(BenchmarkEvaluator, "evaluate", return_value={"sample_count": 1}) as evaluate:
            result = CliRunner().invoke(app, ["compare-hybrid", "--dataset", str(dataset), "--split", "dev",
                                              "--mock", "--output", str(output)])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual([c.args for c in evaluate.call_args_list], [
            (dataset, "hybrid", None), (dataset, "hybrid", None),
            (dataset, "hybrid_dense", None), (dataset, "hybrid_dense", None),
        ])
        report = json.loads(output.read_text())
        self.assertTrue(report["mock_mode"])
        self.assertEqual(set(report["reports"]["hybrid_dense"]), {"first_pass", "warm_repeat"})


if __name__ == "__main__":
    unittest.main()
