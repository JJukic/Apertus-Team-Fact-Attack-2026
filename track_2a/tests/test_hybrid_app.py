"""UI selection is lazy; local embedding errors are recoverable. No API calls."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from streamlit.testing.v1 import AppTest
from src.embeddings import EmbeddingError


class TestHybridApp(unittest.TestCase):
    def test_strategy_options_keep_default_and_selection_does_not_encode(self):
        with patch("src.embeddings.SentenceTransformerBackend.encode", side_effect=AssertionError("No model on selection")), \
             patch("src.apertus_client.ApertusClient.infer", side_effect=AssertionError("No API on selection")):
            app = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=20).run()
            self.assertFalse(app.exception)
            self.assertEqual(app.radio(key="context_strategy").value, "retrieval")
            app.radio(key="context_strategy").set_value("hybrid_dense").run()
            self.assertFalse(app.exception)
            self.assertEqual(app.radio(key="context_strategy").value, "hybrid_dense")
            self.assertIsNotNone(app.text_input(key="retrieval_vote_title"))

    def test_embedding_failure_is_shown_without_api_or_app_crash(self):
        with patch("src.embeddings.SentenceTransformerBackend.encode", side_effect=EmbeddingError("Test weights missing")), \
             patch("src.apertus_client.ApertusClient.infer", side_effect=AssertionError("No API on embedding failure")):
            app = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=20).run()
            app.radio(key="context_strategy").set_value("hybrid_dense").run()
            # Isolate from an on-disk index created by a manual CPU warm-up.
            with patch("src.embeddings.DenseIndex.ensure", side_effect=EmbeddingError("Test weights missing")):
                app.button(key="verify_claim").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any("Test weights missing" in error.value for error in app.error))
