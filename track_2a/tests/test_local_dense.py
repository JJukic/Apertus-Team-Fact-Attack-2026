import unittest
import numpy as np
from src.local_dense import normalized, page_scores, ranking, minmax, fuse, select

class TestLocalDense(unittest.TestCase):
    def test_page_scores_use_maximum_not_chunk_count(self):
        scores = page_scores([.3, .3, .3, .8], [0, 0, 0, 1], 2)
        self.assertEqual(ranking(scores), [1, 0])
        with self.assertRaises(ValueError): page_scores([.2], [0, 1], 2)
    def test_normalization_rejects_invalid_vectors(self):
        for matrix in ([[0, 0]], [[float('nan'), 1]], [1, 2]):
            with self.assertRaises(ValueError): normalized(matrix)
        self.assertTrue(np.allclose(np.linalg.norm(normalized([[3, 4]]), axis=1), [1]))
        self.assertTrue(np.array_equal(minmax([.7, .7]), [0, 0]))
    def test_rrf_deduplicates_and_ties_are_stable(self):
        self.assertEqual(fuse([[1, 1, 0], [0, 1]]), [0, 1])
        with self.assertRaises(ValueError): fuse([[0]], 0)
    def test_speaker_boost_does_not_expand_context(self):
        passages = [{'section': 'other'}, {'section': 'other'}, {'section': 'committee'}]
        chosen = select([0, 1, 2], passages, 2, section='committee', boost=2)
        self.assertEqual(chosen, [passages[0], passages[2]])
        self.assertEqual(select([0, 1, 2], passages, 2, excluded={'committee'}), passages[:2])

    def test_title_weight_changes_order_and_returns_original_objects(self):
        from src.local_dense import retrieve_dense_title
        class Backend:
            def document(self, passages):
                return np.eye(3, dtype=np.float32), [0, 1, 2], 'fake'
            def encode(self, texts, kind):
                return np.asarray([[1, 0, 0], [0, 0, 1]][:len(texts)], dtype=np.float32)
        passages = [{'text': 'first'}, {'text': 'second'}, {'text': 'third'}]
        self.assertIs(retrieve_dense_title(passages, 'claim', 'title', 1, backend=Backend(), boost=0)[0], passages[2])
        self.assertIs(retrieve_dense_title(passages, 'claim', top_k=1, backend=Backend(), boost=0)[0], passages[0])
        with self.assertRaises(ValueError): retrieve_dense_title(passages, '', backend=Backend())

    def test_engine_preview_never_calls_language_model(self):
        from src.inference import ClaimVerificationEngine
        from src.apertus_client import ApertusClient
        from src import config
        from unittest.mock import Mock
        class Backend:
            def document(self, passages):
                self.count = len(passages)
                return np.eye(self.count, dtype=np.float32), list(range(self.count)), 'fake'
            def encode(self, texts, kind):
                return np.asarray([np.eye(self.count)[0] for text in texts], dtype=np.float32)
        client = ApertusClient(mock=True)
        client.infer = Mock(side_effect=AssertionError('preview must not classify'))
        engine = ClaimVerificationEngine(apertus_client=client, dense_backend=Backend())
        selected = engine.retrieve_context('Une initiative pour la Suisse.', config.BOOKLETS_DIR/'2026-06-14_de.pdf', 'dense_title', 3, 'Initiative')
        self.assertEqual(len(selected), 3)
        self.assertTrue(all(p['page_number'] > 0 and p['text'] for p in selected))
        client.infer.assert_not_called()
        from src.apertus_client import NLIOutput
        client.infer.side_effect = None
        client.infer.return_value = NLIOutput(label=0, reasoning="simulated", evidence_ids=[1])
        result = engine.verify_claim('Une initiative pour la Suisse.', config.BOOKLETS_DIR/'2026-06-14_de.pdf', strategy='dense_title', top_k=3, vote='Initiative')
        self.assertEqual(result.strategy, 'dense_title')
        self.assertIsNone(result.error)
        self.assertEqual(result.to_official_dict()['evidence'][0]['page'], selected[0]['page_number'])
        self.assertGreaterEqual(result.retrieval_ms, 0)


    def test_similarity_matches_reference_and_rejects_invalid_queries(self):
        from src.local_dense import cosine_scores
        matrix = normalized([[1, 2, 3], [4, 5, 6]])
        query = normalized([[2, 4, 1]])[0]
        self.assertTrue(np.allclose(cosine_scores(matrix, query), matrix @ query, atol=1e-6))
        with self.assertRaises(ValueError): cosine_scores(matrix, [1, 2])
        with self.assertRaises(ValueError): cosine_scores(matrix, [1, float('nan'), 3])
