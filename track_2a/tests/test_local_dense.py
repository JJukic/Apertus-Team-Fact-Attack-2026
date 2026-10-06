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
