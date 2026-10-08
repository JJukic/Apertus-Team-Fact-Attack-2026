import hashlib
import json
from pathlib import Path
import unittest

from src.official_evaluation import evidence_hit5, macro_f1, normalize, quote_matches


class TestOfficialEvaluation(unittest.TestCase):
    def test_vendored_evaluator_matches_pinned_source(self):
        folder = Path(__file__).resolve().parents[1] / 'vendor' / 'hackapertus_starter'
        manifest = json.loads((folder / 'provenance.json').read_text())
        self.assertEqual(hashlib.sha256((folder / 'evaluate.py').read_bytes()).hexdigest(), manifest['sha256'])
        self.assertEqual(manifest['commit'], '5e6957729b77a126006049305ad1ef241211a5b4')

    def test_first_five_positions_include_invalid_items(self):
        reference = 'Der Bundesrat empfiehlt die Vorlage abzulehnen.'
        prediction = {'evidence': [None] * 5 + [{'page': 43, 'text': reference}]}
        self.assertFalse(evidence_hit5(prediction, reference))
        prediction['evidence'][4] = {'page': 43, 'text': reference}
        self.assertTrue(evidence_hit5(prediction, reference))

    def test_normalization_unicode_soft_hyphen_and_hyphenation(self):
        self.assertEqual(normalize('  Ｃｏｎｓｅｉｌ\u00ad  fédé-\nral\t '), 'conseil fédéral')
        self.assertTrue(quote_matches('Consiglio\n federale', 'Il Consiglio federale raccomanda il sì.', 90, 5000))

    def test_length_limit_applies_after_normalization(self):
        self.assertTrue(quote_matches('x' + ' ' * 6000, 'x', 90, 5000))
        self.assertTrue(quote_matches('x' * 5000, 'x', 90, 5000))
        self.assertFalse(quote_matches('x' * 5001, 'x', 90, 5000))
        # Compatibility normalization expands this ligature to three characters.
        self.assertFalse(quote_matches('ﬃ' * 1700, 'ffi', 90, 5000))

    def test_empty_and_duplicate_evidence(self):
        reference = 'Le Conseil fédéral recommande de rejeter cette initiative.'
        self.assertFalse(evidence_hit5({'evidence': []}, reference))
        evidence = {'page': 19, 'text': reference}
        self.assertTrue(evidence_hit5({'evidence': [evidence, evidence]}, reference))

    def test_missing_response_counts_as_wrong_instead_of_being_dropped(self):
        score, classes = macro_f1([(0, 0), (1, 1), (2, -1)])
        self.assertAlmostEqual(score, 2 / 3)
        self.assertEqual(classes['contradiction']['support'], 1)
        self.assertEqual(classes['contradiction']['recall'], 0)


if __name__ == '__main__':
    unittest.main()
