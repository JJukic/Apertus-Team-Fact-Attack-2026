import unittest
from src.error_analysis import filter_cases, speaker

class TestErrorAnalysis(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {'id': 'wrong', 'claim': 'Das Komitee sagt etwas.', 'vote': 'Steuer', 'pair': 'de->fr', 'true_label': 0, 'pred_label': 2},
            {'id': 'ok', 'claim': 'Le Conseil fédéral recommande non.', 'pair': 'fr->de', 'true_label': 2, 'pred_label': 2},
            {'id': 'failure', 'claim': 'Test', 'pair': 'it->de', 'true_label': 0, 'pred_label': 1, 'error': 'api failure'},
        ]
    def test_failures_are_not_classification_mistakes(self):
        self.assertEqual([r['id'] for r in filter_cases(self.rows)], ['wrong'])
        self.assertEqual([r['id'] for r in filter_cases(self.rows, status='Technische Fehler')], ['failure'])
    def test_combined_filters_and_search(self):
        self.assertEqual(len(filter_cases(self.rows, pair='de->fr', expected=0, predicted=2, actor='Komitee', query='STEUER')), 1)
        self.assertEqual(filter_cases(self.rows, expected=1), [])
    def test_actor_and_correct_cases(self):
        self.assertEqual(speaker(self.rows[1]), 'Bundesrat / Parlament')
        self.assertEqual([r['id'] for r in filter_cases(self.rows, status='Korrekte Vorhersagen')], ['ok'])
