import importlib.util
import unittest
from pathlib import Path
spec = importlib.util.spec_from_file_location('measure_retrieval', Path(__file__).resolve().parents[1] / 'scripts/measure_retrieval.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class TestRetrievalMeasurement(unittest.TestCase):
    def test_pdf_hyphenation_and_whitespace_are_normalized(self):
        source = 'Die Wohnbevölkerung in der Schweiz wächst und der Bundesrat empfiehlt diese Vorlage abzulehnen.'
        wrapped = source.replace('Wohnbevölkerung', 'Wohnbevöl-\nkerung').replace(' ', '\n')
        self.assertEqual(module.anchors(source), module.anchors(wrapped))
    def test_short_or_reordered_reference_is_not_a_match(self):
        self.assertEqual(module.anchors('very short reference'), set())
        source = 'eins zwei drei vier fünf sechs sieben acht neun zehn elf zwölf'
        self.assertFalse(module.anchors(source) & module.anchors(' '.join(reversed(source.split()))))
