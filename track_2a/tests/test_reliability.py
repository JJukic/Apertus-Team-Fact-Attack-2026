import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from test_compact_and_eval import _stub_client
from src.inference import ClaimVerificationEngine

class TestReliability(unittest.TestCase):
    def test_replaced_pdf_invalidates_cache(self):
        engine = ClaimVerificationEngine(apertus_client=_stub_client('1|', {}))
        def parsed(path, parser):
            text = Path(path).read_text()
            return {'paragraphs': [{'text': text, 'page_number': 1}], 'pages': [], 'full_text': text}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'same.pdf'
            with patch('src.inference.load_parsed_booklet', side_effect=parsed):
                path.write_text('first unique booklet')
                first = engine._get_booklet_data(path)
                path.write_text('second unique booklet')
                second = engine._get_booklet_data(path)
        self.assertEqual(first['full_text'], 'first unique booklet')
        self.assertEqual(second['full_text'], 'second unique booklet')

    def test_invalid_answers_are_errors(self):
        for answer in ('nonsense', '{"label": 3}', '{"label": true}', '{"label": 0, "p_entail": 2}', '{"label": 0, "p_entail": NaN}', '{"label": 0, "evidence_ids": [1.5]}', '{"label": 0, "evidence_ids": [99]}'):
            with self.subTest(answer=answer):
                out = _stub_client(answer, {}).infer('', 'claim', passages=['text'])
                self.assertIsNotNone(out.error)
                self.assertEqual(out.evidence, [])

    def test_valid_ids_remain_grounded(self):
        out = _stub_client('{"label": 0, "p_entail": 1, "evidence_ids": [2, 2]}', {}).infer('', 'claim', passages=['first', 'second'])
        self.assertIsNone(out.error)
        self.assertEqual(out.evidence, ['second'])

    def test_invalid_compact_ids_are_errors(self):
        for answer in ('0|P9', '0|P1.5', '0 garbage', 'text'):
            with self.subTest(answer=answer):
                out = _stub_client(answer, {}).infer_compact(['text'], 'claim')
                self.assertIsNotNone(out.error)
                self.assertEqual(out.evidence_ids, [])
