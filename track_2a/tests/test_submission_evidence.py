"""Submission invariants exercised against physical, generated PDF pages."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    import pymupdf  # optional (requirements-evidence.txt): AGPL, so not in the submission image
except ImportError:
    raise unittest.SkipTest("pymupdf not installed")

from src import config
from src.apertus_client import ApertusClient
from src.evidence import SourcePages
from src.inference import ClaimVerificationEngine, EvidenceSource, PredictionResult
from src.official_evaluation import normalize
from src.text_utils import official_normalize


ROOT = Path(__file__).resolve().parents[1]


def result(**kwargs):
    values = dict(id='', claim='Le Conseil federal recommande le rejet.', label=0,
                  label_name='incorrect-name', reasoning='', evidence=[],
                  strategy='hybrid', booklet_path='source.pdf', tokens_prompt=10,
                  tokens_completion=5, tokens_total=15, latency_ms=10)
    values.update(kwargs)
    return PredictionResult(**values)


def pdf(path, texts):
    with pymupdf.open() as document:
        for text in texts:
            document.new_page().insert_text((50, 80), text)
        document.save(path)


class TestSubmissionEvidence(unittest.TestCase):
    def test_export_preserves_source_digits_and_quote_marks_and_id(self):
        quote = '2050: «Le Conseil fédéral recommande le rejet.»\n'
        prediction = result(evidence_sources=[EvidenceSource(quote=quote, page_number=43)])
        output = prediction.to_official_dict()
        self.assertEqual(output['id'], '')
        self.assertEqual(output['label_name'], 'entailment')
        self.assertEqual(output['evidence'], [{'page': 43, 'text': quote}])
        self.assertEqual(prediction.to_official_dict(case_id='explicit')['id'], 'explicit')

    def test_unknown_pages_do_not_export_and_the_label_is_kept_without_evidence(self):
        for page in (None, 0, -1):
            prediction = result(evidence=['Unattributed text'],
                                evidence_sources=[EvidenceSource(quote='Original text', page_number=page)])
            with patch.object(config, 'OFFICIAL_IO', False):
                self.assertEqual(prediction.to_official_dict()['evidence'], [])
            # evaluate.py scores the label regardless of evidence: no error (which would become a neutral record)
            with patch.object(config, 'OFFICIAL_IO', True):
                self.assertEqual(prediction.to_official_dict()['label'], 0)

    def test_beginner_does_not_invent_a_pdf_page(self):
        prediction = result(strategy='direct_reference', evidence_sources=[EvidenceSource(quote='Source text')])
        self.assertEqual(prediction.to_official_dict()['evidence'], [{'page': None, 'text': 'Source text'}])

    def test_export_deduplicates_and_applies_normalized_length_and_first_five(self):
        sources = [EvidenceSource(quote='ﬃ' * 1700, page_number=43)]
        sources += [EvidenceSource(quote=f'Original text {i}', page_number=i) for i in [2, 2, 3, 4, 5, 6, 7]]
        output = result(evidence_sources=sources).to_official_dict()
        self.assertEqual([e['page'] for e in output['evidence']], [2, 3, 4, 5, 6])

    def test_source_extraction_and_engine_integration_preserve_physical_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.pdf'
            pdf(path, ['Summary unrelated to the claim.', '2050: Le Conseil federal recommande le rejet.'])
            source = SourcePages(Path(directory))
            case = {'booklet': {'path': 'source.pdf'}}
            text = source.get(case, 2)['text']
            engine = ClaimVerificationEngine(apertus_client=ApertusClient(mock=True))
            prediction = result(booklet_path=str(path), evidence_sources=[EvidenceSource(quote='clipped', page_number=2)])
            original_scores = prediction.model_dump(include={'label', 'p_entail', 'p_neutral', 'p_contra', 'tokens_total'})
            with patch.object(config, 'EVIDENCE_POLICY', 'raw_pages_and_blocks'):
                updated = engine._finalize_evidence(prediction, vote='Vorlage')
            self.assertEqual(updated.to_official_dict()['evidence'], [{'page': 2, 'text': text}])
            self.assertEqual(updated.model_dump(include=set(original_scores)), original_scores)
            self.assertIsNone(source.get(case, 99))

    def test_lazy_extraction_is_identical_and_caches_only_requested_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.pdf'
            pdf(path, ['Unrelated first page.', 'Le Conseil federal recommande le rejet.', 'Third page.'])
            eager, lazy = SourcePages(directory), SourcePages(directory, lazy=True)
            case = {'booklet': {'path': 'source.pdf'}}
            with patch.object(lazy, '_extract_page', wraps=lazy._extract_page) as extract:
                self.assertEqual(lazy.get(case, 2), eager.get(case, 2))
                self.assertEqual(lazy.get(case, 2), eager.get(case, 2))
                self.assertEqual(extract.call_count, 1)
                self.assertEqual(len(lazy.cache), 1)
                self.assertIsNone(lazy.get(case, 99))
                self.assertEqual(lazy.get(case, 1), eager.get(case, 1))

    def test_normalization_matches_pinned_evaluator_for_multilingual_edge_cases(self):
        for text in ['Ｃｏｎｓｅｉｌ\u00ad fédé-\nral', 'Personenfreizü -\ngigkeit',
                     'ﬃ\nConsiglio\t federale', 'Égalité\r\n法律', 'a- \nB', 'a -\nb']:
            self.assertEqual(official_normalize(text), normalize(text))

    def test_single_flight_parses_once_and_invalidates_changed_pdf(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.pdf'
            pdf(path, ['Original document.'])
            engine = ClaimVerificationEngine(apertus_client=ApertusClient(mock=True))
            with patch.object(config, 'BOOKLET_CACHE_DIR', Path(directory) / 'cache'), patch.object(config, 'CACHE_SINGLE_FLIGHT', True), \
                 patch.object(engine.pdf_parser, 'extract_pages', wraps=engine.pdf_parser.extract_pages) as extract:
                with ThreadPoolExecutor(max_workers=8) as pool:
                    values = list(pool.map(engine._get_booklet_data, [path] * 16))
                self.assertEqual(extract.call_count, 1)
                self.assertTrue(all(value is values[0] for value in values))
                path.unlink()
                pdf(path, ['Changed source document with different content.'])
                changed = engine._get_booklet_data(path)
                self.assertEqual(extract.call_count, 2)
                self.assertIn('Changed source', changed['full_text'])

    def test_official_bare_flags_mixed_cases_jsonl_and_order_independence(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            pdf(directory / 'source.pdf', ['Le Conseil federal recommande le rejet de la proposition.'])
            cases = [
                {'id': '', 'booklet': {'path': 'source.pdf'}, 'vote': 'Proposition',
                 'claim': {'text': 'Le Conseil federal recommande le rejet.', 'language': 'fr'}},
                {'id': 'beginner', 'reference': {'text': 'Der Bundesrat empfiehlt Nein.'},
                 'claim': {'text': 'Der Bundesrat empfiehlt die Ablehnung.', 'language': 'de'}}]
            env = {**os.environ, 'EVIDENCE_POLICY': 'raw_pages_and_blocks', 'CACHE_SINGLE_FLIGHT': 'true',
                   'BOOKLET_CACHE_DIR': str(directory / 'cache'), 'PYTHONPATH': str(ROOT)}
            predictions = []
            for ordering in (cases, list(reversed(cases))):
                (directory / 'input.jsonl').write_text(''.join(json.dumps(c) + '\n' for c in ordering))
                process = subprocess.run([sys.executable, '-m', 'src', '--input', str(directory / 'input.jsonl'),
                                          '--output', str(directory / 'output.data'), '--mock'],
                                         cwd=directory, env=env, capture_output=True, text=True)
                self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
                output = [json.loads(line) for line in (directory / 'output.data').read_text().splitlines()]
                self.assertEqual([r['id'] for r in output], [c['id'] for c in ordering])
                predictions.append({r['id']: {k: r[k] for k in ('id', 'label', 'label_name', 'evidence')} for r in output})
            self.assertEqual(predictions[0], predictions[1])


if __name__ == '__main__':
    unittest.main()
