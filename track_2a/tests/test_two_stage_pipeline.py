"""Real PDF/retrieval and simulated endpoint; no network calls."""
import json
import re
import unittest
from types import SimpleNamespace
from src import config
from src.apertus_client import ApertusClient
from src.inference import ClaimVerificationEngine

class Endpoint:
    def __init__(self, label=0, bad=False, empty=False, bad_judgment=False):
        self.calls = []
        self.label, self.bad, self.empty, self.bad_judgment = label, bad, empty, bad_judgment
    def create(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            document = kwargs['messages'][1]['content'].split('DOCUMENT:\n')[1].split('\n\nCLAIM:')[0]
            passages = re.split(r'\[P\d+\] ', document)[1:]
            self.source_id = min(2, len(passages))
            quote = re.sub(r"^\([^)]*\)\s*", "", passages[self.source_id - 1].strip())[:100]
            data = {'statements': [] if self.empty else [{'text': 'invented quote' if self.bad else quote, 'passage_id': self.source_id}]}
        else:
            data = {'label': self.label, 'p_entail': int(self.label == 0), 'p_neutral': int(self.label == 1), 'p_contra': int(self.label == 2), 'evidence_ids': [] if self.label == 1 else [1]}
        content = 'broken response' if len(self.calls) == 2 and self.bad_judgment else json.dumps(data)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20, total_tokens=120))

class TestTwoStagePipeline(unittest.TestCase):
    def run_case(self, **kwargs):
        endpoint = Endpoint(**kwargs)
        client = ApertusClient(mock=True)
        client.mock = False
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=endpoint))
        engine = ClaimVerificationEngine(apertus_client=client, prompt_mode='two_stage')
        path = config.BOOKLETS_DIR / '2026-06-14_de.pdf'
        self.assertTrue(path.exists())
        result = engine.verify_claim('Le Conseil fédéral recommande de rejeter cette initiative.', path, claim_language='fr', strategy='hybrid', top_k=3, case_id='offline')
        return engine, endpoint, result
    def test_pdf_to_official_export(self):
        for label in (0, 1, 2):
            with self.subTest(label=label):
                engine, endpoint, result = self.run_case(label=label)
                self.assertEqual(len(endpoint.calls), 2)
                self.assertIsNone(result.error)
                self.assertEqual(result.label, label)
                self.assertEqual(result.tokens_total, 240)
                self.assertEqual(result.extracted_statements[0]['passage_id'], endpoint.source_id)
                official = result.to_official_dict()
                self.assertEqual(official['id'], 'offline')
                self.assertEqual(official['metrics']['input_tokens'], 200)
                if label == 1:
                    self.assertEqual(official['evidence'], [])
                else:
                    self.assertTrue(result.evidence_sources)
                    pages = {p['page_number']: p['text'] for p in engine._get_booklet_data(result.booklet_path)['pages']}
                    for source in result.evidence_sources:
                        self.assertIn(" ".join(source.quote.split()), " ".join(pages[source.page_number].split()))
                    self.assertEqual(official['evidence'][0]['page'], result.evidence_sources[0].page_number)
    def test_invalid_extraction_stops(self):
        _, endpoint, result = self.run_case(bad=True)
        self.assertEqual(len(endpoint.calls), 1)
        self.assertIn('extraction parse', result.error)
        self.assertEqual(result.evidence, [])
    def test_empty_extraction_stops(self):
        _, endpoint, result = self.run_case(empty=True)
        self.assertEqual(len(endpoint.calls), 1)
        self.assertEqual(result.label, 1)
        self.assertEqual(result.tokens_total, 120)
    def test_invalid_judgment(self):
        _, endpoint, result = self.run_case(bad_judgment=True)
        self.assertEqual(len(endpoint.calls), 2)
        self.assertIn('parse:', result.error)
        self.assertEqual(result.tokens_total, 240)
        self.assertEqual(result.evidence, [])
