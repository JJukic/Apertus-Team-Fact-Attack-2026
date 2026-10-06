import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from src.apertus_client import ApertusClient
from src.inference import ClaimVerificationEngine
from src.sentence_review import build_sentences, infer_sentence_review
from src import config

def response(data):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=data if isinstance(data, str) else json.dumps(data)))], usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))

class TestSentenceReview(unittest.TestCase):
    def setUp(self):
        self.passages = [{'text': 'Le comité recommande oui. Il demande une hausse.', 'page_number': 5, 'section': 'committee'}, {'text': 'Der Bundesrat empfiehlt Nein. Die Steuer bleibt gleich.', 'page_number': 9, 'section': 'Federal Council'}]
    def client(self, selector, judge):
        client = ApertusClient(mock=True)
        client.mock = False
        client._chat = Mock(side_effect=[(response(selector), None), (response(judge), None)])
        return client
    def test_full_context_and_evidence_outside_selector(self):
        client = self.client({'sentence_ids': [1]}, {'label': 2, 'p_contra': 1, 'sentence_ids': [3]})
        out = infer_sentence_review(client, self.passages, 'Le Conseil fédéral recommande oui.')
        self.assertIsNone(out.error)
        self.assertEqual(out.evidence, ['Der Bundesrat empfiehlt Nein.'])
        self.assertEqual(out.evidence_ids, [2])
        self.assertEqual(out.evidence_spans[0]['page'], 9)
        self.assertEqual(out.tokens_total, 240)
        context = client._chat.call_args_list[1].args[0][1]['content']
        for item in build_sentences(self.passages):
            self.assertIn(item['text'], context)
            self.assertIn(f"[S{item['sentence_id']};", context)
    def test_empty_or_invalid_selection_still_runs_judge(self):
        for selection in ({'sentence_ids': []}, {'sentence_ids': [99]}, 'bad json'):
            client = self.client(selection, {'label': 0, 'p_entail': 1, 'sentence_ids': [1]})
            out = infer_sentence_review(client, self.passages, 'claim')
            self.assertEqual(out.label, 0)
            self.assertEqual(client._chat.call_count, 2)
            self.assertIsNone(out.error)
            if selection != {'sentence_ids': []}: self.assertTrue(out.stage_warnings)
    def test_judge_errors_retain_usage(self):
        for judgment in ('bad json', {'label': 0, 'sentence_ids': [99]}, {'label': 0, 'sentence_ids': []}, {'label': True, 'sentence_ids': [1]}):
            out = infer_sentence_review(self.client({'sentence_ids': [1]}, judgment), self.passages, 'claim')
            self.assertIsNotNone(out.error)
            self.assertEqual(out.tokens_total, 240)
            self.assertEqual(out.evidence, [])
    def test_neutral_and_exact_spans(self):
        out = infer_sentence_review(self.client({'sentence_ids': [1]}, {'label': 1, 'p_neutral': 1, 'sentence_ids': []}), self.passages, 'claim')
        self.assertEqual(out.evidence_spans, [])
        for item in build_sentences(self.passages): self.assertIn(item['text'], self.passages[item['passage_id'] - 1]['text'])
        self.assertEqual([s['sentence_id'] for s in build_sentences(self.passages)], [1, 2, 3, 4])
    def test_real_pdf_pipeline_official_export(self):
        client = self.client({'sentence_ids': [1]}, {'label': 0, 'p_entail': 1, 'sentence_ids': [1]})
        engine = ClaimVerificationEngine(apertus_client=client, prompt_mode='sentence_review')
        result = engine.verify_claim('Le comité recommande oui.', config.BOOKLETS_DIR / '2026-06-14_de.pdf', top_k=3, strategy='hybrid')
        self.assertIsNone(result.error)
        src = result.evidence_sources[0]
        pages = engine._get_booklet_data(result.booklet_path)['pages']
        text = next(p['text'] for p in pages if p['page_number'] == src.page_number)
        self.assertIn(' '.join(src.quote.split()), ' '.join(text.split()))
        self.assertEqual(result.to_official_dict()['evidence'][0]['page'], src.page_number)
        self.assertEqual(result.tokens_total, 240)
