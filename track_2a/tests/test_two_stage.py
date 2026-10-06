import unittest
from unittest.mock import Mock
from types import SimpleNamespace
from src.apertus_client import ApertusClient, NLIOutput
from src.two_stage import infer_two_stage

class TestTwoStage(unittest.TestCase):
    def client(self, content):
        client = ApertusClient(mock=True)
        client.mock = False
        response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))], usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))
        client._chat = Mock(return_value=(response, None))
        client.infer = Mock(return_value=NLIOutput(label=0, reasoning='supports', evidence_ids=[1], evidence=['third'], tokens_prompt=50, tokens_completion=10, tokens_total=60))
        return client
    def test_original_ids_and_usage(self):
        client = self.client('{"statements": [{"text": "third", "passage_id": 3}]}')
        out = infer_two_stage(client, ['first', 'second', 'third original page'], 'claim')
        self.assertEqual(out.evidence_ids, [3])
        self.assertEqual(out.tokens_total, 180)
        self.assertEqual(client.infer.call_args.kwargs['passages'], ['third original page'])
        self.assertEqual(len(out.extracted_statements), 1)
    def test_bad_extraction_stops_judgment(self):
        for text in ('{"statements": [{"text": "invented", "passage_id": 1}]}', '{"statements": [{"text": "first", "passage_id": 9}]}', 'garbage'):
            client = self.client(text)
            self.assertIsNotNone(infer_two_stage(client, ['first'], 'claim').error)
            client.infer.assert_not_called()
    def test_empty_extraction_is_neutral(self):
        client = self.client('{"statements": []}')
        out = infer_two_stage(client, ['first'], 'claim')
        self.assertEqual(out.label, 1)
        self.assertIsNone(out.error)
        self.assertEqual(out.tokens_total, 120)
        client.infer.assert_not_called()
    def test_api_failure(self):
        client = self.client('')
        client._chat.return_value = (None, RuntimeError('unavailable'))
        self.assertIn('extraction api', infer_two_stage(client, ['first'], 'claim').error)
        client.infer.assert_not_called()

    def test_metadata_only_quote_rejected(self):
        client = self.client('{"statements": [{"text": "Arguments of committee", "passage_id": 1}]}')
        out = infer_two_stage(client, ['(Arguments of committee) source text'], 'claim', source_texts=['source text'])
        self.assertIsNotNone(out.error)
        client.infer.assert_not_called()

    def test_speaker_hint_uses_selected_ids(self):
        client = self.client('{"statements": [{"text": "third", "passage_id": 3}]}')
        infer_two_stage(client, ['first', 'second', 'third'], 'claim', sections=['other', 'other', 'committee'], attributed_side='committee')
        judge_claim = client.infer.call_args.args[1]
        self.assertIn('committee: P1', judge_claim)
        self.assertNotIn('P3', judge_claim)
