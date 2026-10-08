"""Remote transport and runtime configuration without a live endpoint."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch

from src import config
from src.apertus_client import ApertusClient
from src.measurement import RequestRecorder
from src.streaming import assemble_completion


class Stream:
    def __init__(self, chunks):
        self.chunks = chunks
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def __iter__(self):
        return iter(self.chunks)


class TestSubmissionTransport(unittest.TestCase):
    def test_json_repair_is_one_counted_attempt_with_unchanged_prompt_and_no_caps(self):
        calls = []
        def create(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise ConnectionError('transport unavailable')
            return N(choices=[N(message=N(content='{"label":0,"p_entail":1,"p_neutral":0,"p_contra":0,"evidence_ids":[1]}'),
                                finish_reason='stop')],
                     usage=N(prompt_tokens=100, completion_tokens=50, total_tokens=150))
        client = ApertusClient(mock=True)
        client.mock = False
        client.client = N(chat=N(completions=N(create=create)))
        with tempfile.TemporaryDirectory() as directory:
            recorder = RequestRecorder(Path(directory) / 'attempts.jsonl', 'repair-test')
            client._request_completion = recorder.wrap(client._request_completion)
            with patch.object(config, 'LLM_JSON_REPAIR', True), patch.object(config, 'LLM_MAX_RETRIES', 0), \
                 patch.object(config, 'LLM_STREAMING', False), recorder.case('repair-case'):
                result = client.infer(context='', claim='Source claim', passages=['Original passage'])
            self.assertEqual(result.label, 0)
            self.assertIsNone(result.error)
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0]['messages'], calls[1]['messages'])
            self.assertNotIn('response_format', calls[0])
            self.assertEqual(calls[1]['response_format'], {'type': 'json_object'})
            self.assertEqual([e['status'] for e in recorder.events], ['error', 'success'])
            self.assertFalse(recorder.events[0]['usage_known'])
            self.assertEqual(recorder.events[1]['output_tokens'], 50)
            for call in calls:
                self.assertNotIn('_json_output', call)
                self.assertNotIn('max_tokens', call)

    def test_plain_text_translation_is_never_forced_to_json(self):
        calls = []
        def unavailable(**kwargs):
            calls.append(kwargs)
            raise ConnectionError('transport unavailable')
        client = ApertusClient(mock=True)
        client.mock = False
        client.client = N(chat=N(completions=N(create=unavailable)))
        with patch.object(config, 'LLM_JSON_REPAIR', True), patch.object(config, 'LLM_MAX_RETRIES', 0):
            translation, *_ = client.translate('Original claim', 'fr')
        self.assertIsNone(translation)
        self.assertEqual(len(calls), 1)
        self.assertNotIn('response_format', calls[0])

    def test_all_token_caps_are_removed_and_complete_stream_usage_is_recorded(self):
        usage = N(prompt_tokens=80, completion_tokens=20, total_tokens=100)
        logprob = N(token='0', logprob=-0.1, top_logprobs=[])
        stream = Stream([
            N(choices=[N(index=0, delta=N(content='0|P1'), finish_reason='stop',
                         logprobs=N(content=[logprob]))], usage=None),
            N(choices=[], usage=usage)])
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            return stream

        client = ApertusClient(mock=True)
        client.mock = False
        client.client = N(chat=N(completions=N(create=create)))
        with tempfile.TemporaryDirectory() as directory:
            recorder = RequestRecorder(Path(directory) / 'attempts.jsonl', 'test')
            client._request_completion = recorder.wrap(client._request_completion)
            with patch.object(config, 'LLM_STREAMING', True), recorder.case('test-case'):
                response, error = client._create_with_retry(model='Apertus-v1.5', messages=[],
                                                            max_tokens=1, max_completion_tokens=1)
            self.assertIsNone(error)
            self.assertTrue(stream.closed)
            self.assertEqual(response.choices[0].message.content, '0|P1')
            self.assertEqual(response.choices[0].logprobs.content, [logprob])
            self.assertNotIn('max_tokens', calls[0])
            self.assertNotIn('max_completion_tokens', calls[0])
            self.assertEqual(calls[0]['stream_options'], {'include_usage': True})
            self.assertEqual(recorder.events[0]['input_tokens'], 80)
            self.assertEqual(recorder.events[0]['output_tokens'], 20)
            self.assertEqual(recorder.events[0]['case_id'], 'test-case')

    def test_incomplete_stream_is_a_failed_attempt_and_cannot_become_neutral_success(self):
        stream = Stream([N(choices=[N(index=0, delta=N(content='{"label":1'), finish_reason=None)], usage=None)])
        client = ApertusClient(mock=True)
        client.mock = False
        client.client = N(chat=N(completions=N(create=lambda **kwargs: stream)))
        with patch.object(config, 'LLM_STREAMING', True), patch.object(config, 'LLM_MAX_RETRIES', 0):
            response, error = client._create_with_retry(messages=[])
        self.assertIsNone(response)
        self.assertIsInstance(error, ValueError)
        self.assertTrue(stream.closed)

    def test_runtime_aliases_outrank_dotenv_values(self):
        source = Path(config.__file__).read_text()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / 'src').mkdir()
            (directory / 'src/config.py').write_text(source)
            (directory / '.env').write_text('BASE_URL=https://file.invalid/v1\nAPI_KEY=file-fixture\nLLM_BASE_URL=https://legacy-file.invalid/v1\nLLM_API_KEY=legacy-file-fixture\n')
            code = "import config, json; print(json.dumps([config.LLM_BASE_URL, config.LLM_API_KEY]))"
            for variables, expected in [
                ({'BASE_URL': 'https://runtime.invalid/v1', 'API_KEY': 'runtime-fixture'},
                 ['https://runtime.invalid/v1', 'runtime-fixture']),
                ({'LLM_BASE_URL': 'https://legacy-runtime.invalid/v1', 'LLM_API_KEY': 'legacy-runtime-fixture'},
                 ['https://legacy-runtime.invalid/v1', 'legacy-runtime-fixture']),
                ({'BASE_URL': '', 'API_KEY': ''}, ['', ''])]:
                env = {k: v for k, v in os.environ.items() if k not in ('BASE_URL', 'API_KEY', 'LLM_BASE_URL', 'LLM_API_KEY')}
                env.update(variables, PYTHONPATH=str(directory / 'src'))
                process = subprocess.run([sys.executable, '-c', code], cwd=directory,
                                         env=env, capture_output=True, text=True, check=True)
                self.assertEqual(json.loads(process.stdout), expected)


if __name__ == '__main__':
    unittest.main()
