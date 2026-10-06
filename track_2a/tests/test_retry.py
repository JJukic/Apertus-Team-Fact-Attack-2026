"""
Retry behaviour of ApertusClient on API errors, and the CLI refusing to run without an API key.
No network access: chat.completions.create is replaced by a stub and time.sleep by a no-op.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from src import config
from src.apertus_client import ApertusClient


class _ApiError(Exception):
    def __init__(self, status_code=None, message="error"):
        super().__init__(message)
        self.status_code = status_code


class _Completions:
    """Raises the queued errors, then returns a valid answer."""

    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        message = SimpleNamespace(content='{"label": 0, "p_entail": 1.0, "p_neutral": 0.0, "p_contra": 0.0, "evidence_ids": [1]}')
        return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                               usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15))


def _client(errors):
    client = ApertusClient(api_key="test", mock=True)
    client.mock = False
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions(errors)))
    return client


@mock.patch("src.apertus_client.time.sleep", lambda s: None)
class TestRetry(unittest.TestCase):
    def test_transient_errors_are_retried_until_success(self):
        client = _client([_ApiError(504), _ApiError(None, "timeout"), _ApiError(429)])
        out = client.infer(context="", claim="c", passages=["p"])
        self.assertEqual(out.label, 0)
        self.assertIsNone(out.error)
        self.assertEqual(client.client.chat.completions.calls, 4)

    def test_bad_request_is_not_retried(self):
        client = _client([_ApiError(400)])
        out = client.infer(context="", claim="c", passages=["p"])
        self.assertTrue(out.error.startswith("api:"))
        self.assertEqual(client.client.chat.completions.calls, 1)

    def test_invalid_key_gets_two_quick_retries_only(self):
        client = _client([_ApiError(401, "invalid API key")] * 5)
        out = client.infer(context="", claim="c", passages=["p"])
        self.assertIsNotNone(out.error)
        self.assertEqual(client.client.chat.completions.calls, 3)

    def test_retries_stop_at_the_limit(self):
        client = _client([_ApiError(503)] * (config.LLM_MAX_RETRIES + 5))
        out = client.infer(context="", claim="c", passages=["p"])
        self.assertIsNotNone(out.error)
        self.assertEqual(client.client.chat.completions.calls, config.LLM_MAX_RETRIES + 1)


class _Sequence:
    """Returns the queued answer texts in order and records the messages it was called with."""

    def __init__(self, contents):
        self.contents, self.calls = list(contents), []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.contents.pop(0))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150))


class TestThinkingBudget(unittest.TestCase):
    def test_exhausted_budget_forces_an_answer(self):
        client = ApertusClient(api_key="test", mock=True)
        client.mock = False
        seq = _Sequence(["The committee says that rents", '{"label": 0, "p_entail": 1.0, "p_neutral": 0.0, "p_contra": 0.0, "evidence_ids": [1]}'])
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=seq))
        with mock.patch.object(config, "THINKING", True), mock.patch.object(config, "THINKING_BUDGET", 200):
            out = client.infer(context="", claim="c", passages=["p"])
        self.assertEqual(out.label, 0)
        self.assertEqual(len(seq.calls), 2)
        self.assertEqual(seq.calls[0]["max_tokens"], 200)
        self.assertTrue(seq.calls[1]["messages"][-1]["content"].endswith("<|inner_suffix|>"))
        self.assertEqual((out.tokens_prompt, out.tokens_completion), (200, 100))

    def test_finished_reasoning_needs_no_second_call(self):
        client = ApertusClient(api_key="test", mock=True)
        client.mock = False
        seq = _Sequence(['Short.<|inner_suffix|>{"label": 1, "p_neutral": 1.0, "evidence_ids": []}'])
        client.client = SimpleNamespace(chat=SimpleNamespace(completions=seq))
        with mock.patch.object(config, "THINKING", True), mock.patch.object(config, "THINKING_BUDGET", 200):
            out = client.infer(context="", claim="c", passages=["p"])
        self.assertEqual(out.label, 1)
        self.assertEqual(len(seq.calls), 1)


class TestCliRequiresKey(unittest.TestCase):
    def test_benchmark_without_key_exits_instead_of_mocking(self):
        from typer.testing import CliRunner
        from src.cli import app
        with mock.patch.object(config, "LLM_API_KEY", ""), mock.patch.object(config, "MOCK_APERTUS", False):
            result = CliRunner().invoke(app, ["benchmark", "-n", "3"])
        self.assertEqual(result.exit_code, 2)


if __name__ == "__main__":
    unittest.main()
