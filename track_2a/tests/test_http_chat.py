"""
Standard-library chat client (replaces the openai SDK): request format, SDK-like response access, errors with
status_code, and reconnecting when the server closes a kept-alive connection. Runs against a local test server.
"""

import json
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_pkg_root = Path(__file__).resolve().parent.parent
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

from src.apertus_client import ApertusClient
from src.http_chat import APIError, HTTPChatClient

ANSWER = {
    "choices": [{
        "message": {"role": "assistant", "content": "2 | 3, 1"},
        "logprobs": {"content": [{"token": "2", "logprob": -0.1, "top_logprobs": [
            {"token": "2", "logprob": -0.1}, {"token": "0", "logprob": -2.5}, {"token": "1", "logprob": -4.0}]}]},
    }],
    "usage": {"prompt_tokens": 1200, "completion_tokens": 7, "total_tokens": 1207},
}


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive, like the CSCS endpoint
    requests = []
    mode = "ok"

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.requests.append((self.path, self.headers.get("Authorization"), body))
        if _Handler.mode == "forbidden":
            self._send(403, b"invalid API key", "text/plain")
        elif _Handler.mode == "json_error":
            self._send(400, json.dumps({"error": {"message": "unknown model"}}).encode(), "application/json")
        else:
            self._send(200, json.dumps(ANSWER).encode(), "application/json")
        if _Handler.mode == "close_after":
            self.close_connection = True

    def _send(self, status, data, ctype):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


class TestHTTPChatClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_address[1]}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        _Handler.requests = []
        _Handler.mode = "ok"
        self.client = HTTPChatClient(self.base_url, "team-key")

    def test_request_and_sdk_like_response(self):
        resp = self.client.chat.completions.create(model="m", messages=[{"role": "user", "content": "x"}],
                                                   temperature=0.0, max_tokens=24, timeout=5.0)
        path, auth, body = _Handler.requests[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(auth, "Bearer team-key")
        self.assertEqual(body["max_tokens"], 24)
        self.assertNotIn("timeout", body)
        self.assertEqual(resp.choices[0].message.content, "2 | 3, 1")
        self.assertEqual(resp.usage.prompt_tokens, 1200)
        self.assertEqual(resp.choices[0].logprobs.content[0].top_logprobs[1].token, "0")
        resp.choices[0].message.content = "edited"  # the thinking path rewrites the content
        self.assertEqual(resp.choices[0].message.content, "edited")

    def test_errors_carry_status_code(self):
        _Handler.mode = "forbidden"
        with self.assertRaises(APIError) as ctx:
            self.client.chat.completions.create(model="m", messages=[], timeout=5.0)
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("invalid API key", str(ctx.exception))
        _Handler.mode = "json_error"
        with self.assertRaises(APIError) as ctx:
            self.client.chat.completions.create(model="m", messages=[], timeout=5.0)
        self.assertEqual((ctx.exception.status_code, str(ctx.exception)), (400, "unknown model"))

    def test_network_error_has_no_status(self):
        client = HTTPChatClient("http://127.0.0.1:9/v1", "k")  # nothing listens on the discard port
        with self.assertRaises(APIError) as ctx:
            client.chat.completions.create(model="m", messages=[], timeout=2.0)
        self.assertIsNone(ctx.exception.status_code)  # retried as transient

    def test_reconnects_after_server_closed_connection(self):
        _Handler.mode = "close_after"
        for _ in range(3):
            resp = self.client.chat.completions.create(model="m", messages=[], timeout=5.0)
            self.assertEqual(resp.usage.completion_tokens, 7)
        self.assertEqual(len(_Handler.requests), 3)

    def test_apertus_client_end_to_end(self):
        client = ApertusClient(base_url=self.base_url, api_key="team-key", mock=False)
        out = client.infer_compact(["Erste Passage.", "Zweite Passage.", "Dritte Passage."], "Behauptung", "de")
        self.assertEqual(out.label, 2)
        self.assertEqual(out.evidence_ids, [3, 1])
        self.assertEqual((out.tokens_prompt, out.tokens_completion), (1200, 7))
        self.assertGreater(out.p_contra, 0.8)


if __name__ == "__main__":
    unittest.main()
