"""
Minimal OpenAI-compatible chat client on the standard library.

The official processing time counts everything outside LLM requests, start-up included. Importing the openai SDK
takes ~1.1 s in the container (thousands of pydantic types), while this client imports in ~0.1 s. It offers the one
call we use, `client.chat.completions.create(**kwargs)`, returns the response with attribute access like the SDK
(`choices[0].message.content`, `usage.prompt_tokens`, `logprobs.content[i].top_logprobs`) and raises errors that
carry `status_code`, so the retry logic in apertus_client works unchanged. Each thread keeps one open connection.
"""

import http.client
import json
import threading
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlsplit


class APIError(Exception):
    """A failed request; `status_code` is the HTTP status, or None for network errors and timeouts."""

    def __init__(self, message: str, status_code: Any = None):
        super().__init__(message)
        self.status_code = status_code


def _namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_namespace(v) for v in value]
    return value


class _Completions:
    def __init__(self, client: "HTTPChatClient"):
        self._client = client

    def create(self, timeout: float = 60.0, **body: Any) -> SimpleNamespace:
        return _namespace(self._client.post("/chat/completions", body, timeout))


class HTTPChatClient:
    def __init__(self, base_url: str, api_key: str):
        parts = urlsplit(base_url.rstrip("/"))
        self._https = parts.scheme != "http"
        self._host = parts.hostname or ""
        self._port = parts.port
        self._prefix = parts.path
        self._headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Accept": "application/json"}
        self._local = threading.local()
        self.chat = SimpleNamespace(completions=_Completions(self))

    def _connection(self, timeout: float) -> http.client.HTTPConnection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            cls = http.client.HTTPSConnection if self._https else http.client.HTTPConnection
            conn = self._local.conn = cls(self._host, self._port, timeout=timeout)
        conn.timeout = timeout
        if conn.sock is not None:
            conn.sock.settimeout(timeout)
        return conn

    def _drop(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
        self._local.conn = None

    def post(self, path: str, body: dict, timeout: float) -> Any:
        payload = json.dumps(body).encode("utf-8")
        for attempt in (1, 2):  # a kept-alive connection the server has closed fails once: reconnect and resend
            conn = self._connection(timeout)
            try:
                conn.request("POST", self._prefix + path, body=payload, headers=self._headers)
                resp = conn.getresponse()
                data = resp.read()
            except (http.client.RemoteDisconnected, ConnectionError) as e:  # reset, aborted, broken pipe
                self._drop()
                if attempt == 1:
                    continue
                raise APIError(f"connection error: {e}") from e
            except (OSError, http.client.HTTPException) as e:  # timeouts, DNS, TLS
                self._drop()
                raise APIError(f"{type(e).__name__}: {e}") from e
            if resp.getheader("Connection", "").lower() == "close":
                self._drop()
            if resp.status >= 400:
                text = data.decode("utf-8", "replace").strip()
                try:
                    detail = json.loads(text)
                    err = detail.get("error")
                    text = (err.get("message") if isinstance(err, dict) else err) or detail.get("message") or text
                except (ValueError, AttributeError):
                    pass
                raise APIError(text[:500] or f"HTTP {resp.status}", status_code=resp.status)
            try:
                return json.loads(data)
            except ValueError as e:
                raise APIError(f"invalid JSON response: {data[:200]!r}", status_code=resp.status) from e
        raise APIError("unreachable")
