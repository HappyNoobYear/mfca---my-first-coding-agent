"""
Ollama token- and tool-call-counting proxy.

Sits between benchmark agents and the real Ollama server. Every request is
forwarded verbatim; every response is inspected for prompt_eval_count,
eval_count, and any message.tool_calls. The accumulated totals can be read
and reset between agent runs, giving a single authoritative token- and
tool-call-count source regardless of what each agent framework chooses to
self-report (which, per ohpi and smolagents, cannot be trusted on its own).

Usage (called automatically by BenchmarkRunner):
    proxy = OllamaProxy(real_host="http://localhost:11434", proxy_port=11435)
    proxy.start()
    ...
    proxy.reset()
    # run agent
    stats = proxy.get_stats()
    proxy.stop()
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.request import urlopen, Request as URLRequest
from urllib.error import URLError


class OllamaProxy:
    """Token-counting HTTP proxy for Ollama."""

    def __init__(self, real_host: str = "http://localhost:11434", proxy_port: int = 11435):
        self.real_host = real_host.rstrip("/")
        self.proxy_port = proxy_port
        self._lock = threading.Lock()
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._request_count = 0
        self._tool_calls = 0
        self._server: HTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ #
    # Public API used by the test runner                                   #
    # ------------------------------------------------------------------ #

    def start(self):
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # silence per-request logs

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)

                # Forward to real Ollama
                url = f"{proxy.real_host}{self.path}"
                req = URLRequest(url, data=body, headers={"Content-Type": "application/json"})
                try:
                    with urlopen(req) as resp:
                        raw = resp.read()
                        status = resp.status
                        content_type = resp.headers.get("Content-Type", "application/json")
                except URLError as e:
                    self.send_response(502)
                    self.end_headers()
                    self.wfile.write(str(e).encode())
                    return

                # Parse and accumulate token counts from non-streaming responses.
                # Streaming responses (newline-delimited JSON) carry counts in the
                # final chunk that has "done": true.
                proxy._accumulate(raw)

                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                url = f"{proxy.real_host}{self.path}"
                req = URLRequest(url)
                try:
                    with urlopen(req) as resp:
                        raw = resp.read()
                        status = resp.status
                except URLError as e:
                    self.send_response(502)
                    self.end_headers()
                    self.wfile.write(str(e).encode())
                    return
                self.send_response(status)
                self.end_headers()
                self.wfile.write(raw)

        self._server = HTTPServer(("localhost", self.proxy_port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self):
        if self._server:
            self._server.shutdown()
            self._server = None

    def reset(self):
        """Reset counters before each agent's test run."""
        with self._lock:
            self._prompt_tokens = 0
            self._completion_tokens = 0
            self._request_count = 0
            self._tool_calls = 0

    def get_stats(self) -> dict:
        """Return accumulated token and tool-call counts since last reset()."""
        with self._lock:
            return {
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "total_tokens": self._prompt_tokens + self._completion_tokens,
                "api_calls": self._request_count,
                "tool_calls": self._tool_calls,
            }

    @property
    def url(self) -> str:
        return f"http://localhost:{self.proxy_port}"

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _accumulate(self, raw: bytes):
        """Extract token and tool-call counts from one or more Ollama JSON chunks."""
        # Ollama returns either a single JSON object (non-streaming) or
        # newline-delimited JSON objects (streaming). We scan all lines.
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            prompt = obj.get("prompt_eval_count", 0) or 0
            completion = obj.get("eval_count", 0) or 0
            # Ollama's /api/chat returns tool calls the model requested as
            # message.tool_calls (OpenAI-style function-calling format).
            # Counting them here, independent of what the agent framework
            # wrapping this call chooses to self-report, is what would have
            # caught ohpi's tool-call counter never incrementing.
            tool_calls = len(obj.get("message", {}).get("tool_calls") or [])

            if prompt or completion or tool_calls:
                with self._lock:
                    self._prompt_tokens += prompt
                    self._completion_tokens += completion
                    self._tool_calls += tool_calls
                    self._request_count += 1
