"""Authenticated Serply requests stay at the configured endpoint."""

import asyncio
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tool.serply_search import SerplySearchTool, SerplySearchToolConfig, SerplySearchToolInputSchema  # noqa: E402


@pytest.fixture
def serply_http():
    calls = []
    response = {"status": 200}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append((self.path, self.headers.get("X-Api-Key")))
            status = response["status"] if self.path.startswith("/search/") else 200
            payload = json.dumps({"results": [{"title": "A page", "link": "https://example.test/page"}]}).encode()
            self.send_response(status)
            if status != 200:
                self.send_header("Location", f"http://127.0.0.1:{destination.server_port}/result")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    destination = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    origin = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    workers = [threading.Thread(target=server.serve_forever, daemon=True) for server in (origin, destination)]
    for worker in workers:
        worker.start()
    try:
        yield f"http://127.0.0.1:{origin.server_port}", response, calls
    finally:
        for server in (origin, destination):
            server.shutdown()
            server.server_close()
        for worker in workers:
            worker.join()


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("mode", ["sync", "async"])
def test_search_does_not_follow_redirects(serply_http, status, mode, caplog):
    origin, response, calls = serply_http
    response["status"] = status
    tool = SerplySearchTool(SerplySearchToolConfig(api_key="test-only-key", base_url=origin))
    params = SerplySearchToolInputSchema(queries=["test"])

    result = tool.run(params) if mode == "sync" else asyncio.run(tool.run_async(params))

    assert len(calls) == 1
    assert calls[0][0].startswith("/search/")
    assert calls[0][1] == "test-only-key"
    assert result.results == []
    assert str(status) in caplog.text
    assert "test-only-key" not in caplog.text


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_direct_success_still_sends_the_key_and_returns_results(serply_http, mode):
    origin, response, calls = serply_http
    tool = SerplySearchTool(SerplySearchToolConfig(api_key="test-only-key", base_url=origin))
    params = SerplySearchToolInputSchema(queries=["test"])

    result = tool.run(params) if mode == "sync" else asyncio.run(tool.run_async(params))

    assert len(calls) == 1
    assert calls[0][1] == "test-only-key"
    assert [item.url for item in result.results] == ["https://example.test/page"]
