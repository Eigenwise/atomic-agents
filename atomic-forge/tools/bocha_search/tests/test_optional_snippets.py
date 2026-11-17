"""Exercise optional BoCha snippets through the public API and real HTTP."""

import asyncio
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import aiohttp
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tool.bocha_search import BoChaSearchTool, BoChaSearchToolConfig, BoChaSearchToolInputSchema  # noqa: E402


@pytest.fixture
def bocha_http(monkeypatch):
    payload = {"data": {"webPages": {"value": []}}}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            encoded = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    original_post = aiohttp.ClientSession.post

    def local_post(self, url, **kwargs):
        assert url == "https://api.bochaai.com/v1/web-search"
        return original_post(self, f"http://127.0.0.1:{server.server_port}/search", **kwargs)

    monkeypatch.setattr(aiohttp.ClientSession, "post", local_post)
    try:
        yield payload["data"]["webPages"]["value"]
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("snippet", ["omitted", None, "", "A useful excerpt"])
def test_public_search_retains_hits_without_optional_snippets(bocha_http, mode, snippet):
    hit = {"name": "A useful page", "url": "https://example.test/page"}
    if snippet != "omitted":
        hit["snippet"] = snippet
    bocha_http.append(hit)
    tool = BoChaSearchTool(BoChaSearchToolConfig(api_key="test"))
    params = BoChaSearchToolInputSchema(queries=["page"])

    output = tool.run(params) if mode == "sync" else asyncio.run(tool.run_async(params))

    assert len(output.results) == 1
    assert output.results[0].name == "A useful page"
    assert output.results[0].url == "https://example.test/page"
    assert output.results[0].query == "page"
    assert output.results[0].snippet == ("" if snippet == "omitted" else snippet)


def test_optional_snippet_hit_keeps_its_rank_in_a_limited_batch(bocha_http):
    bocha_http.extend(
        [
            {"name": "First", "url": "https://example.test/first"},
            {"name": "No URL", "snippet": "incomplete"},
            {"url": "https://example.test/no-name", "snippet": "incomplete"},
            {"name": "Second", "url": "https://example.test/second", "snippet": "second"},
        ]
    )
    tool = BoChaSearchTool(BoChaSearchToolConfig(api_key="test", count=1))

    output = tool.run(BoChaSearchToolInputSchema(queries=["page"]))

    assert [item.url for item in output.results] == ["https://example.test/first"]
