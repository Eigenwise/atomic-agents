"""Verify per-call result limits through real HTTP requests."""

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
def request_counts(monkeypatch):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append((request["query"], request["count"]))
            hits = [
                {"name": str(index), "url": f"https://example.test/{index}", "snippet": "excerpt"}
                for index in range(request["count"])
            ]
            encoded = json.dumps({"data": {"webPages": {"value": hits}}}).encode()
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
        yield requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("count", [None, 1, 5])
def test_result_limit_is_sent_to_the_provider_for_each_query(request_counts, mode, count):
    tool = BoChaSearchTool(BoChaSearchToolConfig(api_key="test", count=3))
    params = BoChaSearchToolInputSchema(queries=["first", "second"])

    result = tool.run(params, count=count) if mode == "sync" else asyncio.run(tool.run_async(params, count=count))

    expected = 3 if count is None else count
    assert sorted(request_counts) == [("first", expected), ("second", expected)]
    assert [item.query for item in result.results] == ["first"] * expected + ["second"] * expected
    assert tool.count == 3


def test_concurrent_overrides_do_not_change_the_default_or_each_other(request_counts):
    tool = BoChaSearchTool(BoChaSearchToolConfig(api_key="test", count=3))

    async def search():
        return await asyncio.gather(
            tool.run_async(BoChaSearchToolInputSchema(queries=["small"]), count=1),
            tool.run_async(BoChaSearchToolInputSchema(queries=["large"]), count=5),
        )

    small, large = asyncio.run(search())
    default = tool.run(BoChaSearchToolInputSchema(queries=["default"]))

    assert sorted(request_counts) == [("default", 3), ("large", 5), ("small", 1)]
    assert [len(result.results) for result in (small, large, default)] == [1, 5, 3]
