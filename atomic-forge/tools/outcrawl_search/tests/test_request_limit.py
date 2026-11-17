import asyncio
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread

import pytest
from pydantic import ValidationError

from tool.outcrawl_search import OutcrawlSearchTool, OutcrawlSearchToolConfig, OutcrawlSearchToolInputSchema


@pytest.fixture
def outcrawl_server():
    state = {"active": 0, "peak": 0, "calls": [], "failures": False}
    lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
                state["calls"].append((self.path, payload))
            try:
                time.sleep(0.025)
                status = 200
                if self.path == "/search":
                    query = payload["query"]
                    data = {
                        "hits": [
                            {"title": f"{query}/{index}", "url": f"https://example.com/{query}/{index}"} for index in range(4)
                        ]
                    }
                    if state["failures"] and query == "q1":
                        status = 503
                else:
                    data = {"markdown": f"Content: {payload['url']}"}
                    if state["failures"] and payload["url"].endswith("q2/0"):
                        status = 503
                body = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            finally:
                with lock:
                    state["active"] -= 1

        def log_message(self, *_args):
            pass

    class Server(ThreadingHTTPServer):
        request_queue_size = 64

    server = Server(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("limit", [1, 3])
@pytest.mark.parametrize("scrape", [False, True])
def test_limit_is_shared_by_searches_and_page_fetches(outcrawl_server, mode, limit, scrape):
    base_url, state = outcrawl_server
    tool = OutcrawlSearchTool(OutcrawlSearchToolConfig(api_key="test", base_url=base_url, max_concurrent_requests=limit))
    params = OutcrawlSearchToolInputSchema(queries=[f"q{i}" for i in range(5)], scrape_content=scrape)

    result = tool.run(params) if mode == "sync" else asyncio.run(tool.run_async(params))

    assert 1 <= state["peak"] <= limit
    assert len(state["calls"]) == 5 + (20 if scrape else 0)
    assert [(item.query, item.title) for item in result.results] == [
        (f"q{q}", f"q{q}/{i}") for q in range(5) for i in range(4)
    ]
    assert all(item.content == (f"Content: {item.url}" if scrape else None) for item in result.results)


def test_failures_release_connections_and_tool_can_run_on_another_event_loop(outcrawl_server):
    base_url, state = outcrawl_server
    state["failures"] = True
    tool = OutcrawlSearchTool(OutcrawlSearchToolConfig(api_key="test", base_url=base_url, max_concurrent_requests=1))
    params = OutcrawlSearchToolInputSchema(queries=["q0", "q1", "q2"], scrape_content=True)

    first = asyncio.run(tool.run_async(params))
    assert [item.query for item in first.results] == ["q0"] * 4 + ["q2"] * 4
    assert first.results[4].content is None
    assert all(item.content for item in first.results[:4] + first.results[5:])
    state["failures"] = False
    second = tool.run(params)

    assert len(second.results) == 12
    assert all(item.content for item in second.results)
    assert state["peak"] == 1


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2", None])
def test_request_budget_requires_a_positive_integer(limit):
    with pytest.raises(ValidationError):
        OutcrawlSearchToolConfig(max_concurrent_requests=limit)


def test_default_budget_is_finite():
    assert OutcrawlSearchToolConfig().max_concurrent_requests == 10
