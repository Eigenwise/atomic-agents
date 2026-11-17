import asyncio
import json
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock, Thread
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

from tool.wikipedia_search import WikipediaSearchTool, WikipediaSearchToolInputSchema


@pytest.fixture
def wikipedia_server(monkeypatch):
    calls = Counter()
    state = {"summary_status": 200, "extract_status": 200, "revision": 1}
    lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlsplit(self.path)
            params = parse_qs(parsed.query)
            if "srsearch" in params:
                key = ("search", params["srsearch"][0])
                data = {"query": {"search": [{"title": "Shared Page"}, {"title": "Second Page"}]}}
                status = 200
            elif "titles" in params:
                title = params["titles"][0]
                key = ("extract", title)
                data = {"query": {"pages": {"1": {"extract": f"Full {title} revision {state['revision']}"}}}}
                status = state["extract_status"]
            else:
                title = unquote(parsed.path.rsplit("/", 1)[-1])
                key = ("summary", title)
                data = {"title": title, "extract": f"Summary revision {state['revision']}"}
                status = state["summary_status"] if title == "Shared Page" else 200
            with lock:
                calls[key] += 1
            payload = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(
        WikipediaSearchTool, "_api_base", staticmethod(lambda _language: f"http://127.0.0.1:{server.server_port}")
    )
    try:
        yield calls, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def run_search(tool, mode, **kwargs):
    params = WikipediaSearchToolInputSchema(queries=["first topic", "second topic", "third topic"], **kwargs)
    return tool.run(params) if mode == "sync" else asyncio.run(tool.run_async(params))


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("full_text", [False, True])
def test_overlapping_queries_share_fetches_but_keep_query_results(wikipedia_server, mode, full_text):
    calls, _ = wikipedia_server
    result = run_search(WikipediaSearchTool(), mode, full_text=full_text)

    assert [(article.query, article.title) for article in result.results] == [
        (query, title) for query in ("first topic", "second topic", "third topic") for title in ("Shared Page", "Second Page")
    ]
    assert calls["summary", "Shared Page"] == 1
    assert calls["summary", "Second Page"] == 1
    assert calls["extract", "Shared Page"] == int(full_text)
    assert calls["extract", "Second Page"] == 0
    assert sum(calls.values()) == 5 + int(full_text)
    for article in result.results:
        assert article.summary == "Summary revision 1"
        assert article.full_text == ("Full Shared Page revision 1" if full_text and article.title == "Shared Page" else None)


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("status", [404, 503])
def test_shared_summary_failure_still_uses_first_available_article(wikipedia_server, mode, status):
    calls, state = wikipedia_server
    state["summary_status"] = status

    result = run_search(WikipediaSearchTool(), mode, full_text=True)

    assert len(result.results) == 3
    assert all(
        article.title == "Second Page" and article.full_text == "Full Second Page revision 1" for article in result.results
    )
    assert calls["summary", "Shared Page"] == 1
    assert calls["summary", "Second Page"] == 1
    assert calls["extract", "Second Page"] == 1


@pytest.mark.parametrize("mode", ["sync", "async"])
def test_separate_calls_fetch_fresh_data_and_retry_failed_extracts(wikipedia_server, mode):
    calls, state = wikipedia_server
    tool = WikipediaSearchTool()
    state["extract_status"] = 503

    first = run_search(tool, mode, full_text=True)
    state.update(extract_status=200, revision=2)
    second = run_search(tool, mode, full_text=True)

    assert all(article.full_text is None for article in first.results)
    assert all(article.summary == "Summary revision 2" for article in second.results)
    assert second.results[0].full_text == "Full Shared Page revision 2"
    assert calls["summary", "Shared Page"] == 2
    assert calls["summary", "Second Page"] == 2
    assert calls["extract", "Shared Page"] == 2
