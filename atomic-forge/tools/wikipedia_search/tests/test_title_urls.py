"""Returned fallback links must preserve article titles when followed over HTTP."""

import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.parse import parse_qs, unquote, urlsplit
from urllib.request import urlopen

import pytest

from tool.wikipedia_search import WikipediaSearchTool, WikipediaSearchToolInputSchema


@pytest.fixture
def wiki_server(monkeypatch):
    state = {"title": "Article", "summary_extra": {}, "requests": []}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlsplit(self.path)
            state["requests"].append(parsed)
            if parsed.path == "/w/api.php":
                params = parse_qs(parsed.query)
                if params.get("list") == ["search"]:
                    body = {"query": {"search": [{"title": state["title"]}]}}
                else:
                    body = {"query": {"pages": {"1": {"extract": "Full article"}}}}
            elif parsed.path.startswith("/api/rest_v1/page/summary/"):
                body = {"title": state["title"], "extract": "Summary", **state["summary_extra"]}
            else:
                body = {"path": unquote(parsed.path), "query": parsed.query}
            encoded = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setattr(WikipediaSearchTool, "_api_base", staticmethod(lambda language: origin))
    try:
        yield state, origin
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "title,encoded_title",
    [
        ("Who Are You?", "Who_Are_You%3F"),
        ("What Is Love? (song)", "What_Is_Love%3F_%28song%29"),
        ("100% Love (2011 film)", "100%25_Love_%282011_film%29"),
        ("C++", "C%2B%2B"),
        ("Björk", "Bj%C3%B6rk"),
        ("AC/DC", "AC/DC"),
        ("Ordinary article", "Ordinary_article"),
    ],
)
def test_fallback_title_link_survives_http_navigation(wiki_server, title, encoded_title, asynchronous):
    state, origin = wiki_server
    state["title"] = title
    tool = WikipediaSearchTool()
    params = WikipediaSearchToolInputSchema(queries=["topic"], full_text=True)
    result = asyncio.run(tool.run_async(params)) if asynchronous else tool.run(params)
    assert len(result.results) == 1
    article = result.results[0]
    assert article.title == title
    assert article.summary == "Summary"
    assert article.full_text == "Full article"
    with urlopen(article.page_url) as response:
        destination = json.load(response)
    assert destination == {"path": "/wiki/" + title.replace(" ", "_"), "query": ""}
    assert article.page_url == f"{origin}/wiki/{encoded_title}"
    summary = next(request for request in state["requests"] if request.path.startswith("/api/rest_v1/"))
    assert unquote(summary.path) == "/api/rest_v1/page/summary/" + title


@pytest.mark.parametrize("content_urls", [{}, {"desktop": {}}, {"desktop": {"page": ""}}])
def test_partial_link_metadata_uses_encoded_fallback(wiki_server, content_urls):
    state, origin = wiki_server
    state.update(title="Who Are You?", summary_extra={"content_urls": content_urls})
    article = WikipediaSearchTool().run(WikipediaSearchToolInputSchema(queries=["topic"])).results[0]
    assert article.page_url == f"{origin}/wiki/Who_Are_You%3F"


def test_explicit_canonical_url_remains_unchanged(wiki_server):
    state, _ = wiki_server
    canonical = "https://fr.wikipedia.org/wiki/Qui_%C3%AAtes-vous%3F?oldid=123#Histoire"
    state.update(title="Who Are You?", summary_extra={"content_urls": {"desktop": {"page": canonical}}})
    article = WikipediaSearchTool().run(WikipediaSearchToolInputSchema(queries=["topic"], language="fr")).results[0]
    assert article.page_url == canonical
