import logging
import os
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tool.firecrawl_search import (  # noqa: E402
    FirecrawlSearchResultItem,
    FirecrawlSearchTool,
    FirecrawlSearchToolConfig,
    FirecrawlSearchToolInputSchema,
    FirecrawlSearchToolOutputSchema,
)


WEB_HIT = {
    "url": "https://www.python.org/",
    "title": "Welcome to Python.org",
    "description": "Python is a programming language that lets you work quickly.",
    "position": 1,
}

NEWS_HIT = {
    "url": "https://example.com/python-3-14",
    "title": "Python 3.14 released",
    "snippet": "The Python core team shipped 3.14 today.",
    "date": "2 days ago",
    "imageUrl": "https://example.com/thumb.png",
    "position": 2,
}

CONTENT_HIT = {
    "url": "https://example.com/guide",
    "description": "A guide to agents.",
    "position": 3,
    "markdown": "# Agent guide\n\nBuild agents step by step.",
    "metadata": {"title": "Agent guide | Example", "description": "Metadata description", "statusCode": 200},
}


DUMMY_KEY = "fc-dummy-secret-0123456789"


@asynccontextmanager
async def fake_firecrawl(respond):
    """Serve POST /v2/search on a local aiohttp server; `respond(body)` builds each response."""
    seen = []

    async def handler(request):
        body = await request.json()
        seen.append({"headers": dict(request.headers), "body": body})
        return respond(body)

    app = web.Application()
    app.router.add_post("/v2/search", handler)
    server = TestServer(app)
    await server.start_server()
    try:
        yield str(server.make_url("/v2")), seen
    finally:
        await server.close()


def _http_tool(base_url: str, **config) -> FirecrawlSearchTool:
    return FirecrawlSearchTool(config=FirecrawlSearchToolConfig(api_key=DUMMY_KEY, base_url=base_url, **config))


def _mock_session(status: int, payload: dict, reason: str = "OK") -> MagicMock:
    session = MagicMock()
    response = SimpleNamespace(status=status, reason=reason, json=AsyncMock(return_value=payload))
    session.post.return_value.__aenter__.return_value = response
    return session


@pytest.fixture
def tool():
    return FirecrawlSearchTool(config=FirecrawlSearchToolConfig(api_key="test-key"))


def test_to_item_web():
    item = FirecrawlSearchTool._to_item(WEB_HIT, "python")
    assert isinstance(item, FirecrawlSearchResultItem)
    assert item.query == "python"
    assert item.title == "Welcome to Python.org"
    assert item.url == "https://www.python.org/"
    assert item.description.startswith("Python is a programming language")
    assert item.position == 1
    assert item.published is None
    assert item.content is None


def test_to_item_news():
    """News results carry a snippet and a relative date instead of a description."""
    item = FirecrawlSearchTool._to_item(NEWS_HIT, "python")
    assert item.description == "The Python core team shipped 3.14 today."
    assert item.published == "2 days ago"
    assert item.position == 2


def test_to_item_with_content():
    """With scrapeOptions each result carries markdown and page metadata."""
    item = FirecrawlSearchTool._to_item(CONTENT_HIT, "agents")
    assert item.title == "Agent guide | Example"
    assert item.description == "A guide to agents."
    assert item.content.startswith("# Agent guide")


def test_to_item_falls_back_to_url_for_title():
    item = FirecrawlSearchTool._to_item({"url": "https://example.com/untitled"}, "q")
    assert item.title == "https://example.com/untitled"
    assert item.description is None


def test_build_body_defaults(tool):
    body = tool._build_body("python", FirecrawlSearchToolInputSchema(queries=["python"]))
    assert body == {"query": "python", "limit": 5, "sources": ["web"]}


def test_build_body_with_options():
    tool = FirecrawlSearchTool(config=FirecrawlSearchToolConfig(api_key="k", location="Germany", time_range="week"))
    params = FirecrawlSearchToolInputSchema(queries=["python"], max_results_per_query=3, include_content=True)

    body = tool._build_body("python", params)

    assert body["sources"] == ["web"]
    assert body["limit"] == 3
    assert body["location"] == "Germany"
    assert body["tbs"] == "qdr:w"
    assert body["scrapeOptions"] == {"formats": ["markdown"], "onlyMainContent": True}


def test_build_body_skips_time_range_for_news():
    """The API's tbs filter only applies to web results."""
    tool = FirecrawlSearchTool(config=FirecrawlSearchToolConfig(api_key="k", time_range="day"))

    body = tool._build_body("python", FirecrawlSearchToolInputSchema(queries=["python"], search_type="news"))

    assert body["sources"] == ["news"]
    assert "tbs" not in body


def test_to_item_truncates_content_and_ignores_bad_metadata():
    hit = {**CONTENT_HIT, "metadata": ["not", "a", "dict"]}

    item = FirecrawlSearchTool._to_item(hit, "agents", max_content_chars=7)

    assert item.content == "# Agent"
    assert item.title == "https://example.com/guide"


@pytest.mark.asyncio
async def test_fetch_builds_request(tool):
    session = _mock_session(200, {"success": True, "data": {"web": [WEB_HIT]}})
    params = FirecrawlSearchToolInputSchema(queries=["python"])

    items = await tool._fetch(session, "python", params)

    assert len(items) == 1
    assert items[0].url == "https://www.python.org/"
    call_args = session.post.call_args
    assert call_args[0][0] == "https://api.firecrawl.dev/v2/search"
    assert call_args[1]["json"]["query"] == "python"


@pytest.mark.asyncio
async def test_fetch_reads_news_results(tool):
    session = _mock_session(200, {"success": True, "data": {"news": [NEWS_HIT] * 4}})
    params = FirecrawlSearchToolInputSchema(queries=["python"], search_type="news", max_results_per_query=2)

    items = await tool._fetch(session, "python", params)

    assert len(items) == 2
    assert items[0].published == "2 days ago"


@pytest.mark.asyncio
async def test_fetch_handles_missing_source_key(tool):
    """The API omits the source key entirely when there are no results."""
    session = _mock_session(200, {"success": True, "data": {}})

    items = await tool._fetch(session, "nothing", FirecrawlSearchToolInputSchema(queries=["nothing"]))

    assert items == []


@pytest.mark.asyncio
async def test_fetch_skips_hits_without_url(tool):
    session = _mock_session(200, {"success": True, "data": {"web": [WEB_HIT, {"title": "no url"}]}})

    items = await tool._fetch(session, "python", FirecrawlSearchToolInputSchema(queries=["python"]))

    assert [item.url for item in items] == ["https://www.python.org/"]


@pytest.mark.asyncio
async def test_fetch_raises_on_http_error(tool):
    session = _mock_session(401, {"success": False, "error": "Unauthorized: Invalid token"}, reason="Unauthorized")

    with pytest.raises(Exception, match="Firecrawl search failed for 'python': 401 Unauthorized: Invalid token"):
        await tool._fetch(session, "python", FirecrawlSearchToolInputSchema(queries=["python"]))


@pytest.mark.asyncio
async def test_fetch_raises_when_success_is_false(tool):
    session = _mock_session(200, {"success": False})

    with pytest.raises(Exception, match="Firecrawl search failed for 'python': 200 OK"):
        await tool._fetch(session, "python", FirecrawlSearchToolInputSchema(queries=["python"]))


@pytest.mark.asyncio
async def test_fetch_reports_status_for_non_json_body(tool):
    session = MagicMock()
    response = SimpleNamespace(status=502, reason="Bad Gateway", json=AsyncMock(side_effect=ValueError("not json")))
    session.post.return_value.__aenter__.return_value = response

    with pytest.raises(Exception, match="Firecrawl search failed for 'python': 502 Bad Gateway"):
        await tool._fetch(session, "python", FirecrawlSearchToolInputSchema(queries=["python"]))


@pytest.mark.asyncio
async def test_fetch_applies_max_content_chars():
    tool = FirecrawlSearchTool(config=FirecrawlSearchToolConfig(api_key="k", max_content_chars=5))
    session = _mock_session(200, {"success": True, "data": {"web": [CONTENT_HIT, "not a dict"]}})

    items = await tool._fetch(session, "agents", FirecrawlSearchToolInputSchema(queries=["agents"], include_content=True))

    assert [item.content for item in items] == ["# Age"]


@pytest.mark.asyncio
async def test_run_async_aggregates_results(tool):
    async def fake_fetch(self, session, query, params):
        return [FirecrawlSearchTool._to_item(WEB_HIT, query), FirecrawlSearchTool._to_item(NEWS_HIT, query)]

    with patch.object(FirecrawlSearchTool, "_fetch", fake_fetch):
        out = await tool.run_async(FirecrawlSearchToolInputSchema(queries=["q1", "q2"]))

    assert isinstance(out, FirecrawlSearchToolOutputSchema)
    assert len(out.results) == 4
    assert {r.query for r in out.results} == {"q1", "q2"}


@pytest.mark.asyncio
async def test_run_async_skips_failing_query(tool):
    async def fake_fetch(self, session, query, params):
        if query == "bad":
            raise Exception("rate limited")
        return [FirecrawlSearchTool._to_item(WEB_HIT, query)]

    with patch.object(FirecrawlSearchTool, "_fetch", fake_fetch):
        out = await tool.run_async(FirecrawlSearchToolInputSchema(queries=["bad", "good"]))

    assert len(out.results) == 1
    assert out.results[0].query == "good"


@pytest.mark.asyncio
async def test_run_async_raises_when_every_query_fails(tool):
    """A bad key or exhausted credits should surface as an error, not as an empty result."""

    async def fake_fetch(self, session, query, params):
        raise Exception("Firecrawl search failed for 'q': 401 Unauthorized: Invalid token")

    with patch.object(FirecrawlSearchTool, "_fetch", fake_fetch):
        with pytest.raises(Exception, match="401"):
            await tool.run_async(FirecrawlSearchToolInputSchema(queries=["q1", "q2"]))


@pytest.mark.asyncio
async def test_http_round_trip():
    def respond(body):
        return web.json_response(
            {"success": True, "data": {"web": [{**CONTENT_HIT, "url": f"https://example.com/{body['query']}"}]}}
        )

    async with fake_firecrawl(respond) as (base_url, seen):
        out = await _http_tool(base_url).run_async(FirecrawlSearchToolInputSchema(queries=["agents"], include_content=True))

    assert [r.url for r in out.results] == ["https://example.com/agents"]
    assert out.results[0].content.startswith("# Agent guide")
    assert seen[0]["headers"]["Authorization"] == f"Bearer {DUMMY_KEY}"
    assert seen[0]["body"]["scrapeOptions"] == {"formats": ["markdown"], "onlyMainContent": True}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"success": True, "data": None},
        {"success": True},
        {"success": True, "data": {"web": {"url": "https://example.com/not-a-list"}}},
        {"success": True, "data": {"web": None}},
        {"success": True, "data": [WEB_HIT]},
    ],
)
async def test_http_malformed_success_response_raises(payload):
    """A 200 with success=true but no usable result list is an error, not an empty result."""
    async with fake_firecrawl(lambda body: web.json_response(payload)) as (base_url, _):
        with pytest.raises(Exception, match="unexpected response shape for 'web' results"):
            await _http_tool(base_url).run_async(FirecrawlSearchToolInputSchema(queries=["q"]))


@pytest.mark.asyncio
async def test_http_non_json_error_page_reports_status():
    def respond(body):
        return web.Response(status=502, text="<html>Bad Gateway</html>", content_type="text/html")

    async with fake_firecrawl(respond) as (base_url, _):
        with pytest.raises(Exception, match="Firecrawl search failed for 'q': 502 Bad Gateway"):
            await _http_tool(base_url).run_async(FirecrawlSearchToolInputSchema(queries=["q"]))


@pytest.mark.asyncio
async def test_http_transport_error_does_not_expose_key(caplog):
    """aiohttp's own errors (here a redirect loop) carry the request headers; the tool must not re-raise them as is."""

    def respond(body):
        return web.Response(status=307, headers={"Location": "/v2/search"})

    async with fake_firecrawl(respond) as (base_url, seen):
        with caplog.at_level(logging.WARNING, logger="tool.firecrawl_search"):
            with pytest.raises(Exception) as raised:
                await _http_tool(base_url).run_async(FirecrawlSearchToolInputSchema(queries=["q"]))

    assert len(seen) > 1
    assert "Firecrawl search failed for 'q': TooManyRedirects" in str(raised.value)
    assert raised.value.__cause__ is None
    for text in (str(raised.value), repr(raised.value), caplog.text):
        assert DUMMY_KEY not in text


@pytest.mark.asyncio
async def test_http_dummy_key_kept_out_of_repr_errors_and_logs(caplog):
    """Even when the provider echoes the key back, it never reaches repr, raised errors, or logs."""

    def respond(body):
        if body["query"] == "bad":
            return web.json_response({"success": False, "error": f"Invalid token {DUMMY_KEY}"}, status=401)
        return web.json_response({"success": True, "data": {"web": [WEB_HIT]}})

    async with fake_firecrawl(respond) as (base_url, _):
        tool = _http_tool(base_url)
        with caplog.at_level(logging.WARNING, logger="tool.firecrawl_search"):
            out = await tool.run_async(FirecrawlSearchToolInputSchema(queries=["bad", "good"]))
        with pytest.raises(Exception) as raised:
            await tool.run_async(FirecrawlSearchToolInputSchema(queries=["bad"]))

    assert [r.query for r in out.results] == ["good"]
    assert "401 Invalid token [redacted]" in caplog.text
    assert "401 Invalid token [redacted]" in str(raised.value)
    for text in (caplog.text, str(raised.value), repr(raised.value), repr(tool), repr(tool.config), str(tool.config)):
        assert DUMMY_KEY not in text


@pytest.mark.asyncio
async def test_run_async_sends_bearer_header(tool):
    captured = {}

    class FakeSession:
        def __init__(self, headers=None, timeout=None):
            captured["headers"] = headers

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    async def fake_fetch(self, session, query, params):
        return []

    with (
        patch("tool.firecrawl_search.aiohttp.ClientSession", FakeSession),
        patch.object(FirecrawlSearchTool, "_fetch", fake_fetch),
    ):
        await tool.run_async(FirecrawlSearchToolInputSchema(queries=["q"]))

    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["headers"]["Content-Type"] == "application/json"


@pytest.mark.asyncio
async def test_run_async_requires_api_key(monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    tool = FirecrawlSearchTool(config=FirecrawlSearchToolConfig())

    with pytest.raises(ValueError, match="FIRECRAWL_API_KEY"):
        await tool.run_async(FirecrawlSearchToolInputSchema(queries=["q"]))


def test_api_key_falls_back_to_environment(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "env-key")
    tool = FirecrawlSearchTool(config=FirecrawlSearchToolConfig())
    assert tool._api_key == "env-key"


def test_run_invokes_run_async(tool):
    sentinel = FirecrawlSearchToolOutputSchema(results=[])
    with patch.object(FirecrawlSearchTool, "run_async", AsyncMock(return_value=sentinel)):
        out = tool.run(FirecrawlSearchToolInputSchema(queries=["x"]))
    assert out is sentinel


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
