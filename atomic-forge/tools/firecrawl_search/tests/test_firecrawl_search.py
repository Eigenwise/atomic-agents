import asyncio
import copy
import logging
import os
import socket
import sys
import traceback
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from pydantic import ValidationError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tool.firecrawl_search import (  # noqa: E402
    FirecrawlSearchFailure,
    FirecrawlSearchResultItem,
    FirecrawlSearchTool,
    FirecrawlSearchToolConfig,
    FirecrawlSearchToolInputSchema,
    FirecrawlSearchToolOutputSchema,
)


DUMMY_KEY = "fc-dummy0secret0key0123456789"
OTHER_KEY = "fc-0123456789abcdef0123456789abcdef"

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
    "title": "Agent guide",
    "description": "A guide to agents.",
    "position": 3,
    "markdown": "# Agent guide\n\nBuild agents step by step.",
    "metadata": {"title": "Agent guide | Example", "statusCode": 200},
}


def ok(source: str, hits: list) -> web.Response:
    return web.json_response({"success": True, "data": {source: hits}})


@asynccontextmanager
async def fake_firecrawl(respond):
    """Serve POST /v2/search on a local aiohttp server; `respond(body)` builds each response."""
    seen = []

    async def handler(request):
        body = await request.json()
        seen.append({"headers": dict(request.headers), "body": body})
        response = respond(body)
        return await response if asyncio.iscoroutine(response) else response

    app = web.Application()
    app.router.add_post("/v2/search", handler)
    server = TestServer(app)
    await server.start_server()
    try:
        yield str(server.make_url("/v2")), seen
    finally:
        await server.close()


def http_tool(base_url: str, **config) -> FirecrawlSearchTool:
    return FirecrawlSearchTool(config=FirecrawlSearchToolConfig(api_key=DUMMY_KEY, base_url=base_url, **config))


async def search(respond, queries=("q",), **params):
    """Run one search against a fake server and return (output, requests seen)."""
    async with fake_firecrawl(respond) as (base_url, seen):
        out = await http_tool(base_url).run_async(FirecrawlSearchToolInputSchema(queries=list(queries), **params))
    return out, seen


async def search_error(respond, queries=("q",), **params) -> Exception:
    with pytest.raises(Exception) as raised:
        await search(respond, queries, **params)
    return raised.value


def leak_surfaces(error: BaseException) -> str:
    """Every text an error can surface through, including the locals of the tool's frames in its traceback."""
    stack = traceback.TracebackException.from_exception(error, capture_locals=True).stack
    tool_frames = [
        repr(frame.locals) for frame in stack if frame.filename.endswith(os.path.join("tool", "firecrawl_search.py"))
    ]
    assert tool_frames, "expected the traceback to pass through the tool"
    return "\n".join([str(error), repr(error), repr(error.__context__), repr(error.__cause__), *tool_frames])


##################
# Result parsing #
##################
def test_to_item_web():
    item = FirecrawlSearchTool._to_item(WEB_HIT, "python")
    assert item == FirecrawlSearchResultItem(
        query="python",
        title="Welcome to Python.org",
        url="https://www.python.org/",
        description="Python is a programming language that lets you work quickly.",
        position=1,
    )


def test_to_item_news_uses_snippet_and_date():
    item = FirecrawlSearchTool._to_item(NEWS_HIT, "python")
    assert (item.description, item.published, item.position) == ("The Python core team shipped 3.14 today.", "2 days ago", 2)


def test_to_item_content_and_truncation():
    assert FirecrawlSearchTool._to_item(CONTENT_HIT, "q").content == CONTENT_HIT["markdown"]
    assert FirecrawlSearchTool._to_item(CONTENT_HIT, "q", max_content_chars=7).content == "# Agent"


@pytest.mark.parametrize(
    "field, value, attribute, expected",
    [
        ("title", {"nested": True}, "title", "https://example.com/a"),
        ("title", 5, "title", "https://example.com/a"),
        ("description", ["a", "b"], "description", None),
        ("position", "N/A", "position", None),
        ("position", 1.5, "position", None),
        ("position", True, "position", None),
        ("position", -3, "position", None),
        ("position", 0, "position", None),
        ("date", 20250101, "published", None),
        ("markdown", 42, "content", None),
        ("markdown", {}, "content", None),
    ],
)
def test_to_item_drops_badly_typed_fields(field, value, attribute, expected):
    """One badly typed field must not discard the result or raise a raw ValidationError."""
    item = FirecrawlSearchTool._to_item({"url": "https://example.com/a", field: value}, "q")
    assert getattr(item, attribute) == expected


@pytest.mark.parametrize(
    "hit",
    [
        1,
        "s",
        None,
        [],
        {"title": "no url"},
        {"url": ""},
        {"url": 123},
        {"url": ["https://example.com"]},
        {"url": " https://example.com"},
        {"url": "javascript:alert(1)"},
        {"url": "ftp://example.com/file"},
    ],
)
def test_to_item_without_usable_url_is_none(hit):
    assert FirecrawlSearchTool._to_item(hit, "q") is None


################
# Request body #
################
def test_build_body_defaults():
    tool = FirecrawlSearchTool(config=FirecrawlSearchToolConfig(api_key="k"))
    assert tool._build_body("python", FirecrawlSearchToolInputSchema(queries=["python"])) == {
        "query": "python",
        "limit": 5,
        "sources": ["web"],
    }


def test_build_body_with_options():
    tool = FirecrawlSearchTool(config=FirecrawlSearchToolConfig(api_key="k", location="Germany", time_range="week"))
    params = FirecrawlSearchToolInputSchema(queries=["python"], max_results_per_query=3, include_content=True)
    assert tool._build_body("python", params) == {
        "query": "python",
        "limit": 3,
        "sources": ["web"],
        "location": "Germany",
        "tbs": "qdr:w",
        "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": True},
    }


def test_build_body_skips_time_range_for_news():
    """The API's tbs filter only applies to web results."""
    tool = FirecrawlSearchTool(config=FirecrawlSearchToolConfig(api_key="k", time_range="day"))
    body = tool._build_body("python", FirecrawlSearchToolInputSchema(queries=["python"], search_type="news"))
    assert body == {"query": "python", "limit": 5, "sources": ["news"]}


def test_empty_queries_rejected():
    with pytest.raises(ValidationError):
        FirecrawlSearchToolInputSchema(queries=[])


###################################
# Round trips over real transport #
###################################
@pytest.mark.asyncio
async def test_web_round_trip():
    out, seen = await search(lambda body: ok("web", [CONTENT_HIT]), queries=["agents"], include_content=True)

    assert out == FirecrawlSearchToolOutputSchema(results=[FirecrawlSearchTool._to_item(CONTENT_HIT, "agents")])
    assert seen[0]["headers"]["Authorization"] == f"Bearer {DUMMY_KEY}"
    assert seen[0]["body"] == {
        "query": "agents",
        "limit": 5,
        "sources": ["web"],
        "scrapeOptions": {"formats": ["markdown"], "onlyMainContent": True},
    }


@pytest.mark.asyncio
async def test_news_round_trip_reads_news_key_only():
    def respond(body):
        return web.json_response({"success": True, "data": {"web": [WEB_HIT], "news": [NEWS_HIT]}})

    out, _ = await search(respond, search_type="news")
    assert [r.url for r in out.results] == [NEWS_HIT["url"]]


@pytest.mark.asyncio
async def test_missing_source_key_is_an_empty_search():
    """The API omits the source key when there are no results; that is a real empty result."""
    out, _ = await search(lambda body: web.json_response({"success": True, "data": {}}))
    assert out == FirecrawlSearchToolOutputSchema(results=[], failures=[])


@pytest.mark.asyncio
async def test_results_trimmed_to_max_results():
    hits = [{"url": f"https://example.com/{i}"} for i in range(5)]
    out, _ = await search(lambda body: ok("web", hits), max_results_per_query=2)
    assert [r.url for r in out.results] == ["https://example.com/0", "https://example.com/1"]


@pytest.mark.asyncio
async def test_invalid_hits_skipped_valid_ones_kept():
    hits = [1, "s", None, {"title": "x"}, {"url": ""}, WEB_HIT, {"url": "https://example.com/b", "position": "N/A"}]
    out, _ = await search(lambda body: ok("web", hits))
    assert [(r.url, r.position) for r in out.results] == [("https://www.python.org/", 1), ("https://example.com/b", None)]


@pytest.mark.asyncio
async def test_all_hits_invalid_raises():
    error = await search_error(lambda body: ok("web", [1, None, {"title": "x"}, {"url": ""}]))
    assert str(error) == "Firecrawl search failed for 'q': 200 none of the 4 'web' results had a usable url"


@pytest.mark.asyncio
async def test_duplicate_queries_sent_once():
    out, seen = await search(lambda body: ok("web", [WEB_HIT]), queries=["q", "q"])
    assert len(seen) == 1
    assert [r.query for r in out.results] == ["q"]


#######################
# Response validation #
#######################
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
async def test_malformed_success_response_raises(payload):
    error = await search_error(lambda body: web.json_response(payload))
    assert str(error) == "Firecrawl search failed for 'q': 200 unexpected response shape for 'web' results"


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{"success": "false"}, {"success": "true"}, {"success": 1}, {}])
async def test_success_must_be_literally_true(payload):
    error = await search_error(lambda body: web.json_response({**payload, "data": {"web": [WEB_HIT]}}))
    assert str(error) == "Firecrawl search failed for 'q': 200 response did not report success"


@pytest.mark.asyncio
@pytest.mark.parametrize("status, reason", [(201, "Created"), (500, "Internal Server Error")])
async def test_non_200_with_success_true_raises(status, reason):
    def respond(body):
        return web.json_response({"success": True, "data": {"web": [WEB_HIT]}}, status=status)

    error = await search_error(respond)
    assert str(error) == f"Firecrawl search failed for 'q': {status} {reason}"


@pytest.mark.asyncio
async def test_provider_error_message_is_reported():
    error = await search_error(lambda body: web.json_response({"success": False, "error": "Payment required"}, status=402))
    assert str(error) == "Firecrawl search failed for 'q': 402 Payment required"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, expected",
    [
        (lambda: web.Response(status=502, text="<html>Bad Gateway</html>", content_type="text/html"), "502 Bad Gateway"),
        (lambda: web.Response(status=200, text=""), "200 response did not report success"),
        (lambda: web.Response(status=204), "204 No Content"),
    ],
)
async def test_non_json_bodies_raise_with_status(response, expected):
    error = await search_error(lambda body: response())
    assert str(error) == f"Firecrawl search failed for 'q': {expected}"


@pytest.mark.asyncio
async def test_redirect_is_not_followed():
    """A redirect, even to another host, is reported as an error; the query is never re-sent elsewhere."""
    async with fake_firecrawl(lambda body: ok("web", [WEB_HIT])) as (other_url, other_seen):
        error = await search_error(lambda body: web.Response(status=307, headers={"Location": f"{other_url}/search"}))

    assert str(error) == "Firecrawl search failed for 'q': 307 Temporary Redirect"
    assert other_seen == []


###################
# Batch semantics #
###################
@pytest.mark.asyncio
async def test_partial_failure_reports_failed_queries(caplog):
    def respond(body):
        if body["query"] == "bad":
            return web.json_response({"success": False, "error": "Rate limit exceeded"}, status=429)
        return ok("web", [WEB_HIT])

    with caplog.at_level(logging.WARNING, logger="tool.firecrawl_search"):
        out, _ = await search(respond, queries=["bad", "good"])

    reason = "Firecrawl search failed for 'bad': 429 Rate limit exceeded"
    assert [r.query for r in out.results] == ["good"]
    assert out.failures == [FirecrawlSearchFailure(query="bad", error=reason)]
    assert f"Firecrawl query 'bad' failed: {reason}" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad, expected",
    [
        ({"success": True, "data": None}, "200 unexpected response shape for 'web' results"),
        ({"success": False, "error": "Rate limit exceeded"}, "200 Rate limit exceeded"),
    ],
)
async def test_empty_plus_failed_query_raises(bad, expected):
    """No results anywhere plus a failure must not look like an empty search."""

    def respond(body):
        return web.json_response({"success": True, "data": {}} if body["query"] == "empty" else bad)

    error = await search_error(respond, queries=["empty", "bad"])
    assert str(error) == f"Firecrawl search failed for 'bad': {expected}"


@pytest.mark.asyncio
async def test_every_query_failing_raises_first_error():
    def respond(body):
        return web.json_response({"success": False, "error": "Unauthorized"}, status=401)

    error = await search_error(respond, queries=["a", "b"])
    assert str(error) == "Firecrawl search failed for 'a': 401 Unauthorized"


####################
# Transport errors #
####################
@pytest.mark.asyncio
async def test_connection_error_is_wrapped_without_context():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    tool = http_tool(f"http://127.0.0.1:{port}/v2")

    with pytest.raises(Exception) as raised:
        await tool.run_async(FirecrawlSearchToolInputSchema(queries=["q"]))

    assert str(raised.value).startswith("Firecrawl search failed for 'q': request failed: ClientConnectorError")
    assert raised.value.__context__ is None and raised.value.__cause__ is None
    assert DUMMY_KEY not in leak_surfaces(raised.value)


@pytest.mark.asyncio
async def test_timeout_is_wrapped_with_query():
    async def slow(body):
        await asyncio.sleep(1)
        return ok("web", [WEB_HIT])

    async with fake_firecrawl(slow) as (base_url, _):
        tool = http_tool(base_url)
        tool.timeout = 0.05
        with pytest.raises(Exception) as raised:
            await tool.run_async(FirecrawlSearchToolInputSchema(queries=["q"]))

    assert str(raised.value) == "Firecrawl search failed for 'q': request failed: TimeoutError after 0.05s"
    assert raised.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url",
    [
        "http://exa\nmple.com/v2",
        f"http://{'a' * 70}.example.com/v2",
        f"{DUMMY_KEY}/v2",  # aiohttp's InvalidUrlClientError echoes the URL, so its text must be redacted too
    ],
)
async def test_invalid_base_url_is_wrapped_and_no_frame_holds_the_key(base_url):
    """aiohttp raises ValueError/UnicodeError for such URLs from a frame whose locals hold the request headers."""
    with pytest.raises(Exception) as raised:
        await http_tool(base_url).run_async(FirecrawlSearchToolInputSchema(queries=["q"]))

    assert str(raised.value).startswith("Firecrawl search failed for 'q': request failed: ")
    assert raised.value.__context__ is None
    stack = traceback.TracebackException.from_exception(raised.value, capture_locals=True).stack
    non_test_frames = [repr(frame.locals) for frame in stack if frame.filename != __file__]
    assert non_test_frames
    assert DUMMY_KEY not in str(raised.value) + repr(raised.value) + "".join(non_test_frames)


################
# Key handling #
################
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "echo",
    [DUMMY_KEY, f"Bearer {DUMMY_KEY}", DUMMY_KEY.upper(), {"token": DUMMY_KEY}, OTHER_KEY, OTHER_KEY.upper()],
)
async def test_echoed_key_kept_out_of_errors_and_logs(caplog, echo):
    """Even when the provider echoes a key back, it never reaches raised errors, tracebacks or logs."""

    def respond(body):
        if body["query"] == "bad":
            return web.json_response({"success": False, "error": echo}, status=401)
        return ok("web", [WEB_HIT])

    with caplog.at_level(logging.DEBUG):
        out, _ = await search(respond, queries=["bad", "good"])
        error = await search_error(respond, queries=["bad"])

    assert [failure.query for failure in out.failures] == ["bad"]
    assert "[redacted]" in str(error) and "[redacted]" in caplog.text and "[redacted]" in out.failures[0].error
    for text in (caplog.text, leak_surfaces(error), repr(out)):
        assert all(key not in text for key in (DUMMY_KEY, DUMMY_KEY.upper(), OTHER_KEY, OTHER_KEY.upper()))


def test_key_kept_out_of_tool_and_config_state():
    tool = FirecrawlSearchTool(config=FirecrawlSearchToolConfig(api_key=DUMMY_KEY))
    surfaces = [repr(tool), repr(tool.config), str(tool.config), tool.config.model_dump_json(), repr(vars(tool))]
    surfaces.append(repr(vars(copy.copy(tool))))
    assert all(DUMMY_KEY not in text for text in surfaces)
    assert tool._api_key.get_secret_value() == DUMMY_KEY


def test_redaction_does_not_mangle_ordinary_text():
    """A one-letter key is not redacted (it would hit every 'a'), and fc- words that are not keys are left alone."""
    tool = FirecrawlSearchTool(config=FirecrawlSearchToolConfig(api_key="a"))
    text = "Firecrawl search failed for 'fc-barcelona-2024'"
    assert tool._redact(text) == text


@pytest.mark.asyncio
async def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("FIRECRAWL_API_KEY", raising=False)
    tool = FirecrawlSearchTool(config=FirecrawlSearchToolConfig())

    with pytest.raises(ValueError, match="FIRECRAWL_API_KEY"):
        await tool.run_async(FirecrawlSearchToolInputSchema(queries=["q"]))


def test_api_key_falls_back_to_environment(monkeypatch):
    monkeypatch.setenv("FIRECRAWL_API_KEY", "env-key")
    tool = FirecrawlSearchTool(config=FirecrawlSearchToolConfig())
    assert tool._api_key.get_secret_value() == "env-key"


def test_run_invokes_run_async():
    tool = FirecrawlSearchTool(config=FirecrawlSearchToolConfig(api_key="k"))
    sentinel = FirecrawlSearchToolOutputSchema(results=[])
    with patch.object(FirecrawlSearchTool, "run_async", AsyncMock(return_value=sentinel)):
        assert tool.run(FirecrawlSearchToolInputSchema(queries=["x"])) is sentinel


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
