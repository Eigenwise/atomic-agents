import asyncio
import copy
import logging
import os
import socket
import sys
import traceback
from contextlib import asynccontextmanager
from typing import AsyncIterator, Awaitable, Callable, Dict, List, Sequence, Tuple, Union
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

Body = Dict[str, object]
Responder = Callable[[Body], Union[web.Response, Awaitable[web.Response]]]

DUMMY_KEY = "fc-dummy0secret0key0123456789"  # a configured key in a custom format
CANONICAL_KEY = "fc-0123456789abcdef0123456789abcdef"  # a key in Firecrawl's own format
KEYS = (DUMMY_KEY, DUMMY_KEY.upper(), CANONICAL_KEY, CANONICAL_KEY.upper())
TOOL_FILE = os.path.join("tool", "firecrawl_search.py")
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}

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


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Keep the suite offline: any attempt to resolve a host other than the local test server fails the test."""
    attempts: List[str] = []
    real_getaddrinfo = socket.getaddrinfo

    def loopback_only(host, *args, **kwargs):
        if host not in LOOPBACK_HOSTS:
            attempts.append(str(host))
            raise socket.gaierror("network access is disabled in these tests")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", loopback_only)
    yield
    assert attempts == []


#####################
# Response builders #
#####################
def respond_json(payload: object, status: int = 200) -> Responder:
    def respond(body: Body) -> web.Response:
        return web.json_response(payload, status=status)

    return respond


def respond_hits(source: str, hits: Sequence[object]) -> Responder:
    return respond_json({"success": True, "data": {source: list(hits)}})


def respond_raw(**response) -> Responder:
    def respond(body: Body) -> web.Response:
        return web.Response(**response)

    return respond


def respond_per_query(bad_query: str, bad: Responder, good: Responder) -> Responder:
    def respond(body: Body) -> Union[web.Response, Awaitable[web.Response]]:
        return bad(body) if body["query"] == bad_query else good(body)

    return respond


################
# Test harness #
################
@asynccontextmanager
async def fake_firecrawl(respond: Responder) -> AsyncIterator[Tuple[str, List[Body]]]:
    """Serve POST /v2/search on a local aiohttp server; `respond(body)` builds each response."""
    seen: List[Body] = []

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        seen.append({"headers": dict(request.headers), "body": body})
        response = respond(body)
        return await response if asyncio.iscoroutine(response) else response

    app = web.Application()
    app.router.add_post("/v2/search", handler)
    server = TestServer(app, host="127.0.0.1")
    await server.start_server()
    try:
        yield str(server.make_url("/v2")), seen
    finally:
        await server.close()


def http_tool(base_url: str, api_key: str = DUMMY_KEY, **config) -> FirecrawlSearchTool:
    return FirecrawlSearchTool(config=FirecrawlSearchToolConfig(api_key=api_key, base_url=base_url, **config))


async def search(
    respond: Responder, queries: Sequence[str] = ("q",), api_key: str = DUMMY_KEY, **params
) -> Tuple[FirecrawlSearchToolOutputSchema, List[Body]]:
    """Run one search against a fake server and return (output, requests seen)."""
    async with fake_firecrawl(respond) as (base_url, seen):
        tool = http_tool(base_url, api_key)
        out = await tool.run_async(FirecrawlSearchToolInputSchema(queries=list(queries), **params))
    return out, seen


async def search_error(respond: Responder, queries: Sequence[str] = ("q",), api_key: str = DUMMY_KEY, **params) -> Exception:
    with pytest.raises(Exception) as raised:
        await search(respond, queries, api_key, **params)
    return raised.value


def frame_locals(error: BaseException, keep: Callable[[str], bool]) -> List[str]:
    stack = traceback.TracebackException.from_exception(error, capture_locals=True).stack
    return [repr(frame.locals) for frame in stack if keep(frame.filename)]


def is_tool_frame(filename: str) -> bool:
    return filename.endswith(TOOL_FILE)


def is_outside_this_test_file(filename: str) -> bool:
    """Library and tool frames; this file's own frames hold the test's inputs, which may contain the key."""
    return filename != __file__


def error_surfaces(error: BaseException) -> str:
    """Every text an error surfaces: str, repr, its chain, and the locals of every tool frame in its traceback."""
    tool_frames = frame_locals(error, is_tool_frame)
    assert tool_frames, "expected the traceback to pass through the tool"
    return "\n".join([str(error), repr(error), repr(error.__context__), repr(error.__cause__), *tool_frames])


def assert_no_key(*texts: str) -> None:
    for key in KEYS:
        assert not any(key in text for text in texts), key


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
    "url",
    [
        "HTTPS://Example.com/a",
        "http://example.com",
        "https://example.com:8443/a?b=c#d",
        "http://[::1]:8080/x",
        "https://user@example.com/",
    ],
)
def test_to_item_keeps_valid_http_urls(url):
    assert FirecrawlSearchTool._to_item({"url": url}, "q").url == url


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
        {"url": "javascript:alert(1)"},
        {"url": "ftp://example.com/file"},
        {"url": "httpx://example.com"},
        {"url": "https://"},
        {"url": "http:///path-without-host"},
        {"url": "https://@/a"},
        {"url": "https://:443/a"},
        {"url": "https://user@:443/a"},
        {"url": " https://example.com"},
        {"url": "https://example.com/a b"},
        {"url": "https://exa mple.com/a"},
        {"url": "https://example.com/a\n"},
        {"url": "https://example.com\t/a"},
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
    out, seen = await search(respond_hits("web", [CONTENT_HIT]), queries=["agents"], include_content=True)

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
    out, _ = await search(respond_json({"success": True, "data": {"web": [WEB_HIT], "news": [NEWS_HIT]}}), search_type="news")
    assert [r.url for r in out.results] == [NEWS_HIT["url"]]


@pytest.mark.asyncio
async def test_missing_source_key_is_an_empty_search():
    """The API omits the source key when there are no results; that is a real empty result."""
    out, _ = await search(respond_json({"success": True, "data": {}}))
    assert out == FirecrawlSearchToolOutputSchema(results=[], failures=[])


@pytest.mark.asyncio
async def test_results_trimmed_to_max_results():
    hits = [{"url": f"https://example.com/{i}"} for i in range(5)]
    out, _ = await search(respond_hits("web", hits), max_results_per_query=2)
    assert [r.url for r in out.results] == ["https://example.com/0", "https://example.com/1"]


@pytest.mark.asyncio
async def test_invalid_hits_skipped_valid_ones_kept():
    hits = [1, "s", None, {"title": "x"}, {"url": "https://@/a"}, WEB_HIT, {"url": "https://example.com/b", "position": "N/A"}]
    out, _ = await search(respond_hits("web", hits))
    assert [(r.url, r.position) for r in out.results] == [("https://www.python.org/", 1), ("https://example.com/b", None)]


@pytest.mark.asyncio
async def test_all_hits_invalid_raises():
    error = await search_error(respond_hits("web", [1, None, {"title": "x"}, {"url": "https://:443/a"}]))
    assert str(error) == "Firecrawl search failed for 'q': 200 none of the 4 'web' results had a usable url"


@pytest.mark.asyncio
async def test_duplicate_queries_sent_once():
    out, seen = await search(respond_hits("web", [WEB_HIT]), queries=["q", "q"])
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
    error = await search_error(respond_json(payload))
    assert str(error) == "Firecrawl search failed for 'q': 200 unexpected response shape for 'web' results"


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{"success": "false"}, {"success": "true"}, {"success": 1}, {}])
async def test_success_must_be_literally_true(payload):
    error = await search_error(respond_json({**payload, "data": {"web": [WEB_HIT]}}))
    assert str(error) == "Firecrawl search failed for 'q': 200 response did not report success"


@pytest.mark.asyncio
@pytest.mark.parametrize("status, reason", [(201, "Created"), (500, "Internal Server Error")])
async def test_non_200_with_success_true_raises(status, reason):
    error = await search_error(respond_json({"success": True, "data": {"web": [WEB_HIT]}}, status=status))
    assert str(error) == f"Firecrawl search failed for 'q': {status} {reason}"


@pytest.mark.asyncio
async def test_provider_error_message_is_reported():
    error = await search_error(respond_json({"success": False, "error": "Payment required"}, status=402))
    assert str(error) == "Firecrawl search failed for 'q': 402 Payment required"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response, expected",
    [
        ({"status": 502, "text": "<html>Bad Gateway</html>", "content_type": "text/html"}, "502 Bad Gateway"),
        ({"status": 200, "text": ""}, "200 response did not report success"),
        ({"status": 204}, "204 No Content"),
    ],
)
async def test_non_json_bodies_raise_with_status(response, expected):
    error = await search_error(respond_raw(**response))
    assert str(error) == f"Firecrawl search failed for 'q': {expected}"


@pytest.mark.asyncio
async def test_redirect_is_not_followed():
    """A redirect, even to another server, is reported as an error; the query is never re-sent elsewhere."""
    async with fake_firecrawl(respond_hits("web", [WEB_HIT])) as (other_url, other_seen):
        error = await search_error(respond_raw(status=307, headers={"Location": f"{other_url}/search"}))

    assert str(error) == "Firecrawl search failed for 'q': 307 Temporary Redirect"
    assert other_seen == []


###################
# Batch semantics #
###################
@pytest.mark.asyncio
async def test_partial_failure_reports_failures(caplog):
    rate_limited = respond_json({"success": False, "error": "Rate limit exceeded"}, status=429)
    with caplog.at_level(logging.WARNING, logger="tool.firecrawl_search"):
        out, _ = await search(respond_per_query("bad", rate_limited, respond_hits("web", [WEB_HIT])), queries=["bad", "good"])

    reason = "Firecrawl search failed for 'bad': 429 Rate limit exceeded"
    assert [r.query for r in out.results] == ["good"]
    assert out.failures == [FirecrawlSearchFailure(query="bad", error=reason)]
    assert caplog.messages == [reason]


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
    respond = respond_per_query("bad", respond_json(bad), respond_json({"success": True, "data": {}}))
    error = await search_error(respond, queries=["empty", "bad"])
    assert str(error) == f"Firecrawl search failed for 'bad': {expected}"


@pytest.mark.asyncio
async def test_every_query_failing_raises_first_error():
    error = await search_error(respond_json({"success": False, "error": "Unauthorized"}, status=401), queries=["a", "b"])
    assert str(error) == "Firecrawl search failed for 'a': 401 Unauthorized"


####################
# Transport errors #
####################
@pytest.mark.asyncio
async def test_connection_error_is_wrapped_without_context():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    with pytest.raises(Exception) as raised:
        await http_tool(f"http://127.0.0.1:{port}/v2").run_async(FirecrawlSearchToolInputSchema(queries=["q"]))

    assert str(raised.value).startswith("Firecrawl search failed for 'q': request failed: ClientConnectorError")
    assert raised.value.__context__ is None and raised.value.__cause__ is None
    assert_no_key(error_surfaces(raised.value))


@pytest.mark.asyncio
async def test_timeout_is_wrapped_with_query():
    async def slow(body: Body) -> web.Response:
        await asyncio.sleep(1)
        return web.json_response({"success": True, "data": {"web": [WEB_HIT]}})

    async with fake_firecrawl(slow) as (base_url, _):
        tool = http_tool(base_url)
        tool.timeout = 0.05
        with pytest.raises(Exception) as raised:
            await tool.run_async(FirecrawlSearchToolInputSchema(queries=["q"]))

    assert str(raised.value) == "Firecrawl search failed for 'q': request failed: TimeoutError after 0.05s"
    assert raised.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url, error_type",
    [
        (f"{DUMMY_KEY}/v2", "InvalidUrlClientError"),  # the error echoes the URL, so its text must be redacted
        ("http://[::1/v2", "InvalidUrlClientError"),
        ("http://127.0.0.1:99999/v2", "InvalidUrlClientError"),
        ("http://user:pw@127.0.0.1:1/v2", "ValueError"),
    ],
)
async def test_invalid_base_url_is_wrapped_and_no_frame_holds_the_key(base_url, error_type):
    """aiohttp raises for such URLs from frames whose locals hold the request headers; none of that may surface."""
    with pytest.raises(Exception) as raised:
        await http_tool(base_url).run_async(FirecrawlSearchToolInputSchema(queries=["q"]))

    assert str(raised.value).startswith(f"Firecrawl search failed for 'q': request failed: {error_type} ")
    assert raised.value.__context__ is None
    assert_no_key(str(raised.value), repr(raised.value), *frame_locals(raised.value, is_outside_this_test_file))


################
# Key handling #
################
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "echo", [DUMMY_KEY, f"Bearer {DUMMY_KEY}", DUMMY_KEY.upper(), {"token": DUMMY_KEY}, CANONICAL_KEY, CANONICAL_KEY.upper()]
)
async def test_echoed_key_kept_out_of_errors_and_logs(caplog, echo):
    """Even when the provider echoes a key back, it never reaches raised errors, tracebacks, logs or failures."""
    respond = respond_per_query(
        "bad", respond_json({"success": False, "error": echo}, status=401), respond_hits("web", [WEB_HIT])
    )
    with caplog.at_level(logging.DEBUG):
        out, _ = await search(respond, queries=["bad", "good"])
        error = await search_error(respond, queries=["bad"])

    assert out.failures == [FirecrawlSearchFailure(query="bad", error=str(error))]
    assert "[redacted]" in str(error)
    assert_no_key(caplog.text, error_surfaces(error), repr(out))


def malformed_body_echoing(key: str) -> Body:
    return {"success": True, "data": None, "error": {"token": key}}


def invalid_hits_echoing(key: str) -> Body:
    return {"success": True, "data": {"web": [{"url": "https://@/a", "title": key}, {"description": key}]}}


@pytest.mark.asyncio
@pytest.mark.parametrize("api_key", [DUMMY_KEY, CANONICAL_KEY])
@pytest.mark.parametrize("build_body", [malformed_body_echoing, invalid_hits_echoing])
async def test_rejected_success_body_not_kept_in_traceback_locals(api_key, build_body):
    """A malformed or all-invalid success body is dropped before the raise, so captured locals cannot show a key."""
    error = await search_error(respond_json(build_body(api_key)), api_key=api_key)
    assert str(error).startswith("Firecrawl search failed for 'q': 200 ")
    assert_no_key(error_surfaces(error))


@pytest.mark.asyncio
@pytest.mark.parametrize("api_key", [DUMMY_KEY, CANONICAL_KEY])
async def test_key_in_query_text_is_redacted_in_diagnostics(caplog, api_key):
    """A query that contains the key (for example pasted by mistake) must not leak it through errors or logs."""
    query = f"why does {api_key} fail"
    respond = respond_per_query(
        query, respond_json({"success": False, "error": "Unauthorized"}, status=401), respond_hits("web", [WEB_HIT])
    )
    with caplog.at_level(logging.DEBUG):
        out, _ = await search(respond, queries=[query, "good"], api_key=api_key)
        error = await search_error(respond, queries=[query], api_key=api_key)

    assert out.failures == [FirecrawlSearchFailure(query="why does [redacted] fail", error=str(error))]
    assert str(error) == "Firecrawl search failed for 'why does [redacted] fail': 401 Unauthorized"
    assert_no_key(caplog.text, repr(out.failures), str(error), repr(error))


def test_key_kept_out_of_tool_and_config_state():
    tool = FirecrawlSearchTool(config=FirecrawlSearchToolConfig(api_key=DUMMY_KEY))
    assert_no_key(
        repr(tool),
        repr(tool.config),
        str(tool.config),
        tool.config.model_dump_json(),
        repr(vars(tool)),
        repr(vars(copy.copy(tool))),
    )
    assert tool._api_key.get_secret_value() == DUMMY_KEY


def test_redaction_does_not_mangle_ordinary_text():
    """A one-letter key is not redacted (it would hit every 'a'), and fc- words that are not keys are left alone."""
    tool = FirecrawlSearchTool(config=FirecrawlSearchToolConfig(api_key="a"))
    text = "Firecrawl search failed for 'fc-barcelona-2024' and 'fc-0123abcd'"
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
