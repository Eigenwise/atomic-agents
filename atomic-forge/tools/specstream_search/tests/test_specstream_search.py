import asyncio
import copy
import errno
import logging
import os
import socket
import sys
import traceback
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)
from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from pydantic import SecretStr, ValidationError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tool.specstream_search import (  # noqa: E402
    SpecstreamSearchAnswer,
    SpecstreamSearchFailure,
    SpecstreamSearchResultItem,
    SpecstreamSearchTool,
    SpecstreamSearchToolConfig,
    SpecstreamSearchToolInputSchema,
    SpecstreamSearchToolOutputSchema,
)

Body = Dict[str, object]
Responder = Callable[[Body], Union[web.Response, Awaitable[web.Response]]]


Address = Union[Tuple[object, ...], str, bytes]
AddrInfo = Tuple[
    socket.AddressFamily,
    socket.SocketKind,
    int,
    str,
    Union[Tuple[str, int], Tuple[str, int, int, int], Tuple[int, bytes]],
]


class Seen(TypedDict):
    """One request as the fake server received it."""

    headers: Dict[str, str]
    body: Body


DUMMY_KEY = "custom0dummy0secret0key0123456789"
CANONICAL_KEY = "ss_live_0123456789abcdefABCDEF0123456789abcdef"
KEYS = (DUMMY_KEY, DUMMY_KEY.upper(), CANONICAL_KEY, CANONICAL_KEY.upper())
TOOL_FILE = os.path.join("tool", "specstream_search.py")
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
CALLER_INPUT_LOCALS = {"query", "queries", "params"}

WEB_SOURCE = {
    "url": "https://www.python.org/",
    "title": "Welcome to Python.org",
    "site": "python.org",
    "snippet": "Python is a programming language that lets you work quickly.",
    "cited": True,
    "excerpt": "Python is a programming language that lets you work quickly and integrate systems more effectively.",
}

PLAIN_SOURCE = {
    "url": "https://example.com/guide",
    "title": "Agent guide",
    "site": "example.com",
    "snippet": "A guide to agents.",
    "cited": False,
}

NEWS_SOURCE = {
    "url": "https://example.com/python-3-14",
    "title": "Python 3.14 released",
    "site": "example.com",
    "snippet": "The Python core team shipped 3.14 today.",
    "cited": False,
    "published": "2026-10-07",
    "source": "Example News",
}

ANSWER_OBJ = {
    "value": "Guido van Rossum",
    "page": "https://www.python.org/",
    "seen_at": "2026-11-05T01:06:52+00:00",
}


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> Iterator[List[str]]:
    """Keep the suite offline. Resolving or connecting outside loopback fails."""
    attempts: List[str] = []
    real_getaddrinfo = socket.getaddrinfo
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def resolve_loopback_only(
        host: Union[bytes, str, None], port: Union[bytes, str, int, None], *args: int, **kwargs: int
    ) -> Sequence[AddrInfo]:
        if host not in LOOPBACK_HOSTS:
            attempts.append(str(host))
            raise socket.gaierror("network access is disabled in these tests")
        return real_getaddrinfo(host, port, *args, **kwargs)

    def blocked(address: Address) -> bool:
        if isinstance(address, tuple) and address[0] not in LOOPBACK_HOSTS:
            attempts.append(str(address[0]))
            return True
        return False

    def connect_loopback_only(sock: socket.socket, address: Address) -> None:
        if blocked(address):
            raise OSError("network access is disabled in these tests")
        real_connect(sock, address)

    def connect_ex_loopback_only(sock: socket.socket, address: Address) -> int:
        return errno.ECONNREFUSED if blocked(address) else real_connect_ex(sock, address)

    monkeypatch.setattr(socket, "getaddrinfo", resolve_loopback_only)
    monkeypatch.setattr(socket.socket, "connect", connect_loopback_only)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex_loopback_only)
    yield attempts
    assert attempts == []


#####################
# Response builders #
#####################
def respond_json(payload: object, status: int = 200) -> Responder:
    def respond(body: Body) -> web.Response:
        return web.json_response(payload, status=status)

    return respond


def respond_sources(
    sources: Sequence[object],
    answer: Optional[object] = None,
    passages: Optional[Sequence[object]] = None,
) -> Responder:
    data: Dict[str, object] = {
        "request_id": "req_123",
        "query": "q",
        "kind": "fact",
        "columns": [],
        "answer": answer,
        "rows": [],
        "sources": list(sources),
        "handoff": None,
        "took_ms": 150,
        "price_usd": 0.005,
    }
    if passages is not None:
        data["passages"] = list(passages)
    return respond_json({"data": data})


def respond_raw(
    status: int, text: Optional[str] = None, content_type: Optional[str] = None, location: Optional[str] = None
) -> Responder:
    def respond(body: Body) -> web.Response:
        headers = {"Location": location} if location else None
        return web.Response(status=status, text=text, content_type=content_type, headers=headers)

    return respond


def respond_per_query(bad_query: str, bad: Responder, good: Responder) -> Responder:
    def respond(body: Body) -> Union[web.Response, Awaitable[web.Response]]:
        return bad(body) if body["q"] == bad_query else good(body)

    return respond


################
# Test harness #
################
@asynccontextmanager
async def fake_specstream(respond: Responder) -> AsyncIterator[Tuple[str, List[Seen]]]:
    """Serve POST /v1/web-search on a local aiohttp server; `respond(body)` builds each response."""
    seen: List[Seen] = []

    async def handler(request: web.Request) -> web.Response:
        body = await request.json()
        seen.append(Seen(headers=dict(request.headers), body=body))
        response = respond(body)
        if isinstance(response, web.Response):
            return response
        return await response

    app = web.Application()
    app.router.add_post("/v1/web-search", handler)
    server = TestServer(app, host="127.0.0.1")
    await server.start_server()
    try:
        yield str(server.make_url("/v1")), seen
    finally:
        await server.close()


def http_tool(base_url: str, api_key: str = DUMMY_KEY, **options: Any) -> SpecstreamSearchTool:
    return SpecstreamSearchTool(config=SpecstreamSearchToolConfig(api_key=SecretStr(api_key), base_url=base_url, **options))


def search_input(
    queries: Sequence[str],
    topic: Literal["web", "news"] = "web",
    max_results_per_query: int = 10,
    include_passages: bool = False,
) -> SpecstreamSearchToolInputSchema:
    return SpecstreamSearchToolInputSchema(
        queries=list(queries),
        topic=topic,
        max_results_per_query=max_results_per_query,
        include_passages=include_passages,
    )


async def search(
    respond: Responder,
    queries: Sequence[str] = ("q",),
    api_key: str = DUMMY_KEY,
    topic: Literal["web", "news"] = "web",
    max_results_per_query: int = 10,
    include_passages: bool = False,
    **options: Any,
) -> Tuple[SpecstreamSearchToolOutputSchema, List[Seen]]:
    """Run one search against a fake server and return (output, requests seen)."""
    params = search_input(queries, topic, max_results_per_query, include_passages)
    async with fake_specstream(respond) as (base_url, seen):
        out = await http_tool(base_url, api_key, **options).run_async(params)
    return out, seen


async def search_error(respond: Responder, queries: Sequence[str] = ("q",), api_key: str = DUMMY_KEY) -> BaseException:
    with pytest.raises(Exception) as raised:
        await search(respond, queries, api_key)
    return raised.value


def kept_item(
    hit: object,
    query: str = "q",
    max_content_chars: Optional[int] = None,
    passage_by_url: Optional[Dict[str, str]] = None,
) -> SpecstreamSearchResultItem:
    """_to_item for a hit the test expects to be kept."""
    result = SpecstreamSearchTool._to_item(hit, query, max_content_chars, passage_by_url)
    assert result is not None
    return result


def without_names(names: Dict[str, str], skip_names: Sequence[str]) -> Dict[str, str]:
    return {name: value for name, value in names.items() if name not in skip_names}


def frame_locals(error: BaseException, keep: Callable[[str], bool], skip_names: Sequence[str] = ()) -> List[str]:
    """The captured locals of every frame in the error's traceback whose file passes `keep`."""
    stack = traceback.TracebackException.from_exception(error, capture_locals=True).stack
    return [repr(without_names(frame.locals or {}, skip_names)) for frame in stack if keep(frame.filename)]


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
def test_to_item_with_excerpt() -> None:
    item = kept_item(WEB_SOURCE, "python")
    assert item == SpecstreamSearchResultItem(
        query="python",
        title="Welcome to Python.org",
        url="https://www.python.org/",
        description="Python is a programming language that lets you work quickly.",
        site="python.org",
        cited=True,
        content=WEB_SOURCE["excerpt"],
    )


def test_to_item_news_has_date_and_source() -> None:
    item = kept_item(NEWS_SOURCE, "python")
    assert (item.description, item.published, item.source) == (
        "The Python core team shipped 3.14 today.",
        "2026-10-07",
        "Example News",
    )


def test_to_item_falls_back_to_passages_map() -> None:
    item = kept_item(PLAIN_SOURCE, passage_by_url={"https://example.com/guide": "Full passage text"})
    assert item.content == "Full passage text"


def test_to_item_content_truncation() -> None:
    assert kept_item(WEB_SOURCE, "q", max_content_chars=6).content == "Python"


def test_to_answer_valid_and_invalid() -> None:
    ans = SpecstreamSearchTool._to_answer(ANSWER_OBJ, "who created python")
    assert ans == SpecstreamSearchAnswer(
        query="who created python",
        value="Guido van Rossum",
        page="https://www.python.org/",
        seen_at="2026-11-05T01:06:52+00:00",
    )
    assert SpecstreamSearchTool._to_answer(None, "q") is None
    assert SpecstreamSearchTool._to_answer({"value": "", "page": "https://python.org"}, "q") is None
    assert SpecstreamSearchTool._to_answer({"value": "x", "page": "javascript:alert(1)"}, "q") is None


def test_passage_map_skips_bad_entries_and_duplicates() -> None:
    data = {
        "passages": [
            "not-a-dict",
            {"url": "", "text": "a"},
            {"url": "https://example.com/1", "text": "first"},
            {"url": "https://example.com/1", "text": "second"},
        ]
    }
    assert SpecstreamSearchTool._passage_map(data) == {"https://example.com/1": "first"}


@pytest.mark.parametrize(
    "field, value, attribute, expected",
    [
        ("title", {"nested": True}, "title", "https://example.com/a"),
        ("title", 5, "title", "https://example.com/a"),
        ("snippet", ["a", "b"], "description", None),
        ("site", 123, "site", None),
        ("published", 20250101, "published", None),
        ("source", {"name": "x"}, "source", None),
        ("cited", "true", "cited", None),
        ("cited", 1, "cited", None),
        ("excerpt", 42, "content", None),
        ("excerpt", {}, "content", None),
    ],
)
def test_to_item_drops_badly_typed_fields(field: str, value: object, attribute: str, expected: object) -> None:
    """One badly typed field must not discard the result or raise a raw ValidationError."""
    hit = {"url": "https://example.com/a", "excerpt": "page", field: value}
    kept = kept_item(hit)
    assert getattr(kept, attribute) == expected


@pytest.mark.parametrize(
    "url",
    [
        "HTTPS://Example.com/a",
        "http://example.com",
        "https://example.com:8443/a?b=c#d",
        "http://[::1]:8080/x",
        "https://user@example.com/",
        "http://192.0.2.1:8080/x",
        "https://müller.de/straße",
        "https://xn--mller-kva.de/",
        "https://example.com/%C3%BC%20x.pdf",
        "https://example.com/a\\b",
        "https://example.com./",
        "https://my_host.example.com/",
        f"https://{'a' * 63}.com/",
        f"https://{'a' * 63}.{'b' * 63}.{'c' * 63}.{'d' * 61}/",
        "https://1.2.3.4/",
        "https://stra\u00dfe.de/",
        "https://\u05e9\u05dc\u05d5\u05dd.co.il/",
        "https://\u4f8b\u5b50.\u6d4b\u8bd5/",
        f"https://{'\u00e9' * 20}.com/",
        "https://example\u3002com/",
        "https://example.com\u3002/",
        "https://1\uff0e2\uff0e3\uff0e4/",
        "https://3com.example.123abc/",
    ],
)
def test_to_item_keeps_valid_http_urls(url: str) -> None:
    assert kept_item({"url": url}).url == url


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
        {"url": "https://./"},
        {"url": "https://-/"},
        {"url": "https://%/"},
        {"url": "https://_x.com/"},
        {"url": "https://[:]/"},
        {"url": "https://[zz]/"},
        {"url": "https://a\x00b.com/"},
        {"url": "https://example.com/a\x7f"},
        {"url": "https://exa\u200bmple.com/"},
        {"url": "https://example.com/\xa0"},
        {"url": "https://example.com:abc/"},
        {"url": "https://example.com:99999/"},
        {"url": "https://example.com:65536/"},
        {"url": "https://example.com:0/"},
        {"url": "https://example.com:\u0664\u0664\u0663/"},
        {"url": "https://a b@example.com/"},
        {"url": "https://[v1.x]/"},
        {"url": "https://[::1%25eth0]/"},
        {"url": "https://a.com\\@evil.com/"},
        {"url": "https://evil.com\\.example.com/"},
        {"url": "https://ev\\il.com/"},
        {"url": "https://example..com/a"},
        {"url": "https://a%.com/a"},
        {"url": "https://a%41b.com/a"},
        {"url": "https://.example.com/"},
        {"url": "https://-a.com/"},
        {"url": "https://a-.com/"},
        {"url": f"https://{'a' * 64}.com/"},
        {"url": "https://example.com../"},
        {"url": f"https://{('a' * 63 + '.') * 4}com/"},
        {"url": f"https://{'a' * 63}.{'b' * 63}.{'c' * 63}.{'d' * 62}/"},
        {"url": "https://a.com:1\\@b.com/"},
        {"url": "https://a.com:x@/"},
        {"url": "https://256.0.0.1/"},
        {"url": "https://1.2.3/"},
        {"url": "https://999999/"},
        {"url": "https://example.\u0661\u0662\u0663/"},
        {"url": "https://a\u05d0.com/a"},
        {"url": f"https://{'\u00e9' * 63}.com/"},
        {"url": f"https://{('\u00e9' * 20 + '.') * 10}com/"},
        {"url": "https://256.0.0\uff0e1/"},
        {"url": "https://1.2\u30023/"},
        {"url": "https://a\uff61999999/"},
        {"url": "https://a\u2488./"},
    ],
)
def test_to_item_without_usable_url_is_none(hit: object) -> None:
    assert SpecstreamSearchTool._to_item(hit, "q") is None


################
# Request body #
################
def test_build_body_defaults() -> None:
    tool = SpecstreamSearchTool(config=SpecstreamSearchToolConfig(api_key=SecretStr("k")))
    assert tool._build_body("python", SpecstreamSearchToolInputSchema(queries=["python"])) == {
        "q": "python",
        "num": 10,
        "speed": "default",
    }


def test_build_body_with_options() -> None:
    config = SpecstreamSearchToolConfig(
        api_key=SecretStr("k"),
        speed="fast",
        language="de",
        country="DE",
        safe="moderate",
        engines="-CN",
        pages_wait_ms=1500,
    )
    params = SpecstreamSearchToolInputSchema(
        queries=["python"],
        topic="web",
        max_results_per_query=20,
        include_passages=True,
    )
    assert SpecstreamSearchTool(config=config)._build_body("python", params) == {
        "q": "python",
        "num": 20,
        "speed": "fast",
        "passages": True,
        "pages_wait_ms": 1500,
        "language": "de",
        "country": "DE",
        "safe": "moderate",
        "engines": "-CN",
    }


def test_build_body_news() -> None:
    tool = SpecstreamSearchTool(config=SpecstreamSearchToolConfig(api_key=SecretStr("k")))
    params = SpecstreamSearchToolInputSchema(queries=["ai"], topic="news", max_results_per_query=5)
    assert tool._build_body("ai", params) == {"q": "ai", "num": 5, "speed": "default", "news": True}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"language": "eng"},
        {"country": "USA"},
        {"safe": "invalid"},
        {"pages_wait_ms": 4000},
        {"speed": "turbo"},
    ],
)
def test_invalid_config_rejected(kwargs: Dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        SpecstreamSearchToolConfig(**kwargs)


@pytest.mark.parametrize(
    "params",
    [
        {"queries": []},
        {"queries": ["   "]},
        {"queries": ["x" * 2001]},
        {"queries": ["q"], "max_results_per_query": 101},
        {"queries": ["q"], "topic": "images"},
        {"queries": ["q"], "topic": "news", "include_passages": True},
    ],
)
def test_invalid_input_rejected(params: Dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        SpecstreamSearchToolInputSchema(**params)


###################################
# Round trips over real transport #
###################################
@pytest.mark.asyncio
async def test_round_trip_with_answer_and_passages() -> None:
    passages = [{"url": "https://example.com/guide", "title": "Agent guide", "text": "Extracted passage"}]
    out, seen = await search(
        respond_sources([WEB_SOURCE, PLAIN_SOURCE], answer=ANSWER_OBJ, passages=passages),
        queries=["who created python"],
        include_passages=True,
    )

    assert out == SpecstreamSearchToolOutputSchema(
        results=[
            kept_item(WEB_SOURCE, "who created python"),
            kept_item(PLAIN_SOURCE, "who created python", passage_by_url={"https://example.com/guide": "Extracted passage"}),
        ],
        answers=[
            SpecstreamSearchAnswer(
                query="who created python",
                value="Guido van Rossum",
                page="https://www.python.org/",
                seen_at="2026-11-05T01:06:52+00:00",
            )
        ],
    )
    assert seen[0]["headers"]["Authorization"] == f"Bearer {DUMMY_KEY}"
    assert seen[0]["body"] == {"q": "who created python", "num": 10, "speed": "default", "passages": True}


@pytest.mark.asyncio
async def test_news_round_trip_sends_news_and_config_filters() -> None:
    out, seen = await search(
        respond_sources([NEWS_SOURCE]),
        topic="news",
        speed="fast",
        language="en",
        country="US",
    )
    assert [(r.published, r.source) for r in out.results] == [("2026-10-07", "Example News")]
    assert seen[0]["body"] == {
        "q": "q",
        "num": 10,
        "speed": "fast",
        "news": True,
        "language": "en",
        "country": "US",
    }


@pytest.mark.asyncio
async def test_empty_sources_list_is_an_empty_search() -> None:
    out, _ = await search(respond_sources([]))
    assert out == SpecstreamSearchToolOutputSchema(results=[], answers=[], failures=[])


@pytest.mark.asyncio
async def test_results_trimmed_to_max_results() -> None:
    hits = [{"url": f"https://example.com/{i}"} for i in range(5)]
    out, _ = await search(respond_sources(hits), max_results_per_query=2)
    assert [r.url for r in out.results] == ["https://example.com/0", "https://example.com/1"]


@pytest.mark.asyncio
async def test_invalid_hits_skipped_valid_ones_kept() -> None:
    hits = [1, "s", None, {"title": "x"}, {"url": "https://@/a"}, WEB_SOURCE, {"url": "https://example.com/b", "title": 3}]
    out, _ = await search(respond_sources(hits))
    assert [(r.url, r.title) for r in out.results] == [
        ("https://www.python.org/", "Welcome to Python.org"),
        ("https://example.com/b", "https://example.com/b"),
    ]


@pytest.mark.asyncio
async def test_all_hits_invalid_raises() -> None:
    error = await search_error(respond_sources([1, None, {"title": "x"}, {"url": "https://:443/a"}]))
    assert str(error) == "SpecStream search failed for 'q': 200 none of the 4 results had a usable url"


@pytest.mark.asyncio
async def test_duplicate_queries_sent_once() -> None:
    out, seen = await search(respond_sources([WEB_SOURCE]), queries=["q", "q"])
    assert len(seen) == 1
    assert [r.query for r in out.results] == ["q"]


#######################
# Response validation #
#######################
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload, expected",
    [
        ({"data": None}, "200 unexpected response shape: no data object"),
        ({"sources": []}, "200 unexpected response shape: no data object"),
        ({"data": {"sources": None}}, "200 unexpected response shape: no sources list"),
        ({"data": {"query": "q"}}, "200 unexpected response shape: no sources list"),
        ({"data": {"sources": {"url": "https://example.com"}}}, "200 unexpected response shape: no sources list"),
    ],
)
async def test_malformed_success_response_raises(payload: Body, expected: str) -> None:
    error = await search_error(respond_json(payload))
    assert str(error) == f"SpecStream search failed for 'q': {expected}"


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [[WEB_SOURCE], {}, "sources", 1])
async def test_success_body_must_be_a_json_object(payload: object) -> None:
    error = await search_error(respond_json(payload))
    assert str(error) == "SpecStream search failed for 'q': 200 response was not a JSON object"


@pytest.mark.asyncio
@pytest.mark.parametrize("status, reason", [(201, "Created"), (500, "Internal Server Error")])
async def test_non_200_with_sources_raises(status: int, reason: str) -> None:
    error = await search_error(respond_json({"data": {"sources": [WEB_SOURCE]}}, status=status))
    assert str(error) == f"SpecStream search failed for 'q': {status} {reason}"


@pytest.mark.asyncio
async def test_error_envelope_is_reported() -> None:
    body = {"error": {"code": "invalid_token", "message": "Invalid API key"}}
    error = await search_error(respond_json(body, status=401))
    assert str(error) == "SpecStream search failed for 'q': 401 invalid_token: Invalid API key"


@pytest.mark.asyncio
async def test_insufficient_balance_is_reported() -> None:
    body = {
        "error": {
            "code": "insufficient_balance",
            "message": "Balance is empty. Add credit at https://specstream.ai/workspace",
        }
    }
    error = await search_error(respond_json(body, status=402))
    assert (
        str(error) == "SpecStream search failed for 'q': 402 insufficient_balance: Balance is empty. "
        "Add credit at https://specstream.ai/workspace"
    )


@pytest.mark.asyncio
async def test_validation_detail_fallback_are_joined() -> None:
    body = {"detail": [{"loc": ["body", "language"], "msg": "bad language"}, {"msg": "bad country"}, "junk"]}
    error = await search_error(respond_json(body, status=422))
    assert str(error) == "SpecStream search failed for 'q': 422 bad language; bad country"


@pytest.mark.asyncio
async def test_rate_limit_reports_retry_after() -> None:
    def rate_limited(body: Body) -> web.Response:
        return web.json_response(
            {"error": {"code": "rate_limited", "message": "Rate limit exceeded"}},
            status=503,
            headers={"Retry-After": "2"},
        )

    error = await search_error(rate_limited)
    assert str(error) == "SpecStream search failed for 'q': 503 rate_limited: Rate limit exceeded (retry after 2s)"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, text, content_type, expected",
    [
        (502, "<html>Bad Gateway</html>", "text/html", "502 Bad Gateway"),
        (200, "", None, "200 response was not a JSON object"),
        (204, None, None, "204 No Content"),
    ],
)
async def test_non_json_bodies_raise_with_status(
    status: int, text: Optional[str], content_type: Optional[str], expected: str
) -> None:
    error = await search_error(respond_raw(status, text, content_type))
    assert str(error) == f"SpecStream search failed for 'q': {expected}"


@pytest.mark.asyncio
async def test_deeply_nested_json_is_a_wrapped_failure() -> None:
    """JSON too deep for the parser must fail like any other bad body, not escape with the response in scope."""
    error = await search_error(respond_raw(200, "[" * 200_000, "application/json"))
    assert str(error) == "SpecStream search failed for 'q': 200 response was not a JSON object"
    assert_no_key(error_surfaces(error))


@pytest.mark.asyncio
async def test_redirect_is_not_followed() -> None:
    """A redirect, even to another server, is reported as an error; the query is never re-sent elsewhere."""
    async with fake_specstream(respond_sources([WEB_SOURCE])) as (other_url, other_seen):
        error = await search_error(respond_raw(307, location=f"{other_url}/web-search"))

    assert str(error) == "SpecStream search failed for 'q': 307 Temporary Redirect"
    assert other_seen == []


###################
# Batch semantics #
###################
@pytest.mark.asyncio
async def test_partial_failure_reports_failures(caplog: pytest.LogCaptureFixture) -> None:
    unavailable = respond_json({"error": {"code": "unavailable", "message": "Search is temporarily unavailable"}}, status=503)
    with caplog.at_level(logging.WARNING, logger="tool.specstream_search"):
        out, _ = await search(respond_per_query("bad", unavailable, respond_sources([WEB_SOURCE])), queries=["bad", "good"])

    reason = "SpecStream search failed for 'bad': 503 unavailable: Search is temporarily unavailable"
    assert [r.query for r in out.results] == ["good"]
    assert out.failures == [SpecstreamSearchFailure(query="bad", error=reason)]
    assert caplog.messages == [reason]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad, status, expected",
    [
        ({"data": {"sources": None}}, 200, "200 unexpected response shape: no sources list"),
        ({"error": {"code": "bad_gateway", "message": "Search failed"}}, 502, "502 bad_gateway: Search failed"),
    ],
)
async def test_empty_plus_failed_query_raises(bad: Body, status: int, expected: str) -> None:
    """No results anywhere plus a failure must not look like an empty search."""
    respond = respond_per_query("bad", respond_json(bad, status=status), respond_sources([]))
    error = await search_error(respond, queries=["empty", "bad"])
    assert str(error) == f"SpecStream search failed for 'bad': {expected}"


@pytest.mark.asyncio
async def test_every_query_failing_raises_first_error() -> None:
    error = await search_error(
        respond_json({"error": {"code": "invalid_token", "message": "Invalid API key"}}, status=401),
        queries=["a", "b"],
    )
    assert str(error) == "SpecStream search failed for 'a': 401 invalid_token: Invalid API key"


####################
# Transport errors #
####################
@pytest.mark.asyncio
async def test_connection_error_is_wrapped_without_context() -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    with pytest.raises(Exception) as raised:
        await http_tool(f"http://127.0.0.1:{port}/v1").run_async(SpecstreamSearchToolInputSchema(queries=["q"]))

    assert str(raised.value).startswith("SpecStream search failed for 'q': request failed: ClientConnectorError")
    assert raised.value.__context__ is None and raised.value.__cause__ is None
    assert_no_key(error_surfaces(raised.value))


@pytest.mark.asyncio
async def test_timeout_is_wrapped_with_query() -> None:
    async def slow(body: Body) -> web.Response:
        await asyncio.sleep(1)
        return web.json_response({"data": {"sources": [WEB_SOURCE]}})

    async with fake_specstream(slow) as (base_url, _):
        tool = http_tool(base_url)
        tool.timeout = 0.05
        with pytest.raises(Exception) as raised:
            await tool.run_async(SpecstreamSearchToolInputSchema(queries=["q"]))

    assert str(raised.value) == "SpecStream search failed for 'q': request failed: TimeoutError after 0.05s"
    assert raised.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "base_url, error_type",
    [
        (f"{DUMMY_KEY}/v1", "InvalidUrlClientError"),
        ("http://[::1/v1", "InvalidUrlClientError"),
        ("http://127.0.0.1:99999/v1", "InvalidUrlClientError"),
        ("http://user:pw@127.0.0.1:1/v1", "ValueError"),
    ],
)
async def test_invalid_base_url_is_wrapped_and_no_frame_holds_the_key(base_url: str, error_type: str) -> None:
    """aiohttp raises for such URLs from frames whose locals hold the request headers; none of that may surface."""
    with pytest.raises(Exception) as raised:
        await http_tool(base_url).run_async(SpecstreamSearchToolInputSchema(queries=["q"]))

    assert str(raised.value).startswith(f"SpecStream search failed for 'q': request failed: {error_type} ")
    assert raised.value.__context__ is None
    assert_no_key(str(raised.value), repr(raised.value), *frame_locals(raised.value, is_outside_this_test_file))


@pytest.mark.asyncio
@pytest.mark.parametrize("base_url", ["http://localhost:9/v1", "http://local\nhost:9/v1"])
async def test_unresolvable_base_url_is_wrapped(monkeypatch: pytest.MonkeyPatch, base_url: str) -> None:
    """A DNS failure is reported as a request failure without external lookups."""
    lookups: List[str] = []

    def unresolvable(host: Union[bytes, str, None], *args: object, **kwargs: object) -> Sequence[AddrInfo]:
        lookups.append(str(host))
        raise socket.gaierror("name resolution failed")

    monkeypatch.setattr(socket, "getaddrinfo", unresolvable)
    with pytest.raises(Exception) as raised:
        await http_tool(base_url).run_async(search_input(["q"]))

    assert str(raised.value).startswith("SpecStream search failed for 'q': request failed: ClientConnectorDNSError ")
    assert_no_key(str(raised.value), repr(raised.value), *frame_locals(raised.value, is_outside_this_test_file))
    assert set(lookups) == {"localhost"}


################
# Key handling #
################
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "echo", [DUMMY_KEY, f"Bearer {DUMMY_KEY}", DUMMY_KEY.upper(), {"token": DUMMY_KEY}, CANONICAL_KEY, CANONICAL_KEY.upper()]
)
async def test_echoed_key_kept_out_of_errors_and_logs(caplog: pytest.LogCaptureFixture, echo: object) -> None:
    """Even when the provider echoes a key back, it never reaches raised errors, tracebacks, logs or failures."""
    respond = respond_per_query("bad", respond_json({"detail": echo}, status=401), respond_sources([WEB_SOURCE]))
    with caplog.at_level(logging.DEBUG):
        out, _ = await search(respond, queries=["bad", "good"])
        error = await search_error(respond, queries=["bad"])

    assert out.failures == [SpecstreamSearchFailure(query="bad", error=str(error))]
    assert "[redacted]" in str(error)
    assert_no_key(caplog.text, error_surfaces(error), repr(out))


def malformed_body_echoing(key: str) -> Body:
    return {"data": {"sources": None}, "detail": {"token": key}}


def invalid_hits_echoing(key: str) -> Body:
    return {"data": {"sources": [{"url": "https://@/a", "title": key}, {"snippet": key}]}}


@pytest.mark.asyncio
@pytest.mark.parametrize("api_key", [DUMMY_KEY, CANONICAL_KEY])
@pytest.mark.parametrize("build_body", [malformed_body_echoing, invalid_hits_echoing])
async def test_rejected_success_body_not_kept_in_traceback_locals(api_key: str, build_body: Callable[[str], Body]) -> None:
    """A malformed or all-invalid success body is dropped before the raise, so captured locals cannot show a key."""
    error = await search_error(respond_json(build_body(api_key)), api_key=api_key)
    assert str(error).startswith("SpecStream search failed for 'q': 200 ")
    assert_no_key(error_surfaces(error))


@pytest.mark.asyncio
@pytest.mark.parametrize("api_key", [DUMMY_KEY, CANONICAL_KEY])
async def test_key_in_query_text_is_redacted_in_diagnostics(caplog: pytest.LogCaptureFixture, api_key: str) -> None:
    """A query that contains the key must not leak it through errors, logs, failures, or tool-frame locals."""
    query = f"why does {api_key} fail"
    respond = respond_per_query(
        query,
        respond_json({"error": {"code": "unauthorized", "message": "Unauthorized"}}, status=401),
        respond_sources([WEB_SOURCE]),
    )
    with caplog.at_level(logging.DEBUG):
        out, _ = await search(respond, queries=[query, "good"], api_key=api_key)
        error = await search_error(respond, queries=[query], api_key=api_key)

    assert out.failures == [SpecstreamSearchFailure(query="why does [redacted] fail", error=str(error))]
    assert str(error) == "SpecStream search failed for 'why does [redacted] fail': 401 unauthorized: Unauthorized"
    assert_no_key(caplog.text, repr(out.failures), str(error), repr(error), repr(error.__context__))
    assert_no_key(*frame_locals(error, is_tool_frame, skip_names=sorted(CALLER_INPUT_LOCALS)))


def test_key_kept_out_of_tool_and_config_state() -> None:
    tool = SpecstreamSearchTool(config=SpecstreamSearchToolConfig(api_key=SecretStr(DUMMY_KEY)))
    assert_no_key(
        repr(tool),
        repr(tool.config),
        str(tool.config),
        tool.config.model_dump_json(),
        repr(vars(tool)),
        repr(vars(copy.copy(tool))),
    )
    assert tool._api_key.get_secret_value() == DUMMY_KEY


def test_redaction_does_not_mangle_ordinary_text() -> None:
    """A one-letter key is not redacted, and short ss_ words that are not keys are left alone."""
    tool = SpecstreamSearchTool(config=SpecstreamSearchToolConfig(api_key=SecretStr("a")))
    text = "SpecStream search failed for 'ss_live_demo' and 'ss_test_0123abcd'"
    assert tool._redact(text) == text


@pytest.mark.parametrize(
    "key",
    [
        "ss_live_0123456789abcdefABCDEF0123456789abcdef",
        "ss_test_0123456789abcdefABCDEF0123456789abcdef",
        "specstream_live_0123456789abcdef_-xyz",
        "specstream_test_0123456789abcdef_-xyz",
    ],
)
def test_specstream_key_formats_are_redacted_even_when_not_configured(key: str) -> None:
    tool = SpecstreamSearchTool(config=SpecstreamSearchToolConfig(api_key=SecretStr("another-configured-key")))
    assert tool._redact(f"bad key {key}.") == "bad key [redacted]."


@pytest.mark.asyncio
async def test_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPECSTREAM_API_KEY", raising=False)
    tool = SpecstreamSearchTool(config=SpecstreamSearchToolConfig())

    with pytest.raises(ValueError, match="SPECSTREAM_API_KEY"):
        await tool.run_async(SpecstreamSearchToolInputSchema(queries=["q"]))


def test_api_key_falls_back_to_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPECSTREAM_API_KEY", "env-key")
    tool = SpecstreamSearchTool(config=SpecstreamSearchToolConfig())
    assert tool._api_key.get_secret_value() == "env-key"


def test_run_invokes_run_async() -> None:
    tool = SpecstreamSearchTool(config=SpecstreamSearchToolConfig(api_key=SecretStr("k")))
    sentinel = SpecstreamSearchToolOutputSchema(results=[])
    with patch.object(SpecstreamSearchTool, "run_async", AsyncMock(return_value=sentinel)):
        assert tool.run(SpecstreamSearchToolInputSchema(queries=["x"])) is sentinel


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
