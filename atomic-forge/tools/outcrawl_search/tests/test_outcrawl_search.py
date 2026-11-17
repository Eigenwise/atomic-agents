import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tool.outcrawl_search import (  # noqa: E402
    OutcrawlSearchResultItem,
    OutcrawlSearchTool,
    OutcrawlSearchToolConfig,
    OutcrawlSearchToolInputSchema,
    OutcrawlSearchToolOutputSchema,
)


SEARCH_HIT = {
    "url": "https://www.python.org/",
    "title": "Welcome to Python.org",
    "snippet": "Python is a programming language that lets you work quickly.",
    "rank": 1,
}

OTHER_HIT = {
    "url": "https://docs.python.org/3/",
    "title": "Python 3 documentation",
    "snippet": "",
    "rank": 2,
}

SCRAPE_RESPONSE = {
    "url": "https://www.python.org/",
    "statusCode": 200,
    "markdown": "# Welcome to Python.org",
    "metadata": {"title": "Welcome to Python.org"},
}


def _response(status: int, payload: dict, reason: str = "OK", text: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        status=status,
        reason=reason,
        json=AsyncMock(return_value=payload),
        text=AsyncMock(return_value=text),
    )


def _mock_session(*responses: SimpleNamespace) -> MagicMock:
    session = MagicMock()
    contexts = []
    for response in responses:
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=response)
        ctx.__aexit__ = AsyncMock(return_value=False)
        contexts.append(ctx)
    session.post.side_effect = contexts
    return session


@pytest.fixture
def tool():
    return OutcrawlSearchTool(config=OutcrawlSearchToolConfig(api_key="test-key"))


def test_to_item():
    item = OutcrawlSearchTool._to_item(SEARCH_HIT, "python")
    assert isinstance(item, OutcrawlSearchResultItem)
    assert item.query == "python"
    assert item.title == "Welcome to Python.org"
    assert item.url == "https://www.python.org/"
    assert item.snippet.startswith("Python is a programming language")
    assert item.rank == 1
    assert item.content is None


def test_to_item_empty_snippet_is_none():
    item = OutcrawlSearchTool._to_item(OTHER_HIT, "python")
    assert item.snippet is None
    assert item.rank == 2


@pytest.mark.asyncio
async def test_fetch_builds_request(tool):
    session = _mock_session(_response(200, {"hits": [SEARCH_HIT], "usage": {"credits": "0.5"}}))

    items = await tool._fetch(session, "python", 5, False)

    assert len(items) == 1
    assert items[0].url == "https://www.python.org/"
    call_args = session.post.call_args
    assert call_args[0][0] == "https://api.outcrawl.ai/v1/search"
    assert call_args[1]["json"] == {"query": "python", "limit": 5}


@pytest.mark.asyncio
async def test_fetch_trims_to_max_results(tool):
    session = _mock_session(_response(200, {"hits": [SEARCH_HIT] * 5}))

    items = await tool._fetch(session, "python", 2, False)

    assert len(items) == 2


@pytest.mark.asyncio
async def test_fetch_skips_hits_without_title_or_url(tool):
    hits = [SEARCH_HIT, {"title": "no url"}, {"url": "https://example.com/no-title"}]
    session = _mock_session(_response(200, {"hits": hits}))

    items = await tool._fetch(session, "python", 10, False)

    assert [item.url for item in items] == ["https://www.python.org/"]


@pytest.mark.asyncio
async def test_fetch_handles_missing_hits(tool):
    session = _mock_session(_response(200, {"hits": None}))

    assert await tool._fetch(session, "python", 10, False) == []


@pytest.mark.asyncio
async def test_fetch_raises_on_http_error(tool):
    session = _mock_session(_response(401, {}, reason="Unauthorized", text='{"error":"invalid key"}'))

    with pytest.raises(Exception, match="Outcrawl search failed for 'python': 401 Unauthorized"):
        await tool._fetch(session, "python", 10, False)


@pytest.mark.asyncio
async def test_fetch_scrapes_content(tool):
    session = _mock_session(
        _response(200, {"hits": [SEARCH_HIT]}),
        _response(200, SCRAPE_RESPONSE),
    )

    items = await tool._fetch(session, "python", 5, True)

    assert items[0].content == "# Welcome to Python.org"
    scrape_call = session.post.call_args_list[1]
    assert scrape_call[0][0] == "https://api.outcrawl.ai/v1/scrape"
    assert scrape_call[1]["json"] == {"url": "https://www.python.org/", "formats": ["markdown"], "onlyMainContent": True}


@pytest.mark.asyncio
async def test_fetch_keeps_result_when_scrape_fails(tool):
    session = _mock_session(
        _response(200, {"hits": [SEARCH_HIT, OTHER_HIT]}),
        _response(402, {}, reason="Payment Required"),
        _response(200, {"markdown": "# Python 3 documentation"}),
    )

    items = await tool._fetch(session, "python", 5, True)

    assert len(items) == 2
    assert items[0].content is None
    assert items[1].content == "# Python 3 documentation"


@pytest.mark.asyncio
async def test_run_async_aggregates_results(tool):
    async def fake_fetch(self, session, query, max_results, scrape_content):
        return [OutcrawlSearchTool._to_item(SEARCH_HIT, query), OutcrawlSearchTool._to_item(OTHER_HIT, query)]

    with patch.object(OutcrawlSearchTool, "_fetch", fake_fetch):
        out = await tool.run_async(OutcrawlSearchToolInputSchema(queries=["q1", "q2"]))

    assert isinstance(out, OutcrawlSearchToolOutputSchema)
    assert len(out.results) == 4
    assert {r.query for r in out.results} == {"q1", "q2"}


@pytest.mark.asyncio
async def test_run_async_skips_failing_query(tool):
    async def fake_fetch(self, session, query, max_results, scrape_content):
        if query == "bad":
            raise Exception("rate limited")
        return [OutcrawlSearchTool._to_item(SEARCH_HIT, query)]

    with patch.object(OutcrawlSearchTool, "_fetch", fake_fetch):
        out = await tool.run_async(OutcrawlSearchToolInputSchema(queries=["bad", "good"]))

    assert len(out.results) == 1
    assert out.results[0].query == "good"


@pytest.mark.asyncio
async def test_run_async_sends_bearer_token(tool):
    captured = {}

    class FakeSession:
        def __init__(self, headers=None, timeout=None):
            captured["headers"] = headers

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    async def fake_fetch(self, session, query, max_results, scrape_content):
        return []

    with (
        patch("tool.outcrawl_search.aiohttp.ClientSession", FakeSession),
        patch.object(OutcrawlSearchTool, "_fetch", fake_fetch),
    ):
        await tool.run_async(OutcrawlSearchToolInputSchema(queries=["q"]))

    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert "User-Agent" in captured["headers"]


@pytest.mark.asyncio
async def test_run_async_requires_api_key(monkeypatch):
    monkeypatch.delenv("OUTCRAWL_API_KEY", raising=False)
    tool = OutcrawlSearchTool(config=OutcrawlSearchToolConfig())

    with pytest.raises(ValueError, match="OUTCRAWL_API_KEY"):
        await tool.run_async(OutcrawlSearchToolInputSchema(queries=["q"]))


def test_api_key_falls_back_to_environment(monkeypatch):
    monkeypatch.setenv("OUTCRAWL_API_KEY", "env-key")
    tool = OutcrawlSearchTool(config=OutcrawlSearchToolConfig())
    assert tool.api_key == "env-key"


def test_max_results_is_bounded():
    with pytest.raises(ValueError):
        OutcrawlSearchToolInputSchema(queries=["q"], max_results_per_query=21)


def test_run_invokes_run_async(tool):
    sentinel = OutcrawlSearchToolOutputSchema(results=[])
    with patch.object(OutcrawlSearchTool, "run_async", AsyncMock(return_value=sentinel)):
        out = tool.run(OutcrawlSearchToolInputSchema(queries=["x"]))
    assert out is sentinel


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
