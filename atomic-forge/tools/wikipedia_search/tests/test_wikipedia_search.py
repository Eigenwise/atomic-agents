from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tool.wikipedia_search import (
    WikipediaSearchTool,
    WikipediaSearchToolConfig,
    WikipediaSearchToolInputSchema,
    WikipediaSearchToolOutputSchema,
)


@pytest.fixture
def tool():
    return WikipediaSearchTool(config=WikipediaSearchToolConfig())


def test_input_schema_rejects_invalid_max_results():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        WikipediaSearchToolInputSchema(queries=["x"], max_results_per_query=0)
    with pytest.raises(ValidationError):
        WikipediaSearchToolInputSchema(queries=["x"], max_results_per_query=21)


@pytest.mark.asyncio
async def test_run_async_returns_articles(tool):
    """Mock all three coroutines and assert the output schema is built correctly."""

    async def fake_search_titles(self, session, query, language, limit):
        return ["Atomic Agents", "Atomic Theory"]

    async def fake_fetch_summary(self, session, title, language):
        return {
            "title": title,
            "extract": f"Summary of {title}.",
            "description": f"desc {title}",
            "content_urls": {"desktop": {"page": f"https://{language}.wikipedia.org/wiki/{title.replace(' ', '_')}"}},
            "thumbnail": {"source": "https://example.com/thumb.png"},
        }

    async def fake_fetch_full_extract(self, session, title, language):
        return f"FULL: {title}"

    with (
        patch.object(WikipediaSearchTool, "_search_titles", fake_search_titles),
        patch.object(WikipediaSearchTool, "_fetch_summary", fake_fetch_summary),
        patch.object(WikipediaSearchTool, "_fetch_full_extract", fake_fetch_full_extract),
    ):
        result = await tool.run_async(
            WikipediaSearchToolInputSchema(queries=["Atomic"], full_text=True, max_results_per_query=2)
        )

    assert isinstance(result, WikipediaSearchToolOutputSchema)
    assert len(result.results) == 2
    assert result.results[0].title == "Atomic Agents"
    assert result.results[0].full_text == "FULL: Atomic Agents"  # only top result has full_text
    assert result.results[1].full_text is None
    assert result.results[0].page_url.endswith("Atomic_Agents")


@pytest.mark.asyncio
async def test_run_async_skips_failing_query(tool):
    async def fake_search_titles(self, session, query, language, limit):
        if query == "good":
            return ["Page A"]
        raise Exception("boom")

    async def fake_fetch_summary(self, session, title, language):
        return {"title": title, "extract": "ok", "content_urls": {"desktop": {"page": "https://x"}}}

    with (
        patch.object(WikipediaSearchTool, "_search_titles", fake_search_titles),
        patch.object(WikipediaSearchTool, "_fetch_summary", fake_fetch_summary),
    ):
        out = await tool.run_async(WikipediaSearchToolInputSchema(queries=["good", "bad"]))

    assert len(out.results) == 1
    assert out.results[0].query == "good"


@pytest.mark.asyncio
async def test_run_async_skips_404_summaries(tool):
    async def fake_search_titles(self, session, query, language, limit):
        return ["Real Page", "Missing Page"]

    async def fake_fetch_summary(self, session, title, language):
        if title == "Missing Page":
            return None
        return {"title": title, "extract": "ok", "content_urls": {"desktop": {"page": "https://x"}}}

    with (
        patch.object(WikipediaSearchTool, "_search_titles", fake_search_titles),
        patch.object(WikipediaSearchTool, "_fetch_summary", fake_fetch_summary),
    ):
        out = await tool.run_async(WikipediaSearchToolInputSchema(queries=["q"], max_results_per_query=2))

    assert len(out.results) == 1
    assert out.results[0].title == "Real Page"


@pytest.mark.asyncio
@pytest.mark.parametrize("summary_failure", [None, RuntimeError("unavailable")])
@pytest.mark.parametrize("full_text", [True, False])
async def test_full_text_follows_first_returned_article(tool, summary_failure, full_text):
    summaries = [
        summary_failure,
        {"title": "Available Page", "extract": "Summary"},
        {"title": "Another Page", "extract": "Another summary"},
    ]
    with (
        patch.object(tool, "_search_titles", AsyncMock(return_value=["Missing Page", "Available Page", "Another Page"])),
        patch.object(tool, "_fetch_summary", AsyncMock(side_effect=summaries)),
        patch.object(tool, "_fetch_full_extract", AsyncMock(return_value="Full article")) as extract,
    ):
        result = await tool._process_query(MagicMock(), "topic", "en", 3, full_text)

    assert [article.title for article in result] == ["Available Page", "Another Page"]
    assert result[0].full_text == ("Full article" if full_text else None)
    assert result[1].full_text is None
    if full_text:
        extract.assert_awaited_once()
        assert extract.await_args.args[1:] == ("Available Page", "en")
    else:
        extract.assert_not_awaited()


def test_run_invokes_run_async_in_thread(tool):
    """run() should drive run_async in a worker thread and return its result."""
    sentinel = WikipediaSearchToolOutputSchema(results=[])
    fake_run_async = AsyncMock(return_value=sentinel)
    with patch.object(WikipediaSearchTool, "run_async", fake_run_async):
        out = tool.run(WikipediaSearchToolInputSchema(queries=["x"]))
    assert out is sentinel


@pytest.mark.asyncio
async def test_search_titles_raises_on_non_200(tool):
    """Direct test of _search_titles error path."""

    class Resp:
        status = 500
        reason = "boom"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def json(self):
            return {}

    session = MagicMock()
    session.get = MagicMock(return_value=Resp())

    with pytest.raises(Exception, match="Wikipedia search failed"):
        await tool._search_titles(session, "q", "en", 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
