import asyncio
import os
import sys
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from aiohttp import ClientSession

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tool.searxng_search import (  # noqa: E402
    SearXNGSearchTool,
    SearXNGSearchToolInputSchema,
    SearXNGSearchToolOutputSchema,
    SearXNGSearchToolConfig,
)


@pytest.fixture
def mock_aiohttp_session():
    # Patch the ClientSession in the module where it's used
    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = MagicMock(spec=ClientSession)
        # Mock the async context manager __aenter__ to return the mock session
        mock_session_class.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        yield mock_session


@pytest.mark.asyncio
async def test_searxng_search_tool_with_category(mock_aiohttp_session):
    mock_searxng_url = "https://searxng.example.com"
    mock_query = "test query with category"
    mock_response_data = {
        "results": [
            {
                "title": "Test Result with Category",
                "url": "https://example.com/test-category",
                "content": "This is a test result content with category.",
                "category": "news",
                "query": mock_query,
            }
        ]
    }

    # Create a mock response object
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = mock_response_data

    # Configure the mock to return the mock response
    mock_aiohttp_session.get.return_value.__aenter__.return_value = mock_response

    # Initialize the tool
    searxng_tool = SearXNGSearchTool(SearXNGSearchToolConfig(base_url=mock_searxng_url))
    input_schema = SearXNGSearchToolInputSchema(queries=[mock_query], category="news")

    # Run the tool
    result = await searxng_tool.run_async(input_schema)

    # Assertions
    assert isinstance(result, SearXNGSearchToolOutputSchema)
    assert len(result.results) == 1
    assert result.results[0].title == "Test Result with Category"
    assert result.results[0].url == "https://example.com/test-category"
    assert result.results[0].content == "This is a test result content with category."
    assert result.category == "news"


@pytest.mark.asyncio
async def test_searxng_search_tool_missing_fields(mock_aiohttp_session):
    mock_searxng_url = "https://searxng.example.com"
    mock_query = "query with missing fields"
    mock_response_data = {
        "results": [
            {"title": "Result Missing Content", "url": "https://example.com/1", "query": mock_query},
            {"content": "Result Missing Title", "url": "https://example.com/2", "query": mock_query},
            {"title": "Result Missing URL", "content": "Some content", "query": mock_query},
            {"title": "Result Missing Query", "url": "https://example.com/4", "content": "Some content"},
            {"title": "Valid Result", "url": "https://example.com/5", "content": "Valid content", "query": mock_query},
        ]
    }

    # Create a mock response object
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = mock_response_data

    # Configure the mock
    mock_aiohttp_session.get.return_value.__aenter__.return_value = mock_response

    # Initialize the tool
    searxng_tool = SearXNGSearchTool(SearXNGSearchToolConfig(base_url=mock_searxng_url))
    input_schema = SearXNGSearchToolInputSchema(queries=[mock_query], category=None)

    # Run the tool
    result = await searxng_tool.run_async(input_schema)

    # The last two results should be included
    assert len(result.results) == 2
    assert result.results[0].title == "Result Missing Query"
    assert result.results[1].title == "Valid Result"


@pytest.mark.asyncio
async def test_searxng_search_tool_with_metadata_and_published_date(mock_aiohttp_session):
    mock_searxng_url = "https://searxng.example.com"
    mock_query = "query with dates"
    mock_response_data = {
        "results": [
            {
                "title": "Result with Metadata",
                "url": "https://example.com/metadata",
                "content": "Content with metadata",
                "query": mock_query,
                "metadata": "2021-01-01",
            },
            {
                "title": "Result with Published Date",
                "url": "https://example.com/published-date",
                "content": "Content with published date",
                "query": mock_query,
                "publishedDate": "2022-01-01",
            },
            {
                "title": "Result without Dates",
                "url": "https://example.com/no-dates",
                "content": "Content without dates",
                "query": mock_query,
            },
        ]
    }

    # Create a mock response object
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = mock_response_data

    # Configure the mock
    mock_aiohttp_session.get.return_value.__aenter__.return_value = mock_response

    # Initialize the tool
    searxng_tool = SearXNGSearchTool(SearXNGSearchToolConfig(base_url=mock_searxng_url))
    input_schema = SearXNGSearchToolInputSchema(queries=[mock_query], category=None)

    # Run the tool
    result = await searxng_tool.run_async(input_schema)

    # Assertions
    assert len(result.results) == 3
    assert result.results[0].title == "Result with Metadata - (Published 2021-01-01)"
    assert result.results[1].title == "Result with Published Date - (Published 2022-01-01)"
    assert result.results[2].title == "Result without Dates"


def test_searxng_search_tool_sync_run_method(mock_aiohttp_session):
    mock_searxng_url = "https://searxng.example.com"
    mock_query = "sync query"
    mock_response_data = {
        "results": [
            {
                "title": "Sync Test Result",
                "url": "https://example.com/sync",
                "content": "Sync content",
                "query": mock_query,
            }
        ]
    }

    # Create a mock response object
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = mock_response_data

    # Configure the mock
    mock_aiohttp_session.get.return_value.__aenter__.return_value = mock_response

    # Initialize the tool
    searxng_tool = SearXNGSearchTool(SearXNGSearchToolConfig(base_url=mock_searxng_url))
    input_schema = SearXNGSearchToolInputSchema(queries=[mock_query], category=None)

    # Run the tool synchronously
    result = searxng_tool.run(input_schema)

    # Assertions
    assert len(result.results) == 1
    assert result.results[0].title == "Sync Test Result"


@pytest.mark.asyncio
async def test_searxng_search_tool_with_max_results(mock_aiohttp_session):
    mock_searxng_url = "https://searxng.example.com"
    mock_query = "test query with max results"
    mock_response_data = {
        "results": [
            {"title": f"Result {i}", "url": f"https://example.com/{i}", "content": f"Content {i}", "query": mock_query}
            for i in range(10)
        ]
    }

    # Create a mock response object
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = mock_response_data

    # Configure the mock
    mock_aiohttp_session.get.return_value.__aenter__.return_value = mock_response

    # Initialize the tool
    searxng_tool = SearXNGSearchTool(SearXNGSearchToolConfig(base_url=mock_searxng_url, max_results=10))
    input_schema = SearXNGSearchToolInputSchema(queries=[mock_query], category=None)

    # Run the tool with max_results=5
    result = await searxng_tool.run_async(input_schema, max_results=5)

    # Assertions
    assert len(result.results) == 5


@pytest.mark.asyncio
async def test_searxng_search_tool_no_results(mock_aiohttp_session):
    # Existing test case remains the same
    mock_searxng_url = "https://searxng.example.com"
    mock_query = "no results query"
    mock_response_data = {"results": []}

    # Create a mock response object
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = mock_response_data

    # Configure the mock
    mock_aiohttp_session.get.return_value.__aenter__.return_value = mock_response

    # Initialize the tool
    searxng_tool = SearXNGSearchTool(SearXNGSearchToolConfig(base_url=mock_searxng_url))
    input_schema = SearXNGSearchToolInputSchema(queries=[mock_query], category=None)

    # Run the tool
    result = await searxng_tool.run_async(input_schema)

    # Assertions
    assert len(result.results) == 0


@pytest.mark.asyncio
async def test_searxng_search_tool_error(mock_aiohttp_session):
    # Existing test case remains the same
    mock_searxng_url = "https://searxng.example.com"
    mock_query = "error query"

    # Create a mock response object with a 500 status
    mock_response = AsyncMock()
    mock_response.status = 500
    mock_response.reason = "Internal Server Error"

    # Configure the mock
    mock_aiohttp_session.get.return_value.__aenter__.return_value = mock_response

    # Initialize the tool
    searxng_tool = SearXNGSearchTool(SearXNGSearchToolConfig(base_url=mock_searxng_url))
    input_schema = SearXNGSearchToolInputSchema(queries=[mock_query], category=None)

    # Run the tool and expect an exception
    with pytest.raises(Exception) as excinfo:
        await searxng_tool.run_async(input_schema)

    # Assertion on the exception message
    assert "Failed to fetch search results" in str(excinfo.value)


if __name__ == "__main__":
    pytest.main([__file__])


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_mode", ["request_error", "request_cancelled", "caller_cancelled"])
async def test_pending_requests_finish_before_session_closes(failure_mode):
    started = asyncio.Event()
    cleaned = asyncio.Event()
    session_states = []
    requests = []
    failure = RuntimeError("search failed") if failure_mode == "request_error" else asyncio.CancelledError()

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            session_states.append(cleaned.is_set())

    async def fetch(session, query, setting):
        requests.append(asyncio.current_task())
        if query == "slow":
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                # Response cleanup may itself need an event-loop turn.
                await asyncio.sleep(0)
                cleaned.set()
        await started.wait()
        if failure_mode == "caller_cancelled":
            await asyncio.Event().wait()
        raise failure

    tool = SearXNGSearchTool(SearXNGSearchToolConfig(base_url="https://searxng.example.com"))
    with (
        patch("aiohttp.ClientSession", return_value=Session()),
        patch.object(tool, "_fetch_search_results", side_effect=fetch),
    ):
        operation = asyncio.create_task(tool.run_async(SearXNGSearchToolInputSchema(queries=["slow", "failure"])))
        try:
            await asyncio.wait_for(started.wait(), timeout=1)
            if failure_mode == "caller_cancelled":
                operation.cancel()
            with pytest.raises(type(failure)) as caught:
                await asyncio.wait_for(operation, timeout=1)
            if failure_mode != "caller_cancelled":
                assert caught.value is failure
            assert session_states == [True]
            assert all(request.done() for request in requests)
        finally:
            operation.cancel()
            for request in requests:
                request.cancel()
            await asyncio.gather(operation, *requests, return_exceptions=True)
