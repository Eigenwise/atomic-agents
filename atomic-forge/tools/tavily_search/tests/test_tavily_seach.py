import asyncio
import os
import sys
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tool.tavily_search import (  # noqa: E402
    TavilySearchTool,
    TavilySearchToolInputSchema,
    TavilySearchToolConfig,
)


@pytest.fixture
def mock_aiohttp_session():
    # Patch the ClientSession in the module where it's used
    with patch("tool.tavily_search.aiohttp.ClientSession") as mock_session_class:
        # Create a mock session
        mock_session = MagicMock()

        # Mock `post` method with a response
        mock_response = SimpleNamespace(
            status=200,
            json=AsyncMock(
                return_value={
                    "results": [
                        {"title": "Mock Title", "url": "https://example.com", "content": "Mock Content", "score": 1.0},
                    ]
                }
            ),
        )
        mock_session.post.return_value.__aenter__.return_value = mock_response

        # Mock the session returned by `ClientSession`
        mock_session_class.return_value.__aenter__.return_value = mock_session
        yield mock_session


@pytest.mark.asyncio
async def test_tavily_search_tool_missing_fields(mock_aiohttp_session):
    mock_api_key = "KEY"
    mock_query = "query with missing fields"
    mock_response_data = {
        "results": [
            {"title": "Result Missing Content", "url": "https://example.com/1", "score": 0.8},
            {"content": "Result Missing Title", "url": "https://example.com/2", "score": 0.8},
            {"title": "Result Missing URL", "content": "Some content", "score": 0.8},
            {"title": "Result Missing Score", "url": "https://example.com/4", "content": "Some content"},
            {"title": "Result Missing Query", "url": "https://example.com/4", "content": "Some content", "score": 0.7},
            {"title": "Valid Result", "url": "https://example.com/2", "content": "Valid content", "score": 1.0},
        ]
    }

    # Set up the mocked response for post
    mock_aiohttp_session.post.return_value.__aenter__.return_value.json.return_value = mock_response_data

    # Initialize the tool
    tavily_tool = TavilySearchTool(TavilySearchToolConfig(api_key=mock_api_key))
    input_schema = TavilySearchToolInputSchema(queries=[mock_query])

    # Run the tool
    result = await tavily_tool.run_async(input_schema)

    # The last two results should be included
    assert len(result.results) == 2
    assert result.results[0].title == "Result Missing Query"
    assert result.results[1].title == "Valid Result"


def test_tavily_search_tool_sync_run_method(mock_aiohttp_session):
    mock_api_key = "KEY"
    mock_query = "sync query"
    mock_response_data = {
        "results": [
            {
                "title": "Sync Test Result",
                "url": "https://example.com/sync",
                "content": "Sync content",
                "query": mock_query,
                "score": 1.0,
            }
        ]
    }

    # Create a mock response object
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = mock_response_data

    # Configure the mock
    mock_aiohttp_session.post.return_value.__aenter__.return_value = mock_response

    # Initialize the tool
    tavily_tool = TavilySearchTool(TavilySearchToolConfig(api_key=mock_api_key))
    input_schema = TavilySearchToolInputSchema(queries=[mock_query])

    # Run the tool synchronously
    result = tavily_tool.run(input_schema)

    # Assertions
    assert len(result.results) == 1
    assert result.results[0].title == "Sync Test Result"


@pytest.mark.asyncio
async def test_tavily_search_tool_with_max_results(mock_aiohttp_session):
    mock_api_key = "KEY"
    mock_query = "test query with max results"
    mock_response_data = {
        "results": [
            {
                "title": f"Result {i}",
                "url": f"https://example.com/{i}",
                "content": f"Content {i}",
                "query": mock_query,
                "score": 0.5,
            }
            for i in range(10)
        ]
    }

    # Create a mock response object
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = mock_response_data

    # Configure the mock
    mock_aiohttp_session.post.return_value.__aenter__.return_value = mock_response

    # Initialize the tool
    tavily_tool = TavilySearchTool(TavilySearchToolConfig(api_key=mock_api_key, max_results=10))
    input_schema = TavilySearchToolInputSchema(queries=[mock_query])

    # Run the tool with max_results=5
    result = await tavily_tool.run_async(input_schema, max_results=5)

    # Assertions
    assert len(result.results) == 5

    # Verify the API was called with correct max_results parameter
    mock_aiohttp_session.post.assert_called_once()
    call_args = mock_aiohttp_session.post.call_args
    assert call_args[1]["json"]["max_results"] == 5

    # Test with default max_results from config
    mock_aiohttp_session.post.reset_mock()
    result2 = await tavily_tool.run_async(input_schema)
    assert len(result2.results) == 10  # Uses config default
    call_args2 = mock_aiohttp_session.post.call_args
    assert call_args2[1]["json"]["max_results"] == 10


@pytest.mark.asyncio
async def test_tavily_search_tool_no_results(mock_aiohttp_session):
    # Existing test case remains the same
    mock_api_key = "KEY"
    mock_query = "no results query"
    mock_response_data = {"results": []}

    # Create a mock response object
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = mock_response_data

    # Configure the mock
    mock_aiohttp_session.post.return_value.__aenter__.return_value = mock_response

    # Initialize the tool
    tavily_tool = TavilySearchTool(TavilySearchToolConfig(api_key=mock_api_key))
    input_schema = TavilySearchToolInputSchema(queries=[mock_query])

    # Run the tool
    result = await tavily_tool.run_async(input_schema)

    # Assertions
    assert len(result.results) == 0


@pytest.mark.asyncio
async def test_tavily_search_tool_error(mock_aiohttp_session):
    # Existing test case remains the same
    mock_api_key = "KEY"
    mock_query = "error query"

    # Create a mock response object with a 500 status
    mock_response = AsyncMock()
    mock_response.status = 500
    mock_response.reason = "Internal Server Error"

    # Configure the mock
    mock_aiohttp_session.post.return_value.__aenter__.return_value = mock_response

    # Initialize the tool
    tavily_tool = TavilySearchTool(TavilySearchToolConfig(api_key=mock_api_key))
    input_schema = TavilySearchToolInputSchema(queries=[mock_query])

    # Run the tool and expect an exception
    with pytest.raises(Exception) as excinfo:
        await tavily_tool.run_async(input_schema)

    # Assertion on the exception message
    assert "Failed to fetch search results" in str(excinfo.value)


@pytest.mark.asyncio
async def test_tavily_search_tool_include_answer(mock_aiohttp_session):
    mock_api_key = "KEY"
    mock_query = "test query with answer"
    mock_answer = "This is a test answer."
    mock_response_data = {
        "results": [
            {"title": "Result 1", "url": "https://example.com/1", "content": "Content 1", "score": 1.0},
            {"title": "Result 2", "url": "https://example.com/2", "content": "Content 2", "score": 0.8},
        ],
        "answer": mock_answer,
    }

    # Set up the mocked response for post
    mock_aiohttp_session.post.return_value.__aenter__.return_value.json.return_value = mock_response_data

    # Initialize the tool with `include_answer=True`
    tavily_tool = TavilySearchTool(TavilySearchToolConfig(api_key=mock_api_key))
    tavily_tool.include_answer = True  # Enable the include_answer feature
    input_schema = TavilySearchToolInputSchema(queries=[mock_query])

    # Run the tool
    result = await tavily_tool.run_async(input_schema)

    # Assertions
    assert len(result.results) == 2
    assert result.results[0].title == "Result 1"
    assert result.results[1].title == "Result 2"

    # Validate that the answer is included in each result
    assert result.results[0].answer == mock_answer, f"{result.results[0].answer}"
    assert result.results[1].answer == mock_answer, f"{result.results[1].answer}"


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

    tool = TavilySearchTool(TavilySearchToolConfig(api_key="KEY"))
    with (
        patch("aiohttp.ClientSession", return_value=Session()),
        patch.object(tool, "_fetch_search_results", side_effect=fetch),
    ):
        operation = asyncio.create_task(tool.run_async(TavilySearchToolInputSchema(queries=["slow", "failure"])))
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
