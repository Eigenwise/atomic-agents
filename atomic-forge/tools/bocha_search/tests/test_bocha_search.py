import os
import sys
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from types import SimpleNamespace
import asyncio

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tool.bocha_search import (  # noqa: E402
    BoChaSearchTool,
    BoChaSearchToolInputSchema,
    BoChaSearchToolConfig,
    BoChaSearchToolOutputSchema,
)


@pytest.fixture
def mock_aiohttp_session():
    """Mock aiohttp.ClientSession for all tests"""
    with patch("tool.bocha_search.aiohttp.ClientSession") as mock_session_class:
        mock_session = MagicMock()

        # Default mock response
        mock_response = SimpleNamespace(
            status=200,
            json=AsyncMock(
                return_value={
                    "data": {
                        "webPages": {
                            "value": [
                                {
                                    "name": "Mock Title",
                                    "url": "https://example.com",
                                    "snippet": "Mock content",
                                }
                            ]
                        }
                    }
                }
            ),
        )
        mock_session.post.return_value.__aenter__.return_value = mock_response
        mock_session_class.return_value.__aenter__.return_value = mock_session
        yield mock_session


@pytest.mark.asyncio
async def test_bocha_search_tool_missing_fields(mock_aiohttp_session):
    """Test that results missing required fields are skipped"""
    mock_api_key = "KEY"
    mock_query = "test query"
    mock_response_data = {
        "data": {
            "webPages": {
                "value": [
                    {"name": "Missing URL", "snippet": "Content"},
                    {"url": "https://example.com", "snippet": "Content"},
                    {"name": "Valid Result", "url": "https://example.com/valid", "snippet": "Valid content"},
                ]
            }
        }
    }

    mock_aiohttp_session.post.return_value.__aenter__.return_value.json.return_value = mock_response_data

    tool = BoChaSearchTool(BoChaSearchToolConfig(api_key=mock_api_key))
    input_schema = BoChaSearchToolInputSchema(queries=[mock_query])

    result: BoChaSearchToolOutputSchema = await tool.run_async(input_schema)

    assert len(result.results) == 1
    assert result.results[0].name == "Valid Result"


def test_bocha_search_tool_sync_run(mock_aiohttp_session):
    """Test the synchronous run method"""
    mock_api_key = "KEY"
    mock_query = "sync query"
    mock_response_data = {
        "data": {"webPages": {"value": [{"name": "Sync Result", "url": "https://example.com", "snippet": "Sync content"}]}}
    }

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = mock_response_data
    mock_aiohttp_session.post.return_value.__aenter__.return_value = mock_response

    tool = BoChaSearchTool(BoChaSearchToolConfig(api_key=mock_api_key))
    input_schema = BoChaSearchToolInputSchema(queries=[mock_query])
    result = tool.run(input_schema)

    assert len(result.results) == 1
    assert result.results[0].name == "Sync Result"


@pytest.mark.asyncio
async def test_bocha_search_tool_count(mock_aiohttp_session):
    """Test count parameter"""
    mock_api_key = "KEY"
    mock_query = "max results query"

    mock_response_data = {
        "data": {
            "webPages": {
                "value": [
                    {"name": f"Result {i}", "url": f"https://example.com/{i}", "snippet": f"Content {i}"} for i in range(10)
                ]
            }
        }
    }

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = mock_response_data
    mock_aiohttp_session.post.return_value.__aenter__.return_value = mock_response

    tool = BoChaSearchTool(BoChaSearchToolConfig(api_key=mock_api_key, count=10))
    input_schema = BoChaSearchToolInputSchema(queries=[mock_query])

    # Test limiting to 5 results
    result = await tool.run_async(input_schema, count=5)
    assert len(result.results) == 5

    # Test default count from config
    mock_aiohttp_session.post.reset_mock()
    result2 = await tool.run_async(input_schema)
    assert len(result2.results) == 10


@pytest.mark.asyncio
async def test_bocha_search_tool_no_results(mock_aiohttp_session):
    """Test behavior when no results returned"""
    mock_api_key = "KEY"
    mock_query = "empty query"

    mock_response_data = {"data": {"webPages": {"value": []}}}
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.json.return_value = mock_response_data
    mock_aiohttp_session.post.return_value.__aenter__.return_value = mock_response

    tool = BoChaSearchTool(BoChaSearchToolConfig(api_key=mock_api_key))
    input_schema = BoChaSearchToolInputSchema(queries=[mock_query])
    result = await tool.run_async(input_schema)

    assert len(result.results) == 0


@pytest.mark.asyncio
async def test_bocha_search_tool_http_error_handled(mock_aiohttp_session, caplog):
    """Test that HTTP errors are logged and skipped, not raised"""
    mock_api_key = "KEY"
    mock_query = "error query"

    mock_response = AsyncMock()
    mock_response.status = 500
    mock_response.reason = "Internal Server Error"
    mock_response.text = AsyncMock(return_value="Server error")
    mock_aiohttp_session.post.return_value.__aenter__.return_value = mock_response

    tool = BoChaSearchTool(BoChaSearchToolConfig(api_key=mock_api_key))
    input_schema = BoChaSearchToolInputSchema(queries=[mock_query])

    with caplog.at_level("WARNING"):
        result = await tool.run_async(input_schema)

    # The result should be empty because the query failed
    assert len(result.results) == 0
    # Check that a warning was logged
    assert any("Query 'error query' failed" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_bocha_search_tool_empty_queries(mock_aiohttp_session):
    """Test behavior when queries=[]"""
    mock_api_key = "KEY"
    tool = BoChaSearchTool(BoChaSearchToolConfig(api_key=mock_api_key))
    input_schema = BoChaSearchToolInputSchema(queries=[])
    result = await tool.run_async(input_schema)
    assert len(result.results) == 0


@pytest.mark.asyncio
async def test_bocha_search_tool_concurrent_queries(mock_aiohttp_session):
    """Test multiple concurrent queries"""
    mock_api_key = "KEY"
    tool = BoChaSearchTool(BoChaSearchToolConfig(api_key=mock_api_key))
    queries = ["query1", "query2", "query3"]
    input_schemas = [BoChaSearchToolInputSchema(queries=[q]) for q in queries]

    results = await asyncio.gather(*(tool.run_async(schema) for schema in input_schemas))
    assert all(len(r.results) == 1 for r in results)
    assert [r.results[0].name for r in results] == ["Mock Title"] * 3


@pytest.mark.asyncio
async def test_bocha_search_tool_config_params_real_case_cn(mock_aiohttp_session):
    """Test that config params freshness/include/exclude are included in request payload with Chinese websites"""
    mock_api_key = "mock-api-key"

    # Tool configuration with domestic websites
    tool = BoChaSearchTool(
        BoChaSearchToolConfig(
            api_key=mock_api_key,
            freshness="oneMonth",  # Search results from the past month
            include="baidu.com|sina.com.cn",  # Only include these domestic websites
            exclude="ads.example.com|spam.cn",  # Exclude ad/spam websites
        )
    )

    # Use a realistic Chinese query
    input_schema = BoChaSearchToolInputSchema(queries=["人工智能发展趋势"])

    await tool.run_async(input_schema)

    # Retrieve the payload sent in the request
    payload_sent = mock_aiohttp_session.post.call_args[1]["json"]

    # Assert that the configuration parameters are correctly included
    assert payload_sent["freshness"] == "oneMonth"
    assert "baidu.com" in payload_sent["include"]
    assert "sina.com.cn" in payload_sent["include"]
    assert "ads.example.com" in payload_sent["exclude"]
    assert "spam.cn" in payload_sent["exclude"]

    # Assert that the query keyword is included in the payload
    assert "人工智能发展趋势" in payload_sent["query"] or "queries" in payload_sent


@pytest.mark.asyncio
async def test_bocha_search_tool_malformed_response(mock_aiohttp_session):
    """Test API response missing 'data' or 'webPages' keys"""
    mock_api_key = "KEY"
    tool = BoChaSearchTool(BoChaSearchToolConfig(api_key=mock_api_key))
    input_schema = BoChaSearchToolInputSchema(queries=["test query"])

    # response completely missing 'data'
    mock_response_data = {"unexpected": "structure"}
    mock_aiohttp_session.post.return_value.__aenter__.return_value.json.return_value = mock_response_data

    result = await tool.run_async(input_schema)
    assert len(result.results) == 0

    # response missing 'webPages'
    mock_response_data2 = {"data": {"otherKey": []}}
    mock_aiohttp_session.post.return_value.__aenter__.return_value.json.return_value = mock_response_data2

    result2 = await tool.run_async(input_schema)
    assert len(result2.results) == 0


if __name__ == "__main__":
    pytest.main([__file__])


@pytest.mark.parametrize(
    "freshness",
    [
        "noLimit",
        "oneDay",
        "oneWeek",
        "oneMonth",
        "oneYear",
        "2026-11-07",
        "2024-02-29",
        "2026-11-01..2026-11-07",
        "2026-11-07..2026-11-07",
    ],
)
def test_freshness_accepts_presets_and_dates(freshness):
    assert BoChaSearchToolConfig(freshness=freshness).freshness == freshness


@pytest.mark.parametrize(
    "freshness",
    [
        "YYYY-MM-DD",
        "YYYY-MM-DD..YYYY-MM-DD",
        "2026-02-29",
        "2026-13-01",
        "2026-11-07..2026-11-01",
        "2026-11-01..2026-02-30",
        "20261107",
        "2026-1-7",
        "2026-11-01..",
        "",
        "someday",
    ],
)
def test_freshness_rejects_invalid_dates(freshness):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        BoChaSearchToolConfig(freshness=freshness)


@pytest.mark.asyncio
@pytest.mark.parametrize("freshness", ["2026-11-07", "2026-11-01..2026-11-07"])
async def test_freshness_date_is_sent_unchanged(mock_aiohttp_session, freshness):
    tool = BoChaSearchTool(BoChaSearchToolConfig(api_key="mock-key", freshness=freshness))
    await tool.run_async(BoChaSearchToolInputSchema(queries=["test query"]))
    assert mock_aiohttp_session.post.call_args.kwargs["json"]["freshness"] == freshness
