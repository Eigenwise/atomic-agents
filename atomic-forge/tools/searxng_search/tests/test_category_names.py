"""Exercise the API category names in every distributed SearXNG tool."""

import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest


TOOL_PATHS = [
    "atomic-forge/tools/searxng_search/tool/searxng_search.py",
    "atomic-examples/deep-research/deep_research/tools/searxng_search.py",
    "atomic-examples/orchestration-agent/orchestration_agent/tools/searxng_search.py",
    "atomic-examples/web-search-agent/web_search_agent/tools/searxng_search.py",
]


@pytest.fixture(params=TOOL_PATHS, ids=["forge", "deep-research", "orchestration", "web-search"])
def tool_module(request):
    root = Path(__file__).resolve().parents[4]
    spec = importlib.util.spec_from_file_location("searxng_category_test", root / request.param)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
@pytest.mark.parametrize("category", ["general", "news", "social_media", None])
async def test_category_round_trip(tool_module, category, monkeypatch):
    expected_category = "social media" if category == "social_media" else category
    # SearXNG uses a space in the category of social results, including replies
    # without a category filter. General/news categories retain their spelling.
    response_category = expected_category or "social media"
    response = AsyncMock()
    response.status = 200
    response.json.return_value = {
        "results": [
            {
                "title": "Matching result",
                "url": "https://example.com/result",
                "content": "Search result content",
                "category": response_category,
            }
        ]
    }
    session = MagicMock()
    session.get.return_value.__aenter__.return_value = response
    session_factory = MagicMock()
    session_factory.return_value.__aenter__.return_value = session
    monkeypatch.setattr(tool_module.aiohttp, "ClientSession", session_factory)

    tool = tool_module.SearXNGSearchTool(tool_module.SearXNGSearchToolConfig(base_url="https://search.example"))
    inputs = tool_module.SearXNGSearchToolInputSchema(queries=["community discussion"], category=category)
    output = await tool.run_async(inputs)

    sent = session.get.call_args.kwargs["params"]
    assert sent["q"] == "community discussion"
    assert sent["format"] == "json"
    if expected_category is None:
        assert "categories" not in sent
    else:
        assert sent["categories"] == expected_category
    assert len(output.results) == 1
    assert output.results[0].url == "https://example.com/result"
    assert output.results[0].query == inputs.queries[0]
    assert output.category == inputs.category
