"""Exercise failure reporting through the real HTTP client and tool schema."""

import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

from tool.hackernews_search import HackerNewsSearchTool, HackerNewsSearchToolConfig, HackerNewsSearchToolInputSchema


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["http", "missing", "null", "array", "invalid_json"])
async def test_failed_query_is_preserved_alongside_successful_results(failure):
    async def respond(request):
        if request.query["query"] == "good":
            return web.json_response({"hits": [{"objectID": "123", "title": "Result"}]})
        if failure == "http":
            return web.json_response({"message": "rate limited"}, status=429)
        if failure == "invalid_json":
            return web.Response(text="not JSON", content_type="application/json")
        return web.json_response({"missing": {}, "null": {"hits": None}, "array": []}[failure])

    app = web.Application()
    app.router.add_get("/search", respond)
    async with TestServer(app) as server:
        tool = HackerNewsSearchTool(HackerNewsSearchToolConfig(base_url=str(server.make_url("/")).rstrip("/")))
        output = await tool.run_async(HackerNewsSearchToolInputSchema(queries=["bad", "good"]))

    assert [(item.query, item.object_id) for item in output.results] == [("good", "123")]
    assert len(output.failures) == 1
    assert output.failures[0].query == "bad"
    assert output.failures[0].error
    assert output.model_dump(mode="json")["failures"][0]["query"] == "bad"


@pytest.mark.asyncio
async def test_all_failed_queries_are_not_reported_as_an_empty_success():
    async def respond(request):
        return web.Response(status=503)

    app = web.Application()
    app.router.add_get("/search", respond)
    async with TestServer(app) as server:
        tool = HackerNewsSearchTool(HackerNewsSearchToolConfig(base_url=str(server.make_url("/")).rstrip("/")))
        output = await tool.run_async(HackerNewsSearchToolInputSchema(queries=["first", "second"]))

    assert output.results == []
    assert [failure.query for failure in output.failures] == ["first", "second"]
    assert all("503" in failure.error for failure in output.failures)


@pytest.mark.asyncio
async def test_successful_empty_queries_have_no_failures():
    async def respond(request):
        return web.json_response({"hits": []})

    app = web.Application()
    app.router.add_get("/search", respond)
    async with TestServer(app) as server:
        tool = HackerNewsSearchTool(HackerNewsSearchToolConfig(base_url=str(server.make_url("/")).rstrip("/")))
        output = await tool.run_async(HackerNewsSearchToolInputSchema(queries=["no matches"]))
        empty = await tool.run_async(HackerNewsSearchToolInputSchema(queries=[]))

    assert output.results == []
    assert output.failures == []
    assert empty.results == []
    assert empty.failures == []


@pytest.mark.asyncio
async def test_transport_failure_is_reported_by_sync_wrapper():
    # Reserve then close a local listening socket; no external host is contacted.
    server = await asyncio.start_server(lambda reader, writer: writer.close(), "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    server.close()
    await server.wait_closed()
    tool = HackerNewsSearchTool(HackerNewsSearchToolConfig(base_url=f"http://127.0.0.1:{port}"))
    output = tool.run(HackerNewsSearchToolInputSchema(queries=["offline"]))

    assert output.results == []
    assert [failure.query for failure in output.failures] == ["offline"]
    assert output.failures[0].error
