"""Exercise query failure reporting through aiohttp and a local HTTP service."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tool.outcrawl_search import OutcrawlSearchTool, OutcrawlSearchToolConfig, OutcrawlSearchToolInputSchema


@pytest.fixture
def service():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            query = payload.get("query")
            if self.path == "/v1/scrape":
                status, body = 503, {"error": "page unavailable"}
            elif query.startswith("error-"):
                status, body = int(query.removeprefix("error-")), {"error": "provider request rejected"}
            elif query == "secret-error":
                status, body = 401, {"error": "key test-outcrawl-credential rejected"}
            elif query == "invalid-json":
                status, body = 200, None
            elif query == "empty":
                status, body = 200, {"hits": []}
            else:
                status, body = 200, {"hits": [{"title": "Python", "url": "https://www.python.org/", "rank": 1}]}
            data = json.dumps(body).encode() if body is not None else b"not JSON"
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


def make_tool(service):
    return OutcrawlSearchTool(OutcrawlSearchToolConfig(api_key="test-outcrawl-credential", base_url=service))


@pytest.mark.parametrize("status", [401, 402, 429, 500])
def test_sync_all_queries_failed_raises(service, status):
    with pytest.raises(RuntimeError, match=str(status)):
        make_tool(service).run(OutcrawlSearchToolInputSchema(queries=[f"error-{status}"]))


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 402, 429, 500])
async def test_async_all_queries_failed_raises(service, status):
    with pytest.raises(RuntimeError, match=str(status)):
        await make_tool(service).run_async(OutcrawlSearchToolInputSchema(queries=[f"error-{status}"]))


@pytest.mark.asyncio
async def test_partial_results_retain_query_failure(service):
    output = await make_tool(service).run_async(OutcrawlSearchToolInputSchema(queries=["good", "error-429", "empty"]))

    assert [item.query for item in output.results] == ["good"]
    assert len(output.failures) == 1
    assert output.failures[0].query == "error-429"
    assert "429" in output.failures[0].error
    assert "provider request rejected" in output.failures[0].error


@pytest.mark.asyncio
async def test_successful_empty_query_does_not_hide_failed_query(service):
    with pytest.raises(RuntimeError, match="401"):
        await make_tool(service).run_async(OutcrawlSearchToolInputSchema(queries=["empty", "error-401"]))


@pytest.mark.asyncio
async def test_real_empty_search_remains_a_success(service):
    output = await make_tool(service).run_async(OutcrawlSearchToolInputSchema(queries=["empty"]))
    assert output.results == []
    assert output.failures == []


@pytest.mark.asyncio
async def test_invalid_json_does_not_look_like_empty_search(service):
    with pytest.raises(RuntimeError, match="Outcrawl query"):
        await make_tool(service).run_async(OutcrawlSearchToolInputSchema(queries=["invalid-json"]))


@pytest.mark.asyncio
async def test_failed_optional_scrape_keeps_search_results(service):
    output = await make_tool(service).run_async(OutcrawlSearchToolInputSchema(queries=["good"], scrape_content=True))
    assert len(output.results) == 1
    assert output.results[0].content is None
    assert output.failures == []


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", [False, True])
async def test_provider_echoed_key_is_not_exposed(service, caplog, partial):
    params = OutcrawlSearchToolInputSchema(queries=["secret-error", "good"] if partial else ["secret-error"])
    if partial:
        output = await make_tool(service).run_async(params)
        detail = output.model_dump_json()
    else:
        with pytest.raises(RuntimeError) as error:
            await make_tool(service).run_async(params)
        detail = str(error.value)

    assert "test-outcrawl-credential" not in detail
    assert "test-outcrawl-credential" not in caplog.text
    assert "[redacted]" in detail
