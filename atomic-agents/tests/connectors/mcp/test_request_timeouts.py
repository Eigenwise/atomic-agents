import asyncio
import json
import shlex
import sys
import textwrap
import time

import pytest
from mcp import ClientSession, StdioServerParameters
import mcp.client.stdio as stdio_module
from mcp.client.stdio import stdio_client

from atomic_agents.connectors.mcp import (
    MCPDefinitionService,
    MCPFactory,
    MCPTransportType,
    fetch_mcp_attributes_with_schema,
    fetch_mcp_tools,
    fetch_mcp_tools_async,
    fetch_mcp_resources,
    fetch_mcp_resources_async,
    fetch_mcp_prompts,
    fetch_mcp_prompts_async,
)


@pytest.fixture
def server(tmp_path, monkeypatch):
    processes = []
    create_process = stdio_module._create_platform_compatible_process

    async def track_process(**kwargs):
        process = await create_process(**kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(stdio_module, "_create_platform_compatible_process", track_process)
    script = tmp_path / "timeout_server.py"
    log = tmp_path / "requests.jsonl"
    script.write_text(
        textwrap.dedent(
            """
        import json
        import os
        import sys
        import threading

        log_path, stall = sys.argv[1:]
        # A broken timeout regression must still terminate instead of hanging pytest.
        watchdog = threading.Timer(5, lambda: os._exit(0))
        watchdog.daemon = True
        watchdog.start()

        def record(method):
            with open(log_path, "a") as log:
                log.write(json.dumps({"pid": os.getpid(), "method": method}) + "\\n")

        for line in sys.stdin:
            request = json.loads(line)
            method = request.get("method")
            record(method)
            if "id" not in request or method == stall:
                continue
            if method == "initialize":
                result = {
                    "protocolVersion": request["params"]["protocolVersion"],
                    "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                    "serverInfo": {"name": "timeout-fixture", "version": "1"},
                }
            elif method == "tools/list":
                result = {"tools": [{"name": "ready", "description": "Ready", "inputSchema": {"type": "object", "properties": {}}}]}
            elif method == "resources/list":
                result = {"resources": [{"name": "ready", "uri": "test://ready", "description": "Ready"}]}
            elif method == "resources/templates/list":
                result = {"resourceTemplates": []}
            elif method == "prompts/list":
                result = {"prompts": [{"name": "ready", "description": "Ready", "arguments": []}]}
            elif method == "tools/call":
                result = {"content": [{"type": "text", "text": "ready"}]}
            elif method == "resources/read":
                result = {"contents": [{"uri": "test://ready", "mimeType": "text/plain", "text": "ready"}]}
            elif method == "prompts/get":
                result = {"messages": [{"role": "user", "content": {"type": "text", "text": "ready"}}]}
            else:
                raise AssertionError(method)
            print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
        record("closed")
        watchdog.cancel()
    """
        ),
        encoding="utf-8",
    )

    def command(stall="none"):
        return shlex.join([sys.executable, str(script), str(log), stall])

    def events():
        return [json.loads(line) for line in log.read_text().splitlines()]

    return command, events, processes


def assert_closed(events, method, processes):
    rows = events()
    assert any(row["method"] == method for row in rows)
    assert {row["pid"] for row in rows} == {process.pid for process in processes}
    assert all(process.returncode is not None for process in processes)


@pytest.mark.parametrize(
    "kind,method",
    [
        ("tool", "initialize"),
        ("tool", "tools/list"),
        ("resource", "resources/list"),
        ("resource", "resources/templates/list"),
        ("prompt", "prompts/list"),
    ],
)
def test_definition_requests_time_out_and_close_owned_process(server, kind, method):
    command, events, processes = server
    service = MCPDefinitionService(command(method), MCPTransportType.STDIO, request_timeout=0.5)
    started = time.monotonic()

    with pytest.raises(RuntimeError, match="Timed out while waiting for response"):
        asyncio.run(getattr(service, f"fetch_{kind}_definitions")())

    assert time.monotonic() - started < 4
    assert_closed(events, method, processes)


@pytest.mark.parametrize(
    "kind,fetch,method",
    [
        ("tool", fetch_mcp_tools, "tools/call"),
        ("resource", fetch_mcp_resources, "resources/read"),
        ("prompt", fetch_mcp_prompts, "prompts/get"),
    ],
)
def test_sync_helpers_keep_timeout_for_later_invocations(server, kind, fetch, method):
    command, events, processes = server
    components = fetch(command(method), MCPTransportType.STDIO, request_timeout=0.5)
    component = components[0]()
    params = component.input_schema(**{kind + "_name": "ready"})
    started = time.monotonic()

    with pytest.raises(RuntimeError, match="Timed out while waiting for response"):
        getattr(component, {"tool": "run", "resource": "read", "prompt": "generate"}[kind])(params)

    assert time.monotonic() - started < 4
    assert_closed(events, method, processes)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,fetch,method",
    [
        ("tool", fetch_mcp_tools_async, "tools/call"),
        ("resource", fetch_mcp_resources_async, "resources/read"),
        ("prompt", fetch_mcp_prompts_async, "prompts/get"),
    ],
)
async def test_async_helpers_keep_timeout_for_later_invocations(server, kind, fetch, method):
    command, events, processes = server
    components = await fetch(command(method), MCPTransportType.STDIO, request_timeout=0.5)
    component = components[0]()
    params = component.input_schema(**{kind + "_name": "ready"})
    started = time.monotonic()

    with pytest.raises(RuntimeError, match="Timed out while waiting for response"):
        await getattr(component, {"tool": "arun", "resource": "aread", "prompt": "agenerate"}[kind])(params)

    assert time.monotonic() - started < 4
    assert_closed(events, method, processes)


@pytest.mark.parametrize("options", [{}, {"request_timeout": None}, {"request_timeout": 1}])
def test_successful_combined_discovery_and_execution_with_and_without_timeout(server, options):
    command, events, processes = server
    tools, resources, prompts, schema = fetch_mcp_attributes_with_schema(command(), MCPTransportType.STDIO, **options)

    assert schema is not None
    for kind, components in zip(("tool", "resource", "prompt"), (tools, resources, prompts)):
        component = components[0]()
        params = component.input_schema(**{kind + "_name": "ready"})
        result = getattr(component, {"tool": "run", "resource": "read", "prompt": "generate"}[kind])(params)
        assert "ready" in str(result)
    assert_closed(events, "prompts/get", processes)


@pytest.mark.parametrize("constructor", [MCPFactory, MCPDefinitionService])
@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True, "1", 1e-20, 1e100])
def test_invalid_timeouts_fail_before_connecting(constructor, timeout):
    with pytest.raises(ValueError, match="request_timeout"):
        constructor("unused", request_timeout=timeout)


@pytest.mark.asyncio
@pytest.mark.parametrize("fetch", [fetch_mcp_tools_async, fetch_mcp_resources_async, fetch_mcp_prompts_async])
async def test_timeout_cannot_silently_override_borrowed_session_policy(server, fetch):
    command, events, _ = server
    executable, *args = shlex.split(command())
    async with stdio_client(StdioServerParameters(command=executable, args=args)) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            await session.list_tools()
            before = events()
            with pytest.raises(ValueError, match="client_session"):
                await fetch(client_session=session, request_timeout=0.5)
            assert events() == before
            # The caller still owns and can use the session after configuration is rejected.
            assert (await session.list_tools()).tools[0].name == "ready"
