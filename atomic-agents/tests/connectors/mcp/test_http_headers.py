"""Exercise authenticated MCP discovery and generated calls over real local HTTP."""

import asyncio
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from queue import Empty, Queue
from threading import Event, Thread
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

import pytest
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client

from atomic_agents.connectors.mcp import mcp_factory as factory_module
from atomic_agents.connectors.mcp.mcp_definition_service import MCPDefinitionService, MCPTransportType
from atomic_agents.connectors.mcp.mcp_factory import MCPFactory


HEADERS = {"Authorization": "Bearer local-test-token", "X-Tenant": "research"}


@pytest.fixture
def mcp_server():
    stopped = Event()
    state = {"requests": [], "queues": {}, "required": HEADERS.copy(), "denied": []}

    def result_for(message):
        method = message["method"]
        params = message.get("params", {})
        if method == "initialize":
            return {
                "protocolVersion": params["protocolVersion"],
                "capabilities": {"tools": {}, "resources": {}, "prompts": {}},
                "serverInfo": {"name": "header-test", "version": "1"},
            }
        if method == "tools/list":
            return {
                "tools": [
                    {
                        "name": "echo",
                        "description": "Echo text.",
                        "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
                    }
                ]
            }
        if method == "resources/list":
            return {"resources": []}
        if method == "resources/templates/list":
            return {"resourceTemplates": [{"name": "page", "uriTemplate": "echo://item/{text}", "mimeType": "text/plain"}]}
        if method == "prompts/list":
            return {"prompts": [{"name": "greet", "arguments": [{"name": "text", "required": True}]}]}
        if method == "tools/call":
            return {"content": [{"type": "text", "text": "tool:" + params["arguments"]["text"]}]}
        if method == "resources/read":
            return {"contents": [{"uri": params["uri"], "mimeType": "text/plain", "text": "resource:" + params["uri"]}]}
        if method == "prompts/get":
            return {
                "messages": [{"role": "user", "content": {"type": "text", "text": "prompt:" + params["arguments"]["text"]}}]
            }
        if method == "ping":
            return {}
        raise AssertionError(f"Unexpected method: {method}")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def record_and_authorize(self, method):
            headers = {name: self.headers.get(name) for name in HEADERS}
            state["requests"].append((self.command, method, headers))
            if any(self.headers.get(name) != value for name, value in state["required"].items()):
                state["denied"].append(method)
                self.reply(401)
                return False
            return True

        def reply(self, status, payload=None):
            data = json.dumps(payload).encode() if payload is not None else b""
            self.send_response(status)
            if payload is not None:
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            message = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if not self.record_and_authorize(message["method"]):
                return
            if "id" not in message:
                self.reply(202)
                return
            result = {"jsonrpc": "2.0", "id": message["id"], "result": result_for(message)}
            if urlsplit(self.path).path == "/messages/":
                session_id = parse_qs(urlsplit(self.path).query)["session_id"][0]
                state["queues"][session_id].put(result)
                self.reply(202)
            else:
                self.reply(200, result)

        def do_GET(self):
            if not self.record_and_authorize("GET"):
                return
            if urlsplit(self.path).path != "/sse":
                self.reply(405)
                return
            session_id = str(uuid4())
            pending = state["queues"][session_id] = Queue()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                self.wfile.write(f"event: endpoint\ndata: /messages/?session_id={session_id}\n\n".encode())
                self.wfile.flush()
                while not stopped.is_set():
                    try:
                        message = pending.get(timeout=0.05)
                        data = "event: message\ndata: " + json.dumps(message) + "\n\n"
                    except Empty:
                        data = ": keepalive\n\n"
                    self.wfile.write(data.encode())
                    self.wfile.flush()
            except OSError:
                pass
            finally:
                state["queues"].pop(session_id, None)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, f"http://127.0.0.1:{server.server_port}"
    finally:
        stopped.set()
        server.shutdown()
        server.server_close()
        thread.join()


def check_output(kind, output):
    if kind == "tools":
        assert output.result[0].text == "tool:sample"
    elif kind == "resources":
        assert output.content[0].text == "resource:echo://item/sample"
    else:
        assert output.content == "prompt:sample"


def call_arguments(kind, component):
    route = {"tools": ("tool_name", "echo"), "resources": ("resource_name", "page"), "prompts": ("prompt_name", "greet")}
    field, name = route[kind]
    return component.input_schema(**{field: name, "text": "sample"})


def call_sync(kind, generated):
    component = generated[0]()
    method = {"tools": "run", "resources": "read", "prompts": "generate"}[kind]
    check_output(kind, getattr(component, method)(call_arguments(kind, component)))


async def call_async(kind, generated):
    component = generated[0]()
    method = {"tools": "arun", "resources": "aread", "prompts": "agenerate"}[kind]
    check_output(kind, await getattr(component, method)(call_arguments(kind, component)))


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", [MCPTransportType.HTTP_STREAM, MCPTransportType.SSE])
@pytest.mark.parametrize("kind", ["tools", "resources", "prompts"])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_headers_reach_discovery_and_generated_operations(mcp_server, transport, kind, asynchronous):
    state, endpoint = mcp_server
    if asynchronous:
        fetch = getattr(factory_module, f"fetch_mcp_{kind}_async")
        generated = await fetch(endpoint, transport, http_headers=HEADERS)
        await call_async(kind, generated)
    else:

        def run():
            generated = getattr(factory_module, f"fetch_mcp_{kind}")(endpoint, transport, http_headers=HEADERS)
            call_sync(kind, generated)

        await asyncio.to_thread(run)
    assert state["requests"]
    assert all(headers == HEADERS for _, _, headers in state["requests"])
    methods = {method for _, method, _ in state["requests"]}
    assert "initialize" in methods
    assert {"tools": "tools/call", "resources": "resources/read", "prompts": "prompts/get"}[kind] in methods
    if transport == MCPTransportType.SSE:
        assert "GET" in methods


@pytest.mark.parametrize("transport", [MCPTransportType.HTTP_STREAM, MCPTransportType.SSE])
def test_combined_helper_keeps_headers_for_every_catalog_and_call(mcp_server, transport):
    state, endpoint = mcp_server
    tools, resources, prompts, schema = factory_module.fetch_mcp_attributes_with_schema(
        endpoint, transport, http_headers=HEADERS
    )
    assert schema is not None
    for kind, components in zip(["tools", "resources", "prompts"], [tools, resources, prompts]):
        call_sync(kind, components)
    assert all(headers == HEADERS for _, _, headers in state["requests"])


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", [MCPTransportType.HTTP_STREAM, MCPTransportType.SSE])
async def test_definition_service_copies_caller_headers(mcp_server, transport):
    state, endpoint = mcp_server
    headers = HEADERS.copy()
    service = MCPDefinitionService(endpoint, transport, http_headers=headers)
    headers["Authorization"] = "changed after construction"
    assert len(await service.fetch_tool_definitions()) == 1
    assert len(await service.fetch_resource_definitions()) == 1
    assert len(await service.fetch_prompt_definitions()) == 1
    assert all(actual == HEADERS for _, _, actual in state["requests"])


@pytest.mark.parametrize("transport", [MCPTransportType.HTTP_STREAM, MCPTransportType.SSE])
def test_factory_copies_caller_headers_and_preserves_each_factory(mcp_server, transport):
    state, endpoint = mcp_server
    headers = HEADERS.copy()
    first = MCPFactory(endpoint, transport, http_headers=headers)
    second = MCPFactory(endpoint, transport, http_headers={**HEADERS, "X-Tenant": "different"})
    headers["Authorization"] = "changed after construction"
    call_sync("tools", first.create_tools())
    state["required"]["X-Tenant"] = "different"
    call_sync("tools", second.create_tools())
    state["required"] = HEADERS.copy()
    call_sync("tools", first.create_tools())


@pytest.mark.parametrize("transport", [MCPTransportType.HTTP_STREAM, MCPTransportType.SSE])
def test_default_headers_still_work_for_public_servers(mcp_server, transport):
    state, endpoint = mcp_server
    state["required"] = {}
    call_sync("tools", factory_module.fetch_mcp_tools(endpoint, transport))
    assert all(headers == {name: None for name in HEADERS} for _, _, headers in state["requests"])


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", [MCPTransportType.HTTP_STREAM, MCPTransportType.SSE])
async def test_protected_server_rejects_requests_without_configured_headers(mcp_server, transport):
    state, endpoint = mcp_server
    # Old SDKs leave initialize waiting after the transport receives a 401.
    # Bound that existing behavior while checking the actual server denial.
    with pytest.raises(Exception):
        await asyncio.wait_for(factory_module.fetch_mcp_tools_async(endpoint, transport), timeout=2)
    assert state["denied"]
    assert set(state["denied"]) <= {"initialize", "GET"}
    assert state["requests"]
    assert all(headers == {name: None for name in HEADERS} for _, _, headers in state["requests"])


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", [MCPTransportType.HTTP_STREAM, MCPTransportType.SSE])
async def test_borrowed_session_keeps_its_own_headers_and_lifetime(mcp_server, transport):
    state, endpoint = mcp_server
    connection = (
        streamablehttp_client(endpoint + "/mcp/", headers=HEADERS)
        if transport == MCPTransportType.HTTP_STREAM
        else sse_client(endpoint + "/sse", headers=HEADERS)
    )
    async with connection as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            for kind in ["tools", "resources", "prompts"]:
                fetch = getattr(factory_module, f"fetch_mcp_{kind}_async")
                generated = await fetch(client_session=session, http_headers={"Authorization": "unused"})
                await call_async(kind, generated)
            await session.send_ping()
    assert all(headers == HEADERS for _, _, headers in state["requests"])
