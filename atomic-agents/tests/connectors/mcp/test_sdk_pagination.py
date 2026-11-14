from contextlib import asynccontextmanager

import anyio
import pytest
from mcp import ClientSession, types
from mcp.shared.exceptions import McpError
from mcp.shared.memory import create_client_server_memory_streams
from mcp.shared.message import SessionMessage

from atomic_agents.connectors.mcp import MCPDefinitionService


CATALOGS = {
    "tools": (types.ListToolsRequest, types.ListToolsResult, "tools"),
    "resources": (types.ListResourcesRequest, types.ListResourcesResult, "resources"),
    "templates": (types.ListResourceTemplatesRequest, types.ListResourceTemplatesResult, "resourceTemplates"),
    "prompts": (types.ListPromptsRequest, types.ListPromptsResult, "prompts"),
}
LOADERS = {
    "tools": MCPDefinitionService.fetch_tool_definitions_from_session,
    "resources": MCPDefinitionService.fetch_resource_definitions_from_session,
    "templates": MCPDefinitionService.fetch_resource_definitions_from_session,
    "prompts": MCPDefinitionService.fetch_prompt_definitions_from_session,
}
METHODS = {
    "tools": "tools/list",
    "resources": "resources/list",
    "templates": "resources/templates/list",
    "prompts": "prompts/list",
}


def item(kind, name):
    if kind == "tools":
        return types.Tool(name=name, inputSchema={"type": "object"})
    if kind == "resources":
        return types.Resource(name=name, uri=f"test://catalog/{name}")
    if kind == "templates":
        return types.ResourceTemplate(name=name, uriTemplate=f"test://catalog/{{{name}}}")
    return types.Prompt(name=name)


@asynccontextmanager
async def catalog_session(kind, handler, call_handler=None):
    # Read the actual JSON-RPC envelope instead of an SDK's list request model:
    # MCP 1.8 models cursor at the root and would discard a correct params.cursor.
    async def serve(reader, writer):
        async for message in reader:
            request = message.message.root
            if not isinstance(request, types.JSONRPCRequest):
                continue
            assert "cursor" not in request.model_dump(exclude_none=True)
            try:
                if request.method == "initialize":
                    result = types.InitializeResult(
                        protocolVersion=request.params["protocolVersion"],
                        capabilities=types.ServerCapabilities(
                            tools=types.ToolsCapability(),
                            resources=types.ResourcesCapability(),
                            prompts=types.PromptsCapability(),
                        ),
                        serverInfo=types.Implementation(name="sdk-pagination-test", version="1"),
                    )
                elif request.method == METHODS[kind]:
                    result = await handler(request)
                elif request.method == "tools/call" and call_handler is not None:
                    result = await call_handler(request)
                elif request.method == "ping":
                    result = types.EmptyResult()
                else:
                    other = next(name for name, method in METHODS.items() if method == request.method)
                    _, result_type, field = CATALOGS[other]
                    result = result_type(**{field: []})
                response = types.JSONRPCResponse(
                    jsonrpc="2.0", id=request.id, result=result.model_dump(by_alias=True, mode="json", exclude_none=True)
                )
            except McpError as error:
                response = types.JSONRPCError(jsonrpc="2.0", id=request.id, error=error.error)
            await writer.send(SessionMessage(types.JSONRPCMessage(response)))

    async with create_client_server_memory_streams() as (client_streams, server_streams):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(serve, *server_streams)
            try:
                async with ClientSession(*client_streams) as session:
                    yield session
            finally:
                tasks.cancel_scope.cancel()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", CATALOGS)
@pytest.mark.parametrize("empty_first", [False, True])
async def test_sdk_discovers_every_page_with_opaque_and_empty_cursors(kind, empty_first):
    _, result_type, field = CATALOGS[kind]
    cursors = []

    async def list_items(request):
        cursor = request.params.get("cursor") if request.params is not None else None
        cursors.append(cursor)
        if cursor is None:
            values = [] if empty_first else [item(kind, "first")]
            next_cursor = "opaque:/?=a+b 雪"
        elif cursor == "opaque:/?=a+b 雪":
            values, next_cursor = [item(kind, "middle")], ""
        else:
            assert cursor == ""
            values, next_cursor = [item(kind, "last")], None
        return types.ServerResult(result_type(**{field: values, "nextCursor": next_cursor}))

    async with catalog_session(kind, list_items) as session:
        definitions = await LOADERS[kind](session)
        await session.send_ping()

    assert [definition.name for definition in definitions] == ([] if empty_first else ["first"]) + ["middle", "last"]
    assert cursors == [None, "opaque:/?=a+b 雪", ""]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", CATALOGS)
@pytest.mark.parametrize("code", [types.METHOD_NOT_FOUND, types.INTERNAL_ERROR])
async def test_sdk_continuation_errors_propagate_without_losing_the_session(kind, code):
    _, result_type, field = CATALOGS[kind]
    cursors = []

    async def list_items(request):
        cursor = request.params.get("cursor") if request.params is not None else None
        cursors.append(cursor)
        if cursor is None:
            return types.ServerResult(result_type(**{field: [item(kind, "first")], "nextCursor": "continue"}))
        raise McpError(types.ErrorData(code=code, message="continuation failed"))

    async with catalog_session(kind, list_items) as session:
        with pytest.raises(McpError) as failure:
            await LOADERS[kind](session)
        await session.send_ping()

    assert failure.value.error.code == code
    assert failure.value.error.message == "continuation failed"
    assert cursors == [None, "continue"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", CATALOGS)
async def test_sdk_repeated_cursor_still_stops_discovery(kind):
    _, result_type, field = CATALOGS[kind]
    cursors = []

    async def list_items(request):
        cursors.append(request.params.get("cursor") if request.params is not None else None)
        return types.ServerResult(result_type(**{field: [], "nextCursor": "loop"}))

    async with catalog_session(kind, list_items) as session:
        with pytest.raises(RuntimeError, match="repeated cursor"):
            await LOADERS[kind](session)
        await session.send_ping()

    assert cursors == [None, "loop"]


@pytest.mark.asyncio
@pytest.mark.skipif("outputSchema" not in types.Tool.model_fields, reason="This SDK predates tool output validation")
async def test_native_later_page_tool_output_validation_is_preserved():
    cursors = []

    async def list_tools(request):
        cursor = request.params.get("cursor") if request.params is not None else None
        cursors.append(cursor)
        if cursor is None:
            return types.ServerResult(types.ListToolsResult(tools=[item("tools", "first")], nextCursor="last-page"))
        return types.ServerResult(
            types.ListToolsResult(
                tools=[
                    types.Tool(
                        name="last",
                        inputSchema={"type": "object"},
                        outputSchema={"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]},
                    )
                ]
            )
        )

    async def call_tool(request):
        return types.ServerResult(types.CallToolResult(content=[], structuredContent={"count": "not an integer"}))

    async with catalog_session("tools", list_tools, call_tool) as session:
        await LOADERS["tools"](session)
        with pytest.raises(RuntimeError, match="Invalid structured content"):
            await session.call_tool("last", {})
        await session.send_ping()

    assert cursors == [None, "last-page"]
