from contextlib import asynccontextmanager
from inspect import signature

import anyio
import pytest
from mcp import ClientSession, types
from mcp.server.lowlevel import Server
from mcp.shared.exceptions import McpError
from mcp.shared.memory import create_client_server_memory_streams

from atomic_agents.connectors.mcp import (
    MCPDefinitionService,
    MCPTransportType,
    fetch_mcp_attributes_with_schema,
    fetch_mcp_prompts_async,
    fetch_mcp_resources_async,
    fetch_mcp_tools_async,
)


CATALOGS = {
    "tools": (types.ListToolsRequest, types.ListToolsResult, "tools"),
    "resources": (types.ListResourcesRequest, types.ListResourcesResult, "resources"),
    "templates": (types.ListResourceTemplatesRequest, types.ListResourceTemplatesResult, "resourceTemplates"),
    "prompts": (types.ListPromptsRequest, types.ListPromptsResult, "prompts"),
}
FETCHERS = {
    "tools": fetch_mcp_tools_async,
    "resources": fetch_mcp_resources_async,
    "prompts": fetch_mcp_prompts_async,
}
DEFINITIONS = {
    "tools": MCPDefinitionService.fetch_tool_definitions_from_session,
    "resources": MCPDefinitionService.fetch_resource_definitions_from_session,
    "templates": MCPDefinitionService.fetch_resource_definitions_from_session,
    "prompts": MCPDefinitionService.fetch_prompt_definitions_from_session,
}


@pytest.fixture
def catalog_server(monkeypatch):
    def create(supported, advertised=None):
        server = Server("optional-catalog-test")
        calls = []
        entries = {
            "tools": types.Tool(name="echo", inputSchema={"type": "object", "properties": {}}),
            "resources": types.Resource(name="static", uri="test://resource/static"),
            "templates": types.ResourceTemplate(name="template", uriTemplate="test://resource/{name}"),
            "prompts": types.Prompt(name="welcome"),
        }
        for kind in supported:
            request_type, result_type, field = CATALOGS[kind]

            async def list_items(request, result_type=result_type, field=field, entry=entries[kind], kind=kind):
                calls.append(kind)
                return types.ServerResult(result_type(**{field: [entry]}))

            server.request_handlers[request_type] = list_items

        if "templates" in supported and "resources" not in supported:

            async def list_static_resources(request):
                calls.append("resources")
                return types.ServerResult(types.ListResourcesResult(resources=[]))

            server.request_handlers[types.ListResourcesRequest] = list_static_resources

        @asynccontextmanager
        async def transport(endpoint=None):
            async with create_client_server_memory_streams() as (client_streams, server_streams):
                async with anyio.create_task_group() as tasks:
                    options = server.create_initialization_options()
                    # The SDK infers resources only from the static-list handler.
                    # Template-only servers still advertise the resources capability.
                    if "templates" in supported:
                        options.capabilities.resources = types.ResourcesCapability()
                    if advertised is not None:
                        for kind in ("tools", "resources", "prompts"):
                            if kind not in advertised:
                                setattr(options.capabilities, kind, None)
                    tasks.start_soon(server.run, *server_streams, options)
                    try:
                        yield (*client_streams, None)
                    finally:
                        tasks.cancel_scope.cancel()

        monkeypatch.setattr("atomic_agents.connectors.mcp.mcp_factory.streamablehttp_client", transport)
        monkeypatch.setattr("atomic_agents.connectors.mcp.mcp_definition_service.streamablehttp_client", transport)
        return server, transport, calls

    return create


@pytest.mark.parametrize("supported", [(), ("tools",), ("resources",), ("templates",), ("prompts",), tuple(CATALOGS)])
def test_combined_discovery_retains_every_supported_catalog(catalog_server, supported):
    catalog_server(supported)

    tools, resources, prompts, schema = fetch_mcp_attributes_with_schema("http://catalog.test")

    assert [tool.mcp_tool_name for tool in tools] == (["echo"] if "tools" in supported else [])
    assert [resource.mcp_resource_name for resource in resources] == (
        (["static"] if "resources" in supported else []) + (["template"] if "templates" in supported else [])
    )
    assert [prompt.mcp_prompt_name for prompt in prompts] == (["welcome"] if "prompts" in supported else [])
    assert (schema is not None) == bool(supported)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", FETCHERS)
@pytest.mark.parametrize("borrowed", [False, True], ids=["owned-session", "borrowed-session"])
async def test_absent_catalog_is_empty_through_async_factory(catalog_server, kind, borrowed):
    _, transport, _ = catalog_server(())
    if borrowed:
        async with transport() as (reader, writer, _):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                result = await FETCHERS[kind]("http://catalog.test", MCPTransportType.HTTP_STREAM, client_session=session)
                await session.send_ping()
    else:
        result = await FETCHERS[kind]("http://catalog.test", MCPTransportType.HTTP_STREAM)

    assert result == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", FETCHERS)
async def test_unnegotiated_catalogs_are_not_requested(catalog_server, kind):
    _, transport, calls = catalog_server(tuple(CATALOGS), advertised=())
    async with transport() as (reader, writer, _):
        async with ClientSession(reader, writer) as session:
            result = await DEFINITIONS[kind](session)
            await session.send_ping()

    assert result == []
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", CATALOGS)
@pytest.mark.parametrize(
    "later,code",
    [(False, types.INTERNAL_ERROR), (True, types.METHOD_NOT_FOUND), (True, types.INTERNAL_ERROR)],
    ids=["initial-error", "later-method-missing", "later-error"],
)
async def test_real_catalog_failures_are_not_treated_as_absent(catalog_server, kind, later, code):
    method = "list_resource_templates" if kind == "templates" else f"list_{kind}"
    if later and "cursor" not in signature(getattr(ClientSession, method)).parameters:
        pytest.skip("Installed MCP SDK cannot request continuation pages")
    server, transport, _ = catalog_server(tuple(CATALOGS))
    request_type, result_type, field = CATALOGS[kind]
    cursors = []

    async def list_items(request):
        cursor = request.params.cursor if request.params is not None else None
        cursors.append(cursor)
        if later and cursor is None:
            return types.ServerResult(result_type(**{field: [], "nextCursor": "continue"}))
        raise McpError(types.ErrorData(code=code, message="catalog failed"))

    server.request_handlers[request_type] = list_items
    async with transport() as (reader, writer, _):
        async with ClientSession(reader, writer) as session:
            await session.initialize()
            with pytest.raises(McpError) as failure:
                await DEFINITIONS[kind](session)
            await session.send_ping()

    assert failure.value.error.code == code
    assert failure.value.error.message == "catalog failed"
    assert cursors == ([None, "continue"] if later else [None])


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", FETCHERS)
async def test_advertised_catalog_method_not_found_remains_an_error(catalog_server, kind):
    server, transport, _ = catalog_server(tuple(CATALOGS))

    async def list_items(request):
        raise McpError(types.ErrorData(code=types.METHOD_NOT_FOUND, message="advertised method missing"))

    server.request_handlers[CATALOGS[kind][0]] = list_items
    async with transport() as (reader, writer, _):
        async with ClientSession(reader, writer) as session:
            with pytest.raises(McpError) as failure:
                await DEFINITIONS[kind](session)

    assert failure.value.error.code == types.METHOD_NOT_FOUND
    assert failure.value.error.message == "advertised method missing"
