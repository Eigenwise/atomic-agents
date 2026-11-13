import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from mcp import ClientSession, types
from mcp.server import Server
from mcp.shared.memory import create_connected_server_and_client_session

from atomic_agents.connectors.mcp import (
    MCPDefinitionService,
    fetch_mcp_prompts,
    fetch_mcp_prompts_async,
    fetch_mcp_resources,
    fetch_mcp_resources_async,
    fetch_mcp_tools,
    fetch_mcp_tools_async,
)


def discovery_session():
    session = AsyncMock(spec=ClientSession)
    session.list_tools.return_value = types.ListToolsResult(tools=[])
    session.list_resources.return_value = types.ListResourcesResult(resources=[])
    session.list_resource_templates.return_value = types.ListResourceTemplatesResult(resourceTemplates=[])
    session.list_prompts.return_value = types.ListPromptsResult(prompts=[])
    return session


def test_sync_discovery_reuses_the_supplied_initialized_session():
    session = discovery_session()
    loop = asyncio.new_event_loop()
    try:
        for fetch in (fetch_mcp_tools, fetch_mcp_resources, fetch_mcp_prompts):
            assert fetch(client_session=session, event_loop=loop) == []
        session.initialize.assert_not_awaited()
        session.list_tools.assert_awaited_once()
        session.list_resources.assert_awaited_once()
        session.list_resource_templates.assert_awaited_once()
        session.list_prompts.assert_awaited_once()
    finally:
        loop.close()


@pytest.mark.asyncio
async def test_async_discovery_does_not_repeat_the_real_session_handshake():
    server = Server("reused-discovery-test")

    @server.list_tools()
    async def tools_list():
        return [types.Tool(name="example", inputSchema={"type": "object", "properties": {}})]

    @server.list_resources()
    async def resources_list():
        return [types.Resource(name="example", uri="memory://example")]

    @server.list_resource_templates()
    async def templates_list():
        return []

    @server.list_prompts()
    async def prompts_list():
        return [types.Prompt(name="example")]

    # The SDK helper establishes and initializes a real client/server session.
    async with create_connected_server_and_client_session(server) as session:
        with patch.object(session, "send_request", wraps=session.send_request) as send_request:
            for fetch in (fetch_mcp_tools_async, fetch_mcp_resources_async, fetch_mcp_prompts_async):
                assert len(await fetch(client_session=session)) == 1
            methods = [call.args[0].root.method for call in send_request.await_args_list]
            assert methods == ["tools/list", "resources/list", "resources/templates/list", "prompts/list"]


@pytest.mark.parametrize("kind", ["tool", "resource", "prompt"])
@pytest.mark.asyncio
async def test_direct_session_discovery_still_initializes_by_default(kind):
    session = discovery_session()
    fetch = getattr(MCPDefinitionService, f"fetch_{kind}_definitions_from_session")
    assert await fetch(session) == []
    session.initialize.assert_awaited_once()
