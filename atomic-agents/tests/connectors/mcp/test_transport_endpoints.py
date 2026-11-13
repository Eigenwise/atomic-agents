"""MCP URL handling must agree between discovery and invocation."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp import ClientSession

from atomic_agents.connectors.mcp import (
    MCPDefinitionService,
    MCPFactory,
    MCPPromptDefinition,
    MCPResourceDefinition,
    MCPToolDefinition,
    MCPTransportType,
)


@pytest.fixture(params=[MCPTransportType.HTTP_STREAM, MCPTransportType.SSE])
def endpoint_case(request):
    transport_type = request.param
    suffix = "/mcp/" if transport_type == MCPTransportType.HTTP_STREAM else "/sse"
    return transport_type, suffix


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["tool", "resource", "prompt"])
@pytest.mark.parametrize("discovery", [True, False], ids=["discovery", "invocation"])
@pytest.mark.parametrize("url_kind", ["root", "mount", "complete", "complete_query", "query"])
async def test_transport_uses_valid_endpoint(monkeypatch, endpoint_case, kind, discovery, url_kind):
    transport_type, suffix = endpoint_case
    if url_kind == "root":
        endpoint, expected = "https://example.com", f"https://example.com{suffix}"
    elif url_kind == "mount":
        endpoint, expected = "https://example.com/api/", f"https://example.com/api{suffix}"
    elif url_kind == "complete":
        endpoint = expected = f"https://example.com/api{suffix.rstrip('/')}"
    elif url_kind == "complete_query":
        endpoint = expected = f"https://example.com/api{suffix.rstrip('/')}/?token=a%2Fb#section"
    else:
        endpoint = "https://example.com?token=a%2Fb#section"
        expected = f"https://example.com{suffix}?token=a%2Fb#section"

    module = "atomic_agents.connectors.mcp." + ("mcp_definition_service" if discovery else "mcp_factory")
    session = AsyncMock(spec=ClientSession)
    session.__aenter__.return_value = session
    session.call_tool.return_value = {"ok": True}
    session.read_resource.return_value = {"contents": []}
    session.get_prompt.return_value = {"messages": []}
    transport = AsyncMock()
    transport.__aenter__.return_value = (None, None, None) if transport_type == MCPTransportType.HTTP_STREAM else (None, None)
    client = MagicMock(return_value=transport)
    client_name = "streamablehttp_client" if transport_type == MCPTransportType.HTTP_STREAM else "sse_client"
    monkeypatch.setattr(f"{module}.{client_name}", client)
    monkeypatch.setattr(f"{module}.ClientSession", MagicMock(return_value=session))

    if discovery:
        service = MCPDefinitionService(endpoint, transport_type)
        monkeypatch.setattr(service, f"fetch_{kind}_definitions_from_session", AsyncMock(return_value=[]))
        assert await getattr(service, f"fetch_{kind}_definitions")() == []
    else:
        factory = MCPFactory(endpoint, transport_type)
        schema = {"type": "object", "properties": {}}
        if kind == "tool":
            definition = MCPToolDefinition("sample", "Example tool", schema)
            instance = factory._create_tool_classes([definition])[0]()
            await instance.arun(instance.input_schema(tool_name="sample"))
        elif kind == "resource":
            definition = MCPResourceDefinition("sample", "Example resource", "example://item", schema)
            instance = factory._create_resource_classes([definition])[0]()
            await instance.aread(instance.input_schema(resource_name="sample"))
        else:
            definition = MCPPromptDefinition("sample", "Example prompt", schema)
            instance = factory._create_prompt_classes([definition])[0]()
            await instance.agenerate(instance.input_schema(prompt_name="sample"))

    client.assert_called_once_with(expected)
    session.__aexit__.assert_awaited_once()
    transport.__aexit__.assert_awaited_once()
