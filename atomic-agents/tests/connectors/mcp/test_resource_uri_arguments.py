"""Resource template values must not become URI delimiters."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp import ClientSession
from mcp.types import ReadResourceResult

from atomic_agents.connectors.mcp import MCPResourceDefinition
from atomic_agents.connectors.mcp.mcp_factory import MCPFactory


@pytest.mark.asyncio
@pytest.mark.parametrize("persistent", [False, True], ids=["owned", "borrowed"])
@pytest.mark.parametrize("template", ["resource://items/prefix%2F/{item}", "resource://search?q={item}&fixed=1"])
@pytest.mark.parametrize(
    "value,encoded",
    [
        ("report-1~final", "report-1~final"),
        ("folder/name?draft#section", "folder%2Fname%3Fdraft%23section"),
        ("x&limit=99", "x%26limit%3D99"),
        ("测 试", "%E6%B5%8B%20%E8%AF%95"),
        ("%2F{raw}", "%252F%7Braw%7D"),
    ],
)
async def test_resource_uri_encodes_template_values(monkeypatch, persistent, template, value, encoded):
    session = AsyncMock(spec=ClientSession)
    session.__aenter__.return_value = session
    session.read_resource.return_value = ReadResourceResult(contents=[])
    transport = AsyncMock()
    transport.__aenter__.return_value = (None, None, None)
    monkeypatch.setattr("atomic_agents.connectors.mcp.mcp_factory.ClientSession", MagicMock(return_value=session))
    monkeypatch.setattr("atomic_agents.connectors.mcp.mcp_factory.streamablehttp_client", MagicMock(return_value=transport))
    definition = MCPResourceDefinition(
        name="Item",
        description="Read a named item",
        uri=template,
        input_schema={"type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]},
    )
    resource = MCPFactory("http://example.test")._create_resource_classes([definition])[0]()
    if persistent:
        resource._client_session = session
    await resource.aread(resource.input_schema(resource_name="Item", item=value))
    session.read_resource.assert_awaited_once_with(uri=template.replace("{item}", encoded))
    assert session.initialize.await_count == int(not persistent)
