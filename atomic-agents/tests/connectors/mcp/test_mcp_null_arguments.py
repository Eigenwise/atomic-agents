from contextlib import asynccontextmanager
from unittest.mock import create_autospec

import pytest
from mcp import ClientSession
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from atomic_agents.connectors.mcp import fetch_mcp_tools_async, MCPTransportType
from atomic_agents.connectors.mcp import mcp_factory, mcp_definition_service


@pytest.mark.asyncio
@pytest.mark.parametrize("session_mode", ["persistent", "fresh"])
@pytest.mark.parametrize(
    "properties, required, arguments, expected",
    [
        ({"value": {"type": ["string", "null"]}}, ["value"], {"value": None}, {"value": None}),
        ({"value": {"type": ["string", "null"]}}, [], {"value": None}, {"value": None}),
        ({"value": {"type": ["string", "null"]}}, [], {}, {}),
        ({"value": {"type": "string", "default": "server-default"}}, [], {}, {}),
        (
            {"value": {"type": "string", "default": "server-default"}},
            [],
            {"value": "server-default"},
            {"value": "server-default"},
        ),
        ({"value": {"type": "integer"}}, ["value"], {"value": 0}, {"value": 0}),
        ({"value": {"type": "boolean"}}, ["value"], {"value": False}, {"value": False}),
        ({"value": {"type": "string"}}, [], {"value": None}, {}),
        ({"value": {"type": "integer"}}, [], {"value": None}, {}),
        ({"value": {"type": "boolean"}}, [], {"value": None}, {}),
        ({"value": {"anyOf": [{"type": "string"}, {"type": "null"}]}}, [], {"value": None}, {"value": None}),
        ({"value": {"oneOf": [{"type": "string"}, {"type": "null"}]}}, [], {"value": None}, {"value": None}),
        ({"value": {"type": "null"}}, [], {"value": None}, {"value": None}),
        ({"value": {}}, [], {"value": None}, {"value": None}),
        ({"value": {"enum": ["brief"]}}, [], {"value": None}, {}),
        ({"value": {"const": "brief"}}, [], {"value": None}, {}),
        ({"value": {"const": None}}, [], {"value": None}, {"value": None}),
    ],
    ids=[
        "required-null",
        "explicit-optional-null",
        "omitted-optional",
        "omitted-default",
        "explicit-default",
        "zero",
        "false",
        "nonnullable-string-null",
        "nonnullable-integer-null",
        "nonnullable-boolean-null",
        "nullable-anyof",
        "nullable-oneof",
        "null-type",
        "unrestricted-null",
        "nonnullable-enum-null",
        "nonnullable-const-null",
        "nullable-const-null",
    ],
)
async def test_generated_tool_preserves_argument_presence(
    properties, required, arguments, expected, session_mode, monkeypatch
):
    """The generated tool must preserve what the caller actually sends to MCP."""
    session = create_autospec(ClientSession, instance=True)
    session.list_tools.return_value = ListToolsResult(
        tools=[
            Tool(
                name="update_value",
                description="Update or clear a value",
                inputSchema={"type": "object", "properties": properties, "required": required},
            )
        ]
    )
    session.call_tool.return_value = CallToolResult(content=[TextContent(type="text", text="updated")])

    if session_mode == "persistent":
        tool_classes = await fetch_mcp_tools_async(client_session=session)
    else:

        @asynccontextmanager
        async def transport(*args, **kwargs):
            yield None, None, None

        session.__aenter__.return_value = session
        for module in (mcp_factory, mcp_definition_service):
            monkeypatch.setattr(module, "streamablehttp_client", transport)
            monkeypatch.setattr(module, "ClientSession", lambda *args: session)
        tool_classes = await fetch_mcp_tools_async(
            mcp_endpoint="https://example.com", transport_type=MCPTransportType.HTTP_STREAM
        )
    tool = tool_classes[0]()
    params = tool.input_schema(tool_name="update_value", **arguments)

    await tool.arun(params)

    session.call_tool.assert_awaited_once_with(name="update_value", arguments=expected)


@pytest.mark.asyncio
async def test_generated_tools_keep_independent_referenced_nullability():
    """Each tool retains its own declared nullability when definitions share an argument name."""
    session = create_autospec(ClientSession, instance=True)
    session.list_tools.return_value = ListToolsResult(
        tools=[
            Tool(
                name=name,
                inputSchema={
                    "type": "object",
                    "properties": {"value": {"$ref": "#/$defs/Value"}},
                    "$defs": {"Value": {"type": types}},
                },
            )
            for name, types in [("clear", ["string", "null"]), ("keep", "string")]
        ]
    )
    session.call_tool.return_value = CallToolResult(content=[TextContent(type="text", text="updated")])
    for tool_cls in await fetch_mcp_tools_async(client_session=session):
        tool = tool_cls()
        await tool.arun(tool.input_schema(tool_name=tool.mcp_tool_name, value=None))
    assert session.call_tool.await_args_list[0].kwargs == {"name": "clear", "arguments": {"value": None}}
    assert session.call_tool.await_args_list[1].kwargs == {"name": "keep", "arguments": {}}
