from unittest.mock import create_autospec

import pytest
from mcp import ClientSession
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from atomic_agents.connectors.mcp import fetch_mcp_tools_async


@pytest.mark.asyncio
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
    ],
    ids=[
        "required-null",
        "explicit-optional-null",
        "omitted-optional",
        "omitted-default",
        "explicit-default",
        "zero",
        "false",
    ],
)
async def test_generated_tool_preserves_argument_presence(properties, required, arguments, expected):
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

    tool_classes = await fetch_mcp_tools_async(client_session=session)
    tool = tool_classes[0]()
    params = tool.input_schema(tool_name="update_value", **arguments)

    await tool.arun(params)

    session.call_tool.assert_awaited_once_with(name="update_value", arguments=expected)
