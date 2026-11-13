"""MCP property names remain JSON keys even when Python reserves their spelling."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from mcp import ClientSession
from mcp.types import CallToolResult
from pydantic import ValidationError

from atomic_agents.connectors.mcp import MCPFactory, MCPToolDefinition, SchemaTransformer


@pytest.mark.parametrize("name", ["_id", "__value", "model_dump", "model_config"])
@pytest.mark.parametrize("output", [False, True])
def test_reserved_property_names_are_required_and_serialized_under_original_keys(name, output):
    schema = {"type": "object", "properties": {name: {"type": "integer"}}, "required": [name]}
    model = SchemaTransformer.create_model_from_schema(schema, "NamedModel", "lookup", is_output_schema=output)
    identifier = {} if output else {"tool_name": "lookup"}
    value = {name: 7, **identifier}
    assert model.model_validate(value).model_dump() == value
    assert name in model.model_json_schema()["properties"]
    assert name in model.model_json_schema()["required"]
    with pytest.raises(ValidationError):
        model.model_validate(identifier)
    with pytest.raises(ValidationError):
        model.model_validate({name: "invalid", **identifier})


def test_generated_field_names_do_not_shadow_real_properties():
    values = {"_id": 1, "mcp_field_0": 2, "mcp_field_1": 3, "model_dump": 4}
    schema = {
        "type": "object",
        "properties": {name: {"type": "integer"} for name in values},
        "required": list(values),
    }
    model = SchemaTransformer.create_model_from_schema(schema, "CollisionModel", "lookup", is_output_schema=True)
    assert model.model_validate(values).model_dump() == values
    assert set(model.model_json_schema()["properties"]) == set(values)


@pytest.mark.asyncio
async def test_generated_tool_keeps_reserved_keys_in_arguments_and_structured_results():
    schema = {
        "type": "object",
        "properties": {"_id": {"type": "integer"}, "item": {"$ref": "#/$defs/Item"}},
        "required": ["_id", "item"],
        "$defs": {
            "Item": {
                "type": "object",
                "properties": {"_key": {"type": "string"}, "model_config": {"type": "integer"}},
                "required": ["_key", "model_config"],
            }
        },
    }
    session = AsyncMock(spec=ClientSession)
    output = {"_id": 9, "item": {"_key": "returned", "model_config": 4}}
    session.call_tool.return_value = CallToolResult(content=[], structuredContent=output)
    factory = MCPFactory(client_session=session, event_loop=asyncio.get_running_loop())
    tools = factory._create_tool_classes([MCPToolDefinition("lookup", "Look up an item", schema, schema)])
    assert len(tools) == 1
    tool = tools[0]()
    arguments = {"_id": 2, "item": {"_key": "requested", "model_config": 3}}
    params = tool.input_schema.model_validate({"tool_name": "lookup", **arguments})
    result = await tool.arun(params)
    session.call_tool.assert_awaited_once_with(name="lookup", arguments=arguments)
    assert result.model_dump() == output
    item_schema = tool.input_schema.model_json_schema()["$defs"]["Item"]
    assert set(item_schema["properties"]) == {"_key", "model_config"}
    with pytest.raises(ValidationError):
        tool.input_schema.model_validate({"tool_name": "lookup", "_id": 2, "item": {"_key": "missing"}})
