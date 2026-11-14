"""MCP set-like arrays retain their JSON Schema uniqueness contract."""

import asyncio
import json

import pytest
from jsonschema import Draft202012Validator
from mcp import types
from mcp.server import Server
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import ValidationError

from atomic_agents.connectors.mcp import MCPFactory, MCPToolDefinition, SchemaTransformer


def array_model(items=None, unique=True, required=True):
    field = {"type": "array", "items": items or {}, "uniqueItems": unique}
    schema = {"type": "object", "properties": {"values": field}, "required": ["values"] if required else []}
    return SchemaTransformer.create_model_from_schema(schema, "UniqueOutput", "unique", is_output_schema=True)


@pytest.mark.parametrize(
    "values",
    [
        [1, 1],
        [1, 1.0],
        ["red", "red"],
        [None, None],
        [{"a": 1, "b": 2}, {"b": 2, "a": 1}],
        [[1, {"name": "same"}], [1.0, {"name": "same"}]],
    ],
)
def test_duplicate_json_values_are_rejected_locally_and_in_the_emitted_schema(values):
    model = array_model()
    with pytest.raises(ValidationError):
        model(values=values)
    assert not Draft202012Validator(model.model_json_schema()).is_valid({"values": values})


@pytest.mark.parametrize("values", [[], [1], [True, 1], [False, 0], [[True], [1]], [{"a": 1}, {"a": 2}]])
def test_distinct_values_preserve_json_scalar_identity_and_input_order(values):
    model = array_model()
    assert model(values=values).model_dump() == {"values": values}
    assert Draft202012Validator(model.model_json_schema()).is_valid({"values": values})


def test_uniqueness_is_checked_after_typed_array_normalization():
    model = array_model({"type": "integer"})
    with pytest.raises(ValidationError):
        model(values=[1, "1"])
    with pytest.raises(ValidationError):
        model(values=(1, 1))


def test_referenced_record_arrays_compare_the_serialized_records():
    schema = {
        "type": "object",
        "properties": {"values": {"type": "array", "uniqueItems": True, "items": {"$ref": "#/$defs/Item"}}},
        "required": ["values"],
        "$defs": {"Item": {"type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"]}},
    }
    model = SchemaTransformer.create_model_from_schema(schema, "RecordsOutput", "records", is_output_schema=True)
    assert model(values=[{"count": 1}, {"count": 2}]).model_dump() == {"values": [{"count": 1}, {"count": 2}]}
    with pytest.raises(ValidationError):
        model(values=[{"count": 1}, {"count": "1"}])


def test_optional_unique_arrays_keep_the_existing_absent_value_behavior():
    model = array_model(required=False)
    assert model().values is None
    assert model(values=None).values is None
    assert model(values=[]).values == []
    with pytest.raises(ValidationError):
        model(values=["same", "same"])
    assert not Draft202012Validator(model.model_json_schema()).is_valid({"values": ["same", "same"]})


def test_false_uniqueness_keeps_duplicate_values():
    model = array_model(unique=False)
    value = {"values": [1, 1]}
    assert model.model_validate(value).model_dump() == value
    assert Draft202012Validator(model.model_json_schema()).is_valid(value)


@pytest.mark.asyncio
async def test_generated_tool_rejects_duplicate_arguments_before_calling_the_server():
    schema = {
        "type": "object",
        "properties": {"values": {"type": "array", "items": {"type": "integer"}, "uniqueItems": True}},
        "required": ["values"],
    }
    server = Server("unique-array-regression")
    received = []

    @server.list_tools()
    async def list_tools():
        return [types.Tool(name="batch", inputSchema=schema)]

    @server.call_tool()
    async def call_tool(name, arguments):
        received.append(arguments)
        return [types.TextContent(type="text", text=json.dumps(arguments))]

    async with create_connected_server_and_client_session(server) as session:
        factory = MCPFactory(client_session=session, event_loop=asyncio.get_running_loop())
        tool = factory._create_tool_classes([MCPToolDefinition("batch", "Process unique IDs", schema, schema)])[0]()
        result = await tool.arun(tool.input_schema(tool_name="batch", values=[3, 1, 2]))
        assert result.values == [3, 1, 2]
        with pytest.raises(ValidationError):
            tool.input_schema(tool_name="batch", values=[3, 3])
        assert received == [{"values": [3, 1, 2]}]
