"""Explicit object extension policies survive MCP model conversion."""

import asyncio
import json

import pytest
from jsonschema import Draft202012Validator
from mcp import types
from mcp.server import Server
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import ValidationError

from atomic_agents.connectors.mcp import MCPFactory, MCPToolDefinition, SchemaTransformer


def object_schema(extra):
    return {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": extra,
    }


def model_for(schema, nested):
    if nested:
        schema = {
            "type": "object",
            "properties": {"record": {"$ref": "#/$defs/Record"}},
            "required": ["record"],
            "$defs": {"Record": schema},
        }
    return SchemaTransformer.create_model_from_schema(schema, "ExtensionOutput", "extension", is_output_schema=True)


@pytest.mark.parametrize("nested", [False, True], ids=["root", "referenced"])
@pytest.mark.parametrize("extra", [True, {}, {"type": "integer"}])
def test_explicit_open_objects_preserve_extension_values(nested, extra):
    model = model_for(object_schema(extra), nested)
    record = {"name": "result", "count": 7, "_counter": 4}
    value = {"record": record} if nested else record
    assert model.model_validate(value).model_dump() == value
    emitted = model.model_json_schema()
    assert Draft202012Validator(emitted).is_valid(value)
    obj = emitted["$defs"]["Record"] if nested else emitted
    assert obj["additionalProperties"] == (True if extra == {} else extra)


@pytest.mark.parametrize("nested", [False, True], ids=["root", "referenced"])
@pytest.mark.parametrize("extra", [False, {"type": "integer"}])
def test_explicit_object_policy_rejects_invalid_extension_values(nested, extra):
    model = model_for(object_schema(extra), nested)
    record = {"name": "result", "count": "not-an-integer"}
    value = {"record": record} if nested else record
    with pytest.raises(ValidationError):
        model.model_validate(value)
    assert not Draft202012Validator(model.model_json_schema()).is_valid(value)


def test_extra_values_can_reference_structured_records():
    schema = object_schema({"$ref": "#/$defs/Count"})
    schema["$defs"] = {"Count": {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]}}
    model = model_for(schema, False)
    value = {"name": "result", "first": {"value": 3}, "second": {"value": 9}}
    assert model.model_validate(value).model_dump() == value
    assert Draft202012Validator(model.model_json_schema()).is_valid(value)
    with pytest.raises(ValidationError):
        model.model_validate({"name": "result", "first": {"value": "invalid"}})


@pytest.mark.parametrize("extra", [False, True, {"type": "integer"}])
def test_explicit_policy_does_not_make_named_fields_optional(extra):
    model = model_for(object_schema(extra), False)
    assert model(name="result").model_dump() == {"name": "result"}
    with pytest.raises(ValidationError):
        model.model_validate({})


def test_closed_empty_objects_reject_unadvertised_arguments():
    model = model_for({"type": "object", "additionalProperties": False}, False)
    assert model().model_dump() == {}
    with pytest.raises(ValidationError):
        model(unadvertised=1)


def test_unspecified_object_policy_keeps_existing_behavior():
    schema = object_schema(True)
    del schema["additionalProperties"]
    model = model_for(schema, False)
    assert model(name="result", ignored=1).model_dump() == {"name": "result"}


@pytest.mark.asyncio
@pytest.mark.parametrize("structured", [False, True], ids=["json-text", "structured"])
async def test_extra_arguments_and_output_cross_real_mcp_transport(structured):
    if structured and "structuredContent" not in types.CallToolResult.model_fields:
        pytest.skip("MCP version predates structured tool output")
    schema = object_schema({"type": "integer"})
    server = Server("extension-regression")
    received = []

    @server.list_tools()
    async def list_tools():
        return [types.Tool(name="extend", inputSchema=schema, outputSchema=schema if structured else None)]

    @server.call_tool()
    async def call_tool(name, arguments):
        received.append(arguments)
        output = {**arguments, "returned": 11}
        if structured:
            return types.CallToolResult(content=[], structuredContent=output)
        return [types.TextContent(type="text", text=json.dumps(output))]

    async with create_connected_server_and_client_session(server) as session:
        factory = MCPFactory(client_session=session, event_loop=asyncio.get_running_loop())
        tool = factory._create_tool_classes([MCPToolDefinition("extend", "Extend a record", schema, schema)])[0]()
        value = {"name": "sample", "count": 7, "_counter": 4}
        result = await tool.arun(tool.input_schema(tool_name="extend", **value))
        assert received == [value]
        assert result.model_dump() == {**value, "returned": 11}
