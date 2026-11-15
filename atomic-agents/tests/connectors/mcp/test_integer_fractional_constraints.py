"""Integer JSON Schema bounds need not themselves be integers."""

import copy
import json
from datetime import timedelta

import pytest
from jsonschema import Draft202012Validator
from mcp import types
from mcp.server import Server
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import ValidationError

from atomic_agents.connectors.mcp import SchemaTransformer
from atomic_agents.connectors.mcp.mcp_factory import fetch_mcp_tools_async


CASES = [
    ({"minimum": 1.5}, [2, 3], [0, 1]),
    ({"minimum": -1.5}, [-1, 0], [-3, -2]),
    ({"maximum": 1.5}, [0, 1], [2, 3]),
    ({"maximum": -1.5}, [-3, -2], [-1, 0]),
    ({"exclusiveMinimum": 1.5}, [2, 3], [0, 1]),
    ({"exclusiveMinimum": -1.5}, [-1, 0], [-3, -2]),
    ({"exclusiveMaximum": 1.5}, [0, 1], [2, 3]),
    ({"exclusiveMaximum": -1.5}, [-3, -2], [-1, 0]),
    ({"multipleOf": 0.25}, [-2, 0, 1, 2], []),
    ({"multipleOf": 1.5}, [-3, 0, 3, 6], [-2, -1, 1, 2, 4]),
    ({"multipleOf": 2.5}, [-5, 0, 5, 10], [-4, -1, 1, 2, 4, 6]),
    ({"minimum": 0.5, "maximum": 9.5, "multipleOf": 1.5}, [3, 6, 9], [0, 1, 4, 10, 12]),
]


def make_model(constraints, *, referenced=False, output=False):
    value_schema = {"type": "integer", **constraints}
    schema = {
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/Value"} if referenced else value_schema},
        "required": ["value"],
        "$defs": {"Value": value_schema},
    }
    original = copy.deepcopy(schema)
    model = SchemaTransformer.create_model_from_schema(schema, "IntegerBounds", "integer_bounds", is_output_schema=output)
    assert schema == original
    return model, schema


@pytest.mark.parametrize("referenced,output", [(False, False), (True, False), (False, True)])
@pytest.mark.parametrize("constraints,valid,invalid", CASES)
def test_fractional_integer_bounds_validate_and_remain_advertised(constraints, valid, invalid, referenced, output):
    model, original = make_model(constraints, referenced=referenced, output=output)
    base_payload = {} if output else {"tool_name": "integer_bounds"}
    original_validator = Draft202012Validator(original)
    emitted_validator = Draft202012Validator(model.model_json_schema())
    for value in valid:
        payload = {**base_payload, "value": value}
        assert original_validator.is_valid(payload)
        assert emitted_validator.is_valid(payload)
        result = model.model_validate(payload)
        assert type(result.value) is int
        assert result.value == value
        assert result.model_dump()["value"] == value
    for value in invalid:
        payload = {**base_payload, "value": value}
        assert not original_validator.is_valid(payload)
        assert not emitted_validator.is_valid(payload)
        with pytest.raises(ValidationError):
            model.model_validate(payload)


def test_fractional_bounds_keep_integer_coercion_and_type_checks():
    model, _ = make_model({"minimum": 1.5, "maximum": 3.5})
    assert model(tool_name="integer_bounds", value="2").value == 2
    for value in ["1", 2.5, "not an integer"]:
        with pytest.raises(ValidationError):
            model(tool_name="integer_bounds", value=value)


def test_fractional_bounds_do_not_round_large_integer_values():
    model, _ = make_model({"multipleOf": 2.5})
    valid = 10**30
    assert model(tool_name="integer_bounds", value=valid).value == valid
    with pytest.raises(ValidationError):
        model(tool_name="integer_bounds", value=valid + 1)


@pytest.mark.parametrize("keyword", ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"])
def test_integral_float_bound_uses_the_existing_integer_semantics(keyword):
    model, original = make_model({keyword: 2.0})
    validator = Draft202012Validator(original)
    for value in range(-4, 5):
        payload = {"tool_name": "integer_bounds", "value": value}
        if validator.is_valid(payload):
            assert model.model_validate(payload).value == value
        else:
            with pytest.raises(ValidationError):
                model.model_validate(payload)


@pytest.mark.asyncio
async def test_fractional_integer_tool_stays_discoverable_and_callable():
    server = Server("fractional-integer-bounds")
    calls = []
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer", "minimum": 1.5, "maximum": 9.5, "multipleOf": 1.5}},
        "required": ["value"],
    }

    @server.list_tools()
    async def list_tools():
        return [types.Tool(name="integer_bounds", inputSchema=schema)]

    @server.call_tool()
    async def call_tool(name, arguments):
        assert name == "integer_bounds"
        Draft202012Validator(schema).validate(arguments)
        calls.append(arguments)
        return [types.TextContent(type="text", text=json.dumps(arguments))]

    async with create_connected_server_and_client_session(server, read_timeout_seconds=timedelta(seconds=3)) as session:
        tools = await fetch_mcp_tools_async(client_session=session)
        assert len(tools) == 1
        tool = tools[0]()
        for value in [3, 6, 9]:
            result = await tool.arun(tool.input_schema(tool_name="integer_bounds", value=value))
            assert json.loads(result.result[0].text) == {"value": value}
        with pytest.raises(ValidationError):
            tool.input_schema(tool_name="integer_bounds", value=4)
        assert calls == [{"value": value} for value in [3, 6, 9]]
        await session.send_ping()
