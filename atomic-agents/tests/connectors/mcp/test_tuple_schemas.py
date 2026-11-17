import json

import pytest
from jsonschema import Draft202012Validator
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import ValidationError

from atomic_agents.connectors.mcp import SchemaTransformer, fetch_mcp_tools_async


def fixed_tuple_schema(closed_with="maxItems"):
    schema = {
        "type": "array",
        "prefixItems": [{"type": "integer"}, {"type": "string", "enum": ["east", "north"]}],
        "minItems": 2,
    }
    schema[closed_with] = 2 if closed_with == "maxItems" else False
    return schema


def make_model(field_schema, output=False, definitions=None):
    schema = {"type": "object", "properties": {"point": field_schema}, "required": ["point"]}
    if definitions:
        schema["$defs"] = definitions
    model = SchemaTransformer.create_model_from_schema(schema, "Location", "locate", is_output_schema=output)
    return schema, model


@pytest.mark.parametrize("closed_with", ["maxItems", "items"])
@pytest.mark.parametrize("output", [False, True])
@pytest.mark.parametrize("point", [["invalid", "east"], [1, "west"], [1], [1, "east", 2]])
def test_fixed_tuple_rejects_wrong_positions_and_lengths(closed_with, output, point):
    schema, model = make_model(fixed_tuple_schema(closed_with), output)
    payload = {"point": point}
    if not output:
        payload["tool_name"] = "locate"

    assert not Draft202012Validator(schema).is_valid(payload)
    with pytest.raises(ValidationError):
        model.model_validate(payload)
    assert not Draft202012Validator(model.model_json_schema()).is_valid(payload)


@pytest.mark.parametrize("closed_with", ["maxItems", "items"])
@pytest.mark.parametrize("output", [False, True])
def test_fixed_tuple_preserves_valid_json_and_advertised_positions(closed_with, output):
    schema, model = make_model(fixed_tuple_schema(closed_with), output)
    payload = {"point": [7, "east"]}
    if not output:
        payload["tool_name"] = "locate"

    assert Draft202012Validator(schema).is_valid(payload)
    value = model.model_validate(payload)
    assert json.loads(value.model_dump_json()) == payload
    emitted = model.model_json_schema()
    assert Draft202012Validator(emitted).is_valid(payload)
    assert emitted["properties"]["point"]["prefixItems"][0]["type"] == "integer"
    assert emitted["properties"]["point"]["prefixItems"][1]["enum"] == ["east", "north"]


@pytest.mark.parametrize("wrapper", ["array", "mapping", "nullable"])
def test_fixed_tuple_is_converted_recursively(wrapper):
    field = fixed_tuple_schema()
    if wrapper == "array":
        field = {"type": "array", "items": field}
        valid, invalid = [[7, "east"]], [["invalid", "east"]]
    elif wrapper == "mapping":
        field = {"type": "object", "additionalProperties": field}
        valid, invalid = {"home": [7, "east"]}, {"home": ["invalid", "east"]}
    else:
        field = {"anyOf": [field, {"type": "null"}]}
        valid, invalid = [7, "east"], ["invalid", "east"]

    _, model = make_model(field, output=True)
    assert json.loads(model(point=valid).model_dump_json()) == {"point": valid}
    with pytest.raises(ValidationError):
        model(point=invalid)
    assert not Draft202012Validator(model.model_json_schema()).is_valid({"point": invalid})
    if wrapper == "nullable":
        assert model(point=None).point is None


def test_fixed_tuple_position_resolves_local_reference():
    field = fixed_tuple_schema()
    field["prefixItems"][0] = {"$ref": "#/$defs/Coordinate"}
    _, model = make_model(field, output=True, definitions={"Coordinate": {"type": "integer"}})
    assert json.loads(model(point=[7, "east"]).model_dump_json()) == {"point": [7, "east"]}
    with pytest.raises(ValidationError):
        model(point=["invalid", "east"])


def test_fixed_tuple_does_not_apply_the_tail_schema_to_prefix_positions():
    field = fixed_tuple_schema()
    field["items"] = {"type": "boolean"}
    schema, model = make_model(field, output=True)
    payload = {"point": [7, "east"]}
    assert Draft202012Validator(schema).is_valid(payload)
    assert json.loads(model.model_validate(payload).model_dump_json()) == payload


def test_closed_tuple_with_contradictory_bounds_does_not_become_valid():
    field = fixed_tuple_schema("items")
    field["maxItems"] = 1
    schema, model = make_model(field, output=True)
    payload = {"point": [7, "east"]}
    assert not Draft202012Validator(schema).is_valid(payload)
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_homogeneous_arrays_keep_list_values_and_item_validation():
    _, model = make_model({"type": "array", "items": {"type": "integer"}}, output=True)
    assert model(point=[1, 2, 3]).point == [1, 2, 3]
    with pytest.raises(ValidationError):
        model(point=[1, "invalid"])


@pytest.mark.asyncio
async def test_fastmcp_fixed_tuple_survives_discovery_and_a_real_call():
    server = FastMCP("tuple-arguments")
    received = []

    @server.tool()
    def locate(point: tuple[int, str]) -> str:
        """Render a numbered direction."""
        received.append(point)
        return f"{point[0]}:{point[1]}"

    async with create_connected_server_and_client_session(server._mcp_server) as session:
        tools = await fetch_mcp_tools_async(client_session=session)
        tool = tools[0]()
        params = tool.input_schema(tool_name="locate", point=[7, "east"])
        result = await tool.arun(params)
        # Newer SDKs advertise structured string output; older SDKs return text blocks.
        text = result.result if isinstance(result.result, str) else result.result[0].text
        assert text == "7:east"
        assert received == [(7, "east")]

        with pytest.raises(ValidationError):
            tool.input_schema(tool_name="locate", point=["invalid", "east"])
        assert received == [(7, "east")]
        await session.send_ping()
