"""Keep finite JSON object/array choices in validation and model-facing schemas."""

import asyncio
import json
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest
from jsonschema import Draft202012Validator
from mcp import ClientSession
from mcp.types import CallToolResult, TextContent
from pydantic import ValidationError

from atomic_agents.connectors.mcp import MCPFactory, SchemaTransformer
from atomic_agents.connectors.mcp.mcp_definition_service import MCPToolDefinition


CHOICES = [
    pytest.param({"type": "array", "const": [1, "x", False]}, id="array-const"),
    pytest.param({"type": "object", "const": {"x": 1, "ready": True}}, id="object-const"),
    pytest.param({"const": [[1], {"x": None}]}, id="nested-array-const"),
    pytest.param({"const": {"a": []}}, id="untyped-object-const"),
    pytest.param({"type": "array", "enum": [[], [1, 2], [2, 1]]}, id="array-enum"),
    pytest.param({"type": "object", "enum": [{"x": 1}, {"x": 2}]}, id="object-enum"),
    pytest.param({"enum": [None, False, 1, "manual", [], {"x": 1}]}, id="mixed-enum"),
    pytest.param({"const": {"x": 1}, "enum": [{"x": 1}, {"x": 2}]}, id="const-enum-intersection"),
    pytest.param(
        {
            "enum": [{"x": 1}, {"x": 2}, {"x": "2"}],
            "allOf": [{"properties": {"x": {"type": "integer", "minimum": 2}}}],
        },
        id="object-allof-intersection",
    ),
    pytest.param(
        {"enum": [[], [1], [1, 2]], "oneOf": [{"minItems": 1}, {"maxItems": 1}]},
        id="array-oneof-intersection",
    ),
    pytest.param(
        {"enum": [{"x": 1}, {"x": "2"}, []], "anyOf": [{"$ref": "#/$defs/IntegerObject"}]},
        id="locally-referenced-intersection",
    ),
    pytest.param({"enum": [1, {"x": 1}], "type": "integer"}, id="scalar-survivor"),
    pytest.param({"type": ["boolean", "integer"], "enum": [True, 1]}, id="existing-scalar-enum"),
    pytest.param({"type": "string", "const": "fixed"}, id="existing-scalar-const"),
]

VALUES = [
    None,
    True,
    False,
    0,
    1,
    1.0,
    "manual",
    "fixed",
    "unknown",
    [],
    [1],
    [1, 2],
    [2, 1],
    [1, "x", False],
    [1, "x", 0],
    [[1], {"x": None}],
    [[True], {"x": None}],
    {},
    {"a": []},
    {"x": 1},
    {"x": 1.0},
    {"x": True},
    {"x": 2},
    {"x": "2"},
    {"x": 1, "extra": 0},
    {"ready": True, "x": 1},
    {"ready": 1, "x": True},
]


@pytest.mark.parametrize("prop_schema", CHOICES)
@pytest.mark.parametrize("is_output", [False, True], ids=["input", "output"])
def test_finite_choices_match_advertised_and_emitted_schemas(prop_schema, is_output):
    schema = {
        "type": "object",
        "properties": {"choice": prop_schema},
        "required": ["choice"],
        "$defs": {"IntegerObject": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}},
    }
    Draft202012Validator.check_schema(schema)
    original = deepcopy(schema)
    model = SchemaTransformer.create_model_from_schema(schema, "FiniteChoice", "choose", is_output_schema=is_output)
    advertised = Draft202012Validator(schema)
    emitted = Draft202012Validator(model.model_json_schema())

    for value in VALUES:
        payload = {"choice": deepcopy(value)}
        if not is_output:
            payload["tool_name"] = "choose"
        allowed = advertised.is_valid(payload)
        assert emitted.is_valid(payload) == allowed, value
        if allowed:
            validated = model.model_validate(payload)
            assert advertised.is_valid(json.loads(validated.model_dump_json()))
            assert model.model_validate_json(json.dumps(payload)).choice == validated.choice
        else:
            with pytest.raises(ValidationError):
                model.model_validate(payload)
            with pytest.raises(ValidationError):
                model.model_validate_json(json.dumps(payload))
    assert schema == original


@pytest.mark.parametrize(
    "prop_schema",
    [
        {"const": {"x": 1}, "type": "array"},
        {"enum": [[], {"x": 1}], "type": "string"},
        {"const": [1], "oneOf": [{"minItems": 1}, {"maxItems": 1}]},
        {"const": {"x": 1}, "enum": [{"x": 2}]},
    ],
)
def test_impossible_structured_choice_intersections_reject_definition(prop_schema):
    schema = {"type": "object", "properties": {"choice": prop_schema}, "required": ["choice"]}
    Draft202012Validator.check_schema(schema)
    with pytest.raises(ValueError, match="choices"):
        SchemaTransformer.create_model_from_schema(schema, "ImpossibleChoice", "choose")


def test_optional_defaults_and_array_items_keep_their_existing_roles():
    schema = {
        "type": "object",
        "properties": {
            "defaulted": {"enum": [{"x": 1}, {"x": 2}], "default": {"x": 2}},
            "optional": {"const": [1, 2]},
            "rows": {"type": "array", "items": {"enum": [{"x": 1}, {"x": 2}]}},
        },
        "required": ["rows"],
    }
    model = SchemaTransformer.create_model_from_schema(schema, "NestedChoice", "choose")
    result = model(tool_name="choose", rows=[{"x": 1}])
    assert result.defaulted == {"x": 2}
    assert result.optional is None
    assert result.rows == [{"x": 1}]
    assert model(tool_name="choose", optional=None, rows=[]).optional is None
    assert Draft202012Validator(model.model_json_schema()).is_valid(result.model_dump())
    for override in [{"defaulted": {"x": 3}}, {"optional": [2, 1]}, {"rows": [{"x": True}]}]:
        with pytest.raises(ValidationError):
            model.model_validate({"tool_name": "choose", "rows": [], **override})


@pytest.mark.parametrize("with_properties", [False, True], ids=["const-only", "const-with-properties"])
def test_referenced_structured_choices_keep_the_reference_constraints(with_properties):
    schema = {
        "type": "object",
        "properties": {"choice": {"$ref": "#/$defs/Preset"}},
        "required": ["choice"],
        "$defs": {"Preset": {"const": {"values": [False, 1]}}},
    }
    if with_properties:
        schema["$defs"]["Preset"].update(type="object", properties={"values": {"type": "array"}}, required=["values"])
    model = SchemaTransformer.create_model_from_schema(schema, "ReferencedChoice", "choose")
    assert model(tool_name="choose", choice={"values": [False, 1]}).model_dump()["choice"] == {"values": [False, 1]}
    with pytest.raises(ValidationError):
        model(tool_name="choose", choice={"values": [0, True]})
    emitted = Draft202012Validator(model.model_json_schema())
    assert not emitted.is_valid({"tool_name": "choose", "choice": {"values": [0, True]}})


@pytest.mark.asyncio
@pytest.mark.parametrize("valid_output", [True, False], ids=["valid-output", "invalid-output"])
async def test_generated_tools_validate_and_preserve_structured_choices(valid_output):
    input_schema = {
        "type": "object",
        "properties": {"choice": {"enum": [{"format": "json", "fields": ["x"]}, {"format": "text"}]}},
        "required": ["choice"],
    }
    output_schema = {
        "type": "object",
        "properties": {"receipt": {"const": {"ack": [True, 1]}}},
        "required": ["receipt"],
    }
    session = AsyncMock(spec=ClientSession)
    result = {"receipt": {"ack": [True, 1] if valid_output else [1, True]}}
    session.call_tool.return_value = CallToolResult(content=[TextContent(type="text", text=json.dumps(result))])
    factory = MCPFactory(client_session=session, event_loop=asyncio.get_running_loop())
    (tool_class,) = factory._create_tool_classes(
        [
            MCPToolDefinition(
                name="choose", description="Select an exact preset.", input_schema=input_schema, output_schema=output_schema
            )
        ]
    )
    with pytest.raises(ValidationError):
        tool_class.input_schema(tool_name="choose", choice={"format": "unknown"})
    session.call_tool.assert_not_awaited()

    arguments = {"choice": {"fields": ["x"], "format": "json"}}
    params = tool_class.input_schema(tool_name="choose", **arguments)
    if valid_output:
        assert (await tool_class().arun(params)).model_dump() == result
    else:
        with pytest.raises(RuntimeError) as error:
            await tool_class().arun(params)
        assert isinstance(error.value.__cause__, ValidationError)
    session.call_tool.assert_awaited_once_with(name="choose", arguments=arguments)
