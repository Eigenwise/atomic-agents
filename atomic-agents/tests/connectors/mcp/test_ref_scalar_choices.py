"""Keep finite scalar choices that narrow an MCP schema reference."""

from enum import Enum
from typing import Annotated

import pytest
from jsonschema import Draft202012Validator
from mcp.server.fastmcp import FastMCP
from pydantic import Field, ValidationError

from atomic_agents.connectors.mcp import SchemaTransformer


class Size(str, Enum):
    SMALL = "small"
    LARGE = "large"


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", [{"enum": ["small"]}, {"const": "small"}])
async def test_fastmcp_enum_reference_retains_field_choices(choice):
    server = FastMCP("reference-choices")

    @server.tool()
    def reserve(size: Annotated[Size, Field(json_schema_extra=choice)]) -> str:
        """Reserve the advertised size."""
        return size.value

    definition = (await server.list_tools())[0]
    assert definition.inputSchema["properties"]["size"]["$ref"] == "#/$defs/Size"
    model = SchemaTransformer.create_model_from_schema(definition.inputSchema, "ReserveInput", "reserve")
    advertised = Draft202012Validator(definition.inputSchema)
    emitted = Draft202012Validator(model.model_json_schema())

    allowed = {"tool_name": "reserve", "size": "small"}
    forbidden = {"tool_name": "reserve", "size": "large"}
    assert advertised.is_valid(allowed)
    assert not advertised.is_valid(forbidden)
    assert model.model_validate(allowed).size == "small"
    with pytest.raises(ValidationError):
        model.model_validate(forbidden)
    assert emitted.is_valid(allowed)
    assert not emitted.is_valid(forbidden)


@pytest.mark.parametrize("required", [True, False])
@pytest.mark.parametrize(
    "definition, choice",
    [
        ({"type": "integer", "minimum": 1, "maximum": 3}, {"enum": [0, 1, 2, 4]}),
        ({"type": "integer"}, {"const": 1}),
        ({"type": ["boolean", "integer"]}, {"enum": [True]}),
        ({"type": ["string", "null"]}, {"enum": ["small", None]}),
        ({"type": ["string", "null"]}, {"const": None}),
        ({"$ref": "#/$defs/Text"}, {"enum": ["small", "invalid"]}),
    ],
)
def test_reference_and_choices_agree_in_validation_and_emitted_schema(definition, choice, required):
    schema = {
        "type": "object",
        "$defs": {"Value": definition, "Text": {"type": "string", "maxLength": 5}},
        "properties": {"value": {"$ref": "#/$defs/Value", **choice}},
        "required": ["value"] if required else [],
    }
    model = SchemaTransformer.create_model_from_schema(schema, "ReferenceChoice", "choose")
    advertised = Draft202012Validator(schema)
    emitted = Draft202012Validator(model.model_json_schema())
    for value in [None, False, True, 0, 1, 1.0, 2, 3, 4, "small", "large", "invalid", [], {}]:
        payload = {"tool_name": "choose", "value": value}
        # Omission and optional None keep the transformer's existing behavior.
        allowed = advertised.is_valid(payload) or (not required and value is None)
        assert emitted.is_valid(payload) == allowed
        if allowed:
            assert model.model_validate(payload).value == value
        else:
            with pytest.raises(ValidationError):
                model.model_validate(payload)
    if not required:
        assert model(tool_name="choose").value is None


@pytest.mark.parametrize("container", ["array", "mapping", "reference"])
def test_choices_are_preserved_in_nested_values(container):
    choice = {"$ref": "#/$defs/Text", "const": "small"}
    shapes = {
        "array": ({"type": "array", "items": choice}, ["small"], ["large"]),
        "mapping": ({"type": "object", "additionalProperties": choice}, {"key": "small"}, {"key": "large"}),
        "reference": ({"$ref": "#/$defs/Nested"}, {"size": "small"}, {"size": "large"}),
    }
    prop, allowed, forbidden = shapes[container]
    schema = {
        "type": "object",
        "$defs": {
            "Text": {"type": "string"},
            "Nested": {"type": "object", "properties": {"size": choice}, "required": ["size"]},
        },
        "properties": {"value": prop},
        "required": ["value"],
    }
    model = SchemaTransformer.create_model_from_schema(schema, "NestedChoice", "choose", is_output_schema=True)
    assert model.model_validate({"value": allowed}).model_dump() == {"value": allowed}
    with pytest.raises(ValidationError):
        model.model_validate({"value": forbidden})
    emitted = Draft202012Validator(model.model_json_schema())
    assert emitted.is_valid({"value": allowed})
    assert not emitted.is_valid({"value": forbidden})


@pytest.mark.parametrize("choice", [{"enum": ["bad"]}, {"const": None}])
def test_reference_with_no_admissible_choice_rejects_definition(choice):
    schema = {
        "type": "object",
        "$defs": {"Integer": {"type": "integer"}},
        "properties": {"value": {"$ref": "#/$defs/Integer", **choice}},
        "required": ["value"],
    }
    with pytest.raises(ValueError, match="choices"):
        SchemaTransformer.create_model_from_schema(schema, "ImpossibleChoice", "choose")


@pytest.mark.parametrize("reference", ["#/$defs/Missing", "https://example.invalid/schema.json"])
def test_unresolved_reference_retains_known_choices(reference):
    schema = {
        "type": "object",
        "properties": {"value": {"$ref": reference, "enum": [1, "bad"], "type": "integer"}},
        "required": ["value"],
    }
    model = SchemaTransformer.create_model_from_schema(schema, "UnresolvedChoice", "choose")
    assert model(tool_name="choose", value=1).value == 1
    for value in [2, "bad", True]:
        with pytest.raises(ValidationError):
            model(tool_name="choose", value=value)
    assert not Draft202012Validator(model.model_json_schema()).is_valid({"tool_name": "choose", "value": 2})


def test_plain_reference_keeps_existing_coercion():
    schema = {
        "type": "object",
        "$defs": {"Integer": {"type": "integer", "minimum": 1}},
        "properties": {"value": {"$ref": "#/$defs/Integer"}},
        "required": ["value"],
    }
    model = SchemaTransformer.create_model_from_schema(schema, "PlainReference", "choose")
    assert model(tool_name="choose", value="2").value == 2
    with pytest.raises(ValidationError):
        model(tool_name="choose", value=0)
