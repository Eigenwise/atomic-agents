import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from atomic_agents.connectors.mcp import SchemaTransformer

NUMERIC = {"oneOf": [{"type": "integer"}, {"type": "number"}]}


def check(value_schema, value):
    schema = {"type": "object", "properties": {"value": value_schema}, "required": ["value"]}
    model = SchemaTransformer.create_model_from_schema(schema, "OneOfInput", "oneof")
    payload = {"tool_name": "oneof", "value": value}
    expected = Draft202012Validator(schema).is_valid(payload)
    assert Draft202012Validator(model.model_json_schema()).is_valid(payload) == expected
    try:
        model.model_validate(payload)
        accepted = True
    except ValidationError:
        accepted = False
    assert accepted == expected


@pytest.mark.parametrize(
    "value_schema, value",
    [
        (NUMERIC, 1),
        (NUMERIC, 1.0),
        (NUMERIC, 1.5),
        (NUMERIC, "text"),
        ({"oneOf": [True, {"type": "string"}]}, "text"),
        ({"oneOf": [True, {"type": "string"}]}, {"x": 1}),
        ({"type": "array", "items": NUMERIC}, [1]),
        ({"type": "array", "items": NUMERIC}, [1.5]),
        ({"anyOf": NUMERIC["oneOf"]}, 1),
    ],
)
def test_oneof_matches_json_schema(value_schema, value):
    check(value_schema, value)


@pytest.mark.parametrize("value", [None, 1, 1.5])
def test_optional_oneof_matches_json_schema(value):
    schema = {"type": "object", "properties": {"value": NUMERIC}}
    model = SchemaTransformer.create_model_from_schema(schema, "OneOfInput", "oneof")
    payload = {"tool_name": "oneof", "value": value}
    accepted = value != 1
    assert Draft202012Validator(model.model_json_schema()).is_valid(payload) == accepted
    if accepted:
        assert model.model_validate(payload).value == value
    else:
        with pytest.raises(ValidationError):
            model.model_validate(payload)


def test_oneof_is_advertised_as_oneof():
    schema = {"type": "object", "properties": {"value": NUMERIC}, "required": ["value"]}
    model = SchemaTransformer.create_model_from_schema(schema, "OneOfInput", "oneof")
    value_schema = model.model_json_schema()["properties"]["value"]
    assert value_schema["oneOf"] == [{"type": "integer"}, {"type": "number"}]
    assert "anyOf" not in value_schema


def test_oneof_counts_json_values_not_coerced_ones():
    # Pydantic would coerce "1" to int, but in JSON it only matches the string branch.
    check({"oneOf": [{"type": "string"}, {"type": "integer"}]}, "1")


def test_oneof_ref_branches_accept_model_instances():
    schema = {
        "type": "object",
        "properties": {"value": {"oneOf": [{"$ref": "#/$defs/Named"}, {"type": "string"}]}},
        "required": ["value"],
        "$defs": {"Named": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    }
    model = SchemaTransformer.create_model_from_schema(schema, "OneOfInput", "oneof")
    first = model(tool_name="oneof", value={"name": "a"})
    assert model(tool_name="oneof", value=first.value).value.name == "a"
