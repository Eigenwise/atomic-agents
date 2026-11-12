import pytest
from pydantic import ValidationError

from atomic_agents.connectors.mcp import SchemaTransformer


CONSTRAINT_CASES = [
    ({"type": "integer", "minimum": 0}, 0, -1),
    ({"type": "integer", "maximum": 10}, 10, 11),
    ({"type": "number", "exclusiveMinimum": 0}, 0.5, 0),
    ({"type": "number", "exclusiveMaximum": 10}, 9.5, 10),
    ({"type": "number", "multipleOf": 0.25}, 1.25, 1.1),
    ({"type": "string", "minLength": 2}, "é字", "é"),
    ({"type": "string", "maxLength": 2}, "é字", "é字x"),
    ({"type": "string", "pattern": "(?<=prefix:)[a-z]+$"}, "prefix:hello", "other:hello"),
    ({"type": "array", "items": {"type": "integer"}, "minItems": 2}, [1, 2], [1]),
    ({"type": "array", "items": {"type": "integer"}, "maxItems": 2}, [1, 2], [1, 2, 3]),
]


def build_model(prop_schema, definitions=None):
    schema = {"type": "object", "properties": {"value": prop_schema}, "required": ["value"]}
    if definitions:
        schema["$defs"] = definitions
    return SchemaTransformer.create_model_from_schema(schema, "ConstrainedInput", "constrained")


@pytest.mark.parametrize("prop_schema,valid,invalid", CONSTRAINT_CASES)
def test_property_constraints_validate_values(prop_schema, valid, invalid):
    model = build_model(prop_schema)
    assert model(tool_name="constrained", value=valid).value == valid
    with pytest.raises(ValidationError):
        model(tool_name="constrained", value=invalid)


@pytest.mark.parametrize("prop_schema,valid,invalid", CONSTRAINT_CASES)
def test_property_constraints_are_advertised_to_models(prop_schema, valid, invalid):
    model = build_model(prop_schema)
    advertised = model.model_json_schema()["properties"]["value"]
    for key, value in prop_schema.items():
        if key not in {"type", "items"}:
            assert advertised[key] == value


@pytest.mark.parametrize(
    "prop_schema,valid,invalid",
    [
        ({"type": "array", "items": {"type": "integer", "minimum": 2}}, [2, 3], [1, 2]),
        ({"type": "object", "additionalProperties": {"type": "integer", "minimum": 2}}, {"n": 2}, {"n": 1}),
        ({"$ref": "#/$defs/Bounded"}, 2, 1),
        ({"anyOf": [{"type": "integer", "minimum": 2}, {"type": "null"}]}, 2, 1),
        ({"type": ["integer", "null"], "minimum": 2}, 2, 1),
    ],
)
def test_constraints_survive_recursive_conversion(prop_schema, valid, invalid):
    model = build_model(prop_schema, {"Bounded": {"type": "integer", "minimum": 2}})
    assert model(tool_name="constrained", value=valid).value == valid
    with pytest.raises(ValidationError):
        model(tool_name="constrained", value=invalid)


def test_numeric_constraints_apply_after_existing_type_coercion():
    model = build_model({"type": "integer", "minimum": 5})
    assert model(tool_name="constrained", value="5").value == 5
    with pytest.raises(ValidationError):
        model(tool_name="constrained", value="4")


def test_multiple_of_preserves_floating_point_tolerance():
    model = build_model({"type": "number", "multipleOf": 0.1})
    assert model(tool_name="constrained", value=0.3).value == 0.3


def test_optional_constrained_field_can_be_omitted():
    model = SchemaTransformer.create_model_from_schema(
        {"type": "object", "properties": {"value": {"type": "integer", "minimum": 2}}},
        "OptionalConstrainedInput",
        "constrained",
    )
    assert model(tool_name="constrained").value is None
    with pytest.raises(ValidationError):
        model(tool_name="constrained", value=1)
