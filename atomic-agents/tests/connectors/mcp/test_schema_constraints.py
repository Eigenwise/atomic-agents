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
    ({"type": "string", "pattern": "^[a-z]+$"}, "hello", "hello123"),
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


@pytest.mark.parametrize("pattern", ["(?<=prefix:)[a-z]+$", "["])
def test_unsupported_pattern_preserves_tool_and_other_constraints(pattern, caplog):
    model = build_model({"type": "string", "pattern": pattern, "minLength": 2})
    assert model(tool_name="constrained", value="other:hello").value == "other:hello"
    assert "Unsupported JSON Schema pattern" in caplog.text
    assert "pattern" not in model.model_json_schema()["properties"]["value"]
    with pytest.raises(ValidationError):
        model(tool_name="constrained", value="x")


def test_unicode_property_pattern_builds_and_validates():
    model = build_model({"type": "string", "pattern": r"^\p{L}+$"})
    assert model(tool_name="constrained", value="é字").value == "é字"
    with pytest.raises(ValidationError):
        model(tool_name="constrained", value="é字1")
    assert model.model_json_schema()["properties"]["value"]["pattern"] == r"^\p{L}+$"


@pytest.mark.parametrize("keyword", ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"])
@pytest.mark.parametrize("invalid_bound", [True, False, "5"])
def test_nonnumeric_bounds_do_not_become_numeric_constraints(keyword, invalid_bound):
    model = build_model({"type": "number", keyword: invalid_bound})
    for value in [-2, 0, 2.5]:
        assert model(tool_name="constrained", value=value).value == value
    assert keyword not in model.model_json_schema()["properties"]["value"]


@pytest.mark.parametrize("exclusive", [True, False])
def test_draft04_boolean_exclusivity_does_not_override_numeric_bounds(exclusive):
    model = build_model(
        {"type": "number", "minimum": 5, "maximum": 10, "exclusiveMinimum": exclusive, "exclusiveMaximum": exclusive}
    )
    assert model(tool_name="constrained", value=5).value == 5
    assert model(tool_name="constrained", value=10).value == 10
    with pytest.raises(ValidationError):
        model(tool_name="constrained", value=4)
    with pytest.raises(ValidationError):
        model(tool_name="constrained", value=11)


def test_nested_quantifier_pattern_rejects_long_input():
    model = build_model({"type": "string", "pattern": "^(a+)+$"})
    with pytest.raises(ValidationError):
        model(tool_name="constrained", value="a" * 5000 + "!")
