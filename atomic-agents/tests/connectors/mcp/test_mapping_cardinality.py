import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from atomic_agents.connectors.mcp import SchemaTransformer


def create_model(mapping_schema, wrapper="direct", required=True):
    wrappers = {
        "direct": mapping_schema,
        "array": {"type": "array", "items": mapping_schema},
        "values": {"type": "object", "additionalProperties": mapping_schema},
        "reference": {"$ref": "#/$defs/Mapping"},
        "nullable": {"anyOf": [mapping_schema, {"type": "null"}]},
        "type-array": {**mapping_schema, "type": ["object", "null"]},
    }
    schema = {
        "type": "object",
        "properties": {"value": wrappers[wrapper]},
        "$defs": {"Mapping": mapping_schema},
    }
    if required:
        schema["required"] = ["value"]
    return SchemaTransformer.create_model_from_schema(schema, "MappingInput", "mapping")


def wrap_value(value, wrapper):
    return [value] if wrapper == "array" else {"entry": value} if wrapper == "values" else value


@pytest.mark.parametrize("wrapper", ["direct", "array", "values", "reference", "nullable", "type-array"])
@pytest.mark.parametrize(
    "keyword,bound,valid,invalid",
    [
        ("minProperties", 1, {"one": 1}, {}),
        ("maxProperties", 1, {"one": 1}, {"one": 1, "two": 2}),
        ("maxProperties", 0, {}, {"one": 1}),
    ],
)
def test_mapping_size_constraints_reach_validation_and_advertised_schema(wrapper, keyword, bound, valid, invalid):
    model = create_model({"type": "object", "additionalProperties": {"type": "integer"}, keyword: bound}, wrapper)
    valid_input = {"tool_name": "mapping", "value": wrap_value(valid, wrapper)}
    invalid_input = {"tool_name": "mapping", "value": wrap_value(invalid, wrapper)}

    assert model(**valid_input).value == valid_input["value"]
    with pytest.raises(ValidationError):
        model(**invalid_input)
    advertised = Draft202012Validator(model.model_json_schema())
    assert advertised.is_valid(valid_input)
    assert not advertised.is_valid(invalid_input)


def test_mapping_cardinality_does_not_replace_value_validation_or_optional_defaults():
    model = create_model(
        {
            "type": "object",
            "additionalProperties": {"type": "integer", "minimum": 1},
            "minProperties": 1,
            "maxProperties": 2,
        },
        required=False,
    )

    assert model(tool_name="mapping").value is None
    assert model(tool_name="mapping", value={"one": 1, "two": 2}).value == {"one": 1, "two": 2}
    for invalid in ({}, {"one": 0}, {"one": 1, "two": 2, "three": 3}):
        with pytest.raises(ValidationError):
            model(tool_name="mapping", value=invalid)


@pytest.mark.parametrize("schema", [{"type": "object"}, {"type": "object", "minProperties": 0}])
def test_unbounded_mappings_keep_empty_and_larger_values(schema):
    model = create_model(schema)

    assert model(tool_name="mapping", value={}).value == {}
    value = {str(index): index for index in range(20)}
    assert model(tool_name="mapping", value=value).value == value
