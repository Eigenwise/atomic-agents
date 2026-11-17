from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from atomic_agents.connectors.mcp import SchemaTransformer


def dependency_model(wrapper):
    dependent = {
        "type": "object",
        "properties": {"option": {}, "details": {}},
        "dependentRequired": {"option": ["details"]},
    }
    value_schemas = {
        "inline": dependent,
        "reference": {"$ref": "#/$defs/Dependency"},
        "array": {"type": "array", "items": dependent},
        "mapping": {"type": "object", "additionalProperties": dependent},
        "nullable": {"anyOf": [dependent, {"type": "null"}]},
        "type-array": {**dependent, "type": ["object", "null"]},
        "untyped": {"dependentRequired": dependent["dependentRequired"]},
    }
    if wrapper == "top":
        schema = dependent
    else:
        schema = {
            "type": "object",
            "properties": {"value": value_schemas[wrapper]},
            "required": ["value"],
            "$defs": {"Dependency": dependent},
        }
    model = SchemaTransformer.create_model_from_schema(schema, "DependentInput", "configure")
    return model, schema


def wrap(value, wrapper):
    if wrapper == "top":
        return {"tool_name": "configure", **value}
    nested = [value] if wrapper == "array" else {"item": value} if wrapper == "mapping" else value
    return {"tool_name": "configure", "value": nested}


@pytest.mark.parametrize("wrapper", ["top", "inline", "reference", "array", "mapping", "nullable", "type-array", "untyped"])
@pytest.mark.parametrize("trigger", [None, False, 0, ""])
def test_present_properties_require_dependencies_even_when_falsey(wrapper, trigger):
    model, original = dependency_model(wrapper)
    valid = wrap({"option": trigger, "details": None}, wrapper)
    missing = wrap({"option": trigger}, wrapper)
    oracle = Draft202012Validator(original)
    assert oracle.is_valid(valid)
    assert not oracle.is_valid(missing)

    model.model_validate(valid)
    with pytest.raises(ValidationError, match="details.*option"):
        model.model_validate(missing)

    advertised = Draft202012Validator(model.model_json_schema())
    assert advertised.is_valid(valid)
    assert not advertised.is_valid(missing)


@pytest.mark.parametrize("wrapper", ["top", "reference", "inline"])
def test_dependencies_are_directional_and_checked_before_model_defaults(wrapper):
    model, _ = dependency_model(wrapper)

    model.model_validate(wrap({}, wrapper))
    model.model_validate(wrap({"details": "standalone"}, wrapper))

    assert Draft202012Validator(model.model_json_schema()).is_valid(wrap({}, wrapper))


def test_output_models_enforce_transitive_dependencies_and_keep_field_validation():
    schema = {
        "type": "object",
        "properties": {"start": {"type": "integer"}, "end": {"type": "integer"}, "unit": {"type": "string"}},
        "dependentRequired": {"start": ["end"], "end": ["unit"]},
    }
    model = SchemaTransformer.create_model_from_schema(schema, "RangeOutput", "range", is_output_schema=True)

    assert model(start=0, end=10, unit="seconds").end == 10
    for value in ({"start": 0}, {"start": 0, "end": 10}, {"end": 10}, {"start": 0, "end": "bad", "unit": "seconds"}):
        with pytest.raises(ValidationError):
            model.model_validate(value)
        assert not Draft202012Validator(model.model_json_schema()).is_valid(value)


def test_defaulted_dependency_must_still_be_present_in_the_submitted_object():
    schema = {
        "type": "object",
        "properties": {"option": {}, "details": {"type": "string", "default": "default details"}},
        "dependentRequired": {"option": ["details"]},
    }
    model = SchemaTransformer.create_model_from_schema(schema, "DefaultedInput", "configure")

    assert model(tool_name="configure").details == "default details"
    with pytest.raises(ValidationError, match="details.*option"):
        model(tool_name="configure", option=True)


def test_dependency_rules_are_snapshotted_with_the_advertised_schema():
    model, original = dependency_model("top")
    advertised = deepcopy(model.model_json_schema())
    original["dependentRequired"]["option"].append("later")

    model(tool_name="configure", option=True, details="provided")
    assert model.model_json_schema() == advertised


@pytest.mark.parametrize("value", [None, "text", 42, ["option"]])
def test_dependency_only_schemas_leave_non_objects_alone(value):
    model, _ = dependency_model("untyped")

    assert model(tool_name="configure", value=value).value == value
    assert Draft202012Validator(model.model_json_schema()).is_valid({"tool_name": "configure", "value": value})
