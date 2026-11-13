import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from atomic_agents.connectors.mcp import SchemaTransformer


def make_model(value_schema, required=True, definitions=None):
    schema = {
        "type": "object",
        "properties": {"value": value_schema},
        "required": ["value"] if required else [],
        "$defs": definitions or {},
    }
    return SchemaTransformer.create_model_from_schema(schema, "BooleanInput", "boolean"), schema


@pytest.mark.parametrize("required", [True, False])
@pytest.mark.parametrize("value", [None, False, 1, "text", [1, None], {"x": True}])
def test_true_property_accepts_every_json_value(required, value):
    model, original = make_model(True, required)
    payload = {"tool_name": "boolean", "value": value}
    assert Draft202012Validator(original).is_valid(payload)
    assert Draft202012Validator(model.model_json_schema()).is_valid(payload)
    assert model.model_validate(payload).value == value


@pytest.mark.parametrize("required", [True, False])
@pytest.mark.parametrize("value", [None, False, 1, "text", [], {}])
def test_false_property_rejects_every_present_json_value(required, value):
    model, original = make_model(False, required)
    payload = {"tool_name": "boolean", "value": value}
    assert not Draft202012Validator(original).is_valid(payload)
    assert not Draft202012Validator(model.model_json_schema()).is_valid(payload)
    with pytest.raises(ValidationError):
        model.model_validate(payload)


@pytest.mark.parametrize("value_schema", [True, False])
def test_optional_boolean_property_can_be_omitted(value_schema):
    model, _ = make_model(value_schema, required=False)
    result = model(tool_name="boolean")
    assert result.model_dump(exclude_unset=True) == {"tool_name": "boolean"}
    assert Draft202012Validator(model.model_json_schema()).is_valid({"tool_name": "boolean"})


@pytest.mark.parametrize("value_schema", [True, False])
def test_required_boolean_property_cannot_be_omitted(value_schema):
    model, _ = make_model(value_schema)
    with pytest.raises(ValidationError):
        model(tool_name="boolean")


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf"])
def test_false_union_member_keeps_the_other_branch(keyword):
    model, _ = make_model({keyword: [False, {"type": "string"}]})
    assert model(tool_name="boolean", value="text").value == "text"
    with pytest.raises(ValidationError):
        model(tool_name="boolean", value={})
    assert not Draft202012Validator(model.model_json_schema()).is_valid({"tool_name": "boolean", "value": {}})


def test_true_union_member_accepts_other_values():
    model, _ = make_model({"anyOf": [True, {"type": "string"}]})
    assert model(tool_name="boolean", value={"x": 1}).value == {"x": 1}


@pytest.mark.parametrize("items, valid, invalid", [(True, [1, None, "x"], None), (False, [], [None])])
def test_boolean_array_items(items, valid, invalid):
    model, _ = make_model({"type": "array", "items": items})
    assert model(tool_name="boolean", value=valid).value == valid
    if invalid is not None:
        with pytest.raises(ValidationError):
            model(tool_name="boolean", value=invalid)
        assert not Draft202012Validator(model.model_json_schema()).is_valid({"tool_name": "boolean", "value": invalid})


@pytest.mark.parametrize("definition", [True, False])
@pytest.mark.parametrize("required", [True, False])
def test_references_to_boolean_definitions(definition, required):
    model, _ = make_model({"$ref": "#/$defs/Value"}, required=required, definitions={"Value": definition})
    if not required:
        assert model(tool_name="boolean").model_dump(exclude_unset=True) == {"tool_name": "boolean"}
    if definition:
        assert model(tool_name="boolean", value={"x": 1}).value == {"x": 1}
    else:
        with pytest.raises(ValidationError):
            model(tool_name="boolean", value={"x": 1})
        with pytest.raises(ValidationError):
            model(tool_name="boolean", value=None)
