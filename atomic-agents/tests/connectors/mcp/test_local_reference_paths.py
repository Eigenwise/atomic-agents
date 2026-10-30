import pytest
from pydantic import ValidationError

from atomic_agents.connectors.mcp import SchemaTransformer


def _model(schema):
    return SchemaTransformer.create_model_from_schema(schema, "PointerOutput", "pointer", is_output_schema=True)


def test_definition_containers_do_not_share_cache_entries():
    schema = {
        "type": "object",
        "properties": {"count": {"$ref": "#/$defs/Value"}, "name": {"$ref": "#/definitions/Value"}},
        "required": ["count", "name"],
        "$defs": {"Value": {"type": "integer"}},
        "definitions": {"Value": {"type": "string"}},
    }
    model = _model(schema)
    assert model(count=7, name="seven").model_dump() == {"count": 7, "name": "seven"}
    with pytest.raises(ValidationError):
        model(count="invalid", name="seven")
    with pytest.raises(ValidationError):
        model(count=7, name={})


@pytest.mark.parametrize(
    "path, definition_key",
    [
        ("#/$defs/Container/properties/count", "Container"),
        ("#/$defs/ns~1Container/properties/count", "ns/Container"),
        ("#/$defs/ns~01Container/properties/count", "ns~1Container"),
        ("#/$defs/Container%20Name/properties/count", "Container Name"),
    ],
)
def test_full_pointer_resolves_nested_schema_instead_of_same_named_definition(path, definition_key):
    schema = {
        "type": "object",
        "properties": {"value": {"$ref": path}, "values": {"type": "array", "items": {"$ref": path}}},
        "required": ["value", "values"],
        "$defs": {
            definition_key: {"type": "object", "properties": {"count": {"type": "integer"}}},
            "count": {"type": "boolean"},
        },
    }
    model = _model(schema)
    assert model(value=7, values=[8]).model_dump() == {"value": 7, "values": [8]}
    assert model.model_json_schema()["properties"]["value"]["type"] == "integer"
    with pytest.raises(ValidationError):
        model(value="invalid", values=[])
    with pytest.raises(ValidationError):
        model(value=7, values=[{}])


def test_pointer_can_select_schema_from_an_array():
    schema = {
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/Choices/anyOf/1"}},
        "required": ["value"],
        "$defs": {"Choices": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
    }
    model = _model(schema)
    assert model(value=7).value == 7
    with pytest.raises(ValidationError):
        model(value={})


@pytest.mark.parametrize("reference", ["https://example.invalid/schema#/$defs/Value", "#/$defs/Missing/Value"])
def test_unresolved_reference_does_not_bind_an_unrelated_local_definition(reference, caplog):
    schema = {
        "type": "object",
        "properties": {"value": {"$ref": reference}},
        "required": ["value"],
        "$defs": {"Value": {"type": "integer"}},
    }
    model = _model(schema)
    assert model(value={"unresolved": True}).value == {"unresolved": True}
    assert "Could not resolve $ref" in caplog.text


def test_recursive_definition_retains_existing_nonrecursive_fields():
    schema = {
        "type": "object",
        "properties": {"node": {"$ref": "#/$defs/Node"}},
        "required": ["node"],
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"value": {"type": "integer"}, "next": {"$ref": "#/$defs/Node"}},
                "required": ["value"],
            }
        },
    }
    model = _model(schema)
    assert model(node={"value": 7}).node.value == 7
    with pytest.raises(ValidationError):
        model(node={"value": "invalid"})
