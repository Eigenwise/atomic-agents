import pytest
from pydantic import ValidationError

from atomic_agents.connectors.mcp import SchemaTransformer


@pytest.mark.parametrize("definitions_key", ["$defs", "definitions"])
@pytest.mark.parametrize(
    "definition, valid, invalid",
    [
        ({"type": "string"}, "query", {}),
        ({"type": "integer"}, 42, {}),
        ({"type": "array", "items": {"type": "integer"}}, [1, 2], ["invalid"]),
        ({"anyOf": [{"type": "string"}, {"type": "null"}]}, None, {}),
    ],
)
def test_non_object_definitions_preserve_their_value_type(definitions_key, definition, valid, invalid):
    reference = {"$ref": f"#/{definitions_key}/Value"}
    schema = {
        "type": "object",
        "properties": {"value": reference, "values": {"type": "array", "items": reference}},
        "required": ["value", "values"],
        definitions_key: {"Value": definition},
    }
    model = SchemaTransformer.create_model_from_schema(schema, "ValuesInput", "values")
    instance = model(tool_name="values", value=valid, values=[valid])
    assert instance.value == valid
    assert instance.values == [valid]
    with pytest.raises(ValidationError):
        model(tool_name="values", value=invalid, values=[])
    with pytest.raises(ValidationError):
        model(tool_name="values", value=valid, values=[invalid])


def test_definition_aliases_resolve_to_the_underlying_scalar():
    schema = {
        "type": "object",
        "properties": {"value": {"$ref": "#/$defs/Alias"}},
        "required": ["value"],
        "$defs": {"Alias": {"$ref": "#/$defs/Value"}, "Value": {"type": "string"}},
    }
    model = SchemaTransformer.create_model_from_schema(schema, "AliasOutput", "alias", is_output_schema=True)
    assert model(value="result").model_dump() == {"value": "result"}
    with pytest.raises(ValidationError):
        model(value={})
