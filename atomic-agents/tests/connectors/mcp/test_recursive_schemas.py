import json
from copy import deepcopy

import pytest
from jsonschema import Draft202012Validator
from mcp.server import Server
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import TextContent, Tool
from pydantic import ValidationError

from atomic_agents.connectors.mcp import SchemaTransformer, fetch_mcp_tools_async


def tree_schema(definitions_key="$defs", name="Node"):
    reference = f"#/{definitions_key}/{name.replace('~', '~0').replace('/', '~1')}"
    return {
        "type": "object",
        "properties": {"tree": {"$ref": reference}},
        "required": ["tree"],
        definitions_key: {
            name: {
                "type": "object",
                "properties": {
                    "value": {"type": "integer"},
                    "children": {"type": "array", "items": {"$ref": reference}},
                },
                "required": ["value", "children"],
            }
        },
    }


def tree_payload(output=False):
    payload = {"tree": {"value": 1, "children": [{"value": 2, "children": [{"value": 3, "children": []}]}]}}
    if not output:
        payload["tool_name"] = "walk"
    return payload


@pytest.mark.parametrize("definitions_key", ["$defs", "definitions"])
@pytest.mark.parametrize("output", [False, True])
@pytest.mark.parametrize("missing", [False, True], ids=["wrong-type", "missing-field"])
def test_recursive_descendants_keep_validation_and_advertised_schema(definitions_key, output, missing):
    schema = tree_schema(definitions_key)
    model = SchemaTransformer.create_model_from_schema(schema, "Tree", "walk", is_output_schema=output)
    payload = tree_payload(output)
    descendant = payload["tree"]["children"][0]["children"][0]
    if missing:
        del descendant["value"]
    else:
        descendant["value"] = "not an integer"

    assert not Draft202012Validator(schema).is_valid(payload)
    with pytest.raises(ValidationError):
        model.model_validate(payload)
    assert not Draft202012Validator(model.model_json_schema()).is_valid(payload)


@pytest.mark.parametrize("definitions_key", ["$defs", "definitions"])
@pytest.mark.parametrize("output", [False, True])
def test_valid_recursive_values_survive_serialization(definitions_key, output):
    schema = tree_schema(definitions_key)
    model = SchemaTransformer.create_model_from_schema(schema, "Tree", "walk", is_output_schema=output)
    payload = tree_payload(output)

    assert model.model_validate(payload).model_dump() == payload
    assert Draft202012Validator(model.model_json_schema()).is_valid(payload)


@pytest.mark.parametrize("name", ["Node/with~name", "class", "Node with spaces"])
def test_recursive_definition_names_do_not_become_python_expressions(name):
    schema = tree_schema(name=name)
    model = SchemaTransformer.create_model_from_schema(schema, "Tree", "walk")
    payload = tree_payload()
    assert model.model_validate(payload).model_dump() == payload
    payload["tree"]["children"][0]["value"] = "invalid"
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_mutually_recursive_objects_with_identical_titles():
    schema = {
        "type": "object",
        "properties": {"tree": {"$ref": "#/$defs/A"}},
        "required": ["tree"],
        "$defs": {
            "A": {
                "type": "object",
                "title": "Node",
                "properties": {"label": {"type": "string"}, "next": {"anyOf": [{"$ref": "#/$defs/B"}, {"type": "null"}]}},
                "required": ["label", "next"],
            },
            "B": {
                "type": "object",
                "title": "Node",
                "properties": {"value": {"type": "integer"}, "next": {"anyOf": [{"$ref": "#/$defs/A"}, {"type": "null"}]}},
                "required": ["value", "next"],
            },
        },
    }
    model = SchemaTransformer.create_model_from_schema(schema, "Tree", "walk")
    payload = {"tree": {"label": "root", "next": {"value": 1, "next": {"label": "leaf", "next": None}}}, "tool_name": "walk"}
    assert model.model_validate(payload).model_dump() == payload
    assert Draft202012Validator(model.model_json_schema()).is_valid(payload)
    del payload["tree"]["next"]["next"]["label"]
    assert not Draft202012Validator(schema).is_valid(payload)
    with pytest.raises(ValidationError):
        model.model_validate(payload)
    assert not Draft202012Validator(model.model_json_schema()).is_valid(payload)


def test_recursive_field_is_ready_without_a_generated_outer_model():
    schema = tree_schema()
    node_type, _ = SchemaTransformer.json_to_pydantic_field(schema["properties"]["tree"], True, schema)
    value = tree_payload()["tree"]
    assert node_type.model_validate(value).model_dump() == value
    value["children"][0]["value"] = "invalid"
    with pytest.raises(ValidationError):
        node_type.model_validate(value)


@pytest.mark.asyncio
async def test_recursive_models_through_a_real_mcp_session():
    schema = tree_schema()
    server = Server("recursive-models")
    received = []

    @server.list_tools()
    async def list_tools():
        return [Tool(name="walk", description="Return a tree.", inputSchema=schema)]

    @server.call_tool()
    async def call_tool(name, arguments):
        received.append(deepcopy(arguments))
        return [TextContent(type="text", text=json.dumps(arguments))]

    async with create_connected_server_and_client_session(server) as session:
        tools = await fetch_mcp_tools_async(client_session=session)
        tool = tools[0]()
        payload = tree_payload()
        result = await tool.arun(tool.input_schema.model_validate(payload))
        expected = {"tree": payload["tree"]}
        assert received == [expected]
        assert json.loads(result.result[0].text) == expected

        bad_input = deepcopy(payload)
        bad_input["tree"]["children"][0]["value"] = "invalid input"
        with pytest.raises(ValidationError):
            tool.input_schema.model_validate(bad_input)
        assert len(received) == 1
