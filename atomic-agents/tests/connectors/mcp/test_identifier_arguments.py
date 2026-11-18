import asyncio
import json
from contextlib import asynccontextmanager
from copy import deepcopy

import anyio
import pytest
from jsonschema import Draft202012Validator
from mcp import ClientSession, types
from mcp.server.lowlevel import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.shared.memory import create_client_server_memory_streams
from pydantic import ValidationError

from atomic_agents.connectors.mcp import (
    MCPFactory,
    SchemaTransformer,
    fetch_mcp_prompts_async,
    fetch_mcp_resources_async,
    fetch_mcp_tools_async,
)


@asynccontextmanager
async def connected(server):
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(server.run, *server_streams, server.create_initialization_options())
            try:
                async with ClientSession(*client_streams) as session:
                    await session.initialize()
                    yield session
            finally:
                tasks.cancel_scope.cancel()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["tool", "resource", "prompt"])
@pytest.mark.parametrize("occupied", [0, 2])
async def test_discovered_component_preserves_name_argument_and_routing(kind, occupied):
    identifier = f"{kind}_name"
    occupied_names = [f"mcp_{identifier}" if i == 0 else f"mcp_{identifier}_{i}" for i in range(occupied)]
    argument_field = f"mcp_{identifier}" + (f"_{occupied}" if occupied else "")
    arguments = {identifier: "target", **{name: f"keep-{index}" for index, name in enumerate(occupied_names)}}
    input_schema = {
        "type": "object",
        "properties": {name: {"type": "string"} for name in arguments},
        "required": list(arguments),
    }
    server = Server("name-argument-test")
    received = []

    @server.list_tools()
    async def list_tools():
        return [types.Tool(name="route", description="Route one request", inputSchema=input_schema)]

    @server.call_tool()
    async def call_tool(name, values):
        received.append((name, values))
        assert name == "route" and values == arguments
        return [types.TextContent(type="text", text="accepted")]

    @server.list_resources()
    async def list_resources():
        return []

    uri = "memo://entry/" + "/".join("{" + name + "}" for name in arguments)
    expected_uri = "memo://entry/" + "/".join(arguments.values())

    @server.list_resource_templates()
    async def list_resource_templates():
        return [types.ResourceTemplate(name="route", description="Read one record", uriTemplate=uri)]

    @server.read_resource()
    async def read_resource(value):
        received.append(str(value))
        assert str(value) == expected_uri
        return [ReadResourceContents(content="accepted", mime_type="text/plain")]

    @server.list_prompts()
    async def list_prompts():
        return [
            types.Prompt(
                name="route",
                description="Render one prompt",
                arguments=[types.PromptArgument(name=name, required=True) for name in arguments],
            )
        ]

    @server.get_prompt()
    async def get_prompt(name, values):
        received.append((name, values))
        assert name == "route" and values == arguments
        return types.GetPromptResult(
            messages=[types.PromptMessage(role="user", content=types.TextContent(type="text", text="accepted"))]
        )

    fetch = {"tool": fetch_mcp_tools_async, "resource": fetch_mcp_resources_async, "prompt": fetch_mcp_prompts_async}[kind]
    async with connected(server) as session:
        component_class = (await fetch(client_session=session))[0]
        component = component_class()
        selection = {identifier: "route", argument_field: "target", **{name: arguments[name] for name in occupied_names}}
        factory = MCPFactory(client_session=session, event_loop=asyncio.get_running_loop())
        orchestrator = factory.create_orchestrator_schema(**{f"{kind}s": [component_class]})
        selected = orchestrator.model_validate({f"{kind}_parameters": selection})
        params = getattr(selected, f"{kind}_parameters")
        assert getattr(params, identifier) == "route"
        if kind == "tool":
            result = await component.arun(params)
            assert result.result[0].text == "accepted"
        elif kind == "resource":
            result = await component.aread(params)
            assert result.content[0].text == "accepted"
        else:
            result = await component.agenerate(params)
            assert result.content == "accepted"
        assert received == ([expected_uri] if kind == "resource" else [("route", arguments)])
        assert params.model_dump() == selection
        assert json.loads(params.model_dump_json()) == selection
        advertised = component.input_schema.model_json_schema()
        assert Draft202012Validator(advertised).is_valid(selection)
        assert set(advertised["required"]) == set(selection)
        assert advertised["properties"][identifier]["const"] == "route"
        await session.send_ping()


@pytest.mark.parametrize("kind", ["tool", "resource", "prompt"])
@pytest.mark.parametrize("required", [False, True])
def test_name_argument_preserves_constraints_default_and_requiredness(kind, required):
    identifier = f"{kind}_name"
    schema = {
        "type": "object",
        "properties": {identifier: {"type": "integer", "description": "Remote numeric identifier", "default": 7}},
        "required": [identifier] if required else [],
    }
    original = deepcopy(schema)
    model = SchemaTransformer.create_model_from_schema(schema, "RouteInput", "route", attribute_type=kind)
    field = f"mcp_{identifier}"
    supplied = {identifier: "route", field: 9}
    assert model(**supplied).model_dump() == supplied
    assert model.model_json_schema()["properties"][field]["description"] == "Remote numeric identifier"
    with pytest.raises(ValidationError):
        model(**{identifier: "route", field: "not-an-integer"})
    if required:
        with pytest.raises(ValidationError):
            model(**{identifier: "route"})
    else:
        assert getattr(model(**{identifier: "route"}), field) == 7
    assert schema == original


@pytest.mark.parametrize("kind", ["tool", "resource", "prompt"])
def test_output_name_field_remains_an_ordinary_server_property(kind):
    identifier = f"{kind}_name"
    model = SchemaTransformer.create_model_from_schema(
        {"type": "object", "properties": {identifier: {"type": "integer"}}, "required": [identifier]},
        "RouteOutput",
        "route",
        attribute_type=kind,
        is_output_schema=True,
    )
    assert model(**{identifier: 42}).model_dump() == {identifier: 42}
    assert set(model.model_json_schema()["properties"]) == {identifier}
