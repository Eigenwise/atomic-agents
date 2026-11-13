"""Generated MCP classes must retain their own output type and metadata."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp import ClientSession
from mcp.types import GetPromptResult, PromptMessage, ReadResourceResult, TextContent, TextResourceContents

from atomic_agents.connectors.mcp import MCPFactory, MCPPromptDefinition, MCPResourceDefinition, MCPToolDefinition


@pytest.fixture(params=[False, True], ids=["owned", "borrowed"])
def connection(request, monkeypatch):
    session = AsyncMock(spec=ClientSession)
    session.__aenter__.return_value = session
    transport = AsyncMock()
    transport.__aenter__.return_value = (None, None, None)
    monkeypatch.setattr("atomic_agents.connectors.mcp.mcp_factory.ClientSession", MagicMock(return_value=session))
    monkeypatch.setattr("atomic_agents.connectors.mcp.mcp_factory.streamablehttp_client", MagicMock(return_value=transport))
    with asyncio.Runner() as runner:
        factory = MCPFactory(
            "http://localhost", client_session=session if request.param else None, event_loop=runner.get_loop()
        )
        yield factory, session, runner


def resource_classes(factory):
    schema = {"type": "object", "properties": {}}
    return factory._create_resource_classes(
        [
            MCPResourceDefinition("first", None, "file:///first", schema, "text/plain"),
            MCPResourceDefinition("last", None, "file:///last", schema, "application/json"),
        ]
    )


def prompt_classes(factory):
    schema = {"type": "object", "properties": {}}
    return factory._create_prompt_classes(
        [MCPPromptDefinition("first", None, schema), MCPPromptDefinition("last", None, schema)]
    )


@pytest.mark.parametrize("synchronous", [False, True], ids=["async", "sync"])
def test_resource_output_class_matches_each_resource(connection, synchronous):
    factory, session, runner = connection
    classes = resource_classes(factory)
    for resource_class in classes:
        resource = resource_class()
        session.read_resource.return_value = ReadResourceResult(
            contents=[TextResourceContents(uri=resource.uri, text=resource.mcp_resource_name, mimeType="text/custom")]
        )
        params = resource.input_schema(resource_name=resource.mcp_resource_name)
        result = resource.read(params) if synchronous else runner.run(resource.aread(params))
        assert type(result) is resource.output_schema
        assert result.content[0].text == resource.mcp_resource_name
        assert result.mime_type == "text/custom"
        session.read_resource.assert_awaited_with(uri=resource.uri)


@pytest.mark.parametrize("payload", [ReadResourceResult(contents=[]), {"contents": []}, "raw content"])
def test_resource_mime_fallback_matches_each_resource(connection, payload):
    factory, session, runner = connection
    session.read_resource.return_value = payload
    for resource_class, mime_type in zip(resource_classes(factory), ["text/plain", "application/json"]):
        resource = resource_class()
        result = runner.run(resource.aread(resource.input_schema(resource_name=resource.mcp_resource_name)))
        assert result.mime_type == mime_type


@pytest.mark.parametrize("synchronous", [False, True], ids=["async", "sync"])
def test_prompt_output_class_matches_each_prompt(connection, synchronous):
    factory, session, runner = connection
    classes = prompt_classes(factory)
    for prompt_class in classes:
        prompt = prompt_class()
        session.get_prompt.return_value = GetPromptResult(
            messages=[PromptMessage(role="user", content=TextContent(type="text", text=prompt.mcp_prompt_name))]
        )
        params = prompt.input_schema(prompt_name=prompt.mcp_prompt_name)
        result = prompt.generate(params) if synchronous else runner.run(prompt.agenerate(params))
        assert type(result) is prompt.output_schema
        assert result.content == prompt.mcp_prompt_name
        session.get_prompt.assert_awaited_with(name=prompt.mcp_prompt_name, arguments={})


@pytest.mark.parametrize(
    "names,index", [(("first", "last"), 0), (("first", "last"), 1), (("only",), 0)], ids=["first", "last", "single"]
)
@pytest.mark.parametrize(
    "kind,method,action", [("tool", "run", "execute"), ("resource", "read", "read"), ("prompt", "generate", "get")]
)
def test_sync_wrapper_error_names_each_generated_class(kind, method, action, names, index):
    """A missing borrowed-session loop must not report the last definition's name."""
    schema = {"type": "object", "properties": {}}
    definitions = {
        "tool": [MCPToolDefinition(name, None, schema) for name in names],
        "resource": [MCPResourceDefinition(name, None, f"file:///{name}", schema) for name in names],
        "prompt": [MCPPromptDefinition(name, None, schema) for name in names],
    }
    session = AsyncMock(spec=ClientSession)
    with asyncio.Runner() as runner:
        factory = MCPFactory(client_session=session, event_loop=runner.get_loop())
        classes = getattr(factory, f"_create_{kind}_classes")(definitions[kind])
        assert len(classes) == len(names)
        component_class, name = classes[index], names[index]
        component = component_class()
        component._event_loop = None
        params = component.input_schema(**{f"{kind}_name": name})
        with pytest.raises(RuntimeError) as caught:
            getattr(component, method)(params)
        assert str(caught.value) == (
            f"Failed to {action} MCP {kind} '{name}': " "'NoneType' object has no attribute 'run_until_complete'"
        )
        assert isinstance(caught.value.__cause__, AttributeError)
    session.call_tool.assert_not_awaited()
    session.read_resource.assert_not_awaited()
    session.get_prompt.assert_not_awaited()
