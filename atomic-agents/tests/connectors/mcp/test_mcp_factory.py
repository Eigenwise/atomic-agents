import pytest
from pydantic import BaseModel, ValidationError
from mcp import ClientSession, StdioServerParameters
from mcp.types import (
    CallToolResult,
    GetPromptResult,
    ImageContent,
    ListResourcesResult,
    ListResourceTemplatesResult,
    ListToolsResult,
    PromptMessage,
    ReadResourceResult,
    ResourceTemplate,
    TextContent,
    Tool,
)
from atomic_agents.base import BasePrompt, BaseResource
from unittest.mock import AsyncMock, MagicMock, call
import asyncio
from atomic_agents.connectors.mcp import (
    fetch_mcp_tools,
    fetch_mcp_resources,
    fetch_mcp_prompts,
    create_mcp_orchestrator_schema,
    fetch_mcp_attributes_with_schema,
    fetch_mcp_tools_async,
    fetch_mcp_resources_async,
    fetch_mcp_prompts_async,
    MCPFactory,
)
from atomic_agents.connectors.mcp import (
    MCPToolDefinition,
    MCPResourceDefinition,
    MCPPromptDefinition,
    MCPDefinitionService,
    MCPTransportType,
)


@pytest.fixture(params=[False, True], ids=["generic", "typed"])
def error_test_tool(request, monkeypatch):
    output_schema = None
    if request.param:
        output_schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
    definition = MCPToolDefinition(
        name="SearchTool", description="Search", input_schema={"type": "object", "properties": {}}, output_schema=output_schema
    )
    factory = MCPFactory("http://example.com")
    tool = factory._create_tool_classes([definition])[0]()
    session = AsyncMock(spec=ClientSession)
    session.__aenter__.return_value = session
    transport = AsyncMock()
    transport.__aenter__.return_value = (None, None, None)
    monkeypatch.setattr("atomic_agents.connectors.mcp.mcp_factory.ClientSession", MagicMock(return_value=session))
    monkeypatch.setattr("atomic_agents.connectors.mcp.mcp_factory.streamablehttp_client", MagicMock(return_value=transport))
    return tool, session, transport, request.param


@pytest.fixture(params=[False, True], ids=["owned", "borrowed"])
def error_test_connection(request, error_test_tool):
    tool, session, transport, typed = error_test_tool
    loop = asyncio.new_event_loop()
    if request.param:
        tool._client_session = session
        tool._event_loop = loop
    yield tool, session, transport, typed, request.param
    loop.close()


@pytest.mark.parametrize("synchronous", [False, True], ids=["async", "sync"])
@pytest.mark.parametrize("mapping", [False, True], ids=["sdk", "mapping"])
def test_execution_error_precedes_output_parsing(error_test_connection, synchronous, mapping, caplog):
    tool, session, transport, typed, borrowed = error_test_connection
    envelope = CallToolResult(
        isError=True,
        content=[
            TextContent(type="text", text="Search unavailable"),
            ImageContent(type="image", data="PRIVATE_BINARY", mimeType="image/png", text=None),
        ],
        structuredContent={"name": "PRIVATE_STRUCTURED"},
        _meta={"secret": "PRIVATE_META"},
    )
    session.call_tool.return_value = envelope.model_dump() if mapping else envelope
    params = tool.input_schema(tool_name="SearchTool")
    with pytest.raises(RuntimeError, match="Failed to execute MCP tool 'SearchTool': Search unavailable") as raised:
        if synchronous:
            tool.run(params)
        else:
            asyncio.run(tool.arun(params))
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert str(raised.value.__cause__) == "Search unavailable"
    assert "PRIVATE_" not in str(raised.value) + caplog.text
    session.call_tool.assert_awaited_once_with(name="SearchTool", arguments={})
    owned_session_calls = int(not borrowed)
    assert session.initialize.await_count == owned_session_calls
    assert session.__aexit__.await_count == owned_session_calls
    assert transport.__aexit__.await_count == owned_session_calls


@pytest.mark.parametrize("mapping", [False, True], ids=["sdk", "mapping"])
@pytest.mark.parametrize(
    "content",
    [[], [ImageContent(type="image", data="PRIVATE_BINARY", mimeType="image/png")], [TextContent(type="text", text="  \n ")]],
)
def test_execution_error_without_text_has_safe_fallback(error_test_connection, mapping, content, caplog):
    tool, session, transport, typed, borrowed = error_test_connection
    envelope = CallToolResult(isError=True, content=content, structuredContent={"name": "PRIVATE_STRUCTURED"})
    session.call_tool.return_value = envelope.model_dump() if mapping else envelope
    with pytest.raises(RuntimeError, match="MCP tool reported an error without text content") as raised:
        asyncio.run(tool.arun(tool.input_schema(tool_name="SearchTool")))
    assert "PRIVATE_" not in str(raised.value) + caplog.text


@pytest.mark.parametrize("mapping", [False, True], ids=["sdk", "mapping"])
@pytest.mark.parametrize("synchronous", [False, True], ids=["async", "sync"])
def test_successful_envelope_keeps_output_precedence(error_test_connection, mapping, synchronous):
    tool, session, transport, typed, borrowed = error_test_connection
    envelope = CallToolResult(
        isError=False, content=[TextContent(type="text", text='{"name":"text"}')], structuredContent={"name": "structured"}
    )
    session.call_tool.return_value = envelope.model_dump() if mapping else envelope
    params = tool.input_schema(tool_name="SearchTool")
    output = tool.run(params) if synchronous else asyncio.run(tool.arun(params))
    if typed:
        assert output.name == "structured"
    else:
        expected_content = envelope.model_dump()["content"] if mapping else envelope.content
        assert output.result == expected_content
    session.call_tool.assert_awaited_once_with(name="SearchTool", arguments={})


@pytest.mark.parametrize("error_test_tool", [False], indirect=True, ids=["generic"])
@pytest.mark.parametrize("structured", [None, {}, {"name": "structured", "count": 0, "items": ["a", "b"]}])
@pytest.mark.parametrize("mapping", [False, True], ids=["sdk", "mapping"])
@pytest.mark.parametrize("synchronous", [False, True], ids=["async", "sync"])
def test_generic_tool_keeps_structured_only_result(error_test_connection, structured, mapping, synchronous):
    tool, session, transport, typed, borrowed = error_test_connection
    envelope = CallToolResult(content=[], structuredContent=structured)
    session.call_tool.return_value = envelope.model_dump() if mapping else envelope
    params = tool.input_schema(tool_name="SearchTool")

    output = tool.run(params) if synchronous else asyncio.run(tool.arun(params))

    assert output.result == ([] if structured is None else structured)
    session.call_tool.assert_awaited_once_with(name="SearchTool", arguments={})


@pytest.mark.parametrize("content", [None, "article"])
@pytest.mark.parametrize("synchronous", [False, True], ids=["async", "sync"])
def test_raw_error_named_dictionary_stays_application_data(error_test_connection, content, synchronous):
    tool, session, transport, typed, borrowed = error_test_connection
    payload = {"isError": True, "name": "test_value"}
    if content is not None:
        payload["content"] = content
    session.call_tool.return_value = payload
    params = tool.input_schema(tool_name="SearchTool")
    output = tool.run(params) if synchronous else asyncio.run(tool.arun(params))
    if typed:
        assert output.name == "test_value"
    else:
        assert output.result == (payload if content is None else content)
    session.call_tool.assert_awaited_once_with(name="SearchTool", arguments={})


class DummySession:
    pass


def test_fetch_mcp_tools_no_endpoint_raises():
    with pytest.raises(ValueError):
        fetch_mcp_tools()


def test_fetch_mcp_resources_no_endpoint_raises():
    with pytest.raises(ValueError):
        fetch_mcp_resources()


def test_fetch_mcp_prompts_no_endpoint_raises():
    with pytest.raises(ValueError):
        fetch_mcp_prompts()


def test_fetch_mcp_tools_event_loop_without_client_session_raises():
    with pytest.raises(ValueError):
        fetch_mcp_tools(None, MCPTransportType.HTTP_STREAM, client_session=DummySession(), event_loop=None)


def test_fetch_mcp_resources_event_loop_without_client_session_raises():
    with pytest.raises(ValueError):
        fetch_mcp_resources(None, MCPTransportType.HTTP_STREAM, client_session=DummySession(), event_loop=None)


def test_fetch_mcp_prompts_event_loop_without_client_session_raises():
    with pytest.raises(ValueError):
        fetch_mcp_prompts(None, MCPTransportType.HTTP_STREAM, client_session=DummySession(), event_loop=None)


def test_fetch_mcp_tools_empty_definitions(monkeypatch):
    monkeypatch.setattr(MCPFactory, "_fetch_tool_definitions", lambda self: [])
    tools = fetch_mcp_tools("http://example.com", MCPTransportType.HTTP_STREAM)
    assert tools == []


def test_fetch_mcp_resources_empty_definitions(monkeypatch):
    monkeypatch.setattr(MCPFactory, "_fetch_resource_definitions", lambda self: [])
    resources = fetch_mcp_resources("http://example.com", MCPTransportType.HTTP_STREAM)
    assert resources == []


def test_fetch_mcp_prompts_empty_definitions(monkeypatch):
    monkeypatch.setattr(MCPFactory, "_fetch_prompt_definitions", lambda self: [])
    prompts = fetch_mcp_prompts("http://example.com", MCPTransportType.HTTP_STREAM)
    assert prompts == []


def test_fetch_mcp_tools_with_definitions_http(monkeypatch):
    input_schema = {"type": "object", "properties": {}, "required": []}
    definitions = [MCPToolDefinition(name="ToolX", description="Dummy tool", input_schema=input_schema)]
    monkeypatch.setattr(MCPFactory, "_fetch_tool_definitions", lambda self: definitions)
    tools = fetch_mcp_tools("http://example.com", MCPTransportType.HTTP_STREAM)
    assert len(tools) == 1
    tool_cls = tools[0]
    # verify class attributes
    assert tool_cls.mcp_endpoint == "http://example.com"
    assert tool_cls.transport_type == MCPTransportType.HTTP_STREAM
    # input_schema has only tool_name field
    Model = tool_cls.input_schema
    assert "tool_name" in Model.model_fields
    # output_schema has result field (generic schema)
    OutModel = tool_cls.output_schema
    assert "result" in OutModel.model_fields
    # verify _has_typed_output_schema is False for generic schema
    assert tool_cls._has_typed_output_schema is False


@pytest.mark.parametrize("nullable_schema_location", ["input", "output"])
def test_fetch_mcp_tools_preserves_nullable_schema(monkeypatch, nullable_schema_location):
    """A tool must not disappear when its input or output contains a type array."""
    nullable_schema = {
        "type": "object",
        "properties": {"query": {"type": ["string", "null"]}},
        "required": ["query"],
    }
    definition = MCPToolDefinition(
        name="SearchTool",
        description="Search tool",
        input_schema=nullable_schema if nullable_schema_location == "input" else {"type": "object"},
        output_schema=nullable_schema if nullable_schema_location == "output" else None,
    )
    monkeypatch.setattr(MCPFactory, "_fetch_tool_definitions", lambda self: [definition])

    tools = fetch_mcp_tools("http://example.com", MCPTransportType.HTTP_STREAM)

    assert len(tools) == 1
    tool_cls = tools[0]
    if nullable_schema_location == "input":
        assert tool_cls.input_schema(tool_name="SearchTool", query=None).query is None
    else:
        assert tool_cls._has_typed_output_schema is True
        assert tool_cls.output_schema(query=None).query is None


def test_fetch_mcp_tools_with_typed_output_schema(monkeypatch):
    """Test that tools with outputSchema get typed output models"""
    input_schema = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    output_schema = {
        "type": "object",
        "properties": {
            "results": {"type": "array", "items": {"type": "string"}, "description": "Search results"},
            "count": {"type": "integer", "description": "Number of results"},
        },
        "required": ["results", "count"],
    }
    definitions = [
        MCPToolDefinition(
            name="SearchTool", description="A tool with typed output", input_schema=input_schema, output_schema=output_schema
        )
    ]
    monkeypatch.setattr(MCPFactory, "_fetch_tool_definitions", lambda self: definitions)
    tools = fetch_mcp_tools("http://example.com", MCPTransportType.HTTP_STREAM)

    assert len(tools) == 1
    tool_cls = tools[0]

    # verify class attributes
    assert tool_cls.mcp_endpoint == "http://example.com"
    assert tool_cls._has_typed_output_schema is True

    # input_schema has tool_name and query fields
    Model = tool_cls.input_schema
    assert "tool_name" in Model.model_fields
    assert "query" in Model.model_fields

    # output_schema has typed fields instead of generic 'result'
    OutModel = tool_cls.output_schema
    assert "results" in OutModel.model_fields
    assert "count" in OutModel.model_fields
    # Should NOT have the generic 'result' field
    assert "result" not in OutModel.model_fields
    # Should NOT have the tool_name field (output schemas don't need it)
    assert "tool_name" not in OutModel.model_fields


def test_fetch_mcp_tools_mixed_output_schemas(monkeypatch):
    """Test that tools with and without outputSchema are handled correctly together"""
    input_schema = {"type": "object", "properties": {}, "required": []}
    output_schema = {
        "type": "object",
        "properties": {"data": {"type": "string"}},
        "required": ["data"],
    }
    definitions = [
        MCPToolDefinition(name="GenericTool", description="No output schema", input_schema=input_schema),
        MCPToolDefinition(
            name="TypedTool", description="With output schema", input_schema=input_schema, output_schema=output_schema
        ),
    ]
    monkeypatch.setattr(MCPFactory, "_fetch_tool_definitions", lambda self: definitions)
    tools = fetch_mcp_tools("http://example.com", MCPTransportType.HTTP_STREAM)

    assert len(tools) == 2

    # First tool should have generic output
    generic_tool = tools[0]
    assert generic_tool._has_typed_output_schema is False
    assert "result" in generic_tool.output_schema.model_fields

    # Second tool should have typed output
    typed_tool = tools[1]
    assert typed_tool._has_typed_output_schema is True
    assert "data" in typed_tool.output_schema.model_fields
    assert "result" not in typed_tool.output_schema.model_fields


# =============================================================================
# Tests for typed output schema result processing
# =============================================================================


class MockStructuredContentResult(BaseModel):
    """Mock MCP result with structuredContent attribute (MCP spec primary path)"""

    structuredContent: dict


class MockContentItem(BaseModel):
    """Mock content item with text attribute"""

    text: str


class MockContentItemWithData(BaseModel):
    """Mock content item with data attribute"""

    data: dict


class MockContentResult(BaseModel):
    """Mock MCP result with content array"""

    content: list


@pytest.mark.asyncio
async def test_typed_output_schema_with_structured_content_dict(monkeypatch):
    """Test result processing when tool returns BaseModel with structuredContent as dict"""
    input_schema = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
    output_schema = {
        "type": "object",
        "properties": {
            "results": {"type": "array", "items": {"type": "string"}},
            "count": {"type": "integer"},
        },
        "required": ["results", "count"],
    }
    definitions = [
        MCPToolDefinition(name="SearchTool", description="Search tool", input_schema=input_schema, output_schema=output_schema)
    ]
    monkeypatch.setattr(MCPFactory, "_fetch_tool_definitions", lambda self: definitions)
    tools = fetch_mcp_tools("http://example.com", MCPTransportType.HTTP_STREAM)
    tool_cls = tools[0]

    tool_instance = tool_cls()
    mock_result = MockStructuredContentResult(structuredContent={"results": ["a", "b"], "count": 2})

    import atomic_agents.connectors.mcp.mcp_factory as factory_module

    class MockClientSession:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def initialize(self):
            pass

        async def call_tool(self, name, arguments):
            return mock_result

    class MockHttpClient:
        async def __aenter__(self):
            return (None, None, None)

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(factory_module, "ClientSession", MockClientSession)
    monkeypatch.setattr(factory_module, "streamablehttp_client", lambda *args, **kwargs: MockHttpClient())

    InputSchema = tool_cls.input_schema
    params = InputSchema(tool_name="SearchTool", query="test")

    result = await tool_instance.arun(params)

    assert result.results == ["a", "b"]
    assert result.count == 2


@pytest.mark.asyncio
async def test_typed_output_schema_with_json_text_content(monkeypatch):
    """Test result processing when tool returns content[0].text as JSON string"""
    input_schema = {"type": "object", "properties": {}, "required": []}
    output_schema = {
        "type": "object",
        "properties": {"data": {"type": "string"}},
        "required": ["data"],
    }
    definitions = [
        MCPToolDefinition(name="JsonTool", description="JSON tool", input_schema=input_schema, output_schema=output_schema)
    ]
    monkeypatch.setattr(MCPFactory, "_fetch_tool_definitions", lambda self: definitions)
    tools = fetch_mcp_tools("http://example.com", MCPTransportType.HTTP_STREAM)
    tool_cls = tools[0]

    tool_instance = tool_cls()
    mock_content_item = MockContentItem(text='{"data": "hello"}')
    mock_result = MockContentResult(content=[mock_content_item])

    import atomic_agents.connectors.mcp.mcp_factory as factory_module

    class MockClientSession:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def initialize(self):
            pass

        async def call_tool(self, name, arguments):
            return mock_result

    class MockHttpClient:
        async def __aenter__(self):
            return (None, None, None)

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(factory_module, "ClientSession", MockClientSession)
    monkeypatch.setattr(factory_module, "streamablehttp_client", lambda *args, **kwargs: MockHttpClient())

    InputSchema = tool_cls.input_schema
    params = InputSchema(tool_name="JsonTool")

    result = await tool_instance.arun(params)

    assert result.data == "hello"


@pytest.mark.asyncio
async def test_typed_output_schema_with_content_data_dict(monkeypatch):
    """Test result processing when content item has .data attribute as dict"""
    input_schema = {"type": "object", "properties": {}, "required": []}
    output_schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
    }
    definitions = [
        MCPToolDefinition(name="DataTool", description="Data tool", input_schema=input_schema, output_schema=output_schema)
    ]
    monkeypatch.setattr(MCPFactory, "_fetch_tool_definitions", lambda self: definitions)
    tools = fetch_mcp_tools("http://example.com", MCPTransportType.HTTP_STREAM)
    tool_cls = tools[0]

    tool_instance = tool_cls()
    mock_content_item = MockContentItemWithData(data={"value": 42})
    mock_result = MockContentResult(content=[mock_content_item])

    import atomic_agents.connectors.mcp.mcp_factory as factory_module

    class MockClientSession:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def initialize(self):
            pass

        async def call_tool(self, name, arguments):
            return mock_result

    class MockHttpClient:
        async def __aenter__(self):
            return (None, None, None)

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(factory_module, "ClientSession", MockClientSession)
    monkeypatch.setattr(factory_module, "streamablehttp_client", lambda *args, **kwargs: MockHttpClient())

    InputSchema = tool_cls.input_schema
    params = InputSchema(tool_name="DataTool")

    result = await tool_instance.arun(params)

    assert result.value == 42


@pytest.mark.asyncio
async def test_typed_output_schema_with_raw_dict(monkeypatch):
    """Test fallback when tool_result is plain dict"""
    input_schema = {"type": "object", "properties": {}, "required": []}
    output_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    definitions = [
        MCPToolDefinition(name="DictTool", description="Dict tool", input_schema=input_schema, output_schema=output_schema)
    ]
    monkeypatch.setattr(MCPFactory, "_fetch_tool_definitions", lambda self: definitions)
    tools = fetch_mcp_tools("http://example.com", MCPTransportType.HTTP_STREAM)
    tool_cls = tools[0]

    tool_instance = tool_cls()
    mock_result = {"name": "test_value"}

    import atomic_agents.connectors.mcp.mcp_factory as factory_module

    class MockClientSession:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def initialize(self):
            pass

        async def call_tool(self, name, arguments):
            return mock_result

    class MockHttpClient:
        async def __aenter__(self):
            return (None, None, None)

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(factory_module, "ClientSession", MockClientSession)
    monkeypatch.setattr(factory_module, "streamablehttp_client", lambda *args, **kwargs: MockHttpClient())

    InputSchema = tool_cls.input_schema
    params = InputSchema(tool_name="DictTool")

    result = await tool_instance.arun(params)

    assert result.name == "test_value"


@pytest.mark.asyncio
async def test_typed_output_schema_raises_on_unparseable_result(monkeypatch):
    """Test that ValueError is raised when typed schema tool returns unparseable result"""
    input_schema = {"type": "object", "properties": {}, "required": []}
    output_schema = {
        "type": "object",
        "properties": {"data": {"type": "string"}},
        "required": ["data"],
    }
    definitions = [
        MCPToolDefinition(
            name="FailingTool", description="Failing tool", input_schema=input_schema, output_schema=output_schema
        )
    ]
    monkeypatch.setattr(MCPFactory, "_fetch_tool_definitions", lambda self: definitions)
    tools = fetch_mcp_tools("http://example.com", MCPTransportType.HTTP_STREAM)
    tool_cls = tools[0]

    tool_instance = tool_cls()
    # Return a string which can't be parsed as structured content
    mock_result = "just a string, not structured"

    import atomic_agents.connectors.mcp.mcp_factory as factory_module

    class MockClientSession:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def initialize(self):
            pass

        async def call_tool(self, name, arguments):
            return mock_result

    class MockHttpClient:
        async def __aenter__(self):
            return (None, None, None)

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(factory_module, "ClientSession", MockClientSession)
    monkeypatch.setattr(factory_module, "streamablehttp_client", lambda *args, **kwargs: MockHttpClient())

    InputSchema = tool_cls.input_schema
    params = InputSchema(tool_name="FailingTool")

    with pytest.raises(RuntimeError) as exc_info:
        await tool_instance.arun(params)

    # The ValueError gets wrapped in RuntimeError by the outer exception handler
    assert "unparseable result" in str(exc_info.value) or "FailingTool" in str(exc_info.value)


@pytest.mark.asyncio
async def test_typed_output_schema_handles_empty_content_array(monkeypatch):
    """Test graceful handling when content array is empty"""
    input_schema = {"type": "object", "properties": {}, "required": []}
    output_schema = {
        "type": "object",
        "properties": {"data": {"type": "string"}},
        "required": ["data"],
    }
    definitions = [
        MCPToolDefinition(
            name="EmptyContentTool", description="Empty content tool", input_schema=input_schema, output_schema=output_schema
        )
    ]
    monkeypatch.setattr(MCPFactory, "_fetch_tool_definitions", lambda self: definitions)
    tools = fetch_mcp_tools("http://example.com", MCPTransportType.HTTP_STREAM)
    tool_cls = tools[0]

    tool_instance = tool_cls()
    # Empty content array - should fall through and raise error
    mock_result = MockContentResult(content=[])

    import atomic_agents.connectors.mcp.mcp_factory as factory_module

    class MockClientSession:
        def __init__(self, *args):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def initialize(self):
            pass

        async def call_tool(self, name, arguments):
            return mock_result

    class MockHttpClient:
        async def __aenter__(self):
            return (None, None, None)

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(factory_module, "ClientSession", MockClientSession)
    monkeypatch.setattr(factory_module, "streamablehttp_client", lambda *args, **kwargs: MockHttpClient())

    InputSchema = tool_cls.input_schema
    params = InputSchema(tool_name="EmptyContentTool")

    # Should raise error since we can't extract structured content
    with pytest.raises(RuntimeError) as exc_info:
        await tool_instance.arun(params)

    assert "EmptyContentTool" in str(exc_info.value)


def test_fetch_mcp_resources_with_definitions_stdio(monkeypatch):
    input_schema = {"type": "object", "properties": {}, "required": []}
    uri = "resource://example-resource"
    definitions = [MCPResourceDefinition(name="ResY", description="Dummy resource", uri=uri, input_schema=input_schema)]
    monkeypatch.setattr(MCPFactory, "_fetch_resource_definitions", lambda self: definitions)
    resources = fetch_mcp_resources("run me", MCPTransportType.STDIO, working_directory="/tmp")
    assert len(resources) == 1
    res_cls = resources[0]
    # verify class attributes
    assert res_cls.mcp_endpoint == "run me"
    assert res_cls.transport_type == MCPTransportType.STDIO
    assert res_cls.working_directory == "/tmp"
    # input_schema has only resource_name field
    Model = res_cls.input_schema
    assert "resource_name" in Model.model_fields
    # output_schema has content field for resources
    OutModel = res_cls.output_schema
    assert "content" in OutModel.model_fields


def test_fetch_mcp_prompts_with_definitions_http(monkeypatch):
    input_schema = {"type": "object", "properties": {}, "required": []}
    definitions = [MCPPromptDefinition(name="PromptZ", description="Dummy prompt", input_schema=input_schema)]
    monkeypatch.setattr(MCPFactory, "_fetch_prompt_definitions", lambda self: definitions)
    prompts = fetch_mcp_prompts("http://example.com", MCPTransportType.HTTP_STREAM)
    assert len(prompts) == 1
    prompt_cls = prompts[0]
    # verify class attributes
    assert prompt_cls.mcp_endpoint == "http://example.com"
    assert prompt_cls.transport_type == MCPTransportType.HTTP_STREAM
    # input_schema has only prompt_name field
    Model = prompt_cls.input_schema
    assert "prompt_name" in Model.model_fields
    # output_schema has content field for prompts
    OutModel = prompt_cls.output_schema
    assert "content" in OutModel.model_fields


def test_create_mcp_orchestrator_schema_empty():
    schema = create_mcp_orchestrator_schema([], [], [])
    assert schema is None


def test_create_mcp_orchestrator_schema_with_tools():
    class FakeInput(BaseModel):
        tool_name: str
        param: int

    class FakeTool:
        input_schema = FakeInput
        mcp_tool_name = "FakeTool"

    schema = create_mcp_orchestrator_schema(tools=[FakeTool], resources=[], prompts=[])
    assert schema is not None
    assert "tool_parameters" in schema.model_fields
    inst = schema(tool_parameters=FakeInput(tool_name="FakeTool", param=1))
    assert inst.tool_parameters.param == 1


def test_create_mcp_orchestrator_schema_with_resources():
    class FakeInput(BaseModel):
        resource_name: str
        param: int

    class FakeResource:
        input_schema = FakeInput
        mcp_resource_name = "FakeResource"

    schema = create_mcp_orchestrator_schema(resources=[FakeResource])
    assert schema is not None
    assert "resource_parameters" in schema.model_fields
    inst = schema(resource_parameters=FakeInput(resource_name="FakeResource", param=2))
    assert inst.resource_parameters.param == 2


def test_create_mcp_orchestrator_schema_with_prompts():
    class FakeInput(BaseModel):
        prompt_name: str
        param: int

    class FakePrompt:
        input_schema = FakeInput
        mcp_prompt_name = "FakePrompt"

    schema = create_mcp_orchestrator_schema(prompts=[FakePrompt])
    assert schema is not None
    assert "prompt_parameters" in schema.model_fields
    inst = schema(prompt_parameters=FakeInput(prompt_name="FakePrompt", param=3))
    assert inst.prompt_parameters.param == 3


def test_fetch_mcp_attributes_with_schema_no_endpoint_raises():
    with pytest.raises(ValueError):
        fetch_mcp_attributes_with_schema()


def test_fetch_mcp_attributes_with_schema_empty(monkeypatch):
    monkeypatch.setattr(MCPFactory, "create_tools", lambda self: [])
    monkeypatch.setattr(MCPFactory, "create_resources", lambda self: [])
    monkeypatch.setattr(MCPFactory, "create_prompts", lambda self: [])
    tools, resources, prompts, schema = fetch_mcp_attributes_with_schema("endpoint", MCPTransportType.HTTP_STREAM)
    assert tools == []
    assert resources == []
    assert prompts == []
    assert schema is None


def test_fetch_mcp_attributes_with_schema_nonempty(monkeypatch):
    dummy_tools = ["a", "b"]
    dummy_resources = ["c", "d"]
    dummy_prompts = ["e", "f"]
    dummy_schema = object()
    monkeypatch.setattr(MCPFactory, "create_tools", lambda self: dummy_tools)
    monkeypatch.setattr(MCPFactory, "create_resources", lambda self: dummy_resources)
    monkeypatch.setattr(MCPFactory, "create_prompts", lambda self: dummy_prompts)
    monkeypatch.setattr(MCPFactory, "create_orchestrator_schema", lambda self, tools, resources, prompts: dummy_schema)
    tools, resources, prompts, schema = fetch_mcp_attributes_with_schema("endpoint", MCPTransportType.STDIO)
    assert tools == dummy_tools
    assert resources == dummy_resources
    assert prompts == dummy_prompts
    assert schema is dummy_schema


def test_fetch_mcp_tools_with_stdio_and_working_directory(monkeypatch):
    input_schema = {"type": "object", "properties": {}, "required": []}
    tool_definitions = [MCPToolDefinition(name="ToolZ", description=None, input_schema=input_schema)]
    monkeypatch.setattr(MCPFactory, "_fetch_tool_definitions", lambda self: tool_definitions)
    tools = fetch_mcp_tools("run me", MCPTransportType.STDIO, working_directory="/tmp")

    assert len(tools) == 1
    tool_cls = tools[0]
    assert tool_cls.transport_type == MCPTransportType.STDIO
    assert tool_cls.mcp_endpoint == "run me"
    assert tool_cls.working_directory == "/tmp"


def test_fetch_mcp_resources_with_stdio_and_working_directory(monkeypatch):
    input_schema = {"type": "object", "properties": {}, "required": []}
    resource_definitions = [
        MCPResourceDefinition(name="ResZ", description=None, uri="resource://ResZ", input_schema=input_schema)
    ]
    monkeypatch.setattr(MCPFactory, "_fetch_resource_definitions", lambda self: resource_definitions)
    resources = fetch_mcp_resources("run me", MCPTransportType.STDIO, working_directory="/tmp")

    assert len(resources) == 1
    res_cls = resources[0]
    assert res_cls.transport_type == MCPTransportType.STDIO
    assert res_cls.mcp_endpoint == "run me"
    assert res_cls.working_directory == "/tmp"


def test_fetch_mcp_prompts_with_stdio_and_working_directory(monkeypatch):
    input_schema = {"type": "object", "properties": {}, "required": []}
    prompt_definitions = [MCPPromptDefinition(name="PromptZ", description=None, input_schema=input_schema)]
    monkeypatch.setattr(MCPFactory, "_fetch_prompt_definitions", lambda self: prompt_definitions)
    prompts = fetch_mcp_prompts("run me", MCPTransportType.STDIO, working_directory="/tmp")

    assert len(prompts) == 1
    prompt_cls = prompts[0]
    assert prompt_cls.transport_type == MCPTransportType.STDIO
    assert prompt_cls.mcp_endpoint == "run me"
    assert prompt_cls.working_directory == "/tmp"


@pytest.mark.parametrize("transport_type", [MCPTransportType.HTTP_STREAM, MCPTransportType.STDIO])
def test_run_tool(monkeypatch, transport_type):
    # Setup dummy transports and session
    import atomic_agents.connectors.mcp.mcp_factory as mtf

    class DummyTransportCM:
        def __init__(self, ret):
            self.ret = ret

        async def __aenter__(self):
            return self.ret

        async def __aexit__(self, exc_type, exc, tb):
            pass

    def dummy_sse_client(endpoint):
        return DummyTransportCM((None, None))

    def dummy_stdio_client(params):
        return DummyTransportCM((None, None))

    class DummySessionCM:
        def __init__(self, rs=None, ws=None):
            pass

        async def initialize(self):
            pass

        async def call_tool(self, name, arguments):
            return {"content": f"{name}-{arguments}-ok"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr(mtf, "sse_client", dummy_sse_client)
    monkeypatch.setattr(mtf, "stdio_client", dummy_stdio_client)
    monkeypatch.setattr(mtf, "ClientSession", DummySessionCM)

    # Prepare definitions
    input_schema = {"type": "object", "properties": {}, "required": []}
    tool_definitions = [MCPToolDefinition(name="ToolA", description="desc", input_schema=input_schema)]

    monkeypatch.setattr(MCPFactory, "_fetch_tool_definitions", lambda self: tool_definitions)

    # Run fetch and execute tool
    endpoint = "cmd run" if transport_type == MCPTransportType.STDIO else "http://e"
    tools = fetch_mcp_tools(
        endpoint, transport_type, working_directory="wd" if transport_type == MCPTransportType.STDIO else None
    )
    tool_cls = tools[0]
    inst = tool_cls()
    result = inst.run(tool_cls.input_schema(tool_name="ToolA"))
    assert result.result == "ToolA-{}-ok"


@pytest.mark.parametrize("transport_type", [MCPTransportType.HTTP_STREAM, MCPTransportType.STDIO])
def test_read_resource(monkeypatch, transport_type):
    # Setup dummy transports and session
    import atomic_agents.connectors.mcp.mcp_factory as mtf

    class DummyTransportCM:
        def __init__(self, ret):
            self.ret = ret

        async def __aenter__(self):
            return self.ret

        async def __aexit__(self, exc_type, exc, tb):
            pass

    def dummy_sse_client(endpoint):
        return DummyTransportCM((None, None))

    def dummy_stdio_client(params):
        return DummyTransportCM((None, None))

    class DummySessionCM:
        def __init__(self, rs=None, ws=None):
            pass

        async def initialize(self):
            pass

        async def read_resource(self, *args, **kwargs):
            return {"content": "resource-ResA-ok"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr(mtf, "sse_client", dummy_sse_client)
    monkeypatch.setattr(mtf, "stdio_client", dummy_stdio_client)
    monkeypatch.setattr(mtf, "ClientSession", DummySessionCM)

    # Prepare definitions
    input_schema = {"type": "object", "properties": {}, "required": []}
    resource_definitions = [
        MCPResourceDefinition(name="ResA", description="desc", uri="resource://ResA", input_schema=input_schema)
    ]
    monkeypatch.setattr(MCPFactory, "_fetch_resource_definitions", lambda self: resource_definitions)

    endpoint = "cmd run" if transport_type == MCPTransportType.STDIO else "http://e"

    # Read data from resource
    resources = fetch_mcp_resources(
        endpoint, transport_type, working_directory="wd" if transport_type == MCPTransportType.STDIO else None
    )
    resource_cls = resources[0]
    inst = resource_cls()
    result = inst.read(resource_cls.input_schema(resource_name="ResA"))
    assert result.content["content"] == "resource-ResA-ok"


@pytest.mark.parametrize("transport_type", [MCPTransportType.HTTP_STREAM, MCPTransportType.STDIO])
def test_generate_prompt(monkeypatch, transport_type):
    # Setup dummy transports and session
    import atomic_agents.connectors.mcp.mcp_factory as mtf

    class DummyTransportCM:
        def __init__(self, ret):
            self.ret = ret

        async def __aenter__(self):
            return self.ret

        async def __aexit__(self, exc_type, exc, tb):
            pass

    def dummy_sse_client(endpoint):
        return DummyTransportCM((None, None))

    def dummy_stdio_client(params):
        return DummyTransportCM((None, None))

    class DummySessionCM:
        def __init__(self, rs=None, ws=None):
            pass

        async def initialize(self):
            pass

        async def get_prompt(self, *, name, arguments):
            class Msg(BaseModel):
                content: str

            return {"messages": [Msg(content=f"prompt-{name}-{arguments}-ok")]}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr(mtf, "sse_client", dummy_sse_client)
    monkeypatch.setattr(mtf, "stdio_client", dummy_stdio_client)
    monkeypatch.setattr(mtf, "ClientSession", DummySessionCM)

    # Prepare definitions
    input_schema = {"type": "object", "properties": {}, "required": []}
    prompt_definitions = [MCPPromptDefinition(name="PromptA", description="desc", input_schema=input_schema)]

    monkeypatch.setattr(MCPFactory, "_fetch_prompt_definitions", lambda self: prompt_definitions)

    endpoint = "cmd run" if transport_type == MCPTransportType.STDIO else "http://e"

    # Generate prompt
    prompts = fetch_mcp_prompts(
        endpoint, transport_type, working_directory="wd" if transport_type == MCPTransportType.STDIO else None
    )
    prompt_cls = prompts[0]
    inst = prompt_cls()
    result = inst.generate(prompt_cls.input_schema(prompt_name="PromptA"))
    assert result.content == "prompt-PromptA-{}-ok"


def test_run_tool_with_persistent_session(monkeypatch):
    import atomic_agents.connectors.mcp.mcp_factory as mtf

    # Setup persistent client
    class DummySessionPersistent:
        async def call_tool(self, name, arguments):
            return {"content": "persist-ok"}

    client = DummySessionPersistent()
    # Stub definition fetch for persistent
    definitions = [
        MCPToolDefinition(name="ToolB", description=None, input_schema={"type": "object", "properties": {}, "required": []})
    ]

    async def fake_fetch_defs(session):
        return definitions

    monkeypatch.setattr(mtf.MCPDefinitionService, "fetch_tool_definitions_from_session", staticmethod(fake_fetch_defs))
    # Create and pass an event loop
    loop = asyncio.new_event_loop()
    try:
        tools = fetch_mcp_tools(None, MCPTransportType.HTTP_STREAM, client_session=client, event_loop=loop)
        tool_cls = tools[0]
        inst = tool_cls()
        result = inst.run(tool_cls.input_schema(tool_name="ToolB"))
        assert result.result == "persist-ok"
    finally:
        loop.close()


def test_read_resource_with_persistent_session(monkeypatch):
    import atomic_agents.connectors.mcp.mcp_factory as mtf

    # Setup persistent client that matches factory expectations
    class DummySessionPersistent:
        async def read_resource(self, *, uri):
            return {"content": "persist-resource-ok"}

    client = DummySessionPersistent()
    # Stub definition fetch for persistent
    definitions = [
        MCPResourceDefinition(
            name="ResB",
            description=None,
            uri="resource://ResB",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
    ]

    async def fake_fetch_defs(session):
        return definitions

    monkeypatch.setattr(mtf.MCPDefinitionService, "fetch_resource_definitions_from_session", staticmethod(fake_fetch_defs))
    # Create and pass an event loop
    loop = asyncio.new_event_loop()
    try:
        resources = fetch_mcp_resources(None, MCPTransportType.HTTP_STREAM, client_session=client, event_loop=loop)
        res_cls = resources[0]
        inst = res_cls()
        result = inst.read(res_cls.input_schema(resource_name="ResB"))
        assert result.content["content"] == "persist-resource-ok"
    finally:
        loop.close()


def test_generate_prompt_with_persistent_session(monkeypatch):
    import atomic_agents.connectors.mcp.mcp_factory as mtf

    # Setup persistent client
    class DummySessionPersistent:
        async def get_prompt(self, *, name, arguments):
            class Msg(BaseModel):
                content: str

            return {"messages": [Msg(content="persist-prompt-ok")]}

    client = DummySessionPersistent()
    # Stub definition fetch for persistent
    definitions = [
        MCPPromptDefinition(
            name="PromptB", description=None, input_schema={"type": "object", "properties": {}, "required": []}
        )
    ]

    async def fake_fetch_defs(session):
        return definitions

    monkeypatch.setattr(mtf.MCPDefinitionService, "fetch_prompt_definitions_from_session", staticmethod(fake_fetch_defs))
    # Create and pass an event loop
    loop = asyncio.new_event_loop()
    try:
        prompts = fetch_mcp_prompts(None, MCPTransportType.HTTP_STREAM, client_session=client, event_loop=loop)
        prompt_cls = prompts[0]
        inst = prompt_cls()
        result = inst.generate(prompt_cls.input_schema(prompt_name="PromptB"))
        assert result.content == "persist-prompt-ok"
    finally:
        loop.close()


def test_fetch_tool_definitions_via_service(monkeypatch):
    from atomic_agents.connectors.mcp.mcp_factory import MCPFactory
    from atomic_agents.connectors.mcp.mcp_definition_service import MCPToolDefinition

    defs = [MCPToolDefinition(name="X", description="d", input_schema={"type": "object", "properties": {}, "required": []})]

    def fake_fetch(self):
        return defs

    monkeypatch.setattr(MCPFactory, "_fetch_tool_definitions", fake_fetch)
    factory_http = MCPFactory("http://e", MCPTransportType.HTTP_STREAM)
    assert factory_http._fetch_tool_definitions() == defs
    factory_stdio = MCPFactory("http://e", MCPTransportType.STDIO, working_directory="/tmp")
    assert factory_stdio._fetch_tool_definitions() == defs


def test_fetch_resource_definitions_via_service(monkeypatch):
    from atomic_agents.connectors.mcp.mcp_factory import MCPFactory
    from atomic_agents.connectors.mcp.mcp_definition_service import MCPResourceDefinition

    defs = [
        MCPResourceDefinition(
            name="Y", description="d", uri="resource://Y", input_schema={"type": "object", "properties": {}, "required": []}
        )
    ]

    def fake_fetch(self):
        return defs

    monkeypatch.setattr(MCPFactory, "_fetch_resource_definitions", fake_fetch)
    factory_http = MCPFactory("http://e", MCPTransportType.HTTP_STREAM)
    assert factory_http._fetch_resource_definitions() == defs
    factory_stdio = MCPFactory("http://e", MCPTransportType.STDIO, working_directory="/tmp")
    assert factory_stdio._fetch_resource_definitions() == defs


def test_fetch_prompt_definitions_via_service(monkeypatch):
    from atomic_agents.connectors.mcp.mcp_factory import MCPFactory
    from atomic_agents.connectors.mcp.mcp_definition_service import MCPPromptDefinition

    defs = [MCPPromptDefinition(name="Z", description="d", input_schema={"type": "object", "properties": {}, "required": []})]

    def fake_fetch(self):
        return defs

    monkeypatch.setattr(MCPFactory, "_fetch_prompt_definitions", fake_fetch)
    factory_http = MCPFactory("http://e", MCPTransportType.HTTP_STREAM)
    assert factory_http._fetch_prompt_definitions() == defs
    factory_stdio = MCPFactory("http://e", MCPTransportType.STDIO, working_directory="/tmp")
    assert factory_stdio._fetch_prompt_definitions() == defs


def test_fetch_tool_definitions_propagates_error(monkeypatch):
    from atomic_agents.connectors.mcp.mcp_factory import MCPFactory

    def fake_fetch(self):
        raise RuntimeError("nope")

    monkeypatch.setattr(MCPFactory, "_fetch_tool_definitions", fake_fetch)
    factory = MCPFactory("http://e", MCPTransportType.HTTP_STREAM)
    with pytest.raises(RuntimeError):
        factory._fetch_tool_definitions()


def test_fetch_resource_definitions_propagates_error(monkeypatch):
    from atomic_agents.connectors.mcp.mcp_factory import MCPFactory

    def fake_fetch(self):
        raise RuntimeError("nope")

    monkeypatch.setattr(MCPFactory, "_fetch_resource_definitions", fake_fetch)
    factory = MCPFactory("http://e", MCPTransportType.HTTP_STREAM)
    with pytest.raises(RuntimeError):
        factory._fetch_resource_definitions()


def test_fetch_prompt_definitions_propagates_error(monkeypatch):
    from atomic_agents.connectors.mcp.mcp_factory import MCPFactory

    def fake_fetch(self):
        raise RuntimeError("nope")

    monkeypatch.setattr(MCPFactory, "_fetch_prompt_definitions", fake_fetch)
    factory = MCPFactory("http://e", MCPTransportType.HTTP_STREAM)
    with pytest.raises(RuntimeError):
        factory._fetch_prompt_definitions()


def test_run_tool_handles_special_result_types(monkeypatch):
    import atomic_agents.connectors.mcp.mcp_factory as mtf

    class DummyTransportCM:
        def __init__(self, ret):
            self.ret = ret

        async def __aenter__(self):
            return self.ret

        async def __aexit__(self, exc_type, exc, tb):
            pass

    def dummy_sse_client(endpoint):
        return DummyTransportCM((None, None))

    def dummy_stdio_client(params):
        return DummyTransportCM((None, None))

    class DynamicSession:
        def __init__(self, *args, **kwargs):
            pass

        async def initialize(self):
            pass

        async def call_tool(self, name, arguments):
            class R(BaseModel):
                content: str

            return R(content="hello")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr(mtf, "sse_client", dummy_sse_client)
    monkeypatch.setattr(mtf, "stdio_client", dummy_stdio_client)
    monkeypatch.setattr(mtf, "ClientSession", DynamicSession)
    definitions = [
        MCPToolDefinition(name="T", description=None, input_schema={"type": "object", "properties": {}, "required": []})
    ]
    monkeypatch.setattr(MCPFactory, "_fetch_tool_definitions", lambda self: definitions)
    tool_cls = fetch_mcp_tools("e", MCPTransportType.HTTP_STREAM)[0]
    result = tool_cls().run(tool_cls.input_schema(tool_name="T"))
    assert result.result == "hello"

    # plain result
    class PlainSession(DynamicSession):
        async def call_tool(self, name, arguments):
            return 123

    monkeypatch.setattr(mtf, "ClientSession", PlainSession)
    result2 = fetch_mcp_tools("e", MCPTransportType.HTTP_STREAM)[0]().run(tool_cls.input_schema(tool_name="T"))
    assert result2.result == 123


def test_run_resource_handles_special_result_types(monkeypatch):
    import atomic_agents.connectors.mcp.mcp_factory as mtf

    class DummyTransportCM:
        def __init__(self, ret):
            self.ret = ret

        async def __aenter__(self):
            return self.ret

        async def __aexit__(self, exc_type, exc, tb):
            pass

    def dummy_sse_client(endpoint):
        return DummyTransportCM((None, None))

    def dummy_stdio_client(params):
        return DummyTransportCM((None, None))

    class DynamicSession:
        def __init__(self, *args, **kwargs):
            pass

        async def initialize(self):
            pass

        async def read_resource(self, *, uri):
            class R(BaseModel):
                contents: str

            return R(contents="res-hello")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr(mtf, "sse_client", dummy_sse_client)
    monkeypatch.setattr(mtf, "stdio_client", dummy_stdio_client)
    monkeypatch.setattr(mtf, "ClientSession", DynamicSession)
    definitions = [
        MCPResourceDefinition(
            name="R", description=None, uri="resource://R", input_schema={"type": "object", "properties": {}, "required": []}
        )
    ]
    monkeypatch.setattr(MCPFactory, "_fetch_resource_definitions", lambda self: definitions)
    resource_cls = fetch_mcp_resources("e", MCPTransportType.HTTP_STREAM)[0]
    result = resource_cls().read(resource_cls.input_schema(resource_name="R"))

    # resource output schema uses 'content' as the field name; the inner value
    # may itself be a BaseModel with attribute 'contents' (legacy) or 'content'.
    def _unwrap_output(out):
        val = getattr(out, "content", out)
        if isinstance(val, BaseModel):
            if hasattr(val, "content"):
                return val.content
            if hasattr(val, "contents"):
                return val.contents
        return val

    assert _unwrap_output(result) == "res-hello"

    # plain result
    class PlainSession(DynamicSession):
        async def read_resource(self, *, uri):
            return 456

    monkeypatch.setattr(mtf, "ClientSession", PlainSession)
    result2 = fetch_mcp_resources("e", MCPTransportType.HTTP_STREAM)[0]().read(resource_cls.input_schema(resource_name="R"))
    assert _unwrap_output(result2) == 456


def test_run_prompt_handles_special_result_types(monkeypatch):
    import atomic_agents.connectors.mcp.mcp_factory as mtf

    class DummyTransportCM:
        def __init__(self, ret):
            self.ret = ret

        async def __aenter__(self):
            return self.ret

        async def __aexit__(self, exc_type, exc, tb):
            pass

    def dummy_sse_client(endpoint):
        return DummyTransportCM((None, None))

    def dummy_stdio_client(params):
        return DummyTransportCM((None, None))

    class DynamicSession:
        def __init__(self, *args, **kwargs):
            pass

        async def initialize(self):
            pass

        async def get_prompt(self, *, name, arguments):
            class Msg(BaseModel):
                content: str

            return {"messages": [Msg(content="prompt-hello")]}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr(mtf, "sse_client", dummy_sse_client)
    monkeypatch.setattr(mtf, "stdio_client", dummy_stdio_client)
    monkeypatch.setattr(mtf, "ClientSession", DynamicSession)
    definitions = [
        MCPPromptDefinition(name="P", description=None, input_schema={"type": "object", "properties": {}, "required": []})
    ]
    monkeypatch.setattr(MCPFactory, "_fetch_prompt_definitions", lambda self: definitions)
    prompt_cls = fetch_mcp_prompts("e", MCPTransportType.HTTP_STREAM)[0]
    result = prompt_cls().generate(prompt_cls.input_schema(prompt_name="P"))
    assert result.content == "prompt-hello"

    # plain result
    class PlainSession(DynamicSession):
        async def get_prompt(self, *, name, arguments):
            return {"messages": ["plain-hello"]}

    monkeypatch.setattr(mtf, "ClientSession", PlainSession)
    result2 = fetch_mcp_prompts("e", MCPTransportType.HTTP_STREAM)[0]().generate(prompt_cls.input_schema(prompt_name="P"))
    assert result2.content == "plain-hello"


def test_run_invalid_stdio_command_raises(monkeypatch):
    import atomic_agents.connectors.mcp.mcp_factory as mtf

    class DummyTransportCM:
        def __init__(self, ret):
            self.ret = ret

        async def __aenter__(self):
            return self.ret

        async def __aexit__(self, exc_type, exc, tb):
            pass

    def dummy_sse_client(endpoint):
        return DummyTransportCM((None, None))

    def dummy_stdio_client(params):
        return DummyTransportCM((None, None))

    monkeypatch.setattr(mtf, "sse_client", dummy_sse_client)
    monkeypatch.setattr(mtf, "stdio_client", dummy_stdio_client)
    monkeypatch.setattr(
        MCPFactory,
        "_fetch_tool_definitions",
        lambda self: [
            MCPToolDefinition(name="Bad", description=None, input_schema={"type": "object", "properties": {}, "required": []})
        ],
    )
    monkeypatch.setattr(
        MCPFactory,
        "_fetch_resource_definitions",
        lambda self: [
            MCPResourceDefinition(
                name="Y",
                description="d",
                uri="resource://Y",
                input_schema={"type": "object", "properties": {}, "required": []},
            )
        ],
    )
    monkeypatch.setattr(
        MCPFactory,
        "_fetch_prompt_definitions",
        lambda self: [
            MCPPromptDefinition(name="Z", description="d", input_schema={"type": "object", "properties": {}, "required": []})
        ],
    )

    # Use a blank-space endpoint to bypass init validation but trigger empty command in STDIO
    tool_cls = fetch_mcp_tools(" ", MCPTransportType.STDIO, working_directory="/wd")[0]
    with pytest.raises(RuntimeError) as exc:
        tool_cls().run(tool_cls.input_schema(tool_name="Bad"))
    assert "STDIO command string cannot be empty" in str(exc.value)

    resource_cls = fetch_mcp_resources(" ", MCPTransportType.STDIO, working_directory="/wd")[0]
    with pytest.raises(RuntimeError) as exc:
        resource_cls().read(resource_cls.input_schema(resource_name="Y"))
    assert "STDIO command string cannot be empty" in str(exc.value)

    prompt_cls = fetch_mcp_prompts(" ", MCPTransportType.STDIO, working_directory="/wd")[0]
    with pytest.raises(RuntimeError) as exc:
        prompt_cls().generate(prompt_cls.input_schema(prompt_name="Z"))
    assert "STDIO command string cannot be empty" in str(exc.value)


def test_create_tool_classes_skips_invalid(monkeypatch):
    factory = MCPFactory("endpoint", MCPTransportType.HTTP_STREAM)
    defs = [
        MCPToolDefinition(name="Bad", description=None, input_schema={"type": "object", "properties": {}, "required": []}),
        MCPToolDefinition(name="Good", description=None, input_schema={"type": "object", "properties": {}, "required": []}),
    ]

    class FakeST:
        def create_model_from_schema(self, schema, model_name, tname, doc, attribute_type="tool"):
            if tname == "Bad":
                raise ValueError("fail")
            return BaseModel

    factory.schema_transformer = FakeST()
    tools = factory._create_tool_classes(defs)
    assert len(tools) == 1
    assert tools[0].mcp_tool_name == "Good"


def test_create_resource_classes_skips_invalid(monkeypatch):
    factory = MCPFactory("endpoint", MCPTransportType.HTTP_STREAM)
    defs = [
        MCPResourceDefinition(
            name="Bad",
            description=None,
            uri="resource://Bad",
            input_schema={"type": "object", "properties": {}, "required": []},
        ),
        MCPResourceDefinition(
            name="Good",
            description=None,
            uri="resource://Good",
            input_schema={"type": "object", "properties": {}, "required": []},
        ),
    ]

    class FakeST:
        def create_model_from_schema(self, schema, model_name, tname, doc, attribute_type="resource"):
            if tname == "Bad":
                raise ValueError("fail")
            return BaseModel

    factory.schema_transformer = FakeST()
    resources = factory._create_resource_classes(defs)
    assert len(resources) == 1
    assert resources[0].mcp_resource_name == "Good"


def test_create_prompt_classes_skips_invalid(monkeypatch):
    factory = MCPFactory("endpoint", MCPTransportType.HTTP_STREAM)
    defs = [
        MCPPromptDefinition(name="Bad", description=None, input_schema={"type": "object", "properties": {}, "required": []}),
        MCPPromptDefinition(name="Good", description=None, input_schema={"type": "object", "properties": {}, "required": []}),
    ]

    class FakeST:
        def create_model_from_schema(self, schema, model_name, tname, doc, attribute_type="prompt"):
            if tname == "Bad":
                raise ValueError("fail")
            return BaseModel

    factory.schema_transformer = FakeST()
    prompts = factory._create_prompt_classes(defs)
    assert len(prompts) == 1
    assert prompts[0].mcp_prompt_name == "Good"


@pytest.fixture
def mocked_mcp_connection(monkeypatch: pytest.MonkeyPatch) -> tuple[AsyncMock, dict[str, MagicMock]]:
    import atomic_agents.connectors.mcp.mcp_factory as factory_module
    import atomic_agents.connectors.mcp.mcp_definition_service as definition_module

    session = AsyncMock(spec=ClientSession)
    session.__aenter__.return_value = session
    session.__aexit__.return_value = None
    session.list_tools.return_value = ListToolsResult(
        tools=[
            Tool(
                name="EchoTool",
                inputSchema={
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "optional": {"type": "string"}},
                    "required": ["query"],
                },
            )
        ]
    )
    session.call_tool.return_value = CallToolResult(content=[TextContent(type="text", text="echo result")])
    monkeypatch.setattr(factory_module, "ClientSession", MagicMock(return_value=session))
    monkeypatch.setattr(definition_module, "ClientSession", MagicMock(return_value=session))

    clients = {}
    for client_name, streams in (
        ("stdio_client", (None, None)),
        ("sse_client", (None, None)),
        ("streamablehttp_client", (None, None, None)),
    ):
        transport = AsyncMock()
        transport.__aenter__.return_value = streams
        transport.__aexit__.return_value = None
        client = MagicMock(return_value=transport)
        clients[client_name] = client
        monkeypatch.setattr(factory_module, client_name, client)
        monkeypatch.setattr(definition_module, client_name, client)
    return session, clients


@pytest.mark.parametrize(
    "transport_type,client_name,endpoint,expected_connection",
    [
        pytest.param(
            MCPTransportType.STDIO,
            "stdio_client",
            "mock-mcp --local",
            StdioServerParameters(command="mock-mcp", args=["--local"], env=None, cwd="workdir"),
            id="stdio",
        ),
        pytest.param(MCPTransportType.SSE, "sse_client", "http://example.test", "http://example.test/sse", id="sse"),
        pytest.param(
            MCPTransportType.HTTP_STREAM,
            "streamablehttp_client",
            "http://example.test",
            "http://example.test/mcp/",
            id="http-stream",
        ),
    ],
)
def test_generated_tool_discovers_and_calls_sdk_tool_through_transport(
    mocked_mcp_connection: tuple[AsyncMock, dict[str, MagicMock]],
    transport_type: MCPTransportType,
    client_name: str,
    endpoint: str,
    expected_connection: str | StdioServerParameters,
) -> None:
    session, clients = mocked_mcp_connection
    tool_class = fetch_mcp_tools(endpoint, transport_type, working_directory="workdir")[0]
    result = tool_class().run(tool_class.input_schema(tool_name="EchoTool", query="hello", optional=None))

    assert result.result == [TextContent(type="text", text="echo result")]
    session.list_tools.assert_awaited_once_with()
    session.call_tool.assert_awaited_once_with(name="EchoTool", arguments={"query": "hello"})
    assert session.initialize.await_count == 2
    assert clients[client_name].call_args_list == [call(expected_connection), call(expected_connection)]
    session.__aexit__.assert_has_awaits([call(session, None, None, None), call(session, None, None, None)])
    transport = clients[client_name].return_value
    transport.__aexit__.assert_has_awaits([call(transport, None, None, None), call(transport, None, None, None)])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_result,expected_count",
    [
        pytest.param(CallToolResult(content=[TextContent(type="text", text='{"count": 7}')]), 7, id="json-text-fallback"),
        pytest.param(
            CallToolResult(content=[TextContent(type="text", text='{"count": 99}')], structuredContent={"count": 7}),
            7,
            id="structured-precedence",
        ),
        pytest.param(
            CallToolResult(content=[TextContent(type="text", text='{"count": 99}')], structuredContent={"count": 0}),
            0,
            id="structured-zero",
        ),
    ],
)
async def test_generated_tool_uses_sdk_typed_output_on_persistent_session(
    mocked_mcp_connection: tuple[AsyncMock, dict[str, MagicMock]], tool_result: CallToolResult, expected_count: int
) -> None:
    session, clients = mocked_mcp_connection
    session.list_tools.return_value.tools[0].outputSchema = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
    }
    session.call_tool.return_value = tool_result
    tool_class = (await fetch_mcp_tools_async(client_session=session))[0]
    result = await tool_class().arun(tool_class.input_schema(tool_name="EchoTool", query="hello"))

    assert result.model_dump() == {"count": expected_count}
    session.call_tool.assert_awaited_once_with(name="EchoTool", arguments={"query": "hello"})
    session.initialize.assert_awaited_once_with()
    for client in clients.values():
        client.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_result,cause_type,message",
    [
        pytest.param(
            CallToolResult(content=[TextContent(type="text", text="invalid")]),
            ValueError,
            "unparseable result",
            id="malformed-json",
        ),
        pytest.param(
            CallToolResult(content=[TextContent(type="text", text="[]")]), ValueError, "unparseable result", id="json-array"
        ),
        pytest.param(
            CallToolResult(content=[TextContent(type="text", text="0")]), ValueError, "unparseable result", id="json-zero"
        ),
        pytest.param(
            CallToolResult(content=[TextContent(type="text", text="false")]), ValueError, "unparseable result", id="json-false"
        ),
        pytest.param(
            CallToolResult(content=[TextContent(type="text", text="null")]), ValueError, "unparseable result", id="json-null"
        ),
        pytest.param(
            CallToolResult(content=[TextContent(type="text", text='{"count": "invalid"}')]),
            ValidationError,
            "validation error",
            id="invalid-count",
        ),
        pytest.param(
            CallToolResult(content=[TextContent(type="text", text='{"count": 7}')], structuredContent={}),
            ValidationError,
            "validation error",
            id="empty-structured-precedence",
        ),
    ],
)
async def test_generated_typed_tool_rejects_invalid_sdk_output(
    mocked_mcp_connection: tuple[AsyncMock, dict[str, MagicMock]],
    tool_result: CallToolResult,
    cause_type: type[Exception],
    message: str,
) -> None:
    session, _ = mocked_mcp_connection
    session.list_tools.return_value.tools[0].outputSchema = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
        "required": ["count"],
    }
    session.call_tool.return_value = tool_result
    tool_class = (await fetch_mcp_tools_async(client_session=session))[0]
    with pytest.raises(RuntimeError, match=message) as failure:
        await tool_class().arun(tool_class.input_schema(tool_name="EchoTool", query="hello"))

    assert isinstance(failure.value.__cause__, cause_type)
    session.call_tool.assert_awaited_once_with(name="EchoTool", arguments={"query": "hello"})


def test__fetch_tool_definitions_service_branch(monkeypatch):
    """Covers lines 112-113: MCPDefinitionService branch in _fetch_tool_definitions."""
    factory = MCPFactory("dummy_endpoint", MCPTransportType.HTTP_STREAM)

    # Patch fetch_tool_definitions to avoid real async work
    async def dummy_fetch_tool_definitions(self):
        return [
            MCPToolDefinition(name="COV", description="cov", input_schema={"type": "object", "properties": {}, "required": []})
        ]

    monkeypatch.setattr(MCPDefinitionService, "fetch_tool_definitions", dummy_fetch_tool_definitions)
    result = factory._fetch_tool_definitions()
    assert result[0].name == "COV"


def test_fetch_resource_definitions_service_branch(monkeypatch):
    """Covers lines of MCPDefinitionService branch in _fetch_resource_definitions."""
    factory = MCPFactory("dummy_endpoint", MCPTransportType.HTTP_STREAM)

    # Patch fetch_resource_definitions to avoid real async work
    async def dummy_fetch_resource_definitions(self):
        return [
            MCPResourceDefinition(
                name="COVR",
                description="covr",
                uri="resource://COVR",
                input_schema={"type": "object", "properties": {}, "required": []},
            )
        ]

    monkeypatch.setattr(MCPDefinitionService, "fetch_resource_definitions", dummy_fetch_resource_definitions)
    result = factory._fetch_resource_definitions()
    assert result[0].name == "COVR"


def test_fetch_prompt_definitions_service_branch(monkeypatch):
    """Covers lines of MCPDefinitionService branch in _fetch_prompt_definitions."""
    factory = MCPFactory("dummy_endpoint", MCPTransportType.HTTP_STREAM)

    # Patch fetch_prompt_definitions to avoid real async work
    async def dummy_fetch_prompt_definitions(self):
        return [
            MCPPromptDefinition(
                name="COVP", description="covp", input_schema={"type": "object", "properties": {}, "required": []}
            )
        ]

    monkeypatch.setattr(MCPDefinitionService, "fetch_prompt_definitions", dummy_fetch_prompt_definitions)
    result = factory._fetch_prompt_definitions()
    assert result[0].name == "COVP"


def test_fetch_mcp_tools_rejects_persistent_session_without_event_loop(
    mocked_mcp_connection: tuple[AsyncMock, dict[str, MagicMock]],
) -> None:
    session, clients = mocked_mcp_connection
    with pytest.raises(ValueError, match="When `client_session` is provided an `event_loop` must also be supplied"):
        fetch_mcp_tools(client_session=session)

    session.initialize.assert_not_awaited()
    session.list_tools.assert_not_awaited()
    session.call_tool.assert_not_awaited()
    for client in clients.values():
        client.assert_not_called()


def test_run_tool_with_persistent_session_no_event_loop(monkeypatch):
    """A persistent session with a removed loop reports its missing owner."""
    import atomic_agents.connectors.mcp.mcp_factory as mtf

    # Setup persistent client
    class DummySessionPersistent:
        async def call_tool(self, name, arguments):
            return {"content": "should not get here"}

    client = DummySessionPersistent()
    definitions = [
        MCPToolDefinition(name="ToolCOV", description=None, input_schema={"type": "object", "properties": {}, "required": []})
    ]

    async def fake_fetch_defs(session):
        return definitions

    monkeypatch.setattr(mtf.MCPDefinitionService, "fetch_tool_definitions_from_session", staticmethod(fake_fetch_defs))
    # Create tool with persistent session and a valid event loop
    loop = asyncio.new_event_loop()
    try:
        tools = fetch_mcp_tools(None, MCPTransportType.HTTP_STREAM, client_session=client, event_loop=loop)
        tool_cls = tools[0]
        inst = tool_cls()
        # Remove the event loop to simulate the error path
        inst._event_loop = None
        with pytest.raises(RuntimeError) as exc:
            inst.run(tool_cls.input_schema(tool_name="ToolCOV"))
        # No coroutine is created when the persistent session has no event loop
        assert "persistent MCP session requires an event loop" in str(exc.value)
    finally:
        loop.close()


def test_run_resource_with_persistent_session_no_event_loop(monkeypatch):
    """A persistent session with a removed loop reports its missing owner."""
    import atomic_agents.connectors.mcp.mcp_factory as mtf

    # Setup persistent client
    class DummySessionPersistent:
        async def read_resource(self, *, uri):
            return {"content": "should not get here"}

    client = DummySessionPersistent()
    definitions = [
        MCPResourceDefinition(
            name="ResCOV",
            description=None,
            uri="resource://ResCOV",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
    ]

    async def fake_fetch_defs(session):
        return definitions

    monkeypatch.setattr(mtf.MCPDefinitionService, "fetch_resource_definitions_from_session", staticmethod(fake_fetch_defs))
    # Create resource with persistent session and a valid event loop
    loop = asyncio.new_event_loop()
    try:
        resources = fetch_mcp_resources(None, MCPTransportType.HTTP_STREAM, client_session=client, event_loop=loop)
        res_cls = resources[0]
        inst = res_cls()
        # Remove the event loop to simulate the error
        inst._event_loop = None
        with pytest.raises(RuntimeError) as exc:
            inst.read(res_cls.input_schema(resource_name="ResCOV"))
        # No coroutine is created when the persistent session has no event loop
        assert "persistent MCP session requires an event loop" in str(exc.value)
    finally:
        loop.close()


def test_run_prompt_with_persistent_session_no_event_loop(monkeypatch):
    """A persistent session with a removed loop reports its missing owner."""
    import atomic_agents.connectors.mcp.mcp_factory as mtf

    # Setup persistent client
    class DummySessionPersistent:
        async def get_prompt(self, *, name, arguments):
            return {"content": "should not get here"}

    client = DummySessionPersistent()
    definitions = [
        MCPPromptDefinition(
            name="PromptCOV", description=None, input_schema={"type": "object", "properties": {}, "required": []}
        )
    ]

    async def fake_fetch_defs(session):
        return definitions

    monkeypatch.setattr(mtf.MCPDefinitionService, "fetch_prompt_definitions_from_session", staticmethod(fake_fetch_defs))
    # Create prompt with persistent session and a valid event loop
    loop = asyncio.new_event_loop()
    try:
        prompts = fetch_mcp_prompts(None, MCPTransportType.HTTP_STREAM, client_session=client, event_loop=loop)
        prompt_cls = prompts[0]
        inst = prompt_cls()
        # Remove the event loop to simulate the error
        inst._event_loop = None
        with pytest.raises(RuntimeError) as exc:
            inst.generate(prompt_cls.input_schema(prompt_name="PromptCOV"))
        # No coroutine is created when the persistent session has no event loop
        assert "persistent MCP session requires an event loop" in str(exc.value)
    finally:
        loop.close()


def test_http_stream_connection_error_handling(monkeypatch):
    """Test HTTP stream connection error handling in MCPToolFactory."""
    from atomic_agents.connectors.mcp.mcp_definition_service import MCPDefinitionService

    # Mock MCPDefinitionService.fetch_tool_definitions to raise ConnectionError for HTTP_STREAM
    original_fetch_tools = MCPDefinitionService.fetch_tool_definitions

    async def mock_fetch_tool_definitions(self):
        if self.transport_type == MCPTransportType.HTTP_STREAM:
            raise ConnectionError("HTTP stream connection failed")
        return await original_fetch_tools(self)

    monkeypatch.setattr(MCPDefinitionService, "fetch_tool_definitions", mock_fetch_tool_definitions)

    factory = MCPFactory("http://test-endpoint", MCPTransportType.HTTP_STREAM)

    with pytest.raises(ConnectionError, match="HTTP stream connection failed"):
        factory._fetch_tool_definitions()

    original_fetch_resources = MCPDefinitionService.fetch_resource_definitions

    async def mock_fetch_resource_definitions(self):
        if self.transport_type == MCPTransportType.HTTP_STREAM:
            raise ConnectionError("HTTP stream connection failed")
        return await original_fetch_resources(self)

    monkeypatch.setattr(MCPDefinitionService, "fetch_resource_definitions", mock_fetch_resource_definitions)
    with pytest.raises(ConnectionError, match="HTTP stream connection failed"):
        factory._fetch_resource_definitions()

    original_fetch_prompts = MCPDefinitionService.fetch_prompt_definitions

    async def mock_fetch_prompt_definitions(self):
        if self.transport_type == MCPTransportType.HTTP_STREAM:
            raise ConnectionError("HTTP stream connection failed")
        return await original_fetch_prompts(self)

    monkeypatch.setattr(MCPDefinitionService, "fetch_prompt_definitions", mock_fetch_prompt_definitions)
    with pytest.raises(ConnectionError, match="HTTP stream connection failed"):
        factory._fetch_prompt_definitions()


def test_http_stream_endpoint_formatting():
    """Test that HTTP stream endpoints are properly formatted with /mcp/ suffix."""
    factory = MCPFactory("http://test-endpoint", MCPTransportType.HTTP_STREAM)

    # Verify the factory was created with correct transport type
    assert factory.transport_type == MCPTransportType.HTTP_STREAM


# Tests for fetch_mcp_tools_async function


@pytest.mark.asyncio
async def test_fetch_mcp_tools_async_with_client_session(monkeypatch):
    """Test fetch_mcp_tools_async with pre-initialized client session."""
    import atomic_agents.connectors.mcp.mcp_factory as mtf

    # Setup persistent client
    class DummySessionPersistent:
        async def call_tool(self, name, arguments):
            return {"content": "async-session-ok"}

    client = DummySessionPersistent()
    definitions = [
        MCPToolDefinition(
            name="AsyncTool", description="Test async tool", input_schema={"type": "object", "properties": {}, "required": []}
        )
    ]

    async def fake_fetch_defs(session):
        return definitions

    monkeypatch.setattr(mtf.MCPDefinitionService, "fetch_tool_definitions_from_session", staticmethod(fake_fetch_defs))

    # Call fetch_mcp_tools_async with client session
    tools = await fetch_mcp_tools_async(None, MCPTransportType.HTTP_STREAM, client_session=client)

    assert len(tools) == 1
    tool_cls = tools[0]
    # Verify the tool was created correctly
    assert hasattr(tool_cls, "mcp_tool_name")


@pytest.mark.asyncio
async def test_fetch_mcp_resources_async_with_client_session(monkeypatch):
    """Test fetch_mcp_resources_async with pre-initialized client session."""
    import atomic_agents.connectors.mcp.mcp_factory as mtf

    # Setup persistent client
    class DummySessionPersistent:
        async def read_resource(self, name, uri):
            return {"content": "async-resource-ok"}

    client = DummySessionPersistent()
    definitions = [
        MCPResourceDefinition(
            name="AsyncRes",
            description="Test async resource",
            uri="resource://AsyncRes",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
    ]

    async def fake_fetch_defs(session):
        return definitions

    monkeypatch.setattr(mtf.MCPDefinitionService, "fetch_resource_definitions_from_session", staticmethod(fake_fetch_defs))

    # Call fetch_mcp_resources_async with client session
    resources = await fetch_mcp_resources_async(None, MCPTransportType.HTTP_STREAM, client_session=client)

    assert len(resources) == 1
    res_cls = resources[0]
    # Verify the resource was created correctly
    assert hasattr(res_cls, "mcp_resource_name")


@pytest.mark.asyncio
async def test_fetch_mcp_prompts_async_with_client_session(monkeypatch):
    """Test fetch_mcp_prompts_async with pre-initialized client session."""
    import atomic_agents.connectors.mcp.mcp_factory as mtf

    # Setup persistent client
    class DummySessionPersistent:
        async def generate_prompt(self, name, arguments):
            return {"content": "async-prompt-ok"}

    client = DummySessionPersistent()
    definitions = [
        MCPPromptDefinition(
            name="AsyncPrompt",
            description="Test async prompt",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
    ]

    async def fake_fetch_defs(session):
        return definitions

    monkeypatch.setattr(mtf.MCPDefinitionService, "fetch_prompt_definitions_from_session", staticmethod(fake_fetch_defs))

    # Call fetch_mcp_prompts_async with client session
    prompts = await fetch_mcp_prompts_async(None, MCPTransportType.HTTP_STREAM, client_session=client)

    assert len(prompts) == 1
    prompt_cls = prompts[0]
    # Verify the prompt was created correctly
    assert hasattr(prompt_cls, "mcp_prompt_name")


@pytest.mark.asyncio
async def test_fetch_mcp_tools_async_without_client_session(monkeypatch):
    """Test fetch_mcp_tools_async without pre-initialized client session."""

    definitions = [
        MCPToolDefinition(
            name="AsyncTool2",
            description="Test async tool 2",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
    ]

    async def fake_fetch_defs(self):
        return definitions

    monkeypatch.setattr(MCPDefinitionService, "fetch_tool_definitions", fake_fetch_defs)

    # Call fetch_mcp_tools_async without client session
    tools = await fetch_mcp_tools_async("http://test-endpoint", MCPTransportType.HTTP_STREAM)

    assert len(tools) == 1
    tool_cls = tools[0]
    # Verify the tool was created correctly
    assert hasattr(tool_cls, "mcp_tool_name")


@pytest.mark.asyncio
async def test_fetch_mcp_resources_async_without_client_session(monkeypatch):
    """Test fetch_mcp_resources_async without pre-initialized client session."""

    definitions = [
        MCPResourceDefinition(
            name="AsyncRes2",
            description="Test async resource 2",
            uri="resource://AsyncRes2",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
    ]

    async def fake_fetch_defs(self):
        return definitions

    monkeypatch.setattr(MCPDefinitionService, "fetch_resource_definitions", fake_fetch_defs)

    # Call fetch_mcp_resources_async without client session
    resources = await fetch_mcp_resources_async("http://test-endpoint", MCPTransportType.HTTP_STREAM)

    assert len(resources) == 1
    res_cls = resources[0]
    # Verify the resource was created correctly
    assert hasattr(res_cls, "mcp_resource_name")


@pytest.mark.asyncio
async def test_fetch_mcp_prompts_async_without_client_session(monkeypatch):
    """Test fetch_mcp_prompts_async without pre-initialized client session."""

    definitions = [
        MCPPromptDefinition(
            name="AsyncPrompt2",
            description="Test async prompt 2",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
    ]

    async def fake_fetch_defs(self):
        return definitions

    monkeypatch.setattr(MCPDefinitionService, "fetch_prompt_definitions", fake_fetch_defs)

    # Call fetch_mcp_prompts_async without client session
    prompts = await fetch_mcp_prompts_async("http://test-endpoint", MCPTransportType.HTTP_STREAM)

    assert len(prompts) == 1
    prompt_cls = prompts[0]
    # Verify the prompt was created correctly
    assert hasattr(prompt_cls, "mcp_prompt_name")


@pytest.mark.asyncio
async def test_fetch_mcp_tools_async_stdio_transport(monkeypatch):
    """Test fetch_mcp_tools_async with STDIO transport."""
    definitions = [
        MCPToolDefinition(
            name="StdioAsyncTool",
            description="Test stdio async tool",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
    ]

    async def fake_fetch_defs(self):
        return definitions

    monkeypatch.setattr(MCPDefinitionService, "fetch_tool_definitions", fake_fetch_defs)

    # Call fetch_mcp_tools_async with STDIO transport
    tools = await fetch_mcp_tools_async("test-command", MCPTransportType.STDIO, working_directory="/tmp")

    assert len(tools) == 1
    tool_cls = tools[0]
    # Verify the tool was created correctly
    assert hasattr(tool_cls, "mcp_tool_name")


@pytest.mark.asyncio
async def test_fetch_mcp_resources_async_stdio_transport(monkeypatch):
    """Test fetch_mcp_resources_async with STDIO transport."""
    definitions = [
        MCPResourceDefinition(
            name="StdioAsyncRes",
            description="Test stdio async resource",
            uri="resource://StdioAsyncRes",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
    ]

    async def fake_fetch_defs(self):
        return definitions

    monkeypatch.setattr(MCPDefinitionService, "fetch_resource_definitions", fake_fetch_defs)

    # Call fetch_mcp_resources_async with STDIO transport
    resources = await fetch_mcp_resources_async("test-command", MCPTransportType.STDIO, working_directory="/tmp")

    assert len(resources) == 1
    res_cls = resources[0]
    # Verify the resource was created correctly
    assert hasattr(res_cls, "mcp_resource_name")


@pytest.mark.asyncio
async def test_fetch_mcp_prompts_async_stdio_transport(monkeypatch):
    """Test fetch_mcp_prompts_async with STDIO transport."""
    definitions = [
        MCPPromptDefinition(
            name="StdioAsyncPrompt",
            description="Test stdio async prompt",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
    ]

    async def fake_fetch_defs(self):
        return definitions

    monkeypatch.setattr(MCPDefinitionService, "fetch_prompt_definitions", fake_fetch_defs)

    # Call fetch_mcp_prompts_async with STDIO transport
    prompts = await fetch_mcp_prompts_async("test-command", MCPTransportType.STDIO, working_directory="/tmp")

    assert len(prompts) == 1
    prompt_cls = prompts[0]
    # Verify the prompt was created correctly
    assert hasattr(prompt_cls, "mcp_prompt_name")


@pytest.mark.asyncio
async def test_fetch_mcp_tools_async_empty_definitions(monkeypatch):
    """Test fetch_mcp_tools_async returns empty list when no definitions found."""

    async def fake_fetch_defs(self):
        return []

    monkeypatch.setattr(MCPDefinitionService, "fetch_tool_definitions", fake_fetch_defs)

    # Call fetch_mcp_tools_async
    tools = await fetch_mcp_tools_async("http://test-endpoint", MCPTransportType.HTTP_STREAM)

    assert tools == []


@pytest.mark.asyncio
async def test_fetch_mcp_resources_async_empty_definitions(monkeypatch):
    """Test fetch_mcp_resources_async returns empty list when no definitions found."""

    async def fake_fetch_defs(self):
        return []

    monkeypatch.setattr(MCPDefinitionService, "fetch_resource_definitions", fake_fetch_defs)

    # Call fetch_mcp_resources_async
    resources = await fetch_mcp_resources_async("http://test-endpoint", MCPTransportType.HTTP_STREAM)

    assert resources == []


@pytest.mark.asyncio
async def test_fetch_mcp_prompts_async_empty_definitions(monkeypatch):
    """Test fetch_mcp_prompts_async returns empty list when no definitions found."""

    async def fake_fetch_defs(self):
        return []

    monkeypatch.setattr(MCPDefinitionService, "fetch_prompt_definitions", fake_fetch_defs)

    # Call fetch_mcp_prompts_async
    prompts = await fetch_mcp_prompts_async("http://test-endpoint", MCPTransportType.HTTP_STREAM)

    assert prompts == []


@pytest.mark.asyncio
async def test_fetch_mcp_tools_async_connection_error(monkeypatch):
    """Test fetch_mcp_tools_async propagates connection errors."""

    async def fake_fetch_defs_error(self):
        raise ConnectionError("Failed to connect to MCP server")

    monkeypatch.setattr(MCPDefinitionService, "fetch_tool_definitions", fake_fetch_defs_error)

    # Call fetch_mcp_tools_async and expect ConnectionError
    with pytest.raises(ConnectionError, match="Failed to connect to MCP server"):
        await fetch_mcp_tools_async("http://test-endpoint", MCPTransportType.HTTP_STREAM)


@pytest.mark.asyncio
async def test_fetch_mcp_resources_async_connection_error(monkeypatch):
    """Test fetch_mcp_resources_async propagates connection errors."""

    async def fake_fetch_defs_error(self):
        raise ConnectionError("Failed to connect to MCP server")

    monkeypatch.setattr(MCPDefinitionService, "fetch_resource_definitions", fake_fetch_defs_error)

    # Call fetch_mcp_resources_async and expect ConnectionError
    with pytest.raises(ConnectionError, match="Failed to connect to MCP server"):
        await fetch_mcp_resources_async("http://test-endpoint", MCPTransportType.HTTP_STREAM)


@pytest.mark.asyncio
async def test_fetch_mcp_prompts_async_connection_error(monkeypatch):
    """Test fetch_mcp_prompts_async propagates connection errors."""

    async def fake_fetch_defs_error(self):
        raise ConnectionError("Failed to connect to MCP server")

    monkeypatch.setattr(MCPDefinitionService, "fetch_prompt_definitions", fake_fetch_defs_error)

    # Call fetch_mcp_prompts_async and expect ConnectionError
    with pytest.raises(ConnectionError, match="Failed to connect to MCP server"):
        await fetch_mcp_prompts_async("http://test-endpoint", MCPTransportType.HTTP_STREAM)


@pytest.mark.asyncio
async def test_fetch_mcp_tools_async_runtime_error(monkeypatch):
    """Test fetch_mcp_tools_async propagates runtime errors."""

    async def fake_fetch_defs_error(self):
        raise RuntimeError("Unexpected error during fetching")

    monkeypatch.setattr(MCPDefinitionService, "fetch_tool_definitions", fake_fetch_defs_error)

    # Call fetch_mcp_tools_async and expect RuntimeError
    with pytest.raises(RuntimeError, match="Unexpected error during fetching"):
        await fetch_mcp_tools_async("http://test-endpoint", MCPTransportType.HTTP_STREAM)


@pytest.mark.asyncio
async def test_fetch_mcp_resources_async_runtime_error(monkeypatch):
    """Test fetch_mcp_resources_async propagates runtime errors."""

    async def fake_fetch_defs_error(self):
        raise RuntimeError("Unexpected error during fetching")

    monkeypatch.setattr(MCPDefinitionService, "fetch_resource_definitions", fake_fetch_defs_error)

    # Call fetch_mcp_resources_async and expect RuntimeError
    with pytest.raises(RuntimeError, match="Unexpected error during fetching"):
        await fetch_mcp_resources_async("http://test-endpoint", MCPTransportType.HTTP_STREAM)


@pytest.mark.asyncio
async def test_fetch_mcp_prompts_async_runtime_error(monkeypatch):
    """Test fetch_mcp_prompts_async propagates runtime errors."""

    async def fake_fetch_defs_error(self):
        raise RuntimeError("Unexpected error during fetching")

    monkeypatch.setattr(MCPDefinitionService, "fetch_prompt_definitions", fake_fetch_defs_error)

    # Call fetch_mcp_prompts_async and expect RuntimeError
    with pytest.raises(RuntimeError, match="Unexpected error during fetching"):
        await fetch_mcp_prompts_async("http://test-endpoint", MCPTransportType.HTTP_STREAM)


@pytest.mark.asyncio
async def test_fetch_mcp_tools_async_with_working_directory(monkeypatch):
    """Test fetch_mcp_tools_async with working directory parameter."""
    definitions = [
        MCPToolDefinition(
            name="WorkingDirTool",
            description="Test tool with working dir",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
    ]

    async def fake_fetch_defs(self):
        return definitions

    monkeypatch.setattr(MCPDefinitionService, "fetch_tool_definitions", fake_fetch_defs)

    # Call fetch_mcp_tools_async with working directory
    tools = await fetch_mcp_tools_async("test-command", MCPTransportType.STDIO, working_directory="/custom/working/dir")

    assert len(tools) == 1
    tool_cls = tools[0]
    # Verify the tool was created correctly
    assert hasattr(tool_cls, "mcp_tool_name")


@pytest.mark.asyncio
async def test_fetch_mcp_resources_async_with_working_directory(monkeypatch):
    """Test fetch_mcp_resources_async with working directory parameter."""
    definitions = [
        MCPResourceDefinition(
            name="WorkingDirRes",
            description="Test resource with working dir",
            uri="resource://WorkingDirRes",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
    ]

    async def fake_fetch_defs(self):
        return definitions

    monkeypatch.setattr(MCPDefinitionService, "fetch_resource_definitions", fake_fetch_defs)

    # Call fetch_mcp_resources_async with working directory
    resources = await fetch_mcp_resources_async(
        "test-command", MCPTransportType.STDIO, working_directory="/custom/working/dir"
    )

    assert len(resources) == 1
    res_cls = resources[0]
    # Verify the resource was created correctly
    assert hasattr(res_cls, "mcp_resource_name")


@pytest.mark.asyncio
async def test_fetch_mcp_prompts_async_with_working_directory(monkeypatch):
    """Test fetch_mcp_prompts_async with working directory parameter."""
    definitions = [
        MCPPromptDefinition(
            name="WorkingDirPrompt",
            description="Test prompt with working dir",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
    ]

    async def fake_fetch_defs(self):
        return definitions

    monkeypatch.setattr(MCPDefinitionService, "fetch_prompt_definitions", fake_fetch_defs)

    # Call fetch_mcp_prompts_async with working directory
    prompts = await fetch_mcp_prompts_async("test-command", MCPTransportType.STDIO, working_directory="/custom/working/dir")

    assert len(prompts) == 1
    prompt_cls = prompts[0]
    # Verify the prompt was created correctly
    assert hasattr(prompt_cls, "mcp_prompt_name")


@pytest.mark.asyncio
async def test_fetch_mcp_tools_async_session_error_propagation(monkeypatch):
    """Test fetch_mcp_tools_async with client session error propagation."""
    import atomic_agents.connectors.mcp.mcp_factory as mtf

    class DummySessionPersistent:
        async def call_tool(self, name, arguments):
            return {"content": "session-ok"}

    client = DummySessionPersistent()

    async def fake_fetch_defs_error(session):
        raise ValueError("Session fetch error")

    monkeypatch.setattr(mtf.MCPDefinitionService, "fetch_tool_definitions_from_session", staticmethod(fake_fetch_defs_error))

    # Call fetch_mcp_tools_async with client session and expect error
    with pytest.raises(ValueError, match="Session fetch error"):
        await fetch_mcp_tools_async(None, MCPTransportType.HTTP_STREAM, client_session=client)


@pytest.mark.asyncio
async def test_fetch_mcp_resources_async_session_error_propagation(monkeypatch):
    """Test fetch_mcp_resources_async with client session error propagation."""
    import atomic_agents.connectors.mcp.mcp_factory as mtf

    class DummySessionPersistent:
        async def read_resource(self, name, uri):
            return {"content": "session-ok"}

    client = DummySessionPersistent()

    async def fake_fetch_defs_error(session):
        raise ValueError("Session fetch error")

    monkeypatch.setattr(
        mtf.MCPDefinitionService, "fetch_resource_definitions_from_session", staticmethod(fake_fetch_defs_error)
    )

    # Call fetch_mcp_resources_async with client session and expect error
    with pytest.raises(ValueError, match="Session fetch error"):
        await fetch_mcp_resources_async(None, MCPTransportType.HTTP_STREAM, client_session=client)


@pytest.mark.asyncio
async def test_fetch_mcp_prompts_async_session_error_propagation(monkeypatch):
    """Test fetch_mcp_prompts_async with client session error propagation."""
    import atomic_agents.connectors.mcp.mcp_factory as mtf

    class DummySessionPersistent:
        async def generate_prompt(self, name, arguments):
            return {"content": "session-ok"}

    client = DummySessionPersistent()

    async def fake_fetch_defs_error(session):
        raise ValueError("Session fetch error")

    monkeypatch.setattr(mtf.MCPDefinitionService, "fetch_prompt_definitions_from_session", staticmethod(fake_fetch_defs_error))

    # Call fetch_mcp_prompts_async with client session and expect error
    with pytest.raises(ValueError, match="Session fetch error"):
        await fetch_mcp_prompts_async(None, MCPTransportType.HTTP_STREAM, client_session=client)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport_type", [MCPTransportType.HTTP_STREAM, MCPTransportType.STDIO, MCPTransportType.SSE])
async def test_fetch_mcp_tools_async_all_transport_types(monkeypatch, transport_type):
    """Test fetch_mcp_tools_async with all supported transport types."""
    definitions = [
        MCPToolDefinition(
            name=f"Tool_{transport_type.value}",
            description=f"Test tool for {transport_type.value}",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
    ]

    async def fake_fetch_defs(self):
        return definitions

    monkeypatch.setattr(MCPDefinitionService, "fetch_tool_definitions", fake_fetch_defs)

    # Determine endpoint based on transport type
    endpoint = "test-command" if transport_type == MCPTransportType.STDIO else "http://test-endpoint"
    working_dir = "/tmp" if transport_type == MCPTransportType.STDIO else None

    # Call fetch_mcp_tools_async with different transport types
    tools = await fetch_mcp_tools_async(endpoint, transport_type, working_directory=working_dir)

    assert len(tools) == 1
    tool_cls = tools[0]
    # Verify the tool was created correctly
    assert hasattr(tool_cls, "mcp_tool_name")


@pytest.mark.asyncio
@pytest.mark.parametrize("transport_type", [MCPTransportType.HTTP_STREAM, MCPTransportType.STDIO, MCPTransportType.SSE])
async def test_fetch_mcp_resources_async_all_transport_types(monkeypatch, transport_type):
    """Test fetch_mcp_resources_async with all supported transport types."""
    definitions = [
        MCPResourceDefinition(
            name=f"Res_{transport_type.value}",
            description=f"Test resource for {transport_type.value}",
            uri=f"resource://Res_{transport_type.value}",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
    ]

    async def fake_fetch_defs(self):
        return definitions

    monkeypatch.setattr(MCPDefinitionService, "fetch_resource_definitions", fake_fetch_defs)

    # Determine endpoint based on transport type
    endpoint = "test-command" if transport_type == MCPTransportType.STDIO else "http://test-endpoint"
    working_dir = "/tmp" if transport_type == MCPTransportType.STDIO else None

    # Call fetch_mcp_resources_async with different transport types
    resources = await fetch_mcp_resources_async(endpoint, transport_type, working_directory=working_dir)

    assert len(resources) == 1
    res_cls = resources[0]
    # Verify the resource was created correctly
    assert hasattr(res_cls, "mcp_resource_name")


@pytest.mark.asyncio
@pytest.mark.parametrize("transport_type", [MCPTransportType.HTTP_STREAM, MCPTransportType.STDIO, MCPTransportType.SSE])
async def test_fetch_mcp_prompts_async_all_transport_types(monkeypatch, transport_type):
    """Test fetch_mcp_prompts_async with all supported transport types."""
    definitions = [
        MCPPromptDefinition(
            name=f"Prompt_{transport_type.value}",
            description=f"Test prompt for {transport_type.value}",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
    ]

    async def fake_fetch_defs(self):
        return definitions

    monkeypatch.setattr(MCPDefinitionService, "fetch_prompt_definitions", fake_fetch_defs)

    # Determine endpoint based on transport type
    endpoint = "test-command" if transport_type == MCPTransportType.STDIO else "http://test-endpoint"
    working_dir = "/tmp" if transport_type == MCPTransportType.STDIO else None

    # Call fetch_mcp_prompts_async with different transport types
    prompts = await fetch_mcp_prompts_async(endpoint, transport_type, working_directory=working_dir)

    assert len(prompts) == 1
    prompt_cls = prompts[0]
    # Verify the prompt was created correctly
    assert hasattr(prompt_cls, "mcp_prompt_name")


@pytest.mark.asyncio
async def test_fetch_mcp_tools_async_multiple_tools(monkeypatch):
    """Test fetch_mcp_tools_async with multiple tool definitions."""
    definitions = [
        MCPToolDefinition(
            name="Tool1", description="First tool", input_schema={"type": "object", "properties": {}, "required": []}
        ),
        MCPToolDefinition(
            name="Tool2",
            description="Second tool",
            input_schema={"type": "object", "properties": {"param": {"type": "string"}}, "required": ["param"]},
        ),
        MCPToolDefinition(
            name="Tool3",
            description="Third tool",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                "required": ["x", "y"],
            },
        ),
    ]

    async def fake_fetch_defs(self):
        return definitions

    monkeypatch.setattr(MCPDefinitionService, "fetch_tool_definitions", fake_fetch_defs)

    # Call fetch_mcp_tools_async
    tools = await fetch_mcp_tools_async("http://test-endpoint", MCPTransportType.HTTP_STREAM)

    assert len(tools) == 3
    tool_names = [getattr(tool_cls, "mcp_tool_name", None) for tool_cls in tools]
    assert "Tool1" in tool_names
    assert "Tool2" in tool_names
    assert "Tool3" in tool_names


@pytest.mark.asyncio
async def test_fetch_mcp_resources_async_multiple_resources(monkeypatch):
    """Test fetch_mcp_resources_async with multiple resource definitions."""
    definitions = [
        MCPResourceDefinition(
            name="Res1",
            description="First resource",
            uri="resource://Res1",
            input_schema={"type": "object", "properties": {}, "required": []},
        ),
        MCPResourceDefinition(
            name="Res2",
            description="Second resource",
            uri="resource://Res2",
            input_schema={"type": "object", "properties": {"param": {"type": "string"}}, "required": ["param"]},
        ),
        MCPResourceDefinition(
            name="Res3",
            description="Third resource",
            uri="resource://Res3",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                "required": ["x", "y"],
            },
        ),
    ]

    async def fake_fetch_defs(self):
        return definitions

    monkeypatch.setattr(MCPDefinitionService, "fetch_resource_definitions", fake_fetch_defs)

    # Call fetch_mcp_resources_async
    resources = await fetch_mcp_resources_async("http://test-endpoint", MCPTransportType.HTTP_STREAM)

    assert len(resources) == 3
    res_names = [getattr(res_cls, "mcp_resource_name", None) for res_cls in resources]
    assert "Res1" in res_names
    assert "Res2" in res_names
    assert "Res3" in res_names


@pytest.mark.asyncio
async def test_fetch_mcp_prompts_async_multiple_prompts(monkeypatch):
    """Test fetch_mcp_prompts_async with multiple prompt definitions."""
    definitions = [
        MCPPromptDefinition(
            name="Prompt1", description="First prompt", input_schema={"type": "object", "properties": {}, "required": []}
        ),
        MCPPromptDefinition(
            name="Prompt2",
            description="Second prompt",
            input_schema={"type": "object", "properties": {"param": {"type": "string"}}, "required": ["param"]},
        ),
        MCPPromptDefinition(
            name="Prompt3",
            description="Third prompt",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                "required": ["x", "y"],
            },
        ),
    ]

    async def fake_fetch_defs(self):
        return definitions

    monkeypatch.setattr(MCPDefinitionService, "fetch_prompt_definitions", fake_fetch_defs)

    # Call fetch_mcp_prompts_async
    prompts = await fetch_mcp_prompts_async("http://test-endpoint", MCPTransportType.HTTP_STREAM)

    assert len(prompts) == 3
    prompt_names = [getattr(prompt_cls, "mcp_prompt_name", None) for prompt_cls in prompts]
    assert "Prompt1" in prompt_names
    assert "Prompt2" in prompt_names
    assert "Prompt3" in prompt_names


# Tests for arun functionality


def test_arun_attribute_exists_on_generated_tools(monkeypatch):
    """Test that dynamically generated tools have the arun attribute."""
    input_schema = {"type": "object", "properties": {}, "required": []}
    definitions = [MCPToolDefinition(name="TestTool", description="test", input_schema=input_schema)]
    monkeypatch.setattr(MCPFactory, "_fetch_tool_definitions", lambda self: definitions)

    # Create tool
    tools = fetch_mcp_tools("http://test", MCPTransportType.HTTP_STREAM)
    tool_cls = tools[0]

    # Verify the class has arun as an attribute
    assert hasattr(tool_cls, "arun")

    # Verify instance has arun
    inst = tool_cls()
    assert hasattr(inst, "arun")
    assert callable(getattr(inst, "arun"))


def test_arun_attribute_exists_on_generated_resources(monkeypatch):
    """Test that dynamically generated resources have the arun attribute."""
    input_schema = {"type": "object", "properties": {}, "required": []}
    definitions = [
        MCPResourceDefinition(name="TestRes", description="test", uri="resource://TestRes", input_schema=input_schema)
    ]
    monkeypatch.setattr(MCPFactory, "_fetch_resource_definitions", lambda self: definitions)

    # Create resource
    resources = fetch_mcp_resources("http://test", MCPTransportType.HTTP_STREAM)
    res_cls = resources[0]

    # Verify the class has aread as an attribute
    assert hasattr(res_cls, "aread")

    # Verify instance has aread
    inst = res_cls()
    assert hasattr(inst, "aread")
    assert callable(getattr(inst, "aread"))


def test_arun_attribute_exists_on_generated_prompts(monkeypatch):
    """Test that dynamically generated prompts have the arun attribute."""
    input_schema = {"type": "object", "properties": {}, "required": []}
    definitions = [MCPPromptDefinition(name="TestPrompt", description="test", input_schema=input_schema)]
    monkeypatch.setattr(MCPFactory, "_fetch_prompt_definitions", lambda self: definitions)

    # Create prompt
    prompts = fetch_mcp_prompts("http://test", MCPTransportType.HTTP_STREAM)
    prompt_cls = prompts[0]

    # Verify the class has aread as an attribute
    assert hasattr(prompt_cls, "agenerate")

    # Verify instance has aread
    inst = prompt_cls()
    assert hasattr(inst, "agenerate")
    assert callable(getattr(inst, "agenerate"))


@pytest.mark.asyncio
async def test_arun_tool_async_execution(monkeypatch):
    """Test that arun method executes tool asynchronously."""
    import atomic_agents.connectors.mcp.mcp_factory as mtf

    class DummyTransportCM:
        def __init__(self, ret):
            self.ret = ret

        async def __aenter__(self):
            return self.ret

        async def __aexit__(self, exc_type, exc, tb):
            pass

    def dummy_http_client(endpoint):
        return DummyTransportCM((None, None, None))

    class DummySessionCM:
        def __init__(self, rs=None, ws=None, *args):
            pass

        async def initialize(self):
            pass

        async def call_tool(self, name, arguments):
            return {"content": f"async-{name}-{arguments}-ok"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr(mtf, "streamablehttp_client", dummy_http_client)
    monkeypatch.setattr(mtf, "ClientSession", DummySessionCM)

    # Prepare definitions
    input_schema = {"type": "object", "properties": {}, "required": []}
    definitions = [MCPToolDefinition(name="AsyncTool", description="async test", input_schema=input_schema)]
    monkeypatch.setattr(MCPFactory, "_fetch_tool_definitions", lambda self: definitions)

    # Create tool and test arun
    tools = fetch_mcp_tools("http://test", MCPTransportType.HTTP_STREAM)
    tool_cls = tools[0]
    inst = tool_cls()

    # Test arun execution
    arun_method = getattr(inst, "arun")  # type: ignore
    params = tool_cls.input_schema(tool_name="AsyncTool")  # type: ignore
    result = await arun_method(params)
    assert result.result == "async-AsyncTool-{}-ok"


@pytest.mark.asyncio
async def test_aread_resource_async_execution(monkeypatch):
    """Test that aread method executes resource asynchronously."""
    import atomic_agents.connectors.mcp.mcp_factory as mtf

    class DummyTransportCM:
        def __init__(self, ret):
            self.ret = ret

        async def __aenter__(self):
            return self.ret

        async def __aexit__(self, exc_type, exc, tb):
            pass

    def dummy_http_client(endpoint):
        return DummyTransportCM((None, None, None))

    class DummySessionCM:
        def __init__(self, rs=None, ws=None, *args):
            pass

        async def initialize(self):
            pass

        async def read_resource(self, uri):
            # If uri is resource://AsyncRes/{id}, name is AsyncRes
            name = uri.split("/")[2].split("-")[0]
            return {"content": f"async-{name}-ok"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr(mtf, "streamablehttp_client", dummy_http_client)
    monkeypatch.setattr(mtf, "ClientSession", DummySessionCM)

    # Prepare definitions
    input_schema = {"type": "object", "properties": {}, "required": []}
    definitions = [
        MCPResourceDefinition(name="AsyncRes", description="async test", uri="resource://AsyncRes", input_schema=input_schema)
    ]
    monkeypatch.setattr(MCPFactory, "_fetch_resource_definitions", lambda self: definitions)

    # Create resource and test aread
    resources = fetch_mcp_resources("http://test", MCPTransportType.HTTP_STREAM)
    res_cls = resources[0]
    inst = res_cls()

    # Test aread execution
    aread_method = getattr(inst, "aread")  # type: ignore
    params = res_cls.input_schema(resource_name="AsyncRes")  # type: ignore
    result = await aread_method(params)
    assert result.content["content"] == "async-AsyncRes-ok"


@pytest.mark.asyncio
async def test_agenerate_prompt_async_execution(monkeypatch):
    """Test that agenerate method executes prompt asynchronously."""
    import atomic_agents.connectors.mcp.mcp_factory as mtf

    class DummyTransportCM:
        def __init__(self, ret):
            self.ret = ret

        async def __aenter__(self):
            return self.ret

        async def __aexit__(self, exc_type, exc, tb):
            pass

    def dummy_http_client(endpoint):
        return DummyTransportCM((None, None, None))

    class DummySessionCM:
        def __init__(self, rs=None, ws=None, *args):
            pass

        async def initialize(self):
            pass

        async def get_prompt(self, *, name, arguments):
            class Msg(BaseModel):
                content: str

            return {"messages": [Msg(content=f"async-{name}-{arguments}-ok")]}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr(mtf, "streamablehttp_client", dummy_http_client)
    monkeypatch.setattr(mtf, "ClientSession", DummySessionCM)

    # Prepare definitions
    input_schema = {"type": "object", "properties": {}, "required": []}
    definitions = [MCPPromptDefinition(name="AsyncPrompt", description="async test", input_schema=input_schema)]
    monkeypatch.setattr(MCPFactory, "_fetch_prompt_definitions", lambda self: definitions)

    # Create prompt and test agenerate
    prompts = fetch_mcp_prompts("http://test", MCPTransportType.HTTP_STREAM)
    prompt_cls = prompts[0]
    inst = prompt_cls()

    # Test agenerate execution
    agenerate_method = getattr(inst, "agenerate")  # type: ignore
    params = prompt_cls.input_schema(prompt_name="AsyncPrompt")  # type: ignore
    result = await agenerate_method(params)
    assert result.content == "async-AsyncPrompt-{}-ok"


@pytest.mark.asyncio
async def test_arun_error_handling(monkeypatch):
    """Test that arun properly handles and wraps errors."""
    import atomic_agents.connectors.mcp.mcp_factory as mtf

    class DummyTransportCM:
        def __init__(self, ret):
            self.ret = ret

        async def __aenter__(self):
            return self.ret

        async def __aexit__(self, exc_type, exc, tb):
            pass

    def dummy_http_client(endpoint):
        return DummyTransportCM((None, None, None))

    class ErrorSessionCM:
        def __init__(self, rs=None, ws=None, *args):
            pass

        async def initialize(self):
            pass

        async def call_tool(self, name, arguments):
            raise RuntimeError("Tool execution failed")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr(mtf, "streamablehttp_client", dummy_http_client)
    monkeypatch.setattr(mtf, "ClientSession", ErrorSessionCM)

    # Prepare definitions
    input_schema = {"type": "object", "properties": {}, "required": []}
    definitions = [MCPToolDefinition(name="ErrorTool", description="error test", input_schema=input_schema)]
    monkeypatch.setattr(MCPFactory, "_fetch_tool_definitions", lambda self: definitions)

    # Create tool and test arun error handling
    tools = fetch_mcp_tools("http://test", MCPTransportType.HTTP_STREAM)
    tool_cls = tools[0]
    inst = tool_cls()

    # Test that arun properly wraps errors
    arun_method = getattr(inst, "arun")  # type: ignore
    params = tool_cls.input_schema(tool_name="ErrorTool")  # type: ignore
    with pytest.raises(RuntimeError) as exc_info:
        await arun_method(params)
    assert "Failed to execute MCP tool 'ErrorTool'" in str(exc_info.value)


@pytest.mark.asyncio
async def test_resource_aread_error_handling(monkeypatch):
    """Test that aread properly handles and wraps errors."""
    import atomic_agents.connectors.mcp.mcp_factory as mtf

    class DummyTransportCM:
        def __init__(self, ret):
            self.ret = ret

        async def __aenter__(self):
            return self.ret

        async def __aexit__(self, exc_type, exc, tb):
            pass

    def dummy_http_client(endpoint):
        return DummyTransportCM((None, None, None))

    class ErrorSessionCM:
        def __init__(self, rs=None, ws=None, *args):
            pass

        async def initialize(self):
            pass

        async def read_resource(self, uri):
            raise RuntimeError("Resource read failed")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr(mtf, "streamablehttp_client", dummy_http_client)
    monkeypatch.setattr(mtf, "ClientSession", ErrorSessionCM)

    # Prepare definitions
    input_schema = {"type": "object", "properties": {}, "required": []}
    definitions = [
        MCPResourceDefinition(name="ErrorRes", description="error test", uri="resource://ErrorRes", input_schema=input_schema)
    ]
    monkeypatch.setattr(MCPFactory, "_fetch_resource_definitions", lambda self: definitions)

    # Create resource and test aread error handling
    resources = fetch_mcp_resources("http://test", MCPTransportType.HTTP_STREAM)
    res_cls = resources[0]
    inst = res_cls()

    # Test that aread properly wraps errors
    aread_method = getattr(inst, "aread")  # type: ignore
    params = res_cls.input_schema(resource_name="ErrorRes")  # type: ignore
    with pytest.raises(RuntimeError) as exc_info:
        await aread_method(params)
    assert "Failed to read MCP resource 'ErrorRes'" in str(exc_info.value)


@pytest.mark.asyncio
async def test_prompt_agenerate_error_handling(monkeypatch):
    """Test that agenerate properly handles and wraps errors."""
    import atomic_agents.connectors.mcp.mcp_factory as mtf

    class DummyTransportCM:
        def __init__(self, ret):
            self.ret = ret

        async def __aenter__(self):
            return self.ret

        async def __aexit__(self, exc_type, exc, tb):
            pass

    def dummy_http_client(endpoint):
        return DummyTransportCM((None, None, None))

    class ErrorSessionCM:
        def __init__(self, rs=None, ws=None, *args):
            pass

        async def initialize(self):
            pass

        async def get_prompt(self, *, name, arguments):
            raise RuntimeError("Prompt generation failed")

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

    monkeypatch.setattr(mtf, "streamablehttp_client", dummy_http_client)
    monkeypatch.setattr(mtf, "ClientSession", ErrorSessionCM)

    # Prepare definitions
    input_schema = {"type": "object", "properties": {}, "required": []}
    definitions = [MCPPromptDefinition(name="ErrorPrompt", description="error test", input_schema=input_schema)]
    monkeypatch.setattr(MCPFactory, "_fetch_prompt_definitions", lambda self: definitions)

    # Create prompt and test agenerate error handling
    prompts = fetch_mcp_prompts("http://test", MCPTransportType.HTTP_STREAM)
    prompt_cls = prompts[0]
    inst = prompt_cls()

    # Test that agenerate properly wraps errors
    agenerate_method = getattr(inst, "agenerate")  # type: ignore
    params = prompt_cls.input_schema(prompt_name="ErrorPrompt")  # type: ignore
    with pytest.raises(RuntimeError) as exc_info:
        await agenerate_method(params)
    assert "Failed to get MCP prompt 'ErrorPrompt'" in str(exc_info.value)


class StructuredModelResult(BaseModel):
    structuredContent: BaseModel | dict[str, str] | str | None


class NamedToolPayload(BaseModel):
    name: str


class TextAndDataContent(BaseModel):
    text: str | int | None
    data: dict[str, str]


@pytest.fixture
def named_output_tool_session(mocked_mcp_connection: tuple[AsyncMock, dict[str, MagicMock]]) -> AsyncMock:
    session, _ = mocked_mcp_connection
    session.list_tools.return_value.tools[0].outputSchema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }
    return session


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_result,expected_name",
    [
        (CallToolResult(content=[TextContent(type="text", text='{"name":"text"}')], structuredContent={"name": "sdk"}), "sdk"),
        (CallToolResult(content=[TextContent(type="text", text='{"name":"text"}')]), "text"),
        (StructuredModelResult(structuredContent=NamedToolPayload(name="model")), "model"),
        (
            MockContentResult(content=[MockContentItem(text='{"name":"first"}'), MockContentItem(text='{"name":"second"}')]),
            "first",
        ),
        (MockContentResult(content=(MockContentItemWithData(data={"name": "tuple"}),)), "tuple"),
        (MockContentResult(content=[TextAndDataContent(text="[]", data={"name": "data"})]), "data"),
        (MockContentResult(content=[TextAndDataContent(text="invalid" * 40, data={"name": "long"})]), "long"),
        (MockContentResult(content=[TextAndDataContent(text="invalid", data={"name": "short"})]), "short"),
        (MockContentResult(content=[TextAndDataContent(text=None, data={"name": "none-text"})]), "none-text"),
        ({"structuredContent": {"name": "structured"}, "content": {"name": "content"}}, "structured"),
        ({"content": {"name": "content"}, "name": "raw"}, "content"),
        ({"isError": True, "name": "test_value", "content": "article"}, "test_value"),
    ],
)
async def test_typed_tool_conversion_preserves_precedence(
    named_output_tool_session: AsyncMock, tool_result: BaseModel | dict[str, object], expected_name: str
) -> None:
    named_output_tool_session.call_tool.return_value = tool_result
    tool_class = (await fetch_mcp_tools_async(client_session=named_output_tool_session))[0]
    result = await tool_class().arun(tool_class.input_schema(tool_name="EchoTool", query="hello"))
    assert result.model_dump() == {"name": expected_name}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_result,cause_type,message",
    [
        (StructuredModelResult(structuredContent="invalid"), TypeError, "structuredContent with unexpected type str"),
        (StructuredModelResult(structuredContent=""), TypeError, "structuredContent with unexpected type str"),
        (MockContentResult(content=[]), ValueError, "unparseable result"),
        (MockContentResult(content=[MockContentItem(text="invalid")]), ValueError, "unparseable result"),
        (
            MockContentResult(content=[TextAndDataContent(text='{"wrong":"field"}', data={"name": "ignored"})]),
            ValueError,
            "validation error",
        ),
        ({"structuredContent": None, "name": "ignored"}, TypeError, "mapping"),
        ("raw string", ValueError, "unparseable result"),
    ],
)
async def test_typed_tool_conversion_preserves_wrapped_failures(
    named_output_tool_session: AsyncMock,
    tool_result: BaseModel | dict[str, object] | str,
    cause_type: type[Exception],
    message: str,
) -> None:
    named_output_tool_session.call_tool.return_value = tool_result
    tool_class = (await fetch_mcp_tools_async(client_session=named_output_tool_session))[0]
    with pytest.raises(RuntimeError, match="Failed to execute MCP tool 'EchoTool':") as failure:
        await tool_class().arun(tool_class.input_schema(tool_name="EchoTool", query="hello"))
    assert isinstance(failure.value.__cause__, cause_type)
    assert message in str(failure.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_result,expected_content",
    [
        (
            CallToolResult(content=[TextContent(type="text", text="first"), TextContent(type="text", text="second")]),
            [TextContent(type="text", text="first"), TextContent(type="text", text="second")],
        ),
        ({"isError": True, "name": "test_value", "content": "article"}, "article"),
        ({"name": "raw"}, {"name": "raw"}),
        (123, 123),
    ],
)
async def test_generic_tool_conversion_preserves_content(
    mocked_mcp_connection: tuple[AsyncMock, dict[str, MagicMock]],
    tool_result: BaseModel | dict[str, object] | int,
    expected_content: list[TextContent] | dict[str, str] | str | int,
) -> None:
    session, _ = mocked_mcp_connection
    session.call_tool.return_value = tool_result
    tool_class = (await fetch_mcp_tools_async(client_session=session))[0]
    result = await tool_class().arun(tool_class.input_schema(tool_name="EchoTool", query="hello"))
    assert result.result == expected_content


@pytest.fixture
def generated_resource() -> BaseResource:
    definition = MCPResourceDefinition(
        name="ReadItem",
        description="Read an item",
        uri="resource://items/{item}",
        input_schema={"type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]},
        mime_type="text/plain",
    )
    return MCPFactory("http://example.test", MCPTransportType.SSE)._create_resource_classes([definition])[0]()


@pytest.fixture
def generated_prompt() -> BasePrompt:
    definition = MCPPromptDefinition(
        name="DescribeItem",
        description="Describe an item",
        input_schema={"type": "object", "properties": {"item": {"type": "string"}}, "required": ["item"]},
    )
    return MCPFactory("http://example.test", MCPTransportType.SSE)._create_prompt_classes([definition])[0]()


@pytest.mark.parametrize(
    "resource_result",
    [ReadResourceResult(contents=[]), {"contents": [], "mime_type": "text/plain"}],
    ids=["sdk-empty", "mapping-empty"],
)
def test_generated_resource_sse_closes_contexts(
    mocked_mcp_connection: tuple[AsyncMock, dict[str, MagicMock]],
    generated_resource: BaseResource,
    resource_result: ReadResourceResult | dict[str, object],
) -> None:
    session, clients = mocked_mcp_connection
    session.read_resource.return_value = resource_result
    result = generated_resource.read(generated_resource.input_schema(resource_name="ReadItem", item="entry"))
    assert result.model_dump() == {"content": [], "mime_type": "text/plain"}
    session.read_resource.assert_awaited_once_with(uri="resource://items/entry")
    clients["sse_client"].assert_called_once_with("http://example.test/sse")
    session.initialize.assert_awaited_once_with()
    session.__aexit__.assert_awaited_once_with(session, None, None, None)
    transport = clients["sse_client"].return_value
    transport.__aexit__.assert_awaited_once_with(transport, None, None, None)


@pytest.mark.parametrize(
    "mime_fields,expected_mime", [({"mime_type": "application/json"}, "application/json"), ({}, "text/plain")]
)
def test_generated_resource_mapping_adapter_preserves_contents_and_mime(
    mocked_mcp_connection: tuple[AsyncMock, dict[str, MagicMock]],
    generated_resource: BaseResource,
    mime_fields: dict[str, str],
    expected_mime: str,
) -> None:
    session, _ = mocked_mcp_connection
    session.read_resource.return_value = {"contents": [{"text": "item"}], **mime_fields}
    result = generated_resource.read(generated_resource.input_schema(resource_name="ReadItem", item="entry"))
    assert result.content == [{"text": "item"}]
    assert result.mime_type == expected_mime
    session.read_resource.assert_awaited_once_with(uri="resource://items/entry")


@pytest.mark.parametrize(
    "content",
    [TextContent(type="text", text="second"), ImageContent(type="image", data="aW1hZ2U=", mimeType="image/png")],
    ids=["ordered-text", "nontext-fallback"],
)
def test_generated_prompt_sse_preserves_sdk_message_order_and_closes_contexts(
    mocked_mcp_connection: tuple[AsyncMock, dict[str, MagicMock]],
    generated_prompt: BasePrompt,
    content: TextContent | ImageContent,
) -> None:
    session, clients = mocked_mcp_connection
    session.get_prompt.return_value = GetPromptResult(
        messages=[
            PromptMessage(role="user", content=TextContent(type="text", text="first")),
            PromptMessage(role="assistant", content=content),
        ]
    )
    result = generated_prompt.generate(generated_prompt.input_schema(prompt_name="DescribeItem", item="entry"))
    assert result.content == f"first\n\n{getattr(content, 'text', str(content))}"
    session.get_prompt.assert_awaited_once_with(name="DescribeItem", arguments={"item": "entry"})
    clients["sse_client"].assert_called_once_with("http://example.test/sse")
    session.initialize.assert_awaited_once_with()
    session.__aexit__.assert_awaited_once_with(session, None, None, None)
    transport = clients["sse_client"].return_value
    transport.__aexit__.assert_awaited_once_with(transport, None, None, None)


@pytest.mark.parametrize(
    "messages,expected_content",
    [
        ([{"content": "first"}, {"content": {"text": "second"}}], "first\n\nsecond"),
        ([{"content": {}}, {"text": "direct"}, 17], "\n\ndirect\n\n17"),
    ],
    ids=["string-and-text-mapping", "empty-and-direct-fallbacks"],
)
def test_generated_prompt_mapping_adapter_renders_messages(
    mocked_mcp_connection: tuple[AsyncMock, dict[str, MagicMock]],
    generated_prompt: BasePrompt,
    messages: list[dict[str, object] | int],
    expected_content: str,
) -> None:
    session, _ = mocked_mcp_connection
    session.get_prompt.return_value = {"messages": messages}
    result = generated_prompt.generate(generated_prompt.input_schema(prompt_name="DescribeItem", item="entry"))
    assert result.content == expected_content
    session.get_prompt.assert_awaited_once_with(name="DescribeItem", arguments={"item": "entry"})


@pytest.mark.parametrize("prompt_result", [{}, "missing messages"], ids=["mapping", "plain"])
def test_generated_prompt_rejects_missing_messages_and_closes_contexts(
    mocked_mcp_connection: tuple[AsyncMock, dict[str, MagicMock]],
    generated_prompt: BasePrompt,
    prompt_result: dict[str, object] | str,
) -> None:
    session, clients = mocked_mcp_connection
    session.get_prompt.return_value = prompt_result
    with pytest.raises(
        RuntimeError, match="Failed to get MCP prompt 'DescribeItem': Prompt response has no messages."
    ) as failure:
        generated_prompt.generate(generated_prompt.input_schema(prompt_name="DescribeItem", item="entry"))
    assert type(failure.value.__cause__) is Exception
    assert str(failure.value.__cause__) == "Prompt response has no messages."
    session.get_prompt.assert_awaited_once_with(name="DescribeItem", arguments={"item": "entry"})
    session.__aexit__.assert_awaited_once_with(session, None, None, None)
    transport = clients["sse_client"].return_value
    transport.__aexit__.assert_awaited_once_with(transport, None, None, None)


@pytest.mark.parametrize(
    "attribute_fixture,method_name,discriminator",
    [
        ("generated_resource", "read", "resource_name"),
        ("generated_prompt", "generate", "prompt_name"),
    ],
)
def test_generated_attributes_reject_unknown_transport_without_connection(
    request: pytest.FixtureRequest,
    mocked_mcp_connection: tuple[AsyncMock, dict[str, MagicMock]],
    attribute_fixture: str,
    method_name: str,
    discriminator: str,
) -> None:
    session, clients = mocked_mcp_connection
    attribute: BaseResource | BasePrompt = request.getfixturevalue(attribute_fixture)
    attribute.transport_type = "unsupported"
    name = getattr(attribute, "mcp_" + discriminator)
    parameters = attribute.input_schema(**{discriminator: name, "item": "entry"})
    with pytest.raises(RuntimeError, match="Unknown transport type: unsupported") as failure:
        getattr(attribute, method_name)(parameters)
    assert isinstance(failure.value.__cause__, ValueError)
    assert "Available transport types: ['sse', 'http_stream', 'stdio']" in str(failure.value.__cause__)
    session.__aenter__.assert_not_awaited()
    for client in clients.values():
        client.assert_not_called()


def test_generated_tool_rejects_unknown_transport_without_connection(
    mocked_mcp_connection: tuple[AsyncMock, dict[str, MagicMock]],
) -> None:
    session, clients = mocked_mcp_connection
    definition = MCPToolDefinition(name="EchoTool", description="Echo", input_schema={"type": "object", "properties": {}})
    tool = MCPFactory("http://example.test")._create_tool_classes([definition])[0]()
    tool.transport_type = "unsupported"
    with pytest.raises(RuntimeError, match="Unknown transport type: unsupported") as failure:
        tool.run(tool.input_schema(tool_name="EchoTool"))
    assert isinstance(failure.value.__cause__, ValueError)
    assert "Available transport types: ['sse', 'http_stream', 'stdio']" in str(failure.value.__cause__)
    session.call_tool.assert_not_awaited()
    for client in clients.values():
        client.assert_not_called()


def test_orchestrator_without_attribute_groups_warns(caplog: pytest.LogCaptureFixture) -> None:
    assert MCPFactory("http://example.test").create_orchestrator_schema() is None
    assert "No tools/resources/prompts provided to create orchestrator schema" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("borrowed", [False, True], ids=["fresh", "persistent"])
async def test_generated_resource_preserves_dotted_sdk_template_when_python_formatting_fails(
    mocked_mcp_connection: tuple[AsyncMock, dict[str, MagicMock]], borrowed: bool
) -> None:
    session, clients = mocked_mcp_connection
    session.list_resources.return_value = ListResourcesResult(resources=[])
    session.list_resource_templates.return_value = ListResourceTemplatesResult(
        resourceTemplates=[ResourceTemplate(name="DottedItem", uriTemplate="resource://items/{item.name}")]
    )
    definitions = await MCPDefinitionService.fetch_resource_definitions_from_session(session)
    session.initialize.reset_mock()
    factory = MCPFactory("http://example.test", MCPTransportType.SSE)
    resource = factory._create_resource_classes(definitions)[0]()
    if borrowed:
        resource._client_session = session
    session.read_resource.return_value = ReadResourceResult(contents=[])
    result = await resource.aread(resource.input_schema(resource_name="DottedItem", **{"item.name": "entry"}))
    assert result.content == []
    session.read_resource.assert_awaited_once_with(uri="resource://items/{item.name}")
    assert (session.initialize.await_count, clients["sse_client"].return_value.__aexit__.await_count) == (
        int(not borrowed),
        int(not borrowed),
    )


@pytest.mark.asyncio
async def test_typed_tool_rejects_model_adapter_without_output_fields(named_output_tool_session: AsyncMock) -> None:
    named_output_tool_session.call_tool.return_value = NamedToolPayload(name="adapter-without-content")
    tool_class = (await fetch_mcp_tools_async(client_session=named_output_tool_session))[0]
    with pytest.raises(RuntimeError, match="unparseable result") as failure:
        await tool_class().arun(tool_class.input_schema(tool_name="EchoTool", query="hello"))
    assert isinstance(failure.value.__cause__, ValueError)
    named_output_tool_session.call_tool.assert_awaited_once_with(name="EchoTool", arguments={"query": "hello"})
