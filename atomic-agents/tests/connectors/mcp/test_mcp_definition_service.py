import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import mcp.types as types
from mcp.shared.exceptions import McpError

from atomic_agents.connectors.mcp import (
    MCPDefinitionService,
    MCPToolDefinition,
    MCPResourceDefinition,
    MCPPromptDefinition,
    MCPTransportType,
)


class MockAsyncContextManager:
    def __init__(self, return_value=None):
        self.return_value = return_value
        self.enter_called = False
        self.exit_called = False

    async def __aenter__(self):
        self.enter_called = True
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.exit_called = True
        return False


@pytest.fixture
def mock_client_session():
    mock_session = AsyncMock()

    # Setup mock responses
    mock_tool = MagicMock()
    mock_tool.name = "TestTool"
    mock_tool.description = "Test tool description"
    mock_tool.inputSchema = {
        "type": "object",
        "properties": {"param1": {"type": "string", "description": "A string parameter"}},
        "required": ["param1"],
    }

    mock_response = MagicMock()
    mock_response.tools = [mock_tool]

    mock_session.list_tools.return_value = mock_response

    # Setup tool result
    mock_tool_result = MagicMock()
    mock_tool_result.content = "Tool result"
    mock_session.call_tool.return_value = mock_tool_result

    # Same for resources and prompts
    mock_resource = MagicMock()
    mock_resource.name = "TestResource"
    mock_resource.description = "A test resource"
    mock_resource.input_schema = {"type": "object", "properties": {"id": {"type": "string"}}}
    mock_response.resources = [mock_resource]
    mock_response.uri = "resource://TestResource/{id}"
    mock_session.list_resources.return_value = mock_response
    mock_session.list_resource_templates.return_value = types.ListResourceTemplatesResult(resourceTemplates=[])

    mock_prompt = MagicMock()
    mock_prompt.name = "welcome"
    mock_prompt.description = "Welcome prompt"
    arguments = [{"name": "id", "description": "The user's ID", "required": True}]
    mock_prompt.input_schema = {
        "type": "object",
        "properties": {arg["name"]: {"type": "string", "description": arg["description"]} for arg in arguments},
        "required": [arg["name"] for arg in arguments if arg["required"]],
    }

    # ensure list_prompts returns the same response object
    mock_response.prompts = [mock_prompt]
    mock_session.list_prompts.return_value = mock_response

    return mock_session


class TestToolDefinitionService:
    @pytest.mark.asyncio
    @patch("atomic_agents.connectors.mcp.mcp_definition_service.sse_client")
    @patch("atomic_agents.connectors.mcp.mcp_definition_service.ClientSession")
    async def test_fetch_via_sse(self, mock_client_session_cls, mock_sse_client, mock_client_session):
        # Setup
        mock_transport = MockAsyncContextManager(return_value=(AsyncMock(), AsyncMock()))
        mock_sse_client.return_value = mock_transport

        mock_session = MockAsyncContextManager(return_value=mock_client_session)
        mock_client_session_cls.return_value = mock_session

        # Create service
        service = MCPDefinitionService("http://test-endpoint", transport_type=MCPTransportType.SSE)

        # Mock the fetch_tool_definitions_from_session to return directly
        original_method = service.fetch_tool_definitions_from_session
        service.fetch_tool_definitions_from_session = AsyncMock(
            return_value=[
                MCPToolDefinition(
                    name="MockTool",
                    description="Mock tool for testing",
                    input_schema={"type": "object", "properties": {"param": {"type": "string"}}},
                )
            ]
        )

        # Execute
        result = await service.fetch_tool_definitions()

        # Verify
        assert len(result) == 1
        assert isinstance(result[0], MCPToolDefinition)
        assert result[0].name == "MockTool"
        assert result[0].description == "Mock tool for testing"

        # Restore the original method
        service.fetch_tool_definitions_from_session = original_method

        # Same for resources and prompts
        original_method_resources = service.fetch_resource_definitions_from_session
        service.fetch_resource_definitions_from_session = AsyncMock(
            return_value=[
                MCPResourceDefinition(
                    name="MockResource",
                    description="Mock resource for testing",
                    uri="resource://MockResource",
                    input_schema={"type": "object", "properties": {}, "required": []},
                )
            ]
        )
        resource_result = await service.fetch_resource_definitions()
        assert len(resource_result) == 1
        assert isinstance(resource_result[0], MCPResourceDefinition)
        assert resource_result[0].name == "MockResource"
        assert resource_result[0].description == "Mock resource for testing"
        service.fetch_resource_definitions_from_session = original_method_resources

        original_method_prompts = service.fetch_prompt_definitions_from_session
        service.fetch_prompt_definitions_from_session = AsyncMock(
            return_value=[
                MCPPromptDefinition(
                    name="welcome",
                    description="Welcome prompt",
                    input_schema={"type": "object", "properties": {}, "required": []},
                )
            ]
        )
        prompt_result = await service.fetch_prompt_definitions()
        assert len(prompt_result) == 1
        assert isinstance(prompt_result[0], MCPPromptDefinition)
        assert prompt_result[0].name == "welcome"
        assert prompt_result[0].description == "Welcome prompt"
        service.fetch_prompt_definitions_from_session = original_method_prompts

    @pytest.mark.asyncio
    @patch("atomic_agents.connectors.mcp.mcp_definition_service.streamablehttp_client")
    @patch("atomic_agents.connectors.mcp.mcp_definition_service.ClientSession")
    async def test_fetch_via_http_stream(self, mock_client_session_cls, mock_http_client, mock_client_session):
        # Setup
        mock_transport = MockAsyncContextManager(return_value=(AsyncMock(), AsyncMock(), AsyncMock()))
        mock_http_client.return_value = mock_transport

        mock_session = MockAsyncContextManager(return_value=mock_client_session)
        mock_client_session_cls.return_value = mock_session

        # Create service with HTTP_STREAM transport
        service = MCPDefinitionService("http://test-endpoint", transport_type=MCPTransportType.HTTP_STREAM)

        # Mock the fetch_tool_definitions_from_session to return directly
        original_method = service.fetch_tool_definitions_from_session
        service.fetch_tool_definitions_from_session = AsyncMock(
            return_value=[
                MCPToolDefinition(
                    name="MockTool",
                    description="Mock tool for testing",
                    input_schema={"type": "object", "properties": {"param": {"type": "string"}}},
                )
            ]
        )

        # Execute
        result = await service.fetch_tool_definitions()

        # Verify
        assert len(result) == 1
        assert isinstance(result[0], MCPToolDefinition)
        assert result[0].name == "MockTool"
        assert result[0].description == "Mock tool for testing"

        # Verify HTTP client was called with correct endpoint (should have /mcp/ suffix)
        mock_http_client.assert_called_once_with("http://test-endpoint/mcp/")

        # Restore the original method
        service.fetch_tool_definitions_from_session = original_method

        # Same for resources and prompts
        original_method_resources = service.fetch_resource_definitions_from_session
        service.fetch_resource_definitions_from_session = AsyncMock(
            return_value=[
                MCPResourceDefinition(
                    name="MockResource",
                    description="Mock resource for testing",
                    uri="resource://MockResource",
                    input_schema={"type": "object", "properties": {}, "required": []},
                )
            ]
        )
        resource_result = await service.fetch_resource_definitions()
        assert len(resource_result) == 1
        assert isinstance(resource_result[0], MCPResourceDefinition)
        assert resource_result[0].name == "MockResource"
        assert resource_result[0].description == "Mock resource for testing"
        service.fetch_resource_definitions_from_session = original_method_resources

        original_method_prompts = service.fetch_prompt_definitions_from_session
        service.fetch_prompt_definitions_from_session = AsyncMock(
            return_value=[
                MCPPromptDefinition(
                    name="welcome",
                    description="Welcome prompt",
                    input_schema={"type": "object", "properties": {}, "required": []},
                )
            ]
        )
        prompt_result = await service.fetch_prompt_definitions()
        assert len(prompt_result) == 1
        assert isinstance(prompt_result[0], MCPPromptDefinition)
        assert prompt_result[0].name == "welcome"
        assert prompt_result[0].description == "Welcome prompt"
        service.fetch_prompt_definitions_from_session = original_method_prompts

    @pytest.mark.asyncio
    async def test_fetch_via_stdio(self):
        # Create service
        service = MCPDefinitionService("command arg1 arg2", MCPTransportType.STDIO)

        # Mock the fetch_tool_definitions_from_session method
        service.fetch_tool_definitions_from_session = AsyncMock(
            return_value=[
                MCPToolDefinition(
                    name="MockTool",
                    description="Mock tool for testing",
                    input_schema={"type": "object", "properties": {"param": {"type": "string"}}},
                )
            ]
        )
        service.fetch_resource_definitions_from_session = AsyncMock(
            return_value=[
                MCPResourceDefinition(
                    name="MockResource",
                    description="Mock resource for testing",
                    uri="resource://MockResource",
                    input_schema={"type": "object", "properties": {"id": {"type": "string"}}},
                )
            ]
        )
        service.fetch_prompt_definitions_from_session = AsyncMock(
            return_value=[
                MCPPromptDefinition(
                    name="welcome",
                    description="Welcome prompt",
                    # arguments=[{"name": "id", "description": "The user's ID", "required": True}],
                    input_schema={"type": "object", "properties": {"id": {"type": "string"}}},
                )
            ]
        )

        # Patch the stdio_client to avoid actual subprocess execution
        with patch("atomic_agents.connectors.mcp.mcp_definition_service.stdio_client") as mock_stdio:
            mock_transport = MockAsyncContextManager(return_value=(AsyncMock(), AsyncMock()))
            mock_stdio.return_value = mock_transport

            with patch("atomic_agents.connectors.mcp.mcp_definition_service.ClientSession") as mock_session_cls:
                mock_session = MockAsyncContextManager(return_value=AsyncMock())
                mock_session_cls.return_value = mock_session

                # Execute
                result = await service.fetch_tool_definitions()

                # Verify
                assert len(result) == 1
                assert result[0].name == "MockTool"

                # Same for resources and prompts
                resource_result = await service.fetch_resource_definitions()
                assert len(resource_result) == 1
                assert resource_result[0].name == "MockResource"
                prompt_result = await service.fetch_prompt_definitions()
                assert len(prompt_result) == 1
                assert prompt_result[0].name == "welcome"

    @pytest.mark.asyncio
    async def test_stdio_empty_command(self):
        # Create service with empty command
        service = MCPDefinitionService("", MCPTransportType.STDIO)

        # Test that ValueError is raised for empty command
        with pytest.raises(ValueError, match="Endpoint is required"):
            await service.fetch_tool_definitions()
        with pytest.raises(ValueError, match="Endpoint is required"):
            await service.fetch_resource_definitions()
        with pytest.raises(ValueError, match="Endpoint is required"):
            await service.fetch_prompt_definitions()

    @pytest.mark.asyncio
    async def test_fetch_tool_definitions_from_session(self, mock_client_session):
        # Execute using the static method
        result = await MCPDefinitionService.fetch_tool_definitions_from_session(mock_client_session)

        # Verify
        assert len(result) == 1
        assert isinstance(result[0], MCPToolDefinition)
        assert result[0].name == "TestTool"

        # Verify session initialization
        mock_client_session.initialize.assert_called_once()
        mock_client_session.list_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_resource_definitions_from_session(self, mock_client_session):
        result = await MCPDefinitionService.fetch_resource_definitions_from_session(mock_client_session)

        assert len(result) == 1
        assert isinstance(result[0], MCPResourceDefinition)
        assert result[0].name == "TestResource"

        mock_client_session.initialize.assert_called()
        mock_client_session.list_resources.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_resource_definitions_from_session_includes_templates_with_static_resources(self):
        mock_session = AsyncMock()
        mock_session.list_resources.return_value = types.ListResourcesResult(
            resources=[types.Resource(name="current_config", uri="config://current")]
        )
        mock_session.list_resource_templates.return_value = types.ListResourceTemplatesResult(
            resourceTemplates=[types.ResourceTemplate(name="user_profile", uriTemplate="users://{user_id}")]
        )

        result = await MCPDefinitionService.fetch_resource_definitions_from_session(mock_session)

        assert [definition.name for definition in result] == ["current_config", "user_profile"]
        assert result[1].uri == "users://{user_id}"
        assert result[1].input_schema == {
            "type": "object",
            "properties": {"user_id": {"type": "string", "description": "URI parameter user_id"}},
            "required": ["user_id"],
        }
        mock_session.list_resources.assert_awaited_once()
        mock_session.list_resource_templates.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_fetch_resource_definitions_from_session_allows_unsupported_templates(self):
        mock_session = AsyncMock()
        mock_session.list_resources.return_value = types.ListResourcesResult(
            resources=[types.Resource(name="current_config", uri="config://current")]
        )
        mock_session.list_resource_templates.side_effect = McpError(
            types.ErrorData(code=types.METHOD_NOT_FOUND, message="Method not found")
        )

        result = await MCPDefinitionService.fetch_resource_definitions_from_session(mock_session)

        assert [definition.name for definition in result] == ["current_config"]
        mock_session.list_resource_templates.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("code", [types.METHOD_NOT_FOUND, types.INTERNAL_ERROR])
    async def test_resource_template_continuation_error_is_propagated(self, code):
        mock_session = AsyncMock()
        mock_session.list_resources.return_value = types.ListResourcesResult(resources=[])
        error = McpError(types.ErrorData(code=code, message="Continuation failed"))
        mock_session.list_resource_templates.side_effect = [
            types.ListResourceTemplatesResult(
                resourceTemplates=[types.ResourceTemplate(name="user", uriTemplate="users://{id}")],
                nextCursor="next-page",
            ),
            error,
        ]
        with pytest.raises(McpError) as exc_info:
            await MCPDefinitionService.fetch_resource_definitions_from_session(mock_session)
        assert exc_info.value is error
        assert mock_session.list_resource_templates.await_count == 2
        assert mock_session.list_resource_templates.await_args.kwargs == {"cursor": "next-page"}

    @pytest.mark.asyncio
    async def test_fetch_resource_definitions_from_session_propagates_template_errors(self):
        mock_session = AsyncMock()
        mock_session.list_resources.return_value = types.ListResourcesResult(
            resources=[types.Resource(name="current_config", uri="config://current")]
        )
        template_error = McpError(types.ErrorData(code=types.INTERNAL_ERROR, message="Template listing failed"))
        mock_session.list_resource_templates.side_effect = template_error

        with pytest.raises(McpError) as exc_info:
            await MCPDefinitionService.fetch_resource_definitions_from_session(mock_session)

        assert exc_info.value is template_error

    @pytest.mark.asyncio
    async def test_fetch_prompt_definitions_from_session(self, mock_client_session):
        result = await MCPDefinitionService.fetch_prompt_definitions_from_session(mock_client_session)

        assert len(result) == 1
        assert isinstance(result[0], MCPPromptDefinition)
        assert result[0].name == "welcome"

        mock_client_session.initialize.assert_called()
        mock_client_session.list_prompts.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_exception(self):
        mock_session = AsyncMock()
        mock_session.initialize.side_effect = Exception("Session error")

        with pytest.raises(Exception, match="Session error"):
            await MCPDefinitionService.fetch_tool_definitions_from_session(mock_session)
        with pytest.raises(Exception, match="Session error"):
            await MCPDefinitionService.fetch_resource_definitions_from_session(mock_session)
        with pytest.raises(Exception, match="Session error"):
            await MCPDefinitionService.fetch_prompt_definitions_from_session(mock_session)

    @pytest.mark.asyncio
    async def test_null_input_schema(self, mock_client_session):
        # Create a tool with null inputSchema
        mock_tool = MagicMock()
        mock_tool.name = "NullSchemaTool"
        mock_tool.description = "Tool with null schema"
        mock_tool.inputSchema = None

        mock_response = MagicMock()
        mock_response.tools = [mock_tool]
        mock_client_session.list_tools.return_value = mock_response

        # Execute
        result = await MCPDefinitionService.fetch_tool_definitions_from_session(mock_client_session)

        # Verify default empty schema is created
        assert len(result) == 1
        assert result[0].name == "NullSchemaTool"
        # input_schema is {"type": "object", "properties": {}, "required": []}
        assert result[0].input_schema.get("type") == "object"
        assert result[0].input_schema.get("properties") == {}

        # Same for resources and prompts
        mock_resource = MagicMock()
        mock_resource.name = "NullSchemaResource"
        mock_resource.description = "Resource with null schema"
        mock_resource.uri = "resource://NullSchemaResource"
        mock_resource.input_schema = None

        mock_response.resources = [mock_resource]
        # ensure the session will return this response for list_resources
        mock_client_session.list_resources.return_value = mock_response
        resource_result = await MCPDefinitionService.fetch_resource_definitions_from_session(mock_client_session)
        assert len(resource_result) == 1
        assert resource_result[0].name == "NullSchemaResource"
        assert resource_result[0].input_schema.get("type") == "object"
        assert resource_result[0].input_schema.get("properties") == {}
        assert resource_result[0].uri == "resource://NullSchemaResource"

        # prompts
        mock_prompt = MagicMock()
        mock_prompt.name = "NullSchemaPrompt"
        mock_prompt.description = "Prompt with null schema"
        mock_prompt.arguments = None
        mock_prompt.input_schema = None

        mock_response.prompts = [mock_prompt]
        mock_client_session.list_prompts.return_value = mock_response
        prompt_result = await MCPDefinitionService.fetch_prompt_definitions_from_session(mock_client_session)
        assert len(prompt_result) == 1
        assert prompt_result[0].name == "NullSchemaPrompt"
        assert prompt_result[0].description == "Prompt with null schema"
        assert prompt_result[0].input_schema.get("type") == "object"
        assert prompt_result[0].input_schema.get("properties") == {}

    @pytest.mark.asyncio
    async def test_stdio_command_parts_empty(self):
        svc = MCPDefinitionService("   ", MCPTransportType.STDIO)
        with pytest.raises(
            RuntimeError, match="Unexpected error during tool definition fetching: STDIO command string cannot be empty"
        ):
            await svc.fetch_tool_definitions()
        with pytest.raises(
            RuntimeError, match="Unexpected error during resource fetching: STDIO command string cannot be empty"
        ):
            await svc.fetch_resource_definitions()
        with pytest.raises(
            RuntimeError, match="Unexpected error during prompt fetching: STDIO command string cannot be empty"
        ):
            await svc.fetch_prompt_definitions()

    @pytest.mark.asyncio
    async def test_sse_connection_error(self):
        with patch("atomic_agents.connectors.mcp.mcp_definition_service.sse_client", side_effect=ConnectionError):
            svc = MCPDefinitionService("http://host", transport_type=MCPTransportType.SSE)
            with pytest.raises(ConnectionError):
                await svc.fetch_tool_definitions()
            with pytest.raises(ConnectionError):
                await svc.fetch_resource_definitions()
            with pytest.raises(ConnectionError):
                await svc.fetch_prompt_definitions()

    @pytest.mark.asyncio
    async def test_http_stream_connection_error(self):
        with patch("atomic_agents.connectors.mcp.mcp_definition_service.streamablehttp_client", side_effect=ConnectionError):
            svc = MCPDefinitionService("http://host", transport_type=MCPTransportType.HTTP_STREAM)
            with pytest.raises(ConnectionError):
                await svc.fetch_tool_definitions()
            with pytest.raises(ConnectionError):
                await svc.fetch_resource_definitions()
            with pytest.raises(ConnectionError):
                await svc.fetch_prompt_definitions()

    @pytest.mark.asyncio
    async def test_generic_error_wrapped(self):
        with patch("atomic_agents.connectors.mcp.mcp_definition_service.sse_client", side_effect=OSError("BOOM")):
            svc = MCPDefinitionService("http://host", transport_type=MCPTransportType.SSE)
            with pytest.raises(RuntimeError):
                await svc.fetch_tool_definitions()
            with pytest.raises(RuntimeError):
                await svc.fetch_resource_definitions()
            with pytest.raises(RuntimeError):
                await svc.fetch_prompt_definitions()


# Helper class for no-tools test
class _NoToolsResponse:
    """Response object that simulates an empty tools list"""

    tools = []


class _NoResourcesResponse:
    """Response object that simulates an empty resources list"""

    resources = []


class _NoPromptsResponse:
    """Response object that simulates an empty prompts list"""

    prompts = []


@pytest.mark.asyncio
async def test_fetch_tool_definitions_from_session_no_tools(caplog):
    """Test handling of empty tools list from session"""
    sess = AsyncMock()
    sess.initialize = AsyncMock()
    sess.list_tools = AsyncMock(return_value=_NoToolsResponse())

    result = await MCPDefinitionService.fetch_tool_definitions_from_session(sess)
    assert result == []
    assert "No tool definitions found on MCP server" in caplog.text


@pytest.mark.asyncio
async def test_fetch_resources_from_session_no_resources(caplog):
    """Test handling of empty resources list from session"""
    sess = AsyncMock()
    sess.initialize = AsyncMock()
    sess.list_resources = AsyncMock(return_value=_NoResourcesResponse())
    sess.list_resource_templates = AsyncMock(return_value=types.ListResourceTemplatesResult(resourceTemplates=[]))

    result = await MCPDefinitionService.fetch_resource_definitions_from_session(sess)
    assert result == []
    assert "No resources found on MCP server" in caplog.text


@pytest.mark.asyncio
async def test_fetch_prompts_from_session_no_prompts(caplog):
    """Test handling of empty prompts list from session"""
    sess = AsyncMock()
    sess.initialize = AsyncMock()
    sess.list_prompts = AsyncMock(return_value=_NoPromptsResponse())

    result = await MCPDefinitionService.fetch_prompt_definitions_from_session(sess)
    assert result == []
    assert "No prompts found on MCP server" in caplog.text


@pytest.mark.asyncio
async def test_fetch_resources_from_session(caplog):
    """Test fetching resources via session"""
    sess = AsyncMock()
    sess.initialize = AsyncMock()

    # Mock resource object as SimpleNamespace-like dict with a URI template
    mock_resource = MagicMock()
    mock_resource.name = "TestResource"
    mock_resource.description = "A test resource"
    mock_resource.uri = "resource://TestResource/{id}"

    mock_response = MagicMock()
    mock_response.resources = [mock_resource]

    sess.list_resources = AsyncMock(return_value=mock_response)
    sess.list_resource_templates = AsyncMock(return_value=types.ListResourceTemplatesResult(resourceTemplates=[]))

    result = await MCPDefinitionService.fetch_resource_definitions_from_session(sess)

    assert len(result) == 1
    rd = result[0]
    assert rd.name == "TestResource"
    assert rd.description == "A test resource"
    assert rd.input_schema["properties"]["id"]["type"] == "string"


@pytest.mark.asyncio
async def test_fetch_prompts_from_session(caplog):
    """Test fetching prompts via session"""
    sess = AsyncMock()
    sess.initialize = AsyncMock()

    # Some MCP clients may return prompt objects or dicts; provide arguments as objects
    mock_prompt = MagicMock()
    mock_prompt.name = "welcome"
    mock_prompt.description = "Welcome prompt"
    arg = MagicMock()
    arg.name = "name"
    arg.description = "The user's name"
    arg.required = True
    mock_prompt.arguments = [arg]

    mock_response = MagicMock()
    mock_response.prompts = [mock_prompt]

    sess.list_prompts = AsyncMock(return_value=mock_response)

    result = await MCPDefinitionService.fetch_prompt_definitions_from_session(sess)

    assert len(result) == 1
    pd = result[0]
    assert pd.name == "welcome"
    # validate input_schema was constructed from arguments
    assert pd.input_schema["properties"]["name"]["description"] == "The user's name"


@pytest.mark.asyncio
async def test_fetch_tool_definitions_with_output_schema():
    """Test that outputSchema is captured from MCP tools when available"""
    sess = AsyncMock()
    sess.initialize = AsyncMock()

    # Create a mock tool with outputSchema
    mock_tool = MagicMock()
    mock_tool.name = "StructuredTool"
    mock_tool.description = "A tool with structured output"
    mock_tool.inputSchema = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Search query"}},
        "required": ["query"],
    }
    mock_tool.outputSchema = {
        "type": "object",
        "properties": {
            "results": {"type": "array", "items": {"type": "string"}, "description": "Search results"},
            "count": {"type": "integer", "description": "Number of results"},
        },
        "required": ["results", "count"],
    }

    mock_response = MagicMock()
    mock_response.tools = [mock_tool]
    sess.list_tools = AsyncMock(return_value=mock_response)

    result = await MCPDefinitionService.fetch_tool_definitions_from_session(sess)

    assert len(result) == 1
    td = result[0]
    assert td.name == "StructuredTool"
    assert td.output_schema is not None
    assert td.output_schema["properties"]["results"]["type"] == "array"
    assert td.output_schema["properties"]["count"]["type"] == "integer"


@pytest.mark.asyncio
async def test_fetch_tool_definitions_without_output_schema():
    """Test that output_schema is None when MCP tool doesn't provide outputSchema"""
    sess = AsyncMock()
    sess.initialize = AsyncMock()

    # Create a mock tool without outputSchema
    mock_tool = MagicMock()
    mock_tool.name = "SimpleTool"
    mock_tool.description = "A simple tool without structured output"
    mock_tool.inputSchema = {"type": "object", "properties": {}}
    # Simulate tool without outputSchema attribute
    del mock_tool.outputSchema

    mock_response = MagicMock()
    mock_response.tools = [mock_tool]
    sess.list_tools = AsyncMock(return_value=mock_response)

    result = await MCPDefinitionService.fetch_tool_definitions_from_session(sess)

    assert len(result) == 1
    td = result[0]
    assert td.name == "SimpleTool"
    assert td.output_schema is None
