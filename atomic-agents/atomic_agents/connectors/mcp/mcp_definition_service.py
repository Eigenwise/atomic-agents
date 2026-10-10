"""Module for fetching tool definitions from MCP endpoints."""

import logging
import re
import shlex
from contextlib import AsyncExitStack
from typing import List, NamedTuple, Optional, Dict, Any, Callable, Awaitable
from enum import Enum

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.exceptions import McpError
import mcp.types as types
from pydantic import AnyUrl
from urllib.parse import unquote as decode_uri

logger = logging.getLogger(__name__)


async def _fetch_all_pages(list_page: Callable[..., Awaitable[Any]], items_attribute: str) -> List[Any]:
    """Collect an MCP list response, following its opaque continuation cursors."""
    response = await list_page()
    items = list(getattr(response, items_attribute) or [])
    cursor = getattr(response, "nextCursor", None)
    while isinstance(cursor, str):
        response = await list_page(cursor=cursor)
        items.extend(getattr(response, items_attribute) or [])
        cursor = getattr(response, "nextCursor", None)
    return items


class MCPTransportType(Enum):
    """Enum for MCP transport types."""

    SSE = "sse"
    HTTP_STREAM = "http_stream"
    STDIO = "stdio"


class MCPAttributeType:
    """MCP attribute types."""

    TOOL = "tool"
    RESOURCE = "resource"
    PROMPT = "prompt"


class MCPToolDefinition(NamedTuple):
    """Definition of an MCP tool."""

    name: str
    description: Optional[str]
    input_schema: Dict[str, Any]
    output_schema: Optional[Dict[str, Any]] = None


class MCPResourceDefinition(NamedTuple):
    """Definition of an MCP resource."""

    name: str
    description: Optional[str]
    uri: str
    input_schema: Dict[str, Any]
    mime_type: Optional[str] = None


class MCPPromptDefinition(NamedTuple):
    """Definition of an MCP prompt/template."""

    name: str
    description: Optional[str]
    input_schema: Dict[str, Any]
    # required: List[str]  # A list of required argument names


class MCPDefinitionService:
    """Service for fetching tool definitions from MCP endpoints."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        transport_type: MCPTransportType = MCPTransportType.HTTP_STREAM,
        working_directory: Optional[str] = None,
    ):
        """
        Initialize the service.

        Args:
            endpoint: URL of the MCP server (for SSE/HTTP stream) or command string (for STDIO)
            transport_type: Type of transport to use (SSE, HTTP_STREAM, or STDIO)
            working_directory: Optional working directory to use when running STDIO commands
        """
        self.endpoint = endpoint
        self.transport_type = transport_type
        self.working_directory = working_directory

    async def fetch_tool_definitions(self) -> List[MCPToolDefinition]:
        """
        Fetch tool definitions from the configured endpoint.

        Returns:
            List of tool definitions

        Raises:
            ConnectionError: If connection to the MCP server fails
            ValueError: If the STDIO command string is empty
            RuntimeError: For other unexpected errors
        """
        if not self.endpoint:
            raise ValueError("Endpoint is required")

        definitions = []
        stack = AsyncExitStack()
        try:
            if self.transport_type == MCPTransportType.STDIO:
                # STDIO transport
                command_parts = shlex.split(self.endpoint)
                if not command_parts:
                    raise ValueError("STDIO command string cannot be empty.")
                command = command_parts[0]
                args = command_parts[1:]
                logger.info(f"Attempting STDIO connection with command='{command}', args={args}")
                server_params = StdioServerParameters(command=command, args=args, env=None, cwd=self.working_directory)
                stdio_transport = await stack.enter_async_context(stdio_client(server_params))
                read_stream, write_stream = stdio_transport
            elif self.transport_type == MCPTransportType.HTTP_STREAM:
                # HTTP Stream transport - use trailing slash to avoid redirect
                # See: https://github.com/modelcontextprotocol/python-sdk/issues/732
                transport_endpoint = f"{self.endpoint}/mcp/"
                logger.info(f"Attempting HTTP Stream connection to {transport_endpoint}")
                transport = await stack.enter_async_context(streamablehttp_client(transport_endpoint))
                read_stream, write_stream, _ = transport
            elif self.transport_type == MCPTransportType.SSE:
                # SSE transport (deprecated)
                transport_endpoint = f"{self.endpoint}/sse"
                logger.info(f"Attempting SSE connection to {transport_endpoint}")
                transport = await stack.enter_async_context(sse_client(transport_endpoint))
                read_stream, write_stream = transport
            else:
                available_types = [t.value for t in MCPTransportType]
                raise ValueError(f"Unknown transport type: {self.transport_type}. Available types: {available_types}")

            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            definitions = await self.fetch_tool_definitions_from_session(session)

        except ConnectionError as e:
            logger.error(f"Error fetching MCP tool definitions from {self.endpoint}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching MCP tool definitions from {self.endpoint}: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error during tool definition fetching: {e}") from e
        finally:
            await stack.aclose()

        return definitions

    @staticmethod
    async def fetch_tool_definitions_from_session(session: ClientSession) -> List[MCPToolDefinition]:
        """
        Fetch tool definitions from an existing session.

        Args:
            session: MCP client session

        Returns:
            List of tool definitions

        Raises:
            Exception: If listing tools fails
        """
        definitions: List[MCPToolDefinition] = []
        try:
            # `initialize` is idempotent – calling it twice is safe and
            # ensures the session is ready.
            await session.initialize()
            mcp_tools = await _fetch_all_pages(session.list_tools, "tools")
            for mcp_tool in mcp_tools:
                # Capture outputSchema if the MCP server provides one
                output_schema = getattr(mcp_tool, "outputSchema", None)
                definitions.append(
                    MCPToolDefinition(
                        name=mcp_tool.name,
                        description=mcp_tool.description,
                        input_schema=mcp_tool.inputSchema or {"type": "object", "properties": {}},
                        output_schema=output_schema,
                    )
                )

            if not definitions:
                logger.warning("No tool definitions found on MCP server")

        except Exception as e:
            logger.error("Failed to list tools via MCP session: %s", e, exc_info=True)
            raise

        return definitions

    async def fetch_resource_definitions(self) -> List[MCPResourceDefinition]:
        """
        Fetch resource definitions from the configured endpoint.

        Returns:
            List of resource definitions
        """
        if not self.endpoint:
            raise ValueError("Endpoint is required")

        resources: List[MCPResourceDefinition] = []
        stack = AsyncExitStack()
        try:
            if self.transport_type == MCPTransportType.STDIO:
                command_parts = shlex.split(self.endpoint)
                if not command_parts:
                    raise ValueError("STDIO command string cannot be empty.")
                command = command_parts[0]
                args = command_parts[1:]
                server_params = StdioServerParameters(command=command, args=args, env=None, cwd=self.working_directory)
                stdio_transport = await stack.enter_async_context(stdio_client(server_params))
                read_stream, write_stream = stdio_transport
            elif self.transport_type == MCPTransportType.HTTP_STREAM:
                transport_endpoint = f"{self.endpoint}/mcp/"
                transport = await stack.enter_async_context(streamablehttp_client(transport_endpoint))
                read_stream, write_stream, _ = transport
            elif self.transport_type == MCPTransportType.SSE:
                transport_endpoint = f"{self.endpoint}/sse"
                transport = await stack.enter_async_context(sse_client(transport_endpoint))
                read_stream, write_stream = transport
            else:
                available_types = [t.value for t in MCPTransportType]
                raise ValueError(f"Unknown transport type: {self.transport_type}. Available types: {available_types}")

            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            resources = await self.fetch_resource_definitions_from_session(session)

        except ConnectionError as e:
            logger.error(f"Error fetching MCP resources from {self.endpoint}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching MCP resources from {self.endpoint}: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error during resource fetching: {e}") from e
        finally:
            await stack.aclose()

        return resources

    @staticmethod
    async def fetch_resource_definitions_from_session(session: ClientSession) -> List[MCPResourceDefinition]:
        """
        Fetch resource definitions from an existing session.

        Args:
            session: MCP client session

        Returns:
            List of resource definitions
        """
        resources: List[MCPResourceDefinition] = []

        try:
            await session.initialize()
            resources_iterable: List[types.Resource] = await _fetch_all_pages(session.list_resources, "resources")

            try:
                res_templates = await _fetch_all_pages(session.list_resource_templates, "resourceTemplates")
            except McpError as error:
                if error.error.code != types.METHOD_NOT_FOUND:
                    raise
            else:
                for template in res_templates:
                    # Resource templates have no "input_schema" value and use URI templates with parameters.
                    resources_iterable.append(
                        types.Resource(
                            name=template.name,
                            description=template.description,
                            uri=AnyUrl(template.uriTemplate),
                        )
                    )

            for mcp_resource in resources_iterable:
                # Support both attribute-style objects and dict-like responses
                if hasattr(mcp_resource, "name"):
                    name = mcp_resource.name
                    description = mcp_resource.description
                    uri = mcp_resource.uri
                elif isinstance(mcp_resource, dict):
                    # assume mapping
                    name = mcp_resource["name"]
                    description = mcp_resource.get("description")
                    uri = mcp_resource.get("uri", "")
                else:
                    raise ValueError(f"Unexpected resource format: {mcp_resource}")

                # Extract placeholders from the chosen source
                uri = decode_uri(str(uri))
                placeholders = re.findall(r"\{([^}]+)\}", uri) if uri else []
                properties: Dict[str, Any] = {}
                for param_name in placeholders:
                    properties[param_name] = {"type": "string", "description": f"URI parameter {param_name}"}

                resources.append(
                    MCPResourceDefinition(
                        name=name,
                        description=description,
                        uri=uri,
                        mime_type=getattr(mcp_resource, "mimeType", None),
                        input_schema={"type": "object", "properties": properties, "required": list(placeholders)},
                    )
                )

            if not resources:
                logger.warning("No resources found on MCP server")

        except Exception as e:
            logger.error("Failed to list resources via MCP session: %s", e, exc_info=True)
            raise

        return resources

    async def fetch_prompt_definitions(self) -> List[MCPPromptDefinition]:
        """
        Fetch prompt/template definitions from the configured endpoint.

        Returns:
            List of prompt definitions
        """
        if not self.endpoint:
            raise ValueError("Endpoint is required")

        prompts: List[MCPPromptDefinition] = []
        stack = AsyncExitStack()
        try:
            if self.transport_type == MCPTransportType.STDIO:
                command_parts = shlex.split(self.endpoint)
                if not command_parts:
                    raise ValueError("STDIO command string cannot be empty.")
                command = command_parts[0]
                args = command_parts[1:]
                server_params = StdioServerParameters(command=command, args=args, env=None, cwd=self.working_directory)
                stdio_transport = await stack.enter_async_context(stdio_client(server_params))
                read_stream, write_stream = stdio_transport
            elif self.transport_type == MCPTransportType.HTTP_STREAM:
                transport_endpoint = f"{self.endpoint}/mcp/"
                transport = await stack.enter_async_context(streamablehttp_client(transport_endpoint))
                read_stream, write_stream, _ = transport
            elif self.transport_type == MCPTransportType.SSE:
                transport_endpoint = f"{self.endpoint}/sse"
                transport = await stack.enter_async_context(sse_client(transport_endpoint))
                read_stream, write_stream = transport
            else:
                available_types = [t.value for t in MCPTransportType]
                raise ValueError(f"Unknown transport type: {self.transport_type}. Available types: {available_types}")

            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            prompts = await self.fetch_prompt_definitions_from_session(session)

        except ConnectionError as e:
            logger.error(f"Error fetching MCP prompts from {self.endpoint}: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unexpected error fetching MCP prompts from {self.endpoint}: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error during prompt fetching: {e}") from e
        finally:
            await stack.aclose()

        return prompts

    @staticmethod
    async def fetch_prompt_definitions_from_session(session: ClientSession) -> List[MCPPromptDefinition]:
        """
        Fetch prompt/template definitions from an existing session.

        Args:
            session: MCP client session

        Returns:
            List of prompt definitions
        """
        prompts: List[MCPPromptDefinition] = []
        try:
            await session.initialize()
            mcp_prompts = await _fetch_all_pages(session.list_prompts, "prompts")
            for mcp_prompt in mcp_prompts:
                arguments: List[types.PromptArgument] = mcp_prompt.arguments or []
                prompts.append(
                    MCPPromptDefinition(
                        name=mcp_prompt.name,
                        description=mcp_prompt.description,
                        input_schema={
                            "type": "object",
                            "properties": {arg.name: {"type": "string", "description": arg.description} for arg in arguments},
                            "required": [arg.name for arg in arguments if arg.required],
                        },
                    )
                )
            if not prompts:
                logger.warning("No prompts found on MCP server")

        except Exception as e:
            logger.error("Failed to list prompts via MCP session: %s", e, exc_info=True)
            raise

        return prompts
