import asyncio
import json
import logging
from typing import Any, Dict, List, Type, Optional, Union, Tuple, cast
from contextlib import AsyncExitStack
import shlex
import types

from pydantic import create_model, Field, BaseModel, TypeAdapter, ValidationError

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
import mcp.types

from atomic_agents.base.base_io_schema import BaseIOSchema
from atomic_agents.base import BaseTool, BaseResource, BasePrompt
from atomic_agents.connectors.mcp.schema_transformer import SchemaTransformer
from atomic_agents.connectors.mcp.mcp_definition_service import (
    MCPAttributeType,
    MCPDefinitionService,
    MCPToolDefinition,
    MCPTransportType,
    MCPResourceDefinition,
    MCPPromptDefinition,
)

logger = logging.getLogger(__name__)


class MCPToolOutputSchema(BaseIOSchema):
    """Generic output schema for dynamically generated MCP tools.

    Used as a fallback when the MCP server does not provide an outputSchema definition.
    Tools with MCP-provided outputSchema will have typed output schemas instead.
    """

    result: Any = Field(..., description="The result returned by the MCP tool.")


class MCPResourceOutputSchema(BaseIOSchema):
    """Generic output schema for dynamically generated MCP resources."""

    content: Any = Field(..., description="The content of the MCP resource.")
    mime_type: Optional[str] = Field(None, description="The MIME type of the resource.")


class MCPPromptOutputSchema(BaseIOSchema):
    """Generic output schema for dynamically generated MCP prompts."""

    content: str = Field(..., description="The content of the MCP prompt.")


class MCPFactory:
    """Factory for creating MCP tool classes."""

    @staticmethod
    def _nullable_arguments(schema: Dict[str, Any]) -> set[str]:
        """Check declared parameter types before optional fields are widened to accept None."""
        nullable = set()
        for name, prop_schema in schema.get("properties", {}).items():
            if "enum" in prop_schema and None not in prop_schema["enum"]:
                continue
            if "const" in prop_schema and prop_schema["const"] is not None:
                continue
            field_type, _ = SchemaTransformer.json_to_pydantic_field(prop_schema, True, schema)
            try:
                TypeAdapter(field_type).validate_python(None)
            except ValidationError:
                continue
            nullable.add(name)
        return nullable

    def __init__(
        self,
        mcp_endpoint: Optional[str] = None,
        transport_type: MCPTransportType = MCPTransportType.HTTP_STREAM,
        client_session: Optional[ClientSession] = None,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
        working_directory: Optional[str] = None,
    ):
        """
        Initialize the factory.

        Args:
            mcp_endpoint: URL of the MCP server (for SSE/HTTP stream) or the full command to run the server (for STDIO)
            transport_type: Type of transport to use (SSE, HTTP_STREAM, or STDIO)
            client_session: Optional pre-initialized ClientSession for reuse
            event_loop: Optional event loop for running asynchronous operations
            working_directory: Optional working directory to use when running STDIO commands
        """
        self.mcp_endpoint = mcp_endpoint
        self.transport_type = transport_type
        self.client_session = client_session
        self.event_loop = event_loop
        self.schema_transformer = SchemaTransformer()
        self.working_directory = working_directory

        # Validate configuration
        if client_session is not None and event_loop is None:
            raise ValueError("When `client_session` is provided an `event_loop` must also be supplied.")
        if not mcp_endpoint and client_session is None:
            raise ValueError("`mcp_endpoint` must be provided when no `client_session` is supplied.")

    def create_tools(self) -> List[Type[BaseTool]]:
        """
        Create tool classes from the configured endpoint or session.

        Returns:
            List of dynamically generated BaseTool subclasses
        """
        tool_definitions = self._fetch_tool_definitions()
        if not tool_definitions:
            return []

        return self._create_tool_classes(tool_definitions)

    def _fetch_tool_definitions(self) -> List[MCPToolDefinition]:
        """
        Fetch tool definitions using the appropriate method.

        Returns:
            List of tool definitions
        """
        if self.client_session is not None:
            # Use existing session
            async def _gather_defs():
                return await MCPDefinitionService.fetch_tool_definitions_from_session(self.client_session)  # pragma: no cover

            return cast(asyncio.AbstractEventLoop, self.event_loop).run_until_complete(_gather_defs())  # pragma: no cover
        else:
            # Create new connection
            service = MCPDefinitionService(
                self.mcp_endpoint,
                self.transport_type,
                self.working_directory,
            )
            return asyncio.run(service.fetch_tool_definitions())

    def _create_tool_classes(self, tool_definitions: List[MCPToolDefinition]) -> List[Type[BaseTool]]:
        """
        Create tool classes from definitions.

        Args:
            tool_definitions: List of tool definitions

        Returns:
            List of dynamically generated BaseTool subclasses
        """
        generated_tools = []

        for definition in tool_definitions:
            try:
                tool_name = definition.name
                tool_description = definition.description or f"Dynamically generated tool for MCP tool: {tool_name}"
                input_schema_dict = definition.input_schema

                # Create input schema
                InputSchema = self.schema_transformer.create_model_from_schema(
                    input_schema_dict,
                    f"{tool_name}InputSchema",
                    tool_name,
                    f"Input schema for {tool_name}",
                    attribute_type=MCPAttributeType.TOOL,
                )

                # Create output schema - use MCP-provided schema if available, otherwise fallback to generic.
                # When a typed output schema is used, _has_typed_output_schema is set on the tool class
                # to enable structured content extraction at runtime (see result processing below).
                output_schema_dict: Optional[Dict[str, Any]] = definition.output_schema
                has_typed_output_schema = False
                if output_schema_dict:
                    # Use the schema transformer to create a proper typed output schema
                    OutputSchema = self.schema_transformer.create_model_from_schema(
                        output_schema_dict,
                        f"{tool_name}OutputSchema",
                        tool_name,
                        f"Output schema for {tool_name}",
                        attribute_type=MCPAttributeType.TOOL,
                        is_output_schema=True,
                    )
                    has_typed_output_schema = True
                else:
                    # Fallback to generic output schema
                    OutputSchema = type(
                        f"{tool_name}OutputSchema", (MCPToolOutputSchema,), {"__doc__": f"Output schema for {tool_name}"}
                    )

                # Async implementation
                async def run_tool_async(self, params: InputSchema) -> OutputSchema:  # type: ignore
                    bound_tool_name = self.mcp_tool_name
                    bound_mcp_endpoint = self.mcp_endpoint  # May be None when using external session
                    bound_transport_type = self.transport_type
                    persistent_session: Optional[ClientSession] = getattr(self, "_client_session", None)
                    bound_working_directory = getattr(self, "working_directory", None)

                    # Get arguments, excluding tool_name
                    arguments = {
                        name: value
                        for name, value in params.model_dump(exclude={"tool_name"}, exclude_unset=True).items()
                        if value is not None or name in self._nullable_argument_names
                    }

                    async def _connect_and_call():
                        stack = AsyncExitStack()
                        try:
                            if bound_transport_type == MCPTransportType.STDIO:
                                # Split the command string into the command and its arguments
                                command_parts = shlex.split(bound_mcp_endpoint)
                                if not command_parts:
                                    raise ValueError("STDIO command string cannot be empty.")
                                command = command_parts[0]
                                args = command_parts[1:]
                                logger.debug(f"Executing tool '{bound_tool_name}' via STDIO: command='{command}', args={args}")
                                server_params = StdioServerParameters(
                                    command=command, args=args, env=None, cwd=bound_working_directory
                                )
                                stdio_transport = await stack.enter_async_context(stdio_client(server_params))
                                read_stream, write_stream = stdio_transport
                            elif bound_transport_type == MCPTransportType.HTTP_STREAM:
                                # HTTP Stream transport - use trailing slash to avoid redirect
                                # See: https://github.com/modelcontextprotocol/python-sdk/issues/732
                                http_endpoint = f"{bound_mcp_endpoint}/mcp/"
                                logger.debug(f"Executing tool '{bound_tool_name}' via HTTP Stream: endpoint={http_endpoint}")
                                http_transport = await stack.enter_async_context(streamablehttp_client(http_endpoint))
                                read_stream, write_stream, _ = http_transport
                            elif bound_transport_type == MCPTransportType.SSE:
                                # SSE transport (deprecated)
                                sse_endpoint = f"{bound_mcp_endpoint}/sse"
                                logger.debug(f"Executing tool '{bound_tool_name}' via SSE: endpoint={sse_endpoint}")
                                sse_transport = await stack.enter_async_context(sse_client(sse_endpoint))
                                read_stream, write_stream = sse_transport
                            else:
                                available_types = [t.value for t in MCPTransportType]
                                raise ValueError(
                                    f"Unknown transport type: {bound_transport_type}. Available transport types: {available_types}"
                                )

                            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
                            await session.initialize()

                            # Ensure arguments is a dict, even if empty
                            call_args = arguments if isinstance(arguments, dict) else {}
                            tool_result = await session.call_tool(name=bound_tool_name, arguments=call_args)
                            return tool_result
                        finally:
                            await stack.aclose()

                    async def _call_with_persistent_session():
                        # Ensure arguments is a dict, even if empty
                        call_args = arguments if isinstance(arguments, dict) else {}
                        return await persistent_session.call_tool(name=bound_tool_name, arguments=call_args)

                    try:
                        if persistent_session is not None:
                            # Use the always‑on session/loop supplied at construction time.
                            tool_result = await _call_with_persistent_session()
                        else:
                            # Legacy behaviour – open a fresh connection per invocation.
                            tool_result = await _connect_and_call()

                        # Process the result based on whether we have a typed output schema.
                        # Extraction precedence for typed schemas:
                        # 1. structuredContent attribute (MCP spec primary path)
                        # 2. content[0].text parsed as JSON (some servers return JSON as text)
                        # 3. content[0].data dict (structured data in content item)
                        # 4. Dict with structuredContent/content keys
                        # 5. Direct dict usage as fallback
                        has_typed_schema = getattr(self, "_has_typed_output_schema", False)
                        BoundOutputSchema = self.output_schema

                        if has_typed_schema:
                            # For typed output schemas, try to extract structured content
                            # MCP tools with output schemas return structured data
                            if isinstance(tool_result, BaseModel) and hasattr(tool_result, "structuredContent"):
                                # Use structured content if available (MCP spec)
                                structured_data = tool_result.structuredContent
                                if isinstance(structured_data, dict):
                                    return BoundOutputSchema(**structured_data)
                                elif hasattr(structured_data, "model_dump"):
                                    return BoundOutputSchema(**structured_data.model_dump())
                                else:
                                    # Unexpected type for structuredContent
                                    logger.error(
                                        f"Unexpected structuredContent type for tool '{bound_tool_name}': "
                                        f"got {type(structured_data).__name__}, expected dict or BaseModel. "
                                        f"Content: {structured_data!r}"
                                    )
                                    raise TypeError(
                                        f"MCP tool '{bound_tool_name}' returned structuredContent with unexpected type "
                                        f"{type(structured_data).__name__}. Expected dict or BaseModel."
                                    )
                            elif isinstance(tool_result, BaseModel) and hasattr(tool_result, "content"):
                                # Try to parse content as structured data
                                content = tool_result.content
                                # Ensure content is a list/tuple before indexing
                                if content and isinstance(content, (list, tuple)) and len(content) > 0:
                                    first_content = content[0]
                                    # Check for text content that might be JSON
                                    if hasattr(first_content, "text"):
                                        try:
                                            parsed = json.loads(first_content.text)
                                            if isinstance(parsed, dict):
                                                return BoundOutputSchema(**parsed)
                                            else:
                                                logger.debug(
                                                    f"Tool '{bound_tool_name}' content parsed as JSON but was "
                                                    f"{type(parsed).__name__}, not dict. Trying other extraction methods."
                                                )
                                        except json.JSONDecodeError as e:
                                            logger.debug(
                                                f"Tool '{bound_tool_name}' content is not valid JSON: {e}. "
                                                f"Content preview: {first_content.text[:200]!r}..."
                                                if len(first_content.text) > 200
                                                else f"Content: {first_content.text!r}"
                                            )
                                        except TypeError as e:
                                            logger.warning(
                                                f"Tool '{bound_tool_name}' content.text has unexpected type "
                                                f"{type(first_content.text).__name__}: {e}"
                                            )
                                    # Check for structured content in the content item
                                    if hasattr(first_content, "data") and isinstance(first_content.data, dict):
                                        return BoundOutputSchema(**first_content.data)
                            elif isinstance(tool_result, dict):
                                if "structuredContent" in tool_result:
                                    return BoundOutputSchema(**tool_result["structuredContent"])
                                elif "content" in tool_result:
                                    content = tool_result["content"]
                                    if isinstance(content, dict):
                                        return BoundOutputSchema(**content)
                            # Fallback: try to use tool_result directly if it's a dict
                            if isinstance(tool_result, dict):
                                return BoundOutputSchema(**tool_result)
                            # If we have a typed schema but couldn't extract structured content,
                            # this is an error - we cannot fall through to generic handling
                            # because the typed schema doesn't have a 'result' field.
                            logger.error(
                                f"Could not parse structured output for tool '{bound_tool_name}'. "
                                f"Expected typed output but got: type={type(tool_result).__name__}, "
                                f"value={tool_result!r}"
                            )
                            raise ValueError(
                                f"MCP tool '{bound_tool_name}' has outputSchema but returned unparseable result. "
                                f"Received type: {type(tool_result).__name__}. "
                                f"Check MCP server implementation."
                            )

                        # Generic output schema handling (original behavior) - only for tools without typed schemas
                        if isinstance(tool_result, BaseModel) and hasattr(tool_result, "content"):
                            actual_result_content = tool_result.content
                        elif isinstance(tool_result, dict) and "content" in tool_result:
                            actual_result_content = tool_result["content"]
                        else:
                            actual_result_content = tool_result

                        return BoundOutputSchema(result=actual_result_content)

                    except Exception as e:
                        logger.error(f"Error executing MCP tool '{bound_tool_name}': {e}", exc_info=True)
                        raise RuntimeError(f"Failed to execute MCP tool '{bound_tool_name}': {e}") from e

                # Create sync wrapper
                def run_tool_sync(self, params: InputSchema) -> OutputSchema:  # type: ignore
                    persistent_session: Optional[ClientSession] = getattr(self, "_client_session", None)
                    loop: Optional[asyncio.AbstractEventLoop] = getattr(self, "_event_loop", None)

                    if persistent_session is not None:
                        # Use the always‑on session/loop supplied at construction time.
                        try:
                            return cast(asyncio.AbstractEventLoop, loop).run_until_complete(self.arun(params))
                        except AttributeError as e:
                            raise RuntimeError(f"Failed to execute MCP tool '{tool_name}': {e}") from e
                    else:
                        # Legacy behaviour – run in new event loop.
                        return asyncio.run(self.arun(params))

                # Create the tool class using types.new_class() instead of type()
                attrs = {
                    "arun": run_tool_async,
                    "run": run_tool_sync,
                    "__doc__": tool_description,
                    "mcp_tool_name": tool_name,
                    "mcp_endpoint": self.mcp_endpoint,
                    "transport_type": self.transport_type,
                    "_client_session": self.client_session,
                    "_event_loop": self.event_loop,
                    "working_directory": self.working_directory,
                    "_has_typed_output_schema": has_typed_output_schema,
                    "_nullable_argument_names": self._nullable_arguments(input_schema_dict),
                }

                # Create the class using new_class() for proper generic type support
                tool_class = types.new_class(
                    tool_name, (BaseTool[InputSchema, OutputSchema],), {}, lambda ns: ns.update(attrs)
                )

                # Add the input_schema and output_schema class attributes explicitly
                # since they might not be properly inherited with types.new_class
                setattr(tool_class, "input_schema", InputSchema)
                setattr(tool_class, "output_schema", OutputSchema)

                generated_tools.append(tool_class)

            except Exception as e:
                logger.error(f"Error generating class for tool '{definition.name}': {e}", exc_info=True)
                continue

        return generated_tools

    def create_orchestrator_schema(
        self,
        tools: Optional[List[Type[BaseTool]]] = None,
        resources: Optional[List[Type[BaseResource]]] = None,
        prompts: Optional[List[Type[BasePrompt]]] = None,
    ) -> Optional[Type[BaseIOSchema]]:
        """
        Create an orchestrator schema for the given tools.

        Args:
            tools: List of tool classes
            resources: List of resource classes
            prompts: List of prompt classes

        Returns:
            Orchestrator schema or None if no tools provided
        """
        if tools is None and resources is None and prompts is None:
            logger.warning("No tools/resources/prompts provided to create orchestrator schema")
            return None
        if tools is None:
            tools = []
        if resources is None:
            resources = []
        if prompts is None:
            prompts = []

        tool_schemas = [ToolClass.input_schema for ToolClass in tools]
        resource_schemas = [ResourceClass.input_schema for ResourceClass in resources]
        prompt_schemas = [PromptClass.input_schema for PromptClass in prompts]

        # Build runtime Union types for each attribute group when present
        field_defs = {}

        if tool_schemas:
            ToolUnion = Union[tuple(tool_schemas)]
            field_defs["tool_parameters"] = (
                ToolUnion,
                Field(
                    ...,
                    description="The parameters for the selected tool, matching its specific schema (which includes the 'tool_name').",
                ),
            )

        if resource_schemas:
            ResourceUnion = Union[tuple(resource_schemas)]
            field_defs["resource_parameters"] = (
                ResourceUnion,
                Field(
                    ...,
                    description="The parameters for the selected resource, matching its specific schema (which includes the 'resource_name').",
                ),
            )

        if prompt_schemas:
            PromptUnion = Union[tuple(prompt_schemas)]
            field_defs["prompt_parameters"] = (
                PromptUnion,
                Field(
                    ...,
                    description="The parameters for the selected prompt, matching its specific schema (which includes the 'prompt_name').",
                ),
            )

        if not field_defs:
            logger.warning("No schemas available to create orchestrator union")
            return None

        # Dynamically create the output schema with the appropriate fields
        orchestrator_schema = create_model(
            "MCPOrchestratorOutputSchema",
            __doc__="Output schema for the MCP Orchestrator Agent. Contains the parameters for the selected tool/resource/prompt.",
            __base__=BaseIOSchema,
            **field_defs,
        )

        return orchestrator_schema

    def create_resources(self) -> List[Type[BaseResource]]:
        """
        Create resource classes from the configured endpoint or session.

        Returns:
            List of dynamically generated resource classes
        """
        resource_definitions = self._fetch_resource_definitions()
        if not resource_definitions:
            return []

        return self._create_resource_classes(resource_definitions)

    def _fetch_resource_definitions(self) -> List[MCPResourceDefinition]:
        """
        Fetch resource definitions using the appropriate method.

        Returns:
            List of resource definitions
        """
        if self.client_session is not None:
            # Use existing session
            async def _gather_defs():
                return await MCPDefinitionService.fetch_resource_definitions_from_session(
                    self.client_session
                )  # pragma: no cover

            return cast(asyncio.AbstractEventLoop, self.event_loop).run_until_complete(_gather_defs())  # pragma: no cover
        else:
            # Create new connection
            service = MCPDefinitionService(
                self.mcp_endpoint,
                self.transport_type,
                self.working_directory,
            )
            return asyncio.run(service.fetch_resource_definitions())

    def _create_resource_classes(self, resource_definitions: List[MCPResourceDefinition]) -> List[Type[BaseResource]]:
        """
        Create resource classes from definitions.

        Args:
            resource_definitions: List of resource definitions

        Returns:
            List of dynamically generated resource classes
        """
        generated_resources = []

        for definition in resource_definitions:
            try:
                resource_name = definition.name
                resource_description = (
                    definition.description or f"Dynamically generated resource for MCP resource: {resource_name}"
                )
                uri = definition.uri
                mime_type = definition.mime_type

                InputSchema = self.schema_transformer.create_model_from_schema(
                    definition.input_schema,
                    f"{resource_name}InputSchema",
                    resource_name,
                    f"Input schema for {resource_name}",
                    attribute_type=MCPAttributeType.RESOURCE,
                )

                # Create output schema
                OutputSchema = type(
                    f"{resource_name}OutputSchema",
                    (MCPResourceOutputSchema,),
                    {"__doc__": f"Output schema for {resource_name}"},
                )

                # Async implementation
                async def read_resource_async(self, params: InputSchema) -> OutputSchema:  # type: ignore
                    bound_uri = self.uri
                    bound_mcp_endpoint = self.mcp_endpoint  # May be None when using external session
                    bound_transport_type = self.transport_type
                    persistent_session: Optional[ClientSession] = getattr(self, "_client_session", None)
                    bound_working_directory = getattr(self, "working_directory", None)

                    arguments = params.model_dump(exclude={"resource_name"}, exclude_none=True)

                    async def _connect_and_read():
                        stack = AsyncExitStack()
                        try:
                            if bound_transport_type == MCPTransportType.STDIO:
                                # Split the command string into the command and its arguments
                                command_parts = shlex.split(bound_mcp_endpoint)
                                if not command_parts:
                                    raise ValueError("STDIO command string cannot be empty.")
                                command = command_parts[0]
                                args = command_parts[1:]
                                logger.debug(
                                    f"Reading resource '{self.mcp_resource_name}' via STDIO: command='{command}', args={args}"
                                )
                                server_params = StdioServerParameters(
                                    command=command, args=args, env=None, cwd=bound_working_directory
                                )
                                stdio_transport = await stack.enter_async_context(stdio_client(server_params))
                                read_stream, write_stream = stdio_transport
                            elif bound_transport_type == MCPTransportType.HTTP_STREAM:
                                # HTTP Stream transport - use trailing slash to avoid redirect
                                # See: https://github.com/modelcontextprotocol/python-sdk/issues/732
                                http_endpoint = f"{bound_mcp_endpoint}/mcp/"
                                logger.debug(
                                    f"Reading resource '{self.mcp_resource_name}' via HTTP Stream: endpoint={http_endpoint}"
                                )
                                http_transport = await stack.enter_async_context(streamablehttp_client(http_endpoint))
                                read_stream, write_stream, _ = http_transport
                            elif bound_transport_type == MCPTransportType.SSE:
                                # SSE transport (deprecated)
                                sse_endpoint = f"{bound_mcp_endpoint}/sse"
                                logger.debug(f"Reading resource '{self.mcp_resource_name}' via SSE: endpoint={sse_endpoint}")
                                sse_transport = await stack.enter_async_context(sse_client(sse_endpoint))
                                read_stream, write_stream = sse_transport
                            else:
                                available_types = [t.value for t in MCPTransportType]
                                raise ValueError(
                                    f"Unknown transport type: {bound_transport_type}. Available transport types: {available_types}"
                                )

                            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
                            await session.initialize()

                            # Substitute URI placeholders with provided parameters when available.
                            call_args = arguments if isinstance(arguments, dict) else {}
                            # If params contain keys, format the URI template.
                            try:
                                concrete_uri = bound_uri.format(**call_args) if call_args else bound_uri
                            except Exception:
                                concrete_uri = bound_uri
                            resource_result: mcp.types.ReadResourceResult = await session.read_resource(uri=concrete_uri)
                            return resource_result
                        finally:
                            await stack.aclose()

                    async def _read_with_persistent_session():
                        call_args = arguments if isinstance(arguments, dict) else {}

                        try:
                            concrete_uri_p = bound_uri.format(**call_args) if call_args else bound_uri
                        except Exception:
                            concrete_uri_p = bound_uri

                        return await persistent_session.read_resource(uri=concrete_uri_p)

                    try:
                        if persistent_session is not None:
                            # Use the always‑on session/loop supplied at construction time.
                            resource_result = await _read_with_persistent_session()
                        else:
                            # Legacy behaviour – open a fresh connection per invocation.
                            resource_result = await _connect_and_read()

                        # Process the result
                        if isinstance(resource_result, BaseModel) and hasattr(resource_result, "contents"):
                            actual_content = resource_result.contents
                            # MCP stores mimeType in each content item, not on the result itself
                            if actual_content and len(actual_content) > 0:
                                # Get mimeType from the first content item
                                first_content = actual_content[0]
                                actual_mime = getattr(first_content, "mimeType", mime_type)
                            else:
                                actual_mime = mime_type
                        elif isinstance(resource_result, dict) and "contents" in resource_result:
                            actual_content = resource_result["contents"]
                            actual_mime = resource_result.get("mime_type", mime_type)
                        else:
                            actual_content = resource_result
                            actual_mime = mime_type

                        return OutputSchema(content=actual_content, mime_type=actual_mime)

                    except Exception as e:
                        logger.error(f"Error reading MCP resource '{self.mcp_resource_name}': {e}", exc_info=True)
                        raise RuntimeError(f"Failed to read MCP resource '{self.mcp_resource_name}': {e}") from e

                # Create sync wrapper
                def read_resource_sync(self, params: InputSchema) -> OutputSchema:  # type: ignore
                    persistent_session: Optional[ClientSession] = getattr(self, "_client_session", None)
                    loop: Optional[asyncio.AbstractEventLoop] = getattr(self, "_event_loop", None)

                    if persistent_session is not None:
                        # Use the always‑on session/loop supplied at construction time.
                        try:
                            return cast(asyncio.AbstractEventLoop, loop).run_until_complete(self.aread(params))
                        except AttributeError as e:
                            raise RuntimeError(f"Failed to read MCP resource '{resource_name}': {e}") from e
                    else:
                        # Legacy behaviour – run in new event loop.
                        return asyncio.run(self.aread(params))

                # Create the resource class using types.new_class() instead of type()
                attrs = {
                    "aread": read_resource_async,
                    "read": read_resource_sync,
                    "__doc__": resource_description,
                    "mcp_resource_name": resource_name,
                    "mcp_endpoint": self.mcp_endpoint,
                    "transport_type": self.transport_type,
                    "_client_session": self.client_session,
                    "_event_loop": self.event_loop,
                    "working_directory": self.working_directory,
                    "uri": uri,
                }

                # Create the class using new_class() for proper generic type support
                resource_class = types.new_class(
                    resource_name, (BaseResource[InputSchema, OutputSchema],), {}, lambda ns: ns.update(attrs)
                )

                # Add the input_schema and output_schema class attributes explicitly
                setattr(resource_class, "input_schema", InputSchema)
                setattr(resource_class, "output_schema", OutputSchema)

                generated_resources.append(resource_class)

            except Exception as e:
                logger.error(f"Error generating class for resource '{definition.name}': {e}", exc_info=True)
                continue

        return generated_resources

    def create_prompts(self) -> List[Type[BasePrompt]]:
        """
        Create prompt classes from the configured endpoint or session.

        Returns:
            List of dynamically generated prompt classes
        """
        prompt_definitions = self._fetch_prompt_definitions()
        if not prompt_definitions:
            return []

        return self._create_prompt_classes(prompt_definitions)

    def _fetch_prompt_definitions(self) -> List[MCPPromptDefinition]:
        """
        Fetch prompt definitions using the appropriate method.

        Returns:
            List of prompt definitions
        """
        if self.client_session is not None:
            # Use existing session
            async def _gather_defs():
                return await MCPDefinitionService.fetch_prompt_definitions_from_session(
                    self.client_session
                )  # pragma: no cover

            return cast(asyncio.AbstractEventLoop, self.event_loop).run_until_complete(_gather_defs())  # pragma: no cover
        else:
            # Create new connection
            service = MCPDefinitionService(
                self.mcp_endpoint,
                self.transport_type,
                self.working_directory,
            )
            return asyncio.run(service.fetch_prompt_definitions())

    def _create_prompt_classes(self, prompt_definitions: List[MCPPromptDefinition]) -> List[Type[BasePrompt]]:
        """
        Create prompt classes from definitions.

        Args:
            prompt_definitions: List of prompt definitions

        Returns:
            List of dynamically generated prompt classes
        """
        generated_prompts = []

        for definition in prompt_definitions:
            try:
                prompt_name = definition.name
                prompt_description = definition.description or f"Dynamically generated prompt for MCP prompt: {prompt_name}"

                InputSchema = self.schema_transformer.create_model_from_schema(
                    definition.input_schema,
                    f"{prompt_name}InputSchema",
                    prompt_name,
                    f"Input schema for {prompt_name}",
                    attribute_type=MCPAttributeType.PROMPT,
                )

                # Create output schema
                OutputSchema = type(
                    f"{prompt_name}OutputSchema", (MCPPromptOutputSchema,), {"__doc__": f"Output schema for {prompt_name}"}
                )

                # Async implementation
                async def generate_prompt_async(self, params: InputSchema) -> OutputSchema:  # type: ignore
                    bound_prompt_name = self.mcp_prompt_name
                    bound_mcp_endpoint = self.mcp_endpoint  # May be None when using external session
                    bound_transport_type = self.transport_type
                    persistent_session: Optional[ClientSession] = getattr(self, "_client_session", None)
                    bound_working_directory = getattr(self, "working_directory", None)

                    # Get arguments
                    arguments = params.model_dump(exclude={"prompt_name"}, exclude_none=True)

                    async def _connect_and_generate():
                        stack = AsyncExitStack()
                        try:
                            if bound_transport_type == MCPTransportType.STDIO:
                                # Split the command string into the command and its arguments
                                command_parts = shlex.split(bound_mcp_endpoint)
                                if not command_parts:
                                    raise ValueError("STDIO command string cannot be empty.")
                                command = command_parts[0]
                                args = command_parts[1:]
                                logger.debug(
                                    f"Getting prompt '{bound_prompt_name}' via STDIO: command='{command}', args={args}"
                                )
                                server_params = StdioServerParameters(
                                    command=command, args=args, env=None, cwd=bound_working_directory
                                )
                                stdio_transport = await stack.enter_async_context(stdio_client(server_params))
                                read_stream, write_stream = stdio_transport
                            elif bound_transport_type == MCPTransportType.HTTP_STREAM:
                                # HTTP Stream transport - use trailing slash to avoid redirect
                                # See: https://github.com/modelcontextprotocol/python-sdk/issues/732
                                http_endpoint = f"{bound_mcp_endpoint}/mcp/"
                                logger.debug(f"Getting prompt '{bound_prompt_name}' via HTTP Stream: endpoint={http_endpoint}")
                                http_transport = await stack.enter_async_context(streamablehttp_client(http_endpoint))
                                read_stream, write_stream, _ = http_transport
                            elif bound_transport_type == MCPTransportType.SSE:
                                # SSE transport (deprecated)
                                sse_endpoint = f"{bound_mcp_endpoint}/sse"
                                logger.debug(f"Getting prompt '{bound_prompt_name}' via SSE: endpoint={sse_endpoint}")
                                sse_transport = await stack.enter_async_context(sse_client(sse_endpoint))
                                read_stream, write_stream = sse_transport
                            else:
                                available_types = [t.value for t in MCPTransportType]
                                raise ValueError(
                                    f"Unknown transport type: {bound_transport_type}. Available transport types: {available_types}"
                                )

                            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
                            await session.initialize()

                            # Ensure arguments is a dict, even if empty
                            call_args = arguments if isinstance(arguments, dict) else {}
                            prompt_result = await session.get_prompt(name=bound_prompt_name, arguments=call_args)
                            return prompt_result
                        finally:
                            await stack.aclose()

                    async def _get_with_persistent_session():
                        # Ensure arguments is a dict, even if empty
                        call_args = arguments if isinstance(arguments, dict) else {}
                        return await persistent_session.get_prompt(name=bound_prompt_name, arguments=call_args)

                    try:
                        if persistent_session is not None:
                            # Use the always‑on session/loop supplied at construction time.
                            prompt_result = await _get_with_persistent_session()
                        else:
                            # Legacy behaviour – open a fresh connection per invocation.
                            prompt_result = await _connect_and_generate()

                        # Process the result
                        messages = None
                        if isinstance(prompt_result, BaseModel) and hasattr(prompt_result, "messages"):
                            messages = prompt_result.messages
                        elif isinstance(prompt_result, dict) and "messages" in prompt_result:
                            messages = prompt_result["messages"]
                        else:
                            raise Exception("Prompt response has no messages.")

                        texts = []
                        for message in messages:
                            if isinstance(message, BaseModel) and hasattr(message, "content"):
                                content = message.content  # type: ignore
                            elif isinstance(message, dict) and "content" in message:
                                content = message["content"]
                            else:
                                content = message

                            if isinstance(content, str):
                                texts.append(content)
                            elif isinstance(content, dict):
                                texts.append(content.get("text", ""))
                            elif getattr(content, "text", None):
                                texts.append(content.text)  # type: ignore
                            else:
                                texts.append(str(content))
                        final_content = "\n\n".join(texts)

                        return OutputSchema(content=final_content)

                    except Exception as e:
                        logger.error(f"Error getting MCP prompt '{bound_prompt_name}': {e}", exc_info=True)
                        raise RuntimeError(f"Failed to get MCP prompt '{bound_prompt_name}': {e}") from e

                # Create sync wrapper
                def generate_prompt_sync(self, params: InputSchema) -> OutputSchema:  # type: ignore
                    persistent_session: Optional[ClientSession] = getattr(self, "_client_session", None)
                    loop: Optional[asyncio.AbstractEventLoop] = getattr(self, "_event_loop", None)

                    if persistent_session is not None:
                        # Use the always‑on session/loop supplied at construction time.
                        try:
                            return cast(asyncio.AbstractEventLoop, loop).run_until_complete(self.agenerate(params))
                        except AttributeError as e:
                            raise RuntimeError(f"Failed to get MCP prompt '{prompt_name}': {e}") from e
                    else:
                        # Legacy behaviour – run in new event loop.
                        return asyncio.run(self.agenerate(params))

                # Create the prompt class using types.new_class() instead of type()
                attrs = {
                    "agenerate": generate_prompt_async,
                    "generate": generate_prompt_sync,
                    "__doc__": prompt_description,
                    "mcp_prompt_name": prompt_name,
                    "mcp_endpoint": self.mcp_endpoint,
                    "transport_type": self.transport_type,
                    "_client_session": self.client_session,
                    "_event_loop": self.event_loop,
                    "working_directory": self.working_directory,
                }

                # Create the class using new_class() for proper generic type support
                prompt_class = types.new_class(
                    prompt_name, (BasePrompt[InputSchema, OutputSchema],), {}, lambda ns: ns.update(attrs)
                )

                # Add the input_schema and output_schema class attributes explicitly
                setattr(prompt_class, "input_schema", InputSchema)
                setattr(prompt_class, "output_schema", OutputSchema)

                generated_prompts.append(prompt_class)

            except Exception as e:
                logger.error(f"Error generating class for prompt '{definition.name}': {e}", exc_info=True)
                continue

        return generated_prompts


# Public API functions
def fetch_mcp_tools(
    mcp_endpoint: Optional[str] = None,
    transport_type: MCPTransportType = MCPTransportType.HTTP_STREAM,
    *,
    client_session: Optional[ClientSession] = None,
    event_loop: Optional[asyncio.AbstractEventLoop] = None,
    working_directory: Optional[str] = None,
) -> List[Type[BaseTool]]:
    """
    Connects to an MCP server via SSE, HTTP Stream or STDIO, discovers tool definitions, and dynamically generates
    synchronous Atomic Agents compatible BaseTool subclasses for each tool.
    Each generated tool will establish its own connection when its `run` method is called.

    Args:
        mcp_endpoint: URL of the MCP server or command for STDIO.
        transport_type: Type of transport to use (SSE, HTTP_STREAM, or STDIO).
        client_session: Optional pre-initialized ClientSession for reuse.
        event_loop: Optional event loop for running asynchronous operations.
        working_directory: Optional working directory for STDIO.
    """
    factory = MCPFactory(mcp_endpoint, transport_type, client_session, event_loop, working_directory)
    return factory.create_tools()


async def fetch_mcp_tools_async(
    mcp_endpoint: Optional[str] = None,
    transport_type: MCPTransportType = MCPTransportType.STDIO,
    *,
    client_session: Optional[ClientSession] = None,
    working_directory: Optional[str] = None,
) -> List[Type[BaseTool]]:
    """
    Asynchronously connects to an MCP server and dynamically generates BaseTool subclasses for each tool.
    Must be called within an existing asyncio event loop context.

    Args:
        mcp_endpoint: URL of the MCP server (for HTTP/SSE) or command for STDIO.
        transport_type: Type of transport to use (SSE, HTTP_STREAM, or STDIO).
        client_session: Optional pre-initialized ClientSession for reuse.
        working_directory: Optional working directory for STDIO transport.
    """
    if client_session is not None:
        tool_defs = await MCPDefinitionService.fetch_tool_definitions_from_session(client_session)
        factory = MCPFactory(mcp_endpoint, transport_type, client_session, asyncio.get_running_loop(), working_directory)
    else:
        service = MCPDefinitionService(mcp_endpoint, transport_type, working_directory)
        tool_defs = await service.fetch_tool_definitions()
        factory = MCPFactory(mcp_endpoint, transport_type, None, None, working_directory)

    return factory._create_tool_classes(tool_defs)


def create_mcp_orchestrator_schema(
    tools: Optional[List[Type[BaseTool]]] = None,
    resources: Optional[List[Type[BaseResource]]] = None,
    prompts: Optional[List[Type[BasePrompt]]] = None,
) -> Optional[Type[BaseIOSchema]]:
    """
    Creates a schema for the MCP Orchestrator's output using the Union of all tool input schemas.

    Args:
        tools: List of dynamically generated MCP tool classes

    Returns:
        A Pydantic model class to be used as the output schema for an orchestrator agent
    """
    # Bypass constructor validation since orchestrator schema does not require endpoint or session
    factory = object.__new__(MCPFactory)
    return MCPFactory.create_orchestrator_schema(factory, tools, resources, prompts)


def fetch_mcp_attributes_with_schema(
    mcp_endpoint: Optional[str] = None,
    transport_type: MCPTransportType = MCPTransportType.HTTP_STREAM,
    *,
    client_session: Optional[ClientSession] = None,
    event_loop: Optional[asyncio.AbstractEventLoop] = None,
    working_directory: Optional[str] = None,
) -> Tuple[List[Type[BaseTool]], List[Type[BaseResource]], List[Type[BasePrompt]], Optional[Type[BaseIOSchema]]]:
    """
    Fetches MCP tools and creates an orchestrator schema for them. Returns both as a tuple.

    Args:
        mcp_endpoint: URL of the MCP server or command for STDIO.
        transport_type: Type of transport to use (SSE, HTTP_STREAM, or STDIO).
        client_session: Optional pre-initialized ClientSession for reuse.
        event_loop: Optional event loop for running asynchronous operations.
        working_directory: Optional working directory for STDIO.

    Returns:
        A tuple containing:
        - List of dynamically generated tool classes
        - List of dynamically generated resource classes
        - List of dynamically generated prompt classes
        - Orchestrator output schema with Union of tool input schemas, or None if no tools found.
    """
    factory = MCPFactory(mcp_endpoint, transport_type, client_session, event_loop, working_directory)
    tools = factory.create_tools()
    resources = factory.create_resources()
    prompts = factory.create_prompts()
    if not tools and not resources and not prompts:
        return [], [], [], None

    orchestrator_schema = factory.create_orchestrator_schema(tools, resources, prompts)
    return tools, resources, prompts, orchestrator_schema


# Resource / Prompt convenience API
def fetch_mcp_resources(
    mcp_endpoint: Optional[str] = None,
    transport_type: MCPTransportType = MCPTransportType.HTTP_STREAM,
    *,
    client_session: Optional[ClientSession] = None,
    event_loop: Optional[asyncio.AbstractEventLoop] = None,
    working_directory: Optional[str] = None,
) -> List[Type[BaseResource]]:
    """
    Fetch resource classes from an MCP server (sync).
    """
    factory = MCPFactory(mcp_endpoint, transport_type, client_session, event_loop, working_directory)
    return factory.create_resources()


async def fetch_mcp_resources_async(
    mcp_endpoint: Optional[str] = None,
    transport_type: MCPTransportType = MCPTransportType.HTTP_STREAM,
    *,
    client_session: Optional[ClientSession] = None,
    working_directory: Optional[str] = None,
) -> List[Type[BaseResource]]:
    """
    Async version of fetch_mcp_resources. Call from within an event loop.
    """
    if client_session is not None:
        resource_defs = await MCPDefinitionService.fetch_resource_definitions_from_session(client_session)
        factory = MCPFactory(mcp_endpoint, transport_type, client_session, asyncio.get_running_loop(), working_directory)
    else:
        service = MCPDefinitionService(mcp_endpoint, transport_type, working_directory)
        resource_defs = await service.fetch_resource_definitions()
        factory = MCPFactory(mcp_endpoint, transport_type, None, None, working_directory)

    return factory._create_resource_classes(resource_defs)


def fetch_mcp_prompts(
    mcp_endpoint: Optional[str] = None,
    transport_type: MCPTransportType = MCPTransportType.HTTP_STREAM,
    *,
    client_session: Optional[ClientSession] = None,
    event_loop: Optional[asyncio.AbstractEventLoop] = None,
    working_directory: Optional[str] = None,
) -> List[Type[BasePrompt]]:
    """
    Fetch prompt classes from an MCP server (sync).
    """
    factory = MCPFactory(mcp_endpoint, transport_type, client_session, event_loop, working_directory)
    return factory.create_prompts()


async def fetch_mcp_prompts_async(
    mcp_endpoint: Optional[str] = None,
    transport_type: MCPTransportType = MCPTransportType.HTTP_STREAM,
    *,
    client_session: Optional[ClientSession] = None,
    working_directory: Optional[str] = None,
) -> List[Type[BasePrompt]]:
    """
    Async version of fetch_mcp_prompts. Call from within an event loop.
    """
    if client_session is not None:
        prompt_defs = await MCPDefinitionService.fetch_prompt_definitions_from_session(client_session)
        factory = MCPFactory(mcp_endpoint, transport_type, client_session, asyncio.get_running_loop(), working_directory)
    else:
        service = MCPDefinitionService(mcp_endpoint, transport_type, working_directory)
        prompt_defs = await service.fetch_prompt_definitions()
        factory = MCPFactory(mcp_endpoint, transport_type, None, None, working_directory)

    return factory._create_prompt_classes(prompt_defs)
