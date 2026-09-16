# MCP Server and Client Example

This guide provides a detailed overview of the Model Context Protocol (MCP) server and client example implementation in the Atomic Agents framework.

## Overview

The MCP example demonstrates how to build an intelligent agent system using the Model Context Protocol, showcasing both server and client implementations. The example supports three transport methods: STDIO, Server-Sent Events (SSE), and HTTP Stream.

## Architecture

### MCP Server

The server component (`example-mcp-server`) is built using:
- FastMCP: A high-performance MCP server implementation
- Starlette: A lightweight ASGI framework
- Uvicorn: An ASGI server implementation

Key components:
1. **Transport Layers**:
   - `server_stdio.py`: Implements STDIO-based communication
   - `server_sse.py`: Implements SSE-based HTTP communication
   - `server_http.py`: Implements unified HTTP streaming transport 

2. **Tools Service**:
   - Manages registration and execution of MCP tools
   - Handles tool discovery and metadata

3. **Resource Service**:
   - Manages static resources
   - Handles resource discovery and access

4. **Built-in Tools**:
   - `AddNumbersTool`: Performs addition
   - `SubtractNumbersTool`: Performs subtraction
   - `MultiplyNumbersTool`: Performs multiplication
   - `DivideNumbersTool`: Performs division

### MCP Client

The client component (`example-client`) is an intelligent agent that:

1. **Tool Discovery**:
   - Dynamically discovers available tools from the MCP server
   - Builds a schema-based tool registry

2. **Query Processing**:
   - Uses GPT models for natural language understanding
   - Extracts parameters from user queries
   - Selects appropriate tools based on intent

3. **Execution Flow**:
   - Maintains conversation context
   - Handles tool execution results
   - Provides conversational responses

## Implementation Details

### Server Implementation

The server supports three transport methods:

1. **SSE Transport** (`server_sse.py`):
```python
# Initialize FastMCP server
mcp = FastMCP("example-mcp-server")

# Register tools and resources
tool_service.register_tools(get_available_tools())
resource_service.register_resources(get_available_resources())

# Create Starlette app with CORS support
app = create_starlette_app(mcp_server)
```

2. **STDIO Transport** (`server_stdio.py`):
- Runs as a subprocess
- Communicates through standard input/output
- Ideal for local development

3. **HTTP Stream Transport** (`server_http.py`):
- Single `/mcp` endpoint for JSON-RPC and SSE-style streaming
- Handles session via `Mcp-Session-Id` header; allows resumable and cancelable streams

### Client Implementation

The client uses a sophisticated orchestration system:

1. **Tool Management**:
```python
# Fetch available tools (synchronous)
tools = fetch_mcp_tools(
    mcp_endpoint=config.mcp_server_url,
    transport_type=MCPTransportType.HTTP_STREAM,
)

# Or fetch tools asynchronously (must be called within async context)
tools = await fetch_mcp_tools_async(
    mcp_endpoint=config.mcp_server_command,
    transport_type=MCPTransportType.STDIO,
    client_session=session,  # Optional pre-initialized session
)

# Build tool schema mapping
tool_schema_to_class_map = {
    ToolClass.input_schema: ToolClass
    for ToolClass in tools
    if hasattr(ToolClass, "input_schema")
}
```

2. **Query Processing**:
- Uses an orchestrator agent to analyze queries
- Extracts parameters and selects appropriate tools
- Maintains conversation context through ChatHistory

3. **Async vs Sync Tool Fetching**:
```python
# Synchronous fetching (suitable for HTTP/SSE)
tools = fetch_mcp_tools(
    mcp_endpoint="http://localhost:6969", 
    transport_type=MCPTransportType.HTTP_STREAM
)

# Asynchronous fetching (optimized for STDIO with persistent sessions)
async def setup_tools():
    tools = await fetch_mcp_tools_async(
        transport_type=MCPTransportType.STDIO,
        client_session=session  # Reuse existing session
    )
    return tools
```

## Remote web search with Parallel

The local examples above use a server base URL. To connect to a remote server
with an exact endpoint and custom headers, open an MCP SDK session and pass it
to `fetch_mcp_tools_async`. The generated tools reuse that session.

This example uses [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
over Streamable HTTP at `https://search.parallel.ai/mcp`. Anonymous access needs
no Parallel account or API key and is rate limited. No local MCP server or LLM
key is needed for these direct tool calls.

Run `uv sync` in the repository root, save the following as `parallel_search.py`,
then run `uv run python parallel_search.py`:

```python
import asyncio
from importlib.metadata import version
from uuid import uuid4

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from atomic_agents.connectors.mcp import MCPTransportType, fetch_mcp_tools_async


async def main() -> None:
    # Identify Atomic Agents for aggregate free MCP usage measurement.
    # Keep this project-wide, without user or installation identifiers.
    headers = {
        "User-Agent": f"atomic-agents/{version('atomic-agents')} python-httpx/{httpx.__version__}",
    }
    async with streamablehttp_client("https://search.parallel.ai/mcp", headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            tools = await fetch_mcp_tools_async(
                transport_type=MCPTransportType.HTTP_STREAM,
                client_session=session,
            )
            tools_by_name = {tool.mcp_tool_name: tool for tool in tools}
            print("Available tools:", ", ".join(tools_by_name))

            # Search and fetch belong to one task, independently of the HTTP session.
            task_session_id = uuid4().hex
            Search = tools_by_name["web_search"]
            search_result = await Search().arun(
                Search.input_schema(
                    tool_name="web_search",
                    objective="Find the official Parallel Search MCP setup instructions",
                    search_queries=["Parallel Search MCP setup"],
                    session_id=task_session_id,
                )
            )
            print(search_result.model_dump_json(indent=2))

            Fetch = tools_by_name["web_fetch"]
            fetch_result = await Fetch().arun(
                Fetch.input_schema(
                    tool_name="web_fetch",
                    urls=["https://docs.parallel.ai/integrations/mcp/search-mcp"],
                    objective="Explain anonymous access and the available tools",
                    full_content=False,
                    session_id=task_session_id,
                )
            )
            print(fetch_result.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())
```

Running this example sends queries, requested URLs, objectives, and the supplied
task identifier to Parallel. Its project `User-Agent` identifies Atomic Agents
and its installed version for aggregate usage measurement. See Parallel's
[Customer Terms](https://parallel.ai/customer-terms) and
[Privacy Policy](https://parallel.ai/privacy-policy).

The tools retain their discovered input and output schemas, including result
URLs, excerpts, warnings, and per-URL fetch errors. `full_content=False` requests
excerpts; fetching does not use a signed-in browser or browser cookies. Keep
retrieved page text separate from trusted instructions. If you give these tools
to an agent, it can invoke them as part of answering a question. Use them only
while the session context is open and call `arun` from the same event loop.

## MCP Transport Methods

The example implements three distinct transport methods via the `MCPTransportType` enum, each with its own advantages:

```python
from atomic_agents.connectors.mcp.mcp_definition_service import MCPTransportType

# Available transport types
MCPTransportType.STDIO       # Standard input/output transport  
MCPTransportType.SSE         # Server-Sent Events transport
MCPTransportType.HTTP_STREAM # HTTP Stream transport
```

### 1. STDIO Transport

STDIO transport uses standard input/output streams for communication between the client and server:

```python
# Client-side STDIO setup (from main_stdio.py)
async def _bootstrap_stdio():
    stdio_exit_stack = AsyncExitStack()
    command_parts = shlex.split(config.mcp_stdio_server_command)
    server_params = StdioServerParameters(command=command_parts[0], args=command_parts[1:], env=None)
    read_stream, write_stream = await stdio_exit_stack.enter_async_context(stdio_client(server_params))
    session = await stdio_exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
    await session.initialize()
    return session
```

**Key advantages**:
- No network configuration required
- Simple local setup
- Direct process communication
- Lower latency for local usage

**Use cases**:
- Development and testing
- Single-user environments
- Embedded agent applications
- Offline operation

### 2. SSE Transport

Server-Sent Events (SSE) transport uses HTTP long-polling for real-time, one-way communication:

```python
# Server-side SSE setup (from server_sse.py)
async def handle_sse(request: Request) -> None:
    async with sse.connect_sse(
        request.scope,
        request.receive,
        request._send,  # noqa: SLF001
    ) as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options(),
        )
```

**Key advantages**:
- Multiple clients can connect to a single server
- Network-based communication
- Stateless server architecture
- Suitable for distributed systems

**Use cases**:
- Production deployments
- Multi-user environments
- Scalable agent infrastructure
- Cross-network operation

### 3. HTTP Stream Transport

HTTP Stream transport uses a single `/mcp` endpoint for JSON-RPC and streaming communication:

```python
# Client-side HTTP Stream setup
tools = fetch_mcp_tools(
    mcp_endpoint="http://localhost:6969",
    transport_type=MCPTransportType.HTTP_STREAM,
)
```

**Key advantages**:
- Single endpoint for all MCP operations
- Session management via headers
- Resumable and cancelable streams
- Modern HTTP-based architecture

**Use cases**:
- Modern web applications
- Cloud-native deployments  
- Microservice architectures
- API gateway integration

## Interfaces

### Tool Interface

The MCP server defines a standardized tool interface that all tools must implement:

```python
class Tool(ABC):
    """Abstract base class for all tools."""
    name: ClassVar[str]
    description: ClassVar[str]
    input_model: ClassVar[Type[BaseToolInput]]
    output_model: ClassVar[Optional[Type[BaseModel]]] = None

    @abstractmethod
    async def execute(self, input_data: BaseToolInput) -> ToolResponse:
        """Execute the tool with given arguments."""
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for the tool."""
        schema = {
            "name": self.name,
            "description": self.description,
            "input": self.input_model.model_json_schema(),
        }

        if self.output_model:
            schema["output"] = self.output_model.model_json_schema()

        return schema
```

The tool interface consists of:

1. **Class Variables**:
   - `name`: Tool identifier used in MCP communications
   - `description`: Human-readable tool description
   - `input_model`: Pydantic model defining input parameters
   - `output_model`: Pydantic model defining output structure (optional)

2. **Execute Method**:
   - Asynchronous method that performs the tool's functionality
   - Takes strongly-typed input data
   - Returns a structured ToolResponse

3. **Schema Method**:
   - Provides JSON Schema for tool discovery
   - Enables automatic documentation generation
   - Facilitates client-side validation

### Resource Interface

The MCP server defines a standardized resource interface that all resources must implement:

```python
class Resource(ABC):
    """Abstract base class for all resources."""
    name: ClassVar[str]
    description: ClassVar[str]
    uri: ClassVar[str]
    mime_type: ClassVar[str]
    input_model: ClassVar[Type[BaseResourceInput]]
    output_model: ClassVar[Optional[Type[BaseModel]]] = None

    @abstractmethod
    async def read(self, input_data: BaseResourceInput) -> ResourceResponse:
        """Read data from the resource."""
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for the resource."""
        schema = {
            "name": self.name,
            "description": self.description,
            "uri": self.uri,
            "mime_type": self.mime_type,
            "input": self.input_model.model_json_schema(),
        }

        if self.output_model:
            schema["output"] = self.output_model.model_json_schema()

        return schema
```

The resource interface consists of:

1. **Class Variables**:
   - `name`: Resource identifier used in MCP communications
   - `description`: Human-readable resource description
   - `uri`: URI pattern for accessing the resource
   - `mime_type`: MIME type of the resource content
   - `input_model`: Pydantic model defining input parameters
   - `output_model`: Pydantic model defining output structure (optional)

2. **Read Method**:
   - Asynchronous method that retrieves data from the resource
   - Takes strongly-typed input data
   - Returns a structured ResourceResponse

3. **Schema Method**:
   - Provides URI Template for resource discovery
   - Enables automatic documentation generation


### Prompt Interface

The MCP client uses a standardized prompt interface for managing prompts:

```python
class Prompt(ABC):
    """Abstract base class for all prompts."""
    name: ClassVar[str]
    description: ClassVar[str]
    input_model: ClassVar[Type[BasePromptInput]]
    output_model: ClassVar[Optional[Type[BaseModel]]] = None

    @abstractmethod
    async def generate(self, input_data: BasePromptInput) -> PromptResponse:
        """Generate the prompt with given arguments."""
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Get JSON schema for the prompt."""
        schema = {
            "name": self.name,
            "description": self.description,
            "input": self.input_model.model_json_schema(),
        }

        if self.output_model:
            schema["output"] = self.output_model.model_json_schema()

        return schema
```

The prompt interface consists of:

1. **Class Variables**:
   - `name`: Prompt identifier used in MCP communications
   - `description`: Human-readable prompt description
   - `input_model`: Pydantic model defining input parameters
   - `output_model`: Pydantic model defining output structure (optional)

2. **Generate Method**:
   - Asynchronous method that generates the prompt
   - Takes strongly-typed input data
   - Returns a structured PromptResponse

3. **Schema Method**:
   - Provides JSON Schema for prompt discovery
   - Enables automatic documentation generation


## Configuration

### Server Configuration

The server can be configured through command-line arguments:
```bash
uv run example-mcp-server --mode=sse --host=0.0.0.0 --port=6969 --reload
uv run example-mcp-server --mode=stdio
uv run example-mcp-server --mode=http_stream --host=0.0.0.0 --port=6969
```

Options:
- `--mode`: Transport mode (sse/stdio/http_stream)
- `--host`: Host to bind to
- `--port`: Port to listen on
- `--reload`: Enable auto-reload for development

### Client Configuration

The client uses a configuration class:
```python
@dataclass
class MCPConfig:
    mcp_server_url: str = "http://localhost:6969"
    openai_model: str = "gpt-5-mini"
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
```

For STDIO transport, additional options are available:
```python
@dataclass
class MCPConfig:
    openai_model: str = "gpt-5-mini"
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    mcp_stdio_server_command: str = "uv run example-mcp-server --mode stdio"
```

## Usage Examples

1. **Start the Server (SSE mode)**:
```bash
cd example-mcp-server
uv run example-mcp-server --mode=sse
```

2. **Run the Client**:

Using the main launcher with transport selection:
```bash
cd example-client
uv run python -m example_client.main --transport sse
```

Directly calling the SSE client:
```bash
cd example-client
uv run python -m example_client.main_sse
```

Directly calling the STDIO client:
```bash
cd example-client
uv run python -m example_client.main_stdio
```

3. **Example Queries**:
```
You: What is 2+2?
You: Calculate the square root of 144
You: Generate a random number between 1 and 100
```

## Best Practices

1. **Development**:
   - Use STDIO transport for local development
   - Enable server auto-reload during development
   - Implement proper error handling

2. **Production**:
   - Use SSE transport for production deployments
   - Configure appropriate CORS settings
   - Implement authentication if needed

3. **Tool Development**:
   - Follow the Tool interface contract
   - Provide clear input/output schemas
   - Include comprehensive documentation

## Extending the Example

### Adding New Tools

To add new tools:

1. Create a new tool class implementing the Tool interface
2. Register the tool in the server's tool service
3. The client will automatically discover and use the new tool

Example tool structure:
```python
class MyNewTool(Tool):
    name = "my_new_tool"
    description = "This tool performs a custom operation"
    input_model = create_model(
        "MyNewToolInput",
        param1=(str, Field(..., description="First parameter")),
        param2=(int, Field(..., description="Second parameter")),
        __base__=BaseToolInput
    )

    async def execute(self, input_data: BaseToolInput) -> ToolResponse:
        # Access params with input_data.param1, input_data.param2
        result = f"Processed {input_data.param1} with {input_data.param2}"
        return ToolResponse.from_text(result)
```

Then register the tool in the server:
```python
def get_available_tools() -> List[Tool]:
    return [
        # ... existing tools ...
        MyNewTool(),
    ]
```

### Adding New Resources

To add new resources:

1. Create a new resource class implementing the Resource interface
2. Register the resource in the server's resource service
3. The client can access the new resource via its URI

Example resource structure:
```python
class MyNewResource(Resource):
    name = "my_new_resource"
    description = "This resource provides custom data"
    uri = "resource://my_new_resource/{param1}"
    mime_type = "application/json"
    input_model = create_model(
        "MyNewResourceInput",
        param1=(str, Field(..., description="Resource parameter")),
        __base__=BaseResourceInput
    )

    async def read(self, input_data: BaseResourceInput) -> ResourceResponse:
        # Access params with input_data.param1
        data = {"message": f"Data for {input_data.param1}"}
        return ResourceResponse.from_data(data, self.mime_type)
```

Then register the resource in the server:
```python
def get_available_resources() -> List[Resource]:
    return [
        # ... existing resources ...
        MyNewResource(),
    ]
```

### Adding New Prompts

To add new prompts:

1. Create a new prompt class implementing the Prompt interface
2. Register the prompt in the server's prompt service
3. The client will automatically discover and use the new prompt

Example prompt structure:
```python
class MyNewPrompt(Prompt):
    name = "my_new_prompt"
    description = "This prompt generates a custom response"
    input_model = create_model(
        "MyNewPromptInput",
        param1=(str, Field(..., description="First parameter")),
        param2=(int, Field(..., description="Second parameter")),
        __base__=BasePromptInput
    )

    async def generate(self, input_data: BasePromptInput) -> PromptResponse:
        # Access params with input_data.param1, input_data.param2
        result = f"Generated response for {input_data.param1} with {input_data.param2}"
        return PromptResponse.from_text(result)
```

Then register the prompt in the server:
```python
def get_available_prompts() -> List[Prompt]:
    return [
        # ... existing prompts ...
        MyNewPrompt(),
    ]
```
