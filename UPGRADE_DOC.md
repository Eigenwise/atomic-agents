# Atomic Agents v2.0 Upgrade Guide

## Overview

Atomic Agents v2.0 introduces breaking changes to improve the developer experience through cleaner imports and better type safety. This guide helps you migrate your code from v1.x to v2.0.

## Prerequisites

- **Python Version**: Ensure your environment is running Python 3.12 or higher (v2.0 requires >=3.12)
- **Update Dependencies**:
  ```bash
  uv sync
  # or
  pip install --upgrade atomic-agents>=2.0.0
  ```

## Breaking Changes and Migration

### 1. Import Path Restructuring

The `.lib` directory has been eliminated from all imports, resulting in cleaner and more intuitive paths.

#### Import Migration Map

**Core Classes**
```python
# OLD (v1.x)
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.base.base_tool import BaseTool, BaseToolConfig

# NEW (v2.0)
from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema, BaseTool, BaseToolConfig
```

**Context Components**
```python
# OLD (v1.x)
from atomic_agents.lib.components.agent_memory import AgentMemory
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator, SystemPromptContextProviderBase

# NEW (v2.0)
from atomic_agents.context import ChatHistory, SystemPromptGenerator, BaseDynamicContextProvider
```

**MCP Integration**
```python
# OLD (v1.x)
from atomic_agents.lib.factories.mcp_tool_factory import fetch_mcp_tools_async
from atomic_agents.lib.factories.tool_definition_service import MCPTransportType

# NEW (v2.0)
from atomic_agents.connectors.mcp import fetch_mcp_tools_async, MCPTransportType
```

**Utilities**
```python
# OLD (v1.x)
from atomic_agents.lib.utils.format_tool_message import format_tool_message

# NEW (v2.0)
from atomic_agents.utils import format_tool_message
```

#### How to Migrate

Use find-and-replace across your codebase:
```
"atomic_agents.lib.base." → "atomic_agents."
"atomic_agents.lib.components." → "atomic_agents.context."
"atomic_agents.lib.factories." → "atomic_agents.connectors.mcp."
"atomic_agents.lib.utils." → "atomic_agents.utils."
```

### 2. Class Renames

Several classes have been renamed for clarity and consistency:

- `BaseAgent` → `AtomicAgent`
- `BaseAgentConfig` → `AgentConfig`
- `BaseAgentInputSchema` → `BasicChatInputSchema`
- `BaseAgentOutputSchema` → `BasicChatOutputSchema`
- `AgentMemory` → `ChatHistory`
- `SystemPromptContextProviderBase` → `BaseDynamicContextProvider`

#### How to Migrate

Find and replace these class names throughout your code:
```
"BaseAgent" → "AtomicAgent"
"BaseAgentConfig" → "AgentConfig"
"BaseAgentInputSchema" → "BasicChatInputSchema"
"BaseAgentOutputSchema" → "BasicChatOutputSchema"
"AgentMemory" → "ChatHistory"
"SystemPromptContextProviderBase" → "BaseDynamicContextProvider"
```

Also update method calls:
```python
# OLD
agent.reset_memory()

# NEW
agent.reset_history()
```

### 3. Generic Type Parameters for Tools

**IMPORTANT**: `BaseTool` now uses generic type parameters similar to `AtomicAgent`. This is a breaking change that affects all custom tools.

#### Tool Definition Changes

```python
# OLD (v1.x) - Schemas as class attributes
from atomic_agents.lib.base.base_tool import BaseTool, BaseToolConfig
from atomic_agents.lib.base.base_io_schema import BaseIOSchema

class MyTool(BaseTool):
    input_schema = MyInputSchema
    output_schema = MyOutputSchema

    def run(self, params: MyInputSchema) -> MyOutputSchema:
        # Tool logic
        pass

# NEW (v2.0) - Schemas as type parameters
from atomic_agents import BaseTool, BaseToolConfig, BaseIOSchema

class MyTool(BaseTool[MyInputSchema, MyOutputSchema]):
    def run(self, params: MyInputSchema) -> MyOutputSchema:
        # Tool logic
        pass
```

#### How to Migrate

1. **Update tool class definitions**:
   ```python
   # Example with calculator tool
   # OLD
   class CalculatorTool(BaseTool):
       input_schema = CalculatorInputSchema
       output_schema = CalculatorOutputSchema

   # NEW
   class CalculatorTool(BaseTool[CalculatorInputSchema, CalculatorOutputSchema]):
       # No need for input_schema and output_schema attributes
   ```

2. **The schemas are now accessed via properties** that use the generic type parameters

### 4. Agent Creation and Configuration Changes

v2.0 moves schemas from configuration to type parameters and updates several configuration fields:

#### Schemas Move from Config to Type Parameters

```python
# OLD (v1.x) - Schemas passed in config
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig
from atomic_agents.lib.components.agent_memory import AgentMemory
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from pydantic import Field
import instructor
from openai import OpenAI

# Define custom schemas
class CustomInputSchema(BaseIOSchema):
    """Custom input schema for specialized agent"""
    query: str = Field(..., description="User's query")
    context: str = Field(default="", description="Additional context")

class CustomOutputSchema(BaseIOSchema):
    """Custom output schema for specialized agent"""
    answer: str = Field(..., description="Agent's response")
    confidence: float = Field(..., description="Confidence score")

# Setup client
client = instructor.from_openai(OpenAI())

# Using default schemas (implicitly)
agent = BaseAgent(
    BaseAgentConfig(
        client=client,
        model="gpt-5-mini",
        memory=AgentMemory()
        # No schema parameters = uses BaseAgentInputSchema and BaseAgentOutputSchema
    )
)

# Using custom schemas (passed in config)
agent = BaseAgent(
    BaseAgentConfig(
        client=client,
        model="gpt-5-mini",
        memory=AgentMemory(),
        input_schema=CustomInputSchema,   # Passed in config
        output_schema=CustomOutputSchema   # Passed in config
    )
)
```

```python
# NEW (v2.0) - Schemas as type parameters
from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema, BasicChatInputSchema, BasicChatOutputSchema
from atomic_agents.context import ChatHistory
from pydantic import Field
import instructor
from openai import OpenAI

# Define custom schemas (same as before)
class CustomInputSchema(BaseIOSchema):
    """Custom input schema for specialized agent"""
    query: str = Field(..., description="User's query")
    context: str = Field(default="", description="Additional context")

class CustomOutputSchema(BaseIOSchema):
    """Custom output schema for specialized agent"""
    answer: str = Field(..., description="Agent's response")
    confidence: float = Field(..., description="Confidence score")

# Setup client
client = instructor.from_openai(OpenAI())

# Using default schemas (explicitly as type parameters)
agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
    AgentConfig(
        client=client,
        model="gpt-5-mini",
        history=ChatHistory()
    )
)

# Using custom schemas (as type parameters)
agent = AtomicAgent[CustomInputSchema, CustomOutputSchema](
    AgentConfig(
        client=client,
        model="gpt-5-mini",
        history=ChatHistory()
        # No schema parameters in config!
    )
)
```

#### How to Migrate

1. **Move schemas from config to type parameters**:
   ```python
   # Example with custom schemas:
   from atomic_agents import BaseIOSchema
   from pydantic import Field

   class TranslationInput(BaseIOSchema):
       text: str = Field(..., description="Text to translate")
       target_language: str = Field(..., description="Target language code")

   class TranslationOutput(BaseIOSchema):
       translated_text: str = Field(..., description="The translated text")
       confidence: float = Field(..., description="Translation confidence score")

   # OLD - Schemas passed in config
   agent = BaseAgent(
       BaseAgentConfig(
           client=client,
           model="gpt-5-mini",
           memory=AgentMemory(),
           input_schema=TranslationInput,  # Was here
           output_schema=TranslationOutput  # Was here
       )
   )

   # NEW - Schemas as type parameters
   agent = AtomicAgent[TranslationInput, TranslationOutput](
       AgentConfig(
           client=client,
           model="gpt-5-mini",
           history=ChatHistory()  # Note: memory → history
           # Schemas no longer in config!
       )
   )
   ```

2. **Update configuration fields**:
   ```python
   # OLD - Direct parameters
   config = BaseAgentConfig(
       client=client,
       model="gpt-5-mini",
       memory=AgentMemory(),  # Old field name
       temperature=0.7,       # Direct parameter
       max_tokens=1000        # Direct parameter
   )

   # NEW - Grouped parameters
   config = AgentConfig(
       client=client,
       model="gpt-5-mini",
       history=ChatHistory(),  # New field name
       model_api_parameters={  # Temperature and max_tokens moved here
           "temperature": 0.7,
           "max_tokens": 1000,
           "top_p": 0.9  # Can add any API parameters here
       }
   )
   ```

### 5. Module Organization

The package structure has been reorganized for better logical grouping:

- `lib/components/` → `context/` - Better reflects the purpose of these components
- `lib/factories/` → `connectors/` - Groups all connectivity-related functionality
- MCP-specific functionality is now under `connectors.mcp/`
- All base classes are available from the main package

### 6. Benefits of v2.0

The v2.0 upgrade brings several key benefits:

1. **Shorter imports**: Eliminated `.lib` from import paths
2. **Consistent API**: All base classes from main package
3. **Cleaner code**: More readable import statements
4. **Better organization**:
   - `components` → `context` (better reflects purpose)
   - `factories` → `connectors` with MCP-specific functionality grouped under `connectors.mcp`
   - `connectors` structure allows future extension for agent-to-agent communications and other connectivity modules
   - More intuitive class naming (`BaseDynamicContextProvider` vs `SystemPromptContextProviderBase`)

### 7. Streaming and Async Support

v2.0 reorganizes methods to clearly separate streaming from non-streaming operations:

#### Method Overview

| Client Type | Non-Streaming | Streaming |
|-------------|---------------|-----------|
| Sync (OpenAI) | `run()` | `run_stream()` (NEW) |
| Async (AsyncOpenAI) | `run_async()` (behavior changed) | `run_async_stream()` (NEW) |

#### Synchronous Operations

```python
# Example: A chatbot that responds to user queries
from openai import OpenAI
from atomic_agents import AtomicAgent, AgentConfig, BasicChatInputSchema, BasicChatOutputSchema
from atomic_agents.context import ChatHistory

client = instructor.from_openai(OpenAI())
agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
    AgentConfig(client=client, model="gpt-5-mini", history=ChatHistory())
)

# Non-streaming (same as v1.x) - Wait for complete response
user_input = BasicChatInputSchema(chat_message="Explain quantum computing")
response = agent.run(user_input)
print(response.chat_message)  # Prints complete response at once

# Streaming (NEW in v2.0) - Show response as it's generated
user_input = BasicChatInputSchema(chat_message="Write a story about a robot")
for partial in agent.run_stream(user_input):
    # Print incrementally as content arrives
    print(partial.chat_message, end='', flush=True)
print()  # New line after complete
```

#### Asynchronous Operations

```python
# Example: An async chatbot for handling multiple conversations
from openai import AsyncOpenAI
import asyncio

client = instructor.from_openai(AsyncOpenAI())
agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
    AgentConfig(client=client, model="gpt-5-mini", history=ChatHistory())
)

# OLD (v1.x) - run_async was a streaming generator
async def old_chat():
    user_input = BasicChatInputSchema(chat_message="Hello!")
    async for partial in agent.run_async(user_input):
        print(partial.chat_message)  # This was streaming

# NEW (v2.0) - run_async returns complete response
async def new_chat_complete():
    user_input = BasicChatInputSchema(chat_message="Hello!")
    response = await agent.run_async(user_input)
    print(response.chat_message)  # Complete response

# NEW (v2.0) - run_async_stream for streaming
async def new_chat_stream():
    user_input = BasicChatInputSchema(chat_message="Tell me a joke")
    async for partial in agent.run_async_stream(user_input):
        print(partial.chat_message, end='', flush=True)
    print()  # New line after complete
```

**Key Breaking Change**: In v1.x, `run_async()` was a streaming generator. In v2.0, it returns a complete response. Use `run_async_stream()` for async streaming.

#### How to Migrate

If you were using `run_async()` for streaming:
```python
# OLD
async for partial in agent.run_async(user_input):
    print(partial.chat_message)

# NEW - Option 1: Get complete response
response = await agent.run_async(user_input)
print(response.chat_message)

# NEW - Option 2: Keep streaming behavior
async for partial in agent.run_async_stream(user_input):
    print(partial.chat_message)
```

### 8. MCP Tool Factory Enhancements

The MCP (Model Context Protocol) integration has been significantly enhanced:

#### New Transport Types
```python
from atomic_agents.connectors.mcp import MCPTransportType

# v2.0 adds HTTP_STREAM transport as the new default
transport_type = MCPTransportType.HTTP_STREAM  # New default (changed from SSE)
# Also available: MCPTransportType.SSE, MCPTransportType.STDIO
```

**Important**: The default transport type has changed from `SSE` in v1.x to `HTTP_STREAM` in v2.0. If your MCP servers only support SSE, you'll need to explicitly specify it.

#### Async Tool Execution
MCP tools now expose an `arun` method for async execution:

```python
# Each MCP tool now has both sync and async entry points
result = tool.run(params)  # Synchronous
result = await tool.arun(params)  # Asynchronous (new in v2.0)
```

**Note**: The `arun` method is automatically generated for MCP tools during tool factory creation.

#### Fetching Tools Without Event Loop
```python
from atomic_agents.connectors.mcp import fetch_mcp_tools_async

# New async fetcher that doesn't require an active event loop
tools = await fetch_mcp_tools_async(
    endpoint="http://localhost:3000",
    transport_type=MCPTransportType.HTTP_STREAM
)
```

## Migration Strategy

### Required Migration
This upgrade introduces breaking changes. All projects must be updated to use the new import paths and class names.

### Recommended Approach
1. **Update all imports**: Replace old `.lib` imports with the new, shorter paths.
2. **Rename all classes**: Use find-and-replace to update all class names as specified above.
3. **Update tool definitions**: Convert tools to use generic type parameters.
4. **Test thoroughly**: Ensure all parts of your application work as expected after the migration.

## Additional Updates

### YouTube API Update
The YouTube API dependency has been updated to version 1.1.1 to resolve compatibility issues with lower versions.

### Package Dependencies
Examples and forge dependencies have been updated using `uv sync` to ensure compatibility with the new package structure.

## Version Information

These improvements are available starting from Atomic Agents v2.0.0.

## Complete Migration Examples

### Example 1: Migrating a Custom Tool

Here's how to migrate a custom tool from v1.x to v2.0:

#### Before (v1.x)

```python
from atomic_agents.lib.base.base_tool import BaseTool, BaseToolConfig
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from pydantic import Field
import requests

class WeatherToolInputSchema(BaseIOSchema):
    """Input schema for weather tool"""
    city: str = Field(..., description="City name to get weather for")
    units: str = Field(default="metric", description="Temperature units (metric/imperial)")

class WeatherToolOutputSchema(BaseIOSchema):
    """Output schema for weather tool"""
    temperature: float = Field(..., description="Current temperature")
    description: str = Field(..., description="Weather description")
    humidity: int = Field(..., description="Humidity percentage")

class WeatherTool(BaseTool):
    """Tool for fetching weather information"""

    input_schema = WeatherToolInputSchema
    output_schema = WeatherToolOutputSchema

    def __init__(self, api_key: str, config: BaseToolConfig = BaseToolConfig()):
        super().__init__(config)
        self.api_key = api_key

    def run(self, params: WeatherToolInputSchema) -> WeatherToolOutputSchema:
        # Tool implementation
        response = requests.get(
            f"https://api.weather.com/v1/weather",
            params={"city": params.city, "units": params.units, "api_key": self.api_key}
        )
        data = response.json()
        return WeatherToolOutputSchema(
            temperature=data["temp"],
            description=data["description"],
            humidity=data["humidity"]
        )
```

#### After (v2.0)

```python
from atomic_agents import BaseTool, BaseToolConfig, BaseIOSchema
from pydantic import Field
import requests

class WeatherToolInputSchema(BaseIOSchema):
    """Input schema for weather tool"""
    city: str = Field(..., description="City name to get weather for")
    units: str = Field(default="metric", description="Temperature units (metric/imperial)")

class WeatherToolOutputSchema(BaseIOSchema):
    """Output schema for weather tool"""
    temperature: float = Field(..., description="Current temperature")
    description: str = Field(..., description="Weather description")
    humidity: int = Field(..., description="Humidity percentage")

class WeatherTool(BaseTool[WeatherToolInputSchema, WeatherToolOutputSchema]):
    """Tool for fetching weather information"""

    def __init__(self, api_key: str, config: BaseToolConfig = BaseToolConfig()):
        super().__init__(config)
        self.api_key = api_key

    def run(self, params: WeatherToolInputSchema) -> WeatherToolOutputSchema:
        # Tool implementation (unchanged)
        response = requests.get(
            f"https://api.weather.com/v1/weather",
            params={"city": params.city, "units": params.units, "api_key": self.api_key}
        )
        data = response.json()
        return WeatherToolOutputSchema(
            temperature=data["temp"],
            description=data["description"],
            humidity=data["humidity"]
        )
```

### Example 2: Migrating a Customer Support Agent

Here's a real-world example of migrating a customer support agent:

#### Before (v1.x)

```python
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseAgentInputSchema
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from atomic_agents.lib.components.agent_memory import AgentMemory
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from pydantic import Field
from typing import List
import instructor
from openai import OpenAI

# Custom schemas for a support agent
class SupportTicketInput(BaseIOSchema):
    customer_name: str = Field(..., description="Customer's name")
    issue_description: str = Field(..., description="Description of the issue")
    priority: str = Field(..., description="Priority level: low, medium, high")

class SupportTicketOutput(BaseIOSchema):
    response_message: str = Field(..., description="Response to the customer")
    suggested_actions: List[str] = Field(..., description="Suggested actions to resolve the issue")
    estimated_resolution_time: str = Field(..., description="Estimated time to resolve")
    needs_escalation: bool = Field(..., description="Whether this needs escalation")

# Setup
client = instructor.from_openai(OpenAI())
memory = AgentMemory()

# System prompt configuration
system_prompt = SystemPromptGenerator(
    background=["You are a helpful customer support agent."],
    steps=["Analyze the issue", "Provide a solution", "Determine if escalation is needed"],
    output_instructions=["Be empathetic and professional"]
)

# Create agent with old configuration
agent = BaseAgent(
    BaseAgentConfig(
        client=client,
        model="gpt-5-mini",
        memory=memory,
        system_prompt_generator=system_prompt,
        input_schema=SupportTicketInput,
        output_schema=SupportTicketOutput,
        temperature=0.7,
        max_tokens=500
    )
)

# Use the agent
ticket = SupportTicketInput(
    customer_name="John Doe",
    issue_description="My order hasn't arrived after 2 weeks",
    priority="high"
)
response = agent.run(ticket)
print(f"Response: {response.response_message}")
print(f"Needs escalation: {response.needs_escalation}")
```

#### After (v2.0)

```python
from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from atomic_agents.context import ChatHistory, SystemPromptGenerator
from pydantic import Field
from typing import List
import instructor
from openai import OpenAI

# Same custom schemas (no changes needed)
class SupportTicketInput(BaseIOSchema):
    customer_name: str = Field(..., description="Customer's name")
    issue_description: str = Field(..., description="Description of the issue")
    priority: str = Field(..., description="Priority level: low, medium, high")

class SupportTicketOutput(BaseIOSchema):
    response_message: str = Field(..., description="Response to the customer")
    suggested_actions: List[str] = Field(..., description="Suggested actions to resolve the issue")
    estimated_resolution_time: str = Field(..., description="Estimated time to resolve")
    needs_escalation: bool = Field(..., description="Whether this needs escalation")

# Setup
client = instructor.from_openai(OpenAI())
history = ChatHistory()  # Renamed from AgentMemory

# System prompt configuration (same structure)
system_prompt = SystemPromptGenerator(
    background=["You are a helpful customer support agent."],
    steps=["Analyze the issue", "Provide a solution", "Determine if escalation is needed"],
    output_instructions=["Be empathetic and professional"]
)

# Create agent with new configuration
agent = AtomicAgent[SupportTicketInput, SupportTicketOutput](  # Schemas as type parameters
    AgentConfig(
        client=client,
        model="gpt-5-mini",
        history=history,  # Changed from memory
        system_prompt_generator=system_prompt,
        # No input_schema or output_schema in config
        model_api_parameters={  # Temperature and max_tokens moved here
            "temperature": 0.7,
            "max_tokens": 500
        }
    )
)

# Use the agent (same usage)
ticket = SupportTicketInput(
    customer_name="John Doe",
    issue_description="My order hasn't arrived after 2 weeks",
    priority="high"
)
response = agent.run(ticket)
print(f"Response: {response.response_message}")
print(f"Needs escalation: {response.needs_escalation}")

# NEW: Can also use streaming for real-time responses
print("\nStreaming response:")
for partial in agent.run_stream(ticket):
    print(partial.response_message, end='', flush=True)
```

## Quick Reference

| Feature | v1.x | v2.0 |
|---------|------|------|
| Python version | >=3.10 | >=3.12 |
| Base imports | `from atomic_agents.lib.base.base_io_schema import BaseIOSchema` | `from atomic_agents import BaseIOSchema` |
| Context imports | `from atomic_agents.lib.components.agent_memory import AgentMemory` | `from atomic_agents.context import ChatHistory` |
| MCP imports | `from atomic_agents.lib.factories.mcp_tool_factory import ...` | `from atomic_agents.connectors.mcp import ...` |
| Agent class | `BaseAgent` | `AtomicAgent` |
| Agent config | `BaseAgentConfig` | `AgentConfig` |
| Default input schema | `BaseAgentInputSchema` | `BasicChatInputSchema` |
| Default output schema | `BaseAgentOutputSchema` | `BasicChatOutputSchema` |
| Memory class | `AgentMemory()` | `ChatHistory()` |
| Context provider base | `class MyProvider(SystemPromptContextProviderBase)` | `class MyProvider(BaseDynamicContextProvider)` |
| Agent with custom schemas | `BaseAgent(config(input_schema=QueryInput, output_schema=AnalysisOutput))` | `AtomicAgent[QueryInput, AnalysisOutput](config)` |
| Tool with custom schemas | `class MyTool(BaseTool): input_schema = In; output_schema = Out` | `class MyTool(BaseTool[In, Out]): pass` |
| Async non-streaming | `async for partial in agent.run_async(input)` | `response = await agent.run_async(input)` |
| Async streaming | `async for partial in agent.run_async(input)` | `async for partial in agent.run_async_stream(input)` |
| Sync streaming | Not available | `for partial in agent.run_stream(input)` |
| Reset conversation | `agent.reset_memory()` | `agent.reset_history()` |
| Model parameters | `config(temperature=0.7, max_tokens=500)` | `config(model_api_parameters={"temperature": 0.7, "max_tokens": 500})` |

## Troubleshooting

### Import Errors

If you encounter import errors after upgrading:

1. Ensure all old `.lib` imports are updated
2. Check that class renames are applied:
   - `BaseAgent` → `AtomicAgent`
   - `BaseAgentConfig` → `AgentConfig`
   - `BaseAgentInputSchema` → `BasicChatInputSchema`
   - `BaseAgentOutputSchema` → `BasicChatOutputSchema`
   - `AgentMemory` → `ChatHistory`
3. Verify that the package is correctly installed with v2.0

### Type Errors

If you get type errors with the new generic AtomicAgent or BaseTool:

1. Ensure you're specifying both input and output schemas: `AtomicAgent[InputSchema, OutputSchema]`
2. Remove `input_schema` and `output_schema` from `AgentConfig`
3. For tools, update from `class MyTool(BaseTool)` to `class MyTool(BaseTool[InputSchema, OutputSchema])`
4. Make sure your schemas inherit from `BaseIOSchema`
5. If using default schemas, specify them explicitly: `AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema]`

### Runtime Errors

If you encounter runtime errors:

1. Check that all references to `memory` are updated to `history`
2. Verify that custom context providers inherit from `BaseDynamicContextProvider`
3. Ensure Python version is 3.12 or higher

## Support

If you encounter issues during migration:

1. Check the [GitHub Issues](https://github.com/eigenwise/atomic-agents/issues)
2. Join the [Discord community](https://discord.gg/J3W9b5AZJR)
3. Visit the [subreddit](https://www.reddit.com/r/AtomicAgents/)

