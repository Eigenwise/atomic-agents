# Frequently Asked Questions

Common questions and answers about using Atomic Agents.

## Installation & Setup

### How do I install Atomic Agents?

Install using pip:

```bash
pip install atomic-agents
```

Or using uv (recommended):

```bash
uv add atomic-agents
```

You also need to install your LLM provider. OpenAI is included by default. For other providers, use instructor extras:

```bash
# For Anthropic
pip install instructor[anthropic]

# For Groq
pip install instructor[groq]

# For Gemini
pip install instructor[google-genai]
```

### What Python version is required?

Atomic Agents requires **Python 3.12 or higher**.

```bash
# Check your Python version
python --version
```

### How do I set up my API key?

Set your API key as an environment variable:

```bash
# OpenAI
export OPENAI_API_KEY="your-api-key"

# Anthropic
export ANTHROPIC_API_KEY="your-api-key"

# Or use a .env file with python-dotenv
```

In your code:

```python
import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file

# Keys are read from environment
api_key = os.getenv("OPENAI_API_KEY")
```

## Agent Configuration

### How do I create a basic agent?

```python
import instructor
import openai
from atomic_agents import AtomicAgent, AgentConfig, BasicChatInputSchema, BasicChatOutputSchema
from atomic_agents.context import ChatHistory

# Create instructor client
client = instructor.from_openai(openai.OpenAI())

# Create agent
agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
    config=AgentConfig(
        client=client,
        model="gpt-5.6-terra",
        history=ChatHistory()
    )
)

# Use the agent
response = agent.run(BasicChatInputSchema(chat_message="Hello!"))
print(response.chat_message)
```

### How do I use different LLM providers?

Atomic Agents works with any provider supported by Instructor:

**OpenAI:**
```python
import instructor
import openai

client = instructor.from_openai(openai.OpenAI())
```

**Anthropic:**
```python
import instructor
from anthropic import Anthropic

client = instructor.from_anthropic(Anthropic())
```

**Groq:**
```python
import instructor
from groq import Groq

client = instructor.from_groq(Groq(), mode=instructor.Mode.JSON)
```

**Ollama (local models):**
```python
import instructor
from openai import OpenAI

client = instructor.from_openai(
    OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    ),
    mode=instructor.Mode.JSON
)
```

**Google Gemini:**
```python
import instructor
from openai import OpenAI
import os

client = instructor.from_openai(
    OpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
    ),
    mode=instructor.Mode.JSON
)
```

### How do I customize the system prompt?

Use `SystemPromptGenerator` to define agent behavior:

```python
from atomic_agents.context import SystemPromptGenerator

system_prompt = SystemPromptGenerator(
    background=[
        "You are a helpful coding assistant.",
        "You specialize in Python programming."
    ],
    steps=[
        "Analyze the user's question.",
        "Provide clear, working code examples.",
        "Explain the code step by step."
    ],
    output_instructions=[
        "Always include code examples.",
        "Use markdown formatting.",
        "Keep explanations concise."
    ]
)

agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
    config=AgentConfig(
        client=client,
        model="gpt-5.6-terra",
        system_prompt_generator=system_prompt
    )
)
```

### How do I add memory/conversation history?

Use `ChatHistory` to maintain conversation context:

```python
from atomic_agents.context import ChatHistory

# Create history
history = ChatHistory()

# Create agent with history
agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
    config=AgentConfig(
        client=client,
        model="gpt-5.6-terra",
        history=history
    )
)

# Conversation is automatically maintained
agent.run(BasicChatInputSchema(chat_message="My name is Alice"))
agent.run(BasicChatInputSchema(chat_message="What's my name?"))  # Will remember "Alice"

# Reset history when needed
agent.reset_history()
```

## Custom Schemas

### How do I create custom input/output schemas?

Inherit from `BaseIOSchema`:

```python
from typing import List, Optional
from pydantic import Field
from atomic_agents import BaseIOSchema


class CustomInputSchema(BaseIOSchema):
    """Custom input with additional fields."""

    question: str = Field(..., description="The user's question")
    context: Optional[str] = Field(None, description="Additional context")
    max_length: int = Field(default=500, description="Max response length")


class CustomOutputSchema(BaseIOSchema):
    """Custom output with structured data."""

    answer: str = Field(..., description="The answer to the question")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    sources: List[str] = Field(default_factory=list, description="Source references")
    follow_up_questions: List[str] = Field(default_factory=list, description="Suggested follow-ups")


# Use with agent
agent = AtomicAgent[CustomInputSchema, CustomOutputSchema](
    config=AgentConfig(client=client, model="gpt-5.6-terra")
)

response = agent.run(CustomInputSchema(question="What is Python?"))
print(f"Answer: {response.answer}")
print(f"Confidence: {response.confidence}")
```

### How do I add validation to schemas?

Use Pydantic validators:

```python
from pydantic import Field, field_validator, model_validator
from atomic_agents import BaseIOSchema


class ValidatedInputSchema(BaseIOSchema):
    """Input with validation rules."""

    query: str = Field(..., min_length=1, max_length=1000)
    category: str = Field(...)

    @field_validator('category')
    @classmethod
    def validate_category(cls, v: str) -> str:
        valid = ['tech', 'science', 'business']
        if v.lower() not in valid:
            raise ValueError(f"Category must be one of: {valid}")
        return v.lower()

    @field_validator('query')
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode='after')
    def validate_combination(self):
        # Cross-field validation
        if self.category == 'tech' and len(self.query) < 10:
            raise ValueError("Tech queries must be at least 10 characters")
        return self
```

## Tools

### How do I create a custom tool?

Inherit from `BaseTool`:

```python
import os
from pydantic import Field
from atomic_agents import BaseTool, BaseToolConfig, BaseIOSchema


class WeatherInputSchema(BaseIOSchema):
    """Input for weather tool."""
    city: str = Field(..., description="City name to get weather for")


class WeatherOutputSchema(BaseIOSchema):
    """Output from weather tool."""
    temperature: float = Field(..., description="Temperature in Celsius")
    condition: str = Field(..., description="Weather condition")
    humidity: int = Field(..., description="Humidity percentage")


class WeatherToolConfig(BaseToolConfig):
    """Configuration for weather tool."""
    api_key: str = Field(default_factory=lambda: os.getenv("WEATHER_API_KEY"))


class WeatherTool(BaseTool[WeatherInputSchema, WeatherOutputSchema]):
    """Tool to fetch current weather."""

    def __init__(self, config: WeatherToolConfig = None):
        super().__init__(config or WeatherToolConfig())
        self.api_key = self.config.api_key

    def run(self, params: WeatherInputSchema) -> WeatherOutputSchema:
        # Implement your tool logic here
        # This is a mock implementation
        return WeatherOutputSchema(
            temperature=22.5,
            condition="Sunny",
            humidity=45
        )


# Use the tool
tool = WeatherTool()
result = tool.run(WeatherInputSchema(city="London"))
print(f"Temperature: {result.temperature}°C")
```

### How do I use the built-in tools?

Use the Atomic Assembler CLI to download tools:

```bash
atomic
```

Then import and use them:

```python
from calculator.tool.calculator import CalculatorTool, CalculatorInputSchema

calculator = CalculatorTool()
result = calculator.run(CalculatorInputSchema(expression="2 + 2 * 3"))
print(result.value)  # 8.0
```

## Streaming & Async

### How do I stream responses?

Use `run_stream()` for synchronous streaming:

```python
from atomic_agents import AtomicAgent, AgentConfig, BasicChatInputSchema, BasicChatOutputSchema

# Synchronous streaming
for partial in agent.run_stream(BasicChatInputSchema(chat_message="Write a poem")):
    print(partial.chat_message, end='', flush=True)
print()  # Newline at end
```

### How do I use async methods?

Use `run_async()` for async operations:

```python
import asyncio
from openai import AsyncOpenAI
import instructor
from atomic_agents import AtomicAgent, AgentConfig, BasicChatInputSchema, BasicChatOutputSchema
from atomic_agents.context import ChatHistory


async def main():
    # Use async client
    client = instructor.from_openai(AsyncOpenAI())

    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
        config=AgentConfig(
            client=client,
            model="gpt-5.6-terra",
            history=ChatHistory()
        )
    )

    # Non-streaming async
    response = await agent.run_async(BasicChatInputSchema(chat_message="Hello"))
    print(response.chat_message)

    # Streaming async
    async for partial in agent.run_async_stream(BasicChatInputSchema(chat_message="Write a story")):
        print(partial.chat_message, end='', flush=True)


asyncio.run(main())
```

## Context Providers

### How do I inject dynamic context?

Create a custom context provider:

```python
from typing import List
from atomic_agents.context import BaseDynamicContextProvider


class SearchResultsProvider(BaseDynamicContextProvider):
    """Provides search results as context."""

    def __init__(self, title: str = "Search Results"):
        super().__init__(title=title)
        self.results: List[str] = []

    def add_result(self, result: str):
        self.results.append(result)

    def clear(self):
        self.results = []

    def get_info(self) -> str:
        if not self.results:
            return "No search results available."
        return "\n".join(f"- {r}" for r in self.results)


# Register with agent
provider = SearchResultsProvider()
provider.add_result("Python is a programming language")
provider.add_result("Python was created by Guido van Rossum")

agent.register_context_provider("search_results", provider)

# The context is now included in the system prompt
response = agent.run(BasicChatInputSchema(chat_message="Tell me about Python"))
```

## Common Issues

### Why am I getting validation errors?

Check that your input matches the schema:

```python
from pydantic import ValidationError

try:
    response = agent.run(BasicChatInputSchema(chat_message=""))
except ValidationError as e:
    print("Validation errors:")
    for error in e.errors():
        print(f"  {error['loc']}: {error['msg']}")
```

### How do I handle API rate limits?

Implement retry logic:

```python
import time
from openai import RateLimitError


def run_with_retry(agent, input_data, max_retries=3):
    for attempt in range(max_retries):
        try:
            return agent.run(input_data)
        except RateLimitError:
            if attempt < max_retries - 1:
                wait = 2 ** attempt  # Exponential backoff
                print(f"Rate limited. Waiting {wait}s...")
                time.sleep(wait)
            else:
                raise
```

### How do I debug agent behavior?

1. **Check the system prompt:**
```python
print(agent.system_prompt_generator.generate_prompt())
```

2. **Inspect history:**
```python
for msg in agent.history.get_history():
    print(f"{msg['role']}: {msg['content']}")
```

3. **Enable logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## MCP Integration

### How do I connect to an MCP server?

```python
from atomic_agents.connectors.mcp import fetch_mcp_tools_async, MCPTransportType


async def setup_mcp_tools():
    tools = await fetch_mcp_tools_async(
        server_url="http://localhost:8000",
        transport_type=MCPTransportType.HTTP_STREAM
    )
    return tools


# Use tools with your agent
tools = asyncio.run(setup_mcp_tools())
```

## Migration

### How do I upgrade from v1.x to v2.0?

Key changes:

1. **Import paths:**
```python
# Old
from atomic_agents.lib.base.base_io_schema import BaseIOSchema

# New
from atomic_agents import BaseIOSchema
```

2. **Class names:**
```python
# Old
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig

# New
from atomic_agents import AtomicAgent, AgentConfig
```

3. **Schemas as type parameters:**
```python
# Old
agent = BaseAgent(BaseAgentConfig(
    client=client,
    model="gpt-5.6-terra",
    input_schema=MyInput,
    output_schema=MyOutput
))

# New
agent = AtomicAgent[MyInput, MyOutput](
    AgentConfig(client=client, model="gpt-5.6-terra")
)
```

See the [Upgrade Guide](../../UPGRADE_DOC.md) for complete migration instructions.
