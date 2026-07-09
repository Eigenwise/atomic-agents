# Context

```{seealso}
For a comprehensive guide on memory management, multi-agent patterns, and best practices, see the **[Memory and Context Guide](/guides/memory)**.
```

## Agent History

The `ChatHistory` class manages conversation history and state for AI agents. It implements `BaseChatHistory`, an interface-only abstract base class that declares the memory contract `AtomicAgent` depends on (the type `AgentConfig.history` accepts). Implement `BaseChatHistory` directly, or subclass `ChatHistory`, to plug in your own persistent or backend-specific memory store. See the [Memory guide's "Writing a Custom Memory Backend" section](/guides/memory#writing-a-custom-memory-backend) for the full contract and a recommended pattern.

```python
from atomic_agents.context import ChatHistory
from atomic_agents import BaseIOSchema

# Initialize history with optional max messages
history = ChatHistory(max_messages=10)

# Add messages
history.add_message(
    role="user",
    content=BaseIOSchema(...)
)

# Initialize a new turn
history.initialize_turn()
turn_id = history.get_current_turn_id()

# Access history
history = history.get_history()

# Manage history
history.get_message_count()  # Get number of messages
history.delete_turn_id(turn_id)  # Delete messages by turn

# Persistence
serialized = history.dump()  # Save to string
history.load(serialized)  # Load from string

# Create copy
new_history = history.copy()
```

Key features:
- Message history management with role-based messages
- Turn-based conversation tracking
- Support for multimodal content (images, etc.)
- Serialization and persistence
- History size management
- Deep copy functionality

### Message Structure

Messages in history are structured as:

```python
class Message(BaseModel):
    role: str  # e.g., 'user', 'assistant', 'system'
    content: BaseIOSchema  # Message content following schema
    turn_id: Optional[str]  # Unique ID for grouping messages
```

### Multimodal Support

The history system automatically handles multimodal content:

```python
# For content with images
history = history.get_history()
for message in history:
    if isinstance(message.content, list):
        text_content = message.content[0]  # JSON string
        images = message.content[1:]  # List of images
```

## System Prompt Generator

The `SystemPromptGenerator` creates structured system prompts for AI agents:

```python
from atomic_agents.context import (
    SystemPromptGenerator,
    BaseDynamicContextProvider
)

# Create generator with static content
generator = SystemPromptGenerator(
    background=[
        "You are a helpful AI assistant.",
        "You specialize in technical support."
    ],
    steps=[
        "1. Understand the user's request",
        "2. Analyze available information",
        "3. Provide clear solutions"
    ],
    output_instructions=[
        "Use clear, concise language",
        "Include step-by-step instructions",
        "Cite relevant documentation"
    ]
)

# Generate prompt
prompt = generator.generate_prompt()
```

### Custom System Prompt Generator

If you require finer control over system prompt construction, subclass `BaseSystemPromptGenerator` and implement `generate_prompt()`. This approach is useful when prompt content should be maintained in a human-readable format (e.g., Markdown or text file) to allow review or editing by non-developers.

```python
from pathlib import Path
from typing import Dict, Optional, Union

from atomic_agents.context import (
    BaseDynamicContextProvider, 
    BaseSystemPromptGenerator
)


class MarkdownFileSystemPromptGenerator(BaseSystemPromptGenerator):
    def __init__(
        self,
        md_file: Union[Path, str],
        context_providers: Optional[Dict[str, BaseDynamicContextProvider]] = None,
    ):
        super().__init__(context_providers=context_providers)

        path = Path(md_file)
        if not path.exists():
            raise FileNotFoundError(f"System prompt file not found: {md_file}")

        self.system_prompt = path.read_text(encoding="utf-8")

    def generate_prompt(self) -> str:
        return f"{self.system_prompt}\n\n{self._build_context_string()}"

    def _build_context_string(self) -> str:
        if not self.context_providers:
            return ""

        context_sections = ["# Additional Context"]
        for provider in self.context_providers.values():
            info = provider.get_info()
            if info:
                context_sections.append(f"## {provider.title}")
                context_sections.append(info)
                context_sections.append("")
        return "\n".join(context_sections).strip()

generator = MarkdownFileSystemPromptGenerator("path/to/system_prompt.md")
prompt = generator.generate_prompt()
```

### Dynamic Context Providers

Context providers inject dynamic information into prompts:

```python
from dataclasses import dataclass
from typing import List

@dataclass
class SearchResult:
    content: str
    metadata: dict

class SearchResultsProvider(BaseDynamicContextProvider):
    def __init__(self, title: str):
        super().__init__(title=title)
        self.results: List[SearchResult] = []

    def get_info(self) -> str:
        """Format search results for the prompt"""
        if not self.results:
            return "No search results available."

        return "\n\n".join([
            f"Result {idx}:\nMetadata: {result.metadata}\nContent:\n{result.content}\n{'-' * 80}"
            for idx, result in enumerate(self.results, 1)
        ])

# Use with generator
generator = SystemPromptGenerator(
    background=["You answer based on search results."],
    context_providers={
        "search_results": SearchResultsProvider("Search Results")
    }
)
```

The generated prompt will include:
1. Background information
2. Processing steps (if provided)
3. Dynamic context from providers
4. Output instructions

## Base Components

### BaseIOSchema

Base class for all input/output schemas:

```python
from atomic_agents import BaseIOSchema
from pydantic import Field

class CustomSchema(BaseIOSchema):
    """Schema description (required)"""
    field: str = Field(..., description="Field description")
```

Key features:
- Requires docstring description
- Rich representation support
- Automatic schema validation
- JSON serialization

### BaseTool

Base class for creating tools:

```python
from atomic_agents import BaseTool, BaseToolConfig
from pydantic import Field

class MyToolConfig(BaseToolConfig):
    """Tool configuration"""
    api_key: str = Field(
        default=os.getenv("API_KEY"),
        description="API key for the service"
    )

class MyTool(BaseTool[MyToolInputSchema, MyToolOutputSchema]):
    """Tool implementation"""
    input_schema = MyToolInputSchema
    output_schema = MyToolOutputSchema

    def __init__(self, config: MyToolConfig = MyToolConfig()):
        super().__init__(config)
        self.api_key = config.api_key

    def run(self, params: MyToolInputSchema) -> MyToolOutputSchema:
        # Implement tool logic
        pass
```

Key features:

- Structured input/output schemas
- Configuration management
- Title and description overrides
- Error handling

For full API details:

```{eval-rst}
.. automodule:: atomic_agents.context.chat_history
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: atomic_agents.context.base_chat_history
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: atomic_agents.context.system_prompt_generator
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: atomic_agents.base.base_io_schema
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: atomic_agents.base.base_tool
   :members:
   :undoc-members:
   :show-inheritance:
```
