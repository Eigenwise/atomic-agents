# Quickstart Guide

**See also:**

- [Quickstart runnable examples on GitHub](https://github.com/eigenwise/atomic-agents/tree/main/atomic-examples/quickstart)
- [All Atomic Agents examples on GitHub](https://github.com/eigenwise/atomic-agents/tree/main/atomic-examples)

This guide will help you get started with the Atomic Agents framework. We'll cover basic usage, custom agents, and different AI providers.

## Installation

First, install the package using pip:

```bash
pip install atomic-agents
```

## Basic Chatbot

Let's start with a simple chatbot:

```python
import os
import instructor
import openai
from rich.console import Console
from atomic_agents.context import ChatHistory
from atomic_agents import AtomicAgent, AgentConfig, BasicChatInputSchema, BasicChatOutputSchema

# Initialize console for pretty outputs
console = Console()

# History setup
history = ChatHistory()

# Initialize history with an initial message from the assistant
initial_message = BasicChatOutputSchema(chat_message="Hello! How can I assist you today?")
history.add_message("assistant", initial_message)

# OpenAI client setup using the Instructor library
client = instructor.from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

# Create agent with type parameters
agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
    config=AgentConfig(
        client=client,
        model="gpt-5-mini",  # Using the latest model
        history=history,
        model_api_parameters={"max_tokens": 2048}
    )
)

# Start a loop to handle user inputs and agent responses
while True:
    # Prompt the user for input
    user_input = console.input("[bold blue]You:[/bold blue] ")
    # Check if the user wants to exit the chat
    if user_input.lower() in ["/exit", "/quit"]:
        console.print("Exiting chat...")
        break

    # Process the user's input through the agent and get the response
    input_schema = BasicChatInputSchema(chat_message=user_input)
    response = agent.run(input_schema)

    # Display the agent's response
    console.print("Agent: ", response.chat_message)
```

## Token Counting

Monitor your context usage with the `get_context_token_count()` method. Token counts are computed accurately on-demand by serializing the context exactly as Instructor does, including the output schema overhead. This works with any provider (OpenAI, Anthropic, Google, Groq, etc.) and supports multimodal content (images, PDFs, audio):

```python
# Get accurate token count at any time - no need to make an API call first
token_info = agent.get_context_token_count()
print(f"Total tokens: {token_info.total}")
print(f"System prompt (with schema): {token_info.system_prompt} tokens")
print(f"History: {token_info.history} tokens")

# Check context utilization (if model's max tokens is known)
if token_info.max_tokens:
    print(f"Max context: {token_info.max_tokens} tokens")
if token_info.utilization:
    print(f"Context utilization: {token_info.utilization:.1%}")
```

You can add a `/tokens` command to your chatbot for easy monitoring:

```python
while True:
    user_input = console.input("[bold blue]You:[/bold blue] ")

    if user_input.lower() in ["/exit", "/quit"]:
        break

    # Add token counting command
    if user_input.lower() == "/tokens":
        token_info = agent.get_context_token_count()
        console.print(f"[bold magenta]Token Usage:[/bold magenta]")
        console.print(f"  Total: {token_info.total} tokens")
        console.print(f"  System prompt: {token_info.system_prompt} tokens")
        console.print(f"  History: {token_info.history} tokens")
        if token_info.utilization:
            console.print(f"  Context utilization: {token_info.utilization:.1%}")
        continue

    # Process normal input
    input_schema = BasicChatInputSchema(chat_message=user_input)
    response = agent.run(input_schema)
    console.print("Agent: ", response.chat_message)
```

## Streaming Responses

For a more interactive experience, you can use streaming with async processing:

```python
import os
import instructor
import openai
import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from atomic_agents.context import ChatHistory
from atomic_agents import AtomicAgent, AgentConfig, BasicChatInputSchema, BasicChatOutputSchema

# Initialize console for pretty outputs
console = Console()

# History setup
history = ChatHistory()

# Initialize history with an initial message from the assistant
initial_message = BasicChatOutputSchema(chat_message="Hello! How can I assist you today?")
history.add_message("assistant", initial_message)

# OpenAI client setup using the Instructor library for async operations
client = instructor.from_openai(openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))

# Agent setup with specified configuration
agent = AtomicAgent(
    config=AgentConfig(
        client=client,
        model="gpt-5-mini",
        history=history,
    )
)

# Display the initial message from the assistant
console.print(Text("Agent:", style="bold green"), end=" ")
console.print(Text(initial_message.chat_message, style="green"))

async def main():
    # Start an infinite loop to handle user inputs and agent responses
    while True:
        # Prompt the user for input with a styled prompt
        user_input = console.input("\n[bold blue]You:[/bold blue] ")
        # Check if the user wants to exit the chat
        if user_input.lower() in ["/exit", "/quit"]:
            console.print("Exiting chat...")
            break

        # Process the user's input through the agent and get the streaming response
        input_schema = BasicChatInputSchema(chat_message=user_input)
        console.print()  # Add newline before response

        # Use Live display to show streaming response
        with Live("", refresh_per_second=10, auto_refresh=True) as live:
            current_response = ""
            async for partial_response in agent.run_async(input_schema):
                if hasattr(partial_response, "chat_message") and partial_response.chat_message:
                    # Only update if we have new content
                    if partial_response.chat_message != current_response:
                        current_response = partial_response.chat_message
                        # Combine the label and response in the live display
                        display_text = Text.assemble(("Agent: ", "bold green"), (current_response, "green"))
                        live.update(display_text)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Custom Input/Output Schema

For more structured interactions, define custom schemas:

```python
import os
import instructor
import openai
from rich.console import Console
from typing import List
from pydantic import Field
from atomic_agents.context import ChatHistory, SystemPromptGenerator
from atomic_agents import AtomicAgent, AgentConfig, BasicChatInputSchema, BaseIOSchema

# Initialize console for pretty outputs
console = Console()

# History setup
history = ChatHistory()

# Custom output schema
class CustomOutputSchema(BaseIOSchema):
    """This schema represents the response generated by the chat agent, including suggested follow-up questions."""

    chat_message: str = Field(
        ...,
        description="The chat message exchanged between the user and the chat agent.",
    )
    suggested_user_questions: List[str] = Field(
        ...,
        description="A list of suggested follow-up questions the user could ask the agent.",
    )

# Initialize history with an initial message from the assistant
initial_message = CustomOutputSchema(
    chat_message="Hello! How can I assist you today?",
    suggested_user_questions=["What can you do?", "Tell me a joke", "Tell me about how you were made"],
)
history.add_message("assistant", initial_message)

# OpenAI client setup using the Instructor library
client = instructor.from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

# Custom system prompt
system_prompt_generator = SystemPromptGenerator(
    background=[
        "This assistant is a knowledgeable AI designed to be helpful, friendly, and informative.",
        "It has a wide range of knowledge on various topics and can engage in diverse conversations.",
    ],
    steps=[
        "Analyze the user's input to understand the context and intent.",
        "Formulate a relevant and informative response based on the assistant's knowledge.",
        "Generate 3 suggested follow-up questions for the user to explore the topic further.",
    ],
    output_instructions=[
        "Provide clear, concise, and accurate information in response to user queries.",
        "Maintain a friendly and professional tone throughout the conversation.",
        "Conclude each response with 3 relevant suggested questions for the user.",
    ],
)

# Agent setup with specified configuration and custom output schema
agent = AtomicAgent[BasicChatInputSchema, CustomOutputSchema](
    config=AgentConfig(
        client=client,
        model="gpt-5-mini",
        system_prompt_generator=system_prompt_generator,
        history=history,
    )
)

# Start a loop to handle user inputs and agent responses
while True:
    # Prompt the user for input
    user_input = console.input("[bold blue]You:[/bold blue] ")
    # Check if the user wants to exit the chat
    if user_input.lower() in ["/exit", "/quit"]:
        console.print("Exiting chat...")
        break

    # Process the user's input through the agent
    input_schema = BasicChatInputSchema(chat_message=user_input)
    response = agent.run(input_schema)

    # Display the agent's response
    console.print("[bold green]Agent:[/bold green] ", response.chat_message)
    
    # Display the suggested questions
    console.print("\n[bold cyan]Suggested questions you could ask:[/bold cyan]")
    for i, question in enumerate(response.suggested_user_questions, 1):
        console.print(f"[cyan]{i}. {question}[/cyan]")
    console.print()  # Add an empty line for better readability
```

## Multiple AI Providers Support

The framework supports multiple AI providers:

```json
{
    "openai": "gpt-5-mini",
    "anthropic": "claude-3-5-haiku-20241022",
    "groq": "mixtral-8x7b-32768",
    "ollama": "llama3",
    "gemini": "gemini-2.0-flash-exp",
    "openrouter": "mistral/ministral-8b"
}
```

Here's how to set up clients for different providers:

```python
import os
import instructor
from rich.console import Console
from rich.text import Text
from atomic_agents.context import ChatHistory
from atomic_agents import AtomicAgent, AgentConfig, BasicChatInputSchema, BasicChatOutputSchema
from dotenv import load_dotenv

load_dotenv()

# Initialize console for pretty outputs
console = Console()

# History setup
history = ChatHistory()

# Initialize history with an initial message from the assistant
initial_message = BasicChatOutputSchema(chat_message="Hello! How can I assist you today?")
history.add_message("assistant", initial_message)

# Function to set up the client based on the chosen provider
def setup_client(provider):
    if provider == "openai":
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        client = instructor.from_openai(OpenAI(api_key=api_key))
        model = "gpt-5-mini"

    elif provider == "anthropic":
        from anthropic import Anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        client = instructor.from_anthropic(Anthropic(api_key=api_key))
        model = "claude-3-5-haiku-20241022"

    elif provider == "groq":
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY")
        client = instructor.from_groq(
            Groq(api_key=api_key),
            mode=instructor.Mode.JSON
        )
        model = "mixtral-8x7b-32768"

    elif provider == "ollama":
        from openai import OpenAI as OllamaClient
        client = instructor.from_openai(
            OllamaClient(
                base_url="http://localhost:11434/v1",
                api_key="ollama"
            ),
            mode=instructor.Mode.JSON
        )
        model = "llama3"

    elif provider == "gemini":
        from openai import OpenAI
        api_key = os.getenv("GEMINI_API_KEY")
        client = instructor.from_openai(
            OpenAI(
                api_key=api_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
            ),
            mode=instructor.Mode.JSON
        )
        model = "gemini-2.0-flash-exp"
        
    elif provider == "openrouter":
        from openai import OpenAI as OpenRouterClient
        api_key = os.getenv("OPENROUTER_API_KEY")
        client = instructor.from_openai(
            OpenRouterClient(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )
        )
        model = "mistral/ministral-8b"

    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return client, model

# Prompt for provider choice
provider = console.input("Choose a provider (openai/anthropic/groq/ollama/gemini/openrouter): ").lower()

# Set up client and model
client, model = setup_client(provider)

# Create agent with chosen provider
agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
    config=AgentConfig(
        client=client,
        model=model,
        history=history,
        model_api_parameters={"max_tokens": 2048}
    )
)
```

The framework supports multiple providers through Instructor:

- **OpenAI**: Standard GPT models
- **Anthropic**: Claude models
- **Groq**: Fast inference for open models
- **Ollama**: Local models (requires Ollama running)
- **Gemini**: Google's Gemini models

Each provider requires its own API key (except Ollama) which should be set in environment variables:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-key"

# Anthropic
export ANTHROPIC_API_KEY="your-anthropic-key"

# Groq
export GROQ_API_KEY="your-groq-key"

# Gemini
export GEMINI_API_KEY="your-gemini-key"

# OpenRouter
export OPENROUTER_API_KEY="your-openrouter-key"
```

## Running the Examples

To run any of these examples:

1. Save the code in a Python file (e.g., `chatbot.py`)
2. Set your API key as an environment variable:

   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```

3. Run the script:

   ```bash
   uv run python chatbot.py
   ```

## Set Up Your Project for AI Coding Assistants

If you build with an AI assistant (Claude Code, Cursor, Copilot, Codex, Gemini CLI, ...), two small steps make it know Atomic Agents instead of guessing at the API:

1. **Install the agent skills** — guided workflows for creating schemas, agents, tools, and context providers, plus a troubleshooting diagnostic:

   ```bash
   # Claude Code
   /plugin marketplace add eigenwise/atomic-agents
   /plugin install atomic-agents@eigenwise

   # Any other skills-compatible assistant
   npx skills add eigenwise/atomic-agents
   ```

2. **Drop an `AGENTS.md` into your project root** so every assistant picks up the framework conventions. Add a `CLAUDE.md` containing just `@AGENTS.md` so Claude Code reads the same file:

   ```markdown
   # My Project

   Built with [Atomic Agents](https://github.com/eigenwise/atomic-agents) — a schema-driven
   framework on Instructor + Pydantic. Docs for LLMs:
   https://eigenwise.github.io/atomic-agents/llms.txt

   ## Framework conventions

   - Import from the top-level package: `from atomic_agents import AtomicAgent, AgentConfig,
     BaseIOSchema, BaseTool`; context pieces from `atomic_agents.context`.
   - Agents are `AtomicAgent[InputSchema, OutputSchema](config=AgentConfig(...))` — the type
     parameters carry runtime information, keep them accurate.
   - The LLM client must be wrapped with Instructor before it goes into `AgentConfig.client`.
   - Every `BaseIOSchema` subclass needs a non-empty docstring and `Field(..., description=...)`
     on each field — both flow into the LLM prompt.
   - Provider knobs (`temperature`, `max_tokens`, ...) go in `AgentConfig.model_api_parameters`.
   ```

Projects scaffolded with the `new-app` skill get both files automatically. Assistants can also pull the full docs from [llms.txt](https://eigenwise.github.io/atomic-agents/llms.txt) or query them via [Context7](https://context7.com/eigenwise/atomic-agents).

## Next Steps

After trying these examples, you can:

1. Learn about [tools and their integration](tools.md)
2. Review the [API reference](../api/index) for detailed documentation

## Explore More Examples

For more advanced usage and examples, please check out the [Atomic Agents examples on GitHub](https://github.com/eigenwise/atomic-agents/tree/main/atomic-examples). These examples demonstrate various capabilities of the framework including custom schemas, advanced history usage, tool integration, and more.
