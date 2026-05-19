# Atomic Agents

<img src="./.assets/logo.png" alt="Atomic Agents" width="350"/>

[![PyPI version](https://badge.fury.io/py/atomic-agents.svg)](https://badge.fury.io/py/atomic-agents)
[![Documentation](https://img.shields.io/badge/docs-read%20the%20docs-blue?logo=readthedocs&style=flat-square)](https://brainblend-ai.github.io/atomic-agents/)
[![Build Docs](https://github.com/BrainBlend-AI/atomic-agents/actions/workflows/docs.yml/badge.svg)](https://github.com/BrainBlend-AI/atomic-agents/actions/workflows/docs.yml)
[![Code Quality](https://github.com/BrainBlend-AI/atomic-agents/actions/workflows/code-quality.yml/badge.svg)](https://github.com/BrainBlend-AI/atomic-agents/actions/workflows/code-quality.yml)
[![Discord](https://img.shields.io/badge/chat-on%20discord-7289DA?logo=discord&style=flat-square)](https://discord.gg/J3W9b5AZJR)
[![PyPI downloads](https://img.shields.io/pypi/dm/atomic-agents?style=flat-square)](https://pypi.org/project/atomic-agents/)
[![Python Versions](https://img.shields.io/pypi/pyversions/atomic-agents?style=flat-square)](https://pypi.org/project/atomic-agents/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow?style=flat-square)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/BrainBlend-AI/atomic-agents?style=social)](https://github.com/BrainBlend-AI/atomic-agents/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/BrainBlend-AI/atomic-agents?style=social)](https://github.com/BrainBlend-AI/atomic-agents/network/members)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/BrainBlend-AI/atomic-agents)

## What is Atomic Agents?

The Atomic Agents framework is designed around the concept of atomicity to be an extremely lightweight and modular framework for building Agentic AI pipelines and applications without sacrificing developer experience and maintainability.

Think of it like building AI applications with LEGO blocks - each component (agent, tool, context provider) is:
- **Single-purpose**: Does one thing well
- **Reusable**: Can be used in multiple pipelines
- **Composable**: Easily combines with other components
- **Predictable**: Produces consistent, reliable outputs

Built on [Instructor](https://github.com/jxnl/instructor) and [Pydantic](https://docs.pydantic.dev/latest/), it enables you to create AI applications with the same software engineering principles you already know and love.

**NEW: Join our community on Discord at [discord.gg/J3W9b5AZJR](https://discord.gg/J3W9b5AZJR) and our official subreddit at [/r/AtomicAgents](https://www.reddit.com/r/AtomicAgents/)!**

## Table of Contents

- [Atomic Agents](#atomic-agents)
  - [What is Atomic Agents?](#what-is-atomic-agents)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Quick Example](#quick-example)
  - [Why Atomic Agents?](#why-atomic-agents)
  - [Core Concepts](#core-concepts)
    - [Anatomy of an Agent](#anatomy-of-an-agent)
    - [Context Providers](#context-providers)
    - [Chaining Schemas and Agents](#chaining-schemas-and-agents)
  - [Examples \& Documentation](#examples--documentation)
    - [Quickstart Examples](#quickstart-examples)
    - [Complete Examples](#complete-examples)
  - [🚀 Version 2.0 Released!](#-version-20-released)
    - [Key Changes in v2.0:](#key-changes-in-v20)
    - [⚠️ Upgrading from v1.x](#️-upgrading-from-v1x)
  - [Atomic Forge \& CLI](#atomic-forge--cli)
    - [Running the CLI](#running-the-cli)
  - [Project Structure](#project-structure)
  - [Provider \& Model Compatibility](#provider--model-compatibility)
  - [Contributing](#contributing)
  - [License](#license)
  - [Additional Resources](#additional-resources)
  - [Star History](#star-history)

## Getting Started

### Installation
To install Atomic Agents, you can use pip:

```bash
pip install atomic-agents
```

Make sure you also install the provider you want to use. Provider SDKs are available as instructor extras:

```bash
pip install instructor[groq]        # for Groq
pip install instructor[anthropic]   # for Anthropic
pip install instructor[google-genai] # for Gemini
```

OpenAI is included by default. For a full list of supported providers, see the [Instructor docs](https://python.useinstructor.com/integrations/).

This also installs the CLI *Atomic Assembler*, which can be used to download Tools (and soon also Agents and Pipelines).

### Quick Example

Here's a quick snippet demonstrating how easy it is to create a powerful agent with Atomic Agents:

```python
from pydantic import Field
from openai import OpenAI
import instructor
from atomic_agents import AtomicAgent, AgentConfig, BasicChatInputSchema, BaseIOSchema
from atomic_agents.context import SystemPromptGenerator, ChatHistory

# Define a custom output schema
class CustomOutputSchema(BaseIOSchema):
    """
    docstring for the custom output schema
    """
    chat_message: str = Field(..., description="The chat message from the agent.")
    suggested_questions: list[str] = Field(..., description="Suggested follow-up questions.")

# Set up the system prompt
system_prompt_generator = SystemPromptGenerator(
    background=["This assistant is knowledgeable, helpful, and suggests follow-up questions."],
    steps=[
        "Analyze the user's input to understand the context and intent.",
        "Formulate a relevant and informative response.",
        "Generate 3 suggested follow-up questions for the user."
    ],
    output_instructions=[
        "Provide clear and concise information in response to user queries.",
        "Conclude each response with 3 relevant suggested questions for the user."
    ]
)

# Initialize OpenAI client
client = instructor.from_openai(OpenAI())

# Initialize the agent
agent = AtomicAgent[BasicChatInputSchema, CustomOutputSchema](
    config=AgentConfig(
        client=client,
        model="gpt-5-mini",
        system_prompt_generator=system_prompt_generator,
        history=ChatHistory(),
    )
)

# Example usage
if __name__ == "__main__":
    user_input = "Tell me about atomic agents framework"
    response = agent.run(BasicChatInputSchema(chat_message=user_input))
    print(f"Agent: {response.chat_message}")
    print("Suggested questions:")
    for question in response.suggested_questions:
        print(f"- {question}")
```

## Why Atomic Agents?
While existing frameworks for agentic AI focus on building autonomous multi-agent systems, they often lack the control and predictability required for real-world applications. Businesses need AI systems that produce consistent, reliable outputs aligned with their brand and objectives.

Atomic Agents addresses this need by providing:

- **Modularity:** Build AI applications by combining small, reusable components.
- **Predictability:** Define clear input and output schemas to ensure consistent behavior.
- **Extensibility:** Easily swap out components or integrate new ones without disrupting the entire system.
- **Control:** Fine-tune each part of the system individually, from system prompts to tool integrations.

All logic and control flows are written in Python, enabling developers to apply familiar best practices and workflows from traditional software development without compromising flexibility or clarity.

## Core Concepts

### Anatomy of an Agent
In Atomic Agents, an agent is composed of several key components:

- **System Prompt:** Defines the agent's behavior and purpose.
- **Input Schema:** Specifies the structure and validation rules for the agent's input.
- **Output Schema:** Specifies the structure and validation rules for the agent's output.
- **History:** Stores conversation history or other relevant data.
- **Context Providers:** Inject dynamic context into the agent's system prompt at runtime.

Here's a high-level architecture diagram:
<!-- ![alt text](./.assets/architecture_highlevel_overview.png) -->
<img src="./.assets/architecture_highlevel_overview.png" alt="High-level architecture overview of Atomic Agents" width="600"/>
<img src="./.assets/what_is_sent_in_prompt.png" alt="Diagram showing what is sent to the LLM in the prompt" width="600"/>

### Context Providers

Atomic Agents allows you to enhance your agents with dynamic context using **Context Providers**. Context Providers enable you to inject additional information into the agent's system prompt at runtime, making your agents more flexible and context-aware.

To use a Context Provider, create a class that inherits from `BaseDynamicContextProvider` and implements the `get_info()` method, which returns the context string to be added to the system prompt.

Here's a simple example:

```python
from atomic_agents.context import BaseDynamicContextProvider

class SearchResultsProvider(BaseDynamicContextProvider):
    def __init__(self, title: str, search_results: List[str]):
        super().__init__(title=title)
        self.search_results = search_results

    def get_info(self) -> str:
        return "\n".join(self.search_results)
```

You can then register your Context Provider with the agent:

```python
# Initialize your context provider with dynamic data
search_results_provider = SearchResultsProvider(
    title="Search Results",
    search_results=["Result 1", "Result 2", "Result 3"]
)

# Register the context provider with the agent
agent.register_context_provider("search_results", search_results_provider)
```

This allows your agent to include the search results (or any other context) in its system prompt, enhancing its responses based on the latest information.

### Chaining Schemas and Agents

Atomic Agents makes it easy to chain agents and tools together by aligning their input and output schemas. This design allows you to swap out components effortlessly, promoting modularity and reusability in your AI applications.

Suppose you have an agent that generates search queries and you want to use these queries with different search tools. By aligning the agent's output schema with the input schema of the search tool, you can easily chain them together or switch between different search providers.

Here's how you can achieve this:

```python
import instructor
import openai
from pydantic import Field
from atomic_agents import BaseIOSchema, AtomicAgent, AgentConfig
from atomic_agents.context import SystemPromptGenerator

# Import the search tool you want to use
from web_search_agent.tools.searxng_search import SearXNGSearchTool

# Define the input schema for the query agent
class QueryAgentInputSchema(BaseIOSchema):
    """Input schema for the QueryAgent."""
    instruction: str = Field(..., description="Instruction to generate search queries for.")
    num_queries: int = Field(..., description="Number of queries to generate.")

# Initialize the query agent
query_agent = AtomicAgent[QueryAgentInputSchema, SearXNGSearchTool.input_schema](
    config=AgentConfig(
        client=instructor.from_openai(openai.OpenAI()),
        model="gpt-5-mini",
        system_prompt_generator=SystemPromptGenerator(
            background=[
                "You are an intelligent query generation expert.",
                "Your task is to generate a specified number of diverse and highly relevant queries based on a given instruction."
            ],
            steps=[
                "Receive the instruction and the number of queries to generate.",
                "Generate the queries in JSON format."
            ],
            output_instructions=[
                "Ensure each query is unique and relevant.",
                "Provide the queries in the expected schema."
            ],
        ),
    )
)
```

In this example:

- **Modularity**: By setting the `output_schema` of the `query_agent` to match the `input_schema` of `SearXNGSearchTool`, you can directly use the output of the agent as input to the tool.
- **Swapability**: If you decide to switch to a different search provider, you can import a different search tool and update the `output_schema` accordingly.

For instance, to switch to another search service:

```python
# Import a different search tool
from web_search_agent.tools.another_search import AnotherSearchTool

# Update the output schema
query_agent.config.output_schema = AnotherSearchTool.input_schema
```

This design pattern simplifies the process of chaining agents and tools, making your AI applications more adaptable and easier to maintain.

## Examples & Documentation

[![Read the Docs](https://img.shields.io/badge/docs-read%20the%20docs-blue?logo=readthedocs&style=for-the-badge)](https://brainblend-ai.github.io/atomic-agents/)

[Visit the Documentation Site »](https://brainblend-ai.github.io/atomic-agents/)

### Quickstart Examples

A complete list of examples can be found in the [examples](./atomic-examples/) directory. We strive to thoroughly document each example, but if something is unclear, please don't hesitate to open an issue or pull request to improve the documentation.

For full, runnable examples, please refer to the following files in the `atomic-examples/quickstart/quickstart/` directory:

- [Basic Chatbot](/atomic-examples/quickstart/quickstart/1_0_basic_chatbot.py) - A minimal chatbot example to get you started.
- [Custom Chatbot](/atomic-examples/quickstart/quickstart/2_basic_custom_chatbot.py) - A more advanced example with a custom system prompt.
- [Custom Chatbot with Schema](/atomic-examples/quickstart/quickstart/3_0_basic_custom_chatbot_with_custom_schema.py) - An advanced example featuring a custom output schema.
- [Multi-Provider Chatbot](/atomic-examples/quickstart/quickstart/4_basic_chatbot_different_providers.py) - Demonstrates how to use different providers such as Ollama or Groq.

### Complete Examples

In addition to the quickstart examples, we have more complex examples demonstrating the power of Atomic Agents:

- [Hooks System](/atomic-examples/hooks-example/README.md): Comprehensive demonstration of the AtomicAgent hook system for monitoring, error handling, and performance metrics with intelligent retry mechanisms.
- [Basic Multimodal](/atomic-examples/basic-multimodal/README.md): Demonstrates how to analyze images with text, focusing on extracting structured information from nutrition labels using GPT-4 Vision capabilities.
- [Deep Research](/atomic-examples/deep-research/README.md): An advanced example showing how to perform deep research tasks.
- [Orchestration Agent](/atomic-examples/orchestration-agent/README.md): Shows how to create an Orchestrator Agent that intelligently decides between using different tools (search or calculator) based on user input.
- [RAG Chatbot](/atomic-examples/rag-chatbot/README.md): A chatbot implementation using Retrieval-Augmented Generation (RAG) to provide context-aware responses.
- [Web Search Agent](/atomic-examples/web-search-agent/README.md): An intelligent agent that performs web searches and answers questions based on the results.
- [YouTube Summarizer](/atomic-examples/youtube-summarizer/README.md): An agent that extracts and summarizes knowledge from YouTube videos.
- [YouTube to Recipe](/atomic-examples/youtube-to-recipe/README.md): An example that extracts structured recipe information from cooking videos, demonstrating complex information extraction and structuring.

For a complete list of examples, see the [examples directory](/atomic-examples/).

## 🚀 Version 2.0 Released!

[![What's New in V2.0](./.assets/whats-new-v2-thumbnail.png)](https://www.youtube.com/watch?v=ro-pM2bPmr0)

[**Watch: What's New in Atomic Agents V2.0**](https://www.youtube.com/watch?v=ro-pM2bPmr0)

**Atomic Agents v2.0 is here with major improvements!** This release includes breaking changes that significantly improve the developer experience:

### Key Changes in v2.0:
- **Cleaner imports**: Eliminated `.lib` from import paths
- **Renamed classes**: `BaseAgent` → `AtomicAgent`, `BaseAgentConfig` → `AgentConfig`, and more
- **Better type safety**: Generic type parameters for tools and agents
- **Enhanced streaming**: New `run_stream()` and `run_async_stream()` methods
- **Improved organization**: Better module structure with `context`, `connectors`, and more

### ⚠️ Upgrading from v1.x
If you're upgrading from v1.x, please read our comprehensive [**Upgrade Guide**](UPGRADE_DOC.md) for detailed migration instructions.

## Atomic Forge & CLI

Atomic Forge is a collection of tools that can be used with Atomic Agents to extend its functionality. Current tools include:

- arXiv Search
- BoCha Search
- Calculator
- DateTime
- Fía Signals
- Hacker News Search
- PDF Reader
- SearXNG Search
- Tavily Search
- Webpage Scraper
- Weather
- Wikipedia Search
- YouTube Transcript Scraper

For more information on using and creating tools, see the [Atomic Forge README](/atomic-forge/README.md).

### Running the CLI

To run the CLI, simply run the following command:

```bash
atomic
```

Or if you're running from a cloned repository with uv:

```bash
uv run atomic
```

After running this command, you will be presented with a menu allowing you to download tools.

Each tool's has its own:

- Input schema
- Output schema
- Usage example
- Dependencies
- Installation instructions

![Atomic CLI tool example](./.assets/atomic-cli-tool-menu.png)

The `atomic-assembler` CLI gives you complete control over your tools, avoiding the clutter of unnecessary dependencies. It makes modifying tools straightforward additionally, each tool comes with its own set of tests for reliability.

**But you're not limited to the CLI!** If you prefer, you can directly access the tool folders and manage them manually by simply copying and pasting as needed.

![Atomic CLI menu](./.assets/atomic-cli.png)

## Project Structure

Atomic Agents uses a monorepo structure with the following main components:

1. `atomic-agents/`: The core Atomic Agents library
2. `atomic-assembler/`: The CLI tool for managing Atomic Agents components
3. `atomic-examples/`: Example projects showcasing Atomic Agents usage
4. `atomic-forge/`: A collection of tools that can be used with Atomic Agents

For local development, you can install from the repository:

```bash
git clone https://github.com/BrainBlend-AI/atomic-agents.git
cd atomic-agents
uv sync
```

To install all workspace packages (examples and tools):

```bash
uv sync --all-packages
```

## Provider & Model Compatibility

Atomic Agents depends on the [Instructor](https://github.com/jxnl/instructor) package. This means that in all examples where OpenAI is used, any other API supported by Instructor can also be used—such as Ollama, Groq, Mistral, Cohere, Anthropic, Gemini, [MiniMax](https://www.minimax.io/), and more. For a complete list, please refer to the Instructor documentation on its [GitHub page](https://github.com/jxnl/instructor).

## Contributing

We welcome contributions! Please see the [Contributing Guide](/docs/contributing.md) for detailed information on how to contribute to Atomic Agents. Here are some quick steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature-branch`)
3. Make your changes
4. Run tests (`uv run pytest --cov=atomic_agents atomic-agents`)
5. Format your code (`uv run black atomic-agents atomic-assembler atomic-examples atomic-forge`)
6. Lint your code (`uv run flake8 --extend-exclude=.venv atomic-agents atomic-assembler atomic-examples atomic-forge`)
7. Commit your changes (`git commit -m 'Add some feature'`)
8. Push to the branch (`git push origin feature-branch`)
9. Open a pull request

For full development setup and guidelines, please refer to the [Developer Guide](/guides/DEV_GUIDE.md).

## License

This project is licensed under the MIT License—see the [LICENSE](LICENSE) file for details.

## Additional Resources

If you want to learn more about the motivation and philosophy behind Atomic Agents, [I suggest reading this Medium article (no account needed)](https://ai.gopubby.com/want-to-build-ai-agents-c83ab4535411?sk=b9429f7c57dbd3bda59f41154b65af35).

**Video Resources:**
- [Watch the Overview Video](https://www.youtube.com/watch?v=Sp30YsjGUW0) - Learn about the framework's philosophy and design principles
- [Watch the Quickstart Video](https://www.youtube.com/watch?v=CyZxRU0ax3Q) - Get started with code examples


## FAQ

### What is Atomic Agents?

Atomic Agents is an extremely lightweight and modular framework for building Agentic AI pipelines and applications. Think of it like building AI applications with LEGO blocks - each component (agent, tool, context provider) is single-purpose, reusable, composable, and predictable.

### How does Atomic Agents compare to LangChain or CrewAI?

| Framework | Philosophy | Key Strength |
|-----------|------------|--------------|
| **Atomic Agents** | Atomicity & modularity | LEGO-like components, single-purpose |
| **LangChain** | Chains & sequences | Pre-built chains for common tasks |
| **CrewAI** | Role-playing agents | Multi-agent collaboration |

Atomic Agents focuses on **control and predictability** - each component does one thing well and produces consistent outputs, making it ideal for production applications where reliability matters.

### What are the core components of Atomic Agents?

**Agent Anatomy:**
- **System Prompt**: Defines the agent's behavior and purpose
- **Input Schema**: Specifies structure and validation for inputs (Pydantic)
- **Output Schema**: Specifies structure and validation for outputs (Pydantic)
- **History**: Stores conversation history
- **Context Providers**: Inject dynamic context at runtime

**Context Providers:** Enhance agents with dynamic information (e.g., search results, user data) injected into system prompts at runtime.

### What LLM providers are supported?

Atomic Agents uses [Instructor](https://github.com/jxnl/instructor) for LLM integration:

**Native Support:**
- OpenAI (default)
- Anthropic
- Groq
- Google Gemini
- Mistral
- Cohere
- MiniMax

**Via Instructor extras:**
```bash
pip install instructor[groq]        # for Groq
pip install instructor[anthropic]   # for Anthropic
pip install instructor[google-genai] # for Gemini
```

### How do I get started?

```bash
# Install Atomic Agents
pip install atomic-agents

# Install your preferred provider
pip install instructor[groq]  # example for Groq

# Run the CLI to download tools
atomic
```

### What is Atomic Forge?

Atomic Forge is a collection of pre-built tools that extend Atomic Agents functionality:

- arXiv Search
- BoCha Search
- Calculator
- DateTime
- Fía Signals
- Hacker News Search
- PDF Reader
- SearXNG Search
- Tavily Search
- Webpage Scraper
- Weather
- Wikipedia Search
- YouTube Transcript Scraper

Each tool has its own input schema, output schema, usage example, dependencies, and installation instructions.

### How do I chain agents and tools together?

By aligning input/output schemas between agents and tools, you can chain them directly:

```python
# Agent output matches tool input schema
query_agent = AtomicAgent[QueryInputSchema, SearchTool.input_schema](
    config=AgentConfig(...)
)

# Output can be directly used as tool input
search_results = SearchTool.run(query_agent.run(input_data))
```

This promotes modularity and makes swapping components effortless.

### What changed in Version 2.0?

**Key improvements:**
- Cleaner imports: eliminated `.lib` from paths
- Renamed classes: `BaseAgent` → `AtomicAgent`, `BaseAgentConfig` → `AgentConfig`
- Better type safety: Generic type parameters for tools and agents
- Enhanced streaming: New `run_stream()` and `run_async_stream()` methods
- Improved organization: Better module structure (`context`, `connectors`)

**Upgrading from v1.x:** See the [Upgrade Guide](UPGRADE_DOC.md) for migration instructions.

### What are the complete examples available?

| Example | Description |
|---------|-------------|
| **Hooks System** | Monitoring, error handling, performance metrics |
| **Basic Multimodal** | Image analysis with GPT-4 Vision |
| **Deep Research** | Advanced research task execution |
| **Orchestration Agent** | Tool selection (search vs calculator) |
| **RAG Chatbot** | Retrieval-Augmented Generation |
| **Web Search Agent** | Web search and question answering |
| **YouTube Summarizer** | Video knowledge extraction |
| **YouTube to Recipe** | Cooking video to structured recipe |

### What is the license?

MIT License - see [LICENSE](LICENSE) file.

### Where can I get help?

- **Documentation**: [brainblend-ai.github.io/atomic-agents](https://brainblend-ai.github.io/atomic-agents/)
- **Discord**: [discord.gg/J3W9b5AZJR](https://discord.gg/J3W9b5AZJR)
- **Subreddit**: [/r/AtomicAgents](https://www.reddit.com/r/AtomicAgents/)
- **GitHub Issues**: For bug reports and feature requests
- **Video Resources**: Overview & Quickstart videos available

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=BrainBlend-AI/atomic-agents&type=Date)](https://star-history.com/#BrainBlend-AI/atomic-agents&Date)

