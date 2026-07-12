# Atomic Agents Monorepo

## What is this Repository?

**atomic-monorepo** is the development repository for **Atomic Agents**, a lightweight and modular Python framework for building Agentic AI applications. The framework is built around the principle of **atomicity** - creating single-purpose, reusable, and composable components for AI pipelines.

### Core Philosophy

Atomic Agents bridges the gap between flexibility and reliability in production AI applications by providing:

- **Predictable AI Behavior**: Controlled, schema-driven agent construction vs. autonomous but unpredictable multi-agent systems
- **Modular Development**: Build AI applications using familiar software engineering principles (LEGO-like composability)
- **Type Safety**: Consistent input/output contracts through Pydantic schemas
- **Developer Control**: Full visibility and control over AI behavior with no hidden abstractions

Built on top of [Instructor](https://github.com/jxnl/instructor) (for structured LLM outputs) and [Pydantic](https://docs.pydantic.dev/) (for data validation).

---

## Monorepo Structure

This repository contains four main packages/projects:

```
atomic-monorepo/
├── atomic-agents/          # Core framework library (main package)
├── atomic-assembler/       # CLI tool for managing components
├── atomic-examples/        # Example projects and use cases
├── atomic-forge/           # Collection of downloadable tools
├── docs/                   # Sphinx documentation
├── guides/                 # Development guides
├── .github/workflows/      # CI/CD pipelines
├── pyproject.toml          # Project configuration
└── README.md              # Main documentation
```

---

## Package Details

### 1. atomic-agents/ - Core Framework

**Published as:** `atomic-agents` on PyPI
**Current Version:** 2.2.2
**Purpose:** Main Python package containing all core framework components

**Key Components:**
- `agents/` - AtomicAgent class and agent configuration
- `base/` - Base abstractions (BaseIOSchema, BaseTool, BasePrompt, BaseResource)
- `context/` - Chat history and system prompt generation
- `connectors/` - External integrations (MCP support)
- `lib/` - Extended library components (components, factories, utils)
- `memory/` - Memory management systems
- `prompting/` - Prompt engineering utilities
- `services/` - Service integrations

**Installation:**
```bash
pip install atomic-agents
```

### 2. atomic-assembler/ - CLI Tool

**Command:** `atomic`
**Purpose:** Terminal UI application for browsing, downloading, and managing tools from atomic-forge
**Built With:** [Textual](https://textual.textualize.io/) (TUI framework)

**Features:**
- Interactive tool exploration
- Download tools directly into your project
- Manage tool dependencies

**Usage:**
```bash
atomic  # Launch interactive TUI
```

### 3. atomic-examples/ - Example Projects

**Purpose:** Complete, runnable example implementations demonstrating framework capabilities

**Available Examples:**
- `quickstart/` - Basic getting started examples
- `deep-research/` - Research agent implementations
- `web-search-agent/` - Web search integration
- `rag-chatbot/` - RAG (Retrieval Augmented Generation)
- `youtube-summarizer/` - Video transcript summarization
- `orchestration-agent/` - Multi-agent orchestration patterns
- `mcp-agent/` - Model Context Protocol integration
- `fastapi-memory/` - API with memory/state management
- And 5+ more examples...

Each example is self-contained and demonstrates specific patterns and capabilities.

### 4. atomic-forge/ - Tool Repository

**Not a Python package** - This is a downloadable tool collection that users integrate into their projects.

**Philosophy:** Tools are NOT bundled with the framework. Instead, users download individual tools for full control and customization.

**Available Tools:**
- `arxiv_search/` - Academic paper search via the public arXiv API
- `bocha_search/` - BoCha web search
- `calculator/` - Mathematical computation tool
- `datetime_tool/` - Timezone-aware now / parse / convert / shift / diff
- `fia_signals/` - Crypto market intelligence (regime, signals, yields, gas, trending, wallet risk)
- `hackernews_search/` - Hacker News search via the free Algolia API
- `pdf_reader/` - PDF text + metadata extraction (local file or URL, page-range support)
- `searxng_search/` - Privacy-focused search integration
- `tavily_search/` - Tavily API search tool
- `weather/` - Current conditions and daily/hourly forecast via Open-Meteo (no key)
- `webpage_scraper/` - Web scraping capabilities
- `wikipedia_search/` - Wikipedia search in any language edition (no key)
- `youtube_transcript_scraper/` - YouTube transcript extraction

**Benefits:**
- Only install what you need (reduces dependency bloat)
- Full control to modify tools for your specific use case
- Tools become part of your codebase (no black boxes)

---

## Core Architecture

### Agent Anatomy

Every Atomic Agent consists of:

1. **System Prompt** - Defines agent behavior and purpose
2. **Input Schema** (Pydantic model) - Validates and structures input
3. **Output Schema** (Pydantic model) - Ensures consistent output format
4. **Chat History** - Maintains conversation context
5. **Context Providers** - Injects dynamic runtime context
6. **Tools** (optional) - Function calling capabilities

### Schema-Driven Development

All data flows through typed Pydantic schemas:

```python
from atomic_agents.base import BaseIOSchema
from pydantic import Field

class CustomOutputSchema(BaseIOSchema):
    chat_message: str = Field(..., description="The response message")
    suggested_questions: list[str] = Field(..., description="Follow-up questions")
```

### Composability Pattern

Agents can be chained by connecting output schemas to input schemas:

```
Agent A (OutputSchemaA) → Agent B (InputSchemaB) → Agent C (OutputSchemaC)
```

This creates predictable data flow pipelines with type safety at each step.

### Context Providers

Dynamic context injection pattern allows runtime enhancement of prompts without modifying agent code:

```python
context_provider = CustomContextProvider()
agent = AtomicAgent(
    system_prompt=prompt,
    input_schema=InputSchema,
    output_schema=OutputSchema,
    context_providers=[context_provider]  # Inject runtime context
)
```

### Model Context Protocol (MCP)

Full support for MCP enables connection to external tools and resources through standardized interfaces:

- Connect to MCP servers
- Use remote tools as if they were local
- Access external resources seamlessly
- Standard protocol for tool integration

---

## Technologies & Frameworks

### Core Stack

- **Language:** Python 3.12+
- **AI/LLM:** Instructor (supports multiple providers)
- **Data Validation:** Pydantic v2
- **CLI Framework:** Textual (TUI), Rich (terminal formatting)
- **Testing:** pytest with coverage
- **Documentation:** Sphinx + MyST + ReadTheDocs theme
- **Build System:** uv (with hatchling)

### LLM Provider Compatibility

Through Instructor, Atomic Agents supports all major LLM providers:

- OpenAI
- Anthropic (Claude)
- Groq
- Mistral
- Cohere
- Google Gemini
- Ollama (local models)
- Any Instructor-compatible provider

### Execution Modes

- **Synchronous:** `agent.run(input_data)`
- **Asynchronous:** `await agent.run_async(input_data)`
- **Streaming:** `agent.run_stream(input_data)` and `agent.run_async_stream(input_data)`

---

## Development Setup

### Prerequisites

- Python ≥3.12, <4.0
- [uv](https://docs.astral.sh/uv/) (package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/eigenwise/atomic-agents.git
cd atomic-monorepo

# Install dependencies
uv sync

# Install all workspace packages (examples and tools)
uv sync --all-packages
```

### Development Commands

```bash
# Code formatting
uv run black atomic-agents atomic-assembler atomic-examples atomic-forge

# Linting
uv run flake8 --extend-exclude=.venv atomic-agents atomic-assembler atomic-examples atomic-forge

# Testing
uv run pytest --cov=atomic_agents atomic-agents

# Build documentation
cd docs && make html
```

### Code Quality Standards

- **Formatter:** Black (line length: 127)
- **Linter:** Flake8
- **Testing:** pytest with coverage tracking
- **Type Safety:** Pydantic runtime validation

### CI/CD

Two GitHub Actions workflows:

1. **code-quality.yml** - Runs Black, Flake8, and pytest on push/PR to main/v2.0
2. **docs.yml** - Builds and deploys Sphinx documentation

---

## Key Design Patterns

### 1. Single Responsibility Principle
Each component does one thing well. Agents, tools, and schemas are atomic units of functionality.

### 2. Type-Safe Contracts
Pydantic schemas enforce contracts at runtime, catching errors before they propagate.

### 3. Composition Over Inheritance
Build complex behaviors by composing simple agents rather than deep inheritance hierarchies.

### 4. Explicit Over Implicit
No hidden magic - full control over what your agent does and how it does it.

### 5. Separation of Concerns
- **Prompts** define behavior
- **Schemas** define data contracts
- **Tools** define capabilities
- **Context Providers** inject runtime information

---

## Version 2.0+ Features

Recent improvements include:

- Cleaner imports (removed `.lib` from import paths)
- Renamed classes for clarity (`BaseAgent` → `AtomicAgent`)
- Better type safety with generic type parameters
- Enhanced streaming capabilities
- Improved module organization
- Backward compatibility layer for v1.x migration

---

## Community & Resources

### Documentation
- **Main Docs:** https://eigenwise.github.io/atomic-agents/
- **Repository:** https://github.com/eigenwise/atomic-agents
- **PyPI:** https://pypi.org/project/atomic-agents/

### Community
- **Discord:** https://discord.gg/J3W9b5AZJR
- **Subreddit:** /r/AtomicAgents

### Learning Resources
- YouTube overview and quickstart videos
- Medium articles on philosophy and patterns
- Comprehensive examples in `atomic-examples/`
- Interactive tutorials in documentation

---

## Quick Start

```python
from atomic_agents import AtomicAgent
from atomic_agents.base import BaseIOSchema
from pydantic import Field

# Define schemas
class InputSchema(BaseIOSchema):
    query: str = Field(..., description="User's question")

class OutputSchema(BaseIOSchema):
    answer: str = Field(..., description="Agent's response")
    confidence: float = Field(..., description="Confidence score 0-1")

# Create agent
agent = AtomicAgent(
    system_prompt="You are a helpful assistant. Provide accurate answers.",
    input_schema=InputSchema,
    output_schema=OutputSchema,
    model="gpt-5-mini",
)

# Run agent
result = agent.run(InputSchema(query="What is atomic computing?"))
print(result.answer)
print(f"Confidence: {result.confidence}")
```

---

## Philosophy Summary

Atomic Agents emphasizes:

1. **Developer Experience** - Clean APIs, type safety, familiar patterns
2. **Modularity** - LEGO-like composability of components
3. **Control** - Full visibility and control over AI behavior
4. **Production Readiness** - Schema validation, testing, CI/CD
5. **Community** - Active development with examples, docs, and support

This framework is designed for developers who want to build **reliable, maintainable AI applications** using standard software engineering practices, rather than dealing with unpredictable autonomous agent systems.

---

<!-- skill-forge:begin -->
## Skill system (self-improving)

This project captures repeatable agent workflows as small, single-purpose,
verified skills under `.claude/skills/`. Build and maintain them with the
**skill-forge** skill — never do a repeatable workflow ad hoc.

**Proactively (notice → propose → ask):** while working, if you notice a
multi-step workflow being repeated with no skill, a SKILL.md growing past ~500
lines or sprouting a second capability, a skill whose instructions have drifted
from reality, or two skills with overlapping triggers — briefly propose forging or
refactoring, then wait for my go-ahead. Never forge or restructure silently or
mid-task without consent.

**When building/changing a skill:** keep it atomic (one capability; if it needs an
"and", it's two skills), compose the shared surfaces below instead of
reimplementing, always include a runnable/documented proof of its success
criteria, and cross-reference related skills both ways.

### Project profile (for skill-forge auto-detect)
- Stack / package manager: Python ≥3.12, uv workspace (hatchling build); Pydantic v2 + Instructor for structured LLM output
- Verify command(s): `uv run black atomic-agents atomic-assembler atomic-examples atomic-forge` (line-length 127); `uv run flake8 --extend-exclude=.venv atomic-agents atomic-assembler atomic-examples atomic-forge`; `uv run pytest --cov=atomic_agents atomic-agents`
- Shared surfaces to compose: core framework `atomic-agents/atomic_agents/` (AtomicAgent, BaseIOSchema, BaseTool, context providers, MCP connectors); CLI `atomic` (atomic-assembler); downloadable tools `atomic-forge/tools/*`; runnable examples `atomic-examples/*`
- Skill convention: frontmatter `name` + `description` (optional `allowed-tools`); one skill per dir at `.claude/skills/<name>/SKILL.md`
- Current skills: `release` (project-local); `skill-forge` + `skill-forge-setup` (installed via plugin `claude-dev-kitchen`)
<!-- skill-forge:end -->
