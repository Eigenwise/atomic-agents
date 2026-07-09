# Entry Points

*Last Updated: 2026-07-05*

## 1. Library API (the primary entry point)

```python
from atomic_agents import AtomicAgent, AgentConfig, BasicChatInputSchema, BasicChatOutputSchema
import instructor
from openai import OpenAI

agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
    AgentConfig(client=instructor.from_openai(OpenAI()), model="gpt-5-mini")
)
result = agent.run(BasicChatInputSchema(chat_message="Hello"))
```

- **Module:** `atomic-agents/atomic_agents/agents/atomic_agent.py`
- **Run methods:** `run`, `run_stream`, `run_async`, `run_async_stream`.
- Custom agents define their own `BaseIOSchema` subclasses and pass them as `AtomicAgent[In, Out]`.

## 2. CLI — `atomic`

| Entry | Type | Purpose | File |
|-------|------|---------|------|
| `atomic` | Textual TUI | Browse & install forge tools into a project | `atomic-assembler/atomic_assembler/main.py:main` |

Declared in `pyproject.toml` (`[project.scripts]  atomic = "atomic_assembler.main:main"`). Flags:
`--enable-logging`, `--version`.

## 3. MCP

MCP server tools/resources/prompts become agent components via `atomic_agents.connectors.mcp`
(`fetch_mcp_tools(...)`, `MCPFactory`). See `communication.md`.

## 4. Examples (each runnable on its own, under `atomic-examples/<name>/`)

| Example | Demonstrates |
|---------|--------------|
| `quickstart` | basic chatbot, custom schema, multi-provider (4 scripts) |
| `rag-chatbot` | retrieval-augmented generation |
| `web-search-agent` | query generation + web search + answer |
| `deep-research` | a pipeline of single-purpose agents |
| `orchestration-agent` | choosing between search / calculator tools |
| `mcp-agent` | MCP client/server, multiple transports |
| `progressive-disclosure` | efficient MCP tool loading (3 servers, 24 tools) |
| `fastapi-memory` | multi-user / multi-session memory behind a FastAPI service |
| `persistent-memory` | custom `BaseChatHistory` backend: cross-session recall via stdlib `sqlite3` (no extra deps) |
| `hooks-example` | monitoring, error handling, retry via hooks |
| `basic-multimodal` / `nested-multimodal` | images / PDFs in (nested) schemas |
| `basic-pdf-analysis` | PDF analysis with a multimodal model |
| `youtube-summarizer` / `youtube-to-recipe` | transcript → summary / structured recipe |
| `dspy-integration` | combining DSPy with Atomic Agents |

## Representative Flow

`agent.run(input)` → `SystemPromptGenerator.generate_prompt()` (+ context providers) → serialize
`ChatHistory` → `instructor` `client.chat.completions.create(response_model=Out)` → validated `Out`
appended to history → returned. See `architecture.md`.
