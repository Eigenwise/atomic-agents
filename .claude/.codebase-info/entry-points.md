# Entry Points

*Last Updated: 2026-10-21*

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
- **Public multimodal type:** import `VideoURL` from `atomic_agents` for video inputs; `ChatHistory`
  serializes it as an OpenAI-compatible `video_url` content part before the provider call.

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
| `phoenix-tracing` | Arize Phoenix traces: planner agent -> calculator tool -> answer agent under one CHAIN span, via `register(auto_instrument=True)`; depends on `atomic-agents[phoenix]` and sets `ATOMIC_AGENTS_TRACE_LLM_SPANS=false` because the OpenAI instrumentor owns LLM spans there |
| `basic-multimodal` / `nested-multimodal` | images / PDFs in (nested) schemas |
| `basic-pdf-analysis` | PDF analysis with a multimodal model |
| `youtube-summarizer` / `youtube-to-recipe` | transcript → summary / structured recipe |
| `dspy-integration` | combining DSPy with Atomic Agents |

## 5. OpenTelemetry discovery

`atomic_agents.observability:AtomicAgentsInstrumentor` is registered as `atomic_agents` in both
`openinference_instrumentor` (Phoenix) and `opentelemetry_instrumentor` (the standard
`opentelemetry-instrument` launcher). Both use the same installation state. The launcher needs
its own SDK/exporter setup; only `opentelemetry-api` is required for the instrumentor, and
`opentelemetry-bootstrap` does not install Atomic Agents. Set `OTEL_PYTHON_DISABLED_INSTRUMENTATIONS=atomic_agents`
to disable launcher discovery; `ATOMIC_AGENTS_TRACE_*` environment variables configure hiding, size cap and
LLM attempt spans on both automatic paths. See `docs/guides/observability.md`.

## Representative Flow

`agent.run(input)` → `SystemPromptGenerator.generate_prompt()` (+ context providers) → serialize
`ChatHistory` → `instructor` `client.chat.completions.create(response_model=Out)` → validated `Out`
appended to history → returned. See `architecture.md`.
