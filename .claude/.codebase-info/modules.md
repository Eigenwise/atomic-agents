# Key Modules

*Last Updated: 2026-10-21*

## Core framework — `atomic-agents/atomic_agents/`

### agents
- **Location:** `atomic-agents/atomic_agents/agents/atomic_agent.py`
- **Purpose:** The `AtomicAgent[InputSchema, OutputSchema]` class and `AgentConfig`. Orchestrates the
  run lifecycle (sync / stream / async), context-provider registration, hook registration, and
  context-window trimming.
- **Exposes:** `AtomicAgent`, `AgentConfig`, `BasicChatInputSchema`, `BasicChatOutputSchema`.
- **Depends on:** `base/`, `context/`, `utils/`, `instructor`.

### base
- **Location:** `atomic-agents/atomic_agents/base/`
- **Purpose:** The typed contracts everything else implements.
- **Key files:** `base_io_schema.py` (`BaseIOSchema` — Pydantic base; non-empty docstring enforced and
  used as the schema description), `base_tool.py` (`BaseTool[In, Out]`, `BaseToolConfig`),
  `base_resource.py` (`BaseResource`), `base_prompt.py` (`BasePrompt`), and `multimodal.py`
  (`VideoURL` — OpenAI-compatible `video_url` content-part model for video inputs).
- **Exposes via package root:** `BaseIOSchema`, `BaseTool`, `BaseToolConfig`, `VideoURL`.

### context
- **Location:** `atomic-agents/atomic_agents/context/`
- **Purpose:** System-prompt assembly and conversation memory.
- **Key files:** `system_prompt_generator.py` (`SystemPromptGenerator`, `BaseDynamicContextProvider`),
  `base_chat_history.py` (`BaseChatHistory` — interface-only ABC declaring the memory contract
  `AtomicAgent` depends on; the pluggable seam for custom/persistent backends),
  `chat_history.py` (`ChatHistory`, `Message` — the built-in in-memory implementation of
  `BaseChatHistory`: multimodal Image/Audio/PDF plus `VideoURL`, turn grouping, `dump()`/`load()`
  serialization).
- **Note:** `AgentConfig.history` is typed to `BaseChatHistory`, so any conforming backend drops in.

### connectors/mcp
- **Location:** `atomic-agents/atomic_agents/connectors/mcp/`
- **Purpose:** Model Context Protocol integration — expose MCP server tools/resources/prompts as
  Atomic Agents components.
- **Exposes:** `MCPFactory`, `MCPDefinitionService`, `SchemaTransformer`,
  `fetch_mcp_tools` / `fetch_mcp_resources` / `fetch_mcp_prompts` (sync + async).

### utils
- **Location:** `atomic-agents/atomic_agents/utils/`
- **Key files:** `token_counter.py` (`TokenCounter` via LiteLLM; `get_context_token_count`),
  `format_tool_message.py`.

### observability (opt-in)
- **Location:** `atomic-agents/atomic_agents/observability/`
- **Purpose:** plain OpenTelemetry tracing for any backend (OTLP collector, Grafana Tempo, Sentry, Datadog,
  Honeycomb, PostHog, Arize Phoenix, Langfuse). Only `opentelemetry-api` is required; the OpenTelemetry GenAI
  semantic-convention attributes are primary and the OpenInference family is an optional layer, auto-on when the
  `openinference-instrumentation` package is importable (`phoenix` extra).
- **Modules:** `__init__.py` (loader, `AtomicAgentsInstrumentor`, stdlib-only imports), `tracing.py` (thin facade),
  `tracing_config.py` (`TracingConfig`, `PayloadVisibility`, env parsing), `span_payloads.py` (serialize, hide before
  serialize, truncate with `<key>.original_length`, bounded/redacted error text), `span_attributes.py` (key constants,
  Instructor-to-semconv provider map, None filtering, best-effort attribute collection, OpenInference context seam,
  `using_conversation`), `agent_spans.py` (four `AtomicAgent.run*` wrappers, stream cursors, run state),
  `tool_spans.py` (`BaseTool` wrappers via `__subclasses__` walk + wrapped `__init_subclass__`, re-entrancy guard),
  `installation.py` (install / uninstall / replace-in-place), `llm_attempt_spans.py` (Instructor hook handlers).
- **Contract:** `instrument(tracer_provider=, config=TracingConfig | PayloadVisibility | None, **options)` /
  `uninstrument()` / `is_instrumented_by_opentelemetry`. Same arguments: no-op; different provider or config:
  replaced in place with a WARNING, `None` keeps the installed value; in-flight spans finish on their original
  provider. Hooks are registered once per Instructor client under a lock and removed on uninstall;
  `suppress_instrumentation` is honored; hooks never raise into user code.
- **Spans:** `invoke_agent {AgentConfig.name or class}` (`gen_ai.operation.name=invoke_agent`, `gen_ai.agent.name`,
  `gen_ai.request.model`, `gen_ai.provider.name`, `gen_ai.conversation.id`, `atomic_agents.agent.class`, `.input`,
  `.output`, model/mode/schema/history/provider/prompt-hash keys, usage totals and `atomic_agents.llm.attempts` at end);
  `execute_tool {tool.tool_name}` (`gen_ai.tool.*`, `atomic_agents.tool.class`); one `chat {model}` CLIENT span per
  Instructor attempt from `completion:kwargs`/`completion:response`/`completion:error` with per-attempt usage
  (cached/reasoning tokens for both chat-completion and Responses shapes), `parse:error` as a `validation_error`
  event; `error.type` on failures; cancellation = `atomic_agents.cancelled`, never ERROR. Spans always attach to the
  current context. `llm_spans=False` keeps totals and drops chat spans.
- **Config:** `TracingConfig(hide_inputs, hide_outputs, max_attribute_length=4096, openinference_attributes=None,
  llm_spans=True)`; env `ATOMIC_AGENTS_TRACE_HIDE_INPUTS/_OUTPUTS` (fallback `OPENINFERENCE_HIDE_*`),
  `ATOMIC_AGENTS_TRACE_MAX_ATTRIBUTE_LENGTH` (tolerant, 4096 fallback), `ATOMIC_AGENTS_TRACE_LLM_SPANS`. Under
  `hide_outputs` every exception becomes `<Type>: <N> attempts (__REDACTED__)`. Nothing in `atomic_agents/__init__.py`
  imports the package.
- **Tests:** `atomic-agents/tests/observability/` (in-memory exporter; `instructor_transport.py` builds real Instructor
  clients over `httpx.MockTransport`; `test_instructor_modes.py` covers TOOLS/JSON/JSON_SCHEMA/RESPONSES_TOOLS).

## Subprojects

### atomic-assembler (CLI)
- **Location:** `atomic-assembler/atomic_assembler/`
- **Purpose:** Textual TUI to browse and install forge tools into a user project.
- **Key files:** `main.py` (`main()`; argparse `--enable-logging`, `--version`), `app.py`
  (`AtomicAssembler(App)`), `screens/` (`main_menu`, `atomic_tool_explorer`, `file_explorer`,
  `tool_info_screen`), `widgets/`, `utils.py` (`GithubRepoCloner`, `AtomicToolManager`),
  `constants.py` (GitHub URL, `TOOLS_SUBFOLDER`).

### atomic-forge (tools)
- **Location:** `atomic-forge/tools/`
- **Purpose:** Standalone tools, each an independent mini-project following the `BaseTool` pattern,
  copied into user projects by the assembler (build files such as `pyproject.toml` / `requirements.txt`
  / `uv.lock` are skipped on copy).
- **Recent additions:** `firecrawl_search` provides web/news search; `fxmacrodata` provides economic releases, FX and COT data. Both keep their own dependencies and setup.
- **YouTube compatibility:** `youtube_transcript_scraper` uses the instance-based transcript API and converts SDK transcript objects to raw entries.
- **Authoring guide:** `atomic-forge/guides/tool_structure.md`.

### atomic-examples
- **Location:** `atomic-examples/`
- **Purpose:** 16 runnable reference apps. Catalog in `entry-points.md`.
