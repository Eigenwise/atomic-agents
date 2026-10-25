# Communication & Integrations

*Last Updated: 2026-10-24*

The framework exposes no HTTP API of its own; "communication" here means how it talks to LLM
providers and external tools.

## LLM providers (via Instructor)
- All model calls go through an **Instructor-wrapped client** the caller supplies
  (`instructor.from_openai(...)`, `from_anthropic(...)`, etc.). The agent calls
  `client.chat.completions.create(response_model=OutputSchema)` (and `create_partial` for streaming).
- Provider-agnostic: OpenAI, Anthropic, Google Gemini, MiniMax, and 100+ models via LiteLLM. Mode is
  configurable (`Mode.TOOLS` is the default).
- Multimodal content uses Instructor's Image/Audio/PDF types plus the framework's `VideoURL`;
  `ChatHistory.get_history()` converts `VideoURL` to an OpenAI-compatible `video_url` dict, which
  Instructor forwards unchanged to providers that support video inputs.
- Provider quirks live in `AgentConfig`: `system_role`, `assistant_role` (use `"model"` for Gemini),
  `tool_result_role` (auto-detected).
- Code: `atomic-agents/atomic_agents/agents/atomic_agent.py`.

## Token accounting (via LiteLLM)
- `atomic-agents/atomic_agents/utils/token_counter.py` uses LiteLLM's `token_counter` for
  provider-agnostic counts, which drive the context-window trimming in `AtomicAgent`. Video content
  parts are represented by a `[video content]` text placeholder because LiteLLM cannot count them.

## MCP — Model Context Protocol
- `atomic-agents/atomic_agents/connectors/mcp/` turns MCP server capabilities into agent
  tools/resources/prompts: `MCPFactory`, `MCPDefinitionService`, `SchemaTransformer`, and
  `fetch_mcp_tools` / `fetch_mcp_resources` / `fetch_mcp_prompts` (sync + async). See the
  `mcp-agent` and `progressive-disclosure` examples.
- Discovery follows tool, resource, template and prompt cursors. Template continuation errors propagate.
- Schema references decode escaped definition names and support non-object definitions.
- Typed SDK tool results use structured content only when it is non-None, then use the existing validated JSON-text fallback. Present empty structured data retains precedence. Mapping adapters keep their separate validation contract.
- SDK `CallToolResult` and strictly validated mapping envelopes with `isError=True` raise a `RuntimeError` before typed or generic output parsing. The message uses `TextContent` only (meaningful fallback when no text); `structuredContent`, `_meta` and binary blocks never enter messages or logs. Raw application dictionaries that merely contain `isError`/`name` keys (even with a string `content`) stay ordinary outputs.

## Hooks / observability
- Built on Instructor's hook system: `register_hook(event, handler)`, `unregister_hook`,
  `clear_hooks`. Events include `parse:error`, `completion:kwargs`, `completion:response`,
  `token:counted`. See the `hooks-example` example and `docs/guides/hooks.md`.
- `disable_hooks()` detaches this agent's registered Instructor callbacks. Registrations made while disabled stay
  paused; `enable_hooks()` restores the remaining callbacks. Direct client callbacks and tracing hooks remain independent.
- OpenTelemetry tracing (`atomic_agents.observability`, `observability` extra = `opentelemetry-api` only): an AGENT
  span per `AtomicAgent` run, a TOOL span per `BaseTool` run and one CLIENT `chat` span per Instructor attempt
  (driven by Instructor hooks registered once per client), all with OpenTelemetry GenAI attributes; OpenInference
  attributes are added only when its package is importable (`phoenix` extra). Activated by
  `phoenix.otel.register(auto_instrument=True)` through `openinference_instrumentor`, by the `opentelemetry-instrument`
  launcher through `opentelemetry_instrumentor`, or by `AtomicAgentsInstrumentor().instrument(tracer_provider=...,
  config=TracingConfig(...))`. A later install with different explicit values replaces the provider/config in place
  with a WARNING; `None` keeps the installed value. Missing `opentelemetry-api` is DEBUG-skipped when the launcher
  passes `skip_dep_check=True`; manual setup raises `ModuleNotFoundError` with the install hint. Application hooks on
  the same client keep running; `AtomicAgent.clear_hooks()` removes only its own. Set `llm_spans=False` (or
  `ATOMIC_AGENTS_TRACE_LLM_SPANS=false`) when a provider instrumentor also emits LLM spans. See
  `docs/guides/observability.md` and the `phoenix-tracing` example.
- Forge tools that run their async body through a `ThreadPoolExecutor` submit via
  `contextvars.copy_context().run` so the caller's OpenTelemetry context (and the tracing re-entrancy guard)
  survives in the pool thread.

## Assembler ↔ GitHub
- `atomic-assembler` clones `https://github.com/eigenwise/atomic-agents.git` (via GitPython), reads
  `atomic-forge/tools/`, and copies the chosen tool into the user's project (skipping build files).
  Source URL and paths are in `atomic-assembler/atomic_assembler/constants.py` and `utils.py`
  (`GithubRepoCloner`, `AtomicToolManager`).
