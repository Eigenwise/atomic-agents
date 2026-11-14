# Communication & Integrations

*Last Updated: 2026-11-14*

The framework exposes no HTTP API of its own; "communication" here means how it talks to LLM
providers and external tools.

## LLM providers (via Instructor)
- All model calls go through an **Instructor-wrapped client** the caller supplies
  (`instructor.from_openai(...)`, `from_anthropic(...)`, etc.). The agent calls
  `client.chat.completions.create(response_model=OutputSchema)` (and `create_partial` for streaming).
- Provider-agnostic: OpenAI, Anthropic, Google Gemini, MiniMax, and 100+ models via LiteLLM.
  `AgentConfig.mode` defaults to `None` and follows the Instructor client's mode (so a `mode=Mode.JSON`
  factory is enough for JSON-only hosts); an explicit mode from a different transmission family than the
  client's is kept but logs a warning.
- Multimodal content uses Instructor's Image/Audio/PDF types plus the framework's `VideoURL`;
  `ChatHistory.get_history()` converts `VideoURL` to an OpenAI-compatible `video_url` dict, which
  Instructor forwards unchanged to providers that support video inputs. Media in fields marked
  `exclude=True` is left out of the request, matching the JSON text.
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
- Discovery follows tool, resource, template and prompt cursors. Template continuation errors propagate, and a
  repeated cursor raises `RuntimeError` instead of looping.
- HTTP and SSE endpoints go through `_transport_endpoint` (`mcp_definition_service.py`): the transport path
  (`/mcp/` or `/sse`) is appended to the URL path, the query string is kept, and a URL whose path already ends
  in that segment is used as given.
- Schema references resolve by their complete JSON Pointer through `referencing` (cached per full reference), so
  same-named definitions under `$defs` and `definitions` stay distinct. Escaped names and non-object targets are
  supported. Boolean schemas are accepted: `true` maps to `Any`, `false` to a field that rejects every value and is
  emitted as `{"not": {}}`.
- Field constraints are carried onto the generated models: numeric `minimum` / `maximum` / `exclusiveMinimum` /
  `exclusiveMaximum` / `multipleOf`, string `minLength` / `maxLength` / `pattern`, array `minItems` / `maxItems`.
  Draft-04 boolean exclusive flags are ignored. A pattern Pydantic's regex engine rejects is logged at WARNING and
  skipped; the field keeps its other constraints.
- Resource URIs are kept exactly as the server sent them (no decoding). Braces in a plain SDK `Resource` URI are
  literal; only resource templates and dict-shaped entries get URI parameters from them. Each generated tool,
  resource and prompt class reports errors and builds results with its own name and output schema.
- Scalar `enum`/`const` choices are intersected with the advertised JSON Schema (compositions and local `$defs`
  included, checked with `jsonschema` at model construction) so the generated `Literal` and emitted schema only
  admit values the server accepts. An empty intersection fails that one tool definition; the factory logs and skips
  it and keeps the others.
- Typed SDK tool results use structured content only when it is non-None, then use the existing validated JSON-text fallback. Present empty structured data retains precedence. Mapping adapters keep their separate validation contract. Generic (untyped) tool output uses `structuredContent` when `content` is empty.
- SDK `CallToolResult` and strictly validated mapping envelopes with `isError=True` raise a `RuntimeError` before typed or generic output parsing. The message uses `TextContent` only (meaningful fallback when no text); `structuredContent`, `_meta` and binary blocks never enter messages or logs. Raw application dictionaries that merely contain `isError`/`name` keys (even with a string `content`) stay ordinary outputs.

## Hooks / observability
- Built on Instructor's hook system: `register_hook(event, handler)`, `unregister_hook`,
  `clear_hooks`. Events include `parse:error`, `completion:kwargs`, `completion:response`,
  `token:counted`. See the `hooks-example` example and `docs/guides/hooks.md`.
- `disable_hooks()` detaches this agent's registered Instructor callbacks. Registrations made while disabled stay
  paused; `enable_hooks()` restores the remaining callbacks. Direct client callbacks and tracing hooks remain independent.
- `token:counted` handlers are kept on the agent, not on the Instructor client, so agents sharing one client don't
  fire each other's token handlers. Every other event goes to Instructor and is client-wide.
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
- Optional `TracingConfig(capture_messages=True)` (or `ATOMIC_AGENTS_TRACE_CAPTURE_MESSAGES=true`) records standard
  `gen_ai.input.messages` / `gen_ai.output.messages` JSON on chat attempt spans for supported ChatCompletion text
  and tool parts. Capture is off by default. Hiding wins before content access: `hide_inputs` drops the request
  attribute, `hide_outputs` drops both (requests replay earlier model output). Oversized complete JSON attributes
  are omitted with their size left in `<key>.original_length`, unparseable tool-call arguments stay as raw text, and native formats/streams/media are not converted, so non-chat providers yield partial request text
  only. Exporter setup remains app-owned; backend presentation is unverified.
- Forge tools that run their async body through a `ThreadPoolExecutor` submit via
  `contextvars.copy_context().run` so the caller's OpenTelemetry context (and the tracing re-entrancy guard)
  survives in the pool thread.

## Assembler ↔ GitHub
- `atomic-assembler` clones `https://github.com/eigenwise/atomic-agents.git` (via GitPython), reads
  `atomic-forge/tools/`, and copies the chosen tool into the user's project (skipping build files).
  Source URL and paths are in `atomic-assembler/atomic_assembler/constants.py` and `utils.py`
  (`GithubRepoCloner`, `AtomicToolManager`).
- `GithubRepoCloner(base_url, branch="main")` removes its temp directory when the clone fails. A missing tools
  folder lists as empty. `copy_atomic_tool` refuses an existing destination and, when a copy fails partway, removes
  only the folder it created, so the copy can be retried.
