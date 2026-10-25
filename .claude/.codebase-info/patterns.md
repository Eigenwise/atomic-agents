# Patterns & Conventions

*Last Updated: 2026-10-24*

## Atomicity
Build with small, single-purpose, composable parts ("LEGO blocks"): each agent, tool, and context
provider does one thing and is reusable. Compose pipelines by feeding one agent's output schema into
the next's input. See `atomic-examples/deep-research` and `orchestration-agent`.

## Schema-driven I/O
- Every input and output is a `BaseIOSchema` (Pydantic) subclass with a **required docstring** (used
  as the LLM-facing description). Naming: `<Thing>InputSchema` / `<Thing>OutputSchema`.
- Agents are parameterized by schemas via PEP 695 generics: `AtomicAgent[InputSchema, OutputSchema]`.
- Fully specialized generic agent subclasses resolve their concrete schemas through intermediate bases per instance.
  Different specializations keep independent input/output bindings; untyped agents retain the basic chat defaults.
- Structured output is enforced by Instructor through `response_model=OutputSchema`.
- Streaming completion accepts field names when reconstructing output schemas with aliases.
- Tool messages use Pydantic JSON-mode serialization, including JSON-only field serializers.
- MCP dictionary schemas preserve mapping values and validate typed `additionalProperties`,
  including mappings reached through `$defs` or `definitions` references.

## Dynamic system prompts (context providers)
- `SystemPromptGenerator(background=[...], steps=[...], output_instructions=[...], context_providers={...})`.
- A `BaseDynamicContextProvider` injects live data (retrieved docs, current date, search results) into
  the system prompt at call time. Register via `agent.register_context_provider(name, provider)`.

## Tools
- A tool is a `BaseTool[InputSchema, OutputSchema]` with a `run(params) -> OutputSchema` method and an
  optional `BaseToolConfig` (override `title`/`description` to disambiguate similar tools).
- Forge tool layout (`atomic-forge/guides/tool_structure.md`): imports → input schema → output
  schema(s) → config → tool class + logic → example usage.

## Memory
- `BaseChatHistory` (`context/base_chat_history.py`) is an interface-only ABC declaring the memory
  contract `AtomicAgent` relies on (`add_message`, `get_history`, turn handling, `dump()`/`load()`,
  `copy()`, plus the `history`/`current_turn_id` attributes). It is the documented, dependency-free
  seam for plugging in custom/persistent backends; `AgentConfig.history` is typed to it.
- `ChatHistory` is the built-in implementation: stores typed `Message`s grouped into turns (a
  user+assistant pair shares a `turn_id`), is multimodal-aware (Image/Audio/PDF plus `VideoURL`),
  and supports `dump()`/`load()`. `VideoURL` is converted to an OpenAI-compatible `video_url`
  content part for provider messages. `AtomicAgent` trims the oldest whole turns to honor
  `max_context_tokens`, using a text placeholder when LiteLLM cannot count video parts.
- Invalid serialized history is validated before replacing existing state. Context trimming includes the incoming message in the token budget.
- Custom backend pattern: subclass `ChatHistory` and override `add_message`/`load` to persist (see
  the `persistent-memory` example and the "Writing a Custom Memory Backend" guide section).

## Error handling & retries
- Lean on Instructor's validation/retry of structured outputs, plus hook events (`parse:error`, …)
  for observability. See `docs/guides/error-handling.md` and `docs/guides/hooks.md`.

## Testing
- `pytest` (+ `pytest-asyncio` for async, `pytest-cov` for coverage), with `unittest.mock` for LLM
  clients. Core tests in `atomic-agents/tests/` mirror the package layout (`agents/`, `base/`,
  `context/`, `connectors/mcp/`, `utils/`), including `tests/base/test_multimodal.py` for `VideoURL`.
  Discovery (`pytest.ini`): files `test_*.py`, classes `Test*`, functions `test_*`.
- Instructor transport regressions use real clients with `httpx.MockTransport`, including sync/async streaming and explicit strict validation.
- MCP tests use real SDK result models with mocked transport/session boundaries. Synthetic execution against production filenames is not used to increase coverage.

## Configuration
- Use explicit `Field(default=...)` for public config defaults so static type checkers infer optional constructor arguments. The AgentConfig constructor regression covers direct calls and typed settings unpacking.
- Omitted `BaseTool`, `BaseResource`, and `BasePrompt` configurations are created per instance.
  Explicitly supplied configuration objects retain their identity.
- Runtime config is explicit via `AgentConfig` (client, model, history, roles, mode,
  `model_api_parameters`, `max_context_tokens`) — no global state. Secrets (API keys) come from the
  environment / `.env`.
