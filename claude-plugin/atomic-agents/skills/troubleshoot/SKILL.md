---
name: troubleshoot
description: Diagnose and fix a failing or misbehaving Atomic Agents app — import errors, schema validation failures, empty or malformed LLM output, provider role/mode errors, history and context problems, MCP transport errors. Use when the user reports an error, traceback, or wrong behavior in atomic-agents code, asks "why is my agent not working / crashing / returning garbage", or pastes a `ValidationError`, `ImportError`, or provider API error from an atomic-agents project.
---

# Troubleshoot an Atomic Agents App

Diagnostic workflow for the "it's broken" moment. Most atomic-agents failures are one of a dozen known causes with mechanical fixes. Match the symptom, apply the fix, verify by re-running.

## Workflow

1. **Capture the failure.** Get the exact traceback or wrong-output sample plus the code that produces it (agent construction, schemas, client wiring). If the user pasted only a fragment of the error, ask for the full traceback before guessing.
2. **Match against the symptom table below.** The quoted strings are the framework's real messages — match on them.
3. **Apply the fix and re-run** the failing snippet. Do not declare the problem solved without a passing re-run.
4. **No match?** Load the reference for the failing area (table at the bottom) and reason from there. For a whole-codebase audit rather than one failure, delegate to the `atomic-reviewer` subagent instead.

## Symptom table

### Import and definition errors

| Symptom | Cause | Fix |
|---|---|---|
| `ImportError`/`ModuleNotFoundError` on `atomic_agents.lib.*`, `atomic_agents.agents.base_agent`, or `BaseAgent` | v1 import paths, removed in v2 | Import from the top level: `from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema, BaseTool`; context pieces from `atomic_agents.context` |
| `ValueError: <Name> must have a non-empty docstring to serve as its description` at import time | `BaseIOSchema` subclass without a docstring | Add a docstring describing the schema — it flows into the LLM prompt, so write it for the model |
| `ValidationError` when constructing `AgentConfig`, complaining about `client` | Raw provider SDK client passed; `AgentConfig.client` requires an Instructor-wrapped client | Wrap it: `instructor.from_openai(...)`, `instructor.from_anthropic(...)`, `instructor.from_genai(...)` |
| `TypeError` about missing type parameters, or output typed as `BasicChatOutputSchema` when a custom schema was expected | `AtomicAgent` instantiated without generics | Write `AtomicAgent[InputSchema, OutputSchema](config=...)` — the type parameters carry runtime information (they drive Instructor's `response_model`) |

### Provider errors at run time

| Symptom | Cause | Fix |
|---|---|---|
| Anthropic error mentioning `max_tokens` is required | Anthropic requires it per request | `model_api_parameters={"max_tokens": 4096}` on `AgentConfig` |
| Gemini error about invalid role / role ordering | Gemini names the assistant role `model` | `assistant_role="model"` on `AgentConfig`; use `instructor.from_genai(...)` with `mode=Mode.GENAI_TOOLS` and match `AgentConfig.mode` |
| Empty output, `None` fields, or JSON-parse noise on Groq / Ollama / MiniMax | Provider needs JSON mode; factory and config modes disagree | Create the client with `Mode.JSON` and set `mode=Mode.JSON` on `AgentConfig` — the two must always match |
| Works on OpenAI, breaks on another provider with no code change | Provider quirks live in `AgentConfig`, not the agent | Check the per-provider matrix in [framework/references/providers.md](../framework/references/providers.md) |

### Wrong or degraded behavior (no exception)

| Symptom | Cause | Fix |
|---|---|---|
| Pydantic `ValidationError` from inside `run()` after retries | Model can't satisfy the output schema | Improve field `description=`s and the schema docstring first; loosen over-tight constraints; only then consider a stronger model. Inspect attempts via the `parse:error` hook |
| Agent "forgets" earlier turns | Fresh `ChatHistory` created per call, or none passed | Create one `ChatHistory` and reuse it across `run()` calls |
| Context/cost grows without bound in long sessions | Unbounded history | Set `max_context_tokens` on `AgentConfig` — oldest turns are trimmed automatically |
| Every `run()` is slow before the LLM call even starts | Blocking I/O in a context provider's `get_info()` | `get_info()` runs on every `run()`; move fetching out, cache, or make the provider hold precomputed state |
| `history.load(...)` "doesn't work" | Called as a classmethod | `load` is an instance method that mutates in place: `history = ChatHistory(); history.load(saved)`. `dump()` returns a JSON string |

### MCP interop

| Symptom | Cause | Fix |
|---|---|---|
| `AttributeError: STREAMABLE_HTTP` | No such member | `MCPTransportType.HTTP_STREAM` (others: `SSE`, `STDIO`) |
| MCP tools fetch but calls fail | Transport/endpoint mismatch | See MCP section of [framework/references/tools.md](../framework/references/tools.md) and the `mcp-agent` example |

## Escalation map

| Failing area | Load |
|---|---|
| Schema design, validators | [framework/references/schemas.md](../framework/references/schemas.md) |
| Agent config, run modes, streaming | [framework/references/agents.md](../framework/references/agents.md) |
| Provider wiring, roles, modes | [framework/references/providers.md](../framework/references/providers.md) |
| History, persistence, multi-agent memory | [framework/references/memory.md](../framework/references/memory.md) |
| Hooks, retries, telemetry | [framework/references/hooks.md](../framework/references/hooks.md) |
| Tools, MCP | [framework/references/tools.md](../framework/references/tools.md) |
| Orchestration between agents | [framework/references/orchestration.md](../framework/references/orchestration.md) |

Related skills: `framework` (general guidance, auto-triggers), `create-atomic-agent` / `create-atomic-schema` / `create-atomic-tool` / `create-atomic-context-provider` (building rather than fixing). For a review of code that isn't currently failing, use the `atomic-reviewer` subagent.
