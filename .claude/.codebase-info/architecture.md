# Architecture

*Last Updated: 2026-07-05*

## Summary

Atomic Agents builds agentic AI applications out of small, single-purpose, composable parts. Its
central idea — "atomicity" — is that an agent is essentially a typed function: a Pydantic **input
schema** in, a Pydantic **output schema** out, with the LLM call mediated by the
[Instructor](https://github.com/instructor-ai/instructor) library so structured outputs work across
providers (OpenAI, Anthropic, Gemini, and anything LiteLLM supports). There are no hidden
abstractions: the developer controls the system prompt, the conversation history, and the schemas.

The repository is a **monorepo** managed as a `uv` workspace with four parts:

- **`atomic-agents/`** — the core framework (PyPI package `atomic-agents`, imported as `atomic_agents`).
- **`atomic-assembler/`** — a Textual TUI (the `atomic` command) that downloads tools from the forge
  into a user's project.
- **`atomic-forge/`** — a library of 13 standalone, copy-into-your-project tools.
- **`atomic-examples/`** — 15 runnable example applications.

## High-Level Flow

```
Agent run (atomic_agents core):

  input schema ─┐
                ▼
  SystemPromptGenerator (+ context providers) ──► system message ─┐
  ChatHistory (typed Messages, multimodal, token-trimmed) ─► msgs ┤
                                                                  ▼
              instructor → client.chat.completions.create(response_model=Out)
                                                                  │
                            LLM provider (OpenAI / Anthropic / Gemini / LiteLLM …)
                                                                  │
                            validated Out (Pydantic) ◄────────────┘
                                  ├──► appended to ChatHistory
                                  └──► returned to caller

Tool install (atomic-assembler TUI):
  `atomic`  →  clone github.com/eigenwise/atomic-agents  →  copy atomic-forge/tools/<tool>  →  user project
```

## Components

| Component | Path | Responsibility |
|-----------|------|----------------|
| Core agent | `atomic-agents/atomic_agents/agents/atomic_agent.py` | `AtomicAgent`, `AgentConfig`, run / stream / async methods |
| Base contracts | `atomic-agents/atomic_agents/base/` | `BaseIOSchema`, `BaseTool`, `BaseToolConfig`, `BaseResource`, `BasePrompt` |
| Context | `atomic-agents/atomic_agents/context/` | `SystemPromptGenerator`, `BaseDynamicContextProvider`, `BaseChatHistory` (pluggable memory contract) + `ChatHistory` |
| Connectors | `atomic-agents/atomic_agents/connectors/mcp/` | Model Context Protocol tools / resources / prompts |
| Utils | `atomic-agents/atomic_agents/utils/` | Token counting (LiteLLM), tool-message formatting |
| Assembler (CLI) | `atomic-assembler/atomic_assembler/` | Textual TUI to fetch/install forge tools |
| Forge (tools) | `atomic-forge/tools/` | 13 standalone tools, each `BaseTool`-based |

## The Agent Run Lifecycle

1. The caller wraps an LLM client with Instructor (e.g. `instructor.from_openai(OpenAI())`) and builds
   an `AgentConfig`, then instantiates `AtomicAgent[InputSchema, OutputSchema](config)`.
2. On `run(input)`, the `SystemPromptGenerator` assembles the system message from
   background / steps / output-instructions plus any registered **context providers** (evaluated live
   at call time).
3. The `ChatHistory` (typed `Message`s, multimodal-aware) is serialized into the provider message
   list; oldest *turns* are trimmed to respect `max_context_tokens`.
4. `client.chat.completions.create(response_model=output_schema)` performs the structured LLM call via
   Instructor. Streaming and async variants exist: `run_stream`, `run_async`, `run_async_stream`.
5. The validated output schema is appended to history and returned. Instructor **hooks**
   (`parse:error`, `completion:kwargs`, `completion:response`, …) provide observability and retry.

## Key Decisions & Constraints

- **Schema-first, no magic.** Every input/output subclasses `BaseIOSchema` (Pydantic); a non-empty
  docstring is required and used as the schema's description.
- **Provider-agnostic.** Instructor + LiteLLM mean any supported model works; the agent never
  hardcodes a provider. (Gemini quirk: pass `assistant_role="model"`.)
- **Python ≥3.12** — uses PEP 695 generic syntax (`AtomicAgent[In, Out]`).
- **Tools are not a dependency.** Forge tools are *copied* into the user's repo (full control, no
  version lock-in) by the assembler, rather than pip-installed.
- **No database and no containers** in the framework itself.
- **v2 was a breaking change** from v1 — see `UPGRADE_DOC.md` (`BaseAgent`→`AtomicAgent`,
  `AgentMemory`→`ChatHistory`, flattened imports, schemas moved to generic type parameters).
