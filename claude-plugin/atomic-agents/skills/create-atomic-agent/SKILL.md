---
name: create-atomic-agent
description: Build and wire an `AtomicAgent[InSchema, OutSchema]` — schemas, `AgentConfig`, `SystemPromptGenerator`, provider client, history, hooks, optional context providers. Use when the user asks to "create an agent", "add another agent", "build an `AtomicAgent`", "wire up an agent", "make a planner/router/extractor agent", or runs `/atomic-agents:create-atomic-agent`.
---

# Create an Atomic Agent

An agent is an LLM-backed transformer from one `BaseIOSchema` to another. Building one means: design the schemas, write the system prompt, wire the provider client, build the `AgentConfig`, instantiate `AtomicAgent[In, Out]`.

For deep material (streaming, token counting, hooks, multi-agent memory), the authority is `../framework/references/agents.md` plus `providers.md`, `prompts.md`, and `memory.md`. This skill is the action-oriented path: clarify → write → run.

## When this fires vs the umbrella `framework` skill

- **This skill**: the user is creating or wiring a specific agent — "add a planner agent", "build a Q&A agent", "make a router that classifies tickets".
- **`framework` skill**: questions about Atomic Agents in general, or the user is doing something other than authoring an agent.

## Phase 1 — Clarify

Bundle into one message:

1. **What should the agent do?** One sentence. Becomes the persona / `background` line.
2. **Inputs and outputs.** Use `BasicChatInputSchema` / `BasicChatOutputSchema` for free-form chat. Use a custom pair for anything structured (extraction, classification, planning, routing). When custom, branch to the `create-atomic-schema` skill for the schema authoring.
3. **Provider.** OpenAI / Anthropic / Groq / Ollama / Gemini / OpenRouter / MiniMax. Default: whatever the project already uses; otherwise OpenAI.
4. **Conversational?** Yes → wire a `ChatHistory`. No (single-shot transformer) → omit it for stateless behavior.
5. **Context providers.** Anything to inject into the prompt at runtime (current time, user identity, retrieved docs)? If yes, plan to also use the `create-atomic-context-provider` skill afterwards.

Skip anything already settled in context.

## Phase 2 — Plan

State the plan in one short block:

- File: `<project>/agents/<agent_name>.py` (or directly in `main.py` for a tiny project — see `../framework/references/project-structure.md`).
- Schemas: which pair, where they live.
- Provider + model + Instructor mode. Default models: OpenAI `gpt-5.6-terra`, Anthropic `claude-haiku-4-5`, Groq `llama-3.3-70b-versatile`, Ollama `llama3.1`, Gemini `gemini-2.5-flash`.
- `SystemPromptGenerator` content — three sections: `background`, `steps`, `output_instructions`.
- History? Hooks? Context providers?

## Phase 3 — Implement

### Canonical imports (do not deviate)

```python
from atomic_agents import (
    AtomicAgent, AgentConfig,
    BasicChatInputSchema, BasicChatOutputSchema,
)
from atomic_agents.context import ChatHistory, SystemPromptGenerator
from instructor import Mode
```

### Wire the provider client (always Instructor-wrapped)

The full per-provider matrix lives in `../framework/references/providers.md`. Quick recap:

```python
# OpenAI — default mode is Mode.TOOLS
import os, instructor, openai
client = instructor.from_openai(openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
model = "gpt-5.6-terra"
api_params: dict = {}

# Anthropic — Mode.TOOLS, max_tokens REQUIRED in model_api_parameters
import anthropic
client = instructor.from_anthropic(anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]))
model = "claude-haiku-4-5"
api_params = {"max_tokens": 4096}

# Gemini — Mode.GENAI_TOOLS, assistant_role="model"
from google import genai
client = instructor.from_genai(genai.Client(api_key=os.environ["GEMINI_API_KEY"]), mode=Mode.GENAI_TOOLS)
model = "gemini-2.5-flash"
api_params = {}

# Groq / Ollama / MiniMax — Mode.JSON in both factory and AgentConfig
```

### Build the agent

```python
from atomic_agents import AtomicAgent, AgentConfig
from atomic_agents.context import ChatHistory, SystemPromptGenerator

agent = AtomicAgent[MyInput, MyOutput](
    config=AgentConfig(
        client=client,
        model=model,
        history=ChatHistory(),  # omit for stateless
        system_prompt_generator=SystemPromptGenerator(
            background=["You are a concise research assistant."],
            steps=[
                "Read the question carefully.",
                "Decide what minimum information answers it.",
                "Produce the answer in the required schema.",
            ],
            output_instructions=[
                "Reply under 100 words.",
                "If unsure, set status='error' and explain why.",
            ],
        ),
        # Provider-specific knobs — match the Instructor factory
        # mode=Mode.TOOLS,                         # OpenAI / Anthropic / OpenRouter
        # mode=Mode.JSON,                          # Groq / Ollama / MiniMax
        # mode=Mode.GENAI_TOOLS, assistant_role="model",  # Gemini
        model_api_parameters=api_params or {"temperature": 0.2},
    )
)
```

### Generics carry the truth

`AtomicAgent[MyInput, MyOutput]` — write the type parameters explicitly. The framework reads them at class-definition time. Do **not** rely on subclass-level `input_schema` / `output_schema` class attributes.

### Provider-specific knobs (most common gotchas)

- **Anthropic** without `max_tokens` in `model_api_parameters` → API rejects every call.
- **Gemini** without `assistant_role="model"` → role mismatch on every turn.
- **Groq / Ollama / MiniMax** with `Mode.TOOLS` → tools formatted in a way the provider does not accept; flip to `Mode.JSON`.
- Reasoning models (o-series, GPT-5 reasoning variants) → often want `system_role=None` and `reasoning_effort` in `model_api_parameters`.

## Phase 4 — Run and verify

```python
out = agent.run(MyInput(...))
print(out)
```

Quick smoke test without paying for a real call:

```bash
uv run python -c "from <project>.agents.<agent_name> import agent; print(type(agent).__name__, '->', agent.input_schema.__name__, '/', agent.output_schema.__name__)"
```

If output validation fails repeatedly, the `parse:error` hook has the details — see `../framework/references/hooks.md` for registration.

## Phase 5 — Hand off

Tell the user:

- How to call `agent.run(...)` (and `run_async`, `run_stream`, `run_async_stream` when appropriate).
- Which env var to set for the provider key.
- Optional next steps:
  - Tools the agent should be able to invoke → `create-atomic-tool` skill.
  - Dynamic data injected into the prompt → `create-atomic-context-provider` skill.
  - Custom schemas → `create-atomic-schema` skill.
  - Multiple agents working together → `../framework/references/orchestration.md`.
  - Telemetry / retries / logging → `../framework/references/hooks.md`.
  - Conversation persistence, summarization, multi-agent memory → `../framework/references/memory.md`.

## Anti-patterns

- Forgetting to wrap the client with `instructor.from_*` — structured outputs silently stop working.
- `BaseModel` instead of `BaseIOSchema` for the agent's input or output type.
- `AgentConfig.mode` out of sync with the Instructor factory mode.
- `assistant_role="assistant"` on Gemini — must be `"model"`.
- Missing `max_tokens` on Anthropic — every call fails.
- Hardcoded API keys in the source — read from env.
- Unbounded `ChatHistory` in a long-running service — monitor `agent.get_context_token_count().utilization` or set `max_messages`.

For deep material — streaming, async, token counting, hooks, multi-agent history — load `../framework/references/agents.md`.
