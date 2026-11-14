# Agents (`AtomicAgent` + `AgentConfig`)

## Contents
- Config anatomy
- Creating the agent (generics)
- Execution modes (sync, async, streaming)
- Token counting
- History and hooks (deep dives)
- Registering context providers
- Common mistakes

## Config anatomy

```python
from atomic_agents import AgentConfig
from atomic_agents.context import ChatHistory, SystemPromptGenerator
from instructor import Mode

config = AgentConfig(
    client=client,                      # required: instructor-wrapped client
    model="gpt-5.6-terra",                 # required: model id
    history=ChatHistory(),              # optional: omit for stateless agents
    system_prompt_generator=SystemPromptGenerator(
        background=["You are a concise research assistant."],
        steps=["Understand the question.", "Search memory.", "Answer directly."],
        output_instructions=["Reply under 100 words.", "Cite sources when used."],
    ),
    system_role="system",               # None = no system prompt sent
    assistant_role="assistant",         # "model" for Gemini; default works for OpenAI/Anthropic
    tool_result_role=None,              # auto-detected; override for odd providers
    mode=Mode.TOOLS,                    # Instructor mode — TOOLS / JSON / GENAI_TOOLS / …
    model_api_parameters={"temperature": 0.2, "max_tokens": 2048},
)
```

Match `mode` on `AgentConfig` to the Instructor factory mode. If the factory was built with `mode=Mode.JSON` but `AgentConfig.mode=Mode.TOOLS` (the default), the agent will format tools the provider doesn't accept. See `providers.md` for the per-provider matrix.

`assistant_role` must be `"model"` for Gemini; every other supported provider uses `"assistant"`.

## Creating the agent

Generics carry runtime information — write the types explicitly:

```python
from atomic_agents import AtomicAgent

agent = AtomicAgent[MyInput, MyOutput](config=config)
```

The PEP 695 generic parameters are the source of truth for both the agent and `BaseTool`. Do not rely on subclass-level `input_schema` / `output_schema` class attributes — the generic parameters feed `BaseTool.__init_subclass__` and populate the property at runtime.

## Execution modes

```python
# Sync single-shot
out: MyOutput = agent.run(MyInput(...))

# Sync, no new user input — rerun over existing history (see memory.md)
out = agent.run()

# Async single-shot
out = await agent.run_async(MyInput(...))

# Sync streaming — yields progressively-filled partials
for partial in agent.run_stream(MyInput(...)):
    ...

# Async streaming
async for partial in agent.run_async_stream(MyInput(...)):
    ...
```

Streaming partials are instances of `MyOutput` with a subset of fields populated; validators fire as fields appear, so keep them cheap. Working examples: `atomic-examples/quickstart/quickstart/1_1_basic_chatbot_streaming.py` (sync) and `1_2_basic_chatbot_async_streaming.py` (async, with Rich `Live` rendering).

## Token counting

```python
info = agent.get_context_token_count()
info.total            # total tokens in the next request
info.system_prompt    # system-prompt tokens
info.history          # history tokens
info.tools            # tokens consumed by tool definitions (TOOLS modes)
info.model            # tokenizer/model the count was computed against
info.max_tokens       # model's max context (None if unknown)
info.utilization      # float in [0, 1], or None
```

Use this to gate long operations or decide when to summarize history. Works across providers that Instructor can tokenize.

## History and hooks

History and hooks are large enough that each has its own reference:

- **[memory.md](memory.md)** — `ChatHistory` turn model, persistence (`dump()` / `load()`), `copy()`, `delete_turn_id()`, multimodal content, multi-agent patterns (shared, independent, agent-to-agent, supervisor).
- **[hooks.md](hooks.md)** — the five events (`parse:error`, `completion:kwargs`, `completion:response`, `completion:error`, `completion:last_attempt`), retry patterns, telemetry, production logging.

Short recap for inline use:

```python
# History
history = ChatHistory()
agent.run(BasicChatInputSchema(chat_message="Question 1"))
agent.run(BasicChatInputSchema(chat_message="Question 2"))   # accumulates
agent.reset_history()

# Hooks
def on_parse_error(error): ...
agent.register_hook("parse:error", on_parse_error)
```

Full working hooks example: `atomic-examples/hooks-example/hooks_example/main.py`.

## Registering context providers

```python
from atomic_agents.context import BaseDynamicContextProvider

class UserCtx(BaseDynamicContextProvider):
    def __init__(self):
        super().__init__(title="User Context")
        self.name = ""

    def get_info(self) -> str:
        return f"Current user: {self.name or 'anonymous'}"

agent.register_context_provider("user", UserCtx())
agent.context_providers["user"].name = "Alice"   # dynamic update before next run
```

See `context-providers.md` for patterns (RAG, time, session, cached database schema).

## Common mistakes

- Forgetting to wrap the client with `instructor.from_*` — structured outputs silently stop working.
- Passing a raw Pydantic `BaseModel` as an input/output type — must be `BaseIOSchema`.
- Using `assistant_role="assistant"` with Gemini — Gemini requires `"model"`.
- `AgentConfig.mode` out of sync with the Instructor factory mode — tools get formatted the provider can't accept.
- Unbounded `ChatHistory` in a long-running service — monitor `get_context_token_count().utilization` or set `max_messages`.
