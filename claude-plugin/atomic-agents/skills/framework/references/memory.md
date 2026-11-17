# Memory and Chat History (`ChatHistory`)

## Contents
- The turn model
- Creating and wiring history
- Automatic memory management during `run()`
- Persistence — `dump()` / `load()` / `copy()`
- Overflow and manipulation (`max_messages`, `delete_turn_id`)
- Multimodal content in history
- Multi-agent patterns (shared, independent, agent-to-agent, supervisor)
- Common mistakes

Working example in the repo: `atomic-examples/fastapi-memory/` and the "Memory" guide at `docs/guides/memory.md` (authoritative deep-dive).

## The turn model

Every `agent.run()` produces **one turn**. A turn groups the user message and the assistant response — and any intermediate messages (tool results, context injections) — under a shared `turn_id` (UUID). Messages are `Message(role, content, turn_id)` where `content` is always a `BaseIOSchema` instance.

Key implications:

- Turns are deleted atomically via `history.delete_turn_id(id)`.
- `history.current_turn_id` tracks the latest turn. `initialize_turn()` rotates it.
- When `agent.run(None)` is called, it reuses existing history without adding a new user message — useful for agent-to-agent loops (see below).

## Creating and wiring history

```python
from atomic_agents import AtomicAgent, AgentConfig
from atomic_agents.context import ChatHistory

history = ChatHistory()                           # unbounded
history = ChatHistory(max_messages=40)            # FIFO overflow

# Optional: seed with an assistant intro so the first user turn has something to reply to
history.add_message("assistant", BasicChatOutputSchema(chat_message="Hi!"))

agent = AtomicAgent[In, Out](config=AgentConfig(client=client, model="gpt-5.6-terra", history=history))
```

Omit `history` entirely for stateless agents — each `run()` is independent.

## Automatic memory management

`agent.run(user_input)` does, in order:

1. Call `history.initialize_turn()` to create a new `turn_id`.
2. Append the user message to history under that turn.
3. Call the model with the full history + system prompt.
4. Parse the response into `OutputSchema`.
5. Append the assistant message to history under the same `turn_id`.

That means:

- History accumulates automatically. Do not call `add_message` for the assistant response — it is added by `run()`.
- `history.delete_turn_id(agent.history.current_turn_id)` undoes the last exchange (both user and assistant messages in one call).
- `run(None)` skips step 2 and re-runs the model over whatever is in history. Useful to let the model continue after a manually-injected message.

## Persistence

`dump()` returns a JSON **string**. `load(s)` mutates the instance in place.

```python
# Save
blob: str = history.dump()
open("session.json", "w").write(blob)

# Restore
restored = ChatHistory()
restored.load(open("session.json").read())

# Clone
clone = history.copy()   # fully independent copy via dump/load
```

`load()` re-imports the schema classes by fully qualified name (stored in the dumped payload), so the same module layout must be importable at load time.

## Overflow and manipulation

```python
history = ChatHistory(max_messages=20)   # drop oldest when exceeded

history.get_message_count()              # -> int
history.get_history()                    # -> list[dict]  (role + JSON content)
history.delete_turn_id(turn_id)          # remove one turn; raises if not found
history.get_current_turn_id()            # -> Optional[str]

agent.reset_history()                    # wipe and restore the initial seed
```

For token-aware trimming (rather than message-count), use `agent.get_context_token_count()` to check utilization and either delete old turns or summarize them into a single synthetic message before continuing.

## Multimodal content

History handles Instructor's multimodal types (`Image`, `Audio`, `PDF`) transparently. Include them inside a `BaseIOSchema` field:

```python
from instructor.processing.multimodal import Image
from atomic_agents import BaseIOSchema
from pydantic import Field

class VisualQuery(BaseIOSchema):
    """Question plus an image."""
    question: str = Field(..., description="User question.")
    image: Image = Field(..., description="Input image.")

history.add_message("user", VisualQuery(question="What is this?", image=Image.from_path("x.jpg")))
```

`dump()` / `load()` round-trip multimodal content. `get_history()` returns a mixed array of JSON + multimodal objects that Instructor consumes directly. See `atomic-examples/basic-multimodal/` for a full example.

## Multi-agent patterns

The framework guide (`docs/guides/memory.md`) names five canonical patterns. The three most common:

### Shared history

Two agents participate in one conversation.

```python
shared = ChatHistory()
router   = AtomicAgent[In, RouteOut](config=AgentConfig(client=c, model=m, history=shared))
worker   = AtomicAgent[In, WorkOut](config=AgentConfig(client=c, model=m, history=shared))
```

Only safe when the agents run **sequentially** on the same thread. Concurrent runs against one history race.

### Independent histories

Each agent has its own `ChatHistory()`. Outputs are passed via typed schemas at the call site. Default for multi-agent orchestration — keeps state contained.

### Agent-to-agent messaging (addresses [GitHub issue #58](https://github.com/eigenwise/atomic-agents/issues/58))

Each agent keeps its own history; the orchestrator manually injects the other agent's output as a `"user"` message and calls `run(None)` to let the recipient continue.

```python
writer   = AtomicAgent[WriterIn, WriterOut](config=AgentConfig(..., history=ChatHistory()))
reviewer = AtomicAgent[ReviewerIn, ReviewerOut](config=AgentConfig(..., history=ChatHistory()))

draft = writer.run(WriterIn(task="draft a product description"))
for _ in range(3):
    verdict = reviewer.run(ReviewerIn(content_to_review=draft.content))
    if verdict.approved:
        break
    # Inject reviewer feedback into writer's own history
    writer.history.add_message(
        "user",
        WriterIn(task=f"Revise based on feedback: {verdict.feedback}")
    )
    draft = writer.run()   # run(None) — continues with injected message in context
```

### Supervisor-worker via context providers

When multiple workers should share a fact (current user, selected dataset, etc.) but not a full history, use a `BaseDynamicContextProvider` registered on each. See `context-providers.md`.

## Common mistakes

- Sharing one `ChatHistory` across concurrent agents — messages interleave and corrupt turns.
- Calling `ChatHistory.load(saved)` as a classmethod — `load` is an **instance** method; construct first, then `.load(...)`.
- Treating `dump()` output as a dict — it's a JSON **string**; parse it with `json.loads` before inspection.
- Storing secrets inside `BaseIOSchema` content — they will be serialized by `dump()` and replayed on every `run()`.
- Unbounded `ChatHistory` in long-running services — set `max_messages` or monitor `agent.get_context_token_count().utilization`.
- Expecting `run(None)` to work without seeded history — it needs existing messages to respond to.
