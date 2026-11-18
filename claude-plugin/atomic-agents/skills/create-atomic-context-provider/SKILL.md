---
name: create-atomic-context-provider
description: Build a `BaseDynamicContextProvider` that injects a named, titled block into an agent's system prompt at every `run()` — current time, user identity, retrieved RAG docs, session state, cached DB schema. Use when the user asks to "add a context provider", "inject X into the prompt", "give the agent dynamic context", "wire up RAG", "make a `BaseDynamicContextProvider`", or runs `/atomic-agents:create-atomic-context-provider`.
---

# Create an Atomic Agents Context Provider

A context provider injects a named, titled block into the agent's system prompt at every `run()`. The base prompt stays static; the context is what changes between calls.

For deep material (caching strategies, async data sources, multi-agent sharing patterns), the authority is `../framework/references/context-providers.md`. This skill is the action-oriented path: clarify → write → register.

## When this fires vs the umbrella `framework` skill

- **This skill**: the user wants to add a provider — "inject the user's name", "make the agent see the current time", "feed retrieved docs into the prompt", "share session state across two agents".
- **`framework` skill**: questions about Atomic Agents in general, or about something other than authoring a provider.

## Phase 1 — Clarify

Bundle into one message:

1. **What goes into the prompt?** One sentence. Defines the provider's job.
2. **Where does the data come from?** In-memory state mutated by your code? A vector DB lookup? A REST call? A clock?
3. **How fresh must it be?** Per-`run()` (default), every N seconds (cache), or refreshed externally before each call (async data).
4. **Which agent(s)?** One agent, or shared across multiple agents?

Skip what's already obvious from context.

## Phase 2 — Plan

Confirm in one short block:

- File: `<project>/context_providers.py` (or alongside the agent that owns it).
- Class: `<Topic>Ctx(BaseDynamicContextProvider)`.
- Title (rendered as the section header in the prompt) — must be unique across providers on the same agent.
- Update mechanism: mutate fields between `run()` calls (most common), set via a method, or `await refresh()` for async sources.

## Phase 3 — Implement

### Skeleton

```python
from atomic_agents.context import BaseDynamicContextProvider


class UserCtx(BaseDynamicContextProvider):
    def __init__(self):
        super().__init__(title="User Context")
        self.name: str = ""
        self.role: str = ""

    def get_info(self) -> str:
        if not self.name:
            return "No user is logged in."
        return f"User: {self.name} (role: {self.role})"
```

`get_info()` is **synchronous** and runs on **every** `agent.run()` — keep it cheap. No HTTP, no DB queries, no file I/O. Cache slow sources (see "Cached" pattern below). For async data sources, `await provider.refresh()` from your loop before calling the agent.

### Common patterns

**Time** — read-only, no state mutation needed:

```python
from datetime import datetime, timezone

class TimeCtx(BaseDynamicContextProvider):
    def __init__(self):
        super().__init__(title="Current Time")

    def get_info(self) -> str:
        return datetime.now(timezone.utc).isoformat()
```

**RAG / retrieved docs** — set externally, read inside `get_info()`:

```python
class RAGCtx(BaseDynamicContextProvider):
    def __init__(self):
        super().__init__(title="Retrieved Documents")
        self.docs: list[dict] = []

    def set(self, docs: list[dict]) -> None:
        self.docs = docs

    def get_info(self) -> str:
        if not self.docs:
            return "No relevant documents retrieved."
        return "\n\n".join(f"[{d['source']}] {d['content']}" for d in self.docs)

# In the calling code, just before agent.run():
rag.set(vector_db.search(query, k=4))
agent.run(query_input)
```

**Session** — mutable key/value state shared across agents:

```python
class SessionCtx(BaseDynamicContextProvider):
    def __init__(self):
        super().__init__(title="Session")
        self._data: dict[str, str] = {}

    def set(self, key: str, value: str) -> None:
        self._data[key] = value

    def get_info(self) -> str:
        if not self._data:
            return "No session state."
        return "\n".join(f"- {k}: {v}" for k, v in self._data.items())
```

**Cached** — for slow sources (DB schema, expensive computation):

```python
import time

class DBSchemaCtx(BaseDynamicContextProvider):
    def __init__(self, conn, ttl_seconds: int = 300):
        super().__init__(title="Database Schema")
        self._conn = conn
        self._ttl = ttl_seconds
        self._cached: str = ""
        self._at: float = 0.0

    def get_info(self) -> str:
        now = time.time()
        if not self._cached or now - self._at > self._ttl:
            self._cached = render_schema(self._conn)
            self._at = now
        return self._cached
```

**Async source** — refresh outside, read sync inside:

```python
class AsyncCtx(BaseDynamicContextProvider):
    def __init__(self):
        super().__init__(title="Async Data")
        self._cached = ""

    async def refresh(self) -> None:
        self._cached = format(await fetch_remote())

    def get_info(self) -> str:
        return self._cached

# Caller
await ctx.refresh()
await agent.run_async(input_data)
```

## Phase 4 — Register

```python
ctx = UserCtx()
agent.register_context_provider("user", ctx)

# Mutate before each run as needed:
ctx.name = "Alice"; ctx.role = "admin"
agent.run(...)
```

Sharing one provider instance across agents is allowed — updates propagate to every agent that registered it:

```python
shared = SessionCtx()
agent_a.register_context_provider("session", shared)
agent_b.register_context_provider("session", shared)
shared.set("locale", "en-GB")  # visible to both agents
```

Inspect or unregister:

```python
"user" in agent.system_prompt_generator.context_providers
agent.unregister_context_provider("user")
```

## Phase 5 — Verify

Quick smoke test that the provider renders:

```bash
uv run python -c "from <project>.context_providers import UserCtx; c = UserCtx(); c.name='Alice'; c.role='admin'; print(c.get_info())"
```

Then confirm the rendered system prompt includes the provider's section by inspecting `agent.system_prompt_generator.generate_prompt(...)` or by running the agent and checking the first request's payload via the `completion:kwargs` hook (see `../framework/references/hooks.md`).

## Phase 6 — Hand off

Tell the user:

- Where the provider lives, what name was used in `register_context_provider`, and how to mutate it.
- The lifecycle: register once, mutate fields between calls, the prompt picks up the change automatically.
- Optional next steps:
  - The agent that consumes it → `create-atomic-agent` skill.
  - A research / RAG loop that updates the provider in a loop → see `atomic-examples/deep-research/` and `../framework/references/orchestration.md`.

## Anti-patterns

- Slow I/O inside `get_info()` — runs on every `agent.run()`. Cache it or refresh externally.
- Returning a non-string from `get_info()` — raises at prompt time.
- Forgetting `register_context_provider(...)` — the provider never reaches the prompt.
- Duplicate titles across providers on the same agent — sections collide. Use unique titles.
- Storing secrets in the provider — they end up in every LLM request. Inject only what the model needs to reason about.

For deeper material — multi-agent sharing patterns, dynamic-update research loops, async source patterns — load `../framework/references/context-providers.md`.
