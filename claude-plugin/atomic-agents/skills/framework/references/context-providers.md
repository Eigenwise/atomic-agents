# Context Providers (`BaseDynamicContextProvider`)

## Contents
- What they do
- Writing a provider
- Registering, updating, sharing
- Patterns: user, time, RAG, session, cached DB schema
- Dynamic updates inside a run loop
- Async data sources
- Common mistakes

## What they do

A context provider injects a named, titled block into the agent's system prompt at each `run()`. The base prompt stays static; the context is what changes.

Every `SystemPromptGenerator` section is followed by registered providers, rendered as:

```
## <provider title>
<get_info() output>
```

## Writing a provider

Subclass `BaseDynamicContextProvider`, set the title in `super().__init__`, implement `get_info()`:

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

`get_info()` must return a string and must be fast — it runs on every `agent.run()`.

## Registering, updating, sharing

```python
ctx = UserCtx()
agent.register_context_provider("user", ctx)

# Update before running
ctx.name = "Alice"; ctx.role = "admin"
agent.run(...)

# Inspect / unregister
"user" in agent.system_prompt_generator.context_providers
agent.unregister_context_provider("user")
```

One provider instance can be registered with many agents — updates propagate everywhere:

```python
shared = SessionCtx()
agent_a.register_context_provider("session", shared)
agent_b.register_context_provider("session", shared)
shared.set("locale", "en-GB")   # visible to both agents
```

## Patterns

**Time**

```python
from datetime import datetime, timezone

class TimeCtx(BaseDynamicContextProvider):
    def __init__(self):
        super().__init__(title="Current Time")

    def get_info(self) -> str:
        return datetime.now(timezone.utc).isoformat()
```

**RAG retrieval**

Update outside the provider, read inside.

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
        return "\n\n".join(
            f"[{d['source']}] {d['content']}" for d in self.docs
        )

# Orchestration
rag.set(vector_db.search(query, k=4))
agent.run(query_input)
```

**Session**

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

**Cached database schema**

Providers are called on every run, so cache slow sources:

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

## Dynamic updates inside a run loop

Context providers are most powerful when mutated between calls. A research loop updates the `ScrapedContentContextProvider` after each scrape, then re-runs the agent with the new facts in prompt — see `atomic-examples/deep-research/deep_research/context_providers.py` and the main loop in `deep_research/main.py` for a working reference.

```python
ctx = ScrapedContentCtx()
agent.register_context_provider("scraped", ctx)

while not done:
    ctx.set(scrape(next_url))          # mutate provider state
    decision = agent.run(NextStep(...))  # provider renders fresh context
    done = decision.stop
```

The key property: register the provider once, mutate its fields between `agent.run()` calls, and the rendered prompt picks up the change automatically.

## Async data sources

`get_info()` is synchronous. When the data source is async, refresh before calling the agent:

```python
class AsyncCtx(BaseDynamicContextProvider):
    def __init__(self):
        super().__init__(title="Async Data")
        self._cached = ""

    async def refresh(self) -> None:
        self._cached = format(await fetch_remote())

    def get_info(self) -> str:
        return self._cached

await ctx.refresh()
await agent.run_async(input_data)
```

## Common mistakes

- Slow I/O in `get_info()` — runs on every `agent.run()`.
- Returning a non-string from `get_info()` — raises at prompt time.
- Forgetting to register the provider after creating it — context silently missing.
- Duplicate titles across providers — they collide in the rendered prompt; use distinct titles.
- Storing secrets in the context — they end up in every LLM request. Inject only what the model needs to reason about.
