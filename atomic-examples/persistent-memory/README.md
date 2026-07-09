# Persistent Memory Example

This example demonstrates how to plug a custom, persistent memory backend into Atomic Agents so a conversation survives across separate process runs (cross-session recall).

It shows:

- Subclassing `ChatHistory` to add persistence, following the pluggable `BaseChatHistory` extension point
- A `SQLiteChatHistory` backend built with **zero extra dependencies**, using only the Python standard library's `sqlite3` module
- Persisting on every `add_message` call and rehydrating the full history on startup via the inherited `dump()`/`load()`

See the Memory guide's ["Writing a Custom Memory Backend"](../../docs/guides/memory.md#writing-a-custom-memory-backend) section for the full write-up of this pattern.

## Getting Started

1. Clone the main Atomic Agents repository:

   ```bash
   git clone https://github.com/eigenwise/atomic-agents
   ```

2. Navigate to this example's directory:

   ```bash
   cd atomic-agents/atomic-examples/persistent-memory
   ```

3. Install the dependencies using uv:

   ```bash
   uv sync
   ```

4. Set your `OPENAI_API_KEY` environment variable (or put it in a `.env` file in this directory).

## Running the Example

```bash
uv run python persistent_memory/main.py
```

Chat with the agent, then exit with `/exit`. A `chat_history.db` file is created in the current directory.

**Run it a second time** and the agent picks up right where you left off: it prints the number of restored messages at startup and remembers everything from the previous run, because the conversation was persisted to `chat_history.db` under the `demo` session ID.

## How It Works

`SQLiteChatHistory` (in `persistent_memory/main.py`) subclasses `ChatHistory` instead of reimplementing the memory contract from scratch:

- `__init__` opens (or creates) a SQLite database with a single `sessions(session_id TEXT PRIMARY KEY, data TEXT)` table, and loads any existing row for the given `session_id` via the inherited `load()`.
- `add_message` calls `super().add_message(...)` to reuse `ChatHistory`'s turn handling, then saves the entire history to SQLite via `self.dump()`.

Because `dump()`/`load()` already handle serialization, the persistence layer only has to move a single JSON string in and out of storage. The same pattern works for any store (Redis, Postgres, a hosted memory service) by swapping out what `_save`/`__init__` talk to.
