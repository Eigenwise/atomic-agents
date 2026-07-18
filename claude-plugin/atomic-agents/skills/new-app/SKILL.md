---
name: new-app
description: Scaffold a new Atomic Agents project from scratch — create the directory, `pyproject.toml`, env file, first agent, and a runnable entry point. Use when the user asks to start a new atomic-agents project from scratch, says "scaffold" / "new project" / "start from zero", or runs `/atomic-agents:new-app`.
disable-model-invocation: true
argument-hint: "[project-name]"
---

# New Atomic Agents Project

Scaffold a fresh Atomic Agents project. The result is a single-package Python project with one working agent, one schema pair, a provider-wrapped client, and a runnable `main.py`.

This skill is opinionated. Produce a complete, tested skeleton the user can run immediately.

## Phase 1 — Interrogate

Ask these questions in one message, not one-at-a-time. Skip any the user already answered (including via `$ARGUMENTS`).

1. **Project name** — used as both directory name and package name. Default from `$ARGUMENTS` if provided. Normalize to `kebab-case` for the directory and `snake_case` for the package.
2. **LLM provider** — OpenAI / Anthropic / Groq / Ollama / Gemini / OpenRouter / MiniMax. Default: OpenAI.
3. **Agent type** — a rough one-liner. Shapes the default `SystemPromptGenerator` content and the starter schema pair. Defaults to a generic chat agent.
4. **Tooling** — `uv` (default, because the repo uses uv) or `pip + venv`.

Do not ask about project layout, Python version, or dependency list. Pick them.

## Phase 2 — Confirm the plan

State the plan in one short block and wait for a yes. Include:

- Directory: `<project-name>/`
- Package: `<project_name>/`
- Python: `>=3.12` (Atomic Agents uses PEP 695 generics)
- Dependencies: `atomic-agents>=2.7`, `instructor[<provider-extra>]>=1.14`, `python-dotenv`, `rich`
- Dev dependencies: `pytest`, `pytest-asyncio`, `ruff`
- First agent: `<agent-type>` — uses `BasicChatInputSchema`/`BasicChatOutputSchema` unless the agent type calls for custom schemas
- Default model for the chosen provider (see `framework/references/providers.md`)
- Entry point: `main.py` with a REPL

## Phase 3 — Scaffold

Create files in this order. Verify each step before proceeding.

### Directory and package

```
<project-name>/
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
├── AGENTS.md
├── CLAUDE.md
└── <project_name>/
    ├── __init__.py
    └── main.py
```

### `pyproject.toml`

Use the template from `framework/references/project-structure.md`, substituting the chosen provider extra and project name.

### `.env.example`

Include the provider's API-key variable with a placeholder. Never the real key.

### `.gitignore`

Use the template from `framework/references/project-structure.md`.

### `<project_name>/main.py`

Produce a runnable REPL. Load `.env`, instantiate the provider client per `framework/references/providers.md`, build an agent, wire a `ChatHistory` with a seed assistant message, loop on `console.input(...)`.

For the agent itself, follow the workflow from the `atomic-agents:create-atomic-agent` skill — same canonical imports, same per-provider `mode` matrix, same `SystemPromptGenerator` shape.

When a custom agent type was requested, build custom `InputSchema` / `OutputSchema` subclasses with field `description=` populated, following the `atomic-agents:create-atomic-schema` skill. Otherwise use `BasicChatInputSchema` / `BasicChatOutputSchema`.

Always use the canonical imports:

```python
from atomic_agents import (
    AtomicAgent, AgentConfig,
    BasicChatInputSchema, BasicChatOutputSchema,
)
from atomic_agents.context import ChatHistory, SystemPromptGenerator
from instructor import Mode
```

Per-provider AgentConfig knobs — match the Instructor factory mode on `AgentConfig.mode`:

- **OpenAI**: defaults work. Omit `mode` (or set `Mode.TOOLS`).
- **Anthropic**: `mode=Mode.TOOLS`; include `max_tokens` in `model_api_parameters`.
- **Groq / Ollama / MiniMax**: `mode=Mode.JSON` (Instructor factory also uses `Mode.JSON`).
- **Gemini**: `assistant_role="model"` and `mode=Mode.GENAI_TOOLS` (Instructor factory uses `Mode.GENAI_TOOLS`).
- **OpenRouter**: `mode=Mode.TOOLS`.

### `README.md`

Short. Include: what the project is, how to install (`uv sync` or `pip install -e .[dev]`), how to set the API key (`cp .env.example .env` and edit), how to run (`uv run python -m <project_name>.main` or equivalent).

### `AGENTS.md` and `CLAUDE.md`

Every scaffolded project ships agent instructions so any coding assistant (Cursor, Codex, Copilot, Gemini CLI, ...) knows the framework conventions from the first commit. `CLAUDE.md` contains exactly one line — `@AGENTS.md` — so Claude Code reads the same file without duplication.

`AGENTS.md` template (substitute project specifics):

```markdown
# <Project Name>

<One-line description from the agent-type answer.>

Built with [Atomic Agents](https://github.com/eigenwise/atomic-agents) — a schema-driven
framework on Instructor + Pydantic. Docs for LLMs:
https://eigenwise.github.io/atomic-agents/llms.txt

## Framework conventions

- Import from the top-level package: `from atomic_agents import AtomicAgent, AgentConfig,
  BaseIOSchema, BaseTool`; context pieces from `atomic_agents.context`.
- Agents are `AtomicAgent[InputSchema, OutputSchema](config=AgentConfig(...))` — the type
  parameters carry runtime information, keep them accurate.
- The LLM client must be wrapped with Instructor before it goes into `AgentConfig.client`.
- Every `BaseIOSchema` subclass needs a non-empty docstring and `Field(..., description=...)`
  on each field — both flow into the LLM prompt.
- Provider knobs (`temperature`, `max_tokens`, ...) go in `AgentConfig.model_api_parameters`.
- Provider: <chosen provider>. <Provider-specific line from the matrix below, if any.>

## Commands

- Install: `uv sync` (or the pip equivalent chosen at scaffold time)
- Run: `uv run python -m <project_name>.main`
- Test: `uv run pytest`
```

Provider-specific lines for the template: Anthropic → "Requires `max_tokens` in `model_api_parameters`; `mode=Mode.TOOLS`." Gemini → "`assistant_role='model'` and `mode=Mode.GENAI_TOOLS`." Groq/Ollama/MiniMax → "`mode=Mode.JSON` on both the Instructor factory and `AgentConfig`." OpenAI/OpenRouter → omit.

## Phase 4 — Install and smoke-test

Execute the install step:

- **uv**: `uv sync`
- **pip**: `python -m venv .venv && .venv/bin/pip install -e ".[dev]"` (Windows: `.venv\Scripts\pip`)

Verify imports without a live API key:

```bash
uv run python -c "from <project_name>.main import agent; print('ok')"
```

If that works, the scaffold is sound. Tell the user to drop their key into `.env` and run the REPL.

## Phase 5 — Hand off

After scaffolding, tell the user:

1. How to set their key (`cp .env.example .env`).
2. How to run (`uv run python -m <project_name>.main`).
3. Next steps, picked from:
   - Replace the starter schemas with domain-specific ones — use the `atomic-agents:create-atomic-schema` skill.
   - Add another agent — use the `atomic-agents:create-atomic-agent` skill.
   - Add a tool — use the `atomic-agents:create-atomic-tool` skill.
   - Add a context provider (time, user, RAG, session) — use the `atomic-agents:create-atomic-context-provider` skill.
   - Split into multiple agents — see `framework/references/orchestration.md`.
4. A pointer to `framework` (auto-triggered) and `review` (auto-triggered before commit).

## Constraints

- Never commit `.env`. Only `.env.example`.
- Never install anything globally. Use the project venv.
- Never pick an old model. Default to current generation: OpenAI `gpt-5-mini`, Anthropic `claude-haiku-4-5`, Groq `llama-3.3-70b-versatile`, Ollama `llama3.1`, Gemini `gemini-2.5-flash`.
- Never hand-roll what `framework/references/project-structure.md` already templates.
