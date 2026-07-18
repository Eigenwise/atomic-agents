# Changelog

All notable changes to the Atomic Agents Claude Code Plugin will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.2.0] - 2026-07-18

### Added

- **`troubleshoot` skill** — diagnostic workflow for the "my agent is broken" moment. Symptom table keyed on the framework's real error messages (docstring enforcement, `AgentConfig.client` validation, provider role/mode errors, `HTTP_STREAM`, history misuse), each with a mechanical fix and a re-run verification step. Escalation map into the `framework` reference files; `framework` routes concrete errors here.
- **`new-app` emits `AGENTS.md` + `CLAUDE.md`** into every scaffolded project — framework conventions any coding assistant reads, with `CLAUDE.md` as a one-line `@AGENTS.md` import (the documented, Windows-safe pattern). Template includes per-provider quirk lines.
- Cross-tool install path documented: `npx skills add eigenwise/atomic-agents` works for Cursor, Copilot, Codex, Windsurf, Gemini CLI, and other skills-compatible assistants (discovery via the repo's `.claude-plugin/marketplace.json`).

### Changed

- Rebranded BrainBlend → Eigenwise: marketplace name, URLs, and install docs now point at the `eigenwise` GitHub home. Install via `/plugin marketplace add eigenwise/atomic-agents`.
- `new-app` MiniMax default model updated to MiniMax M3.

### Fixed

- Quoted `argument-hint` YAML values in skill frontmatter so GitHub Copilot CLI ≥1.0.65 loads all skills ([#265](https://github.com/eigenwise/atomic-agents/pull/265)).
- The repo's maintainer-only `release` skill no longer leaks into `npx skills` discovery (`metadata.internal: true`).

## [2.1.0] - 2026-04-29

### Added

- Four action-oriented creation skills: `create-atomic-schema`, `create-atomic-agent`, `create-atomic-tool`, `create-atomic-context-provider`. Each walks a clarify → write → verify → hand-off workflow for its component type, auto-triggering on the matching phrasing ("create a schema", "add a tool", ...). The `framework` skill and `new-app` scaffolder route to them instead of duplicating the material.

### Fixed

- `atomic-reviewer` no longer flags current-generation model names as errors.

## [2.0.1] - 2026-04-16

Bug-fix release addressing two recurring false positives from `atomic-reviewer` reported in [issue #238](https://github.com/eigenwise/atomic-agents/issues/238).

### Fixed

- **Reviewer no longer flags raw provider-SDK use for embeddings, image generation, audio, or moderation as a framework violation.** The "Instructor-wrapped client" rule now carries an explicit scope note: it applies only to clients passed to `AgentConfig.client`. The framework doesn't expose embeddings/image/audio/moderation APIs, so using the provider SDK directly for those capabilities is correct usage, not a rule break. Scope note added to `agents/atomic-reviewer.md` section 2 and `skills/framework/SKILL.md` anti-patterns.
- **Reviewer no longer confabulates a bug for `ChatHistory.initialize_turn()`.** The method only rotates `current_turn_id` (a UUID) — it does not append a system prompt, add a message, or mutate `history`. Added an explicit "methods that are NOT misuses" block to `agents/atomic-reviewer.md` section 9 documenting both non-misuses so Claude doesn't invent them.

## [2.0.0] - 2026-04-16

Full rewrite to match 2026 Claude Code plugin conventions. Reference knowledge lives in skills with progressive-disclosure references; code review lives in an isolated-context subagent. This mirrors Anthropic's own plugin pattern (`plugin-dev`, `feature-dev`, `pr-review-toolkit`).

### Changed

- Replaced three over-broad subagents (`atomic-architect`, `atomic-explorer`, `schema-designer`) with a single `framework` skill that exposes progressive-disclosure references. The retained subagent (`atomic-reviewer`) was rewritten; see below.
- Retired the `/atomic-create`, `/atomic-agent`, `/atomic-tool` slash commands. The `framework` skill loads automatically when the user works in atomic-agents code; the `new-app` skill handles greenfield scaffolding via `/atomic-agents:new-app`.
- `skills/` directories now follow the standard Anthropic layout: a top-level `SKILL.md` plus a `references/` subdirectory loaded on demand.
- Skill descriptions rewritten in third person with concrete trigger terms, per Anthropic's skill authoring best practices.
- Plugin manifest trimmed to the core fields Anthropic's own plugins use (name, description, author, homepage, repository, license, version).

### Retained / added as subagents

Two tasks met the subagent rubric — read-heavy work whose output is a summary the parent thread benefits from receiving without the exploration load:

- **`atomic-reviewer`** — code review runs in an isolated subagent context with read-only tools (`Glob`, `Grep`, `LS`, `Read`, `NotebookRead`, `TodoWrite`). This matches Anthropic's canonical example in their subagents documentation (`code-reviewer` with no `Write`/`Edit`) and the pattern used in `plugin-dev`, `feature-dev`, and `pr-review-toolkit`. Keeping review in a subagent prevents the file-exploration load from polluting the parent conversation, enables tool-level restriction, and leaves the door open to the parallel-review-lens pattern later. Rewritten for the 2026 API (top-level imports, PEP 695 `BaseTool` generics, `MCPTransportType.HTTP_STREAM`, five hook events, etc.).
- **`atomic-explorer`** — architectural mapping of existing atomic-agents codebases runs in an isolated subagent with the same read-only tool set. Catalogs agents, tools, schemas, context providers, and orchestration patterns; traces data flow; returns a compact summary with file:line references and an essential-reading list. Analogous to Anthropic's `code-explorer` in `feature-dev`. For small projects (a single `main.py` + one or two agents), the main thread handles exploration directly.

The other three original subagents (`atomic-architect`, `schema-designer`, and the legacy scaffolder) were not restored: architect and schema-designer produce *artifacts* the user wants in the parent thread to iterate on, which is anti-pattern for subagents. Scaffolding is now the `new-app` slash-invokable skill.

### Fixed

- Corrected all framework imports to the current top-level paths (`from atomic_agents import ...`, `from atomic_agents.context import ...`). Previous version referenced retired `atomic_agents.lib.*` and `atomic_agents.agents.base_agent` paths.
- Updated `BaseTool` examples to use PEP 695 generics (`class MyTool(BaseTool[In, Out])`) instead of the removed `input_schema = ...` / `output_schema = ...` class-attribute pattern.
- Updated default model references to current-generation models.
- Documented Gemini `assistant_role="model"` requirement, Anthropic `max_tokens` requirement, and per-provider Instructor `Mode` — previously missing or inaccurate.

### Added

- `framework/references/testing.md` — new dedicated testing guide (pytest, async, streaming, integration gates).
- `framework/references/providers.md` — covers OpenAI, Anthropic, Groq, Ollama, Gemini, OpenRouter, MiniMax with correct modes and roles.
- `framework/references/memory.md` — `ChatHistory` turn model, persistence, overflow, multi-agent memory patterns (shared, independent, agent-to-agent messaging addressing GH issue #58, supervisor-worker).
- `framework/references/hooks.md` — the five events, telemetry, validation-error inspection, retry patterns, production logging.
- Orchestration guide includes the search+execute pattern for large tool surfaces, with per-pattern pointers to working `atomic-examples/` projects.
- Inline links from reference files to real example projects (`hooks-example`, `deep-research`, `progressive-disclosure`, `orchestration-agent`, `mcp-agent`, `quickstart`).
- `tools.md` distinguishes in-project tools from distributable Forge tools and links to `atomic-forge/guides/tool_structure.md`.
- MCP interop coverage in `framework/references/tools.md`.

### Fixed (post-review)

- `agents.md`: `ChatHistory.load(saved)` is an instance method, not a classmethod. `dump()` returns a JSON **string**, not "JSON-serializable data". Added missing `completion:kwargs` hook event. `TokenCountResult` now lists all seven fields (`total`, `system_prompt`, `history`, `tools`, `model`, `max_tokens`, `utilization`).
- `tools.md`: `MCPTransportType.STREAMABLE_HTTP` does not exist — correct value is `HTTP_STREAM` (alongside `SSE` and `STDIO`). Async tools expose `run_async`, not `arun`.
- `new-app/SKILL.md`: explicit `AgentConfig.mode` per provider (must match the Instructor factory `mode`).
- All SKILL.md bodies and reference files converted from second-person "you" to imperative/third-person per Anthropic's skill-authoring guide (quoted prompt strings and template placeholders retained as-is).
- `framework` skill description trimmed from 574 chars to 343 chars using a hybrid short-lead + compact-trigger form for better discoverability.

### Removed

- `agents/` directory — no subagents.
- `commands/` directory — slash commands are now skills with `disable-model-invocation: true` where user-only invocation is desired.
- Per-skill `examples/` directories — Anthropic's skill-authoring guide recommends `references/` over `examples/` for progressive disclosure.

## [1.0.0] - 2025-12-21

Initial release with subagents (`atomic-explorer`, `atomic-architect`, `atomic-reviewer`, `schema-designer`), three slash commands (`/atomic-create`, `/atomic-agent`, `/atomic-tool`), and six progressive-disclosure skills.
