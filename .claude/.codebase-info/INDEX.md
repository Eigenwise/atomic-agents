# Codebase Map — Atomic Agents

*Last Updated: 2026-07-18*

Atomic Agents is a lightweight, modular Python framework for building agentic AI applications as
composable, schema-driven building blocks (built on Instructor + Pydantic). This repository is a
`uv`-workspace **monorepo**: the core framework, a TUI tool installer, a tool library, and examples.

**Stack:** Python ≥3.12 · Instructor · Pydantic v2 · LiteLLM · MCP · Textual · uv + Hatchling
**Shape:** Monorepo — `atomic-agents/` (core lib) · `atomic-assembler/` (CLI) · `atomic-forge/` (tools) · `atomic-examples/` (examples)
**Package:** `atomic-agents` v2.9.1 on PyPI · core import package is `atomic_agents`

## Documents

| Document | What's inside |
|----------|---------------|
| [architecture.md](./architecture.md) | Framework + monorepo architecture, the agent run lifecycle |
| [tech-landscape.md](./tech-landscape.md) | Languages, libraries, build/release, infra |
| [directory-structure.md](./directory-structure.md) | Annotated monorepo tree |
| [entry-points.md](./entry-points.md) | Library API, the `atomic` CLI, examples |
| [modules.md](./modules.md) | Core package modules + the monorepo subprojects |
| [communication.md](./communication.md) | LLM providers, MCP, hooks, tool fetching |
| [dependencies.md](./dependencies.md) | Categorized packages |
| [patterns.md](./patterns.md) | Atomicity, schema-driven I/O, context providers, testing |
| [coding-style.md](./coding-style.md) | Formatting, linting, naming conventions |
| [onboarding.md](./onboarding.md) | Setup with uv, tests, docs, common tasks |

## How to use this map

- New here? Read `onboarding.md`, then `architecture.md`.
- Building or maintaining an agent? `patterns.md` + `modules.md` + `entry-points.md`.
- Adding a tool? `modules.md` (atomic-forge section) + `patterns.md`.
- The project's own design *philosophy* lives in `AGENTS.md` at the repo root (imported by
  `CLAUDE.md`); this map is its structural, navigational complement.

## Keeping this map current

After changes to the framework's public API, the monorepo layout, dependencies, the agent
run/context pipeline, MCP/provider integration, or the forge tool set, refresh the affected docs with
`/codebase-mapper:update-codebase-map`. (There are intentionally no `database.md` or `docker.md`
docs — the framework has neither.)
