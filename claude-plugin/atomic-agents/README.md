# Atomic Agents Plugin for Claude Code

A Claude Code plugin that gives Claude deep, just-in-time knowledge of the [Atomic Agents](https://github.com/eigenwise/atomic-agents) Python framework. When you work on an Atomic Agents codebase, Claude picks up the right patterns for schemas, agents, tools, context providers, prompts, orchestration, and provider wiring — without you having to paste documentation or repeat conventions.

## What you get

Two skills + two subagents, loaded on demand:

- **`framework` skill** — auto-triggered when Claude sees atomic-agents code. Orients Claude on the framework and exposes eleven focused reference files (schemas, agents, tools, context providers, prompts, orchestration, memory, hooks, providers, project structure, testing). Progressive disclosure keeps the parent context lean.
- **`new-app` skill** — slash-invokable scaffolder (`/atomic-agents:new-app`). Four short questions and produces a runnable project with the right `pyproject.toml`, environment file, and first agent.
- **`atomic-explorer` subagent** — auto-triggered (or `Task`-invoked) when you ask to explore, map, or understand an existing atomic-agents codebase. Reads the project in isolated context and returns a compact architecture map (agents, tools, schemas, context providers, orchestration, essential-reading list) without polluting the parent thread with every file it had to open to build that map.
- **`atomic-reviewer` subagent** — auto-triggered (or `Task`-invoked) when you ask for a review, audit, or check of atomic-agents code. Runs in isolated context with read-only tools so the file-exploration load never pollutes the parent thread. Focuses only on framework-specific concerns (BaseIOSchema invariants, Instructor wrapping, per-provider role and mode, context-provider I/O hygiene, orchestration hazards, common API misuses). Returns a single confidence-filtered structured report. Complements generic code review; does not replace it.

This follows the hybrid pattern Anthropic themselves ship: reference skills for knowledge that should inform the main thread, specialist subagents for discrete read-heavy tasks whose output is a summary.

## Install

### Recommended — from the Eigenwise marketplace

```
/plugin marketplace add eigenwise/atomic-agents
/plugin install atomic-agents@eigenwise
```

This path gets you the latest version immediately. We control the marketplace, so pushes to `main` in this repo reach users on the next `/plugin marketplace update` (and auto-update is on by default for git-hosted marketplaces you've explicitly added).

### From Anthropic's community marketplace

Once the plugin is approved in Anthropic's public community directory, add that marketplace and install from it:

```
/plugin marketplace add anthropics/claude-plugins-community
/plugin install atomic-agents@claude-community
```

The community directory is submission-only — entries are submitted through the [plugin directory submission form](https://clau.de/plugin-directory-submission) (pull requests against the catalog repo are closed automatically). After approval, CI re-pins the entry to new commits on this repo's `main` as they land, so the Eigenwise marketplace above always tracks the same source.

### Team projects (declarative)

To require this plugin for everyone who opens a given repo, add it to the project's `.claude/settings.json`:

```json
{
  "extraKnownMarketplaces": {
    "eigenwise": {
      "source": { "source": "github", "repo": "eigenwise/atomic-agents" }
    }
  },
  "enabledPlugins": {
    "atomic-agents@eigenwise": true
  }
}
```

Claude Code prompts each teammate to install on first trust of the folder.

### Other coding assistants (Cursor, Copilot, Codex, Windsurf, Gemini CLI, ...)

The skills follow the cross-tool [agent skills](https://agents.md/) format, so any skills-compatible assistant can install them without the Claude Code plugin system:

```bash
npx skills add eigenwise/atomic-agents
```

This installs the six skills (framework guide, four create-* workflows, new-app scaffolder). The `atomic-explorer` and `atomic-reviewer` subagents are Claude Code plugin features and don't travel this path.

### Local dev

```bash
claude --plugin-dir /path/to/atomic-agents/claude-plugin/atomic-agents
```

### Validate

```bash
/plugin validate /path/to/claude-plugin/atomic-agents
```

## How it works

This plugin follows Anthropic's 2026 skill-authoring conventions:

- Each skill is a folder with a `SKILL.md` file (≤500 lines) plus an optional `references/` directory.
- YAML frontmatter has two fields: `name` and `description`. The description is what Claude reads to decide when the skill applies.
- Reference files are one level deep from `SKILL.md`, each ≤ a few pages. Claude loads only the ones relevant to the current task.
- No README, CHANGELOG, or auxiliary docs inside skill folders — those live at the plugin root.

Read the `framework` skill's `SKILL.md` if you want to see the routing table and minimum-viable-agent template up front.

## Typical flows

**Writing new atomic-agents code.** Open your project. Claude notices imports from `atomic_agents`, loads the `framework` skill, and pulls in `references/schemas.md`, `references/agents.md`, etc. as the conversation progresses.

**Understanding a codebase you inherited.** Ask "help me understand how the agents in this project work" or "map this codebase." The `atomic-explorer` subagent fires, reads the relevant files in its own context, and returns a compact architecture map with file:line references — the parent thread skips the file-by-file slog.

**Before committing.** Ask "review my changes for atomic-agents issues." The `atomic-reviewer` subagent fires (or Claude invokes it via the `Task` tool), reads the diff in its own context, and returns a confidence-filtered list of framework-specific issues with ready-to-apply fixes.

**Starting from nothing.** Run `/atomic-agents:new-app my-project`. Four questions, one scaffolded project, one runnable agent.

## Framework compatibility

- Python 3.12+ (PEP 695 generics).
- Atomic Agents 2.7+.
- Instructor 1.14+ with provider extras (`instructor[openai]`, `instructor[anthropic]`, `instructor[groq]`, `instructor[google-genai]`).
- Pydantic 2.

## Development

Edit files in place and re-run `/reload-plugins` in Claude Code to pick up changes without restarting. `--plugin-dir` loads take precedence over installed marketplace copies for the current session.

Skill-authoring guidance comes from Anthropic's [skill-creator](https://github.com/anthropics/skills) and [skill authoring best practices](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/best-practices). Keep descriptions in third person with concrete trigger terms, keep SKILL.md bodies under 500 lines, and push detail into `references/` files so progressive disclosure works.

## License

MIT — see `LICENSE`.

## Links

- Framework: https://github.com/eigenwise/atomic-agents
- Examples: https://github.com/eigenwise/atomic-agents/tree/main/atomic-examples
- Changelog: `CHANGELOG.md`

Built by [Eigenwise](https://github.com/eigenwise).
