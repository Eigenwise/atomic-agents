# Directory Structure

*Last Updated: 2026-07-18*

## Root Layout

```
atomic-agents/                  # repo root (uv workspace)
├── atomic-agents/              # CORE framework project (PyPI: atomic-agents)
│   └── atomic_agents/          #   import package
│       ├── agents/             #     AtomicAgent, AgentConfig
│       ├── base/               #     BaseIOSchema, BaseTool, BaseResource, BasePrompt
│       ├── context/            #     SystemPromptGenerator, BaseChatHistory/ChatHistory, context providers
│       ├── connectors/mcp/     #     Model Context Protocol integration
│       └── utils/              #     token counter, tool-message formatting
│   └── tests/                  #   pytest suite (agents/, base/, context/, connectors/, utils/)
├── atomic-assembler/           # Textual TUI (`atomic` command) to install forge tools
│   └── atomic_assembler/       #   main.py, app.py, screens/, widgets/, utils.py, constants.py
├── atomic-forge/               # library of standalone tools (NOT a package)
│   ├── tools/<tool>/           #   one folder per tool: tool/<tool>.py, tests/, pyproject.toml
│   └── guides/                 #   tool authoring guides (e.g. tool_structure.md)
├── atomic-examples/            # 16 runnable example apps (each its own project)
├── claude-plugin/atomic-agents/ # AI-assistant plugin: 7 skills + 2 subagents (Claude Code plugin,
│                               #   also installable cross-tool via `npx skills add eigenwise/atomic-agents`)
├── .claude-plugin/             # marketplace.json — plugin marketplace manifest (drives npx skills discovery)
├── docs/                       # Sphinx + MyST documentation (api/, guides/, examples/)
├── guides/                     # DEV_GUIDE.md and contributor guides
├── scripts/                    # generate_llms_files.py (llms.txt index + llms-*.txt bundles)
├── pyproject.toml              # package metadata, deps, [tool.black], uv workspace
├── context7.json               # Context7 indexing config + v2 API rules for AI assistants
├── build_and_deploy.ps1        # version bump + uv build/publish
├── AGENTS.md                   # the project's own design philosophy (imported by CLAUDE.md)
└── README.md
```

## Key Directories

### `atomic-agents/atomic_agents/` (core)
The framework itself. `agents/atomic_agent.py` is the heart (`AtomicAgent`). `base/` holds the
Pydantic-based contracts every agent/tool/resource/prompt implements. `context/` assembles system
prompts and stores conversation history. `connectors/mcp/` bridges to MCP servers. `utils/` does
token accounting via LiteLLM.

### `atomic-assembler/atomic_assembler/`
A Textual terminal UI launched by the `atomic` command (`main.py:main`). `app.py` routes between
`screens/` (main menu, tool explorer, file picker, README viewer); `utils.py` clones the GitHub repo
and copies a selected tool into the user's project.

### `atomic-forge/tools/`
13 self-contained tools (`arxiv_search`, `calculator`, `tavily_search`, `weather`,
`webpage_scraper`, `wikipedia_search`, …). Each tool folder contains `tool/<name>.py` (Input/Output
`BaseIOSchema` + a `BaseToolConfig` + a `BaseTool` subclass), `tests/`, and its own
`pyproject.toml`/`requirements.txt`. Tools are copied into user projects, not pip-installed.

### `atomic-examples/`
16 standalone example apps (`quickstart`, `rag-chatbot`, `deep-research`, `web-search-agent`,
`mcp-agent`, `fastapi-memory`, `persistent-memory`, `youtube-summarizer`, …), each with its own
`pyproject.toml`. These are excluded from the workspace build.

### `docs/` and `guides/`
`docs/` is a Sphinx + MyST site (`api/` reference, `guides/`, `examples/`, `conf.py`), deployed to
GitHub Pages. `guides/DEV_GUIDE.md` is the contributor setup/workflow guide.
