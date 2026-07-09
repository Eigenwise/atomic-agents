# Key Modules

*Last Updated: 2026-07-05*

## Core framework — `atomic-agents/atomic_agents/`

### agents
- **Location:** `atomic-agents/atomic_agents/agents/atomic_agent.py`
- **Purpose:** The `AtomicAgent[InputSchema, OutputSchema]` class and `AgentConfig`. Orchestrates the
  run lifecycle (sync / stream / async), context-provider registration, hook registration, and
  context-window trimming.
- **Exposes:** `AtomicAgent`, `AgentConfig`, `BasicChatInputSchema`, `BasicChatOutputSchema`.
- **Depends on:** `base/`, `context/`, `utils/`, `instructor`.

### base
- **Location:** `atomic-agents/atomic_agents/base/`
- **Purpose:** The typed contracts everything else implements.
- **Key files:** `base_io_schema.py` (`BaseIOSchema` — Pydantic base; non-empty docstring enforced and
  used as the schema description), `base_tool.py` (`BaseTool[In, Out]`, `BaseToolConfig`),
  `base_resource.py` (`BaseResource`), `base_prompt.py` (`BasePrompt`).
- **Exposes via package root:** `BaseIOSchema`, `BaseTool`, `BaseToolConfig`.

### context
- **Location:** `atomic-agents/atomic_agents/context/`
- **Purpose:** System-prompt assembly and conversation memory.
- **Key files:** `system_prompt_generator.py` (`SystemPromptGenerator`, `BaseDynamicContextProvider`),
  `base_chat_history.py` (`BaseChatHistory` — interface-only ABC declaring the memory contract
  `AtomicAgent` depends on; the pluggable seam for custom/persistent backends),
  `chat_history.py` (`ChatHistory`, `Message` — the built-in in-memory implementation of
  `BaseChatHistory`: multimodal Image/Audio/PDF, turn grouping, `dump()`/`load()` serialization).
- **Note:** `AgentConfig.history` is typed to `BaseChatHistory`, so any conforming backend drops in.

### connectors/mcp
- **Location:** `atomic-agents/atomic_agents/connectors/mcp/`
- **Purpose:** Model Context Protocol integration — expose MCP server tools/resources/prompts as
  Atomic Agents components.
- **Exposes:** `MCPFactory`, `MCPDefinitionService`, `SchemaTransformer`,
  `fetch_mcp_tools` / `fetch_mcp_resources` / `fetch_mcp_prompts` (sync + async).

### utils
- **Location:** `atomic-agents/atomic_agents/utils/`
- **Key files:** `token_counter.py` (`TokenCounter` via LiteLLM; `get_context_token_count`),
  `format_tool_message.py`.

## Subprojects

### atomic-assembler (CLI)
- **Location:** `atomic-assembler/atomic_assembler/`
- **Purpose:** Textual TUI to browse and install forge tools into a user project.
- **Key files:** `main.py` (`main()`; argparse `--enable-logging`, `--version`), `app.py`
  (`AtomicAssembler(App)`), `screens/` (`main_menu`, `atomic_tool_explorer`, `file_explorer`,
  `tool_info_screen`), `widgets/`, `utils.py` (`GithubRepoCloner`, `AtomicToolManager`),
  `constants.py` (GitHub URL, `TOOLS_SUBFOLDER`).

### atomic-forge (tools)
- **Location:** `atomic-forge/tools/`
- **Purpose:** 13 standalone tools, each an independent mini-project following the `BaseTool` pattern,
  copied into user projects by the assembler (build files such as `pyproject.toml` / `requirements.txt`
  / `uv.lock` are skipped on copy).
- **Authoring guide:** `atomic-forge/guides/tool_structure.md`.

### atomic-examples
- **Location:** `atomic-examples/`
- **Purpose:** 16 runnable reference apps. Catalog in `entry-points.md`.
