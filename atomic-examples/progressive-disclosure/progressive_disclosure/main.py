# pyright: reportInvalidTypeForm=false
"""Progressive Disclosure Demo with Multiple MCP Servers.

This script demonstrates Anthropic's "progressive disclosure" pattern where
MCP tools are discovered on-demand rather than loaded all at once.

We have THREE MCP servers:
- math-server: 8 arithmetic tools (add, subtract, multiply, divide, power, sqrt, modulo, abs)
- text-server: 8 text manipulation tools (uppercase, lowercase, reverse, word_count, etc.)
- data-server: 8 list/data tools (sort, filter, sum, average, min, max, unique)

Total: 24 tools across 3 servers.

The progressive disclosure pattern:
1. Tool Finder Agent searches for relevant tools based on user query
2. Only selected tools (typically 2-5) are loaded into the Main Orchestrator
3. Result: ~90% reduction in context window usage

Without progressive disclosure: All 24 tool schemas in context (~12,000 tokens)
With progressive disclosure: Only 2-5 relevant tools (~1,000 tokens)
"""

import asyncio
import os
import shlex
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import List, Type, Dict

import instructor
import openai
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from atomic_agents.connectors.mcp import (
    fetch_mcp_tools,
    MCPTransportType,
)
from atomic_agents.base.base_tool import BaseTool

from progressive_disclosure.registry.tool_registry import ToolRegistry, MCPToolDefinition
from progressive_disclosure.agents.tool_finder_agent import (
    create_tool_finder_agent,
    run_tool_finder,
)
from progressive_disclosure.agents.orchestrator_agent import (
    OrchestratorFactory,
    execute_orchestrator_loop,
    execute_orchestrator_loop_parallel,
)


########################
# CONFIGURATION        #
########################
@dataclass
class ServerConfig:
    """Configuration for an MCP server."""

    name: str
    command: str
    category: str  # For tool categorization


@dataclass
class ProgressiveDisclosureConfig:
    """Configuration for the Progressive Disclosure demo."""

    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    finder_model: str = "gpt-5.6-terra"  # Lightweight model for tool discovery
    orchestrator_model: str = "gpt-5.1"  # More capable model for execution
    parallel_execution: bool = True  # Enable parallel tool execution

    # Three MCP servers demonstrating multi-server progressive disclosure
    servers: List[ServerConfig] = field(
        default_factory=lambda: [
            ServerConfig(name="math-server", command="uv run pd-math-server", category="math"),
            ServerConfig(name="text-server", command="uv run pd-text-server", category="text"),
            ServerConfig(name="data-server", command="uv run pd-data-server", category="data"),
        ]
    )

    def __post_init__(self):
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")


########################
# SERVER SESSION MGR   #
########################
class MCPServerManager:
    """Manages connections to multiple MCP servers."""

    def __init__(self, server_configs: List[ServerConfig]):
        self.server_configs = server_configs
        self.sessions: Dict[str, ClientSession] = {}
        self.loops: Dict[str, asyncio.AbstractEventLoop] = {}
        self.exit_stacks: Dict[str, AsyncExitStack] = {}
        self.tools_by_server: Dict[str, List[Type[BaseTool]]] = {}
        self.all_tools: List[Type[BaseTool]] = []

    async def _connect_server(self, config: ServerConfig) -> ClientSession:
        """Connect to a single MCP server."""
        exit_stack = AsyncExitStack()
        self.exit_stacks[config.name] = exit_stack

        command_parts = shlex.split(config.command)
        server_params = StdioServerParameters(command=command_parts[0], args=command_parts[1:], env=None)

        read_stream, write_stream = await exit_stack.enter_async_context(stdio_client(server_params))
        session = await exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        return session

    def connect_all(self, console: Console) -> None:
        """Connect to all configured MCP servers."""
        for config in self.server_configs:
            console.print(f"[dim]Connecting to {config.name}...[/dim]")

            # Create event loop for this server
            loop = asyncio.new_event_loop()
            self.loops[config.name] = loop

            # Connect
            session = loop.run_until_complete(self._connect_server(config))
            self.sessions[config.name] = session

            # Fetch tools
            tools = fetch_mcp_tools(
                mcp_endpoint=None,
                transport_type=MCPTransportType.STDIO,
                client_session=session,
                event_loop=loop,
            )

            self.tools_by_server[config.name] = tools
            self.all_tools.extend(tools)

            console.print(f"[green]  Connected: {len(tools)} tools[/green]")

    def close_all(self, console: Console) -> None:
        """Close all server connections."""
        for name in list(self.sessions.keys()):
            console.print(f"[dim]Closing {name}...[/dim]")
            try:
                loop = self.loops.get(name)
                exit_stack = self.exit_stacks.get(name)
                if loop and exit_stack:
                    loop.run_until_complete(exit_stack.aclose())
                    loop.close()
            except Exception as e:
                console.print(f"[red]Error closing {name}: {e}[/red]")


########################
# STATISTICS TRACKING  #
########################
@dataclass
class DisclosureStats:
    """Track statistics to demonstrate progressive disclosure benefits."""

    total_tools_available: int = 0
    tools_selected: int = 0
    servers_with_selected_tools: int = 0
    search_queries_made: int = 0
    tool_executions: int = 0
    parallel_batches: int = 0
    tools_in_parallel: int = 0

    @property
    def tools_filtered_percentage(self) -> float:
        """Percentage of tools that were NOT loaded."""
        if self.total_tools_available == 0:
            return 0.0
        return ((self.total_tools_available - self.tools_selected) / self.total_tools_available) * 100

    def display(self, console: Console) -> None:
        """Display statistics."""
        table = Table(title="Progressive Disclosure Statistics", box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total tools (3 servers)", str(self.total_tools_available))
        table.add_row("Tools selected for query", str(self.tools_selected))
        table.add_row("Context reduction", f"{self.tools_filtered_percentage:.1f}%")
        table.add_row("Search queries made", str(self.search_queries_made))
        table.add_row("Tool executions", str(self.tool_executions))
        if self.parallel_batches > 0:
            table.add_row("Parallel batches", str(self.parallel_batches))
            table.add_row("Tools run in parallel", str(self.tools_in_parallel))

        console.print(table)


########################
# MAIN DEMO FUNCTION   #
########################
def main():
    """Run the progressive disclosure demonstration with multiple MCP servers."""
    load_dotenv()
    console = Console()
    config = ProgressiveDisclosureConfig()

    console.print(
        Panel.fit(
            "[bold cyan]Progressive Disclosure Demo[/bold cyan]\n"
            "[dim]Demonstrating Anthropic's pattern with 3 MCP servers (24 total tools)[/dim]",
            border_style="cyan",
        )
    )

    # Initialize instructor client
    client = instructor.from_openai(openai.OpenAI(api_key=config.openai_api_key))

    # Initialize server manager
    server_manager = MCPServerManager(config.servers)

    try:
        # Connect to all servers
        console.print("\n[bold]Connecting to MCP servers...[/bold]")
        server_manager.connect_all(console)

        all_tools = server_manager.all_tools
        if not all_tools:
            console.print("[red]No tools found across any server.[/red]")
            return

        # Display all available tools by server
        for server_config in config.servers:
            server_tools = server_manager.tools_by_server.get(server_config.name, [])
            table = Table(title=f"{server_config.name} Tools", box=None)
            table.add_column("Tool", style="cyan")
            table.add_column("Description", style="dim", max_width=50)

            for tool in server_tools:
                name = getattr(tool, "mcp_tool_name", tool.__name__)
                desc = (tool.__doc__ or "")[:50]
                table.add_row(name, desc)
            console.print(table)

        console.print(f"\n[bold green]Total: {len(all_tools)} tools across {len(config.servers)} servers[/bold green]")

        # Create lightweight tool registry
        console.print("\n[dim]Building lightweight tool registry (metadata only)...[/dim]")
        registry = ToolRegistry()

        mcp_definitions = []
        for server_config in config.servers:
            for tool in server_manager.tools_by_server.get(server_config.name, []):
                name = getattr(tool, "mcp_tool_name", tool.__name__)
                description = tool.__doc__ or ""
                mcp_definitions.append(
                    MCPToolDefinition(
                        name=name,
                        description=description,
                        input_schema={},
                    )
                )

        registry.register_from_mcp(mcp_definitions)

        # Create Tool Finder Agent
        console.print("[dim]Creating Tool Finder Agent (sub-agent)...[/dim]")
        finder_agent, search_tool, list_tool = create_tool_finder_agent(
            registry=registry,
            client=client,
            model=config.finder_model,
        )
        console.print(f"[green]Tool Finder ready (using {config.finder_model})[/green]")

        # Create Orchestrator Factory
        # We'll pass all tools and let the factory filter
        orchestrator_factory = OrchestratorFactory(
            mcp_endpoint=None,
            transport_type=MCPTransportType.STDIO,
            client=client,
            model=config.orchestrator_model,
            parallel_execution=config.parallel_execution,
            # We don't pass session/loop since tools already have them bound
        )

        # Interactive loop
        console.print("\n[bold green]Ready! Type '/exit' to quit, '/stats' for statistics.[/bold green]")
        console.print("[dim]Example queries:[/dim]")
        console.print("[dim]  - 'Calculate (5 + 3) * 2'              (math tools)[/dim]")
        console.print("[dim]  - 'Convert HELLO WORLD to lowercase'  (text tools)[/dim]")
        console.print("[dim]  - 'Find the average of [1,2,3,4,5]'   (data tools)[/dim]")
        console.print("[dim]  - 'Reverse the text ABC and add 10+5' (multi-server!)[/dim]\n")

        stats = DisclosureStats(total_tools_available=len(all_tools))

        while True:
            query = console.input("[bold yellow]You:[/bold yellow] ").strip()

            if query.lower() in {"/exit", "/quit"}:
                console.print("[bold red]Exiting. Goodbye![/bold red]")
                break

            if query.lower() == "/stats":
                stats.display(console)
                continue

            if not query:
                continue

            try:
                # Phase 1: Tool Discovery (Progressive Disclosure)
                console.print("\n[bold cyan]Phase 1: Tool Discovery[/bold cyan]")
                console.print(f"[dim]Sub-agent searching {len(all_tools)} tools across {len(config.servers)} servers...[/dim]")

                finder_result = run_tool_finder(
                    agent=finder_agent,
                    search_tool=search_tool,
                    list_tool=list_tool,
                    user_query=query,
                )

                stats.search_queries_made += len(finder_result.search_queries_used)
                stats.tools_selected = len(finder_result.selected_tools)

                console.print(
                    f"[green]Selected {len(finder_result.selected_tools)} tools:[/green] {finder_result.selected_tools}"
                )
                console.print(f"[dim]Reasoning: {finder_result.reasoning}[/dim]")

                # Phase 2: Dynamic Orchestrator Creation
                console.print("\n[bold cyan]Phase 2: Creating Focused Orchestrator[/bold cyan]")

                orchestrator, tool_map = orchestrator_factory.create_with_tools(
                    tool_names=finder_result.selected_tools,
                    all_tools=all_tools,
                )

                if finder_result.selected_tools:
                    tools_count = len(finder_result.selected_tools)
                    tokens_saved = (len(all_tools) - tools_count) * 500
                    console.print(
                        f"[green]Orchestrator context: {tools_count} tools "
                        f"(filtered {stats.tools_filtered_percentage:.0f}% = "
                        f"saved ~{tokens_saved} tokens)[/green]"
                    )
                else:
                    console.print("[yellow]No tools needed - conversational response[/yellow]")

                # Phase 3: Query Execution
                console.print("\n[bold cyan]Phase 3: Query Execution[/bold cyan]")

                def on_tool_execution(tool_name: str, params: dict):
                    stats.tool_executions += 1
                    console.print(f"[blue]Executing:[/blue] {tool_name}")
                    console.print(f"[dim]Parameters: {params}[/dim]")

                def on_parallel_batch(count: int):
                    stats.parallel_batches += 1
                    stats.tools_in_parallel += count
                    console.print(f"[magenta]⚡ Parallel batch:[/magenta] {count} tools executing simultaneously")

                if config.parallel_execution:
                    response = execute_orchestrator_loop_parallel(
                        orchestrator=orchestrator,
                        tool_schema_to_class=tool_map,
                        initial_query=query,
                        on_tool_execution=on_tool_execution,
                        on_parallel_batch=on_parallel_batch,
                    )
                else:
                    response = execute_orchestrator_loop(
                        orchestrator=orchestrator,
                        tool_schema_to_class=tool_map,
                        initial_query=query,
                        on_tool_execution=on_tool_execution,
                    )

                console.print(f"\n[bold green]Response:[/bold green] {response}")

                # Show savings summary
                savings_pct = stats.tools_filtered_percentage
                parallel_info = " | ⚡ Parallel mode" if config.parallel_execution else ""
                console.print(
                    Panel(
                        f"[dim]Progressive Disclosure: {len(finder_result.selected_tools)}/{len(all_tools)} tools loaded "
                        f"({savings_pct:.0f}% context reduction){parallel_info}[/dim]",
                        border_style="dim",
                    )
                )

                # Reset histories for next query
                finder_agent.history.history = []
                finder_agent.history.current_turn_id = None
                orchestrator.history.history = []
                orchestrator.history.current_turn_id = None

            except Exception as e:
                console.print(f"[red]Error:[/red] {str(e)}")
                import traceback

                console.print(f"[dim]{traceback.format_exc()}[/dim]")

    finally:
        # Cleanup all servers
        console.print("\n[dim]Cleaning up server connections...[/dim]")
        server_manager.close_all(console)


if __name__ == "__main__":
    main()
