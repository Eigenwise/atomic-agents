# pyright: reportInvalidTypeForm=false
from atomic_agents.connectors.mcp import (
    fetch_mcp_tools,
    fetch_mcp_resources,
    fetch_mcp_prompts,
    MCPTransportType,
)
from atomic_agents import BaseIOSchema, AtomicAgent, AgentConfig
from atomic_agents.context import ChatHistory, SystemPromptGenerator
from rich.console import Console
from rich.table import Table
import openai
import os
import instructor
import asyncio
import shlex
from contextlib import AsyncExitStack
from pydantic import Field
from typing import Union, Type, Dict, Optional
from dataclasses import dataclass
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# 1. Configuration and environment setup
@dataclass
class MCPConfig:
    """Configuration for the MCP Agent system using STDIO transport."""

    openai_model: str = "gpt-5.6-terra"
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    reasoning_effort: str = "low"

    # Command to run the STDIO server.
    # In practice, this could be something like "pipx some-other-persons-server or npx some-other-persons-server
    # if working with a server you did not write yourself.
    mcp_stdio_server_command: str = "uv run example-mcp-server --mode stdio"

    def __post_init__(self):
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")


config = MCPConfig()
console = Console()
client = instructor.from_openai(openai.OpenAI(api_key=config.openai_api_key))


class FinalResponseSchema(BaseIOSchema):
    """Schema for providing a final text response to the user."""

    response_text: str = Field(..., description="The final text response to the user's query")


# --- Bootstrap persistent STDIO session ---
stdio_session: Optional[ClientSession] = None
stdio_loop: Optional[asyncio.AbstractEventLoop] = None
stdio_exit_stack: Optional[AsyncExitStack] = None

# Initialize STDIO session
stdio_loop = asyncio.new_event_loop()


async def _bootstrap_stdio():
    global stdio_exit_stack  # Allow modification of the global variable
    stdio_exit_stack = AsyncExitStack()
    command_parts = shlex.split(config.mcp_stdio_server_command)
    server_params = StdioServerParameters(command=command_parts[0], args=command_parts[1:], env=None)
    read_stream, write_stream = await stdio_exit_stack.enter_async_context(stdio_client(server_params))
    session = await stdio_exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
    await session.initialize()
    return session


stdio_session = stdio_loop.run_until_complete(_bootstrap_stdio())
# The stdio_exit_stack is kept to clean up later

# Fetch tools and build ActionUnion statically
tools = fetch_mcp_tools(
    mcp_endpoint=None,
    transport_type=MCPTransportType.STDIO,
    client_session=stdio_session,  # Pass persistent session
    event_loop=stdio_loop,  # Pass corresponding loop
)
resources = fetch_mcp_resources(
    mcp_endpoint=None, transport_type=MCPTransportType.STDIO, client_session=stdio_session, event_loop=stdio_loop
)
prompts = fetch_mcp_prompts(
    mcp_endpoint=None, transport_type=MCPTransportType.STDIO, client_session=stdio_session, event_loop=stdio_loop
)
if not tools and not resources and not prompts:
    raise RuntimeError("No MCP tools or resources or prompts found. Please ensure the MCP server is running and accessible.")

# Build mapping from input_schema to ToolClass
tool_schema_to_class_map: Dict[Type[BaseIOSchema], Type[AtomicAgent]] = {
    ToolClass.input_schema: ToolClass for ToolClass in tools if hasattr(ToolClass, "input_schema")
}
# Collect all tool input schemas
tool_input_schemas = tuple(tool_schema_to_class_map.keys())

# Build mapping for resources and prompts
resource_schema_to_class_map: Dict[Type[BaseIOSchema], Type[AtomicAgent]] = {
    ResourceClass.input_schema: ResourceClass for ResourceClass in resources if hasattr(ResourceClass, "input_schema")
}  # type: ignore
resource_input_schemas = tuple(resource_schema_to_class_map.keys())

prompt_schema_to_class_map: Dict[Type[BaseIOSchema], Type[AtomicAgent]] = {
    PromptClass.input_schema: PromptClass for PromptClass in prompts if hasattr(PromptClass, "input_schema")
}  # type: ignore
prompt_input_schemas = tuple(prompt_schema_to_class_map.keys())

# Available schemas include all tool input schemas, resource schemas, prompts and the final response schema
available_schemas = tool_input_schemas + resource_input_schemas + prompt_input_schemas + (FinalResponseSchema,)

# Define the Union of all action schemas
ActionUnion = Union[available_schemas]


# 2. Schema and class definitions
class MCPOrchestratorInputSchema(BaseIOSchema):
    """Input schema for the MCP Orchestrator Agent."""

    query: str = Field(..., description="The user's query to analyze.")


class OrchestratorOutputSchema(BaseIOSchema):
    """Output schema for the orchestrator. Contains reasoning and the chosen action."""

    reasoning: str = Field(
        ..., description="Detailed explanation of why this action was chosen and how it will address the user's query."
    )
    action: ActionUnion = Field(  # type: ignore[reportInvalidTypeForm]
        ...,
        description="The chosen action: either a tool/resource/prompt's input schema instance or a final response schema instance.",
    )

    model_config = {"arbitrary_types_allowed": True}


# 3. Main logic and script entry point
def main():
    try:
        console.print("[bold green]Initializing MCP Agent System (STDIO mode)...[/bold green]")
        # Display available tools
        table = Table(title="Available MCP Tools", box=None)
        table.add_column("Tool Name", style="cyan")
        table.add_column("Input Schema", style="yellow")
        table.add_column("Description", style="magenta")
        for ToolClass in tools:
            schema_name = ToolClass.input_schema.__name__ if hasattr(ToolClass, "input_schema") else "N/A"
            table.add_row(ToolClass.mcp_tool_name, schema_name, ToolClass.__doc__ or "")
        console.print(table)

        # Display resources and prompts if available
        if resources:
            rtable = Table(title="Available MCP Resources", box=None)
            rtable.add_column("Name", style="cyan")
            rtable.add_column("Description", style="magenta")
            rtable.add_column("Input Schema", style="yellow")
            for ResourceClass in resources:
                schema_name = ResourceClass.input_schema.__name__ if hasattr(ResourceClass, "input_schema") else "N/A"
                rtable.add_row(ResourceClass.mcp_resource_name, ResourceClass.__doc__ or "", schema_name)
            console.print(rtable)
        if prompts:
            ptable = Table(title="Available MCP Prompts", box=None)
            ptable.add_column("Name", style="cyan")
            ptable.add_column("Description", style="magenta")
            ptable.add_column("Input Schema", style="yellow")
            for PromptClass in prompts:
                schema_name = PromptClass.input_schema.__name__ if hasattr(PromptClass, "input_schema") else "N/A"
                ptable.add_row(PromptClass.mcp_prompt_name, PromptClass.__doc__ or "", schema_name)
            console.print(ptable)

        # Create and initialize orchestrator agent
        console.print("[dim]• Creating orchestrator agent...[/dim]")
        history = ChatHistory()
        orchestrator_agent = AtomicAgent[MCPOrchestratorInputSchema, OrchestratorOutputSchema](
            AgentConfig(
                client=client,
                model=config.openai_model,
                model_api_parameters={"reasoning_effort": config.reasoning_effort},
                history=history,
                system_prompt_generator=SystemPromptGenerator(
                    background=[
                        "You are an MCP Orchestrator Agent, designed to chat with users and",
                        "determine the best way to handle their queries using the available tools, resources, and prompts.",
                    ],
                    steps=[
                        "1. Use the reasoning field to determine if one or more successive "
                        "tool/resource/prompt calls could be used to handle the user's query.",
                        "2. If so, choose the appropriate tool(s), resource(s), or prompt(s) one "
                        "at a time and extract all necessary parameters from the query.",
                        "3. If a single tool/resource/prompt can not be used to handle the user's query, "
                        "think about how to break down the query into "
                        "smaller tasks and route them to the appropriate tool(s)/resource(s)/prompt(s).",
                        "4. If no sequence of tools/resources/prompts could be used, or if you are "
                        "finished processing the user's query, provide a final response to the user.",
                        "5. If the context is sufficient and no more tools/resources/prompts are needed, provide a final response to the user.",
                    ],
                    output_instructions=[
                        "1. Always provide a detailed explanation of your decision-making process in the 'reasoning' field.",
                        "2. Choose exactly one action schema (either a tool/resource/prompt input or FinalResponseSchema).",
                        "3. Ensure all required parameters for the chosen tool/resource/prompt are properly extracted and validated.",
                        "4. Maintain a professional and helpful tone in all responses.",
                        "5. Break down complex queries into sequential tool/resource/prompt calls "
                        "before giving the final answer via `FinalResponseSchema`.",
                    ],
                ),
            )
        )
        console.print("[green]Successfully created orchestrator agent.[/green]")
        console.print("[bold green]MCP Agent Interactive Chat (STDIO mode). Type '/exit' or '/quit' to leave.[/bold green]")
        while True:
            query = console.input("[bold yellow]You:[/bold yellow] ").strip()
            if query.lower() in {"/exit", "/quit"}:
                console.print("[bold red]Exiting chat. Goodbye![/bold red]")
                break
            if not query:
                continue  # Ignore empty input
            try:
                # Initial run with user query
                orchestrator_output = orchestrator_agent.run(MCPOrchestratorInputSchema(query=query))
                action_instance = orchestrator_output.action
                reasoning = orchestrator_output.reasoning
                console.print(f"[cyan]Orchestrator reasoning:[/cyan] {reasoning}")

                # Keep executing until we get a final response
                while not isinstance(action_instance, FinalResponseSchema):
                    schema_type = type(action_instance)
                    schema_type_valid = False

                    ToolClass = tool_schema_to_class_map.get(schema_type)
                    if ToolClass:
                        schema_type_valid = True
                        tool_name = ToolClass.mcp_tool_name
                        console.print(f"[blue]Executing tool:[/blue] {tool_name}")
                        console.print(f"[dim]Parameters:[/dim] " f"{action_instance.model_dump()}")

                        tool_instance = ToolClass()
                        # The persistent session/loop are already part of the ToolClass definition
                        tool_output = tool_instance.run(action_instance)
                        console.print(f"[bold green]Result:[/bold green] {tool_output.result}")

                        # Add tool result to agent history
                        result_message = MCPOrchestratorInputSchema(
                            query=(f"Tool {tool_name} executed with result: " f"{tool_output.result}")
                        )
                        orchestrator_agent.add_tool_result(result_message)

                    ResourceClass = resource_schema_to_class_map.get(schema_type)
                    if ResourceClass:
                        schema_type_valid = True
                        resource_name = ResourceClass.mcp_resource_name
                        console.print(f"[blue]Reading resource:[/blue] {resource_name}")
                        console.print(f"[dim]Parameters:[/dim] " f"{action_instance.model_dump()}")

                        resource_instance = ResourceClass()
                        resource_output = resource_instance.read(action_instance)
                        console.print(f"[bold green]Resource content:[/bold green] {resource_output.content}")

                        # Add resource result to agent history
                        result_message = MCPOrchestratorInputSchema(
                            query=(f"Resource {resource_name} read with content: {resource_output.content}")
                        )
                        orchestrator_agent.add_tool_result(result_message)

                    PromptClass = prompt_schema_to_class_map.get(schema_type)
                    if PromptClass:
                        schema_type_valid = True
                        prompt_name = PromptClass.mcp_prompt_name
                        console.print(f"[blue]Fetching prompt:[/blue] {prompt_name}")
                        console.print(f"[dim]Parameters:[/dim] " f"{action_instance.model_dump()}")

                        prompt_instance = PromptClass()
                        prompt_output = prompt_instance.generate(action_instance)
                        console.print(f"[bold green]Prompt content:[/bold green] {prompt_output.content}")

                        # Add prompt result to agent history
                        result_message = MCPOrchestratorInputSchema(
                            query=(f'Prompt {prompt_name} generated successfully. Content: "{prompt_output.content}"')
                        )
                        orchestrator_agent.add_tool_result(result_message)

                    if not schema_type_valid:
                        raise ValueError(f"Unknown schema type '" f"{schema_type.__name__}" f"' returned by orchestrator")

                    # Run the agent again without parameters to continue the flow
                    orchestrator_output = orchestrator_agent.run()
                    action_instance = orchestrator_output.action
                    reasoning = orchestrator_output.reasoning
                    console.print(f"[cyan]Orchestrator reasoning:[/cyan] {reasoning}")

                # Final response from the agent
                console.print(f"[bold blue]Agent:[/bold blue] {action_instance.response_text}")

            except Exception as e:
                console.print(f"[red]Error processing query:[/red] {str(e)}")
                console.print_exception()
    except Exception as e:
        console.print(f"[bold red]Fatal error:[/bold red] {str(e)}")
        console.print_exception()
        return
    finally:
        # Cleanup persistent STDIO resources
        if stdio_loop and stdio_exit_stack:
            console.print("\n[dim]Cleaning up STDIO resources...[/dim]")
            try:
                stdio_loop.run_until_complete(stdio_exit_stack.aclose())
            except Exception as cleanup_err:
                console.print(f"[red]Error during STDIO cleanup:[/red] {cleanup_err}")
            finally:
                stdio_loop.close()


if __name__ == "__main__":
    main()
