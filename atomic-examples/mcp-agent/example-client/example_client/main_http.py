"""
HTTP Stream transport client for MCP Agent example.
Communicates with the server_http.py `/mcp` endpoint using HTTP GET/POST/DELETE for JSON-RPC streams.
"""

from atomic_agents.connectors.mcp import (
    fetch_mcp_tools,
    fetch_mcp_resources,
    fetch_mcp_prompts,
    MCPTransportType,
)
from atomic_agents.context import ChatHistory, SystemPromptGenerator
from atomic_agents import BaseIOSchema, AtomicAgent, AgentConfig
import sys
from rich.console import Console
from rich.table import Table
from rich.markdown import Markdown
from pydantic import Field
import openai
import os
import instructor
from typing import Union, Type, Dict
from dataclasses import dataclass


@dataclass
class MCPConfig:
    """Configuration for the MCP Agent system using HTTP Stream transport."""

    mcp_server_url: str = "http://localhost:6969"
    openai_model: str = "gpt-5.6-terra"
    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    reasoning_effort: str = "low"

    def __post_init__(self):
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")


def main():
    # Use default HTTP transport settings from MCPConfig
    config = MCPConfig()
    console = Console()
    client = instructor.from_openai(openai.OpenAI(api_key=config.openai_api_key))

    console.print("[bold green]Initializing MCP Agent System (HTTP Stream mode)...[/bold green]")
    tools = fetch_mcp_tools(mcp_endpoint=config.mcp_server_url, transport_type=MCPTransportType.HTTP_STREAM)
    resources = fetch_mcp_resources(mcp_endpoint=config.mcp_server_url, transport_type=MCPTransportType.HTTP_STREAM)
    prompts = fetch_mcp_prompts(mcp_endpoint=config.mcp_server_url, transport_type=MCPTransportType.HTTP_STREAM)
    if not tools and not resources and not prompts:
        console.print(f"[bold red]No MCP tools or resources or prompts found at {config.mcp_server_url}[/bold red]")
        sys.exit(1)

    # Display available tools
    table = Table(title="Available MCP Tools", box=None)
    table.add_column("Tool Name", style="cyan")
    table.add_column("Input Schema", style="yellow")
    table.add_column("Description", style="magenta")
    for ToolClass in tools:
        schema_name = getattr(ToolClass.input_schema, "__name__", "N/A")
        table.add_row(ToolClass.mcp_tool_name, schema_name, ToolClass.__doc__ or "")
    console.print(table)

    # Display resources and prompts if available
    if resources:
        rtable = Table(title="Available MCP Resources", box=None)
        rtable.add_column("Name", style="cyan")
        rtable.add_column("Description", style="magenta")
        rtable.add_column("Input Schema", style="yellow")
        for ResourceClass in resources:
            schema_name = getattr(ResourceClass.input_schema, "__name__", "N/A")
            rtable.add_row(ResourceClass.mcp_resource_name, schema_name, ResourceClass.__doc__ or "")
        console.print(rtable)
    if prompts:
        ptable = Table(title="Available MCP Prompts", box=None)
        ptable.add_column("Name", style="cyan")
        ptable.add_column("Description", style="magenta")
        ptable.add_column("Input Schema", style="yellow")
        for PromptClass in prompts:
            schema_name = getattr(PromptClass.input_schema, "__name__", "N/A")
            ptable.add_row(PromptClass.mcp_prompt_name, schema_name, PromptClass.__doc__ or "")
        console.print(ptable)

    # Build orchestrator
    class MCPOrchestratorInputSchema(BaseIOSchema):
        """Input schema for the MCP orchestrator that processes user queries."""

        query: str = Field(...)

    class FinalResponseSchema(BaseIOSchema):
        """Schema for the final response to the user."""

        response_text: str = Field(...)

    # Map schemas and define ActionUnion
    tool_schema_map: Dict[Type[BaseIOSchema], Type] = {
        ToolClass.input_schema: ToolClass for ToolClass in tools if hasattr(ToolClass, "input_schema")
    }
    resource_schema_to_class_map: Dict[Type[BaseIOSchema], Type[AtomicAgent]] = {
        ResourceClass.input_schema: ResourceClass for ResourceClass in resources if hasattr(ResourceClass, "input_schema")
    }  # type: ignore
    prompt_schema_to_class_map: Dict[Type[BaseIOSchema], Type[AtomicAgent]] = {
        PromptClass.input_schema: PromptClass for PromptClass in prompts if hasattr(PromptClass, "input_schema")
    }  # type: ignore
    available_schemas = (
        tuple(tool_schema_map.keys())
        + tuple(resource_schema_to_class_map.keys())
        + tuple(prompt_schema_to_class_map.keys())
        + (FinalResponseSchema,)
    )
    ActionUnion = Union[available_schemas]

    class OrchestratorOutputSchema(BaseIOSchema):
        """Output schema for the MCP orchestrator containing reasoning and selected action."""

        reasoning: str
        action: ActionUnion = Field(  # type: ignore[reportInvalidTypeForm]
            ...,
            description="The chosen action: either a tool/resource/prompt's input schema instance or a final response schema instance.",
        )

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

    console.print("[bold green]HTTP Stream client ready. Type 'exit' to quit.[/bold green]")
    while True:
        query = console.input("[bold yellow]You:[/bold yellow] ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue

        try:
            # Initial run with user query
            orchestrator_output = orchestrator_agent.run(MCPOrchestratorInputSchema(query=query))

            # Debug output to see what's actually in the output
            console.print(
                f"[dim]Debug - orchestrator_output type: {type(orchestrator_output)}, fields: {orchestrator_output.model_dump()}"
            )

            # Handle the output similar to SSE version
            if hasattr(orchestrator_output, "chat_message") and not hasattr(orchestrator_output, "action"):
                # Convert BasicChatOutputSchema to FinalResponseSchema
                action_instance = FinalResponseSchema(response_text=orchestrator_output.chat_message)
                reasoning = "Response generated directly from chat model"
            elif hasattr(orchestrator_output, "action"):
                action_instance = orchestrator_output.action
                reasoning = (
                    orchestrator_output.reasoning if hasattr(orchestrator_output, "reasoning") else "No reasoning provided"
                )
            else:
                console.print("[yellow]Warning: Unexpected response format. Unable to process.[/yellow]")
                continue

            console.print(f"[cyan]Orchestrator reasoning:[/cyan] {reasoning}")

            # Keep executing until we get a final response
            while not isinstance(action_instance, FinalResponseSchema):
                schema_type = type(action_instance)
                schema_type_valid = False

                try:
                    ToolClass = tool_schema_map.get(schema_type)
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
                        console.print(f"[dim]Parameters: {action_instance.model_dump()}")

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
                            query=(f"Prompt {prompt_name} generated content: {prompt_output.content}")
                        )
                        orchestrator_agent.add_tool_result(result_message)

                    if not schema_type_valid:
                        console.print(f"[red]Error: Unknown schema type {schema_type.__name__}[/red]")
                        action_instance = FinalResponseSchema(
                            response_text="I encountered an internal error. Could not find the appropriate tool/resource/prompt."
                        )
                        break

                    next_output = orchestrator_agent.run()
                    if hasattr(next_output, "action"):
                        action_instance = next_output.action
                        if hasattr(next_output, "reasoning"):
                            console.print(f"[cyan]Orchestrator reasoning:[/cyan] {next_output.reasoning}")
                    else:
                        # If no action, treat as final response
                        action_instance = FinalResponseSchema(response_text=next_output.chat_message)

                except Exception as e:
                    console.print(f"[red]Error executing tool: {e}[/red]")
                    action_instance = FinalResponseSchema(
                        response_text=f"I encountered an error while executing the tool: {str(e)}"
                    )
                    break

            # Display final response
            if isinstance(action_instance, FinalResponseSchema):
                md = Markdown(action_instance.response_text)
                console.print("[bold blue]Agent:[/bold blue]")
                console.print(md)
            else:
                console.print("[red]Error: Expected final response but got something else[/red]")

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()
