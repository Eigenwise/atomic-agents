"""Tool Finder Agent for progressive disclosure.

This agent is responsible for discovering relevant tools for a given user query.
It analyzes the lightweight tool registry to find the most appropriate tools,
allowing the main orchestrator to be created with only the necessary tools
loaded into its context window.

This implements the "search_tools" pattern from Anthropic's progressive disclosure.
"""

from typing import List, Optional
from pydantic import Field
import instructor

from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from atomic_agents.context import SystemPromptGenerator, BaseDynamicContextProvider

from progressive_disclosure.registry.tool_registry import ToolRegistry


########################
# INPUT/OUTPUT SCHEMAS #
########################
class ToolFinderInputSchema(BaseIOSchema):
    """Input for the tool finder agent."""

    user_query: str = Field(
        ...,
        description="The user's original query that needs to be analyzed to determine required tools.",
    )
    task_context: Optional[str] = Field(
        default=None,
        description="Additional context about the task that might help with tool selection.",
    )


class ToolFinderOutputSchema(BaseIOSchema):
    """Output containing selected tools for the main orchestrator."""

    reasoning: str = Field(
        ...,
        description="Detailed explanation of why these specific tools were selected and how they relate to the user's query.",
    )
    selected_tools: List[str] = Field(
        ...,
        description="Names of tools that should be loaded for the main orchestrator. Keep this list minimal.",
    )
    search_queries_used: List[str] = Field(
        default_factory=list,
        description="Keywords or concepts used to identify these tools.",
    )
    confidence: str = Field(
        default="high",
        description="Confidence level in tool selection: 'high', 'medium', or 'low'.",
    )


#####################
# CONTEXT PROVIDERS #
#####################
class ToolRegistryProvider(BaseDynamicContextProvider):
    """Provides the full tool registry to the finder agent."""

    def __init__(self, registry: ToolRegistry, title: str = "Available Tools"):
        super().__init__(title)
        self._registry = registry

    def get_info(self) -> str:
        """Get all available tools with descriptions."""
        tools = self._registry.get_all_tools()
        if not tools:
            return "No tools available in registry."

        lines = ["The following tools are available:\n"]
        for tool in tools:
            lines.append(f"- **{tool.name}**: {tool.description}")

        lines.append("\nSelect ONLY the tools needed to complete the user's query.")
        return "\n".join(lines)


#############################
# TOOL FINDER AGENT FACTORY #
#############################
def create_tool_finder_agent(
    registry: ToolRegistry,
    client: instructor.Instructor,
    model: str = "gpt-5.6-terra",
) -> tuple[AtomicAgent, None, None]:
    """Create a tool finder agent with access to tool metadata.

    The tool finder agent uses a lightweight model to analyze user queries
    and determine which MCP tools should be loaded for the main orchestrator.

    Args:
        registry: Tool registry containing metadata about available tools.
        client: Instructor-wrapped LLM client.
        model: Model to use for the finder agent. Default is gpt-5.6-terra
               for cost efficiency since this is a discovery task.

    Returns:
        Tuple of (agent, None, None) - the None values maintain API compatibility.

    Example:
        >>> registry = ToolRegistry()
        >>> registry.register_from_mcp(mcp_definitions)
        >>> client = instructor.from_openai(openai.OpenAI())
        >>> agent, _, _ = create_tool_finder_agent(registry, client)
        >>> result = run_tool_finder(agent, None, None, "Calculate 2+2")
        >>> print(result.selected_tools)
        ['add_numbers']
    """
    # Create the agent
    agent = AtomicAgent[ToolFinderInputSchema, ToolFinderOutputSchema](
        config=AgentConfig(
            client=client,
            model=model,
            system_prompt_generator=SystemPromptGenerator(
                background=[
                    "You are a Tool Finder Agent specialized in discovering relevant tools for user queries.",
                    "Your role is to analyze user queries and find the MINIMUM set of tools needed to accomplish the task.",
                    "You have access to a list of available MCP tools with their descriptions.",
                    "",
                    "IMPORTANT: Your goal is CONTEXT EFFICIENCY - select only the tools that are directly needed.",
                    "The tools you select will be loaded into another agent's context window.",
                    "Loading unnecessary tools wastes context space and reduces accuracy.",
                ],
                steps=[
                    "1. Analyze the user's query to understand what capabilities are needed",
                    "2. Review the available tools list provided in your context",
                    "3. Select ONLY the tools that are necessary for this specific query",
                    "4. Provide your selection with clear reasoning",
                ],
                output_instructions=[
                    "Select the MINIMUM number of tools needed - prefer fewer tools over more",
                    "Only include tools that are directly relevant to accomplishing the user's task",
                    "If no tools are needed (e.g., general conversation), return an empty list",
                    "Include clear reasoning for each selected tool",
                    "Rate your confidence: 'high' if certain, 'medium' if tools might work, 'low' if unsure",
                    "Use the exact tool names as they appear in the available tools list",
                ],
            ),
        )
    )

    # Register context provider with full tool list
    agent.register_context_provider(
        "tool_registry",
        ToolRegistryProvider(registry, "Available Tools"),
    )

    return agent, None, None


def run_tool_finder(
    agent: AtomicAgent,
    search_tool,  # Not used, kept for API compatibility
    list_tool,  # Not used, kept for API compatibility
    user_query: str,
    task_context: Optional[str] = None,
    max_iterations: int = 5,  # Not used, kept for API compatibility
) -> ToolFinderOutputSchema:
    """Run the tool finder agent to discover relevant tools.

    This is a single-pass approach - the agent sees all tool metadata
    and selects the relevant tools in one call.

    Args:
        agent: The tool finder agent.
        search_tool: Not used (kept for API compatibility).
        list_tool: Not used (kept for API compatibility).
        user_query: The user's query to analyze.
        task_context: Optional additional context.
        max_iterations: Not used (kept for API compatibility).

    Returns:
        ToolFinderOutputSchema with the selected tools.
    """
    input_schema = ToolFinderInputSchema(
        user_query=user_query,
        task_context=task_context,
    )

    # Single-pass tool selection
    result = agent.run(input_schema)
    return result


#################
# EXAMPLE USAGE #
#################
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from rich.console import Console
    import openai
    from progressive_disclosure.registry.tool_registry import ToolMetadata

    load_dotenv()
    console = Console()

    # Create a test registry
    registry = ToolRegistry()
    registry.register(
        ToolMetadata(
            name="add_numbers",
            description="Add two numbers together",
            keywords=["add", "sum", "plus", "arithmetic"],
            category="math",
        )
    )
    registry.register(
        ToolMetadata(
            name="subtract_numbers",
            description="Subtract one number from another",
            keywords=["subtract", "minus", "difference", "arithmetic"],
            category="math",
        )
    )
    registry.register(
        ToolMetadata(
            name="multiply_numbers",
            description="Multiply two numbers together",
            keywords=["multiply", "times", "product", "arithmetic"],
            category="math",
        )
    )
    registry.register(
        ToolMetadata(
            name="divide_numbers",
            description="Divide one number by another",
            keywords=["divide", "quotient", "arithmetic"],
            category="math",
        )
    )
    registry.register(
        ToolMetadata(
            name="uppercase",
            description="Convert text to uppercase",
            keywords=["upper", "capitalize", "text"],
            category="text",
        )
    )
    registry.register(
        ToolMetadata(
            name="reverse_text",
            description="Reverse the characters in text",
            keywords=["reverse", "backwards", "text"],
            category="text",
        )
    )

    # Create client
    client = instructor.from_openai(openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

    # Create finder agent
    agent, _, _ = create_tool_finder_agent(registry, client)

    # Test queries
    test_queries = [
        "What is 5 plus 3?",
        "Calculate (10 - 4) * 2",
        "Reverse the word HELLO and convert ABC to uppercase",
    ]

    for query in test_queries:
        console.print(f"\n[bold cyan]Query:[/bold cyan] {query}")
        result = run_tool_finder(agent, None, None, query)
        console.print(f"[bold green]Selected tools:[/bold green] {result.selected_tools}")
        console.print(f"[dim]Reasoning: {result.reasoning}[/dim]")
        console.print(f"[dim]Confidence: {result.confidence}[/dim]")

        # Reset history for next query
        agent.history.history = []
        agent.history.current_turn_id = None
