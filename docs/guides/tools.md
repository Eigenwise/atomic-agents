# Tools Guide

In Atomic Agents, **tools are not a magic parameter on the agent.** This is the single most common point of confusion for users coming from frameworks like LangChain, CrewAI, or PydanticAI, where you would write:

```python
# ❌ This is NOT how Atomic Agents works
agent = Agent(tools=[calculator, search])
```

There is no `tools=[...]` argument anywhere in the framework, and that is **intentional**. This guide explains the philosophy and shows the two patterns you will use in practice.

## Philosophy: tools are atomic components, not framework citizens

A tool in Atomic Agents is just an object with:

- A typed `input_schema` (a `BaseIOSchema`)
- A typed `output_schema` (a `BaseIOSchema`)
- A `run()` method that takes one and returns the other

It does not know about agents, prompts, memory, or any LLM. **You** decide when to call it. That control is the whole point — you can read the call site, set a breakpoint on it, and reason about cost and latency the same way you reason about any other function call.

This buys you three things other frameworks struggle with:

1. **Determinism where you want it.** If the next step is "always run the search tool," you just call it. No LLM, no prompt overhead, no chance of the model deciding to skip it.
2. **A real call graph.** Tools are functions. Stack traces, profiler output, and code search work normally. There is no opaque agent loop hiding the dispatch.
3. **No coupling.** A tool is reusable in non-agent code. The same `CalculatorTool` instance works in a script, a FastAPI handler, or a unit test, with no agent involved.

## The two patterns

In practice, every tool call in Atomic Agents falls into one of two patterns. Pick based on whether *you* know which tool to call, or whether the *LLM* needs to decide.

### Pattern 1: Direct call (you know which tool to use)

When the workflow is fixed — "first generate a query, then run the search, then summarize" — call the tool directly. This is the default. It's faster, cheaper, more debuggable, and harder for an LLM to derail.

```python
from atomic_agents import AtomicAgent, AgentConfig
from my_tools.search import SearXNGSearchTool, SearXNGSearchToolConfig

# 1. Agent generates structured search queries.
#    Notice: query_agent's output_schema IS SearXNGSearchTool's input_schema.
query_agent = AtomicAgent[QueryAgentInputSchema, SearXNGSearchTool.input_schema](
    AgentConfig(client=client, model="gpt-4o-mini", ...)
)

# 2. Tool is just an object you instantiate.
search_tool = SearXNGSearchTool(config=SearXNGSearchToolConfig(base_url="..."))

# 3. You wire them together with normal Python — no framework glue.
queries = query_agent.run(QueryAgentInputSchema(instruction="Find recent papers on..."))
results = search_tool.run(queries)  # output of agent IS input of tool
```

The schema alignment between `query_agent`'s `output_schema` and `SearXNGSearchTool.input_schema` is what makes this composable: the agent literally cannot produce something the tool cannot accept, because they share the same Pydantic schema.

Use this pattern when:
- The order of operations is known at build time.
- You care about latency and cost (no extra LLM call to "decide").
- You want the call site to show up in stack traces and code search.
- The tool is non-optional — skipping it would be a bug.

### Pattern 2: Choice agent (LLM picks the tool)

When the workflow genuinely depends on the user's input — "if it's math, use the calculator; if it's a fact lookup, search the web" — let an LLM pick. The mechanism is a normal agent whose `output_schema` is a **`Union` of tool input schemas**. Instructor will validate the model's response against the union, so the agent can only return well-formed input for one of your tools.

```python
from typing import Union
from pydantic import Field
from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from atomic_agents.context import SystemPromptGenerator

from my_tools.search import SearXNGSearchToolInputSchema
from my_tools.calculator import CalculatorToolInputSchema


class OrchestratorInput(BaseIOSchema):
    """User's question."""
    chat_message: str = Field(..., description="The user's input message.")


class OrchestratorOutput(BaseIOSchema):
    """Orchestrator picks ONE tool input schema from the union."""
    tool_parameters: Union[SearXNGSearchToolInputSchema, CalculatorToolInputSchema] = Field(
        ..., description="Parameters for the selected tool."
    )


orchestrator = AtomicAgent[OrchestratorInput, OrchestratorOutput](
    AgentConfig(
        client=client,
        model="gpt-4o-mini",
        system_prompt_generator=SystemPromptGenerator(
            background=[
                "You route the user's request to the right tool.",
                "Use the search tool for factual questions and current events.",
                "Use the calculator for mathematical expressions.",
            ],
            output_instructions=[
                "Return only the parameters for the chosen tool.",
            ],
        ),
    )
)

# YOU still dispatch on the type the LLM picked — there's no hidden routing.
result = orchestrator.run(OrchestratorInput(chat_message=user_input))

if isinstance(result.tool_parameters, SearXNGSearchToolInputSchema):
    tool_output = search_tool.run(result.tool_parameters)
elif isinstance(result.tool_parameters, CalculatorToolInputSchema):
    tool_output = calculator_tool.run(result.tool_parameters)
```

The `isinstance` dispatch is deliberate. It keeps tool selection visible and traceable — adding a tool means adding a `Union` member, a system-prompt line, and an `isinstance` branch, all in one file.

Use this pattern when:
- The tool to call genuinely depends on natural-language input.
- The set of candidate tools is small (a handful, not dozens — Union grows the prompt).
- You want the LLM's reasoning for the choice to be inspectable (extend the output schema with a `reasoning: str` field).

A complete, runnable version of this pattern lives in [`atomic-examples/orchestration-agent`](https://github.com/eigenwise/atomic-agents/tree/main/atomic-examples/orchestration-agent). The [Orchestration guide](orchestration.md) covers tool-selection, multi-agent pipelines, dynamic routing, and parallel execution in more depth.

## Picking a pattern

| Question | Pattern 1 (Direct) | Pattern 2 (Choice agent) |
|---|---|---|
| Is the next tool always the same? | ✅ | |
| Does the choice depend on free-form user input? | | ✅ |
| Latency budget tight? | ✅ | (extra LLM round-trip) |
| Want full debuggability? | ✅ | (still good — choice is in the schema) |
| Tool is required for correctness? | ✅ | |
| Tool set growing past ~5–7? | (still works) | (consider hierarchical routing instead) |

When in doubt, start with Pattern 1. Add a choice agent only when you actually have a routing problem that input data can't answer.

## The Atomic Forge: where tools live

Tools themselves are distributed via the **Atomic Forge** — a registry of standalone, modular tool packages that you download into your project. The Forge approach gives you:

1. **Full Control**: You own the tool's source. Modify behavior locally without forking the framework.
2. **Dependency Management**: Tools live in your codebase, so their dependencies are yours to pin.
3. **Lightweight**: Download only what you use. No Sympy unless you use the calculator; no requests unless you use a search tool.

### Available tools

The Atomic Forge ships with several pre-built tools:

- **arXiv Search**: Search arXiv for academic papers (free public API)
- **BoCha Search**: Web search
- **Calculator**: Perform mathematical calculations
- **DateTime**: Timezone-aware now / parse / convert / shift / diff (no key required)
- **Fía Signals**: Crypto market intelligence — market regime, trading signals, DeFi yields, gas prices, Solana trending tokens, and wallet risk scoring
- **FXMacroData**: Macroeconomic releases with official announcement times, release calendars, FX rates, and COT positioning (USD data needs no key)
- **Hacker News Search**: Search HN stories, comments, Show HN, Ask HN, polls (free Algolia API)
- **PDF Reader**: Extract text and metadata from local or remote PDFs, with page-range filtering
- **SearXNG Search**: Search the web using SearXNG
- **Serply Search**: Google web, news, and scholar search via the Serply API
- **Tavily Search**: AI-powered web search
- **Weather**: Current conditions and daily/hourly forecasts via Open-Meteo (no key required)
- **Webpage Scraper**: Extract content from web pages
- **Wikipedia Search**: Search Wikipedia in any language edition (no key required)
- **YouTube Transcript Scraper**: Extract transcripts from YouTube videos

### Downloading a tool

Use the Atomic Assembler CLI to download tools into your project:

```bash
atomic
```

This presents a menu to select and download tools. Each tool ships with input/output schemas, usage examples, dependencies, and installation instructions.

### Tool layout

Each downloaded tool follows a standard structure:

```
tool_name/
│   .coveragerc
│   pyproject.toml
│   README.md
│   requirements.txt
│   uv.lock
│
├── tool/
│   │   tool_name.py
│   │   some_util_file.py
│
└── tests/
    │   test_tool_name.py
    │   test_some_util_file.py
```

### Calling a downloaded tool

Once a tool is in your project, it's just a Python class:

```python
from calculator.tool.calculator import (
    CalculatorTool,
    CalculatorInputSchema,
    CalculatorToolConfig,
)

calculator = CalculatorTool(config=CalculatorToolConfig())

result = calculator.run(CalculatorInputSchema(expression="2 + 2"))
print(f"Result: {result.value}")  # Result: 4
```

This is Pattern 1 in its simplest form: you call `.run()` directly, no agent involved. The tool is reusable in any Python context — agent, script, test, web handler.

## Creating custom tools

Build your own tool by subclassing `BaseTool` with input/output schemas and a config.

### Basic structure

```python
import os
from pydantic import Field
from atomic_agents import BaseTool, BaseToolConfig, BaseIOSchema

################
# Input Schema #
################

class MyToolInputSchema(BaseIOSchema):
    """Define what your tool accepts as input."""
    value: str = Field(..., description="Input value to process")

#####################
# Output Schema(s)  #
#####################

class MyToolOutputSchema(BaseIOSchema):
    """Define what your tool returns."""
    result: str = Field(..., description="Processed result")

#################
# Configuration #
#################

class MyToolConfig(BaseToolConfig):
    """Tool configuration options."""
    api_key: str = Field(
        default=os.getenv("MY_TOOL_API_KEY"),
        description="API key for the service",
    )

#####################
# Main Tool & Logic #
#####################

class MyTool(BaseTool[MyToolInputSchema, MyToolOutputSchema]):
    """Main tool implementation."""
    input_schema = MyToolInputSchema
    output_schema = MyToolOutputSchema

    def __init__(self, config: MyToolConfig = MyToolConfig()):
        super().__init__(config)
        self.api_key = config.api_key

    def run(self, params: MyToolInputSchema) -> MyToolOutputSchema:
        result = self.process_input(params.value)
        return MyToolOutputSchema(result=result)
```

### Best practices

- **Single responsibility**: Each tool should do one thing well.
- **Clear interfaces**: Use explicit input/output schemas with `Field(..., description=...)` — those descriptions become the LLM's prompt when the tool is reached via Pattern 2.
- **Error handling**: Validate inputs and return structured errors via the output schema rather than raising opaquely.
- **Documentation**: Include clear usage examples and runtime requirements.
- **Tests**: Tools are pure Python — test them like any other function, no agent needed.
- **Dependencies**: Manually maintain `requirements.txt` with only runtime dependencies.

### Tool requirements

- Inherit from the appropriate base classes:
  - Input/output schemas from `BaseIOSchema`
  - Configuration from `BaseToolConfig`
  - Tool class from `BaseTool[Input, Output]`
- Include proper documentation and usage examples
- Include tests for the tool's pure logic
- Follow the standard directory structure if shipping via the Atomic Forge

## Next steps

1. Browse available tools in the [Atomic Forge directory](https://github.com/eigenwise/atomic-agents/tree/main/atomic-forge).
2. Try Pattern 1 by chaining a query agent into a search tool — the [README's "Chaining Schemas" example](https://github.com/eigenwise/atomic-agents#chaining-schemas-and-agents) is a good starting point.
3. Try Pattern 2 by running the [orchestration-agent example](https://github.com/eigenwise/atomic-agents/tree/main/atomic-examples/orchestration-agent).
4. Build your own tool and contribute it back via the Atomic Forge.
