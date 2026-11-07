# Progressive Disclosure Example

This example demonstrates **Anthropic's "progressive disclosure" pattern** for efficient MCP tool loading using the Atomic Agents framework with **three MCP servers** and **24 total tools**.

## The Problem

As documented by [Anthropic's Engineering Blog](https://www.anthropic.com/engineering/code-execution-with-mcp):

- **Context window bloat**: Loading all tool definitions upfront consumes massive context space
- **Performance degradation**: Agents connecting to 2-3+ MCP servers see significant accuracy drops
- **Cost inefficiency**: Traditional approach for multi-server setup: ~25,000+ tokens just for tool schemas

## The Solution: Progressive Disclosure

Instead of loading all 24 tool definitions upfront, a **sub-agent discovers relevant tools on-demand**:

```
┌─────────────────────────────────────────────────────────────────┐
│ WITHOUT Progressive Disclosure                                  │
│                                                                 │
│ Agent Context Window:                                          │
│ ┌─────────────────────────────────────────────────────────────┐│
│ │ math-server: 8 tools × ~500 tokens = 4,000 tokens          ││
│ │ text-server: 8 tools × ~500 tokens = 4,000 tokens          ││
│ │ data-server: 8 tools × ~500 tokens = 4,000 tokens          ││
│ │ ─────────────────────────────────────────────────           ││
│ │ Total: ~12,000 tokens just for tool definitions!           ││
│ └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│ WITH Progressive Disclosure                                     │
│                                                                 │
│ Agent Context Window:                                          │
│ ┌─────────────────────────────────────────────────────────────┐│
│ │ add_numbers (500 tokens)                                    ││
│ │ multiply_numbers (500 tokens)                               ││
│ │ ─────────────────────────────────────────────────           ││
│ │ Total: ~1,000 tokens (92% reduction!)                      ││
│ └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
progressive-disclosure/
├── pyproject.toml
├── README.md
├── servers/                           # Three MCP servers
│   ├── math_server/                   # 8 arithmetic tools
│   │   ├── pyproject.toml
│   │   └── math_server/
│   │       ├── __init__.py
│   │       └── server.py              # FastMCP server
│   ├── text_server/                   # 8 text manipulation tools
│   │   ├── pyproject.toml
│   │   └── text_server/
│   │       ├── __init__.py
│   │       └── server.py
│   └── data_server/                   # 8 list/data tools
│       ├── pyproject.toml
│       └── data_server/
│           ├── __init__.py
│           └── server.py
└── progressive_disclosure/            # Client with progressive disclosure
    ├── __init__.py
    ├── main.py                        # Entry point
    ├── registry/
    │   └── tool_registry.py           # Lightweight tool metadata
    ├── tools/
    │   └── search_tools.py            # Tool search functionality
    └── agents/
        ├── tool_finder_agent.py       # Sub-agent for discovery
        └── orchestrator_agent.py      # Dynamic orchestrator factory
```

## Available Tools (24 Total)

### math-server (8 tools)
| Tool | Description |
|------|-------------|
| `add_numbers` | Add two numbers (a + b) |
| `subtract_numbers` | Subtract b from a (a - b) |
| `multiply_numbers` | Multiply two numbers (a * b) |
| `divide_numbers` | Divide a by b (a / b) |
| `power` | Raise base to exponent |
| `square_root` | Calculate square root |
| `modulo` | Calculate remainder (a % b) |
| `absolute_value` | Get absolute value |

### text-server (8 tools)
| Tool | Description |
|------|-------------|
| `uppercase` | Convert to UPPERCASE |
| `lowercase` | Convert to lowercase |
| `reverse_text` | Reverse character order |
| `word_count` | Count words in text |
| `char_count` | Count characters |
| `concatenate` | Join two strings |
| `replace_text` | Find and replace |
| `split_text` | Split by delimiter |

### data-server (8 tools)
| Tool | Description |
|------|-------------|
| `sort_list` | Sort numbers in a list |
| `filter_greater_than` | Filter values > threshold |
| `filter_less_than` | Filter values < threshold |
| `sum_list` | Sum all values |
| `average_list` | Calculate average |
| `min_value` | Find minimum |
| `max_value` | Find maximum |
| `unique_values` | Remove duplicates |

## Architecture

```
User Query: "Calculate (5 + 3) * 2 and reverse 'hello'"
    │
    ▼
┌─────────────────────────────────────────────────────┐
│     Phase 1: Tool Discovery                         │
│     ─────────────────────────                       │
│     Tool Finder Agent (gpt-5.6-terra)                │
│     - Searches lightweight registry                 │
│     - Registry has 24 tool names + descriptions    │
│     - Returns: ["add_numbers", "multiply_numbers",  │
│                 "reverse_text"]                     │
└─────────────┬───────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│     Phase 2: Dynamic Orchestrator Creation          │
│     ─────────────────────────                       │
│     OrchestratorFactory                            │
│     - Loads ONLY 3 tool schemas (not 24!)          │
│     - Creates Union type dynamically               │
│     - 92% context reduction achieved               │
└─────────────┬───────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────┐
│     Phase 3: Query Execution                        │
│     ─────────────────────────                       │
│     Main Orchestrator Agent (gpt-4o)               │
│     - Executes add_numbers(5, 3) → 8               │
│     - Executes multiply_numbers(8, 2) → 16         │
│     - Executes reverse_text("hello") → "olleh"     │
│     - Returns final response                        │
└─────────────────────────────────────────────────────┘
```

## Getting Started

### Prerequisites

- Python 3.12+
- OpenAI API key
- uv package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/eigenwise/atomic-agents
cd atomic-agents/atomic-examples/progressive-disclosure

# Install dependencies
uv sync
```

### Configuration

Create a `.env` file:
```bash
OPENAI_API_KEY=your-api-key-here
```

### Running the Demo

```bash
uv run python -m progressive_disclosure.main
```

## Example Session

```
╭──────────────────────────────────────────────────────╮
│ Progressive Disclosure Demo                          │
│ Demonstrating Anthropic's pattern with 3 MCP servers │
╰──────────────────────────────────────────────────────╯

Connecting to MCP servers...
Connecting to math-server...
  Connected: 8 tools
Connecting to text-server...
  Connected: 8 tools
Connecting to data-server...
  Connected: 8 tools

Total: 24 tools across 3 servers

Ready! Type '/exit' to quit, '/stats' for statistics.
Example queries:
  - 'Calculate (5 + 3) * 2'              (math tools)
  - 'Convert HELLO WORLD to lowercase'  (text tools)
  - 'Find the average of [1,2,3,4,5]'   (data tools)
  - 'Reverse the text ABC and add 10+5' (multi-server!)

You: Calculate (5 + 3) * 2

Phase 1: Tool Discovery
Sub-agent searching 24 tools across 3 servers...
Selected 2 tools: ['add_numbers', 'multiply_numbers']
Reasoning: The query requires addition and multiplication operations

Phase 2: Creating Focused Orchestrator
Orchestrator context: 2 tools (filtered 92% = saved ~11000 tokens)

Phase 3: Query Execution
Executing: add_numbers
Parameters: {'a': 5, 'b': 3}
Executing: multiply_numbers
Parameters: {'a': 8, 'b': 2}

Response: The result of (5 + 3) * 2 is 16.

╭──────────────────────────────────────────────────────╮
│ Progressive Disclosure: 2/24 tools loaded (92%)     │
╰──────────────────────────────────────────────────────╯
```

## Key Benefits

| Metric | Without PD | With PD | Improvement |
|--------|-----------|---------|-------------|
| Tools in context | 24 | 2-5 | 90%+ reduction |
| Token usage | ~12,000 | ~1,000 | 92% savings |
| Tool accuracy | Lower | Higher | Better focus |
| Scalability | Limited | Excellent | Many servers |

## How Atomic Agents Enables This

This example demonstrates several Atomic Agents patterns:

1. **Sub-Agent Pattern**: Tool Finder as specialized discovery agent
2. **Dynamic Schema Creation**: `Union` types built at runtime from selected tools
3. **Multi-Server MCP**: Connecting to multiple MCP servers simultaneously
4. **Tool Registry**: Lightweight metadata storage without full schemas
5. **Context Efficiency**: Only relevant information loaded

## The FastMCP Servers

Each server is a simple FastMCP application:

```python
from fastmcp import FastMCP

mcp = FastMCP("math-server")

@mcp.tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together (a + b)."""
    return a + b

# ... more tools ...

if __name__ == "__main__":
    mcp.run()
```

## References

- [Anthropic: Code Execution with MCP](https://www.anthropic.com/engineering/code-execution-with-mcp)
- [FastMCP Documentation](https://gofastmcp.com)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [Atomic Agents Documentation](https://github.com/eigenwise/atomic-agents)

## See Also

- [MCP Agent Example](../mcp-agent/) - Basic single-server MCP integration
- [Orchestration Agent Example](../orchestration-agent/) - Tool orchestration patterns
- [Deep Research Example](../deep-research/) - Multi-agent pipelines
