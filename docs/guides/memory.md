# Memory and Context Management

This guide covers everything you need to know about managing conversation memory and dynamic context in Atomic Agents. Whether you're building a simple chatbot or orchestrating complex multi-agent systems, understanding memory management is essential.

```{contents}
:local:
:depth: 2
```

## Introduction

### What You'll Learn

- How conversation history works in Atomic Agents
- What "turns" are and how they're tracked
- How messages are automatically managed during agent execution
- How to persist and restore conversation state
- How to use context providers for dynamic information injection
- Advanced multi-agent memory patterns

### Prerequisites

- Basic familiarity with Atomic Agents ([Quickstart Guide](quickstart.md))
- Understanding of Python classes and async/await

### The Problem This Solves

A common question from developers (see [GitHub Issue #58](https://github.com/eigenwise/atomic-agents/issues/58)):

> "In most of the examples only the initial message is added, not any subsequent runs. Is this automatic?"

**Yes, it is automatic!** When you call `agent.run(user_input)`, the framework automatically:
1. Adds your input to the conversation history
2. Sends the full history to the LLM
3. Adds the LLM's response to history

This guide explains exactly how this works and how to leverage it for complex use cases.

---

## Understanding Memory in Atomic Agents

### The Conversation Model

Atomic Agents uses a **turn-based conversation model** where each interaction between user and assistant forms a "turn". The `ChatHistory` class manages this conversation state.

```{mermaid}
flowchart LR
    subgraph Turn1["Turn 1 (turn_id: abc-123)"]
        U1[User Message]
        A1[Assistant Response]
    end

    subgraph Turn2["Turn 2 (turn_id: def-456)"]
        U2[User Message]
        A2[Assistant Response]
    end

    subgraph Turn3["Turn 3 (turn_id: ghi-789)"]
        U3[User Message]
        A3[Assistant Response]
    end

    U1 --> A1
    A1 -.-> U2
    U2 --> A2
    A2 -.-> U3
    U3 --> A3
```

**Key Concepts:**
- **Message**: A single piece of content with a role (user, assistant, system)
- **Turn**: A logical grouping of related messages (typically user input + assistant response)
- **Turn ID**: A UUID that links messages belonging to the same turn
- **History**: The complete sequence of messages in a conversation

### Messages and Turns

Each message in the history has three components:

```python
from atomic_agents.context import Message

# Message structure
message = Message(
    role="user",           # "user", "assistant", or "system"
    content=some_schema,   # Must be a BaseIOSchema instance
    turn_id="abc-123"      # UUID linking related messages
)
```

**Why Turn IDs Matter:**
- Group related messages together
- Enable deletion of complete turns (user message + response)
- Track conversation flow for debugging
- Support conversation branching patterns

---

## ChatHistory Fundamentals

### Creating and Configuring History

```python
from atomic_agents.context import ChatHistory

# Basic history (unlimited messages)
history = ChatHistory()

# History with message limit (oldest messages removed when exceeded)
history = ChatHistory(max_messages=50)
```

### Using History with an Agent

```python
import instructor
import openai
from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from atomic_agents.context import ChatHistory
from pydantic import Field

# Define schemas
class ChatInput(BaseIOSchema):
    """User chat message"""
    message: str = Field(..., description="The user's message")

class ChatOutput(BaseIOSchema):
    """Assistant response"""
    response: str = Field(..., description="The assistant's response")

# Create history and agent
history = ChatHistory(max_messages=100)
client = instructor.from_openai(openai.OpenAI())

agent = AtomicAgent[ChatInput, ChatOutput](
    config=AgentConfig(
        client=client,
        model="gpt-5-mini",
        history=history,
    )
)

# Each run automatically manages history
response1 = agent.run(ChatInput(message="Hello!"))
response2 = agent.run(ChatInput(message="What did I just say?"))
# The agent remembers the previous message!
```

### The Turn Lifecycle

```{mermaid}
stateDiagram-v2
    [*] --> NoTurn: ChatHistory created
    NoTurn --> ActiveTurn: initialize_turn() called
    NoTurn --> ActiveTurn: add_message() called
    ActiveTurn --> ActiveTurn: add_message() same turn
    ActiveTurn --> NewTurn: initialize_turn() called
    NewTurn --> ActiveTurn: Generates new UUID
    ActiveTurn --> NoTurn: All turns deleted

    note right of ActiveTurn: current_turn_id = UUID
    note right of NoTurn: current_turn_id = None
```

**Turn Lifecycle Methods:**

```python
# Initialize a new turn (generates new UUID)
history.initialize_turn()

# Get the current turn ID
turn_id = history.get_current_turn_id()
print(f"Current turn: {turn_id}")  # e.g., "abc-123-def-456"

# Add a message to the current turn
history.add_message("user", ChatInput(message="Hello"))

# Messages added without initialize_turn() use the existing turn
# or auto-initialize if no turn exists
```

---

## Automatic Memory Management

This section addresses the core question from GitHub Issue #58: **How does automatic message management work?**

### How .run() Manages Memory

When you call `agent.run(user_input)`, here's exactly what happens:

```{mermaid}
flowchart TD
    A["agent.run(user_input)"] --> B{user_input<br/>provided?}
    B -->|Yes| C["history.initialize_turn()<br/>Creates new UUID"]
    C --> D["history.add_message('user', user_input)<br/>Stores user message"]
    B -->|No| E["Skip turn initialization<br/>Use existing history"]
    D --> F["_prepare_messages()<br/>Build message list"]
    E --> F
    F --> G["System prompt + history"]
    G --> H["LLM API call"]
    H --> I["Receive response"]
    I --> J["history.add_message('assistant', response)<br/>Stores response"]
    J --> K["_manage_overflow()<br/>Trim if needed"]
    K --> L["Return response"]

    style C fill:#e1f5fe
    style D fill:#e1f5fe
    style J fill:#e1f5fe
```

### Step-by-Step Trace

Let's trace through a complete conversation:

```python
from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from atomic_agents.context import ChatHistory
from pydantic import Field

class Input(BaseIOSchema):
    """Input"""
    text: str = Field(...)

class Output(BaseIOSchema):
    """Output"""
    reply: str = Field(...)

# Create agent with history
history = ChatHistory()
agent = AtomicAgent[Input, Output](config=AgentConfig(
    client=client,
    model="gpt-5-mini",
    history=history
))

# --- TURN 1 ---
print(f"Before run: {history.get_message_count()} messages")  # 0 messages

response1 = agent.run(Input(text="Hi, my name is Alice"))
# Internally:
# 1. history.initialize_turn() -> turn_id = "abc-123"
# 2. history.add_message("user", Input(text="Hi..."))
# 3. LLM called with history
# 4. history.add_message("assistant", Output(reply="Hello Alice!"))

print(f"After run 1: {history.get_message_count()} messages")  # 2 messages
print(f"Turn ID: {history.get_current_turn_id()}")  # "abc-123"

# --- TURN 2 ---
response2 = agent.run(Input(text="What's my name?"))
# Internally:
# 1. history.initialize_turn() -> turn_id = "def-456" (NEW turn)
# 2. history.add_message("user", Input(text="What's..."))
# 3. LLM called with FULL history (all 4 messages)
# 4. history.add_message("assistant", Output(reply="Your name is Alice!"))

print(f"After run 2: {history.get_message_count()} messages")  # 4 messages
print(f"Turn ID: {history.get_current_turn_id()}")  # "def-456"
```

### Running Without Input

You can call `.run()` without input to continue within the same turn:

```python
# First call with input - starts new turn
response = agent.run(Input(text="Start a story"))

# Subsequent call without input - same turn continues
# Useful for: tool follow-ups, multi-step reasoning
continuation = agent.run()  # No new turn created, uses existing history
```

### Streaming and Async Behavior

All execution methods handle memory the same way:

| Method | Memory Behavior |
|--------|-----------------|
| `agent.run(input)` | Automatic turn init + message add |
| `agent.run_stream(input)` | Same as run(), streams response |
| `agent.run_async(input)` | Same as run(), async execution |
| `agent.run_async_stream(input)` | Same as run(), async + streaming |

```python
# Streaming example - memory works identically
async for chunk in agent.run_async_stream(Input(text="Hello")):
    print(chunk.reply, end="", flush=True)
# History is updated with complete response after stream finishes
```

---

## History Persistence and Management

### Serialization: Saving Conversations

Save conversation history to disk or database:

```python
from atomic_agents.context import ChatHistory

# ... after some conversation ...

# Serialize to JSON string
serialized = history.dump()

# Save to file
with open("conversation.json", "w") as f:
    f.write(serialized)

# Save to database
db.save_conversation(user_id=123, data=serialized)
```

### Deserialization: Restoring Conversations

```python
# Load from file
with open("conversation.json", "r") as f:
    serialized = f.read()

# Create new history and load
history = ChatHistory()
history.load(serialized)

# Use with agent
agent = AtomicAgent[Input, Output](config=AgentConfig(
    client=client,
    model="gpt-5-mini",
    history=history,  # Restored history!
))

# Continue the conversation where it left off
response = agent.run(Input(text="Where were we?"))
```

```{warning}
Only load serialized data from trusted sources. The `load()` method reconstructs Python classes from the serialized data.
```

### Overflow Management

Control memory usage with `max_messages`:

```python
# Keep only last 20 messages
history = ChatHistory(max_messages=20)

# When 21st message is added, oldest message is removed
# This is FIFO (First In, First Out) - oldest messages go first
```

**Strategy for Long Conversations:**

```python
# Option 1: Simple limit
history = ChatHistory(max_messages=50)

# Option 2: Monitor and handle manually
if history.get_message_count() > 40:
    # Maybe summarize old messages before they're lost
    old_messages = history.get_history()[:10]
    summary = summarize_messages(old_messages)
    # Store summary in context provider instead
```

### History Manipulation

**Copying History:**

```python
# Create independent copy (deep copy)
history_copy = history.copy()

# Modifications don't affect original
history_copy.add_message("user", Input(text="This only goes in copy"))
```

**Deleting Turns:**

```python
# Get the turn ID you want to delete
turn_id = history.get_current_turn_id()

# Delete all messages with that turn ID
history.delete_turn_id(turn_id)

# Useful for: removing failed attempts, undo functionality
```

**Resetting History:**

```python
# Clear all messages, start fresh
agent.reset_history()
# or
history = ChatHistory()  # Create new instance
```

---

## Writing a Custom Memory Backend

Memory is intentionally kept outside the framework core. `ChatHistory` is a solid in-memory default, but if you want conversations to survive a restart, live in Redis, Postgres, or some hosted memory service, that integration belongs in **your** codebase, not as a framework dependency. `BaseChatHistory` is the stable seam that makes this possible: build against it and `AtomicAgent` doesn't care what's underneath.

### The Contract

`BaseChatHistory` is an interface-only abstract base class (`ABC`). It declares no state and no behavior of its own, just the methods a history implementation must provide:

- `initialize_turn(self) -> None`
- `add_message(self, role: str, content: BaseIOSchema) -> None`
- `get_history(self) -> List[Dict]`
- `get_current_turn_id(self) -> Optional[str]`
- `delete_turn_id(self, turn_id: str)`
- `get_message_count(self) -> int`
- `dump(self) -> str`
- `load(self, serialized_data: str) -> None`
- `copy(self) -> "BaseChatHistory"`

Every implementation also has to maintain two attributes, because `AtomicAgent` reads them directly:

- `history: List[Message]` - the in-memory working list of messages. `AtomicAgent._trim_context` reads this list (via each message's `turn_id`) to decide what to trim when a context-length limit is hit.
- `current_turn_id: Optional[str]` - the ID of the current turn, or `None` if no turn has started yet. Set by `initialize_turn()`.

### Recommended Pattern: Subclass ChatHistory

Don't implement `BaseChatHistory` from scratch unless you have a good reason to. Subclassing `ChatHistory` gets you turn handling and serialization for free, you only need to override the couple of methods that touch your storage layer:

```python
from atomic_agents.context import ChatHistory

class PersistentHistory(ChatHistory):
    def __init__(self, session_id, store, max_messages=None):
        super().__init__(max_messages=max_messages)
        self.session_id = session_id
        self.store = store
        saved = store.get(session_id)
        if saved:
            self.load(saved)

    def add_message(self, role, content):
        super().add_message(role, content)
        self.store.put(self.session_id, self.dump())

    def copy(self):
        # AtomicAgent calls copy() for its initial_history snapshot and reset_history().
        # Without this override, copy() returns a plain ChatHistory and your backend is
        # silently dropped after reset_history() (later writes stop reaching the store).
        new_history = PersistentHistory(self.session_id, self.store, max_messages=self.max_messages)
        new_history.load(self.dump())
        new_history.current_turn_id = self.current_turn_id
        return new_history
```

The core of the pattern is small: load on init if there's saved state, persist on every `add_message`. `dump()`/`load()` already round-trip the full history, so there's no serialization logic to write yourself.

```{important}
Override `copy()` too. `AtomicAgent` snapshots `initial_history` with `copy()` at construction and restores it in `reset_history()`. The built-in `ChatHistory.copy()` hard-codes a plain `ChatHistory`, so a subclass that carries extra state (a store handle, a `session_id`) that forgets to override `copy()` will be **silently replaced by a non-persistent in-memory history** the first time `reset_history()` runs. Any subclass with its own state must override `copy()` to return its own type.
```

```{note}
This pattern re-serializes and rewrites the **entire** history on every message (`self.dump()`), which is fine for a demo but O(n²) over a long conversation. A production backend against Redis/Postgres/etc. should persist incrementally (append the new message) rather than dumping the whole history each write.
```

### From-Scratch Alternative

If you don't want the in-memory list at all (say your backend is a query, not a cache), implement `BaseChatHistory` directly. You give up the free turn tracking and `get_history()` serialization that `ChatHistory` provides, so you own all of it: generating turn IDs, building the wire-format list of dicts, and keeping `history` / `current_turn_id` in sync with whatever `AtomicAgent` expects to read.

### Try It

There's a full runnable example of this pattern (a dependency-free SQLite-backed history that survives across process runs) at [`atomic-examples/persistent-memory`](https://github.com/eigenwise/atomic-agents/tree/main/atomic-examples/persistent-memory).

---

## Multimodal Content in History

ChatHistory supports images, PDFs, and audio through Instructor's multimodal types.

### Adding Multimodal Messages

```python
from instructor import Image, PDF, Audio
from atomic_agents import BaseIOSchema
from pydantic import Field
from typing import List

class ImageAnalysisInput(BaseIOSchema):
    """Input with images for analysis"""
    question: str = Field(..., description="Question about the images")
    images: List[Image] = Field(..., description="Images to analyze")

# Create input with images
input_with_images = ImageAnalysisInput(
    question="What's in these images?",
    images=[
        Image.from_path("photo1.jpg"),
        Image.from_path("photo2.png"),
    ]
)

# Run agent - images are stored in history
response = agent.run(input_with_images)
```

### Multimodal Message Structure

When history contains multimodal content, `get_history()` returns a special structure:

```python
history_data = history.get_history()

for message in history_data:
    if isinstance(message["content"], list):
        # Multimodal message
        json_content = message["content"][0]  # Text/JSON data
        multimodal_objects = message["content"][1:]  # Images, PDFs, etc.
    else:
        # Text-only message
        json_content = message["content"]
```

### Serialization with Multimodal

```{note}
Multimodal content with file paths is serialized by path. Ensure files exist at the same paths when loading.
```

```python
# Serialize (file paths are preserved)
serialized = history.dump()

# When loading, files must be accessible at original paths
history.load(serialized)
```

---

## Dynamic Context with Providers

Context providers inject dynamic information into agent system prompts at runtime, complementing the static conversation history.

### Understanding the Difference

| Aspect | ChatHistory (Memory) | Context Providers |
|--------|---------------------|-------------------|
| **Purpose** | Store conversation turns | Inject dynamic context |
| **Location** | Message history | System prompt |
| **Persistence** | Saved with history | Regenerated each call |
| **Use Case** | Conversation continuity | Real-time data (RAG, user info, time) |

```{mermaid}
flowchart TB
    subgraph SystemPrompt["System Prompt (sent to LLM)"]
        BG[Background Instructions]
        ST[Steps]

        subgraph DC["Dynamic Context"]
            CP1[Context Provider 1]
            CP2[Context Provider 2]
            CP3[Context Provider 3]
        end

        OI[Output Instructions]
    end

    subgraph Messages["Conversation Messages"]
        H[ChatHistory Messages]
    end

    SystemPrompt --> LLM
    Messages --> LLM
    LLM --> Response
```

### Creating Custom Context Providers

```python
from atomic_agents.context import BaseDynamicContextProvider

class UserContextProvider(BaseDynamicContextProvider):
    """Provides current user information to the agent."""

    def __init__(self):
        super().__init__(title="Current User")
        self.user_name: str = ""
        self.user_role: str = ""
        self.preferences: dict = {}

    def get_info(self) -> str:
        """Called every time the agent runs."""
        if not self.user_name:
            return "No user logged in."

        info = f"User: {self.user_name} (Role: {self.user_role})"
        if self.preferences:
            prefs = ", ".join(f"{k}: {v}" for k, v in self.preferences.items())
            info += f"\nPreferences: {prefs}"
        return info
```

### Registering Context Providers

```python
from atomic_agents import AtomicAgent, AgentConfig
from atomic_agents.context import SystemPromptGenerator

# Create provider
user_provider = UserContextProvider()

# Option 1: Register with SystemPromptGenerator
system_prompt = SystemPromptGenerator(
    background=["You are a helpful assistant."],
    context_providers={"user": user_provider}
)

agent = AtomicAgent[Input, Output](config=AgentConfig(
    client=client,
    model="gpt-5-mini",
    system_prompt_generator=system_prompt,
))

# Option 2: Register after agent creation
agent.register_context_provider("user", user_provider)

# Update provider state before running
user_provider.user_name = "Alice"
user_provider.user_role = "Admin"

# Now the agent knows about Alice!
response = agent.run(Input(text="What can I do?"))
```

### Common Context Provider Patterns

**RAG (Retrieval-Augmented Generation):**

```python
class RAGContextProvider(BaseDynamicContextProvider):
    """Injects retrieved documents into the prompt."""

    def __init__(self, vector_db):
        super().__init__(title="Relevant Documents")
        self.vector_db = vector_db
        self.current_query: str = ""
        self._cached_results: list = []

    def search(self, query: str, top_k: int = 3):
        """Call before agent.run() to update context."""
        self.current_query = query
        self._cached_results = self.vector_db.search(query, top_k=top_k)

    def get_info(self) -> str:
        if not self._cached_results:
            return "No relevant documents found."

        docs = []
        for i, doc in enumerate(self._cached_results, 1):
            docs.append(f"Document {i}:\n{doc['content']}\nSource: {doc['source']}")
        return "\n\n".join(docs)

# Usage
rag_provider = RAGContextProvider(vector_db)
agent.register_context_provider("documents", rag_provider)

# Before each query
user_query = "How do I reset my password?"
rag_provider.search(user_query)  # Update context
response = agent.run(Input(text=user_query))
```

**Time-Aware Context:**

```python
from datetime import datetime

class TimeContextProvider(BaseDynamicContextProvider):
    """Provides current time information."""

    def __init__(self):
        super().__init__(title="Current Time")

    def get_info(self) -> str:
        now = datetime.now()
        return f"Current date/time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
```

**Session Context:**

```python
class SessionContextProvider(BaseDynamicContextProvider):
    """Tracks session-specific state."""

    def __init__(self):
        super().__init__(title="Session State")
        self.data: dict = {}

    def set(self, key: str, value: str):
        self.data[key] = value

    def get_info(self) -> str:
        if not self.data:
            return "No session data."
        return "\n".join(f"- {k}: {v}" for k, v in self.data.items())
```

---

## Multi-Agent Memory Patterns

This section addresses the question from GitHub Issue #58:

> "How do I handle a scenario where one agent performs an action, a second agent evaluates it, and then passes results back to the first agent's memory?"

Here are five patterns for managing memory across multiple agents.

### Pattern 1: Shared History

Multiple agents share the same `ChatHistory` instance, seeing each other's messages.

```{mermaid}
flowchart LR
    subgraph SharedHistory["Shared ChatHistory"]
        M1[Message 1]
        M2[Message 2]
        M3[Message 3]
        M4[Message 4]
    end

    A1[Agent A] --> SharedHistory
    A2[Agent B] --> SharedHistory
    A3[Agent C] --> SharedHistory
```

**Use Case:** Agents that need full conversation context (e.g., specialist + generalist).

```python
from atomic_agents import AtomicAgent, AgentConfig
from atomic_agents.context import ChatHistory

# One history shared by all
shared_history = ChatHistory()

# Agent A - Technical Expert
technical_agent = AtomicAgent[Input, Output](config=AgentConfig(
    client=client,
    model="gpt-5-mini",
    history=shared_history,  # Same history
    system_prompt_generator=SystemPromptGenerator(
        background=["You are a technical expert."]
    ),
))

# Agent B - Communication Expert
communication_agent = AtomicAgent[Input, Output](config=AgentConfig(
    client=client,
    model="gpt-5-mini",
    history=shared_history,  # Same history!
    system_prompt_generator=SystemPromptGenerator(
        background=["You simplify technical explanations."]
    ),
))

# Conversation flow
user_input = Input(text="Explain quantum computing")

# Technical agent adds to shared history
technical_response = technical_agent.run(user_input)

# Communication agent sees technical response in history
simple_response = communication_agent.run(
    Input(text="Simplify the above explanation for a child")
)
```

### Pattern 2: Independent Histories

Each agent maintains its own isolated history.

```{mermaid}
flowchart TB
    subgraph Agent_A["Agent A"]
        HA[History A]
    end

    subgraph Agent_B["Agent B"]
        HB[History B]
    end

    subgraph Agent_C["Agent C"]
        HC[History C]
    end

    User --> Agent_A
    User --> Agent_B
    User --> Agent_C
```

**Use Case:** Parallel processing, independent tasks, privacy isolation.

```python
# Each agent has its own history
agent_a = AtomicAgent[Input, Output](config=AgentConfig(
    client=client,
    model="gpt-5-mini",
    history=ChatHistory(),  # Independent
))

agent_b = AtomicAgent[Input, Output](config=AgentConfig(
    client=client,
    model="gpt-5-mini",
    history=ChatHistory(),  # Independent
))

# They don't see each other's conversations
response_a = agent_a.run(Input(text="Research topic A"))
response_b = agent_b.run(Input(text="Research topic B"))
```

(pattern-3-agent-to-agent-messaging)=
### Pattern 3: Agent-to-Agent Messaging

Manually transfer outputs between agent memories. **This directly addresses Issue #58.**

```{mermaid}
sequenceDiagram
    participant U as User
    participant O as Orchestrator
    participant A as Agent A
    participant B as Agent B

    U->>O: Initial request
    O->>A: run(user_input)
    Note over A: Turn 1: User + Response<br/>added to A.history
    A-->>O: Result A

    O->>O: Manual transfer
    Note over O: B.history.add_message(<br/>"user", Result A)

    O->>B: run(None)
    Note over B: Uses existing history<br/>Turn 2: Response added
    B-->>O: Result B

    O->>O: Manual transfer
    Note over O: A.history.add_message(<br/>"user", Result B)

    O->>A: run(None)
    Note over A: Continues with<br/>B's feedback in context
    A-->>O: Final Result
```

**Use Case:** Agent loops, evaluation cycles, iterative refinement.

```python
from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from atomic_agents.context import ChatHistory
from pydantic import Field

class WriterInput(BaseIOSchema):
    """Writer input"""
    task: str = Field(...)

class WriterOutput(BaseIOSchema):
    """Writer output"""
    content: str = Field(...)

class ReviewerInput(BaseIOSchema):
    """Reviewer input"""
    content_to_review: str = Field(...)

class ReviewerOutput(BaseIOSchema):
    """Reviewer output"""
    feedback: str = Field(...)
    approved: bool = Field(...)

# Create agents with independent histories
writer = AtomicAgent[WriterInput, WriterOutput](config=AgentConfig(
    client=client,
    model="gpt-5-mini",
    history=ChatHistory(),
))

reviewer = AtomicAgent[ReviewerInput, ReviewerOutput](config=AgentConfig(
    client=client,
    model="gpt-5-mini",
    history=ChatHistory(),
))

def iterative_writing(task: str, max_iterations: int = 3) -> str:
    """Writer-Reviewer loop with memory transfer."""

    # Initial writing
    writer_response = writer.run(WriterInput(task=task))

    for i in range(max_iterations):
        # Review the content
        review = reviewer.run(ReviewerInput(
            content_to_review=writer_response.content
        ))

        if review.approved:
            return writer_response.content

        # Transfer feedback to writer's memory
        # This is the key pattern from Issue #58!
        writer.history.add_message(
            "user",
            WriterInput(task=f"Revise based on feedback: {review.feedback}")
        )

        # Writer continues with feedback in context
        writer_response = writer.run()  # No input = use existing history

    return writer_response.content

# Usage
final_content = iterative_writing("Write a product description for headphones")
```

### Pattern 4: Supervisor-Worker with Context Providers

Use context providers to share state between supervisor and worker agents.

```{mermaid}
flowchart TB
    subgraph SharedContext["Shared Context Provider"]
        SC[Task State & Results]
    end

    SUP[Supervisor Agent] --> SharedContext
    W1[Worker 1] --> SharedContext
    W2[Worker 2] --> SharedContext
    W3[Worker 3] --> SharedContext

    SUP -->|Delegates| W1
    SUP -->|Delegates| W2
    SUP -->|Delegates| W3

    W1 -->|Updates| SharedContext
    W2 -->|Updates| SharedContext
    W3 -->|Updates| SharedContext
```

```python
class TaskContextProvider(BaseDynamicContextProvider):
    """Shared context for supervisor-worker pattern."""

    def __init__(self):
        super().__init__(title="Task Progress")
        self.current_task: str = ""
        self.subtask_results: dict = {}
        self.overall_status: str = "pending"

    def set_task(self, task: str):
        self.current_task = task
        self.subtask_results = {}
        self.overall_status = "in_progress"

    def add_result(self, subtask: str, result: str):
        self.subtask_results[subtask] = result

    def get_info(self) -> str:
        info = [f"Current Task: {self.current_task}"]
        info.append(f"Status: {self.overall_status}")

        if self.subtask_results:
            info.append("\nCompleted Subtasks:")
            for task, result in self.subtask_results.items():
                info.append(f"  - {task}: {result[:100]}...")

        return "\n".join(info)

# Shared context
task_context = TaskContextProvider()

# All agents see the same context
supervisor = AtomicAgent[Input, Output](config=AgentConfig(
    client=client,
    model="gpt-5-mini",
    history=ChatHistory(),
))
supervisor.register_context_provider("task", task_context)

worker1 = AtomicAgent[Input, Output](config=AgentConfig(
    client=client,
    model="gpt-5-mini",
    history=ChatHistory(),
))
worker1.register_context_provider("task", task_context)

# Orchestration
task_context.set_task("Research and summarize AI trends")

# Worker does subtask
result1 = worker1.run(Input(text="Research NLP trends"))
task_context.add_result("NLP Research", result1.response)

# Supervisor sees worker's result via context provider
summary = supervisor.run(Input(text="Synthesize the research findings"))
```

### Pattern 5: Memory-Augmented Loops

Combine conversation history with external memory for long-running processes.

```python
class LongTermMemory:
    """External memory store for facts and decisions."""

    def __init__(self):
        self.facts: list = []
        self.decisions: list = []

    def add_fact(self, fact: str):
        self.facts.append(fact)

    def add_decision(self, decision: str):
        self.decisions.append(decision)

    def get_summary(self) -> str:
        summary = []
        if self.facts:
            summary.append("Known Facts:\n" + "\n".join(f"- {f}" for f in self.facts))
        if self.decisions:
            summary.append("Decisions Made:\n" + "\n".join(f"- {d}" for d in self.decisions))
        return "\n\n".join(summary) if summary else "No long-term memory yet."

class MemoryContextProvider(BaseDynamicContextProvider):
    def __init__(self, memory: LongTermMemory):
        super().__init__(title="Long-Term Memory")
        self.memory = memory

    def get_info(self) -> str:
        return self.memory.get_summary()

# Setup
long_term = LongTermMemory()
memory_provider = MemoryContextProvider(long_term)

agent = AtomicAgent[Input, Output](config=AgentConfig(
    client=client,
    model="gpt-5-mini",
    history=ChatHistory(max_messages=20),  # Short-term limited
))
agent.register_context_provider("memory", memory_provider)

# Research loop with memory accumulation
topics = ["AI Safety", "Quantum Computing", "Climate Tech"]

for topic in topics:
    response = agent.run(Input(text=f"Research {topic} and identify key facts"))

    # Extract and store important facts in long-term memory
    long_term.add_fact(f"{topic}: {response.response[:200]}")

    # ChatHistory may overflow, but long-term memory persists
    # Agent always has access via context provider

# Final synthesis - agent sees all facts via context provider
final = agent.run(Input(text="Synthesize all research into recommendations"))
```

---

## Best Practices

### When to Use Each Pattern

| Scenario | Recommended Pattern |
|----------|-------------------|
| Single agent chatbot | Basic ChatHistory |
| Multi-turn with context | ChatHistory + Context Providers |
| Parallel independent tasks | Independent Histories |
| Sequential pipeline | Agent-to-Agent Messaging |
| Iterative refinement loops | Agent-to-Agent Messaging |
| Supervisor-worker | Shared Context Providers |
| Long-running processes | Memory-Augmented Loops |

### Managing Context Window Limits

```python
from atomic_agents.utils import get_context_token_count

# Monitor token usage
token_info = agent.get_context_token_count()
print(f"Total tokens: {token_info.total}")
print(f"System prompt: {token_info.system_prompt}")
print(f"History: {token_info.history}")
print(f"Utilization: {token_info.utilization:.1%}")

# Set appropriate limits
if token_info.utilization > 0.8:
    # Consider trimming history or summarizing
    pass
```

### Testing Agents with Memory

```python
import pytest
from atomic_agents.context import ChatHistory

@pytest.fixture
def fresh_history():
    """Provide clean history for each test."""
    return ChatHistory()

@pytest.fixture
def agent_with_history(fresh_history):
    """Agent with clean history."""
    return AtomicAgent[Input, Output](config=AgentConfig(
        client=mock_client,
        model="gpt-5-mini",
        history=fresh_history,
    ))

def test_conversation_continuity(agent_with_history):
    """Test that agent remembers previous messages."""
    agent_with_history.run(Input(text="My name is Bob"))
    response = agent_with_history.run(Input(text="What's my name?"))

    assert "Bob" in response.response

def test_history_persistence(agent_with_history):
    """Test serialization/deserialization."""
    agent_with_history.run(Input(text="Remember: secret=42"))

    # Serialize
    serialized = agent_with_history.history.dump()

    # Create new history and load
    new_history = ChatHistory()
    new_history.load(serialized)

    assert new_history.get_message_count() == 2
```

### Debugging Memory Issues

```python
# Inspect current history
for msg in history.history:
    print(f"[{msg.role}] Turn: {msg.turn_id}")
    print(f"  Content: {msg.content.model_dump_json()[:100]}...")
    print()

# Check turn state
print(f"Current turn ID: {history.get_current_turn_id()}")
print(f"Message count: {history.get_message_count()}")
print(f"Max messages: {history.max_messages}")
```

---

## Troubleshooting

### "Messages aren't being added to history"

**Cause:** Calling `run()` without input after resetting history.

```python
# Wrong - no messages to work with
agent.reset_history()
agent.run()  # Nothing in history!

# Correct
agent.reset_history()
agent.run(Input(text="Start fresh"))  # Provides input
```

### "Agent doesn't remember previous conversation"

**Cause:** Creating new agent instances instead of reusing.

```python
# Wrong - new agent = new history each time
def handle_message(text):
    agent = AtomicAgent[Input, Output](config=config)  # New instance!
    return agent.run(Input(text=text))

# Correct - reuse agent instance
agent = AtomicAgent[Input, Output](config=config)  # Create once

def handle_message(text):
    return agent.run(Input(text=text))  # Reuse
```

### "How do I pass memory between agents?"

See [Pattern 3: Agent-to-Agent Messaging](#pattern-3-agent-to-agent-messaging).

```python
# Transfer output to another agent's memory
agent_b.history.add_message("user", agent_a_output)
agent_b.run()  # Now has context from agent A
```

### "What exactly is a 'turn'?"

A **turn** is a logical unit of conversation, typically containing:
- One user message
- One assistant response
- Both sharing the same `turn_id` (UUID)

```python
# This is ONE turn:
response = agent.run(Input(text="Hello"))
# turn_id "abc-123" assigned to both user message and response

# This starts a NEW turn:
response2 = agent.run(Input(text="Next question"))
# turn_id "def-456" assigned to new pair
```

### "History is too large / context overflow"

```python
# Option 1: Limit history size
history = ChatHistory(max_messages=30)

# Option 2: Monitor and handle
if history.get_message_count() > 40:
    # Summarize or archive old messages
    pass

# Option 3: Use context providers for persistent data
# instead of relying on conversation history
```

---

## API Quick Reference

### ChatHistory

| Method | Description |
|--------|-------------|
| `ChatHistory(max_messages=None)` | Create history with optional limit |
| `add_message(role, content)` | Add message to current turn |
| `initialize_turn()` | Start new turn with new UUID |
| `get_current_turn_id()` | Get current turn's UUID |
| `get_history()` | Get all messages as list of dicts |
| `get_message_count()` | Get number of messages |
| `delete_turn_id(turn_id)` | Delete all messages in a turn |
| `dump()` | Serialize to JSON string |
| `load(data)` | Deserialize from JSON string |
| `copy()` | Create deep copy |

### BaseChatHistory

The abstract base class `ChatHistory` implements. Subclass it (directly, or via `ChatHistory`) to plug in a custom memory backend. See [Writing a Custom Memory Backend](#writing-a-custom-memory-backend).

| Method | Description |
|--------|-------------|
| `initialize_turn()` | Start new turn with new UUID |
| `add_message(role, content)` | Add message to current turn |
| `get_history()` | Get all messages as list of dicts |
| `get_current_turn_id()` | Get current turn's UUID |
| `delete_turn_id(turn_id)` | Delete all messages in a turn |
| `get_message_count()` | Get number of messages |
| `dump()` | Serialize to JSON string |
| `load(data)` | Deserialize from JSON string |
| `copy()` | Create deep copy |

Every implementation must also maintain a `history: List[Message]` attribute and a `current_turn_id: Optional[str]` attribute.

### Message

| Field | Type | Description |
|-------|------|-------------|
| `role` | str | "user", "assistant", or "system" |
| `content` | BaseIOSchema | Message content |
| `turn_id` | Optional[str] | UUID linking related messages |

### BaseDynamicContextProvider

| Method | Description |
|--------|-------------|
| `__init__(title)` | Create with display title |
| `get_info() -> str` | Return context string (override this) |

---

## Next Steps

- [Quickstart Guide](quickstart.md) - Get started with Atomic Agents
- [Tools Guide](tools.md) - Add capabilities to your agents
- [Orchestration Guide](orchestration.md) - Coordinate multiple agents
- [Hooks Guide](hooks.md) - Monitor and customize agent behavior
- [API Reference](/api/context) - Full API documentation

---

## Summary

Key takeaways:

1. **Automatic Memory**: `agent.run(input)` automatically manages history - you don't need to manually add messages
2. **Turns**: A turn groups user input + assistant response with a shared UUID
3. **Persistence**: Use `dump()`/`load()` to save and restore conversations
4. **Context Providers**: Inject dynamic information (RAG, user data, time) into system prompts
5. **Multi-Agent**: Use shared history, agent-to-agent messaging, or context providers depending on your needs

For questions or issues, visit our [GitHub repository](https://github.com/eigenwise/atomic-agents) or [Reddit community](https://www.reddit.com/r/AtomicAgents/).
