# Cookbook

Practical recipes for common Atomic Agents use cases.

## Quick Reference

| Recipe | Description |
|--------|-------------|
| [Basic Chatbot](#basic-chatbot) | Simple conversational agent |
| [Chatbot with Memory](#chatbot-with-memory) | Agent that remembers context |
| [Custom Output Schema](#custom-output-schema) | Structured responses |
| [Multi-Provider Agent](#multi-provider-agent) | Switch between LLM providers |
| [Agent with Tools](#agent-with-tools) | Agent using external tools |
| [Streaming Chatbot](#streaming-chatbot) | Real-time response streaming |
| [Research Agent](#research-agent) | Multi-step research workflow |
| [RAG Agent](#rag-agent) | Retrieval-augmented generation |

## Basic Chatbot

A minimal chatbot implementation.

```python
"""
Basic Chatbot Recipe

A simple conversational agent that responds to user messages.

Requirements:
- pip install atomic-agents openai
- Set OPENAI_API_KEY environment variable
"""

import os
import instructor
import openai
from atomic_agents import AtomicAgent, AgentConfig, BasicChatInputSchema, BasicChatOutputSchema
from atomic_agents.context import ChatHistory


def create_basic_chatbot():
    """Create a basic chatbot agent."""
    client = instructor.from_openai(openai.OpenAI())

    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
        config=AgentConfig(
            client=client,
            model="gpt-5-mini",
            history=ChatHistory()
        )
    )

    return agent


def chat_loop(agent):
    """Interactive chat loop."""
    print("Chatbot ready! Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not user_input:
            continue

        response = agent.run(BasicChatInputSchema(chat_message=user_input))
        print(f"Bot: {response.chat_message}\n")


if __name__ == "__main__":
    agent = create_basic_chatbot()
    chat_loop(agent)
```

## Chatbot with Memory

Agent that maintains conversation history across turns.

```python
"""
Chatbot with Memory Recipe

Demonstrates conversation history and context retention.

Requirements:
- pip install atomic-agents openai
- Set OPENAI_API_KEY environment variable
"""

import os
import instructor
import openai
from atomic_agents import AtomicAgent, AgentConfig, BasicChatInputSchema, BasicChatOutputSchema
from atomic_agents.context import ChatHistory, SystemPromptGenerator


def create_memory_chatbot():
    """Create chatbot with memory and custom personality."""

    client = instructor.from_openai(openai.OpenAI())

    # Initialize history with a greeting
    history = ChatHistory()
    greeting = BasicChatOutputSchema(
        chat_message="Hello! I'm your personal assistant. I'll remember our conversation. How can I help?"
    )
    history.add_message("assistant", greeting)

    # Custom system prompt
    system_prompt = SystemPromptGenerator(
        background=[
            "You are a friendly, helpful personal assistant.",
            "You have an excellent memory and always remember details from the conversation.",
            "You refer back to previous messages when relevant."
        ],
        steps=[
            "Review the conversation history for context.",
            "Provide helpful, personalized responses.",
            "Remember any names, preferences, or facts the user shares."
        ],
        output_instructions=[
            "Be conversational and friendly.",
            "Reference previous context when appropriate.",
            "Ask follow-up questions to engage the user."
        ]
    )

    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
        config=AgentConfig(
            client=client,
            model="gpt-5-mini",
            history=history,
            system_prompt_generator=system_prompt
        )
    )

    return agent


def save_history(agent, filename="chat_history.json"):
    """Save conversation history to file."""
    import json
    history_data = agent.history.dump()
    with open(filename, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"History saved to {filename}")


def load_history(agent, filename="chat_history.json"):
    """Load conversation history from file."""
    import json
    try:
        with open(filename, 'r') as f:
            history_data = json.load(f)
        agent.history.load(history_data)
        print(f"History loaded from {filename}")
    except FileNotFoundError:
        print("No previous history found")


if __name__ == "__main__":
    agent = create_memory_chatbot()

    # Demonstrate memory
    print("Testing memory...")

    response1 = agent.run(BasicChatInputSchema(chat_message="My name is Alice and I love Python"))
    print(f"Bot: {response1.chat_message}\n")

    response2 = agent.run(BasicChatInputSchema(chat_message="What's my name and favorite language?"))
    print(f"Bot: {response2.chat_message}\n")

    # Save for later
    save_history(agent)
```

## Custom Output Schema

Agent with structured output including metadata.

```python
"""
Custom Output Schema Recipe

Agent that returns structured responses with confidence and sources.

Requirements:
- pip install atomic-agents openai
- Set OPENAI_API_KEY environment variable
"""

import os
from typing import List
from pydantic import Field
import instructor
import openai
from atomic_agents import AtomicAgent, AgentConfig, BasicChatInputSchema, BaseIOSchema
from atomic_agents.context import ChatHistory, SystemPromptGenerator


class StructuredOutputSchema(BaseIOSchema):
    """Structured response with metadata."""

    answer: str = Field(..., description="The main answer to the question")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score from 0.0 to 1.0"
    )
    key_points: List[str] = Field(
        default_factory=list,
        description="Key points summarizing the answer"
    )
    follow_up_questions: List[str] = Field(
        default_factory=list,
        description="3 suggested follow-up questions"
    )


def create_structured_agent():
    """Create agent with structured output."""

    client = instructor.from_openai(openai.OpenAI())

    system_prompt = SystemPromptGenerator(
        background=[
            "You are a knowledgeable assistant that provides structured responses.",
            "You always assess your confidence in answers."
        ],
        steps=[
            "Analyze the question thoroughly.",
            "Formulate a clear, accurate answer.",
            "Identify 3-5 key points.",
            "Assess your confidence (0.0-1.0).",
            "Generate 3 relevant follow-up questions."
        ],
        output_instructions=[
            "Provide accurate, well-researched answers.",
            "Be honest about confidence level.",
            "Key points should be concise bullet points.",
            "Follow-up questions should explore the topic deeper."
        ]
    )

    agent = AtomicAgent[BasicChatInputSchema, StructuredOutputSchema](
        config=AgentConfig(
            client=client,
            model="gpt-5-mini",
            history=ChatHistory(),
            system_prompt_generator=system_prompt
        )
    )

    return agent


def display_response(response: StructuredOutputSchema):
    """Pretty-print the structured response."""
    print(f"\n{'='*60}")
    print(f"Answer: {response.answer}")
    print(f"\nConfidence: {response.confidence:.0%}")
    print(f"\nKey Points:")
    for point in response.key_points:
        print(f"  - {point}")
    print(f"\nFollow-up Questions:")
    for i, q in enumerate(response.follow_up_questions, 1):
        print(f"  {i}. {q}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    agent = create_structured_agent()

    response = agent.run(BasicChatInputSchema(
        chat_message="What are the main benefits of using Python for data science?"
    ))

    display_response(response)
```

## Multi-Provider Agent

Switch between different LLM providers dynamically.

```python
"""
Multi-Provider Agent Recipe

Agent that can use different LLM providers based on configuration.

Requirements:
- pip install atomic-agents instructor[anthropic,groq]
- Set API keys for providers you want to use
"""

import os
from enum import Enum
from typing import Optional
import instructor
from atomic_agents import AtomicAgent, AgentConfig, BasicChatInputSchema, BasicChatOutputSchema
from atomic_agents.context import ChatHistory


class Provider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    OLLAMA = "ollama"


def get_client(provider: Provider):
    """Get instructor client for specified provider."""

    if provider == Provider.OPENAI:
        from openai import OpenAI
        return instructor.from_openai(OpenAI()), "gpt-5-mini"

    elif provider == Provider.ANTHROPIC:
        from anthropic import Anthropic
        return instructor.from_anthropic(Anthropic()), "claude-haiku-5-5"

    elif provider == Provider.GROQ:
        from groq import Groq
        return instructor.from_groq(Groq(), mode=instructor.Mode.JSON), "openai/gpt-oss-120b"

    elif provider == Provider.OLLAMA:
        from openai import OpenAI
        client = instructor.from_openai(
            OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
            mode=instructor.Mode.JSON
        )
        return client, "llama3"

    raise ValueError(f"Unknown provider: {provider}")


def create_agent(provider: Provider) -> AtomicAgent:
    """Create agent with specified provider."""
    client, model = get_client(provider)

    return AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
        config=AgentConfig(
            client=client,
            model=model,
            history=ChatHistory()
        )
    )


class MultiProviderAgent:
    """Agent that can switch between providers."""

    def __init__(self, default_provider: Provider = Provider.OPENAI):
        self.current_provider = default_provider
        self.agent = create_agent(default_provider)

    def switch_provider(self, provider: Provider):
        """Switch to a different provider."""
        self.current_provider = provider
        self.agent = create_agent(provider)
        print(f"Switched to {provider.value}")

    def run(self, message: str) -> str:
        """Run agent with current provider."""
        response = self.agent.run(BasicChatInputSchema(chat_message=message))
        return response.chat_message


if __name__ == "__main__":
    agent = MultiProviderAgent(Provider.OPENAI)

    # Use OpenAI
    print(f"Using: {agent.current_provider.value}")
    response = agent.run("Hello! What model are you?")
    print(f"Response: {response}\n")

    # Switch to Groq (if available)
    try:
        agent.switch_provider(Provider.GROQ)
        response = agent.run("Hello! What model are you?")
        print(f"Response: {response}")
    except Exception as e:
        print(f"Could not switch to Groq: {e}")
```

## Agent with Tools

Agent that uses tools to extend capabilities.

```python
"""
Agent with Tools Recipe

Agent that uses a calculator tool for mathematical operations.

Requirements:
- pip install atomic-agents openai sympy
- Set OPENAI_API_KEY environment variable
"""

import os
from pydantic import Field
import instructor
import openai
from atomic_agents import AtomicAgent, AgentConfig, BaseTool, BaseToolConfig, BaseIOSchema
from atomic_agents.context import ChatHistory, SystemPromptGenerator


# Define Calculator Tool
class CalculatorInputSchema(BaseIOSchema):
    """Input for calculator."""
    expression: str = Field(..., description="Mathematical expression to evaluate")


class CalculatorOutputSchema(BaseIOSchema):
    """Output from calculator."""
    result: float = Field(..., description="Calculation result")
    expression: str = Field(..., description="Original expression")


class CalculatorTool(BaseTool[CalculatorInputSchema, CalculatorOutputSchema]):
    """Simple calculator tool."""

    def run(self, params: CalculatorInputSchema) -> CalculatorOutputSchema:
        try:
            # Safe evaluation using sympy
            from sympy import sympify
            result = float(sympify(params.expression))
            return CalculatorOutputSchema(
                result=result,
                expression=params.expression
            )
        except Exception as e:
            raise ValueError(f"Could not evaluate: {params.expression}. Error: {e}")


# Agent output that can use tools
class AgentOutputSchema(BaseIOSchema):
    """Agent response that may include tool usage."""
    message: str = Field(..., description="Response message")
    needs_calculation: bool = Field(
        default=False,
        description="Whether a calculation is needed"
    )
    calculation_expression: str = Field(
        default="",
        description="Expression to calculate if needed"
    )


def create_tool_agent():
    """Create agent with tool capability."""

    client = instructor.from_openai(openai.OpenAI())
    calculator = CalculatorTool()

    system_prompt = SystemPromptGenerator(
        background=[
            "You are a helpful assistant with calculation capabilities.",
            "When the user asks for calculations, indicate what needs to be calculated."
        ],
        steps=[
            "Determine if the request involves mathematical calculation.",
            "If yes, set needs_calculation to true and provide the expression.",
            "Provide a helpful response message."
        ],
        output_instructions=[
            "For math questions, extract the expression to calculate.",
            "Always be helpful and explain your response."
        ]
    )

    agent = AtomicAgent[BasicChatInputSchema, AgentOutputSchema](
        config=AgentConfig(
            client=client,
            model="gpt-5-mini",
            history=ChatHistory(),
            system_prompt_generator=system_prompt
        )
    )

    return agent, calculator


def process_with_tools(agent, calculator, user_message: str) -> str:
    """Process message, using tools as needed."""

    # Get agent response
    response = agent.run(BasicChatInputSchema(chat_message=user_message))

    # Check if calculation is needed
    if response.needs_calculation and response.calculation_expression:
        try:
            calc_result = calculator.run(
                CalculatorInputSchema(expression=response.calculation_expression)
            )
            return f"{response.message}\n\nCalculation: {calc_result.expression} = {calc_result.result}"
        except ValueError as e:
            return f"{response.message}\n\nCalculation error: {e}"

    return response.message


if __name__ == "__main__":
    agent, calculator = create_tool_agent()

    # Test with calculation
    result = process_with_tools(
        agent,
        calculator,
        "What is 15% of 250?"
    )
    print(result)
```

## Streaming Chatbot

Real-time streaming responses.

```python
"""
Streaming Chatbot Recipe

Chatbot that streams responses in real-time.

Requirements:
- pip install atomic-agents openai rich
- Set OPENAI_API_KEY environment variable
"""

import os
import asyncio
import instructor
from openai import AsyncOpenAI
from rich.console import Console
from rich.live import Live
from atomic_agents import AtomicAgent, AgentConfig, BasicChatInputSchema, BasicChatOutputSchema
from atomic_agents.context import ChatHistory


console = Console()


def create_streaming_agent():
    """Create agent configured for streaming."""

    # Use async client for streaming
    client = instructor.from_openai(AsyncOpenAI())

    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
        config=AgentConfig(
            client=client,
            model="gpt-5-mini",
            history=ChatHistory()
        )
    )

    return agent


async def stream_response(agent, message: str):
    """Stream agent response with live display."""

    console.print(f"\n[bold blue]You:[/bold blue] {message}")
    console.print("[bold green]Bot:[/bold green] ", end="")

    with Live("", refresh_per_second=10, console=console) as live:
        current_text = ""
        async for partial in agent.run_async_stream(
            BasicChatInputSchema(chat_message=message)
        ):
            if partial.chat_message:
                current_text = partial.chat_message
                live.update(current_text)

    console.print()  # Newline after response


async def streaming_chat_loop(agent):
    """Interactive streaming chat loop."""

    console.print("[bold]Streaming Chatbot[/bold]")
    console.print("Type 'quit' to exit.\n")

    while True:
        user_input = console.input("[bold blue]You:[/bold blue] ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            console.print("Goodbye!")
            break

        if not user_input:
            continue

        await stream_response(agent, user_input)


if __name__ == "__main__":
    agent = create_streaming_agent()
    asyncio.run(streaming_chat_loop(agent))
```

## Research Agent

Multi-step research workflow.

```python
"""
Research Agent Recipe

Agent that performs multi-step research by generating queries and synthesizing results.

Requirements:
- pip install atomic-agents openai
- Set OPENAI_API_KEY environment variable
"""

import os
from typing import List
from pydantic import Field
import instructor
import openai
from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from atomic_agents.context import ChatHistory, SystemPromptGenerator, BaseDynamicContextProvider


# Schemas
class ResearchQuerySchema(BaseIOSchema):
    """Input for generating research queries."""
    topic: str = Field(..., description="Research topic")
    num_queries: int = Field(default=3, ge=1, le=5)


class GeneratedQueriesSchema(BaseIOSchema):
    """Output with generated search queries."""
    queries: List[str] = Field(..., description="Generated search queries")
    reasoning: str = Field(..., description="Why these queries were chosen")


class SynthesisInputSchema(BaseIOSchema):
    """Input for synthesizing research."""
    original_topic: str = Field(..., description="Original research topic")
    query: str = Field(..., description="Ask a question about the research")


class SynthesisOutputSchema(BaseIOSchema):
    """Synthesized research output."""
    summary: str = Field(..., description="Research summary")
    key_findings: List[str] = Field(..., description="Key findings")
    confidence: float = Field(..., ge=0.0, le=1.0)


# Context Provider for Research Results
class ResearchResultsProvider(BaseDynamicContextProvider):
    """Provides research results as context."""

    def __init__(self):
        super().__init__(title="Research Results")
        self.results: List[dict] = []

    def add_result(self, query: str, result: str):
        self.results.append({"query": query, "result": result})

    def clear(self):
        self.results = []

    def get_info(self) -> str:
        if not self.results:
            return "No research results available yet."
        output = []
        for i, r in enumerate(self.results, 1):
            output.append(f"Query {i}: {r['query']}")
            output.append(f"Result: {r['result']}")
            output.append("")
        return "\n".join(output)


class ResearchAgent:
    """Multi-step research agent."""

    def __init__(self):
        self.client = instructor.from_openai(openai.OpenAI())
        self.results_provider = ResearchResultsProvider()

        # Query generation agent
        self.query_agent = AtomicAgent[ResearchQuerySchema, GeneratedQueriesSchema](
            config=AgentConfig(
                client=self.client,
                model="gpt-5-mini",
                system_prompt_generator=SystemPromptGenerator(
                    background=["You generate effective search queries for research."],
                    steps=["Analyze the topic.", "Generate diverse, specific queries."],
                    output_instructions=["Queries should cover different aspects."]
                )
            )
        )

        # Synthesis agent
        self.synthesis_agent = AtomicAgent[SynthesisInputSchema, SynthesisOutputSchema](
            config=AgentConfig(
                client=self.client,
                model="gpt-5-mini",
                system_prompt_generator=SystemPromptGenerator(
                    background=["You synthesize research findings into clear summaries."],
                    steps=["Review the research results.", "Identify key patterns.", "Synthesize findings."],
                    output_instructions=["Be comprehensive but concise."]
                )
            )
        )
        self.synthesis_agent.register_context_provider("research", self.results_provider)

    def generate_queries(self, topic: str, num_queries: int = 3) -> List[str]:
        """Generate research queries for a topic."""
        response = self.query_agent.run(
            ResearchQuerySchema(topic=topic, num_queries=num_queries)
        )
        print(f"Generated queries: {response.queries}")
        print(f"Reasoning: {response.reasoning}")
        return response.queries

    def add_research_result(self, query: str, result: str):
        """Add a research result (from search, database, etc.)."""
        self.results_provider.add_result(query, result)

    def synthesize(self, topic: str, question: str) -> SynthesisOutputSchema:
        """Synthesize research into a summary."""
        return self.synthesis_agent.run(
            SynthesisInputSchema(original_topic=topic, query=question)
        )


if __name__ == "__main__":
    researcher = ResearchAgent()

    # Step 1: Generate queries
    topic = "Benefits of renewable energy"
    queries = researcher.generate_queries(topic)

    # Step 2: Simulate adding research results
    # (In practice, you'd search and add real results)
    researcher.add_research_result(
        queries[0],
        "Solar energy has seen 89% cost reduction since 2010."
    )
    researcher.add_research_result(
        queries[1],
        "Wind power now provides 10% of global electricity."
    )

    # Step 3: Synthesize
    synthesis = researcher.synthesize(topic, "What are the main benefits?")

    print(f"\n{'='*60}")
    print(f"Summary: {synthesis.summary}")
    print(f"\nKey Findings:")
    for finding in synthesis.key_findings:
        print(f"  - {finding}")
    print(f"\nConfidence: {synthesis.confidence:.0%}")
```

## RAG Agent

Retrieval-augmented generation pattern.

```python
"""
RAG Agent Recipe

Agent that retrieves relevant context before generating responses.

Requirements:
- pip install atomic-agents openai chromadb
- Set OPENAI_API_KEY environment variable
"""

import os
from typing import List
from pydantic import Field
import instructor
import openai
from atomic_agents import AtomicAgent, AgentConfig, BasicChatInputSchema, BaseIOSchema
from atomic_agents.context import ChatHistory, SystemPromptGenerator, BaseDynamicContextProvider


class RAGOutputSchema(BaseIOSchema):
    """RAG agent output with sources."""
    answer: str = Field(..., description="Answer based on retrieved context")
    sources_used: List[int] = Field(
        default_factory=list,
        description="Indices of sources used (1-indexed)"
    )
    confidence: float = Field(..., ge=0.0, le=1.0)


class RetrievedContextProvider(BaseDynamicContextProvider):
    """Provides retrieved documents as context."""

    def __init__(self):
        super().__init__(title="Retrieved Documents")
        self.documents: List[str] = []

    def set_documents(self, docs: List[str]):
        self.documents = docs

    def clear(self):
        self.documents = []

    def get_info(self) -> str:
        if not self.documents:
            return "No relevant documents found."
        output = []
        for i, doc in enumerate(self.documents, 1):
            output.append(f"[Document {i}]: {doc}")
        return "\n\n".join(output)


class SimpleVectorStore:
    """Simple in-memory vector store for demonstration."""

    def __init__(self):
        self.documents: List[str] = []

    def add_documents(self, docs: List[str]):
        self.documents.extend(docs)

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """Simple keyword-based search (replace with real embeddings)."""
        query_words = set(query.lower().split())
        scored = []
        for doc in self.documents:
            doc_words = set(doc.lower().split())
            score = len(query_words & doc_words)
            scored.append((score, doc))
        scored.sort(reverse=True)
        return [doc for _, doc in scored[:top_k]]


class RAGAgent:
    """Retrieval-Augmented Generation agent."""

    def __init__(self):
        self.client = instructor.from_openai(openai.OpenAI())
        self.vector_store = SimpleVectorStore()
        self.context_provider = RetrievedContextProvider()

        self.agent = AtomicAgent[BasicChatInputSchema, RAGOutputSchema](
            config=AgentConfig(
                client=self.client,
                model="gpt-5-mini",
                history=ChatHistory(),
                system_prompt_generator=SystemPromptGenerator(
                    background=[
                        "You are a helpful assistant that answers questions based on provided documents.",
                        "Only use information from the retrieved documents to answer."
                    ],
                    steps=[
                        "Review the retrieved documents carefully.",
                        "Find relevant information to answer the question.",
                        "Cite which documents you used."
                    ],
                    output_instructions=[
                        "Base your answer only on the provided documents.",
                        "If the documents don't contain the answer, say so.",
                        "Always cite your sources by document number."
                    ]
                )
            )
        )
        self.agent.register_context_provider("documents", self.context_provider)

    def add_documents(self, documents: List[str]):
        """Add documents to the knowledge base."""
        self.vector_store.add_documents(documents)

    def query(self, question: str, top_k: int = 3) -> RAGOutputSchema:
        """Query with retrieval-augmented generation."""
        # Retrieve relevant documents
        relevant_docs = self.vector_store.search(question, top_k)
        self.context_provider.set_documents(relevant_docs)

        # Generate response
        response = self.agent.run(BasicChatInputSchema(chat_message=question))
        return response


if __name__ == "__main__":
    rag = RAGAgent()

    # Add knowledge base
    rag.add_documents([
        "Python was created by Guido van Rossum and first released in 1991.",
        "Python emphasizes code readability with significant whitespace.",
        "Python supports multiple programming paradigms including procedural, object-oriented, and functional.",
        "The Python Package Index (PyPI) hosts over 400,000 packages.",
        "Python is widely used in data science, machine learning, and web development."
    ])

    # Query
    response = rag.query("Who created Python and when?")

    print(f"Answer: {response.answer}")
    print(f"Sources used: {response.sources_used}")
    print(f"Confidence: {response.confidence:.0%}")
```

## Summary

These recipes demonstrate common patterns:

| Pattern | Key Components | Use Case |
|---------|---------------|----------|
| Basic Chatbot | AtomicAgent, ChatHistory | Simple Q&A |
| Memory | ChatHistory persistence | Context retention |
| Custom Schema | BaseIOSchema subclass | Structured output |
| Multi-Provider | Provider switching | Flexibility |
| Tools | BaseTool | Extended capabilities |
| Streaming | run_async_stream | Real-time UX |
| Research | Multiple agents | Complex workflows |
| RAG | Context providers | Knowledge-augmented |

Combine these patterns to build sophisticated AI applications.
