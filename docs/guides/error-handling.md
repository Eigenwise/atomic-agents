# Error Handling Guide

This guide covers best practices for handling errors in Atomic Agents applications, including validation errors, API failures, and custom error handling patterns.

## Overview

Atomic Agents provides multiple layers of error handling:

1. **Schema Validation** - Pydantic validates input/output at runtime
2. **API Error Handling** - Handle LLM provider errors gracefully
3. **Hook System** - Monitor and respond to errors via hooks
4. **Custom Exception Handling** - Build robust error recovery patterns

## Schema Validation Errors

Pydantic schemas catch invalid data before it reaches the LLM.

### Basic Validation

```python
import os
from typing import List
from pydantic import Field, field_validator
import instructor
import openai
from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from atomic_agents.context import ChatHistory


class ValidatedInputSchema(BaseIOSchema):
    """Input schema with validation rules."""

    query: str = Field(..., description="User query", min_length=1, max_length=1000)
    max_results: int = Field(default=10, ge=1, le=100, description="Maximum results to return")

    @field_validator('query')
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty or whitespace only")
        return v.strip()


class ValidatedOutputSchema(BaseIOSchema):
    """Output schema with validation."""

    answer: str = Field(..., description="The response")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score 0-1")
    sources: List[str] = Field(default_factory=list, description="Source references")


# Initialize client and agent
client = instructor.from_openai(openai.OpenAI())

agent = AtomicAgent[ValidatedInputSchema, ValidatedOutputSchema](
    config=AgentConfig(
        client=client,
        model="gpt-5.6-terra",
        history=ChatHistory()
    )
)

# Handle validation errors
try:
    response = agent.run(ValidatedInputSchema(query="", max_results=5))
except ValueError as e:
    print(f"Validation error: {e}")
```

### Custom Validators

```python
from pydantic import Field, field_validator, model_validator
from typing import Optional
from atomic_agents import BaseIOSchema


class SearchInputSchema(BaseIOSchema):
    """Search input with complex validation."""

    query: str = Field(..., description="Search query")
    category: Optional[str] = Field(None, description="Category filter")
    date_from: Optional[str] = Field(None, description="Start date YYYY-MM-DD")
    date_to: Optional[str] = Field(None, description="End date YYYY-MM-DD")

    @field_validator('category')
    @classmethod
    def validate_category(cls, v: Optional[str]) -> Optional[str]:
        valid_categories = ['technology', 'science', 'business', 'health']
        if v is not None and v.lower() not in valid_categories:
            raise ValueError(f"Category must be one of: {valid_categories}")
        return v.lower() if v else None

    @model_validator(mode='after')
    def validate_dates(self):
        if self.date_from and self.date_to:
            if self.date_from > self.date_to:
                raise ValueError("date_from must be before date_to")
        return self
```

## API Error Handling

Handle LLM provider errors gracefully with retry logic.

### Basic Retry Pattern

```python
import os
import time
from typing import Optional
import instructor
import openai
from openai import APIError, RateLimitError, APIConnectionError
from atomic_agents import AtomicAgent, AgentConfig, BasicChatInputSchema, BasicChatOutputSchema
from atomic_agents.context import ChatHistory


def create_agent_with_retry(
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> AtomicAgent:
    """Create an agent with retry configuration."""
    client = instructor.from_openai(openai.OpenAI())

    return AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
        config=AgentConfig(
            client=client,
            model="gpt-5.6-terra",
            history=ChatHistory(),
            model_api_parameters={
                "max_tokens": 1000,
                "temperature": 0.7
            }
        )
    )


def run_with_retry(
    agent: AtomicAgent,
    input_data: BasicChatInputSchema,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> Optional[BasicChatOutputSchema]:
    """Run agent with automatic retry on transient failures."""

    last_error = None

    for attempt in range(max_retries):
        try:
            return agent.run(input_data)

        except RateLimitError as e:
            last_error = e
            wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
            print(f"Rate limited. Waiting {wait_time}s before retry {attempt + 1}/{max_retries}")
            time.sleep(wait_time)

        except APIConnectionError as e:
            last_error = e
            print(f"Connection error. Retry {attempt + 1}/{max_retries}")
            time.sleep(retry_delay)

        except APIError as e:
            last_error = e
            if e.status_code and e.status_code >= 500:
                print(f"Server error. Retry {attempt + 1}/{max_retries}")
                time.sleep(retry_delay)
            else:
                raise  # Don't retry client errors (4xx)

    print(f"All retries failed. Last error: {last_error}")
    return None


# Usage
agent = create_agent_with_retry()
user_input = BasicChatInputSchema(chat_message="Explain quantum computing")
response = run_with_retry(agent, user_input)

if response:
    print(f"Response: {response.chat_message}")
else:
    print("Failed to get response after retries")
```

## Using the Hook System for Error Handling

The Atomic Agents hook system provides powerful error monitoring capabilities.

### Error Logging Hook

```python
import os
import logging
from datetime import datetime
from typing import Any, Optional
import instructor
import openai
from atomic_agents import AtomicAgent, AgentConfig, BasicChatInputSchema, BasicChatOutputSchema
from atomic_agents.context import ChatHistory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def on_error_hook(error: Exception, context: dict) -> None:
    """Hook called when an error occurs during agent execution."""
    logger.error(f"Agent error: {type(error).__name__}: {error}")
    logger.error(f"Context: {context}")


def on_completion_hook(response: Any, duration_ms: float) -> None:
    """Hook called on successful completion."""
    logger.info(f"Agent completed in {duration_ms:.2f}ms")


# Create agent with hooks using Instructor's hook system
client = instructor.from_openai(openai.OpenAI())

# Register hooks with the instructor client
client.on("completion", lambda *args: on_completion_hook(*args))

agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
    config=AgentConfig(
        client=client,
        model="gpt-5.6-terra",
        history=ChatHistory()
    )
)
```

### Comprehensive Error Handler

```python
import os
from typing import Callable, Optional, TypeVar
from functools import wraps
import instructor
import openai
from pydantic import ValidationError
from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema

T = TypeVar('T', bound=BaseIOSchema)


class AgentErrorHandler:
    """Centralized error handler for Atomic Agents."""

    def __init__(
        self,
        on_validation_error: Optional[Callable[[ValidationError], None]] = None,
        on_api_error: Optional[Callable[[Exception], None]] = None,
        on_unknown_error: Optional[Callable[[Exception], None]] = None
    ):
        self.on_validation_error = on_validation_error or self._default_validation_handler
        self.on_api_error = on_api_error or self._default_api_handler
        self.on_unknown_error = on_unknown_error or self._default_unknown_handler

    def _default_validation_handler(self, error: ValidationError) -> None:
        print(f"Validation failed: {error.error_count()} errors")
        for err in error.errors():
            print(f"  - {err['loc']}: {err['msg']}")

    def _default_api_handler(self, error: Exception) -> None:
        print(f"API error: {type(error).__name__}: {error}")

    def _default_unknown_handler(self, error: Exception) -> None:
        print(f"Unknown error: {type(error).__name__}: {error}")

    def wrap(self, func: Callable) -> Callable:
        """Decorator to wrap agent calls with error handling."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ValidationError as e:
                self.on_validation_error(e)
                return None
            except (openai.APIError, openai.APIConnectionError) as e:
                self.on_api_error(e)
                return None
            except Exception as e:
                self.on_unknown_error(e)
                return None
        return wrapper


# Usage
error_handler = AgentErrorHandler()

@error_handler.wrap
def ask_agent(agent: AtomicAgent, question: str):
    from atomic_agents import BasicChatInputSchema
    return agent.run(BasicChatInputSchema(chat_message=question))


# Create and use agent
client = instructor.from_openai(openai.OpenAI())
agent = AtomicAgent(
    config=AgentConfig(
        client=client,
        model="gpt-5.6-terra"
    )
)

response = ask_agent(agent, "What is machine learning?")
```

## Graceful Degradation

Implement fallback behavior when the primary agent fails.

### Fallback Agent Pattern

```python
import os
from typing import Optional, List
import instructor
import openai
from atomic_agents import AtomicAgent, AgentConfig, BasicChatInputSchema, BasicChatOutputSchema
from atomic_agents.context import ChatHistory


class FallbackAgentChain:
    """Chain of agents with automatic fallback on failure."""

    def __init__(self, agents: List[AtomicAgent]):
        self.agents = agents

    def run(self, input_data: BasicChatInputSchema) -> Optional[BasicChatOutputSchema]:
        """Try each agent in order until one succeeds."""
        last_error = None

        for i, agent in enumerate(self.agents):
            try:
                print(f"Trying agent {i + 1}/{len(self.agents)}")
                return agent.run(input_data)
            except Exception as e:
                last_error = e
                print(f"Agent {i + 1} failed: {e}")
                continue

        print(f"All agents failed. Last error: {last_error}")
        return None


# Create primary and fallback agents with different models/providers
def create_fallback_chain() -> FallbackAgentChain:
    # Primary: GPT-4
    primary_client = instructor.from_openai(openai.OpenAI())
    primary_agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
        config=AgentConfig(
            client=primary_client,
            model="gpt-4o",
            history=ChatHistory()
        )
    )

    # Fallback: GPT-4o-mini (cheaper, faster)
    fallback_client = instructor.from_openai(openai.OpenAI())
    fallback_agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
        config=AgentConfig(
            client=fallback_client,
            model="gpt-5.6-terra",
            history=ChatHistory()
        )
    )

    return FallbackAgentChain([primary_agent, fallback_agent])


# Usage
chain = create_fallback_chain()
response = chain.run(BasicChatInputSchema(chat_message="Explain quantum computing"))
if response:
    print(response.chat_message)
```

## Best Practices

### 1. Always Validate Input

```python
from pydantic import Field, field_validator
from atomic_agents import BaseIOSchema


class SafeInputSchema(BaseIOSchema):
    """Input schema with comprehensive validation."""

    message: str = Field(..., min_length=1, max_length=10000)

    @field_validator('message')
    @classmethod
    def sanitize_message(cls, v: str) -> str:
        # Remove potential prompt injection attempts
        dangerous_patterns = ['ignore previous', 'disregard instructions']
        for pattern in dangerous_patterns:
            if pattern.lower() in v.lower():
                raise ValueError("Invalid input detected")
        return v.strip()
```

### 2. Log All Errors

```python
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def log_errors(func):
    """Decorator to log all errors from agent operations."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.exception(f"Error in {func.__name__}: {e}")
            raise
    return wrapper
```

### 3. Set Timeouts

```python
import os
import instructor
import openai
from atomic_agents import AtomicAgent, AgentConfig
from atomic_agents.context import ChatHistory


# Configure timeout at client level
client = instructor.from_openai(
    openai.OpenAI(timeout=30.0)  # 30 second timeout
)

agent = AtomicAgent(
    config=AgentConfig(
        client=client,
        model="gpt-5.6-terra",
        history=ChatHistory(),
        model_api_parameters={
            "max_tokens": 500  # Limit response length
        }
    )
)
```

### 4. Implement Circuit Breaker

```python
import time
from typing import Optional, Callable
from dataclasses import dataclass


@dataclass
class CircuitBreaker:
    """Simple circuit breaker for agent calls."""

    failure_threshold: int = 5
    reset_timeout: float = 60.0

    _failure_count: int = 0
    _last_failure_time: float = 0
    _state: str = "closed"  # closed, open, half-open

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self._state == "open":
            if time.time() - self._last_failure_time > self.reset_timeout:
                self._state = "half-open"
            else:
                raise Exception("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _on_success(self):
        self._failure_count = 0
        self._state = "closed"

    def _on_failure(self):
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self.failure_threshold:
            self._state = "open"


# Usage
circuit_breaker = CircuitBreaker(failure_threshold=3, reset_timeout=30.0)

def safe_agent_call(agent, input_data):
    return circuit_breaker.call(agent.run, input_data)
```

## Summary

Key error handling strategies in Atomic Agents:

| Strategy | Use Case | Implementation |
|----------|----------|----------------|
| Schema Validation | Prevent invalid inputs | Pydantic validators |
| Retry Logic | Transient failures | Exponential backoff |
| Hook System | Monitoring & logging | Instructor hooks |
| Fallback Chain | High availability | Multiple agents |
| Circuit Breaker | Prevent cascade failures | State machine |

Always combine multiple strategies for robust production applications.
