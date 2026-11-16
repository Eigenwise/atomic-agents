# Hooks Guide

This guide covers the hook system in Atomic Agents, enabling comprehensive monitoring, error handling, and intelligent retry mechanisms.

## Overview

The Atomic Agents hook system integrates with Instructor's event system to provide:

- **Comprehensive Monitoring**: Track all aspects of agent execution
- **Robust Error Handling**: Graceful handling of validation and completion errors
- **Intelligent Retry Patterns**: Implement smart retry logic based on error context
- **Performance Metrics**: Monitor response times, success rates, and error patterns
- **Zero Overhead**: Hooks only execute when registered and enabled

## Supported Hook Events

| Event | Description | When Triggered |
|-------|-------------|----------------|
| `parse:error` | Pydantic validation failures | When LLM output doesn't match schema |
| `completion:kwargs` | Before API calls | Just before sending request to LLM |
| `completion:response` | After API responses | When LLM returns a response |
| `completion:error` | API or network errors | On connection failures, timeouts, etc. |

## Basic Hook Registration

Register hooks using the `register_hook` method on any `AtomicAgent`:

```python
import os
import instructor
import openai
from atomic_agents import AtomicAgent, AgentConfig, BasicChatInputSchema, BasicChatOutputSchema
from atomic_agents.context import ChatHistory


def on_parse_error(error):
    """Handle validation errors."""
    print(f"Validation failed: {error}")


def on_completion_kwargs(**kwargs):
    """Log API call details before request."""
    model = kwargs.get("model", "unknown")
    print(f"Calling model: {model}")


def on_completion_response(response, **kwargs):
    """Process successful responses."""
    if hasattr(response, "usage"):
        print(f"Tokens used: {response.usage.total_tokens}")


def on_completion_error(error, **kwargs):
    """Handle API errors."""
    print(f"API error: {type(error).__name__}: {error}")


# Create agent
client = instructor.from_openai(openai.OpenAI())
agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
    config=AgentConfig(
        client=client,
        model="gpt-4o-mini",
        history=ChatHistory()
    )
)

# Register hooks
agent.register_hook("parse:error", on_parse_error)
agent.register_hook("completion:kwargs", on_completion_kwargs)
agent.register_hook("completion:response", on_completion_response)
agent.register_hook("completion:error", on_completion_error)

# Use the agent normally - hooks are called automatically
response = agent.run(BasicChatInputSchema(chat_message="Hello!"))
```

## Performance Monitoring

Track request metrics for performance analysis:

```python
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentMetrics:
    """Tracks agent performance metrics."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    parse_errors: int = 0
    total_response_time: float = 0.0
    _request_start: Optional[float] = field(default=None, repr=False)

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests * 100

    @property
    def avg_response_time(self) -> float:
        if self.successful_requests == 0:
            return 0.0
        return self.total_response_time / self.successful_requests


# Create metrics instance
metrics = AgentMetrics()


def on_request_start(**kwargs):
    """Track request start time."""
    metrics.total_requests += 1
    metrics._request_start = time.time()


def on_request_complete(response, **kwargs):
    """Track successful request metrics."""
    if metrics._request_start:
        elapsed = time.time() - metrics._request_start
        metrics.total_response_time += elapsed
        metrics._request_start = None
    metrics.successful_requests += 1


def on_request_error(error, **kwargs):
    """Track failed request metrics."""
    metrics.failed_requests += 1
    metrics._request_start = None


def on_validation_error(error):
    """Track validation errors."""
    metrics.parse_errors += 1


# Register metrics hooks
agent.register_hook("completion:kwargs", on_request_start)
agent.register_hook("completion:response", on_request_complete)
agent.register_hook("completion:error", on_request_error)
agent.register_hook("parse:error", on_validation_error)

# After running queries, check metrics
print(f"Success Rate: {metrics.success_rate:.1f}%")
print(f"Avg Response Time: {metrics.avg_response_time:.2f}s")
```

## Detailed Validation Error Handling

Extract detailed information from validation errors:

```python
from pydantic import ValidationError


def detailed_parse_error_handler(error):
    """Extract detailed validation error information."""
    if isinstance(error, ValidationError):
        print("Validation Error Details:")
        for err in error.errors():
            # Get field path (e.g., "confidence" or "nested.field")
            field_path = " -> ".join(str(x) for x in err["loc"])
            error_type = err["type"]
            message = err["msg"]

            print(f"  Field: {field_path}")
            print(f"  Type: {error_type}")
            print(f"  Message: {message}")

            # Access input value if available
            if "input" in err:
                print(f"  Invalid Value: {err['input']}")
    else:
        print(f"Parse Error: {error}")


agent.register_hook("parse:error", detailed_parse_error_handler)
```

## Retry Strategies with Hooks

Implement intelligent retry logic based on error context:

```python
import time
from functools import wraps


class RetryHandler:
    """Manages retry logic for agent calls."""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.current_attempt = 0
        self.should_retry = False

    def on_error(self, error, **kwargs):
        """Determine if retry is appropriate."""
        self.current_attempt += 1

        # Check if we should retry
        if self.current_attempt < self.max_retries:
            # Retry on rate limits and server errors
            error_str = str(error).lower()
            if any(x in error_str for x in ["rate limit", "timeout", "503", "502"]):
                self.should_retry = True
                delay = self.base_delay * (2 ** (self.current_attempt - 1))
                print(f"Retrying in {delay}s (attempt {self.current_attempt}/{self.max_retries})")
                time.sleep(delay)
            else:
                self.should_retry = False
        else:
            self.should_retry = False
            print(f"Max retries ({self.max_retries}) exceeded")

    def on_success(self, response, **kwargs):
        """Reset retry counter on success."""
        self.current_attempt = 0
        self.should_retry = False

    def reset(self):
        """Reset retry state."""
        self.current_attempt = 0
        self.should_retry = False


def run_with_retry(agent, input_data, retry_handler: RetryHandler):
    """Execute agent with retry logic."""
    retry_handler.reset()

    while True:
        try:
            response = agent.run(input_data)
            return response
        except Exception as e:
            if not retry_handler.should_retry:
                raise

    return None


# Usage
retry_handler = RetryHandler(max_retries=3, base_delay=1.0)
agent.register_hook("completion:error", retry_handler.on_error)
agent.register_hook("completion:response", retry_handler.on_success)
```

## Managing Hooks

### Enable/Disable Hooks

Temporarily disable hooks without unregistering:

```python
# Disable all hooks
agent.disable_hooks()

# Run without hook overhead
response = agent.run(input_data)

# Re-enable hooks
agent.enable_hooks()

# Check if hooks are enabled
if agent.hooks_enabled():
    print("Hooks are active")
```

### Unregister Hooks

Rejected registrations are not retained while hooks are enabled. If Instructor
rejects a deferred registration during `enable_hooks()`, the agent stays disabled
and detaches callbacks added during that attempt. Remove the rejected registration
while disabled, then call `enable_hooks()` again.

Remove specific hooks or clear all:

```python
# Unregister a specific hook
agent.unregister_hook("parse:error", on_parse_error)

# Clear all hooks
agent.clear_hooks()
```

## Production Logging Pattern

A complete production-ready logging setup:

```python
import logging
import json
from datetime import datetime
from typing import Any, Dict


class ProductionAgentLogger:
    """Production-grade agent logging with hooks."""

    def __init__(self, logger_name: str = "atomic_agent"):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.INFO)

        # Add handler if none exists
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            ))
            self.logger.addHandler(handler)

    def log_request(self, **kwargs):
        """Log outgoing request details."""
        self.logger.info(json.dumps({
            "event": "request_start",
            "model": kwargs.get("model"),
            "messages_count": len(kwargs.get("messages", [])),
            "timestamp": datetime.utcnow().isoformat()
        }))

    def log_response(self, response, **kwargs):
        """Log response details."""
        log_data = {
            "event": "request_complete",
            "timestamp": datetime.utcnow().isoformat()
        }

        if hasattr(response, "usage"):
            log_data["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }

        self.logger.info(json.dumps(log_data))

    def log_error(self, error, **kwargs):
        """Log error details."""
        self.logger.error(json.dumps({
            "event": "request_error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.utcnow().isoformat()
        }))

    def log_validation_error(self, error):
        """Log validation error details."""
        self.logger.warning(json.dumps({
            "event": "validation_error",
            "error_type": type(error).__name__,
            "error_message": str(error),
            "timestamp": datetime.utcnow().isoformat()
        }))

    def register_with_agent(self, agent: AtomicAgent):
        """Register all logging hooks with an agent."""
        agent.register_hook("completion:kwargs", self.log_request)
        agent.register_hook("completion:response", self.log_response)
        agent.register_hook("completion:error", self.log_error)
        agent.register_hook("parse:error", self.log_validation_error)


# Usage
logger = ProductionAgentLogger("my_agent")
logger.register_with_agent(agent)
```

## Best Practices

### 1. Keep Hooks Lightweight

Hooks run synchronously - avoid heavy operations:

```python
# Good: Quick logging
def on_response(response, **kwargs):
    logger.info(f"Response received")

# Avoid: Heavy processing in hooks
def on_response_slow(response, **kwargs):
    # Don't do this - blocks the response
    save_to_database(response)
    send_to_analytics(response)
    generate_report(response)
```

### 2. Handle Hook Exceptions

Wrap hook logic to prevent failures from disrupting the agent:

```python
def safe_hook(func):
    """Decorator to catch hook exceptions."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Hook error in {func.__name__}: {e}")
    return wrapper


@safe_hook
def on_completion_response(response, **kwargs):
    # If this fails, the agent continues working
    process_response(response)
```

### 3. Use Hooks for Cross-Cutting Concerns

Hooks are ideal for:
- Logging and monitoring
- Metrics collection
- Error tracking
- Performance profiling
- Audit trails

### 4. Don't Modify Responses in Hooks

Hooks are for observation, not transformation:

```python
# Good: Observe and log
def on_response(response, **kwargs):
    logger.info(f"Got response: {response}")

# Avoid: Trying to modify response
def on_response_bad(response, **kwargs):
    response.chat_message = "Modified"  # Don't do this
```

## Summary

| Feature | Method | Description |
|---------|--------|-------------|
| Register hook | `agent.register_hook(event, callback)` | Add a hook callback |
| Unregister hook | `agent.unregister_hook(event, callback)` | Remove specific hook |
| Clear all hooks | `agent.clear_hooks()` | Remove all hooks |
| Enable hooks | `agent.enable_hooks()` | Activate hook system |
| Disable hooks | `agent.disable_hooks()` | Deactivate hook system |
| Check status | `agent.hooks_enabled()` | Check if hooks active |

Use hooks to add monitoring and error handling to your agents without modifying core business logic.
