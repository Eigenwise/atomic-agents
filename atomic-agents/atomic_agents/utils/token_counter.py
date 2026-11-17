"""Token counting utilities for provider-agnostic context measurement."""

import logging
from typing import Any, Dict, List, NamedTuple, Optional

logger = logging.getLogger(__name__)


class TokenCountError(Exception):
    """Exception raised when token counting fails."""

    pass


class TokenCountResult(NamedTuple):
    """
    Result of a token count operation.

    Attributes:
        total: Total number of tokens in the complete message request. When tools
            are provided without messages, this is schema-only overhead and
            excludes request framing.
        system_prompt: Tokens in the system messages, including request framing
            when a system message is present (0 otherwise).
        history: Incremental tokens added by the history messages. Includes
            request framing when there are no system messages.
        tools: Incremental tokens added by tools to the complete message request.
            When there are no messages, this is the schema-only token overhead.
        model: The model used for tokenization.
        max_tokens: Maximum context window for the model (None if unknown).
        utilization: Percentage of context window used (None if max_tokens unknown).
    """

    total: int
    system_prompt: int
    history: int
    tools: int
    model: str
    max_tokens: Optional[int] = None
    utilization: Optional[float] = None


# Module-level singleton for efficiency
_token_counter_instance: Optional["TokenCounter"] = None


def get_token_counter() -> "TokenCounter":
    """Get the singleton TokenCounter instance."""
    global _token_counter_instance
    if _token_counter_instance is None:
        _token_counter_instance = TokenCounter()
    return _token_counter_instance


class TokenCounter:
    """
    Utility class for counting tokens using LiteLLM's provider-agnostic tokenizer.

    This class provides methods for counting tokens in messages, text, tools,
    and retrieving model context limits. It uses LiteLLM's token_counter which
    automatically selects the appropriate tokenizer based on the model.

    Works with any model supported by LiteLLM including:
    - OpenAI (gpt-4, gpt-3.5-turbo, etc.)
    - Anthropic (claude-3-opus, claude-3-sonnet, etc.)
    - Google (gemini-pro, gemini-1.5-pro, etc.)
    - And 100+ other providers

    Example:
        ```python
        counter = TokenCounter()

        # Count tokens in messages
        messages = [{"role": "user", "content": "Hello, world!"}]
        count = counter.count_messages("gpt-4", messages)

        # Count tokens with tools (for TOOLS mode)
        tools = [{"type": "function", "function": {...}}]
        count = counter.count_messages("gpt-4", messages, tools=tools)

        # Get max tokens for a model
        max_tokens = counter.get_max_tokens("gpt-4")
        ```
    """

    def count_messages(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        Count the number of tokens in a list of messages and optional tools.

        Args:
            model: The model identifier (e.g., "gpt-4", "anthropic/claude-3-opus").
            messages: List of message dictionaries with 'role' and 'content' keys.
            tools: Optional list of tool definitions (for TOOLS mode).

        Returns:
            The number of tokens in the messages (and tools if provided).

        Raises:
            TokenCountError: If token counting fails.
        """
        if not model:
            raise ValueError("model is required for token counting")

        try:
            from litellm import token_counter

            if tools:
                return token_counter(model=model, messages=messages, tools=tools)
            return token_counter(model=model, messages=messages)
        except ImportError as e:
            raise ImportError("litellm is required for token counting. " "Install it with: pip install litellm") from e
        except Exception as e:
            raise TokenCountError(f"Failed to count tokens for model '{model}': {e}") from e

    def count_text(self, model: str, text: str) -> int:
        """
        Count the tokens in a text string without chat-message framing.

        Args:
            model: The model identifier.
            text: The text to tokenize.

        Returns:
            The number of tokens in the text.

        Raises:
            TokenCountError: If token counting fails.
        """
        if not model:
            raise ValueError("model is required for token counting")

        try:
            from litellm import token_counter

            return token_counter(model=model, text=text)
        except ImportError as e:
            raise ImportError("litellm is required for token counting. " "Install it with: pip install litellm") from e
        except Exception as e:
            raise TokenCountError(f"Failed to count tokens for model '{model}': {e}") from e

    def get_max_tokens(self, model: str) -> Optional[int]:
        """
        Get the maximum context window size for a model.

        Args:
            model: The model identifier.

        Returns:
            The maximum number of tokens, or None if unknown.

        Raises:
            TypeError: If model is None or not a string.
            ImportError: If litellm is not installed.
        """
        if not isinstance(model, str):
            raise TypeError(f"model must be a string, got {type(model).__name__}")

        try:
            from litellm import get_model_info
        except ImportError as e:
            raise ImportError("litellm is required for token counting. " "Install it with: pip install litellm") from e

        try:
            info = get_model_info(model)
            # Use max_input_tokens (context window) not max_tokens (output limit)
            max_input = info.get("max_input_tokens")
            return max_input if max_input is not None else info.get("max_tokens")
        except Exception as e:
            logger.warning(f"Could not determine max tokens for model '{model}': {e}")
            return None

    def count_context(
        self,
        model: str,
        system_messages: List[Dict[str, Any]],
        history_messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> TokenCountResult:
        """
        Count tokens with breakdown by system prompt, history, and tools.

        The breakdown is additive: ``system_prompt + history + tools == total``.
        Request framing is attributed to the system prompt when present, or to
        history otherwise. Tool tokens are the increment over the full message
        request without tools. When tools are provided without any messages,
        ``total`` and ``tools`` report schema-only overhead and exclude request
        framing.

        Args:
            model: The model identifier.
            system_messages: System prompt messages (may be empty).
            history_messages: Conversation history messages.
            tools: Optional list of tool definitions (for TOOLS mode).

        Returns:
            TokenCountResult with breakdown and utilization metrics.

        Raises:
            TokenCountError: If token counting fails.
        """
        system_tokens = self.count_messages(model, system_messages) if system_messages else 0
        all_messages = system_messages + history_messages

        # Tokenizers include request-level framing, so independently counting the
        # system prompt and history would count that overhead twice. Count the
        # complete message list and attribute the incremental tokens to history.
        if history_messages:
            messages_tokens = self.count_messages(model, all_messages)
            history_tokens = messages_tokens - system_tokens
        else:
            messages_tokens = system_tokens
            history_tokens = 0

        tools_tokens = 0
        total_tokens = messages_tokens
        if tools:
            if all_messages:
                total_tokens = self.count_messages(model, all_messages, tools=tools)
                tools_tokens = total_tokens - messages_tokens
            else:
                # Preserve schema-only counting by subtracting the same placeholder
                # request without tools. Keep the placeholder out of the total.
                placeholder = [{"role": "user", "content": ""}]
                with_tools_tokens = self.count_messages(model, placeholder, tools=tools)
                without_tools_tokens = self.count_messages(model, placeholder)
                tools_tokens = with_tools_tokens - without_tools_tokens
                total_tokens = tools_tokens

        max_tokens = self.get_max_tokens(model)
        # Prevent division by zero
        utilization = (total_tokens / max_tokens) if max_tokens and max_tokens > 0 else None

        return TokenCountResult(
            total=total_tokens,
            system_prompt=system_tokens,
            history=history_tokens,
            tools=tools_tokens,
            model=model,
            max_tokens=max_tokens,
            utilization=utilization,
        )
