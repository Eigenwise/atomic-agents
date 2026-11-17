# Utilities

## Token Counting

The `TokenCounter` utility provides provider-agnostic token counting for any model supported by LiteLLM. This allows you to monitor context usage regardless of whether you're using OpenAI, Anthropic, Google, or any other supported provider.

### TokenCountResult

A named tuple containing token count information:

```{eval-rst}
.. py:class:: TokenCountResult

    Named tuple containing token count information.

    .. py:attribute:: total
        :type: int

        LiteLLM token count for the complete serialized context. With tools but no
        messages, this is schema-only overhead and excludes request framing.

    .. py:attribute:: system_prompt
        :type: int

        System message tokens, including request framing when a system message is
        present. In JSON modes, this also includes the output schema.

    .. py:attribute:: history
        :type: int

        Incremental tokens added by conversation history, including request
        framing when no system message is present.

    .. py:attribute:: tools
        :type: int

        Incremental tokens added by tool definitions in TOOLS mode.

    .. py:attribute:: model
        :type: str

        The model used for token counting.

    .. py:attribute:: max_tokens
        :type: Optional[int]

        Maximum context window for the model (if known).

    .. py:attribute:: utilization
        :type: Optional[float]

        Context utilization percentage (0.0 to 1.0) if max_tokens is known.
```

### TokenCounter

The main utility class for counting tokens:

```{eval-rst}
.. py:class:: TokenCounter

    Utility class for counting tokens in messages using LiteLLM.

    .. py:method:: count_messages(model: str, messages: List[Dict[str, Any]]) -> int

        Count tokens in a list of messages.

        :param model: The model name (e.g., "gpt-4", "claude-3-opus-20240229")
        :param messages: List of message dictionaries with "role" and "content" keys
        :return: Number of tokens

    .. py:method:: count_text(model: str, text: str) -> int

        Count tokens in a text string without role or chat-request framing.
        Use ``count_messages`` when measuring a complete chat request.

        :param model: The model name
        :param text: The text to count tokens for
        :return: Number of tokens

    .. py:method:: get_max_tokens(model: str) -> Optional[int]

        Get the maximum context window for a model.

        :param model: The model name
        :return: Maximum tokens, or None if unknown

    .. py:method:: count_context(model: str, system_messages: List[Dict], history_messages: List[Dict], tools: Optional[List[Dict]] = None) -> TokenCountResult

        Count tokens for a complete context with an additive breakdown.

        :param model: The model name
        :param system_messages: System prompt messages
        :param history_messages: Conversation history messages
        :param tools: Optional tool definitions
        :return: TokenCountResult with detailed breakdown
```

### Usage Example

```python
from atomic_agents.utils import TokenCounter, TokenCountResult

# Direct usage
counter = TokenCounter()
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there! How can I help?"},
]

# Count tokens in messages
token_count = counter.count_messages("gpt-4", messages)

# Get max context window
max_tokens = counter.get_max_tokens("gpt-4")

# Count complete context with breakdown
result = counter.count_context(
    model="gpt-4",
    system_messages=[{"role": "system", "content": "You are helpful."}],
    history_messages=[{"role": "user", "content": "Hello!"}],
)
print(f"Total: {result.total}, System: {result.system_prompt}, History: {result.history}")
if result.utilization:
    print(f"Context utilization: {result.utilization:.1%}")
```

### Using with AtomicAgent

The easiest way to get token counts is through the agent's `get_context_token_count()` method. The agent computes accurate token counts on-demand by serializing the context exactly as Instructor does, including output schema overhead and multimodal content:

```python
# Get accurate token count at any time - always returns a result
token_info = agent.get_context_token_count()
print(f"Total tokens: {token_info.total}")
print(f"System prompt (with schema): {token_info.system_prompt} tokens")
print(f"History: {token_info.history} tokens")
print(f"Tools: {token_info.tools} tokens")
if token_info.utilization:
    print(f"Context utilization: {token_info.utilization:.1%}")
```

The breakdown is additive: `system_prompt + history + tools == total`. Request
framing is attributed to `system_prompt` when a system message is present, or to
`history` otherwise. When tools are provided without system or history messages,
`total` and `tools` report schema-only overhead and exclude request framing. In
JSON modes, the output schema is included in `system_prompt`; in TOOLS mode, tool
definitions are reported through `tools`.

For requests with messages, the total matches LiteLLM's count for the complete
serialized request.

## Tool Message Formatting

Tool arguments use Pydantic's JSON representations for values such as dates,
UUIDs, decimals, paths, and enums. JSON-only field serializers are honored when
formatting the arguments.

```{eval-rst}
.. automodule:: atomic_agents.utils.format_tool_message
   :members:
   :undoc-members:
   :show-inheritance:
```
