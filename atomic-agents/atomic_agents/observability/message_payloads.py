"""Bounded, opt-in GenAI chat message attributes."""

import json
from collections.abc import Iterator

from opentelemetry import context as context_api
from opentelemetry.trace import Span

from .tracing_config import TracingConfig


def _field(value: object, name: str) -> object:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _text_part(content: object) -> dict[str, object] | None:
    if isinstance(content, str) and content:
        return {"type": "text", "content": content}
    return None


def _text_block(block: object) -> dict[str, object] | None:
    if _field(block, "type") != "text":
        return None
    return _text_part(_field(block, "text"))


def _text_parts(content: object) -> list[dict[str, object]]:
    if isinstance(content, list):
        return [part for block in content if (part := _text_block(block)) is not None]
    part = _text_part(content)
    return [part] if part is not None else []


def _call_id(part: dict[str, object], value: object, field: str) -> dict[str, object]:
    identifier = _field(value, field)
    if isinstance(identifier, str):
        part["id"] = identifier
    return part


def _tool_call(call: object) -> dict[str, object] | None:
    if _field(call, "type") != "function":
        return None
    function = _field(call, "function")
    name = _field(function, "name")
    if not isinstance(name, str) or not name:
        return None
    part: dict[str, object] = {"type": "tool_call", "name": name}
    arguments = _field(function, "arguments")
    if isinstance(arguments, str):
        part["arguments"] = json.loads(arguments)
    return _call_id(part, call, "id")


def _tool_calls(message: object) -> list[dict[str, object]]:
    calls = _field(message, "tool_calls")
    if not isinstance(calls, list):
        return []
    return [part for call in calls if (part := _tool_call(call)) is not None]


def _message_parts(message: object, role: str) -> list[dict[str, object]]:
    content = _field(message, "content")
    if role == "tool":
        if not isinstance(content, str):
            return []
        return [_call_id({"type": "tool_call_response", "response": content}, message, "tool_call_id")]
    return _text_parts(content) + _tool_calls(message)


def _message(message: object) -> dict[str, object] | None:
    role = _field(message, "role")
    if not isinstance(role, str):
        return None
    parts = _message_parts(message, role)
    return {"role": role, "parts": parts} if parts else None


def _messages(values: object) -> Iterator[dict[str, object]]:
    if not isinstance(values, list):
        return
    for value in values:
        if (message := _message(value)) is not None:
            yield message


def _output_messages(response: object) -> Iterator[dict[str, object]]:
    choices = _field(response, "choices")
    if not isinstance(choices, list):
        return
    for choice in choices:
        if (message := _message(_field(choice, "message"))) is not None:
            yield message


def _encode_messages(messages: Iterator[dict[str, object]], limit: int) -> str | None:
    parts = []
    size = 2  # The opening and closing array brackets are part of the limit.
    for message in messages:
        encoded = json.dumps(message, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        size += len(encoded) + bool(parts)
        if size > limit:
            return None
        parts.append(encoded)
    return "[" + ",".join(parts) + "]" if parts else None


def _messages_hidden(direction: str, config: TracingConfig) -> bool:
    if direction == "output":
        return config.hide_outputs
    # Request messages replay model output: earlier assistant turns, and on an Instructor
    # re-ask the rejected answer plus the validation error quoting it.
    return config.hide_inputs or config.hide_outputs


def write_messages(span: Span, value: object, direction: str, config: TracingConfig) -> None:
    if not config.capture_messages or not config.llm_spans or context_api.get_value("suppress_instrumentation"):
        return
    if not _messages_hidden(direction, config):
        _write_visible_messages(span, value, direction, config)


def _write_visible_messages(span: Span, value: object, direction: str, config: TracingConfig) -> None:
    try:
        messages = _messages(value) if direction == "input" else _output_messages(value)
        serialized = _encode_messages(messages, config.max_attribute_length)
        if serialized is not None:
            span.set_attribute(f"gen_ai.{direction}.messages", serialized)
    except Exception:
        return
