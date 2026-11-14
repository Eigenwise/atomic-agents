"""Bounded, opt-in GenAI chat message attributes."""

import json

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
    if _field(block, "type") not in ("text", "input_text", "output_text"):
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


def _content_part(block: object) -> dict[str, object] | None:
    kind = _field(block, "type")
    if kind == "tool_use":
        name, arguments = _field(block, "name"), _field(block, "input")
        if isinstance(name, str) and name and isinstance(arguments, dict):
            return _call_id({"type": "tool_call", "name": name, "arguments": arguments}, block, "id")
        return None
    if kind == "tool_result":
        content = _field(block, "content")
        response = content if isinstance(content, str) else _text_parts(content)
        if isinstance(response, str) or response:
            return _call_id({"type": "tool_call_response", "response": response}, block, "tool_use_id")
        return None
    return _text_block(block)


def _message_parts(message: object, role: str) -> list[dict[str, object]]:
    content = _field(message, "content")
    if role == "tool":
        if not isinstance(content, str):
            return []
        return [_call_id({"type": "tool_call_response", "response": content}, message, "tool_call_id")]
    if isinstance(content, list):
        return [part for block in content if (part := _content_part(block)) is not None] + _tool_calls(message)
    return _text_parts(content) + _tool_calls(message)


def _message(message: object) -> dict[str, object] | None:
    role = _field(message, "role")
    if not isinstance(role, str):
        return None
    parts = _message_parts(message, role)
    return {"role": role, "parts": parts} if parts else None


def _messages(values: object) -> list[dict[str, object]]:
    if not isinstance(values, list):
        return []
    return [message for value in values if (message := _message(value)) is not None]


def _output_messages(response: object) -> list[dict[str, object]]:
    choices = _field(response, "choices")
    if isinstance(choices, list):
        return [message for choice in choices if (message := _message(_field(choice, "message"))) is not None]
    if _field(response, "type") == "message" and _field(response, "role") == "assistant":
        message = _message(response)
        return [message] if message else []
    if _field(response, "object") == "response":
        items = _field(response, "output")
        if isinstance(items, list):
            parts = []
            for item in items:
                kind = _field(item, "type")
                if kind == "message" and _field(item, "role") == "assistant":
                    parts.extend(_text_parts(_field(item, "content")))
                elif kind == "function_call":
                    part = _tool_call({"type": "function", "id": _field(item, "call_id"), "function": item})
                    if part is not None:
                        parts.append(part)
            # Responses output items belong to a single generated choice.
            return [{"role": "assistant", "parts": parts}] if parts else []
    return []


def _messages_hidden(direction: str, config: TracingConfig) -> bool:
    if direction == "output":
        return config.hide_outputs
    # Request messages replay model output: earlier assistant turns, and on an Instructor
    # re-ask the rejected answer plus the validation error quoting it.
    return config.hide_inputs or config.hide_outputs


def write_messages(span: Span, value: object, direction: str, config: TracingConfig, *, system: object = None) -> None:
    if not config.capture_messages or not config.llm_spans or context_api.get_value("suppress_instrumentation"):
        return
    if not _messages_hidden(direction, config):
        _write_visible_messages(span, value, direction, config, system=system)


def _write_visible_messages(
    span: Span, value: object, direction: str, config: TracingConfig, *, system: object = None
) -> None:
    try:
        messages = _messages(value) if direction == "input" else _output_messages(value)
        if direction == "input" and (system_parts := _text_parts(system)):
            messages.insert(0, {"role": "system", "parts": system_parts})
        if not messages:
            return
        serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        if len(serialized) <= config.max_attribute_length:
            span.set_attribute(f"gen_ai.{direction}.messages", serialized)
    except Exception:
        return
