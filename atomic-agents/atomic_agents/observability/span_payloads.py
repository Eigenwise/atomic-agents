"""Payload serialization, redaction, and bounded failures."""

import json
import logging
import traceback
from collections.abc import Iterator

from instructor.core.exceptions import InstructorError

from opentelemetry.trace import Span, Status, StatusCode
from pydantic import BaseModel, ValidationError

from . import installation
from .tracing_config import TracingConfig
from .span_attributes import (
    ATTR_INPUT_VALUE,
    ATTR_INPUT_MIME_TYPE,
    ATTR_OUTPUT_VALUE,
    ATTR_OUTPUT_MIME_TYPE,
    ATTR_INPUT,
    ATTR_OUTPUT,
    ATTR_ERROR_TYPE,
)


logger = logging.getLogger("atomic_agents.observability.tracing")


TRUNCATION_MARKER = "...[truncated]"


def _serialize_payload(value: object) -> str | None:
    try:
        return value.model_dump_json() if isinstance(value, BaseModel) else json.dumps(value, default=str)
    except Exception as error:
        logger.warning("Could not serialize tracing payload: %s", error)
        return None


def _record_payload(
    span: Span, key_pairs: tuple[tuple[str, str | None], ...], value: object, hide: bool, config: TracingConfig
) -> None:
    if value is None:
        return
    serialized = "__REDACTED__" if hide else _serialize_payload(value)
    if serialized is None:
        return
    for key, mime_key in key_pairs:
        _write_payload(span, key, mime_key, serialized, hide, config)


def _write_payload(span: Span, key: str, mime_key: str | None, serialized: str, hide: bool, config: TracingConfig) -> None:
    if hide:
        span.set_attribute(key, "__REDACTED__")
        return
    if len(serialized) > config.max_attribute_length:
        span.set_attribute(key + ".original_length", len(serialized))
        serialized = _bounded_text(serialized, config)
    span.set_attribute(key, serialized)
    if mime_key is not None:
        span.set_attribute(mime_key, "application/json")


def _payload_keys(primary: str, output: bool) -> tuple[tuple[str, str | None], ...]:
    keys: tuple[tuple[str, str | None], ...] = ((primary, None),)
    if installation._current_installation().config.openinference_attributes:
        keys += ((ATTR_OUTPUT_VALUE, ATTR_OUTPUT_MIME_TYPE),) if output else ((ATTR_INPUT_VALUE, ATTR_INPUT_MIME_TYPE),)
    return keys


def _set_input_safely(span: Span, value: object, primary: str = ATTR_INPUT) -> None:
    config = installation._current_installation().config
    _record_payload(span, _payload_keys(primary, False), value, config.hide_inputs, config)


def _set_output_safely(span: Span, value: object, primary: str = ATTR_OUTPUT) -> None:
    config = installation._current_installation().config
    _record_payload(span, _payload_keys(primary, True), value, config.hide_outputs, config)


def _bounded_text(value: str, config: TracingConfig) -> str:
    limit = config.max_attribute_length
    if len(value) > limit:
        marker = TRUNCATION_MARKER[:limit]
        return value[: limit - len(marker)] + marker
    return value


def _exception_chain(error: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    while id(error) not in seen:
        seen.add(id(error))
        yield error
        following = error.__cause__ or error.__context__
        if following is None:
            return
        error = following


def _instructor_attempt_count(error: InstructorError) -> int:
    if error.failed_attempts:
        return len(error.failed_attempts)
    return getattr(error, "n_attempts", 0)


def _failure_count(error: BaseException) -> str | None:
    if isinstance(error, InstructorError):
        return f"{_instructor_attempt_count(error)} attempts"
    if isinstance(error, ValidationError):
        return f"{error.error_count()} errors"
    return None


def _failure_message(error: BaseException, config: TracingConfig) -> str:
    if not config.hide_outputs:
        return str(error)
    count = next((count for cause in _exception_chain(error) if (count := _failure_count(cause)) is not None), "1 errors")
    return f"{type(error).__qualname__}: {count} (__REDACTED__)"


def _failure_stacktrace(error: BaseException, config: TracingConfig) -> str:
    lines = traceback.format_exception(error, chain=not config.hide_outputs)
    if config.hide_outputs:
        lines = _redacted_traceback(error, config)
    return _bounded_text("".join(lines), config)


def _redacted_traceback(error: BaseException, config: TracingConfig) -> list[str]:
    frames = [
        traceback.FrameSummary(frame.filename, frame.lineno, frame.name, line="")
        for frame in traceback.extract_tb(error.__traceback__)
    ]
    return ["Traceback (most recent call last):\n", *traceback.format_list(frames), _failure_message(error, config) + "\n"]


def _record_failure(span: Span, error: BaseException, config: TracingConfig | None = None) -> None:
    config = config or installation._current_installation().config
    message = _bounded_text(_failure_message(error, config), config)
    span.set_attribute(ATTR_ERROR_TYPE, type(error).__qualname__)
    span.record_exception(
        error, attributes={"exception.message": message, "exception.stacktrace": _failure_stacktrace(error, config)}
    )
    span.set_status(Status(StatusCode.ERROR, message))
