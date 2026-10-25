"""Tool spans and tool class instrumentation."""

import asyncio
import contextvars
import functools
import inspect
import json
import logging
from contextlib import contextmanager
from types import FunctionType
from typing import Awaitable, Callable, Coroutine, Iterator, List, Optional, Type, TypeVar

from opentelemetry.trace import Span, Status, StatusCode
from opentelemetry.util.types import AttributeValue

from atomic_agents.base.base_tool import BaseTool

from . import installation
from .span_attributes import _best_effort_attributes, _optional_attributes, _suppressed, _without_none
from .span_payloads import _bounded_text, _record_failure, _set_input_safely, _set_output_safely
from .span_attributes import (
    ATTR_OPERATION,
    ATTR_TOOL_NAME,
    ATTR_TOOL_DESCRIPTION,
    ATTR_TOOL_TYPE,
    ATTR_TOOL_ARGUMENTS,
    ATTR_TOOL_RESULT,
    ATTR_OI_TOOL_NAME,
    ATTR_OI_TOOL_DESCRIPTION,
    ATTR_TOOL_PARAMETERS,
    ATTR_TOOL_CLASS,
)


logger = logging.getLogger("atomic_agents.observability.tracing")


OutputSchema = TypeVar("OutputSchema")

TOOL_METHOD_NAMES = ("run", "run_async", "arun")


_active_tool: contextvars.ContextVar[Optional[BaseTool]] = contextvars.ContextVar("atomic_agents_active_tool", default=None)


def _all_subclasses(base: Type) -> List[Type]:
    found: List[Type] = []
    pending = list(base.__subclasses__())
    while pending:
        current = pending.pop(0)
        if current not in found:
            found.append(current)
            pending.extend(current.__subclasses__())
    return found


def _instrument_existing_tool_classes() -> None:
    for tool_class in _all_subclasses(BaseTool):
        _instrument_tool_class(tool_class)


def _hook_future_tool_classes(original_init_subclass: classmethod) -> None:
    def traced_init_subclass(cls, **kwargs):
        original_init_subclass.__func__(cls, **kwargs)
        _instrument_tool_class(cls)

    setattr(BaseTool, "__init_subclass__", classmethod(traced_init_subclass))


def _instrument_tool_class(tool_class: Type) -> None:
    for method_name in TOOL_METHOD_NAMES:
        method = inspect.getattr_static(tool_class, method_name, None)
        if _is_traceable_function(method):
            inherited = method_name not in tool_class.__dict__
            setattr(tool_class, method_name, installation._mark_traced(_trace_tool_method(method), method, inherited))


def _is_traceable_function(method: object) -> bool:
    if not isinstance(method, FunctionType) or inspect.isasyncgenfunction(method):
        return False
    if getattr(method, "__isabstractmethod__", False):
        return False
    return not getattr(method, installation.TRACED_MARKER, False)


def _restore_tool_class(tool_class: Type) -> None:
    for method_name in TOOL_METHOD_NAMES:
        installation._restore_method(tool_class, method_name)


def _tool_parameter_attributes(tool: BaseTool) -> dict[str, AttributeValue]:
    parameters = _best_effort_attributes({ATTR_TOOL_PARAMETERS: lambda: json.dumps(tool.input_schema.model_json_schema())})
    value = parameters.get(ATTR_TOOL_PARAMETERS)
    if isinstance(value, str):
        config = installation._current_installation().config
        parameters[ATTR_TOOL_PARAMETERS] = _bounded_text(value, config)
        if len(value) > config.max_attribute_length:
            parameters[ATTR_TOOL_PARAMETERS + ".original_length"] = len(value)
    return parameters


def _tool_display_name(tool: BaseTool) -> str:
    try:
        return tool.tool_name
    except Exception as error:
        logger.debug("Could not collect tracing tool name: %s", error)
        return type(tool).__name__


def _tool_start_attributes(tool: BaseTool) -> dict[str, AttributeValue]:
    attributes: dict[str, AttributeValue] = {
        ATTR_OPERATION: "execute_tool",
        ATTR_TOOL_NAME: _tool_display_name(tool),
        ATTR_TOOL_CLASS: type(tool).__name__,
        ATTR_TOOL_TYPE: "function",
        **_optional_attributes("TOOL", {}),
    }
    attributes.update(_best_effort_attributes({ATTR_TOOL_DESCRIPTION: lambda: tool.tool_description}))
    if installation._current_installation().config.openinference_attributes:
        attributes.update(
            _best_effort_attributes(
                {
                    ATTR_OI_TOOL_NAME: lambda: attributes[ATTR_TOOL_NAME],
                    ATTR_OI_TOOL_DESCRIPTION: lambda: tool.tool_description,
                }
            )
        )
        attributes.update(_tool_parameter_attributes(tool))
    return _without_none(attributes)


@contextmanager
def _tool_span(tool: BaseTool, args: tuple[object, ...], kwargs: dict[str, object]) -> Iterator[Span]:
    attributes = _tool_start_attributes(tool)
    with installation._current_installation().tracer.start_as_current_span(
        f"execute_tool {attributes[ATTR_TOOL_NAME]}",
        attributes=attributes,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        _set_input_safely(span, _tool_input_from_call(args, kwargs), ATTR_TOOL_ARGUMENTS)
        try:
            yield span
        except (asyncio.CancelledError, GeneratorExit):
            span.set_attribute("atomic_agents.cancelled", True)
            raise
        except BaseException as error:
            _record_failure(span, error)
            raise


def _tool_input_from_call(args: tuple[object, ...], kwargs: dict[str, object]) -> object:
    return args[0] if args else kwargs.get("params")


def _trace_tool_method(
    wrapped: Callable[..., OutputSchema],
) -> Callable[..., OutputSchema] | Callable[..., Coroutine[object, object, object]]:
    if inspect.iscoroutinefunction(wrapped):
        return _trace_tool_async_method(wrapped)

    @functools.wraps(wrapped)
    def traced(tool: BaseTool, *args: object, **kwargs: object) -> OutputSchema:
        if _active_tool.get() is tool or _suppressed():
            return wrapped(tool, *args, **kwargs)
        with _tool_span(tool, args, kwargs) as span:
            result = _run_as_active_tool(tool, wrapped, args, kwargs)
            _set_output_safely(span, result, ATTR_TOOL_RESULT)
            span.set_status(Status(StatusCode.OK))
            return result

    return traced


def _trace_tool_async_method(
    wrapped: Callable[..., Awaitable[OutputSchema]],
) -> Callable[..., Coroutine[object, object, OutputSchema]]:
    @functools.wraps(wrapped)
    async def traced(tool: BaseTool, *args: object, **kwargs: object) -> OutputSchema:
        if _active_tool.get() is tool or _suppressed():
            return await wrapped(tool, *args, **kwargs)
        with _tool_span(tool, args, kwargs) as span:
            result = await _await_as_active_tool(tool, wrapped, args, kwargs)
            _set_output_safely(span, result, ATTR_TOOL_RESULT)
            span.set_status(Status(StatusCode.OK))
            return result

    return traced


def _run_as_active_tool(
    tool: BaseTool, wrapped: Callable[..., OutputSchema], args: tuple[object, ...], kwargs: dict[str, object]
) -> OutputSchema:
    token = _active_tool.set(tool)
    try:
        return wrapped(tool, *args, **kwargs)
    finally:
        _active_tool.reset(token)


async def _await_as_active_tool(
    tool: BaseTool, wrapped: Callable[..., Awaitable[OutputSchema]], args: tuple[object, ...], kwargs: dict[str, object]
) -> OutputSchema:
    token = _active_tool.set(tool)
    try:
        return await wrapped(tool, *args, **kwargs)
    finally:
        _active_tool.reset(token)
