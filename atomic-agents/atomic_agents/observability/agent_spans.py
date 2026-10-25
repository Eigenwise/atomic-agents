"""Agent call and stream spans."""

from __future__ import annotations

import asyncio
import functools
from contextlib import contextmanager
from typing import AsyncGenerator, Awaitable, Callable, Coroutine, Generator, Generic, Iterator, TypeVar

from opentelemetry import context as context_api
from opentelemetry import trace as trace_api
from opentelemetry.trace import Span, Status, StatusCode

from atomic_agents.agents.atomic_agent import AtomicAgent

from . import installation, llm_attempt_spans
from .span_attributes import ATTR_STREAM_CLOSED_EARLY, _agent_display_name, _agent_start_attributes, _suppressed
from .span_payloads import _record_failure, _set_input_safely, _set_output_safely

OutputSchema = TypeVar("OutputSchema")
CallResult = TypeVar("CallResult")

AGENT_SYNC_METHOD = "run"
AGENT_ASYNC_METHOD = "run_async"
AGENT_SYNC_STREAM_METHOD = "run_stream"
AGENT_ASYNC_STREAM_METHOD = "run_async_stream"


def _user_input_from_call(args: tuple[object, ...], kwargs: dict[str, object]) -> object:
    return args[0] if args else kwargs.get("user_input")


def _finish_agent_span(span: Span, output: object) -> None:
    _set_output_safely(span, output)
    span.set_status(Status(StatusCode.OK))


def _mark_closed_early(span: Span, output: object) -> None:
    span.set_attribute(ATTR_STREAM_CLOSED_EARLY, True)
    _set_output_safely(span, output)


@contextmanager
def _agent_span(agent: AtomicAgent, args: tuple[object, ...], kwargs: dict[str, object]) -> Iterator[Span]:
    installed = installation._current_installation()
    with installed.tracer.start_as_current_span(
        f"invoke_agent {_agent_display_name(agent)}",
        attributes=_agent_start_attributes(agent),
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        _set_input_safely(span, _user_input_from_call(args, kwargs))
        run = llm_attempt_spans.new_run(span, agent)
        try:
            with llm_attempt_spans.active_run(run):
                yield span
        except (asyncio.CancelledError, GeneratorExit):
            span.set_attribute("atomic_agents.cancelled", True)
            raise
        except BaseException as error:
            _record_failure(span, error)
            raise
        finally:
            llm_attempt_spans.finish_run(run)


def _trace_agent_call(wrapped: Callable[..., OutputSchema]) -> Callable[..., OutputSchema]:
    @functools.wraps(wrapped)
    def traced(agent: AtomicAgent, *args: object, **kwargs: object) -> OutputSchema:
        if _suppressed():
            return wrapped(agent, *args, **kwargs)
        with _agent_span(agent, args, kwargs) as span:
            response = wrapped(agent, *args, **kwargs)
            _finish_agent_span(span, response)
            return response

    return traced


def _trace_agent_async_call(
    wrapped: Callable[..., Awaitable[OutputSchema]],
) -> Callable[..., Coroutine[object, object, OutputSchema]]:
    @functools.wraps(wrapped)
    async def traced(agent: AtomicAgent, *args: object, **kwargs: object) -> OutputSchema:
        if _suppressed():
            return await wrapped(agent, *args, **kwargs)
        with _agent_span(agent, args, kwargs) as span:
            response = await wrapped(agent, *args, **kwargs)
            _finish_agent_span(span, response)
            return response

    return traced


def _start_agent_span(agent: AtomicAgent, args: tuple[object, ...], kwargs: dict[str, object]) -> Span:
    if _suppressed():
        return trace_api.INVALID_SPAN
    span = installation._current_installation().tracer.start_span(
        f"invoke_agent {_agent_display_name(agent)}",
        attributes=_agent_start_attributes(agent),
    )
    _set_input_safely(span, _user_input_from_call(args, kwargs))
    return span


def _call_within_span(run: llm_attempt_spans._AgentRun, function: Callable[..., CallResult], *args: object) -> CallResult:
    span = run.span
    token = context_api.attach(trace_api.set_span_in_context(span))
    try:
        with llm_attempt_spans.active_run(run):
            return function(*args)
    finally:
        context_api.detach(token)


async def _await_within_span(run: llm_attempt_spans._AgentRun, awaitable: Awaitable[CallResult]) -> CallResult:
    span = run.span
    token = context_api.attach(trace_api.set_span_in_context(span))
    try:
        with llm_attempt_spans.active_run(run):
            return await awaitable
    finally:
        context_api.detach(token)


class _SyncStreamCursor(Generic[OutputSchema]):
    def __init__(self, run: llm_attempt_spans._AgentRun, inner: Generator[OutputSchema, None, OutputSchema | None]):
        self.run = run
        self.span = run.span
        self.inner = inner
        self.last: OutputSchema
        self.output: OutputSchema | None = None
        self.final: OutputSchema | None = None

    def advance(self) -> bool:
        try:
            self.last = _call_within_span(self.run, next, self.inner)
            self.output = self.last
        except StopIteration as stop:
            self.final = stop.value
            return False
        return True

    def close_early(self) -> None:
        _call_within_span(self.run, self.inner.close)
        _mark_closed_early(self.span, self.output)
        self.span.set_status(Status(StatusCode.OK))


class _AsyncStreamCursor(Generic[OutputSchema]):
    """Async twin of :class:`_SyncStreamCursor`; async generators carry no return value, so the last partial is the output."""

    def __init__(self, run: llm_attempt_spans._AgentRun, inner: AsyncGenerator[OutputSchema, None]):
        self.run = run
        self.span = run.span
        self.inner = inner
        self.last: OutputSchema
        self.output: OutputSchema | None = None

    async def advance(self) -> bool:
        try:
            self.last = await _await_within_span(self.run, self.inner.__anext__())
            self.output = self.last
        except StopAsyncIteration:
            return False
        return True

    async def close_early(self) -> None:
        await _await_within_span(self.run, self.inner.aclose())
        _mark_closed_early(self.span, self.output)
        self.span.set_status(Status(StatusCode.OK))

    def mark_cancelled(self) -> None:
        _mark_closed_early(self.span, self.output)


def _trace_agent_stream(
    wrapped: Callable[..., Generator[OutputSchema, None, OutputSchema | None]],
) -> Callable[..., Generator[OutputSchema, None, OutputSchema | None]]:
    @functools.wraps(wrapped)
    def traced(agent: AtomicAgent, *args: object, **kwargs: object) -> Generator[OutputSchema, None, OutputSchema | None]:
        span = _start_agent_span(agent, args, kwargs)
        run = llm_attempt_spans.new_run(span, agent)
        cursor = _SyncStreamCursor(run, wrapped(agent, *args, **kwargs))
        try:
            while cursor.advance():
                yield cursor.last
            _finish_agent_span(span, cursor.final)
            return cursor.final
        except GeneratorExit:
            cursor.close_early()
            raise
        except asyncio.CancelledError:
            _mark_closed_early(span, cursor.output)
            raise
        except BaseException as error:
            _record_failure(span, error)
            raise
        finally:
            llm_attempt_spans.finish_run(run)
            span.end()

    return traced


def _trace_agent_async_stream(
    wrapped: Callable[..., AsyncGenerator[OutputSchema, None]],
) -> Callable[..., AsyncGenerator[OutputSchema, None]]:
    @functools.wraps(wrapped)
    async def traced(agent: AtomicAgent, *args: object, **kwargs: object) -> AsyncGenerator[OutputSchema, None]:
        span = _start_agent_span(agent, args, kwargs)
        run = llm_attempt_spans.new_run(span, agent)
        cursor = _AsyncStreamCursor(run, wrapped(agent, *args, **kwargs))
        try:
            while await cursor.advance():
                yield cursor.last
            _finish_agent_span(span, cursor.output)
        except GeneratorExit:
            await cursor.close_early()
            raise
        except asyncio.CancelledError:
            cursor.mark_cancelled()
            raise
        except BaseException as error:
            _record_failure(span, error)
            raise
        finally:
            llm_attempt_spans.finish_run(run)
            span.end()

    return traced


_AGENT_WRAPPERS = {
    AGENT_SYNC_METHOD: _trace_agent_call,
    AGENT_ASYNC_METHOD: _trace_agent_async_call,
    AGENT_SYNC_STREAM_METHOD: _trace_agent_stream,
    AGENT_ASYNC_STREAM_METHOD: _trace_agent_async_stream,
}
