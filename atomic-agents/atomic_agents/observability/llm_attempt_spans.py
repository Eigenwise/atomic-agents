"""Instructor attempt spans and cumulative agent usage."""

from __future__ import annotations

import contextvars
import functools
import logging
import threading
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator, ParamSpec, Protocol

from opentelemetry import context as context_api, trace as trace_api
from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer
from opentelemetry.util.types import AttributeValue

from atomic_agents.agents.atomic_agent import AtomicAgent

from . import installation
from .span_attributes import _best_effort_attributes, _optional_attributes, _provider_attributes, _without_none
from .span_payloads import _bounded_text, _failure_message, _record_failure, _write_payload
from .tracing_config import TracingConfig

logger = logging.getLogger("atomic_agents.observability.tracing")
_PARAMETERS = ParamSpec("_PARAMETERS")
_RESPONSE_MODEL = "gen_ai.response.model"
_ATTEMPT = "atomic_agents.llm.attempt"
_VALIDATION_ERROR = "atomic_agents.llm.last_validation_error"


class _PromptTokenDetails(Protocol):
    cached_tokens: int | None


class _CompletionTokenDetails(Protocol):
    reasoning_tokens: int | None


class _InputOutputUsage(Protocol):
    input_tokens: int
    output_tokens: int
    input_tokens_details: _PromptTokenDetails | None
    output_tokens_details: _CompletionTokenDetails | None


class _PromptCompletionUsage(Protocol):
    prompt_tokens: int
    completion_tokens: int
    prompt_tokens_details: _PromptTokenDetails | None
    completion_tokens_details: _CompletionTokenDetails | None


_Usage = _InputOutputUsage | _PromptCompletionUsage


class _Response(Protocol):
    model: str
    usage: _Usage


class _HookClient(Protocol):
    def on(self, event: str, handler: Callable[..., None]) -> object:
        """Append a handler without replacing application hooks."""

    def off(self, event: str, handler: Callable[..., None]) -> object:
        """Remove only this handler, preserving application hooks."""


@dataclass
class _AgentRun:
    span: Span
    tracer: Tracer
    config: TracingConfig
    provider: dict[str, AttributeValue]
    usage_totals: dict[str, int] = field(default_factory=dict)
    attempt_count: int = 0
    suppressed_attempt: bool = False
    response_received: bool = False
    open_attempt: Span | None = None
    context_token: contextvars.Token | None = None


_active_run: contextvars.ContextVar[_AgentRun | None] = contextvars.ContextVar("atomic_agents_llm_run", default=None)
_clients: weakref.WeakSet[_HookClient] = weakref.WeakSet()
_registration_lock = threading.Lock()


def _never_raise(handler: Callable[_PARAMETERS, None]) -> Callable[_PARAMETERS, None]:
    @functools.wraps(handler)
    def guarded(*args: _PARAMETERS.args, **kwargs: _PARAMETERS.kwargs) -> None:
        try:
            handler(*args, **kwargs)
        except Exception:
            logger.warning("Instructor tracing hook failed: %s", handler.__name__, exc_info=True)

    return guarded


@_never_raise
def register_client(client: _HookClient) -> None:
    if not hasattr(client, "on"):
        return
    # Shared clients must acquire all handlers once before another run can use them.
    with _registration_lock:
        if client not in _clients:
            _register_handlers(client)
            _clients.add(client)


def _register_handlers(client: _HookClient) -> None:
    registered: list[tuple[str, Callable[..., None]]] = []
    try:
        for event, handler in _HANDLERS.items():
            client.on(event, handler)
            registered.append((event, handler))
    except Exception:
        for event, handler in registered:
            client.off(event, handler)
        raise


@_never_raise
def unregister_clients() -> None:
    with _registration_lock:
        for client in list(_clients):
            _unregister_client(client)
        _clients.clear()


@_never_raise
def _unregister_client(client: _HookClient) -> None:
    for event, handler in _HANDLERS.items():
        client.off(event, handler)


def new_run(span: Span, agent: AtomicAgent) -> _AgentRun:
    installed = installation._current_installation()
    register_client(getattr(agent, "client"))
    provider = _best_effort_attributes(
        {"gen_ai.provider.name": lambda: _provider_attributes(agent).get("gen_ai.provider.name")}
    )
    return _AgentRun(span, installed.tracer, installed.config, provider)


@contextmanager
def active_run(run: _AgentRun) -> Iterator[None]:
    token = _active_run.set(None if run.span is trace_api.INVALID_SPAN else run)
    _attach_attempt(run)
    try:
        yield
    finally:
        _detach_attempt(run)
        _active_run.reset(token)


def _attach_attempt(run: _AgentRun) -> None:
    if run.open_attempt is not None:
        run.context_token = context_api.attach(trace_api.set_span_in_context(run.open_attempt))


def _detach_attempt(run: _AgentRun) -> None:
    if run.context_token is not None:
        context_api.detach(run.context_token)
        run.context_token = None


@_never_raise
def finish_attempt(run: _AgentRun, status: Status | None = None) -> None:
    if run.open_attempt is None:
        return
    if run.response_received:
        status = Status(StatusCode.OK)
    run.response_received = False
    if status is not None:
        run.open_attempt.set_status(status)
    run.open_attempt.end()
    run.open_attempt = None
    _detach_attempt(run)


@_never_raise
def _completion_kwargs(*args: object, **kwargs: object) -> None:
    run = _active_run.get()
    if run is None:
        return
    finish_attempt(run)
    # Suppressed kwargs must also swallow later response events outside that context.
    run.suppressed_attempt = bool(context_api.get_value("suppress_instrumentation"))
    if run.suppressed_attempt:
        return
    run.attempt_count += 1
    if not run.config.llm_spans:
        return
    model = _request_model(kwargs)
    attributes: dict[str, AttributeValue | None] = {
        "gen_ai.operation.name": "chat",
        "gen_ai.request.model": model,
        _ATTEMPT: run.attempt_count,
    }
    attributes.update(run.provider)
    attributes.update(_optional_attributes("LLM", {}))
    run.open_attempt = run.tracer.start_span(
        f"chat {model}" if model else "chat", kind=SpanKind.CLIENT, attributes=_without_none(attributes)
    )
    _attach_attempt(run)


def _request_model(kwargs: dict[str, object]) -> str | None:
    model = kwargs.get("model")
    return str(model) if model is not None else None


def _usage_attributes(usage: _Usage | None) -> dict[str, int]:
    readers = {
        "input_tokens": getattr(usage, "input_tokens", getattr(usage, "prompt_tokens", None)),
        "output_tokens": getattr(usage, "output_tokens", getattr(usage, "completion_tokens", None)),
        "cache_read.input_tokens": getattr(
            getattr(usage, "input_tokens_details", getattr(usage, "prompt_tokens_details", None)), "cached_tokens", None
        ),
        "reasoning.output_tokens": getattr(
            getattr(usage, "output_tokens_details", getattr(usage, "completion_tokens_details", None)),
            "reasoning_tokens",
            None,
        ),
    }
    return {"gen_ai.usage." + key: value for key, value in readers.items() if isinstance(value, int)}


@_never_raise
def _completion_response(response: _Response, **kwargs: object) -> None:
    run = _active_run.get()
    if run is None or run.suppressed_attempt:
        return
    attributes = _without_none({_RESPONSE_MODEL: getattr(response, "model", None)})
    run.span.set_attributes(attributes)
    usage = _usage_attributes(getattr(response, "usage", None))
    for key, value in usage.items():
        run.usage_totals[key] = run.usage_totals.get(key, 0) + value
    _complete_response_attempt(run, usage, attributes)


def _complete_response_attempt(run: _AgentRun, usage: dict[str, int], attributes: dict[str, AttributeValue]) -> None:
    if run.open_attempt is None:
        return
    run.open_attempt.set_attributes(attributes)
    run.open_attempt.set_attributes(usage)
    run.response_received = True


@_never_raise
def _completion_error(error: BaseException, **kwargs: object) -> None:
    run = _active_run.get()
    if run is None or run.suppressed_attempt or run.open_attempt is None:
        return
    run.response_received = False
    _record_failure(run.open_attempt, error, run.config)
    finish_attempt(run)


@_never_raise
def _parse_error(error: BaseException, **kwargs: object) -> None:
    run = _active_run.get()
    if run is None or run.suppressed_attempt:
        return
    text = "__REDACTED__" if run.config.hide_outputs else _failure_message(error, run.config)
    _write_payload(run.span, _VALIDATION_ERROR, None, text, run.config.hide_outputs, run.config)
    event = {_ATTEMPT: run.attempt_count, _VALIDATION_ERROR: _bounded_text(text, run.config)}
    run.span.add_event("validation_error", event)
    _mark_attempt_validation_error(run, error)


def _mark_attempt_validation_error(run: _AgentRun, error: BaseException) -> None:
    if run.open_attempt is None:
        return
    run.response_received = False
    message = _bounded_text(_failure_message(error, run.config), run.config)
    run.open_attempt.add_event("validation_error", {_ATTEMPT: run.attempt_count, _VALIDATION_ERROR: message})
    run.open_attempt.set_attribute("error.type", type(error).__qualname__)
    run.open_attempt.set_status(Status(StatusCode.ERROR, message))


@_never_raise
def finish_run(run: _AgentRun) -> None:
    finish_attempt(run)
    run.span.set_attributes(run.usage_totals)
    run.span.set_attribute("atomic_agents.llm.attempts", run.attempt_count)


_HANDLERS: dict[str, Callable[..., None]] = {
    "completion:kwargs": _completion_kwargs,
    "completion:response": _completion_response,
    "completion:error": _completion_error,
    "parse:error": _parse_error,
}
