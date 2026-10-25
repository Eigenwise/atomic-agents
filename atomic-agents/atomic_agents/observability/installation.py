"""Tracing installation and method restoration."""

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Callable, Optional, Type

from opentelemetry import trace as trace_api
from opentelemetry.trace import Tracer
from opentelemetry.util.types import AttributeValue

from atomic_agents import __version__
from atomic_agents.agents.atomic_agent import AtomicAgent
from atomic_agents.base.base_tool import BaseTool

from . import agent_spans, llm_attempt_spans, span_attributes, tool_spans
from .tracing_config import PayloadVisibility, TracingConfig

if TYPE_CHECKING:
    from opentelemetry.trace import TracerProvider


logger = logging.getLogger("atomic_agents.observability.tracing")


TRACED_MARKER = "__atomic_agents_traced__"
INHERITED_MARKER = "__atomic_agents_inherited__"
TRACER_NAME = "atomic_agents"


def _current_installation() -> "_Installation":
    assert _installation is not None
    return _installation


class _Installation:
    def __init__(
        self,
        tracer: Tracer,
        tracer_provider: Optional["TracerProvider"],
        config: TracingConfig,
        original_init_subclass: classmethod,
        context_attributes: Callable[[], dict[str, AttributeValue]] | None,
    ):
        self.context_attributes = context_attributes
        self.tracer = tracer
        self.tracer_provider = tracer_provider
        self.config = config
        self.original_init_subclass = original_init_subclass


_installation: Optional[_Installation] = None


def is_installed() -> bool:
    return _installation is not None


def install(
    *, tracer_provider: Optional["TracerProvider"] = None, config: TracingConfig | PayloadVisibility | None = None
) -> None:
    global _installation
    if _installation is not None:
        tracer_provider = _installation.tracer_provider if tracer_provider is None else tracer_provider
        _replace_installation(_installation, tracer_provider, config)
        return
    config = TracingConfig.coerce(config)
    context_attributes = span_attributes._import_context_attributes()
    if config.openinference_attributes is None:
        config = replace(config, openinference_attributes=context_attributes is not None)
    tracer = trace_api.get_tracer(TRACER_NAME, __version__, tracer_provider)
    original_init_subclass = BaseTool.__dict__["__init_subclass__"]
    _installation = _Installation(tracer, tracer_provider, config, original_init_subclass, context_attributes)
    _instrument_agent_class()
    tool_spans._instrument_existing_tool_classes()
    tool_spans._hook_future_tool_classes(original_init_subclass)


def uninstall() -> None:
    global _installation
    if _installation is None:
        return
    setattr(BaseTool, "__init_subclass__", _installation.original_init_subclass)
    for tool_class in tool_spans._all_subclasses(BaseTool):
        tool_spans._restore_tool_class(tool_class)
    for method_name in agent_spans._AGENT_WRAPPERS:
        _restore_method(AtomicAgent, method_name)
    llm_attempt_spans.unregister_clients()
    _installation = None


def _replace_installation(
    installed: _Installation, tracer_provider: Optional["TracerProvider"], config: TracingConfig | PayloadVisibility | None
) -> None:
    config = installed.config if config is None else TracingConfig.coerce(config)
    if config.openinference_attributes is None:
        config = replace(config, openinference_attributes=installed.config.openinference_attributes)
    if tracer_provider is installed.tracer_provider and config == installed.config:
        return
    installed.tracer = trace_api.get_tracer(TRACER_NAME, __version__, tracer_provider)
    installed.tracer_provider = tracer_provider
    installed.config = config
    installed.context_attributes = span_attributes._import_context_attributes()
    logger.warning(
        "AtomicAgentsInstrumentor tracer provider/config replaced; "
        "in-flight spans finish on the provider they started with."
    )


def _instrument_agent_class() -> None:
    for method_name, wrap in agent_spans._AGENT_WRAPPERS.items():
        original = AtomicAgent.__dict__[method_name]
        if not getattr(original, TRACED_MARKER, False):
            setattr(AtomicAgent, method_name, _mark_traced(wrap(original), original, False))


def _mark_traced(wrapper: Callable, original: Callable, inherited: bool) -> Callable:
    setattr(wrapper, TRACED_MARKER, True)
    setattr(wrapper, INHERITED_MARKER, inherited)
    setattr(wrapper, "__wrapped__", original)
    return wrapper


def _restore_method(owner: Type, method_name: str) -> None:
    current = owner.__dict__.get(method_name)
    if current is None or not getattr(current, TRACED_MARKER, False):
        return
    if getattr(current, INHERITED_MARKER, False):
        delattr(owner, method_name)
    else:
        setattr(owner, method_name, current.__wrapped__)
