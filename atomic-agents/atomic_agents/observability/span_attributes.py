"""Span attributes and conversation context."""

import contextvars
import hashlib
import json
import logging
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Callable, Dict, Iterator, List

from opentelemetry import context as context_api
from opentelemetry.util.types import AttributeValue

from atomic_agents.agents.atomic_agent import AtomicAgent

from . import installation


logger = logging.getLogger("atomic_agents.observability.tracing")


ATTR_MODEL = "atomic_agents.model"
ATTR_MODE = "atomic_agents.mode"
ATTR_INPUT_SCHEMA = "atomic_agents.input_schema"
ATTR_OUTPUT_SCHEMA = "atomic_agents.output_schema"
ATTR_HISTORY_MESSAGE_COUNT = "atomic_agents.history.message_count"
ATTR_CONTEXT_PROVIDERS = "atomic_agents.context_providers"
ATTR_SYSTEM_PROMPT_SHA256 = "atomic_agents.system_prompt.sha256"
ATTR_STREAM_CLOSED_EARLY = "atomic_agents.stream.closed_early"
ATTR_OPERATION = "gen_ai.operation.name"
ATTR_AGENT_NAME = "gen_ai.agent.name"
ATTR_REQUEST_MODEL = "gen_ai.request.model"
ATTR_PROVIDER_NAME = "gen_ai.provider.name"
ATTR_CONVERSATION_ID = "gen_ai.conversation.id"
ATTR_TOOL_NAME = "gen_ai.tool.name"
ATTR_TOOL_DESCRIPTION = "gen_ai.tool.description"
ATTR_TOOL_TYPE = "gen_ai.tool.type"
ATTR_TOOL_ARGUMENTS = "gen_ai.tool.call.arguments"
ATTR_TOOL_RESULT = "gen_ai.tool.call.result"
ATTR_SPAN_KIND = "openinference.span.kind"
ATTR_INPUT_VALUE = "input.value"
ATTR_INPUT_MIME_TYPE = "input.mime_type"
ATTR_OUTPUT_VALUE = "output.value"
ATTR_OUTPUT_MIME_TYPE = "output.mime_type"
ATTR_OI_AGENT_NAME = "agent.name"
ATTR_OI_TOOL_NAME = "tool.name"
ATTR_OI_TOOL_DESCRIPTION = "tool.description"
ATTR_TOOL_PARAMETERS = "tool.parameters"
ATTR_SESSION_ID = "session.id"
ATTR_INPUT = "atomic_agents.input"
ATTR_OUTPUT = "atomic_agents.output"
ATTR_AGENT_CLASS = "atomic_agents.agent.class"
ATTR_ERROR_TYPE = "error.type"
ATTR_TOOL_CLASS = "atomic_agents.tool.class"
PROVIDER_NAMES = {
    "azure_openai": "azure.ai.openai",
    "vertexai": "gcp.vertex_ai",
    "gemini": "gcp.gemini",
    "genai": "gcp.gemini",
    "bedrock": "aws.bedrock",
    "mistral": "mistral_ai",
    "xai": "x_ai",
}


def _import_context_attributes() -> Callable[[], dict[str, AttributeValue]] | None:
    try:
        from openinference.instrumentation import get_attributes_from_context
    except ImportError:
        return None
    from atomic_agents.observability import _openinference_is_too_old

    if _openinference_is_too_old():
        logger.warning("OpenInference context attributes disabled: 0.1.38 or newer is required.")
        return None
    return lambda: dict(get_attributes_from_context())


def _context_attributes() -> dict[str, AttributeValue]:
    reader = installation._current_installation().context_attributes
    return reader() if reader is not None else {}


def _suppressed() -> bool:
    return bool(context_api.get_value("suppress_instrumentation"))


_conversation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("atomic_agents_conversation_id", default=None)


@contextmanager
def using_conversation(conversation_id: str) -> Iterator[None]:
    token = _conversation_id.set(conversation_id)
    try:
        yield
    finally:
        _conversation_id.reset(token)


def _agent_display_name(agent: AtomicAgent) -> str:
    attributes = _best_effort_attributes({ATTR_AGENT_NAME: lambda: getattr(agent, "name", None)})
    return str(attributes.get(ATTR_AGENT_NAME) or type(agent).__name__)


def _agent_class_name(agent: AtomicAgent) -> str:
    return f"{type(agent).__name__}[{agent.input_schema.__name__}, {agent.output_schema.__name__}]"


def _best_effort_attributes(
    readers: dict[str, Callable[[], AttributeValue | None]],
) -> dict[str, AttributeValue]:
    attributes: dict[str, AttributeValue] = {}
    for key, read in readers.items():
        try:
            value = read()
            if value is not None:
                attributes[key] = value
        except Exception as error:
            logger.debug("Could not collect tracing attribute %s: %s", key, error)
    return attributes


def _without_none(attributes: Mapping[str, AttributeValue | None]) -> dict[str, AttributeValue]:
    return {key: value for key, value in attributes.items() if value is not None}


def _agent_start_attributes(agent: AtomicAgent) -> dict[str, AttributeValue]:
    attributes: dict[str, AttributeValue] = {
        ATTR_OPERATION: "invoke_agent",
        ATTR_AGENT_NAME: _agent_display_name(agent),
        ATTR_AGENT_CLASS: _agent_class_name(agent),
    }
    attributes.update(
        _best_effort_attributes(
            {
                ATTR_REQUEST_MODEL: lambda: agent.model,
                ATTR_MODEL: lambda: agent.model,
                ATTR_MODE: lambda: getattr(agent.mode, "value", None),
                ATTR_INPUT_SCHEMA: lambda: agent.input_schema.__name__,
                ATTR_OUTPUT_SCHEMA: lambda: agent.output_schema.__name__,
                ATTR_HISTORY_MESSAGE_COUNT: lambda: len(agent.history.history),
                ATTR_CONTEXT_PROVIDERS: lambda: _context_provider_names(agent),
                ATTR_SYSTEM_PROMPT_SHA256: lambda: _system_prompt_version(agent).get(ATTR_SYSTEM_PROMPT_SHA256),
                ATTR_PROVIDER_NAME: lambda: _provider_attributes(agent).get(ATTR_PROVIDER_NAME),
            }
        )
    )
    attributes.update(_optional_attributes("AGENT", {ATTR_OI_AGENT_NAME: _agent_display_name(agent)}))
    return _without_none(attributes)


def _context_provider_names(agent: AtomicAgent) -> List[str]:
    providers = getattr(agent.system_prompt_generator, "context_providers", None) or {}
    return sorted(str(name) for name in providers)


def _system_prompt_version(agent: AtomicAgent) -> Dict[str, str]:
    """Hashes only the static prompt sections, so the value identifies the prompt version across turns
    even though context providers change the rendered prompt on every call."""
    generator = agent.system_prompt_generator
    sections = {name: getattr(generator, name, None) for name in ("background", "steps", "output_instructions")}
    if agent.system_role is None or all(value is None for value in sections.values()):
        return {}
    digest = hashlib.sha256(json.dumps(sections, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return {ATTR_SYSTEM_PROMPT_SHA256: digest}


def _provider_attributes(agent: AtomicAgent) -> dict[str, str]:
    provider = getattr(getattr(agent.client, "provider", None), "value", None)
    return {ATTR_PROVIDER_NAME: PROVIDER_NAMES.get(provider, provider)} if isinstance(provider, str) else {}


def _optional_attributes(kind: str, attributes: dict[str, AttributeValue]) -> dict[str, AttributeValue]:
    context = _context_attributes()
    conversation = _conversation_attributes(context)
    if not installation._current_installation().config.openinference_attributes:
        return conversation
    return _without_none({ATTR_SPAN_KIND: kind, **attributes, **context, **conversation})


def _conversation_attributes(context: dict[str, AttributeValue]) -> dict[str, AttributeValue]:
    conversation_id: AttributeValue | None = _conversation_id.get()
    if conversation_id is None:
        conversation_id = context.get(ATTR_SESSION_ID)
    if conversation_id is None:
        return {}
    attributes = {ATTR_CONVERSATION_ID: conversation_id}
    if installation._current_installation().config.openinference_attributes:
        attributes[ATTR_SESSION_ID] = conversation_id
    return attributes
