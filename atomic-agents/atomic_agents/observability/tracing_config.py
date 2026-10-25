"""Tracing configuration and payload visibility."""

import logging
import os
from dataclasses import dataclass, replace
from typing import Protocol


logger = logging.getLogger("atomic_agents.observability.tracing")
DEFAULT_MAX_ATTRIBUTE_LENGTH = 4096


class PayloadVisibility(Protocol):
    hide_inputs: bool
    hide_outputs: bool


@dataclass(frozen=True)
class TracingConfig:
    hide_inputs: bool = False
    hide_outputs: bool = False
    max_attribute_length: int = DEFAULT_MAX_ATTRIBUTE_LENGTH
    openinference_attributes: bool | None = None
    llm_spans: bool = True

    @classmethod
    def from_environment(cls) -> "TracingConfig":
        return cls(
            hide_inputs=_visibility_environment("INPUTS").lower() in ("true", "1"),
            hide_outputs=_visibility_environment("OUTPUTS").lower() in ("true", "1"),
            max_attribute_length=_attribute_length_environment(),
            llm_spans=os.environ.get("ATOMIC_AGENTS_TRACE_LLM_SPANS", "true").lower() not in ("false", "0"),
        )

    @classmethod
    def coerce(cls, config: "TracingConfig | PayloadVisibility | None") -> "TracingConfig":
        if config is None:
            return cls.from_environment()
        if isinstance(config, cls):
            return config
        return replace(cls.from_environment(), hide_inputs=config.hide_inputs, hide_outputs=config.hide_outputs)


def _visibility_environment(direction: str) -> str:
    return os.environ.get(f"ATOMIC_AGENTS_TRACE_HIDE_{direction}", os.environ.get(f"OPENINFERENCE_HIDE_{direction}", ""))


def _attribute_length_environment() -> int:
    value = os.environ.get("ATOMIC_AGENTS_TRACE_MAX_ATTRIBUTE_LENGTH", str(DEFAULT_MAX_ATTRIBUTE_LENGTH))
    try:
        length = int(value)
        if length >= 1:
            return length
    except ValueError:
        pass
    logger.warning("Invalid ATOMIC_AGENTS_TRACE_MAX_ATTRIBUTE_LENGTH value %r; using %d", value, DEFAULT_MAX_ATTRIBUTE_LENGTH)
    return DEFAULT_MAX_ATTRIBUTE_LENGTH
