"""Opt-in tracing for any OpenTelemetry backend using GenAI attributes.

Install the observability extra for tracing. The optional phoenix extra adds
Arize Phoenix integration and OpenInference attributes.
"""

import importlib.util
import logging
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from atomic_agents.observability.tracing import PayloadVisibility, TracingConfig, using_conversation
    from opentelemetry.trace import TracerProvider

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "Atomic Agents tracing needs opentelemetry-api. "
    'Install it with: pip install "atomic-agents[observability]" '
    '(add "atomic-agents[phoenix]" for Arize Phoenix / OpenInference attributes)'
)
_OPENINFERENCE_DISTRIBUTION = "openinference-instrumentation"
_MINIMUM_OPENINFERENCE_VERSION = (0, 1, 38)


def _opentelemetry_is_installed() -> bool:
    try:
        return importlib.util.find_spec("opentelemetry.trace") is not None
    except ModuleNotFoundError:
        return False


def _openinference_is_too_old() -> bool:
    try:
        installed = version(_OPENINFERENCE_DISTRIBUTION)
    except PackageNotFoundError:
        return True
    release = tuple(int(part) for part in installed.split(".")[:3] if part.isdigit())
    return release < _MINIMUM_OPENINFERENCE_VERSION


def _load_tracing_module():
    from atomic_agents.observability import tracing

    return tracing


def __getattr__(name: str) -> object:
    if name in ("TracingConfig", "using_conversation"):
        return getattr(_load_tracing_module(), name)
    raise AttributeError(name)


def _warn_unknown_options(options: dict[str, object]) -> None:
    for key in options:
        if key != "skip_dep_check":
            logger.warning("Unknown Atomic Agents instrumentation option %s ignored", key)


class AtomicAgentsInstrumentor:
    """
    Installs AGENT, TOOL and LLM-attempt spans around ``AtomicAgent`` and ``BaseTool`` calls.

    The public surface mirrors OpenTelemetry's ``BaseInstrumentor`` (``instrument`` / ``uninstrument``)
    without depending on it, because Arize Phoenix only guarantees ``opentelemetry-api`` and
    ``openinference-instrumentation`` in an auto-instrumented environment.
    """

    def instrument(
        self,
        *,
        tracer_provider: "TracerProvider | None" = None,
        config: "TracingConfig | PayloadVisibility | None" = None,
        **options: object,
    ) -> None:
        """
        Patch the framework classes. Safe to call more than once; a second call with different
        arguments replaces the tracer provider/config and logs a warning.

        Args:
            tracer_provider: An OpenTelemetry ``TracerProvider``. Defaults to the global provider.
            config: Payload visibility and truncation settings. OpenInference TraceConfig is also accepted.
            **options: Launcher options accepted for OpenTelemetry auto-instrumentation compatibility.

        Raises:
            ModuleNotFoundError: When the ``observability`` extra is not installed, except during
                OpenTelemetry auto-discovery, which skips unavailable optional instrumentation.
        """
        _warn_unknown_options(options)
        if not _opentelemetry_is_installed():
            if options.get("skip_dep_check") is True:
                logger.debug(_INSTALL_HINT)
                return
            raise ModuleNotFoundError(_INSTALL_HINT)
        _load_tracing_module().install(tracer_provider=tracer_provider, config=config)

    def uninstrument(self) -> None:
        """Restore the original framework methods. No-op when nothing is installed."""
        if not self.is_instrumented_by_opentelemetry:
            return
        _load_tracing_module().uninstall()

    @property
    def is_instrumented_by_opentelemetry(self) -> bool:
        if not _opentelemetry_is_installed():
            return False
        return _load_tracing_module().is_installed()


__all__ = ["AtomicAgentsInstrumentor", "TracingConfig", "using_conversation"]
