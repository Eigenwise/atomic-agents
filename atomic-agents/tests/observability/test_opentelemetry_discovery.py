import logging
import os
from collections.abc import Iterator, Mapping
from importlib.metadata import EntryPoint, distribution, entry_points
from pathlib import Path
import subprocess
import sys
import sysconfig
from unittest.mock import Mock

import pytest
from openinference.instrumentation import TraceConfig
from openinference.semconv.trace import SpanAttributes
from opentelemetry.instrumentation.auto_instrumentation import _load
from opentelemetry.instrumentation.distro import DefaultDistro
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.util.types import AttributeValue

import atomic_agents.observability as observability
from atomic_agents.observability import AtomicAgentsInstrumentor
from test_tracing import CalculationInputSchema, CalculatorTool, QuestionSchema, make_agent, sync_client


@pytest.fixture
def local_tracing() -> Iterator[tuple[TracerProvider, InMemorySpanExporter]]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider, exporter
    AtomicAgentsInstrumentor().uninstrument()
    provider.shutdown()


def installed_entry_point(group: str) -> EntryPoint:
    points = [point for point in entry_points(group=group) if point.name == "atomic_agents"]
    assert len(points) == 1, f"Expected one installed Atomic Agents entry point in {group}"
    return points[0]


def exported_attributes(span: ReadableSpan) -> Mapping[str, AttributeValue]:
    attributes = span.attributes
    assert attributes is not None
    return attributes


def run_agent_and_tool() -> None:
    make_agent(sync_client()).run(QuestionSchema(question="private question"))
    CalculatorTool().run(CalculationInputSchema(expression="private expression"))


def test_installed_opentelemetry_entry_point_activates_real_loader(
    monkeypatch: pytest.MonkeyPatch, local_tracing: tuple[TracerProvider, InMemorySpanExporter]
) -> None:
    provider, exporter = local_tracing
    instrumentor = installed_entry_point("opentelemetry_instrumentor")
    monkeypatch.setattr(_load, "entry_points", lambda group: (instrumentor,) if group == "opentelemetry_instrumentor" else ())
    monkeypatch.setattr("opentelemetry.trace.get_tracer_provider", lambda: provider)
    monkeypatch.delenv("OTEL_PYTHON_DISABLED_INSTRUMENTATIONS", raising=False)
    monkeypatch.setenv("OPENINFERENCE_HIDE_INPUTS", "true")
    monkeypatch.setenv("OPENINFERENCE_HIDE_OUTPUTS", "true")

    _load._load_instrumentors(DefaultDistro())
    run_agent_and_tool()

    spans = exporter.get_finished_spans()
    assert [exported_attributes(span)[SpanAttributes.OPENINFERENCE_SPAN_KIND] for span in spans] == ["AGENT", "TOOL"]
    assert [exported_attributes(span)[SpanAttributes.INPUT_VALUE] for span in spans] == ["__REDACTED__", "__REDACTED__"]
    assert [exported_attributes(span)[SpanAttributes.OUTPUT_VALUE] for span in spans] == ["__REDACTED__", "__REDACTED__"]


def test_unknown_launcher_options_preserve_provider_and_privacy(
    local_tracing: tuple[TracerProvider, InMemorySpanExporter],
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider, exporter = local_tracing
    AtomicAgentsInstrumentor().instrument(
        tracer_provider=provider,
        config=TraceConfig(hide_inputs=True, hide_outputs=True),
        skip_dep_check=True,
        custom_launcher_option=object(),
    )
    run_agent_and_tool()

    assert "custom_launcher_option" in caplog.text
    assert any(record.levelno == logging.WARNING for record in caplog.records)
    assert "skip_dep_check" not in caplog.text
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    assert [exported_attributes(span)[SpanAttributes.INPUT_VALUE] for span in spans] == ["__REDACTED__", "__REDACTED__"]
    assert [exported_attributes(span)[SpanAttributes.OUTPUT_VALUE] for span in spans] == ["__REDACTED__", "__REDACTED__"]


def test_loader_skips_missing_opentelemetry_and_continues_without_error(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    instrumentor = installed_entry_point("opentelemetry_instrumentor")
    sentinel = Mock(
        name="sentinel", group="opentelemetry_instrumentor", value="sentinel:Instrumentor", dist=distribution("atomic-agents")
    )
    sentinel.name = "sentinel"
    points = (instrumentor, sentinel)
    monkeypatch.setattr(_load, "entry_points", lambda group: points if group == "opentelemetry_instrumentor" else ())
    monkeypatch.setattr(observability, "_opentelemetry_is_installed", lambda: False)
    monkeypatch.delenv("OTEL_PYTHON_DISABLED_INSTRUMENTATIONS", raising=False)

    with caplog.at_level(logging.DEBUG):
        _load._load_instrumentors(DefaultDistro())

    sentinel.load.return_value.return_value.instrument.assert_called_once_with(skip_dep_check=True)
    assert (
        'Atomic Agents tracing needs opentelemetry-api. Install it with: pip install "atomic-agents[observability]" '
        '(add "atomic-agents[phoenix]" for Arize Phoenix / OpenInference attributes)'
    ) in caplog.text
    assert not [record for record in caplog.records if record.levelno >= logging.ERROR]
    assert not AtomicAgentsInstrumentor().is_instrumented_by_opentelemetry


@pytest.mark.parametrize("first_group", ["openinference_instrumentor", "opentelemetry_instrumentor"])
def test_both_entry_points_replace_provider_and_config(
    first_group: str, local_tracing: tuple[TracerProvider, InMemorySpanExporter], caplog: pytest.LogCaptureFixture
) -> None:
    provider, exporter = local_tracing
    second_group = {
        "openinference_instrumentor": "opentelemetry_instrumentor",
        "opentelemetry_instrumentor": "openinference_instrumentor",
    }[first_group]
    first = installed_entry_point(first_group).load()()
    second = installed_entry_point(second_group).load()()
    other_provider = TracerProvider()
    other_exporter = InMemorySpanExporter()
    other_provider.add_span_processor(SimpleSpanProcessor(other_exporter))
    first.instrument(tracer_provider=provider, config=TraceConfig(hide_inputs=True))
    try:
        with caplog.at_level(logging.WARNING):
            second.instrument(tracer_provider=other_provider, config=TraceConfig(hide_inputs=False), skip_dep_check=True)
        run_agent_and_tool()
        assert "tracer provider/config replaced" in caplog.text
        assert exporter.get_finished_spans() == ()
        spans = other_exporter.get_finished_spans()
        assert len(spans) == 2
        assert all(exported_attributes(span)[SpanAttributes.INPUT_VALUE] != "__REDACTED__" for span in spans)
    finally:
        second.uninstrument()
        other_provider.shutdown()


@pytest.mark.parametrize("first_group", ["openinference_instrumentor", "opentelemetry_instrumentor"])
def test_either_entry_point_can_uninstrument_and_reinstall(
    first_group: str, local_tracing: tuple[TracerProvider, InMemorySpanExporter]
) -> None:
    provider, exporter = local_tracing
    second_group = {
        "openinference_instrumentor": "opentelemetry_instrumentor",
        "opentelemetry_instrumentor": "openinference_instrumentor",
    }[first_group]
    first = installed_entry_point(first_group).load()()
    second = installed_entry_point(second_group).load()()
    first.instrument(tracer_provider=provider)
    second.instrument(tracer_provider=provider)
    run_agent_and_tool()
    assert len(exporter.get_finished_spans()) == 2

    second.uninstrument()
    first.uninstrument()
    run_agent_and_tool()
    assert len(exporter.get_finished_spans()) == 2

    second.instrument(tracer_provider=provider, config=TraceConfig(hide_inputs=True), skip_dep_check=True)
    exporter.clear()
    run_agent_and_tool()
    assert [exported_attributes(span)[SpanAttributes.INPUT_VALUE] for span in exporter.get_finished_spans()] == [
        "__REDACTED__",
        "__REDACTED__",
    ]


@pytest.mark.parametrize("disabled_name,expected_kinds", [("", ["AGENT", "TOOL"]), ("atomic_agents", [])])
def test_real_launcher_respects_disabled_name(disabled_name: str, expected_kinds: list[str], tmp_path: Path) -> None:
    executable_name = "opentelemetry-instrument.exe" if os.name == "nt" else "opentelemetry-instrument"
    launcher = Path(sysconfig.get_path("scripts")) / executable_name
    environment = os.environ.copy()
    environment.update(
        OTEL_TRACES_EXPORTER="none",
        OTEL_METRICS_EXPORTER="none",
        OTEL_LOGS_EXPORTER="none",
        OTEL_PYTHON_DISABLED_INSTRUMENTATIONS=disabled_name,
        OPENINFERENCE_HIDE_INPUTS="true",
        OPENINFERENCE_HIDE_OUTPUTS="true",
    )
    script = f"""
import sys
sys.path.insert(0, {str(Path(__file__).parent)!r})
from test_opentelemetry_discovery import exported_attributes, run_agent_and_tool
from atomic_agents.observability import AtomicAgentsInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
assert AtomicAgentsInstrumentor().is_instrumented_by_opentelemetry is {bool(expected_kinds)!r}
provider = TracerProvider()
exporter = InMemorySpanExporter()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)
run_agent_and_tool()
spans = exporter.get_finished_spans()
assert len(spans) == {len(expected_kinds)}
assert [exported_attributes(span)['openinference.span.kind'] for span in spans] == {expected_kinds!r}
assert [exported_attributes(span)['input.value'] for span in spans] == {['__REDACTED__'] * len(expected_kinds)!r}
AtomicAgentsInstrumentor().uninstrument()
provider.shutdown()
print('launcher assertions passed')
"""
    script_path = tmp_path / "launcher_probe.py"
    script_path.write_text(script, encoding="utf-8")
    result = subprocess.run(
        [str(launcher), sys.executable, str(script_path)], env=environment, capture_output=True, text=True, timeout=45
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "launcher assertions passed" in result.stdout
    assert "Traceback" not in result.stderr


def test_normal_import_does_not_require_optional_observability_packages() -> None:
    script = """
import sys
sys.modules['openinference'] = None
sys.modules['opentelemetry'] = None
import atomic_agents
import atomic_agents.observability
assert 'atomic_agents.observability.tracing' not in sys.modules
print('optional-free import passed')
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "optional-free import passed" in result.stdout
