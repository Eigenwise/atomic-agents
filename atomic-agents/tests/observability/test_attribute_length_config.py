from dataclasses import replace

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from atomic_agents.observability import TracingConfig
from atomic_agents.observability.span_payloads import _record_payload


@pytest.mark.parametrize("limit", [-100, -1, 0, True, False, 1.5, "32", None])
def test_invalid_explicit_payload_limit_is_rejected_before_instrumentation(limit):
    with pytest.raises(ValueError, match="max_attribute_length.*positive integer"):
        TracingConfig(max_attribute_length=limit)


def test_replacing_a_config_cannot_install_an_invalid_limit():
    config = TracingConfig(max_attribute_length=32)

    with pytest.raises(ValueError, match="max_attribute_length.*positive integer"):
        replace(config, max_attribute_length=-1)
    assert config.max_attribute_length == 32


@pytest.mark.parametrize("limit", [1, 13, 14, 32, 4096])
def test_valid_explicit_limits_bound_exported_payloads(limit, monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "false")
    monkeypatch.setenv("OTEL_TRACES_SAMPLER", "always_on")
    config = TracingConfig(max_attribute_length=limit)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        with provider.get_tracer(__name__).start_as_current_span("bounded") as span:
            _record_payload(span, (("payload", None),), "x" * 5000, False, config)

        attributes = exporter.get_finished_spans()[0].attributes
        assert len(attributes["payload"]) == limit
        assert attributes["payload.original_length"] == 5002
    finally:
        provider.shutdown()
