import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from opentelemetry import context as context_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from atomic_agents.observability import TracingConfig
from atomic_agents.observability.message_payloads import write_messages


class UnreadableContent:
    def __init__(self):
        self.reads = 0

    @property
    def role(self):
        self.reads += 1
        raise AssertionError("content was accessed")

    @property
    def choices(self):
        self.reads += 1
        raise AssertionError("choices were accessed")


class UnconsumableStream:
    def __iter__(self):
        raise AssertionError("stream was consumed")


def captured(value, direction="input", **options):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with provider.get_tracer(__name__).start_as_current_span("chat") as span:
        write_messages(span, value, direction, replace(TracingConfig(capture_messages=True), **options))
    return dict(exporter.get_finished_spans()[0].attributes)


def decoded(value, direction="input"):
    return json.loads(captured(value, direction)[f"gen_ai.{direction}.messages"])


def test_message_and_part_order():
    messages = [
        {"role": "system", "content": "instructions"},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": UnreadableContent()},
                {"type": "text", "text": "first"},
                {"type": "audio", "data": "private"},
                {"type": "text", "text": "second"},
            ],
        },
        {
            "role": "assistant",
            "tool_calls": [
                {"type": "function", "id": "call-1", "function": {"name": "lookup", "arguments": '{"city":"Ghent"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "sunny"},
    ]
    assert decoded(messages) == [
        {"role": "system", "parts": [{"type": "text", "content": "instructions"}]},
        {"role": "user", "parts": [{"type": "text", "content": "first"}, {"type": "text", "content": "second"}]},
        {
            "role": "assistant",
            "parts": [{"type": "tool_call", "id": "call-1", "name": "lookup", "arguments": {"city": "Ghent"}}],
        },
        {"role": "tool", "parts": [{"type": "tool_call_response", "id": "call-1", "response": "sunny"}]},
    ]


def test_output_choices_remain_separate_and_ordered():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(role="assistant", content="first")),
            {"message": {"role": "assistant", "content": "second"}},
        ]
    )
    assert decoded(response, "output") == [
        {"role": "assistant", "parts": [{"type": "text", "content": "first"}]},
        {"role": "assistant", "parts": [{"type": "text", "content": "second"}]},
    ]


@pytest.mark.parametrize(
    "options",
    [
        {"capture_messages": False},
        {"hide_inputs": True},
        {"llm_spans": False},
    ],
)
def test_input_gates_precede_content_access(options):
    content = UnreadableContent()
    assert captured(content, **options) == {}
    assert content.reads == 0


def test_output_gate_precedes_content_access():
    content = UnreadableContent()
    assert captured(content, "output", hide_outputs=True) == {}
    assert content.reads == 0


@pytest.mark.parametrize("direction", ["input", "output"])
def test_suppression_precedes_content_access(direction):
    token = context_api.attach(context_api.set_value("suppress_instrumentation", True))
    try:
        assert captured(UnreadableContent(), direction) == {}
    finally:
        context_api.detach(token)


@pytest.mark.parametrize(
    "value",
    [
        None,
        UnconsumableStream(),
        UnreadableContent(),
        {},
        [{"content": "missing role"}],
        [{"role": "user", "content": object()}],
        [{"role": "user", "content": ""}],
        [{"role": "user", "content": [{"type": "reasoning", "text": "secret"}]}],
        [{"role": "tool", "content": None}],
        [{"role": "assistant", "tool_calls": [{"type": "other"}]}],
        [{"role": "assistant", "tool_calls": [{"type": "function", "function": {"arguments": "{}"}}]}],
        [{"role": "assistant", "tool_calls": [{"type": "function", "function": {"name": "call", "arguments": "broken"}}]}],
    ],
)
def test_unrepresentable_input_is_omitted(value):
    assert captured(value) == {}


@pytest.mark.parametrize(
    "value",
    [
        UnconsumableStream(),
        {"output": [{"type": "function_call"}]},
        {"choices": UnconsumableStream()},
        {"choices": [{"delta": {"content": "stream"}}]},
    ],
)
def test_unsupported_response_is_omitted(value):
    assert captured(value, "output") == {}


@pytest.mark.parametrize("text", ['quote"\\\n', "Ghent ☀️ 漢字", "long" * 100])
def test_complete_json_length_bound(text):
    messages = [{"role": "user", "content": text}]
    encoded = captured(messages)["gen_ai.input.messages"]
    assert json.loads(encoded)[0]["parts"][0]["content"] == text
    assert captured(messages, max_attribute_length=len(encoded))["gen_ai.input.messages"] == encoded
    assert captured(messages, max_attribute_length=len(encoded) - 1) == {}
    assert captured(messages, max_attribute_length=1) == {}


@pytest.mark.parametrize("setting, expected", [("true", True), ("TRUE", True), ("1", True), ("false", False), ("yes", False)])
def test_environment_and_explicit_config_precedence(monkeypatch, setting, expected):
    monkeypatch.setenv("ATOMIC_AGENTS_TRACE_CAPTURE_MESSAGES", setting)
    assert TracingConfig.from_environment().capture_messages is expected
    assert TracingConfig.coerce(None).capture_messages is expected
    visibility = SimpleNamespace(hide_inputs=True, hide_outputs=False)
    assert TracingConfig.coerce(visibility).capture_messages is expected
    assert TracingConfig.coerce(TracingConfig()).capture_messages is False
    assert TracingConfig.coerce(TracingConfig(capture_messages=True)).capture_messages is True


def test_capture_defaults_preserve_positional_fields(monkeypatch):
    monkeypatch.delenv("ATOMIC_AGENTS_TRACE_CAPTURE_MESSAGES", raising=False)
    assert TracingConfig(True, False, 50, False, False).capture_messages is False
    assert TracingConfig.from_environment().capture_messages is False
