"""Exercise payload work against real SDK sampling and API-only no-op spans."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF, ALWAYS_ON, Decision, Sampler, SamplingResult
from opentelemetry.trace import StatusCode
from pydantic import model_serializer

from atomic_agents import BaseIOSchema, BaseTool, BaseToolConfig
from atomic_agents.observability import AtomicAgentsInstrumentor, TracingConfig
from atomic_agents.observability.message_payloads import write_messages


class RecordOnlySampler(Sampler):
    def should_sample(self, parent_context, trace_id, name, kind=None, attributes=None, links=None, trace_state=None):
        return SamplingResult(Decision.RECORD_ONLY, attributes=attributes, trace_state=trace_state)

    def get_description(self):
        return "RecordOnlySampler"


class SpanCollector(SpanProcessor):
    def __init__(self):
        self.spans = []

    def on_end(self, span):
        self.spans.append(span)


@pytest.fixture(params=["drop", "noop", "record", "sample"])
def runtime(request):
    mode = request.param
    collector = SpanCollector()
    if mode == "noop":
        provider = trace.NoOpTracerProvider()
    else:
        sampler = {"drop": ALWAYS_OFF, "record": RecordOnlySampler(), "sample": ALWAYS_ON}[mode]
        provider = TracerProvider(sampler=sampler)
        provider.add_span_processor(collector)
    instrumentor = AtomicAgentsInstrumentor()
    instrumentor.instrument(tracer_provider=provider, config=TracingConfig(openinference_attributes=False))
    try:
        yield SimpleNamespace(provider=provider, spans=collector.spans, recording=mode in {"record", "sample"}, mode=mode)
    finally:
        instrumentor.uninstrument()
        if isinstance(provider, TracerProvider):
            provider.shutdown()


@pytest.fixture
def counted_payload():
    calls = []

    class Payload(BaseIOSchema):
        """A payload with an observable native Pydantic serializer."""

        text: str

        @model_serializer(mode="wrap")
        def serialize(self, handler):
            calls.append(self.text)
            return handler(self)

    return Payload, Payload(text="Business result"), calls


@pytest.mark.parametrize("asynchronous", [False, True])
def test_tool_payload_serialization_follows_span_recording(runtime, counted_payload, asynchronous):
    schema, payload, calls = counted_payload

    class Echo(BaseTool[schema, schema]):
        def run(self, params):
            return params

        async def run_async(self, params):
            return params

    tool = Echo(BaseToolConfig(title="echo", description="Return the original payload"))
    result = asyncio.run(tool.run_async(payload)) if asynchronous else tool.run(payload)
    assert result is payload
    assert calls == [payload.text] * (2 if runtime.recording else 0)
    assert len(runtime.spans) == int(runtime.recording)
    if runtime.recording:
        span = runtime.spans[0]
        assert json.loads(span.attributes["gen_ai.tool.call.arguments"]) == {"text": payload.text}
        assert json.loads(span.attributes["gen_ai.tool.call.result"]) == {"text": payload.text}
        assert bool(span.context.trace_flags.sampled) == (runtime.mode == "sample")


@pytest.mark.parametrize("asynchronous", [False, True])
def test_tool_errors_preserve_the_original_exception(runtime, counted_payload, asynchronous):
    schema, payload, calls = counted_payload
    error = ValueError("The operation failed")

    class Failing(BaseTool[schema, schema]):
        def run(self, params):
            raise error

        async def run_async(self, params):
            raise error

    tool = Failing(BaseToolConfig(title="failure", description="Fail with the original exception"))
    with pytest.raises(ValueError) as caught:
        if asynchronous:
            asyncio.run(tool.run_async(payload))
        else:
            tool.run(payload)
    assert caught.value is error
    assert calls == [payload.text] * int(runtime.recording)
    assert len(runtime.spans) == int(runtime.recording)
    if runtime.recording:
        assert runtime.spans[0].status.status_code == StatusCode.ERROR
        assert runtime.spans[0].events[0].attributes["exception.message"] == str(error)


@pytest.mark.parametrize("direction", ["input", "output"])
def test_chat_capture_does_not_inspect_unrecorded_messages(runtime, direction):
    reads = []

    class Message:
        @property
        def role(self):
            reads.append("role")
            return "user" if direction == "input" else "assistant"

        @property
        def content(self):
            reads.append("content")
            return "Conversation text"

    value = [Message()] if direction == "input" else SimpleNamespace(choices=[SimpleNamespace(message=Message())])
    with runtime.provider.get_tracer(__name__).start_as_current_span("chat") as span:
        write_messages(span, value, direction, TracingConfig(capture_messages=True))
    assert bool(reads) == runtime.recording
    if runtime.recording:
        message = json.loads(runtime.spans[0].attributes[f"gen_ai.{direction}.messages"])[0]
        assert message["parts"] == [{"type": "text", "content": "Conversation text"}]


def test_dropped_tool_keeps_valid_context_and_child_trace_identity(counted_payload):
    schema, payload, calls = counted_payload
    provider = TracerProvider(sampler=ALWAYS_OFF)
    instrumentor = AtomicAgentsInstrumentor()
    instrumentor.instrument(tracer_provider=provider, config=TracingConfig(openinference_attributes=False))
    seen = []

    class Nested(BaseTool[schema, schema]):
        def run(self, params):
            parent = trace.get_current_span().get_span_context()
            with provider.get_tracer(__name__).start_as_current_span("downstream") as child:
                seen.append((parent, child.get_span_context()))
            assert trace.get_current_span().get_span_context() == parent
            return params

    previous = trace.get_current_span()
    try:
        result = Nested(BaseToolConfig(title="nested", description="Create a child operation")).run(payload)
        assert result is payload
        assert calls == []
        parent, child = seen[0]
        assert parent.is_valid and child.is_valid
        assert parent.trace_id == child.trace_id
        assert parent.span_id != child.span_id
        assert trace.get_current_span() is previous
    finally:
        instrumentor.uninstrument()
        provider.shutdown()
