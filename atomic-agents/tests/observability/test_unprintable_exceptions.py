import asyncio
from unittest.mock import Mock

import instructor
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from opentelemetry.trace import StatusCode

from atomic_agents import AgentConfig, AtomicAgent, BaseIOSchema, BaseTool
from atomic_agents.observability import AtomicAgentsInstrumentor, TracingConfig


class Input(BaseIOSchema):
    """Input to the failing operation."""

    value: str


class Output(BaseIOSchema):
    """Output from a partial operation."""

    value: str


class BrokenMessage(Exception):
    def __str__(self):
        raise RuntimeError("message formatter failed")


class NonStringMessage(Exception):
    def __str__(self):
        return None


@pytest.fixture(params=[False, True], ids=["visible", "redacted"])
def tracing(request):
    exporter = InMemorySpanExporter()
    provider = TracerProvider(sampler=ALWAYS_ON)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = AtomicAgentsInstrumentor()
    instrumentor.instrument(tracer_provider=provider, config=TracingConfig(hide_outputs=request.param))
    try:
        yield exporter, request.param
    finally:
        instrumentor.uninstrument()
        provider.shutdown()


def failing_agent(error, asynchronous):
    client = Mock(spec=instructor.AsyncInstructor if asynchronous else instructor.Instructor)

    def create(**_kwargs):
        raise error

    async def create_async(**_kwargs):
        raise error

    def partial(**_kwargs):
        yield Output(value="partial")
        raise error

    async def partial_async(**_kwargs):
        yield Output(value="partial")
        raise error

    client.chat = Mock()
    client.chat.completions = Mock()
    client.chat.completions.create = create_async if asynchronous else create
    client.chat.completions.create_partial = partial_async if asynchronous else partial
    return AtomicAgent[Input, Output](AgentConfig(client=client, model="test-model"))


def invoke(api, error):
    params = Input(value="question")
    if api.startswith("tool"):

        class FailingTool(BaseTool[Input, Output]):
            def run(self, params: Input) -> Output:
                raise error

            async def run_async(self, params: Input) -> Output:
                raise error

        tool = FailingTool()
        return asyncio.run(tool.run_async(params)) if api == "tool_async" else tool.run(params)

    agent = failing_agent(error, asynchronous="async" in api)
    if api == "agent":
        return agent.run(params)
    if api == "agent_async":
        return asyncio.run(agent.run_async(params))
    if api == "stream":
        return list(agent.run_stream(params))

    async def consume():
        return [output async for output in agent.run_async_stream(params)]

    return asyncio.run(consume())


@pytest.mark.parametrize("api", ["tool", "tool_async", "agent", "agent_async", "stream", "stream_async"])
@pytest.mark.parametrize("error_type", [BrokenMessage, NonStringMessage])
def test_exception_formatting_cannot_replace_an_operation_failure(tracing, api, error_type):
    exporter, redacted = tracing
    original = error_type("private payload")

    with pytest.raises(error_type) as raised:
        invoke(api, original)

    assert raised.value is original
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["error.type"] == error_type.__qualname__
    (event,) = span.events
    assert event.name == "exception"
    assert event.attributes["exception.type"].endswith(error_type.__qualname__)
    assert event.attributes["exception.message"] == span.status.description
    assert "private payload" not in repr(event.attributes)
    if redacted:
        assert "__REDACTED__" in span.status.description
    else:
        assert "unprintable exception" in span.status.description


def test_ordinary_exceptions_keep_the_sdk_event_and_visibility_behavior(tracing):
    exporter, redacted = tracing
    original = ValueError("ordinary message")

    with pytest.raises(ValueError) as raised:
        invoke("tool", original)

    assert raised.value is original
    (span,) = exporter.get_finished_spans()
    (event,) = span.events
    assert event.attributes["exception.type"] == "ValueError"
    assert ("ordinary message" in event.attributes["exception.message"]) is not redacted
