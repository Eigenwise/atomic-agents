"""Input visibility also covers failures that repeat tool arguments."""

import json

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import BaseModel, ValidationError

from atomic_agents import BaseIOSchema, BaseTool
from atomic_agents.observability import AtomicAgentsInstrumentor, TracingConfig


class Request(BaseIOSchema):
    """Private input to a tool."""

    text: str


class Response(BaseIOSchema):
    """Public result from a tool."""

    text: str


class Number(BaseModel):
    value: int


@pytest.fixture
def tracing():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = AtomicAgentsInstrumentor()
    yield instrumentor, provider, exporter
    instrumentor.uninstrument()
    provider.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("validation", [False, True])
@pytest.mark.parametrize("hide_inputs,hide_outputs", [(False, False), (True, False), (False, True), (True, True)])
async def test_failure_visibility_preserves_exception_and_hides_repeated_input(
    tracing, asynchronous, validation, hide_inputs, hide_outputs
):
    instrumentor, provider, exporter = tracing
    instrumentor.instrument(
        tracer_provider=provider,
        config=TracingConfig(hide_inputs=hide_inputs, hide_outputs=hide_outputs, openinference_attributes=False),
    )
    private = "private-tool-input-marker"
    failure = ValueError(private)
    if validation:
        try:
            Number.model_validate({"value": private})
        except ValidationError as error:
            failure = error

    class FailingTool(BaseTool[Request, Response]):
        def run(self, params: Request) -> Response:
            raise failure

        async def run_async(self, params: Request) -> Response:
            raise failure

    tool = FailingTool()
    with pytest.raises(type(failure)) as raised:
        if asynchronous:
            await tool.run_async(Request(text=private))
        else:
            tool.run(Request(text=private))

    assert raised.value is failure
    (span,) = exporter.get_finished_spans()
    assert span.status.status_code is StatusCode.ERROR
    assert span.attributes["error.type"] == type(failure).__qualname__
    (event,) = span.events
    assert event.attributes["exception.message"] == span.status.description
    failure_text = json.dumps({"event": dict(event.attributes), "status": span.status.description})
    if hide_inputs or hide_outputs:
        assert private not in failure_text
        assert "__REDACTED__" in event.attributes["exception.message"]
    else:
        assert private in event.attributes["exception.message"]


def test_input_hiding_keeps_successful_outputs_visible(tracing):
    instrumentor, provider, exporter = tracing
    instrumentor.instrument(tracer_provider=provider, config=TracingConfig(hide_inputs=True, openinference_attributes=False))

    class SuccessfulTool(BaseTool[Request, Response]):
        def run(self, params: Request) -> Response:
            return Response(text="public result")

    SuccessfulTool().run(Request(text="private argument"))
    (span,) = exporter.get_finished_spans()
    assert span.attributes["gen_ai.tool.call.arguments"] == "__REDACTED__"
    assert json.loads(span.attributes["gen_ai.tool.call.result"]) == {"text": "public result"}
