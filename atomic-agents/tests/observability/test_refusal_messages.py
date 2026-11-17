import asyncio
import json
from dataclasses import replace

import httpx
import instructor
import openai
import pytest
from instructor.core.exceptions import InstructorRetryException
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from atomic_agents import AgentConfig, AtomicAgent, BaseIOSchema
from atomic_agents.observability import TracingConfig, installation
from atomic_agents.observability.message_payloads import write_messages


class Question(BaseIOSchema):
    """A question used to exercise native refusal handling."""

    question: str


class Answer(BaseIOSchema):
    """A structured answer that a refusal cannot satisfy."""

    answer: str


@pytest.fixture
def tracing():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    installation.install(tracer_provider=provider, config=TracingConfig(capture_messages=True))
    try:
        yield provider, exporter
    finally:
        installation.uninstall()
        provider.shutdown()


def captured(tracing, messages, direction="input", **options):
    provider, exporter = tracing
    with provider.get_tracer(__name__).start_as_current_span("capture") as span:
        write_messages(span, messages, direction, replace(TracingConfig(capture_messages=True), **options))
    return dict(exporter.get_finished_spans()[-1].attributes)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("mode", [instructor.Mode.JSON, instructor.Mode.TOOLS])
@pytest.mark.parametrize("hidden", [False, True])
def test_native_refusals_are_visible_on_failed_attempts_only_when_enabled(tracing, asynchronous, mode, hidden):
    _, exporter = tracing
    installation.install(config=TracingConfig(capture_messages=True, hide_outputs=hidden))
    refusal = "I cannot provide the requested material."
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-refused",
                "object": "chat.completion",
                "created": 1,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": None, "refusal": refusal},
                    }
                ],
            },
        )

    def agent(client):
        return AtomicAgent[Question, Answer](
            AgentConfig(client=client, model="test-model", model_api_parameters={"max_retries": 1})
        )

    async def run_async():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http_client:
            client = instructor.from_openai(openai.AsyncOpenAI(api_key="test", http_client=http_client), mode=mode)
            await agent(client).run_async(Question(question="test"))

    with pytest.raises(InstructorRetryException):
        if asynchronous:
            asyncio.run(run_async())
        else:
            with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
                client = instructor.from_openai(openai.OpenAI(api_key="test", http_client=http_client), mode=mode)
                agent(client).run(Question(question="test"))

    attempts = [span for span in exporter.get_finished_spans() if span.kind is SpanKind.CLIENT]
    assert len(attempts) == len(requests) > 0
    assert exporter.get_finished_spans()[-1].status.status_code is StatusCode.ERROR
    for attempt in attempts:
        if hidden:
            assert "gen_ai.output.messages" not in attempt.attributes
        else:
            assert json.loads(attempt.attributes["gen_ai.output.messages"]) == [
                {"role": "assistant", "parts": [{"type": "text", "content": refusal, "refusal": True}]}
            ]


@pytest.mark.parametrize("representation", ["field", "blocks"])
def test_refusal_history_keeps_visible_text_and_original_part_order(tracing, representation):
    message = (
        {"role": "assistant", "content": "Before", "refusal": "Cannot comply"}
        if representation == "field"
        else {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Before"},
                {"type": "refusal", "refusal": "Cannot comply"},
                {"type": "text", "text": "After"},
            ],
        }
    )

    attributes = captured(tracing, [message])

    parts = json.loads(attributes["gen_ai.input.messages"])[0]["parts"]
    assert parts[:2] == [{"type": "text", "content": "Before"}, {"type": "text", "content": "Cannot comply", "refusal": True}]
    if representation == "blocks":
        assert parts[2] == {"type": "text", "content": "After"}
    assert captured(tracing, [message], hide_outputs=True) == {}
    assert captured(tracing, [message], hide_inputs=True) == {}


def test_refusal_messages_obey_the_complete_json_size_bound(tracing):
    response = {"choices": [{"message": {"role": "assistant", "content": None, "refusal": "拒绝" * 100}}]}
    attributes = captured(tracing, response, "output")
    length = len(attributes["gen_ai.output.messages"])

    assert captured(tracing, response, "output", max_attribute_length=length - 1) == {
        "gen_ai.output.messages.original_length": length
    }
    assert captured(tracing, response, "output", max_attribute_length=length) == attributes


@pytest.mark.parametrize("refusal", [None, "", 42, {"private": "not text"}])
def test_empty_or_nontext_refusals_do_not_replace_ordinary_content(tracing, refusal):
    attributes = captured(tracing, [{"role": "assistant", "content": "Normal response", "refusal": refusal}])

    assert json.loads(attributes["gen_ai.input.messages"]) == [
        {"role": "assistant", "parts": [{"type": "text", "content": "Normal response"}]}
    ]
