"""Keep rejected model arguments available in opt-in chat transcripts."""

import asyncio
import json

import instructor
import pytest
from openai.types.chat import ChatCompletion
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from instructor_transport import ScriptedResponse, async_instructor_client, instructor_client

from atomic_agents import AgentConfig, AtomicAgent, BaseIOSchema
from atomic_agents.observability import TracingConfig, installation
from atomic_agents.observability.message_payloads import write_messages


def transcript(arguments, direction):
    messages = [
        {"role": "assistant", "content": "before"},
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [
                {"type": "function", "id": "rejected", "function": {"name": "lookup", "arguments": arguments}},
                {"type": "function", "id": "valid", "function": {"name": "lookup", "arguments": '{"city":"東京"}'}},
            ],
        },
        {"role": "assistant", "content": "after"},
    ]
    if direction == "input":
        return messages
    return ChatCompletion.model_validate(
        {
            "id": "test",
            "created": 1,
            "model": "test-model",
            "object": "chat.completion",
            "choices": [{"index": i, "finish_reason": "tool_calls", "message": message} for i, message in enumerate(messages)],
        }
    )


def expected_messages(arguments):
    return [
        {"role": "assistant", "parts": [{"type": "text", "content": "before"}]},
        {
            "role": "assistant",
            "parts": [
                {"type": "text", "content": "checking"},
                {"type": "tool_call", "name": "lookup", "arguments": arguments, "id": "rejected"},
                {"type": "tool_call", "name": "lookup", "arguments": {"city": "東京"}, "id": "valid"},
            ],
        },
        {"role": "assistant", "parts": [{"type": "text", "content": "after"}]},
    ]


def capture(value, direction, **options):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    try:
        with provider.get_tracer(__name__).start_as_current_span("chat") as span:
            write_messages(span, value, direction, TracingConfig(capture_messages=True, **options))
        return dict(exporter.get_finished_spans()[0].attributes)
    finally:
        provider.shutdown()


@pytest.mark.parametrize("direction", ["input", "output"])
@pytest.mark.parametrize(
    "arguments",
    [
        '{"city":"Ghent"',
        "",
        '{"city":"Ghent"} trailing',
        "{'city': 'Ghent'}",
        r'{"path":"C:\Users"}',
        '{"temperature":NaN}',
        '{"values":[Infinity,-Infinity]}',
        '{"temperature":1e400}',
    ],
)
def test_unparsed_arguments_remain_raw_without_losing_other_parts(direction, arguments):
    result = capture(transcript(arguments, direction), direction)
    assert json.loads(result[f"gen_ai.{direction}.messages"]) == expected_messages(arguments)


@pytest.mark.parametrize("direction", ["input", "output"])
@pytest.mark.parametrize(
    "arguments,expected",
    [
        (' {"city":"Ghent","temperature":-1.25e2} ', {"city": "Ghent", "temperature": -125.0}),
        ('[1,true,null,"NaN"]', [1, True, None, "NaN"]),
        ('"plain string"', "plain string"),
        ("null", None),
        ("42", 42),
        ("false", False),
    ],
)
def test_valid_json_arguments_still_use_structured_values(direction, arguments, expected):
    result = capture(transcript(arguments, direction), direction)
    assert json.loads(result[f"gen_ai.{direction}.messages"]) == expected_messages(expected)


@pytest.mark.parametrize("direction", ["input", "output"])
@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_raw_arguments_obey_complete_serialized_attribute_limit(direction, delta):
    arguments = '{"city":"東京",\n"quote":"unfinished'
    expected = json.dumps(expected_messages(arguments), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    result = capture(transcript(arguments, direction), direction, max_attribute_length=len(expected) + delta)
    assert result == ({} if delta < 0 else {f"gen_ai.{direction}.messages": expected})


class Question(BaseIOSchema):
    """Question for the real Instructor request path."""

    question: str


class Answer(BaseIOSchema):
    """Validated answer after the rejected tool call."""

    answer: str


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("hidden", [False, True], ids=["visible", "hidden"])
def test_real_instructor_retry_captures_rejected_output_and_replayed_input(asynchronous, hidden):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    installation.install(
        tracer_provider=provider,
        config=TracingConfig(capture_messages=True, hide_outputs=hidden, max_attribute_length=20000),
    )
    rejected = '{"answer": "unfinished'
    construct = async_instructor_client if asynchronous else instructor_client
    client = construct(instructor.Mode.TOOLS, [ScriptedResponse(arguments=rejected), ScriptedResponse()])
    agent = AtomicAgent[Question, Answer](AgentConfig(client=client, model="test-model"))
    question = Question(question="What is the answer?")

    async def run_async():
        try:
            return await agent.run_async(question)
        finally:
            await client.client.close()

    try:
        if asynchronous:
            answer = asyncio.run(run_async())
        else:
            try:
                answer = agent.run(question)
            finally:
                client.client.close()
        assert answer.answer == "42"
        first, second, parent = exporter.get_finished_spans()
        assert [span.status.status_code for span in (first, second, parent)] == [
            StatusCode.ERROR,
            StatusCode.OK,
            StatusCode.OK,
        ]
        assert first.parent.span_id == second.parent.span_id == parent.context.span_id
        assert parent.attributes["atomic_agents.llm.attempts"] == 2
        assert parent.attributes["gen_ai.usage.input_tokens"] == 6
        if hidden:
            for span in (first, second, parent):
                assert "gen_ai.input.messages" not in span.attributes
                assert "gen_ai.output.messages" not in span.attributes
            return
        failed_output = json.loads(first.attributes["gen_ai.output.messages"])
        assert failed_output[0]["parts"] == [{"type": "tool_call", "id": "call_test", "name": "Answer", "arguments": rejected}]
        retried_input = json.loads(second.attributes["gen_ai.input.messages"])
        assert retried_input[0]["role"] == "system"
        assert any(part == failed_output[0]["parts"][0] for message in retried_input for part in message["parts"])
        assert json.loads(second.attributes["gen_ai.output.messages"])[0]["parts"][0]["arguments"] == {"answer": "42"}
    finally:
        installation.uninstall()
        provider.shutdown()
