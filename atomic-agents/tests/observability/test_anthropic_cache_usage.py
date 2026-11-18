"""Anthropic reports cache reads and writes separately from uncached inputs."""

from types import SimpleNamespace

import httpx
import instructor
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from atomic_agents import AgentConfig, AtomicAgent, BaseIOSchema
from atomic_agents.observability import AtomicAgentsInstrumentor, TracingConfig
from atomic_agents.observability.llm_attempt_spans import _usage_attributes


@pytest.mark.parametrize(
    "cache_read,cache_write,total,cache_attributes",
    [
        (100, 50, 160, {"cache_read.input_tokens": 100, "cache_write.input_tokens": 50}),
        (100, None, 110, {"cache_read.input_tokens": 100}),
        (None, 50, 60, {"cache_write.input_tokens": 50}),
        (0, 0, 10, {"cache_read.input_tokens": 0, "cache_write.input_tokens": 0}),
        (None, None, 10, {}),
    ],
)
def test_anthropic_usage_includes_separately_reported_cache_tokens(cache_read, cache_write, total, cache_attributes):
    usage = SimpleNamespace(
        input_tokens=10,
        output_tokens=4,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
    )
    expected = {"input_tokens": total, "output_tokens": 4, **cache_attributes}
    assert _usage_attributes(usage) == {"gen_ai.usage." + key: value for key, value in expected.items()}


@pytest.mark.parametrize("responses_api", [False, True])
def test_openai_input_total_already_includes_cached_tokens(responses_api):
    details = SimpleNamespace(cached_tokens=100)
    if responses_api:
        usage = SimpleNamespace(input_tokens=110, output_tokens=4, input_tokens_details=details)
    else:
        usage = SimpleNamespace(prompt_tokens=110, completion_tokens=4, prompt_tokens_details=details)
    assert _usage_attributes(usage) == {
        "gen_ai.usage.input_tokens": 110,
        "gen_ai.usage.output_tokens": 4,
        "gen_ai.usage.cache_read.input_tokens": 100,
    }


class Question(BaseIOSchema):
    """A question sent to the model."""

    question: str


class Answer(BaseIOSchema):
    """The validated answer."""

    answer: str


@pytest.fixture
def exporter():
    exported = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exported))
    instrumentor = AtomicAgentsInstrumentor()
    instrumentor.instrument(tracer_provider=provider, config=TracingConfig(openinference_attributes=False))
    yield exported
    instrumentor.uninstrument()
    provider.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("retry", [False, True])
async def test_real_anthropic_transport_records_per_attempt_and_total_cache_usage(exporter, asynchronous, retry):
    anthropic = pytest.importorskip("anthropic")
    counts = [(10, 100, 50, 4), (20, 200, 0, 5)] if retry else [(10, 100, 50, 4)]
    requests = []

    def respond(request):
        assert request.url.path == "/v1/messages"
        number = len(requests)
        requests.append(request)
        uncached, read, write, output = counts[number]
        answer = {"invalid": "retry"} if retry and number == 0 else "42"
        return httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "model": "response-model",
                "content": [{"type": "tool_use", "id": "call_test", "name": "Answer", "input": {"answer": answer}}],
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {
                    "input_tokens": uncached,
                    "output_tokens": output,
                    "cache_read_input_tokens": read,
                    "cache_creation_input_tokens": write,
                },
            },
        )

    if asynchronous:
        sdk = anthropic.AsyncAnthropic(api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    else:
        sdk = anthropic.Anthropic(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    client = instructor.from_anthropic(sdk, mode=instructor.Mode.ANTHROPIC_TOOLS)
    agent = AtomicAgent[Question, Answer](
        AgentConfig(client=client, model="test-model", model_api_parameters={"max_tokens": 64, "max_retries": 2})
    )
    try:
        question = Question(question="What is the answer?")
        answer = await agent.run_async(question) if asynchronous else agent.run(question)
        assert answer.answer == "42"
    finally:
        if asynchronous:
            await sdk.close()
        else:
            sdk.close()

    *attempts, parent = exporter.get_finished_spans()
    assert len(requests) == len(attempts) == len(counts)
    expected_totals = [0, 0, 0, 0]
    keys = ["input_tokens", "cache_read.input_tokens", "cache_write.input_tokens", "output_tokens"]
    for attempt, (uncached, read, write, output) in zip(attempts, counts):
        expected = [uncached + read + write, read, write, output]
        assert attempt.parent.span_id == parent.context.span_id
        for index, (key, value) in enumerate(zip(keys, expected)):
            assert attempt.attributes["gen_ai.usage." + key] == value
            expected_totals[index] += value
    for key, total in zip(keys, expected_totals):
        assert parent.attributes["gen_ai.usage." + key] == total
    assert parent.attributes["atomic_agents.llm.attempts"] == len(counts)
