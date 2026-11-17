import asyncio
import json
from types import SimpleNamespace

import httpx
import instructor
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from atomic_agents import AgentConfig, AtomicAgent, BaseIOSchema
from atomic_agents.observability import TracingConfig
from atomic_agents.observability import installation, llm_attempt_spans


class Question(BaseIOSchema):
    """A question for a local provider fixture."""

    question: str


class Answer(BaseIOSchema):
    """A validated provider answer."""

    answer: str


@pytest.fixture
def tracing():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    installation.install(tracer_provider=provider, config=TracingConfig(openinference_attributes=False))
    yield provider, exporter
    installation.uninstall()
    provider.shutdown()


def record_response(tracing, response, llm_spans=True):
    provider, exporter = tracing
    tracer = provider.get_tracer(__name__)
    config = TracingConfig(openinference_attributes=False, llm_spans=llm_spans)
    with tracer.start_as_current_span("agent") as span:
        run = llm_attempt_spans._AgentRun(span, tracer, config, {})
        with llm_attempt_spans.active_run(run):
            llm_attempt_spans._completion_kwargs(model="request-model")
            llm_attempt_spans._completion_response(response)
        llm_attempt_spans.finish_run(run)
    return exporter.get_finished_spans()


@pytest.mark.parametrize(
    "metadata, expected",
    [
        ({}, {}),
        ({"prompt_token_count": 30, "candidates_token_count": 5}, {"input_tokens": 30, "output_tokens": 5}),
        (
            {"prompt_token_count": 30, "cached_content_token_count": 12},
            {"input_tokens": 30, "cache_read.input_tokens": 12},
        ),
        (
            {"candidates_token_count": 5, "thoughts_token_count": 7},
            {"output_tokens": 12, "reasoning.output_tokens": 7},
        ),
        ({"prompt_token_count": 30, "tool_use_prompt_token_count": 11}, {"input_tokens": 41}),
        ({"thoughts_token_count": 7}, {"output_tokens": 7, "reasoning.output_tokens": 7}),
        ({"tool_use_prompt_token_count": 11}, {"input_tokens": 11}),
        (
            {"prompt_token_count": 0, "candidates_token_count": 0, "thoughts_token_count": 0, "cached_content_token_count": 0},
            {"input_tokens": 0, "output_tokens": 0, "reasoning.output_tokens": 0, "cache_read.input_tokens": 0},
        ),
    ],
)
def test_generate_content_usage_reaches_attempt_and_agent(tracing, metadata, expected):
    response = SimpleNamespace(model_version="returned-model-version", usage_metadata=SimpleNamespace(**metadata))
    child, parent = record_response(tracing, response)
    for span in (child, parent):
        actual = {key: value for key, value in span.attributes.items() if key.startswith("gen_ai.usage.")}
        assert actual == {"gen_ai.usage." + key: value for key, value in expected.items()}
        assert span.attributes["gen_ai.response.model"] == "returned-model-version"
    assert child.parent.span_id == parent.context.span_id


def test_missing_usage_metadata_does_not_invent_token_counts(tracing):
    child, parent = record_response(tracing, SimpleNamespace(usage_metadata=None, model_version=None))
    for span in (child, parent):
        assert not any(key.startswith("gen_ai.usage.") for key in span.attributes)
        assert "gen_ai.response.model" not in span.attributes


def test_existing_usage_and_model_remain_authoritative(tracing):
    response = SimpleNamespace(
        model="existing-model",
        model_version="other-model",
        usage=SimpleNamespace(prompt_tokens=30, completion_tokens=5),
        usage_metadata=SimpleNamespace(prompt_token_count=999, thoughts_token_count=999),
    )
    for span in record_response(tracing, response):
        assert span.attributes["gen_ai.response.model"] == "existing-model"
        assert span.attributes["gen_ai.usage.input_tokens"] == 30
        assert span.attributes["gen_ai.usage.output_tokens"] == 5
        assert "gen_ai.usage.reasoning.output_tokens" not in span.attributes


def test_agent_totals_remain_available_without_attempt_spans(tracing):
    (parent,) = record_response(
        tracing,
        SimpleNamespace(usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=3)),
        llm_spans=False,
    )
    assert parent.attributes["gen_ai.usage.input_tokens"] == 10
    assert parent.attributes["gen_ai.usage.output_tokens"] == 3
    assert parent.attributes["atomic_agents.llm.attempts"] == 1


@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("mode", [instructor.Mode.TOOLS, instructor.Mode.JSON])
@pytest.mark.parametrize("retry", [False, True])
def test_real_genai_sdk_usage_and_validation_retries(tracing, use_async, mode, retry):
    genai = pytest.importorskip("google.genai")
    pytest.importorskip("jsonref")  # Part of instructor[google-genai].
    requests = []
    reported = []

    def respond(request):
        assert request.url.path.endswith("/models/gemini-2.5-flash:generateContent")
        requests.append(json.loads(request.content))
        index = len(requests) - 1
        arguments = {"answer": {"invalid": "first response"}} if retry and index == 0 else {"answer": "42"}
        part = (
            {"functionCall": {"name": "Answer", "args": arguments}}
            if mode == instructor.Mode.TOOLS
            else {"text": json.dumps(arguments)}
        )
        metadata = {
            "promptTokenCount": 100 + 10 * index,
            "toolUsePromptTokenCount": 3 + index,
            "candidatesTokenCount": 5 + index,
            "thoughtsTokenCount": 7 + index,
            "cachedContentTokenCount": 40 + index,
            "totalTokenCount": 115 + 13 * index,
        }
        reported.append(
            {
                "input_tokens": 103 + 11 * index,
                "output_tokens": 12 + 2 * index,
                "cache_read.input_tokens": 40 + index,
                "reasoning.output_tokens": 7 + index,
            }
        )
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"role": "model", "parts": [part]}, "finishReason": "STOP", "index": 0}],
                "usageMetadata": metadata,
                "modelVersion": "gemini-returned-version",
            },
        )

    transport = httpx.MockTransport(respond)
    sdk = genai.Client(
        api_key="local-test-key",
        http_options=genai.types.HttpOptions(
            httpx_client=httpx.Client(transport=transport),
            httpx_async_client=httpx.AsyncClient(transport=transport),
        ),
    )
    client = instructor.from_genai(sdk, mode=mode, use_async=use_async)
    agent = AtomicAgent[Question, Answer](
        AgentConfig(client=client, model="gemini-2.5-flash", assistant_role="model", model_api_parameters={"max_retries": 2})
    )

    async def complete():
        try:
            return await agent.run_async(Question(question="Return 42"))
        finally:
            await sdk.aio.aclose()

    try:
        result = asyncio.run(complete()) if use_async else agent.run(Question(question="Return 42"))
    finally:
        sdk.close()
        if not use_async:
            asyncio.run(sdk.aio.aclose())

    assert result.answer == "42"
    assert len(requests) == (2 if retry else 1)
    assert requests[0]["contents"][-1]["role"] == "user"
    _, exporter = tracing
    spans = exporter.get_finished_spans()
    parent = spans[-1]
    children = spans[:-1]
    assert len(children) == len(requests)
    for child, expected in zip(children, reported):
        assert child.parent.span_id == parent.context.span_id
        assert child.attributes["gen_ai.response.model"] == "gemini-returned-version"
        assert {key: value for key, value in child.attributes.items() if key.startswith("gen_ai.usage.")} == {
            "gen_ai.usage." + key: value for key, value in expected.items()
        }
    assert parent.attributes["gen_ai.response.model"] == "gemini-returned-version"
    assert parent.attributes["atomic_agents.llm.attempts"] == len(requests)
    for key in reported[0]:
        assert parent.attributes["gen_ai.usage." + key] == sum(item[key] for item in reported)
