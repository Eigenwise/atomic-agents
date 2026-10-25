"""Real Instructor transport, validation and tracing across Ada's request modes."""

import asyncio

import instructor
import pytest
from instructor.core.exceptions import InstructorRetryException
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode

from instructor_transport import ScriptedResponse, async_instructor_client, instructor_client

from atomic_agents import AgentConfig, AtomicAgent, BaseIOSchema
from atomic_agents.observability import TracingConfig
from atomic_agents.observability import installation


class Question(BaseIOSchema):
    """A question sent through the real client."""

    question: str


class Answer(BaseIOSchema):
    """The validated model answer."""

    answer: str


@pytest.fixture(params=["TOOLS", "JSON", "JSON_SCHEMA", "RESPONSES_TOOLS"])
def mode(request):
    selected = getattr(instructor.Mode, request.param, None)
    if selected is None:
        pytest.skip(f"Installed Instructor has no Mode.{request.param} for from_openai")
    return selected


@pytest.fixture(params=[False, True], ids=["run", "run_async"])
def asynchronous(request):
    return request.param


@pytest.fixture
def exporter():
    exported = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exported))
    installation.install(tracer_provider=provider, config=TracingConfig(openinference_attributes=False))
    yield exported
    installation.uninstall()
    provider.shutdown()


def run_script(mode, asynchronous, responses):
    construct_client = async_instructor_client if asynchronous else instructor_client
    client = construct_client(mode, responses)
    configured = AtomicAgent[Question, Answer](AgentConfig(client=client, model="test-model"))
    configured.model_api_parameters["max_retries"] = 1
    question = Question(question="What is the answer?")
    if asynchronous:
        return asyncio.run(configured.run_async(question))
    return configured.run(question)


def assert_usage(span, input_tokens, output_tokens, cached_tokens, reasoning_tokens):
    assert span.attributes["gen_ai.usage.input_tokens"] == input_tokens
    assert span.attributes["gen_ai.usage.output_tokens"] == output_tokens
    assert span.attributes["gen_ai.usage.cache_read.input_tokens"] == cached_tokens
    assert span.attributes["gen_ai.usage.reasoning.output_tokens"] == reasoning_tokens


def assert_attempt_spans(exporter, responses):
    spans = exporter.get_finished_spans()
    parent = spans[-1]
    attempts = spans[:-1]
    assert parent.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert len(attempts) == len(responses)
    assert len(spans) == len(responses) + 1
    assert parent.attributes["atomic_agents.llm.attempts"] == len(responses)
    assert_models(parent)
    for number, (attempt, response) in enumerate(zip(attempts, responses), start=1):
        assert attempt.attributes["gen_ai.operation.name"] == "chat"
        assert attempt.name == "chat test-model"
        assert attempt.kind == SpanKind.CLIENT
        assert attempt.parent.span_id == parent.context.span_id
        assert attempt.attributes["atomic_agents.llm.attempt"] == number
        assert_models(attempt)
        assert_usage(attempt, response.input_tokens, response.output_tokens, response.cached_tokens, response.reasoning_tokens)
    return attempts, parent


def assert_models(span):
    assert span.attributes["gen_ai.request.model"] == "test-model"
    assert span.attributes["gen_ai.response.model"] == "response-model"
    assert span.attributes["gen_ai.provider.name"] == "openai"


def test_success_model_usage_and_attempt_count(mode, asynchronous, exporter):
    responses = [ScriptedResponse()]
    assert run_script(mode, asynchronous, responses).answer == "42"
    attempts, parent = assert_attempt_spans(exporter, responses)
    assert attempts[0].status.status_code == parent.status.status_code == StatusCode.OK
    assert_usage(parent, 3, 2, 1, 2)


def test_validation_retry_preserves_each_attempt_and_total_usage(mode, asynchronous, exporter):
    responses = [
        ScriptedResponse(arguments='{"answer": {"secret": "private output"}}'),
        ScriptedResponse(input_tokens=7, output_tokens=5, cached_tokens=3, reasoning_tokens=4),
    ]
    assert run_script(mode, asynchronous, responses).answer == "42"
    attempts, parent = assert_attempt_spans(exporter, responses)
    assert [attempt.status.status_code for attempt in attempts] == [StatusCode.ERROR, StatusCode.OK]
    assert parent.status.status_code == StatusCode.OK
    assert any(event.name == "validation_error" for event in attempts[0].events)
    assert any(event.name == "validation_error" for event in parent.events)
    assert_usage(parent, 10, 7, 4, 6)


def test_hidden_retry_exhaustion_redacts_all_span_failures(mode, asynchronous, exporter):
    installed = installation._current_installation()
    installation.install(
        tracer_provider=installed.tracer_provider, config=TracingConfig(hide_outputs=True, openinference_attributes=False)
    )
    responses = [ScriptedResponse(arguments='{"answer": {"secret": "private model output"}}')] * 2
    with pytest.raises(InstructorRetryException):
        run_script(mode, asynchronous, responses)
    attempts, parent = assert_attempt_spans(exporter, responses)
    assert [attempt.status.status_code for attempt in attempts] == [StatusCode.ERROR, StatusCode.ERROR]
    assert parent.status.status_code == StatusCode.ERROR
    assert "__REDACTED__" in parent.attributes["atomic_agents.llm.last_validation_error"]
    assert_usage(parent, 6, 4, 2, 4)
    for span in (*attempts, parent):
        assert "__REDACTED__" in span.status.description
        assert "private model output" not in str(span.attributes) + str(span.events) + span.status.description
        for event in span.events:
            if event.name == "exception":
                assert "__REDACTED__" in event.attributes["exception.message"]
                assert "__REDACTED__" in event.attributes["exception.stacktrace"]
