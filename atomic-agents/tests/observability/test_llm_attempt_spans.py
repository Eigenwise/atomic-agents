import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from types import SimpleNamespace
from unittest.mock import Mock

import instructor
import pytest
from instructor.core.hooks import Hooks
from opentelemetry import context as context_api, trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind, StatusCode
from pydantic import ValidationError
from instructor.core.exceptions import InstructorRetryException

from instructor_transport import ScriptedResponse, instructor_client

from atomic_agents import AgentConfig, AtomicAgent, BaseIOSchema
from atomic_agents.observability import TracingConfig, using_conversation
from atomic_agents.observability import agent_spans, installation, llm_attempt_spans


class Question(BaseIOSchema):
    """A test question."""

    question: str


class Answer(BaseIOSchema):
    """A test answer."""

    answer: str


@pytest.fixture
def exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    installation.install(tracer_provider=provider, config=TracingConfig(openinference_attributes=False))
    yield exporter
    installation.uninstall()


def usage(input_tokens=3, output_tokens=2):
    return ScriptedResponse(input_tokens=input_tokens, output_tokens=output_tokens).chat_usage()


def finished_spans(exporter, count):
    spans = exporter.get_finished_spans()
    assert len(spans) == count
    return spans


def response():
    return SimpleNamespace(model="response-model", usage=usage())


def scripted_client(script):
    client = Mock(spec=instructor.Instructor)
    client.provider = instructor.Provider.OPENAI
    client.hooks = Hooks()
    client.on.side_effect = client.hooks.on
    client.off.side_effect = client.hooks.off
    client.chat = Mock()
    client.chat.completions = Mock()

    def create(**kwargs):
        script(client.hooks, kwargs)
        return Answer(answer="42")

    client.chat.completions.create.side_effect = create
    return client


def agent(client):
    return AtomicAgent[Question, Answer](AgentConfig(client=client, model="test-model"))


def retry_script(hooks, kwargs):
    hooks.emit_completion_arguments(**kwargs)
    hooks.emit_completion_response(response())
    hooks.emit_parse_error(ValueError("invalid output"))
    hooks.emit_completion_arguments(**kwargs)
    hooks.emit_completion_response(response())


def run_agent(client):
    return agent(client).run(Question(question="test"))


def test_retry_spans_usage_validation_and_parentage(exporter):
    client = instructor_client(
        instructor.Mode.TOOLS, [ScriptedResponse(arguments='{"answer": {"secret": "private output"}}'), ScriptedResponse()]
    )
    with using_conversation("conversation"):
        run_agent(client)
    first, second, parent = finished_spans(exporter, 3)
    assert [first.name, second.name] == ["chat test-model", "chat test-model"]
    assert [first.attributes["atomic_agents.llm.attempt"], second.attributes["atomic_agents.llm.attempt"]] == [1, 2]
    assert first.parent.span_id == second.parent.span_id == parent.context.span_id
    assert first.status.status_code == StatusCode.ERROR
    assert second.status.status_code == StatusCode.OK
    assert first.kind == second.kind == SpanKind.CLIENT
    assert first.events[0].name == "validation_error"
    assert first.attributes["error.type"] == "ValidationError"
    assert first.attributes["gen_ai.provider.name"] == "openai"
    assert first.attributes["gen_ai.conversation.id"] == "conversation"
    assert first.attributes["gen_ai.usage.input_tokens"] == second.attributes["gen_ai.usage.input_tokens"] == 3
    assert first.attributes["gen_ai.usage.output_tokens"] == 2
    assert first.attributes["gen_ai.usage.cache_read.input_tokens"] == 1
    assert first.attributes["gen_ai.usage.reasoning.output_tokens"] == 2
    assert parent.attributes["gen_ai.usage.input_tokens"] == 6
    assert parent.attributes["gen_ai.usage.output_tokens"] == 4
    assert parent.attributes.get("atomic_agents.llm.attempts") == 2
    assert "private output" in parent.attributes["atomic_agents.llm.last_validation_error"]
    assert parent.attributes["gen_ai.response.model"] == first.attributes["gen_ai.response.model"] == "response-model"
    event = parent.events[0]
    assert event.name == "validation_error"
    assert event.attributes["atomic_agents.llm.attempt"] == 1


def test_disabled_attempt_spans_keep_totals(exporter):
    installed = installation._current_installation()
    installation.install(tracer_provider=installed.tracer_provider, config=TracingConfig(llm_spans=False))
    run_agent(scripted_client(retry_script))
    (parent,) = finished_spans(exporter, 1)
    assert parent.attributes.get("atomic_agents.llm.attempts") == 2
    assert parent.attributes["gen_ai.usage.input_tokens"] == 6


def test_completion_error_is_bounded_and_ends_attempt(exporter):
    installed = installation._current_installation()
    installation.install(tracer_provider=installed.tracer_provider, config=TracingConfig(max_attribute_length=32))
    error = ValueError("failed " * 20)
    expected = str(error)[:18] + "...[truncated]"

    def script(hooks, kwargs):
        hooks.emit_completion_arguments(**kwargs)
        hooks.emit_parse_error(error)
        hooks.emit_completion_error(error, attempt_number=1, max_attempts=1, is_last_attempt=True)

    run_agent(scripted_client(script))
    child, parent = finished_spans(exporter, 2)
    assert child.status.status_code == StatusCode.ERROR
    assert child.attributes["error.type"] == "ValueError"
    assert child.events[-1].attributes["exception.message"] == expected
    assert len(child.events[-1].attributes["exception.stacktrace"]) <= 32
    assert child.status.description == expected
    assert len(child.status.description) == 32
    assert parent.attributes["atomic_agents.llm.last_validation_error"] == expected
    assert parent.attributes["atomic_agents.llm.last_validation_error.original_length"] == len(str(error))
    assert parent.status.status_code == StatusCode.OK


def test_registration_once_and_uninstall_preserves_application_hooks(exporter):
    client = scripted_client(retry_script)
    application_hook = Mock()
    client.hooks.on("completion:response", application_hook)
    for _ in range(3):
        run_agent(client)
    assert client.on.call_count == 4
    assert application_hook.call_count == 6
    installation.uninstall()
    assert client.off.call_count == 4
    client.hooks.emit_completion_response(response())
    assert application_hook.call_count == 7
    assert not llm_attempt_spans._clients


@pytest.mark.parametrize("asynchronous", [False, True])
def test_concurrent_shared_client_runs_are_isolated(exporter, asynchronous):
    client = scripted_client(retry_script)

    async def invoke(number):
        async def complete(agent, *args, **kwargs):
            client.hooks.emit_completion_arguments(model="test-model")
            await asyncio.sleep(0)
            client.hooks.emit_completion_response(SimpleNamespace(model="response-model", usage=usage(number)))
            return Answer(answer="ok")

        if asynchronous:
            return await agent_spans._trace_agent_async_call(complete)(agent(client))
        return await asyncio.to_thread(run_agent, client)

    async def gather():
        await asyncio.gather(invoke(10), invoke(20))

    asyncio.run(gather())
    parents = [span for span in exporter.get_finished_spans() if span.name.startswith("invoke_agent")]
    expected = [10, 20] if asynchronous else [6, 6]
    assert sorted(span.attributes.get("gen_ai.usage.input_tokens", -1) for span in parents) == expected
    assert all(span.attributes["atomic_agents.llm.attempts"] == (1 if asynchronous else 2) for span in parents)


@pytest.mark.parametrize("kind", ["sync", "async", "stream", "async_stream"])
@pytest.mark.parametrize("fails", [False, True])
def test_open_attempt_ends_at_agent_end(exporter, kind, fails):
    client = scripted_client(retry_script)

    def start():
        client.hooks.emit_completion_arguments(model="test-model")
        if fails:
            raise ValueError("agent failed")
        return Answer(answer="partial")

    def call(*args, **kwargs):
        return start()

    async def async_call(*args, **kwargs):
        return start()

    def stream(*args, **kwargs):
        yield start()
        yield Answer(answer="final")

    async def async_stream(*args, **kwargs):
        yield start()
        yield Answer(answer="final")

    def invoke():
        if kind == "sync":
            return agent_spans._trace_agent_call(call)(agent(client))
        if kind == "async":
            return asyncio.run(agent_spans._trace_agent_async_call(async_call)(agent(client)))
        if kind == "stream":
            iterator = agent_spans._trace_agent_stream(stream)(agent(client))
            next(iterator)
            assert trace_api.get_current_span() is trace_api.INVALID_SPAN
            iterator.close()
            return None
        return asyncio.run(consume_async_stream())

    async def consume_async_stream():
        iterator = agent_spans._trace_agent_async_stream(async_stream)(agent(client))
        await anext(iterator)
        assert trace_api.get_current_span() is trace_api.INVALID_SPAN
        await iterator.aclose()

    if fails:
        with pytest.raises(ValueError, match="agent failed"):
            invoke()
    else:
        invoke()
    child, parent = finished_spans(exporter, 2)
    assert child.status.status_code == StatusCode.UNSET
    assert child.parent.span_id == parent.context.span_id
    assert child.end_time <= parent.end_time


def test_validation_text_redacted_and_anthropic_usage(exporter):
    installed = installation._current_installation()
    installation.install(tracer_provider=installed.tracer_provider, config=TracingConfig(hide_outputs=True))

    def script(hooks, kwargs):
        hooks.emit_completion_arguments(**kwargs)
        with pytest.raises(ValidationError) as invalid:
            Answer.model_validate({"answer": {"secret": "private output"}})
        hooks.emit_completion_response(SimpleNamespace(model="response-model", usage=usage(9, 4)))
        hooks.emit_parse_error(invalid.value)
        hooks.emit_completion_error(invalid.value, attempt_number=1, max_attempts=1, is_last_attempt=True)

    run_agent(scripted_client(script))
    child, parent = finished_spans(exporter, 2)
    assert parent.attributes["atomic_agents.llm.last_validation_error"] == "__REDACTED__"
    assert parent.events[0].attributes["atomic_agents.llm.last_validation_error"] == "__REDACTED__"
    assert parent.attributes["gen_ai.usage.input_tokens"] == 9
    assert child.status.description == "ValidationError: 1 errors (__REDACTED__)"
    assert child.events[-1].attributes["exception.message"] == child.status.description
    assert "ValidationError: 1 errors (__REDACTED__)" in child.events[-1].attributes["exception.stacktrace"]
    assert "private output" not in str(child.events)


def test_hook_failure_warns_without_breaking_agent(exporter, caplog):
    class BrokenResponse:
        @property
        def model(self):
            raise RuntimeError("broken metadata")

    def script(hooks, kwargs):
        hooks.emit_completion_arguments(**kwargs)
        hooks.emit_completion_response(BrokenResponse())

    assert run_agent(scripted_client(script)).answer == "42"
    child, parent = finished_spans(exporter, 2)
    assert child.status.status_code == StatusCode.UNSET
    assert "Instructor tracing hook failed" in caplog.text


@pytest.mark.parametrize("asynchronous", [False, True])
def test_stream_attempt_context_survives_cursor_advances(exporter, asynchronous):
    client = scripted_client(retry_script)

    def begin():
        client.hooks.emit_completion_arguments(model="test-model")
        return trace_api.get_current_span().get_span_context().span_id

    def complete(expected):
        assert trace_api.get_current_span().get_span_context().span_id == expected
        client.hooks.emit_completion_response(response())
        return Answer(answer="final")

    def stream(*args, **kwargs):
        expected = begin()
        yield Answer(answer="partial")
        yield complete(expected)

    async def async_stream(*args, **kwargs):
        expected = begin()
        yield Answer(answer="partial")
        await asyncio.sleep(0)
        yield complete(expected)

    async def consume():
        return [partial async for partial in agent_spans._trace_agent_async_stream(async_stream)(agent(client))]

    if asynchronous:
        assert len(asyncio.run(consume())) == 2
    else:
        assert len(list(agent_spans._trace_agent_stream(stream)(agent(client)))) == 2
    child, parent = finished_spans(exporter, 2)
    assert child.status.status_code == StatusCode.OK
    assert parent.status.status_code == StatusCode.OK


def test_validation_text_truncation_and_extra_hook_keywords(exporter):
    installed = installation._current_installation()
    installation.install(tracer_provider=installed.tracer_provider, config=TracingConfig(max_attribute_length=4))

    def script(hooks, kwargs):
        hooks.emit_completion_arguments(**kwargs)
        llm_attempt_spans._parse_error(ValueError("long error"), future_option=True)
        llm_attempt_spans._completion_response(response(), future_option=True)

    run_agent(scripted_client(script))
    child, parent = finished_spans(exporter, 2)
    assert parent.attributes["atomic_agents.llm.last_validation_error"] == "...["
    assert parent.attributes["atomic_agents.llm.last_validation_error.original_length"] == 10


def test_threaded_first_use_registers_once_and_uninstalls(exporter, monkeypatch):
    client = scripted_client(retry_script)
    first_registration = threading.Barrier(2, timeout=5)
    release_registration = threading.Event()
    original_on = Hooks.on

    def blocked_on(hooks, event, handler):
        if event == "completion:kwargs":
            first_registration.wait()
            assert release_registration.wait(timeout=5)
        return original_on(hooks, event, handler)

    monkeypatch.setattr(Hooks, "on", blocked_on)
    client.on.side_effect = client.hooks.on
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(run_agent, client)
        first_registration.wait()
        second_started = threading.Event()

        def second_run():
            second_started.set()
            return run_agent(client)

        second = executor.submit(second_run)
        assert second_started.wait(timeout=5)
        try:
            with pytest.raises(TimeoutError):
                second.result(timeout=0.1)
            assert client.on.call_count == 1
        finally:
            release_registration.set()
        first.result(timeout=5)
        second.result(timeout=5)
    assert len(client.hooks._handlers) == 4
    assert all(len(handlers) == 1 for handlers in client.hooks._handlers.values())
    installation.uninstall()
    assert all(not handlers for handlers in client.hooks._handlers.values())


def test_partial_registration_is_rolled_back(exporter):
    client = scripted_client(retry_script)

    def fail_registration(event, handler):
        if event == "parse:error":
            raise RuntimeError("registration failed")
        client.hooks.on(event, handler)

    client.on.side_effect = fail_registration
    llm_attempt_spans.register_client(client)
    assert client not in llm_attempt_spans._clients
    assert all(not handlers for handlers in client.hooks._handlers.values())
    client.on.side_effect = client.hooks.on
    llm_attempt_spans.register_client(client)
    assert all(len(handlers) == 1 for handlers in client.hooks._handlers.values())


@pytest.mark.parametrize("completion", ["response", "error"])
def test_suppressed_attempt_events_are_ignored_then_normal_attempt_works(exporter, completion):
    def script(hooks, kwargs):
        run = llm_attempt_spans._active_run.get()
        token = context_api.attach(context_api.set_value("suppress_instrumentation", True))
        try:
            hooks.emit_completion_arguments(**kwargs)
        finally:
            context_api.detach(token)
        if completion == "response":
            hooks.emit_completion_response(response())
        else:
            hooks.emit_completion_error(ValueError("suppressed"), attempt_number=1, max_attempts=2, is_last_attempt=False)
        hooks.emit_parse_error(ValueError("suppressed"))
        assert run.attempt_count == 0
        assert run.open_attempt is None
        assert not exporter.get_finished_spans()
        hooks.emit_completion_arguments(**kwargs)
        hooks.emit_completion_response(response())

    run_agent(scripted_client(script))
    child, parent = finished_spans(exporter, 2)
    assert child.attributes["atomic_agents.llm.attempt"] == 1
    assert parent.attributes["atomic_agents.llm.attempts"] == 1
    assert parent.attributes["gen_ai.usage.input_tokens"] == 3
    assert not parent.events


@pytest.mark.parametrize(
    "mode", [instructor.Mode.TOOLS, instructor.Mode.JSON, instructor.Mode.JSON_SCHEMA, instructor.Mode.RESPONSES_TOOLS]
)
def test_real_transport_usage_is_accumulated_per_response(exporter, mode):
    client = instructor_client(
        mode,
        [
            ScriptedResponse(arguments='{"answer": {"private": "model text"}}'),
            ScriptedResponse(input_tokens=7, output_tokens=5, cached_tokens=3, reasoning_tokens=4),
        ],
    )
    assert run_agent(client).answer == "42"
    first, second, parent = finished_spans(exporter, 3)
    assert first.status.status_code == StatusCode.ERROR
    assert first.events[0].name == "validation_error"
    assert second.status.status_code == StatusCode.OK
    assert first.attributes["gen_ai.usage.input_tokens"] == 3
    assert second.attributes["gen_ai.usage.input_tokens"] == 7
    assert parent.attributes["gen_ai.usage.input_tokens"] == 10
    assert parent.attributes["gen_ai.usage.output_tokens"] == 7
    assert parent.attributes["gen_ai.usage.cache_read.input_tokens"] == 4
    assert parent.attributes["gen_ai.usage.reasoning.output_tokens"] == 6
    assert parent.attributes["atomic_agents.llm.attempts"] == 2
    assert "completion:usage" not in llm_attempt_spans._HANDLERS


@pytest.mark.parametrize("hidden", [False, True])
def test_real_retry_exhaustion_redacts_or_bounds_every_exception(exporter, hidden):
    installed = installation._current_installation()
    installation.install(
        tracer_provider=installed.tracer_provider, config=TracingConfig(hide_outputs=hidden, max_attribute_length=256)
    )
    secret = "private model text " * 100
    client = instructor_client(
        instructor.Mode.TOOLS, [ScriptedResponse(arguments='{"answer": {"secret": "' + secret + '"}}')] * 3
    )
    configured = agent(client)
    configured.model_api_parameters["max_retries"] = 2
    with pytest.raises(InstructorRetryException):
        configured.run(Question(question="test"))
    first, second, third, parent = finished_spans(exporter, 4)
    assert parent.status.status_code == StatusCode.ERROR
    assert parent.attributes["atomic_agents.llm.attempts"] == 3
    assert parent.attributes["gen_ai.usage.input_tokens"] == 9
    message = parent.events[-1].attributes["exception.message"]
    assert len(message) <= 256
    assert len(parent.status.description) <= 256
    assert len(parent.events[-1].attributes["exception.stacktrace"]) <= 256
    if hidden:
        assert message == parent.status.description == "InstructorRetryException: 3 attempts (__REDACTED__)"
        assert "private model text" not in str(parent.events)
        assert "private model text" not in str(first.events) + str(second.events) + str(third.events)
        assert (
            first.events[0].attributes["atomic_agents.llm.last_validation_error"] == "ValidationError: 1 errors (__REDACTED__)"
        )
    else:
        assert message.endswith("...[truncated]")


def test_direct_instructor_calls_inside_run_add_to_agent_totals(exporter):
    client = instructor_client(instructor.Mode.TOOLS, [ScriptedResponse(), ScriptedResponse(input_tokens=11)])

    def call_twice(configured):
        for _ in range(2):
            client.chat.completions.create(
                model="test-model", response_model=Answer, messages=[{"role": "user", "content": "test"}]
            )
        return Answer(answer="42")

    agent_spans._trace_agent_call(call_twice)(agent(client))
    first, second, parent = finished_spans(exporter, 3)
    assert parent.attributes["gen_ai.usage.input_tokens"] == 14
    assert parent.attributes["atomic_agents.llm.attempts"] == 2
    assert first.parent.span_id == second.parent.span_id == parent.context.span_id


def test_missing_model_and_final_completion_error_still_count(exporter):
    def script(hooks, kwargs):
        hooks.emit_completion_arguments()
        hooks.emit_completion_response(response())
        hooks.emit_completion_arguments()
        hooks.emit_completion_error(RuntimeError("transport failed"))

    run_agent(scripted_client(script))
    first, second, parent = finished_spans(exporter, 3)
    assert first.name == second.name == "chat"
    assert first.kind == second.kind == SpanKind.CLIENT
    assert second.status.status_code == StatusCode.ERROR
    assert parent.attributes["atomic_agents.llm.attempts"] == 2
    assert parent.attributes["gen_ai.usage.input_tokens"] == 3


@pytest.mark.parametrize("chain", ["__cause__", "__context__"])
def test_wrapped_validation_error_and_tool_exception_are_redacted(exporter, chain):
    from atomic_agents.observability import tool_spans
    from atomic_agents.base.base_tool import BaseTool

    installed = installation._current_installation()
    installation.install(tracer_provider=installed.tracer_provider, config=TracingConfig(hide_outputs=True))
    with pytest.raises(ValidationError) as invalid:
        Answer.model_validate({"answer": {"secret": "private model text"}})
    wrapped = RuntimeError("private wrapper text")
    setattr(wrapped, chain, invalid.value)
    with pytest.raises(RuntimeError), tool_spans._tool_span(Mock(spec=BaseTool, tool_name="test"), (), {}):
        raise wrapped
    (span,) = finished_spans(exporter, 1)
    assert span.status.description == "RuntimeError: 1 errors (__REDACTED__)"
    assert span.events[0].attributes["exception.message"] == span.status.description
    stacktrace = span.events[0].attributes["exception.stacktrace"]
    assert "Traceback (most recent call last)" in stacktrace
    assert stacktrace.rstrip().endswith(span.status.description)
    assert "private model text" not in stacktrace
    assert "private wrapper text" not in stacktrace
