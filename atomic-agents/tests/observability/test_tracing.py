import abc
import asyncio
import contextvars
import gc
import json
import logging
import types
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import entry_points
from typing import Dict, List
from unittest.mock import Mock, PropertyMock

import instructor
import pytest
from openinference.instrumentation import OITracer, TraceConfig, using_metadata, using_session
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from pydantic import Field, ValidationError

from atomic_agents import AgentConfig, AtomicAgent, BaseIOSchema, BaseTool
from atomic_agents.context import BaseDynamicContextProvider, SystemPromptGenerator
from atomic_agents.observability import AtomicAgentsInstrumentor
import atomic_agents.observability as observability
from atomic_agents.observability import agent_spans, installation, span_attributes, span_payloads, tracing as tracing


class QuestionSchema(BaseIOSchema):
    """A question for the agent."""

    question: str = Field(...)


class AnswerSchema(BaseIOSchema):
    """An answer from the agent."""

    answer: str = Field(...)


class CalculationInputSchema(BaseIOSchema):
    """Expression to evaluate."""

    expression: str = Field(...)


class CalculationOutputSchema(BaseIOSchema):
    """Result of the evaluation."""

    result: str = Field(...)


class CalculatorTool(BaseTool[CalculationInputSchema, CalculationOutputSchema]):
    def run(self, params: CalculationInputSchema, precision: int = 2) -> CalculationOutputSchema:
        return CalculationOutputSchema(result=f"{len(params.expression):.{precision}f}")


class AsyncCalculatorTool(BaseTool[CalculationInputSchema, CalculationOutputSchema]):
    def run(self, params: CalculationInputSchema) -> CalculationOutputSchema:
        return asyncio.run(self.arun(params))

    async def arun(self, params: CalculationInputSchema) -> CalculationOutputSchema:
        return CalculationOutputSchema(result=str(len(params.expression)))


class ThreadedCalculatorTool(BaseTool[CalculationInputSchema, CalculationOutputSchema]):
    def run(self, params: CalculationInputSchema) -> CalculationOutputSchema:
        return CalculationOutputSchema(result=str(len(params.expression)))

    async def run_async(self, params: CalculationInputSchema) -> CalculationOutputSchema:
        return await asyncio.to_thread(self.run, params)


class RunMixin:
    def run(self, params: CalculationInputSchema) -> CalculationOutputSchema:
        return CalculationOutputSchema(result="mixin")


class MixinCalculatorTool(RunMixin, BaseTool[CalculationInputSchema, CalculationOutputSchema]):
    pass


class StaticRunTool(BaseTool[CalculationInputSchema, CalculationOutputSchema]):
    @staticmethod
    def run(params: CalculationInputSchema) -> CalculationOutputSchema:
        return CalculationOutputSchema(result="static")


class FailingTool(BaseTool[CalculationInputSchema, CalculationOutputSchema]):
    def run(self, params: CalculationInputSchema) -> CalculationOutputSchema:
        raise ValueError("boom")


class AnswerStore:
    """Container object pydantic cannot serialize to JSON."""

    def __init__(self):
        self.values: List[int] = []


class UnserializableInputSchema(BaseIOSchema):
    """Input carrying an arbitrary object."""

    model_config = {"arbitrary_types_allowed": True}
    store: AnswerStore = Field(...)


class EchoTool(BaseTool[UnserializableInputSchema, CalculationOutputSchema]):
    def run(self, params: UnserializableInputSchema) -> CalculationOutputSchema:
        return CalculationOutputSchema(result="echo")


@pytest.fixture
def exporter():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    exporter.provider = provider
    instrumentor = AtomicAgentsInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    yield exporter
    instrumentor.uninstrument()


@pytest.fixture
def provider_tracer(exporter):
    return exporter.provider.get_tracer("test")


def sync_client(response=None, partials=None):
    client = Mock(spec=instructor.Instructor)
    client.chat = Mock()
    client.chat.completions = Mock()
    client.chat.completions.create = Mock(return_value=response or AnswerSchema(answer="42"))

    def create_partial(**kwargs):
        for partial in partials or [AnswerSchema(answer="4"), AnswerSchema(answer="42")]:
            yield partial

    client.chat.completions.create_partial = create_partial
    return client


def async_client(response=None, partials=None):
    client = Mock(spec=instructor.AsyncInstructor)
    client.chat = Mock()
    client.chat.completions = Mock()

    async def create(**kwargs):
        return response or AnswerSchema(answer="42")

    async def create_partial(**kwargs):
        for partial in partials or [AnswerSchema(answer="4"), AnswerSchema(answer="42")]:
            yield partial

    client.chat.completions.create = create
    client.chat.completions.create_partial = create_partial
    return client


def make_agent(client) -> AtomicAgent[QuestionSchema, AnswerSchema]:
    return AtomicAgent[QuestionSchema, AnswerSchema](AgentConfig(client=client, model="test-model"))


def spans_of(exporter) -> List[ReadableSpan]:
    return list(exporter.get_finished_spans())


def kind_of(span: ReadableSpan) -> str:
    return span.attributes[SpanAttributes.OPENINFERENCE_SPAN_KIND]


def input_json(span: ReadableSpan) -> Dict:
    return json.loads(span.attributes[SpanAttributes.INPUT_VALUE])


def output_json(span: ReadableSpan) -> Dict:
    return json.loads(span.attributes[SpanAttributes.OUTPUT_VALUE])


def test_run_emits_agent_span_with_io_and_framework_attributes(exporter):
    agent = make_agent(sync_client())

    response = agent.run(QuestionSchema(question="6*7?"))

    assert response.answer == "42"
    (span,) = spans_of(exporter)
    assert span.name == "invoke_agent AtomicAgent"
    assert kind_of(span) == OpenInferenceSpanKindValues.AGENT.value
    assert span.attributes[SpanAttributes.AGENT_NAME] == "AtomicAgent"
    assert span.attributes[span_attributes.ATTR_AGENT_CLASS] == "AtomicAgent[QuestionSchema, AnswerSchema]"
    assert input_json(span) == {"question": "6*7?"}
    assert output_json(span) == {"answer": "42"}
    assert span.attributes[SpanAttributes.INPUT_MIME_TYPE] == "application/json"
    assert span.attributes[SpanAttributes.OUTPUT_MIME_TYPE] == "application/json"
    assert span.attributes[span_attributes.ATTR_MODEL] == "test-model"
    assert span.attributes[span_attributes.ATTR_MODE] == agent.mode.value
    assert span.attributes[span_attributes.ATTR_INPUT_SCHEMA] == "QuestionSchema"
    assert span.attributes[span_attributes.ATTR_OUTPUT_SCHEMA] == "AnswerSchema"
    assert span.attributes[span_attributes.ATTR_HISTORY_MESSAGE_COUNT] == 0
    assert span.attributes[span_attributes.ATTR_CONTEXT_PROVIDERS] == ()
    assert len(span.attributes[span_attributes.ATTR_SYSTEM_PROMPT_SHA256]) == 64
    assert span.status.status_code == StatusCode.OK


def test_second_turn_reports_history_size_entering_the_turn(exporter):
    agent = make_agent(sync_client())
    agent.run(QuestionSchema(question="first"))

    agent.run(QuestionSchema(question="second"))

    first, second = spans_of(exporter)
    assert first.attributes[span_attributes.ATTR_HISTORY_MESSAGE_COUNT] == 0
    assert second.attributes[span_attributes.ATTR_HISTORY_MESSAGE_COUNT] == 2


def test_agent_without_system_role_has_no_prompt_hash(exporter):
    agent = AtomicAgent[QuestionSchema, AnswerSchema](AgentConfig(client=sync_client(), model="m", system_role=None))

    agent.run(QuestionSchema(question="?"))

    (span,) = spans_of(exporter)
    assert span_attributes.ATTR_SYSTEM_PROMPT_SHA256 not in span.attributes


def test_abstract_intermediate_tool_base_is_not_wrapped(exporter):
    class AbstractCalculator(BaseTool[CalculationInputSchema, CalculationOutputSchema]):
        @abc.abstractmethod
        def run(self, params: CalculationInputSchema) -> CalculationOutputSchema:
            raise NotImplementedError

    assert not hasattr(AbstractCalculator.__dict__["run"], installation.TRACED_MARKER)


def test_run_without_input_sets_no_input_attribute(exporter):
    agent = make_agent(sync_client())

    agent.run()

    (span,) = spans_of(exporter)
    assert SpanAttributes.INPUT_VALUE not in span.attributes
    assert output_json(span) == {"answer": "42"}


def test_run_failure_propagates_and_marks_span_error(exporter):
    client = sync_client()
    client.chat.completions.create = Mock(side_effect=RuntimeError("provider down"))
    agent = make_agent(client)

    with pytest.raises(RuntimeError, match="provider down"):
        agent.run(QuestionSchema(question="?"))

    (span,) = spans_of(exporter)
    assert span.status.status_code == StatusCode.ERROR
    assert [event.name for event in span.events] == ["exception"]


def test_subclassed_agent_is_named_after_the_subclass_and_emits_one_span(exporter):
    class ResearchAgent(AtomicAgent[QuestionSchema, AnswerSchema]):
        def run(self, user_input=None):
            return super().run(user_input)

    agent = ResearchAgent(AgentConfig(client=sync_client(), model="test-model"))

    agent.run(QuestionSchema(question="?"))

    (span,) = spans_of(exporter)
    assert span.name == "invoke_agent ResearchAgent"


def test_agent_span_nests_under_a_user_chain_span(exporter, provider_tracer):
    agent = make_agent(sync_client())

    with provider_tracer.start_as_current_span("turn") as turn:
        agent.run(QuestionSchema(question="?"))

    agent_span, turn_span = spans_of(exporter)
    assert agent_span.parent.span_id == turn.get_span_context().span_id
    assert turn_span.name == "turn"


def test_manual_setup_from_the_guide_groups_agents_and_tools_under_one_chain_span():
    exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace_config = TraceConfig()
    instrumentor = AtomicAgentsInstrumentor()
    instrumentor.instrument(tracer_provider=tracer_provider, config=trace_config)
    tracer = OITracer(tracer_provider.get_tracer(__name__), config=trace_config)
    planner = make_agent(sync_client())
    answerer = make_agent(sync_client())
    calculator = CalculatorTool()
    try:
        with tracer.start_as_current_span("answer_question", openinference_span_kind="chain") as turn:
            turn.set_input("what is 2 + 2?")
            plan = planner.run(QuestionSchema(question="what is 2 + 2?"))
            calculation = calculator.run(CalculationInputSchema(expression=plan.answer))
            answer = answerer.run(QuestionSchema(question=str(calculation.result)))
            turn.set_output(answer.answer)
    finally:
        instrumentor.uninstrument()

    spans = spans_of(exporter)
    chain_span = next(span for span in spans if kind_of(span) == OpenInferenceSpanKindValues.CHAIN.value)
    children = [span for span in spans if span is not chain_span]
    assert [kind_of(span) for span in children] == ["AGENT", "TOOL", "AGENT"]
    assert all(span.parent.span_id == chain_span.get_span_context().span_id for span in children)
    assert chain_span.attributes[SpanAttributes.INPUT_VALUE] == "what is 2 + 2?"
    assert chain_span.attributes[SpanAttributes.OUTPUT_VALUE] == answer.answer


def test_llm_call_inside_run_becomes_child_of_agent_span(exporter, provider_tracer):
    def create(**kwargs):
        with provider_tracer.start_as_current_span("llm"):
            return AnswerSchema(answer="42")

    client = sync_client()
    client.chat.completions.create = create
    agent = make_agent(client)

    agent.run(QuestionSchema(question="?"))

    llm_span, agent_span = spans_of(exporter)
    assert llm_span.parent.span_id == agent_span.context.span_id


def test_run_stream_records_final_output_and_keeps_child_context_per_step(exporter, provider_tracer):
    current_between_yields = []

    def create_partial(**kwargs):
        yield AnswerSchema(answer="4")
        with provider_tracer.start_as_current_span("llm-chunk"):
            yield AnswerSchema(answer="42")

    client = sync_client()
    client.chat.completions.create_partial = create_partial
    agent = make_agent(client)

    partials = []
    for partial in agent.run_stream(QuestionSchema(question="?")):
        partials.append(partial.answer)
        current_between_yields.append(trace_api.get_current_span())

    assert partials == ["4", "42"]
    assert all(span is trace_api.INVALID_SPAN for span in current_between_yields)
    chunk_span, agent_span = spans_of(exporter)
    assert agent_span.name == "invoke_agent AtomicAgent"
    assert chunk_span.parent.span_id == agent_span.context.span_id
    assert output_json(agent_span) == {"answer": "42"}
    assert agent_span.status.status_code == StatusCode.OK


def test_run_stream_preserves_the_generator_return_value(exporter):
    agent = make_agent(sync_client())

    def consume():
        final = yield from agent.run_stream(QuestionSchema(question="?"))
        return final

    generator = consume()
    with pytest.raises(StopIteration) as stop:
        while True:
            next(generator)

    assert stop.value.value.answer == "42"


def test_run_stream_with_no_partials_ends_ok_without_output(exporter):
    client = sync_client()
    client.chat.completions.create_partial = lambda **kwargs: iter([])
    agent = make_agent(client)

    assert list(agent.run_stream(QuestionSchema(question="?"))) == []

    (span,) = spans_of(exporter)
    assert span.status.status_code == StatusCode.OK
    assert SpanAttributes.OUTPUT_VALUE not in span.attributes


def test_run_stream_closed_early_ends_span_and_closes_inner_generator(exporter):
    inner_closed = []

    def create_partial(**kwargs):
        try:
            yield AnswerSchema(answer="4")
            yield AnswerSchema(answer="42")
        finally:
            inner_closed.append(True)

    client = sync_client()
    client.chat.completions.create_partial = create_partial
    agent = make_agent(client)

    for partial in agent.run_stream(QuestionSchema(question="?")):
        break

    assert inner_closed == [True]
    (span,) = spans_of(exporter)
    assert span.attributes[span_attributes.ATTR_STREAM_CLOSED_EARLY] is True
    assert output_json(span) == {"answer": "4"}
    assert span.status.status_code == StatusCode.OK


def test_run_stream_failure_mid_stream_marks_error(exporter):
    def create_partial(**kwargs):
        yield AnswerSchema(answer="4")
        raise RuntimeError("stream broke")

    client = sync_client()
    client.chat.completions.create_partial = create_partial
    agent = make_agent(client)

    with pytest.raises(RuntimeError, match="stream broke"):
        list(agent.run_stream(QuestionSchema(question="?")))

    (span,) = spans_of(exporter)
    assert span.status.status_code == StatusCode.ERROR


@pytest.mark.asyncio
async def test_run_async_emits_agent_span(exporter):
    agent = make_agent(async_client())

    response = await agent.run_async(QuestionSchema(question="?"))

    assert response.answer == "42"
    (span,) = spans_of(exporter)
    assert span.name == "invoke_agent AtomicAgent"
    assert output_json(span) == {"answer": "42"}
    assert span.status.status_code == StatusCode.OK


@pytest.mark.asyncio
async def test_run_async_stream_records_last_partial_and_nests_children(exporter, provider_tracer):
    async def create_partial(**kwargs):
        yield AnswerSchema(answer="4")
        with provider_tracer.start_as_current_span("llm-chunk"):
            yield AnswerSchema(answer="42")

    client = async_client()
    client.chat.completions.create_partial = create_partial
    agent = make_agent(client)

    partials = [partial.answer async for partial in agent.run_async_stream(QuestionSchema(question="?"))]

    assert partials == ["4", "42"]
    chunk_span, agent_span = spans_of(exporter)
    assert agent_span.name == "invoke_agent AtomicAgent"
    assert chunk_span.parent.span_id == agent_span.context.span_id
    assert output_json(agent_span) == {"answer": "42"}


@pytest.mark.asyncio
async def test_run_async_stream_closed_early_ends_span(exporter):
    agent = make_agent(async_client())

    stream = agent.run_async_stream(QuestionSchema(question="?"))
    first = await stream.__anext__()
    await stream.aclose()

    assert first.answer == "4"
    (span,) = spans_of(exporter)
    assert span.attributes[span_attributes.ATTR_STREAM_CLOSED_EARLY] is True
    assert span.status.status_code == StatusCode.OK


@pytest.mark.asyncio
async def test_concurrent_async_runs_keep_separate_spans(exporter):
    first = make_agent(async_client(response=AnswerSchema(answer="one")))
    second = make_agent(async_client(response=AnswerSchema(answer="two")))

    await asyncio.gather(first.run_async(QuestionSchema(question="1")), second.run_async(QuestionSchema(question="2")))

    outputs = {input_json(span)["question"]: output_json(span)["answer"] for span in spans_of(exporter)}
    assert outputs == {"1": "one", "2": "two"}
    assert all(span.parent is None for span in spans_of(exporter))


def test_tool_defined_before_instrumentation_emits_tool_span(exporter):
    tool = CalculatorTool()

    result = tool.run(CalculationInputSchema(expression="6*7"), precision=1)

    assert result.result == "3.0"
    (span,) = spans_of(exporter)
    assert span.name == "execute_tool CalculationInputSchema"
    assert kind_of(span) == OpenInferenceSpanKindValues.TOOL.value
    assert span.attributes[SpanAttributes.TOOL_NAME] == "CalculationInputSchema"
    assert span.attributes[SpanAttributes.TOOL_DESCRIPTION] == "Expression to evaluate."
    assert json.loads(span.attributes[SpanAttributes.TOOL_PARAMETERS])["title"] == "CalculationInputSchema"
    assert input_json(span) == {"expression": "6*7"}
    assert output_json(span) == {"result": "3.0"}
    assert span.status.status_code == StatusCode.OK


def test_tool_accepts_keyword_params(exporter):
    result = CalculatorTool().run(params=CalculationInputSchema(expression="1+1"))

    assert result.result == "3.00"
    (span,) = spans_of(exporter)
    assert input_json(span) == {"expression": "1+1"}


def test_tool_defined_after_instrumentation_emits_tool_span(exporter):
    class LateTool(BaseTool[CalculationInputSchema, CalculationOutputSchema]):
        def run(self, params: CalculationInputSchema) -> CalculationOutputSchema:
            return CalculationOutputSchema(result="late")

    LateTool().run(CalculationInputSchema(expression="x"))

    (span,) = spans_of(exporter)
    assert span.name == "execute_tool CalculationInputSchema"
    assert LateTool().input_schema is CalculationInputSchema


def test_subclass_calling_super_run_emits_one_span(exporter):
    class LoudCalculatorTool(CalculatorTool):
        def run(self, params: CalculationInputSchema, precision: int = 2) -> CalculationOutputSchema:
            return super().run(params, precision)

    LoudCalculatorTool().run(CalculationInputSchema(expression="2*2"))

    (span,) = spans_of(exporter)
    assert span.name == "execute_tool CalculationInputSchema"


def test_sync_run_delegating_to_arun_emits_one_span(exporter):
    AsyncCalculatorTool().run(CalculationInputSchema(expression="2*2"))

    (span,) = spans_of(exporter)
    assert span.name == "execute_tool CalculationInputSchema"
    assert output_json(span) == {"result": "3"}


@pytest.mark.asyncio
async def test_run_async_delegating_to_thread_emits_one_span(exporter):
    await ThreadedCalculatorTool().run_async(CalculationInputSchema(expression="3*3"))

    (span,) = spans_of(exporter)
    assert span.name == "execute_tool CalculationInputSchema"
    assert output_json(span) == {"result": "3"}


def test_mixin_defined_run_is_traced(exporter):
    MixinCalculatorTool().run(CalculationInputSchema(expression="x"))

    (span,) = spans_of(exporter)
    assert span.name == "execute_tool CalculationInputSchema"


def test_static_run_is_left_alone(exporter):
    assert StaticRunTool.run(CalculationInputSchema(expression="x")).result == "static"
    assert spans_of(exporter) == []


def test_tool_failure_propagates_same_exception_and_marks_error(exporter):
    with pytest.raises(ValueError, match="boom") as raised:
        FailingTool().run(CalculationInputSchema(expression="x"))

    (span,) = spans_of(exporter)
    assert span.status.status_code == StatusCode.ERROR
    assert raised.value.args == ("boom",)
    assert span.attributes["error.type"] == "ValueError"
    assert [event.name for event in span.events] == ["exception"]


@pytest.mark.asyncio
async def test_tool_run_async_cancellation_leaves_span_unset(exporter):
    started = asyncio.Event()
    release = asyncio.Event()

    class WaitingTool(CalculatorTool):
        async def run_async(self, params: CalculationInputSchema) -> CalculationOutputSchema:
            started.set()
            await release.wait()
            return CalculationOutputSchema(result="never")

    task = asyncio.create_task(WaitingTool().run_async(CalculationInputSchema(expression="x")))
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    (span,) = spans_of(exporter)
    assert span.attributes.get("atomic_agents.cancelled") is True
    assert span.status.status_code == StatusCode.UNSET
    assert span.events == ()
    assert "error.type" not in span.attributes


@pytest.mark.parametrize("error_type", [GeneratorExit, asyncio.CancelledError])
@pytest.mark.parametrize("method_name", ["run", "run_async"])
@pytest.mark.asyncio
async def test_tool_cancellation_propagates_without_error(exporter, monkeypatch, error_type, method_name):
    def cancelled_result(*args, **kwargs):
        raise error_type

    monkeypatch.setattr(CalculationOutputSchema, "__init__", cancelled_result)
    tool = ThreadedCalculatorTool()
    with pytest.raises(error_type):
        result = getattr(tool, method_name)(CalculationInputSchema(expression="x"))
        if method_name == "run_async":
            await result
    (span,) = spans_of(exporter)
    assert span.attributes.get("atomic_agents.cancelled") is True
    assert span.status.status_code == StatusCode.UNSET
    assert span.events == ()
    assert "error.type" not in span.attributes


def test_unserializable_input_still_returns_result(exporter, caplog):
    with caplog.at_level(logging.WARNING):
        result = EchoTool().run(UnserializableInputSchema(store=AnswerStore()))

    assert result.result == "echo"
    (span,) = spans_of(exporter)
    assert span.status.status_code == StatusCode.OK


def test_agent_run_inside_tool_nests_under_tool_span(exporter):
    agent = make_agent(sync_client())

    class AgentTool(BaseTool[CalculationInputSchema, CalculationOutputSchema]):
        def run(self, params: CalculationInputSchema) -> CalculationOutputSchema:
            return CalculationOutputSchema(result=agent.run(QuestionSchema(question=params.expression)).answer)

    AgentTool().run(CalculationInputSchema(expression="?"))

    agent_span, tool_span = spans_of(exporter)
    assert agent_span.parent.span_id == tool_span.context.span_id


def test_hide_inputs_and_outputs_redacts_values_and_drops_mime_types():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = AtomicAgentsInstrumentor()
    instrumentor.instrument(tracer_provider=provider, config=TraceConfig(hide_inputs=True, hide_outputs=True))
    try:
        make_agent(sync_client()).run(QuestionSchema(question="secret"))
        CalculatorTool().run(CalculationInputSchema(expression="1+1"))
    finally:
        instrumentor.uninstrument()

    for span in spans_of(exporter):
        assert span.attributes[SpanAttributes.INPUT_VALUE] == "__REDACTED__"
        assert span.attributes[SpanAttributes.OUTPUT_VALUE] == "__REDACTED__"
        assert SpanAttributes.INPUT_MIME_TYPE not in span.attributes
        assert SpanAttributes.OUTPUT_MIME_TYPE not in span.attributes


def test_session_and_user_metadata_survive_on_agent_span(exporter):
    agent = make_agent(sync_client())

    with using_session("session-1"), using_metadata({"tenant": "acme"}):
        agent.run(QuestionSchema(question="?"))

    (span,) = spans_of(exporter)
    assert span.attributes[SpanAttributes.SESSION_ID] == "session-1"
    assert json.loads(span.attributes[SpanAttributes.METADATA]) == {"tenant": "acme"}


def test_second_instrument_replaces_provider_and_warns(exporter, caplog):
    other_provider = TracerProvider()
    other_exporter = InMemorySpanExporter()
    other_provider.add_span_processor(SimpleSpanProcessor(other_exporter))
    agent = make_agent(sync_client())
    agent.run(QuestionSchema(question="first"))
    installed = installation._installation
    wrapped_run = AtomicAgent.run

    with caplog.at_level(logging.WARNING):
        AtomicAgentsInstrumentor().instrument(tracer_provider=other_provider)

    agent.run(QuestionSchema(question="second"))
    assert installation._installation is installed
    assert AtomicAgent.run is wrapped_run
    assert len(spans_of(exporter)) == 1
    assert len(spans_of(other_exporter)) == 1
    assert "tracer provider/config replaced" in caplog.text
    assert "in-flight spans finish on the provider they started with" in caplog.text
    assert caplog.records[-1].levelno == logging.WARNING
    other_provider.shutdown()


def test_uninstrument_restores_originals_and_stops_tracing():
    original_run = AtomicAgent.__dict__["run"]
    original_tool_run = CalculatorTool.__dict__["run"]
    original_init_subclass = BaseTool.__dict__["__init_subclass__"]
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = AtomicAgentsInstrumentor()

    instrumentor.instrument(tracer_provider=provider)
    assert instrumentor.is_instrumented_by_opentelemetry
    instrumentor.uninstrument()
    instrumentor.uninstrument()

    assert not instrumentor.is_instrumented_by_opentelemetry
    assert AtomicAgent.__dict__["run"] is original_run
    assert CalculatorTool.__dict__["run"] is original_tool_run
    assert BaseTool.__dict__["__init_subclass__"] is original_init_subclass
    assert "run" not in MixinCalculatorTool.__dict__

    class AfterwardsTool(BaseTool[CalculationInputSchema, CalculationOutputSchema]):
        def run(self, params: CalculationInputSchema) -> CalculationOutputSchema:
            return CalculationOutputSchema(result="plain")

    assert AfterwardsTool().input_schema is CalculationInputSchema
    make_agent(sync_client()).run(QuestionSchema(question="?"))
    AfterwardsTool().run(CalculationInputSchema(expression="x"))
    assert spans_of(exporter) == []


def test_phoenix_entry_point_loads_and_instruments_with_only_a_tracer_provider():
    (entry_point,) = [point for point in entry_points(group="openinference_instrumentor") if point.name == "atomic_agents"]
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    instrumentor = entry_point.load()()
    instrumentor.instrument(tracer_provider=provider)
    try:
        make_agent(sync_client()).run(QuestionSchema(question="?"))
    finally:
        instrumentor.uninstrument()

    assert isinstance(instrumentor, AtomicAgentsInstrumentor)
    assert len(spans_of(exporter)) == 1


@pytest.mark.parametrize("options", [{}, {"skip_dep_check": False}])
def test_instrument_without_opentelemetry_raises_install_hint(monkeypatch: pytest.MonkeyPatch, options: dict[str, object]):
    monkeypatch.setattr(observability, "_opentelemetry_is_installed", lambda: False)

    with pytest.raises(ImportError, match="atomic-agents\\[observability\\]"):
        AtomicAgentsInstrumentor().instrument(**options)


def test_uninstrument_without_opentelemetry_is_a_no_op(monkeypatch):
    monkeypatch.setattr(observability, "_opentelemetry_is_installed", lambda: False)

    AtomicAgentsInstrumentor().uninstrument()

    assert not AtomicAgentsInstrumentor().is_instrumented_by_opentelemetry


def test_openinference_below_the_floor_only_disables_context_with_a_warning(monkeypatch, caplog):
    monkeypatch.setattr(observability, "version", lambda distribution: "0.1.20")

    with caplog.at_level(logging.WARNING):
        AtomicAgentsInstrumentor().instrument()

    assert "0.1.38 or newer is required" in caplog.text
    assert AtomicAgentsInstrumentor().is_instrumented_by_opentelemetry
    assert span_attributes._context_attributes() == {}
    AtomicAgentsInstrumentor().uninstrument()


class CountingProvider(BaseDynamicContextProvider):
    def __init__(self):
        super().__init__(title="Counter")
        self.calls = 0

    def get_info(self) -> str:
        self.calls += 1
        return f"call {self.calls}"


def test_prompt_hash_identifies_static_sections_despite_dynamic_providers(exporter):
    generator = SystemPromptGenerator(background=["v1"], context_providers={"counter": CountingProvider()})
    agent = AtomicAgent[QuestionSchema, AnswerSchema](
        AgentConfig(client=sync_client(), model="m", system_prompt_generator=generator)
    )

    agent.run(QuestionSchema(question="a"))
    agent.run(QuestionSchema(question="b"))

    first, second = spans_of(exporter)
    assert (
        first.attributes[span_attributes.ATTR_SYSTEM_PROMPT_SHA256]
        == second.attributes[span_attributes.ATTR_SYSTEM_PROMPT_SHA256]
    )
    assert first.attributes[span_attributes.ATTR_CONTEXT_PROVIDERS] == ("counter",)


def test_prompt_hash_changes_when_static_sections_change(exporter):
    agent = make_agent(sync_client())
    agent.run(QuestionSchema(question="a"))
    agent.system_prompt_generator.background = ["v2"]

    agent.run(QuestionSchema(question="b"))

    first, second = spans_of(exporter)
    assert (
        first.attributes[span_attributes.ATTR_SYSTEM_PROMPT_SHA256]
        != second.attributes[span_attributes.ATTR_SYSTEM_PROMPT_SHA256]
    )


def test_run_stream_dropped_without_close_still_ends_span(exporter):
    agent = make_agent(sync_client())
    stream = agent.run_stream(QuestionSchema(question="?"))
    next(stream)

    del stream
    gc.collect()

    (span,) = spans_of(exporter)
    assert span.attributes[span_attributes.ATTR_STREAM_CLOSED_EARLY] is True


@pytest.mark.asyncio
async def test_run_async_stream_cancelled_mid_await_is_not_an_error(exporter):
    release = asyncio.Event()

    async def create_partial(**kwargs):
        yield AnswerSchema(answer="4")
        await release.wait()
        yield AnswerSchema(answer="42")

    client = async_client()
    client.chat.completions.create_partial = create_partial
    agent = make_agent(client)

    async def consume():
        async for _ in agent.run_async_stream(QuestionSchema(question="?")):
            pass

    task = asyncio.create_task(consume())
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    (span,) = spans_of(exporter)
    assert span.attributes[span_attributes.ATTR_STREAM_CLOSED_EARLY] is True
    assert output_json(span) == {"answer": "4"}
    assert span.status.status_code != StatusCode.ERROR


def test_forge_style_thread_pool_tool_emits_one_span_in_the_callers_trace(exporter, provider_tracer):
    class ForgeStyleTool(BaseTool[CalculationInputSchema, CalculationOutputSchema]):
        def run(self, params: CalculationInputSchema) -> CalculationOutputSchema:
            with ThreadPoolExecutor() as executor:
                return executor.submit(contextvars.copy_context().run, asyncio.run, self.run_async(params)).result()

        async def run_async(self, params: CalculationInputSchema) -> CalculationOutputSchema:
            return CalculationOutputSchema(result="pooled")

    with provider_tracer.start_as_current_span("root") as root:
        ForgeStyleTool().run(CalculationInputSchema(expression="x"))

    tool_span, root_span = spans_of(exporter)
    assert tool_span.name == "execute_tool CalculationInputSchema"
    assert tool_span.parent.span_id == root.get_span_context().span_id
    assert tool_span.context.trace_id == root_span.context.trace_id


def test_mcp_style_generated_tool_class_emits_one_span(exporter):
    async def arun(self, params):
        return CalculationOutputSchema(result="mcp")

    def run(self, params):
        return asyncio.run(self.arun(params))

    GeneratedTool = types.new_class(
        "GeneratedTool",
        (BaseTool[CalculationInputSchema, CalculationOutputSchema],),
        {},
        lambda namespace: namespace.update({"run": run, "arun": arun}),
    )

    GeneratedTool().run(CalculationInputSchema(expression="x"))

    (span,) = spans_of(exporter)
    assert span.name == "execute_tool CalculationInputSchema"
    assert output_json(span) == {"result": "mcp"}


def test_async_generator_tool_method_is_left_alone(exporter):
    class StreamingTool(BaseTool[CalculationInputSchema, CalculationOutputSchema]):
        def run(self, params: CalculationInputSchema) -> CalculationOutputSchema:
            return CalculationOutputSchema(result="x")

        async def run_async(self, params: CalculationInputSchema):
            yield CalculationOutputSchema(result="chunk")

    assert not hasattr(StreamingTool.__dict__["run_async"], installation.TRACED_MARKER)
    assert hasattr(StreamingTool.__dict__["run"], installation.TRACED_MARKER)


def test_instrument_uninstrument_instrument_cycle_traces_tools_defined_between_cycles():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    instrumentor = AtomicAgentsInstrumentor()
    instrumentor.instrument(tracer_provider=provider)
    instrumentor.uninstrument()

    class BetweenCyclesTool(BaseTool[CalculationInputSchema, CalculationOutputSchema]):
        def run(self, params: CalculationInputSchema) -> CalculationOutputSchema:
            return CalculationOutputSchema(result="between")

    original_run = BetweenCyclesTool.__dict__["run"]
    instrumentor.instrument(tracer_provider=provider)
    try:
        BetweenCyclesTool().run(CalculationInputSchema(expression="x"))
    finally:
        instrumentor.uninstrument()

    assert [span.name for span in spans_of(exporter)] == ["execute_tool CalculationInputSchema"]
    assert BetweenCyclesTool.__dict__["run"] is original_run


@pytest.fixture
def configured_tracing():
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider, exporter
    AtomicAgentsInstrumentor().uninstrument()
    provider.shutdown()


@pytest.mark.parametrize("openinference_enabled", [False, True])
def test_genai_attributes_survive_without_openinference(monkeypatch, configured_tracing, openinference_enabled):
    provider, exporter = configured_tracing
    monkeypatch.setattr(span_attributes, "_import_context_attributes", lambda: None)
    config = tracing.TracingConfig(openinference_attributes=True if openinference_enabled else None)
    AtomicAgentsInstrumentor().instrument(tracer_provider=provider, config=config)
    client = sync_client()
    client.provider = types.SimpleNamespace(value="openai")
    make_agent(client).run(QuestionSchema(question="question"))
    CalculatorTool().run(CalculationInputSchema(expression="1+1"))

    agent_span, tool_span = spans_of(exporter)
    assert [agent_span.name, tool_span.name] == ["invoke_agent AtomicAgent", "execute_tool CalculationInputSchema"]
    assert agent_span.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert agent_span.attributes["gen_ai.agent.name"] == "AtomicAgent"
    assert agent_span.attributes["gen_ai.request.model"] == "test-model"
    assert agent_span.attributes["gen_ai.provider.name"] == "openai"
    assert agent_span.attributes["atomic_agents.agent.class"] == "AtomicAgent[QuestionSchema, AnswerSchema]"
    assert json.loads(agent_span.attributes["atomic_agents.input"]) == {"question": "question"}
    assert json.loads(agent_span.attributes["atomic_agents.output"]) == {"answer": "42"}
    assert tool_span.attributes["gen_ai.operation.name"] == "execute_tool"
    assert tool_span.attributes["gen_ai.tool.name"] == "CalculationInputSchema"
    assert tool_span.attributes["atomic_agents.tool.class"] == "CalculatorTool"
    assert tool_span.attributes["gen_ai.tool.description"] == "Expression to evaluate."
    assert tool_span.attributes["gen_ai.tool.type"] == "function"
    assert json.loads(tool_span.attributes["gen_ai.tool.call.arguments"]) == {"expression": "1+1"}
    assert json.loads(tool_span.attributes["gen_ai.tool.call.result"]) == {"result": "3.00"}
    assert ("openinference.span.kind" in agent_span.attributes) is openinference_enabled
    assert ("openinference.span.kind" in tool_span.attributes) is openinference_enabled
    if openinference_enabled:
        assert agent_span.attributes["input.value"] == agent_span.attributes["atomic_agents.input"]
        assert tool_span.attributes["input.value"] == tool_span.attributes["gen_ai.tool.call.arguments"]
        assert tool_span.attributes["output.value"] == tool_span.attributes["gen_ai.tool.call.result"]
    else:
        assert not ({"input.value", "output.value", "agent.name"} & agent_span.attributes.keys())
        assert not ({"input.value", "output.value", "tool.name", "tool.parameters"} & tool_span.attributes.keys())


@pytest.mark.parametrize("config_class", [tracing.TracingConfig, TraceConfig])
def test_both_visibility_configs_redact_primary_payloads(configured_tracing, config_class):
    provider, exporter = configured_tracing
    AtomicAgentsInstrumentor().instrument(tracer_provider=provider, config=config_class(hide_inputs=True, hide_outputs=True))
    make_agent(sync_client()).run(QuestionSchema(question="secret"))
    CalculatorTool().run(CalculationInputSchema(expression="secret"))

    agent_span, tool_span = spans_of(exporter)
    assert agent_span.attributes["atomic_agents.input"] == "__REDACTED__"
    assert agent_span.attributes["atomic_agents.output"] == "__REDACTED__"
    assert tool_span.attributes["gen_ai.tool.call.arguments"] == "__REDACTED__"
    assert tool_span.attributes["gen_ai.tool.call.result"] == "__REDACTED__"


@pytest.mark.parametrize("hide_outputs", [False, True])
def test_large_tool_result_is_truncated_unless_hidden(configured_tracing, hide_outputs):
    provider, exporter = configured_tracing
    AtomicAgentsInstrumentor().instrument(
        tracer_provider=provider, config=tracing.TracingConfig(hide_outputs=hide_outputs, openinference_attributes=True)
    )

    class LargeResultTool(BaseTool[CalculationInputSchema, CalculationOutputSchema]):
        def run(self, params: CalculationInputSchema) -> CalculationOutputSchema:
            return CalculationOutputSchema(result="x" * 10240)

    result = LargeResultTool().run(CalculationInputSchema(expression="large"))
    (span,) = spans_of(exporter)
    serialized = result.model_dump_json()
    expected = "__REDACTED__" if hide_outputs else serialized[: 4096 - len("...[truncated]")] + "...[truncated]"
    assert span.attributes["gen_ai.tool.call.result"] == expected
    assert span.attributes["output.value"] == expected
    if hide_outputs:
        assert "gen_ai.tool.call.result.original_length" not in span.attributes
        assert "output.value.original_length" not in span.attributes
    else:
        assert span.attributes["gen_ai.tool.call.result.original_length"] == len(serialized)
        assert span.attributes["output.value.original_length"] == len(serialized)


@pytest.mark.parametrize("agent_method", ["run", "run_async", "run_stream", "run_async_stream"])
@pytest.mark.asyncio
async def test_suppression_skips_agent_and_tool_spans(exporter, agent_method):
    from opentelemetry import context

    client = async_client() if "async" in agent_method else sync_client()
    agent = make_agent(client)
    token = context.attach(context.set_value("suppress_instrumentation", True))
    try:
        result = getattr(agent, agent_method)(QuestionSchema(question="suppressed"))
        assert await final_agent_answer(agent_method, result) == "42"
        assert CalculatorTool().run(CalculationInputSchema(expression="1+1")).result == "3.00"
        assert (await ThreadedCalculatorTool().run_async(CalculationInputSchema(expression="1+1"))).result == "3"
    finally:
        context.detach(token)
    assert spans_of(exporter) == []


def test_failures_record_qualified_error_type(exporter):
    client = sync_client()
    client.chat.completions.create.side_effect = ValueError("agent failure")
    with pytest.raises(ValueError, match="agent failure"):
        make_agent(client).run(QuestionSchema(question="fail"))
    with pytest.raises(ValueError, match="boom"):
        FailingTool().run(CalculationInputSchema(expression="fail"))
    for span in spans_of(exporter):
        assert span.attributes["error.type"] == "ValueError"
        assert span.status.status_code == StatusCode.ERROR
        assert len(span.events) == 1


def test_tracing_config_environment_and_coercion(monkeypatch):
    monkeypatch.setenv("OPENINFERENCE_HIDE_INPUTS", "TrUe")
    monkeypatch.setenv("OPENINFERENCE_HIDE_OUTPUTS", "1")
    monkeypatch.setenv("ATOMIC_AGENTS_TRACE_MAX_ATTRIBUTE_LENGTH", "128")
    from atomic_agents.observability import TracingConfig

    config = TracingConfig.from_environment()
    assert config == TracingConfig(hide_inputs=True, hide_outputs=True, max_attribute_length=128)
    assert TracingConfig.coerce(None) == config
    assert TracingConfig.coerce(config) is config
    assert TracingConfig.coerce(TraceConfig(hide_inputs=True)) == TracingConfig(hide_inputs=True, max_attribute_length=128)


@pytest.mark.parametrize(
    "value, expected", [("false", False), ("FaLsE", False), ("0", False), ("true", True), ("1", True), ("", True)]
)
def test_tracing_config_llm_spans_environment(monkeypatch, value, expected):
    from atomic_agents.observability import TracingConfig

    monkeypatch.setenv("ATOMIC_AGENTS_TRACE_LLM_SPANS", value)
    assert TracingConfig.from_environment().llm_spans is expected
    assert TracingConfig.coerce(None).llm_spans is expected


def test_payload_visibility_preserves_environment_settings(monkeypatch):
    monkeypatch.setenv("ATOMIC_AGENTS_TRACE_LLM_SPANS", "false")
    monkeypatch.setenv("ATOMIC_AGENTS_TRACE_MAX_ATTRIBUTE_LENGTH", "128")
    monkeypatch.setenv("ATOMIC_AGENTS_TRACE_HIDE_OUTPUTS", "true")
    config = tracing.TracingConfig.coerce(TraceConfig(hide_inputs=True, hide_outputs=False))
    assert config.llm_spans is False
    assert config.max_attribute_length == 128
    assert config.hide_inputs is True
    assert config.hide_outputs is False


def test_tracing_config_llm_spans_defaults_to_enabled(monkeypatch):
    from atomic_agents.observability import TracingConfig

    monkeypatch.delenv("ATOMIC_AGENTS_TRACE_LLM_SPANS", raising=False)
    assert TracingConfig.from_environment().llm_spans is True
    assert TracingConfig().llm_spans is True


def test_session_is_mirrored_to_genai_conversation(exporter):
    with using_session("session-1"):
        make_agent(sync_client()).run(QuestionSchema(question="?"))
        CalculatorTool().run(CalculationInputSchema(expression="?"))
    for span in spans_of(exporter):
        assert span.attributes["gen_ai.conversation.id"] == "session-1"


async def final_agent_answer(agent_method, result):
    if agent_method == "run_async":
        return (await result).answer
    elif agent_method == "run_stream":
        return list(result)[-1].answer
    elif agent_method == "run_async_stream":
        return [partial async for partial in result][-1].answer
    else:
        return result.answer


@pytest.mark.parametrize(
    "name, expected", [(None, "AtomicAgent"), ("", "AtomicAgent"), ("orchestrator_planning", "orchestrator_planning")]
)
def test_stable_agent_name_is_separate_from_schema_class(exporter, name, expected):
    agent = AtomicAgent[QuestionSchema, AnswerSchema](AgentConfig(client=sync_client(), name=name))
    agent.run(QuestionSchema(question="?"))
    (span,) = spans_of(exporter)
    assert span.name == f"invoke_agent {expected}"
    assert span.attributes["gen_ai.agent.name"] == expected
    assert span.attributes["agent.name"] == expected
    assert span.attributes["atomic_agents.agent.class"] == "AtomicAgent[QuestionSchema, AnswerSchema]"


@pytest.mark.parametrize("openinference_enabled", [False, True])
def test_conversation_context_overrides_session_on_agent_and_tool(exporter, openinference_enabled):
    config = tracing.TracingConfig(openinference_attributes=openinference_enabled)
    AtomicAgentsInstrumentor().instrument(tracer_provider=exporter.provider, config=config)
    with using_session("session-1"), observability.using_conversation("c-1"):
        make_agent(sync_client()).run(QuestionSchema(question="?"))
        CalculatorTool().run(CalculationInputSchema(expression="?"))
    agent_span, tool_span = spans_of(exporter)
    assert agent_span.attributes["gen_ai.conversation.id"] == "c-1"
    assert tool_span.attributes["gen_ai.conversation.id"] == "c-1"
    assert agent_span.attributes.get("session.id") == ("c-1" if openinference_enabled else None)
    assert tool_span.attributes.get("session.id") == ("c-1" if openinference_enabled else None)


def test_session_is_mirrored_even_with_openinference_attributes_disabled(exporter):
    config = tracing.TracingConfig(openinference_attributes=False)
    AtomicAgentsInstrumentor().instrument(tracer_provider=exporter.provider, config=config)
    with using_session("session-1"):
        make_agent(sync_client()).run(QuestionSchema(question="?"))
        CalculatorTool().run(CalculationInputSchema(expression="?"))
    for span in spans_of(exporter):
        assert span.attributes["gen_ai.conversation.id"] == "session-1"
        assert "session.id" not in span.attributes


def test_conversation_context_restores_after_nested_exception(exporter):
    with observability.using_conversation("outer"):
        with pytest.raises(ValueError), observability.using_conversation("inner"):
            CalculatorTool().run(CalculationInputSchema(expression="inner"))
            raise ValueError("reset context")
        CalculatorTool().run(CalculationInputSchema(expression="outer"))
    CalculatorTool().run(CalculationInputSchema(expression="unset"))
    inner, outer, unset = spans_of(exporter)
    assert inner.attributes["gen_ai.conversation.id"] == "inner"
    assert outer.attributes["gen_ai.conversation.id"] == "outer"
    assert "gen_ai.conversation.id" not in unset.attributes


@pytest.mark.asyncio
async def test_conversation_context_is_task_local(exporter):
    async def run_in_conversation(conversation_id):
        with observability.using_conversation(conversation_id):
            await asyncio.sleep(0)
            await make_agent(async_client()).run_async(QuestionSchema(question="?"))
            CalculatorTool().run(CalculationInputSchema(expression="?"))

    await asyncio.gather(run_in_conversation("first"), run_in_conversation("second"))
    assert [span.attributes["gen_ai.conversation.id"] for span in spans_of(exporter)] == ["first", "first", "second", "second"]


def test_plain_parent_span_owns_agent_and_tool_trace(exporter):
    with exporter.provider.get_tracer("app").start_as_current_span("turn") as turn:
        make_agent(sync_client()).run(QuestionSchema(question="?"))
        CalculatorTool().run(CalculationInputSchema(expression="?"))
    agent_span, tool_span, parent = spans_of(exporter)
    assert agent_span.context.trace_id == turn.get_span_context().trace_id
    assert tool_span.context.trace_id == turn.get_span_context().trace_id
    assert agent_span.parent.span_id == parent.context.span_id
    assert tool_span.parent.span_id == parent.context.span_id


def test_same_provider_and_config_is_a_silent_no_op(exporter, caplog):
    installed = installation._installation
    tracer = installed.tracer
    with caplog.at_level(logging.DEBUG, logger=tracing.__name__):
        AtomicAgentsInstrumentor().instrument(tracer_provider=exporter.provider, config=tracing.TracingConfig())
    assert installation._installation is installed
    assert installed.tracer is tracer
    assert not caplog.records


def test_reinstrument_replaces_config_without_rewrapping(exporter, caplog):
    wrapped_run = AtomicAgent.run
    agent = make_agent(sync_client())
    config = tracing.TracingConfig(hide_inputs=True, openinference_attributes=False)
    with caplog.at_level(logging.WARNING):
        AtomicAgentsInstrumentor().instrument(tracer_provider=exporter.provider, config=config)
    agent.run(QuestionSchema(question="secret"))
    CalculatorTool().run(CalculationInputSchema(expression="secret"))
    agent_span, tool_span = spans_of(exporter)
    assert agent_span.attributes["atomic_agents.input"] == "__REDACTED__"
    assert tool_span.attributes["gen_ai.tool.call.arguments"] == "__REDACTED__"
    assert "agent.name" not in agent_span.attributes
    assert AtomicAgent.run is wrapped_run
    assert "tracer provider/config replaced" in caplog.text


def test_in_flight_span_finishes_on_original_provider(exporter):
    other_provider = TracerProvider()
    other_exporter = InMemorySpanExporter()
    other_provider.add_span_processor(SimpleSpanProcessor(other_exporter))
    client = sync_client()

    def replace_provider(**kwargs):
        AtomicAgentsInstrumentor().instrument(tracer_provider=other_provider)
        CalculatorTool().run(CalculationInputSchema(expression="?"))
        return AnswerSchema(answer="42")

    client.chat.completions.create.side_effect = replace_provider
    make_agent(client).run(QuestionSchema(question="?"))
    assert len(spans_of(exporter)) == 1
    assert len(spans_of(other_exporter)) == 1
    (agent_span,) = spans_of(exporter)
    (tool_span,) = spans_of(other_exporter)
    assert agent_span.name == "invoke_agent AtomicAgent"
    assert tool_span.parent.span_id == agent_span.context.span_id
    assert tool_span.context.trace_id == agent_span.context.trace_id
    other_provider.shutdown()


@pytest.mark.parametrize("name", ["model", "mode"])
def test_none_descriptions_are_omitted_without_otel_warnings(exporter, caplog, name):
    agent = make_agent(sync_client())
    setattr(agent, name, None)
    with agent_spans._agent_span(agent, (), {}):
        pass
    (span,) = spans_of(exporter)
    assert f"atomic_agents.{name}" not in span.attributes
    assert "gen_ai.request.model" not in span.attributes if name == "model" else True
    assert "Invalid type" not in caplog.text


def test_broken_history_preserves_mandatory_agent_attributes(exporter, monkeypatch, caplog):
    agent = make_agent(sync_client())
    monkeypatch.setattr(type(agent), "history", PropertyMock(side_effect=RuntimeError("history unavailable")), raising=False)
    with caplog.at_level(logging.DEBUG), agent_spans._agent_span(agent, (), {}):
        pass
    (span,) = spans_of(exporter)
    assert span.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert span.attributes["gen_ai.agent.name"] == "AtomicAgent"
    assert span.attributes["atomic_agents.agent.class"] == "AtomicAgent[QuestionSchema, AnswerSchema]"
    assert span.attributes["gen_ai.request.model"] == "test-model"
    assert "atomic_agents.history.message_count" not in span.attributes
    assert sum("history.message_count" in record.message for record in caplog.records) == 1


def test_broken_tool_description_preserves_other_attributes(exporter, monkeypatch):
    monkeypatch.setattr(CalculatorTool, "tool_description", PropertyMock(side_effect=RuntimeError("unavailable")))
    CalculatorTool().run(CalculationInputSchema(expression="1+1"))
    (span,) = spans_of(exporter)
    assert span.attributes["gen_ai.tool.name"] == "CalculationInputSchema"
    assert span.attributes["atomic_agents.tool.class"] == "CalculatorTool"
    assert span.attributes["tool.name"] == "CalculationInputSchema"
    assert "tool.parameters" in span.attributes
    assert "gen_ai.tool.description" not in span.attributes


def test_configured_tool_name_is_the_span_join_key(exporter):
    tool = CalculatorTool()
    tool.config.title = "calculate"
    tool.run(CalculationInputSchema(expression="1+1"))
    (span,) = spans_of(exporter)
    assert span.name == "execute_tool calculate"
    assert span.attributes["gen_ai.tool.name"] == span.attributes["tool.name"] == "calculate"
    assert span.attributes["atomic_agents.tool.class"] == "CalculatorTool"


@pytest.mark.parametrize("hide", [False, True])
def test_hidden_payload_is_never_serialized(exporter, caplog, hide):
    payload = UnserializableInputSchema(store=AnswerStore())
    with exporter.provider.get_tracer("test").start_as_current_span("payload") as span:
        span_payloads._record_payload(span, (("payload", "payload.mime_type"),), payload, hide, tracing.TracingConfig())
    (span,) = spans_of(exporter)
    assert ("Could not serialize tracing payload" in caplog.text) is not hide
    assert span.attributes.get("payload") == ("__REDACTED__" if hide else None)
    assert "payload.mime_type" not in span.attributes


@pytest.mark.parametrize("hide", [False, True])
def test_hidden_non_model_payload_does_not_call_json_dumps(exporter, monkeypatch, caplog, hide):
    serialize = Mock(side_effect=ValueError("serialization failed"))
    monkeypatch.setattr(span_payloads.json, "dumps", serialize)
    with exporter.provider.get_tracer("test").start_as_current_span("payload") as span:
        span_payloads._record_payload(span, (("payload", None),), object(), hide, tracing.TracingConfig())
    assert serialize.call_count == (0 if hide else 1)
    assert ("Could not serialize tracing payload" in caplog.text) is not hide


@pytest.mark.parametrize(
    "provider,expected",
    [
        ("azure_openai", "azure.ai.openai"),
        ("vertexai", "gcp.vertex_ai"),
        ("gemini", "gcp.gemini"),
        ("genai", "gcp.gemini"),
        ("bedrock", "aws.bedrock"),
        ("mistral", "mistral_ai"),
        ("xai", "x_ai"),
        ("openai", "openai"),
        ("anthropic", "anthropic"),
        ("cohere", "cohere"),
        ("groq", "groq"),
        ("deepseek", "deepseek"),
        ("perplexity", "perplexity"),
        ("custom", "custom"),
    ],
)
def test_instructor_provider_names_follow_genai_conventions(exporter, provider, expected):
    client = sync_client()
    client.provider = types.SimpleNamespace(value=provider)
    make_agent(client).run(QuestionSchema(question="question"))
    (span,) = spans_of(exporter)
    assert span.attributes["gen_ai.provider.name"] == expected


def test_none_context_attributes_are_filtered(exporter, monkeypatch, caplog):
    monkeypatch.setattr(span_attributes, "_context_attributes", lambda: {"optional": None})
    make_agent(sync_client()).run(QuestionSchema(question="question"))
    (span,) = spans_of(exporter)
    assert "optional" not in span.attributes
    assert "gen_ai.provider.name" not in span.attributes
    assert "Invalid type" not in caplog.text


@pytest.mark.parametrize("prefix", ["ATOMIC_AGENTS_TRACE", "OPENINFERENCE"])
def test_visibility_environment_supports_both_spellings(monkeypatch, prefix):
    monkeypatch.delenv("ATOMIC_AGENTS_TRACE_HIDE_INPUTS", raising=False)
    monkeypatch.delenv("ATOMIC_AGENTS_TRACE_HIDE_OUTPUTS", raising=False)
    monkeypatch.setenv(f"{prefix}_HIDE_INPUTS", "TRUE")
    monkeypatch.setenv(f"{prefix}_HIDE_OUTPUTS", "1")
    config = tracing.TracingConfig.from_environment()
    assert config.hide_inputs is True
    assert config.hide_outputs is True


def test_atomic_visibility_environment_overrides_openinference(monkeypatch):
    monkeypatch.setenv("OPENINFERENCE_HIDE_INPUTS", "true")
    monkeypatch.setenv("OPENINFERENCE_HIDE_OUTPUTS", "true")
    monkeypatch.setenv("ATOMIC_AGENTS_TRACE_HIDE_INPUTS", "false")
    monkeypatch.setenv("ATOMIC_AGENTS_TRACE_HIDE_OUTPUTS", "")
    config = tracing.TracingConfig.from_environment()
    assert config.hide_inputs is False
    assert config.hide_outputs is False


def test_error_status_and_recorded_message_are_bounded(configured_tracing):
    provider, exporter = configured_tracing
    AtomicAgentsInstrumentor().instrument(tracer_provider=provider, config=tracing.TracingConfig(max_attribute_length=12))
    error = ValueError("private" * 100)
    agent = make_agent(sync_client())
    with pytest.raises(ValueError) as raised, agent_spans._agent_span(agent, (), {}):
        raise error
    (span,) = spans_of(exporter)
    expected = "...[truncate"
    assert raised.value is error
    assert span.status.description == expected
    assert span.events[0].attributes["exception.message"] == expected
    assert len(span.events[0].attributes["exception.stacktrace"]) <= 12


def test_hidden_validation_errors_do_not_export_model_output(configured_tracing):
    provider, exporter = configured_tracing
    AtomicAgentsInstrumentor().instrument(tracer_provider=provider, config=tracing.TracingConfig(hide_outputs=True))
    try:
        QuestionSchema.model_validate({"question": ["secret model output"]})
    except ValidationError as error:
        with provider.get_tracer("test").start_as_current_span("failure") as span:
            span_payloads._record_failure(span, error)
    (span,) = spans_of(exporter)
    assert span.status.description == "ValidationError: 1 errors (__REDACTED__)"
    assert span.events[0].attributes["exception.message"] == span.status.description
    assert "secret model output" not in str(span.events[0].attributes)


def test_tool_parameters_use_the_payload_size_limit(configured_tracing):
    provider, exporter = configured_tracing
    AtomicAgentsInstrumentor().instrument(
        tracer_provider=provider, config=tracing.TracingConfig(max_attribute_length=12, openinference_attributes=True)
    )
    CalculatorTool().run(CalculationInputSchema(expression="x"))
    (span,) = spans_of(exporter)
    parameters = json.dumps(CalculationInputSchema.model_json_schema())
    assert span.attributes["tool.parameters"] == "...[truncate"
    assert span.attributes["tool.parameters.original_length"] == len(parameters)


@pytest.mark.parametrize("value, expected", [("", 4096), ("4k", 4096), ("0", 4096), ("-5", 4096), ("512", 512)])
@pytest.mark.parametrize("visibility", [None, TraceConfig(hide_inputs=True)])
def test_attribute_length_environment_is_tolerant(monkeypatch, caplog, value, expected, visibility):
    monkeypatch.setenv("ATOMIC_AGENTS_TRACE_MAX_ATTRIBUTE_LENGTH", value)
    with caplog.at_level(logging.WARNING):
        config = tracing.TracingConfig.coerce(visibility)
    assert config.max_attribute_length == expected
    assert config.hide_inputs is (visibility is not None)
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    if expected == 4096:
        assert len(warnings) == 1
        assert repr(value) in warnings[0].message
    else:
        assert warnings == []


def test_reinstall_without_config_preserves_redaction(exporter, caplog):
    config = tracing.TracingConfig(hide_inputs=True, openinference_attributes=False)
    AtomicAgentsInstrumentor().instrument(tracer_provider=exporter.provider, config=config)
    other_provider = TracerProvider()
    other_exporter = InMemorySpanExporter()
    other_provider.add_span_processor(SimpleSpanProcessor(other_exporter))
    caplog.clear()
    try:
        with caplog.at_level(logging.WARNING):
            AtomicAgentsInstrumentor().instrument(tracer_provider=other_provider)
        assert installation._current_installation().config is config
        assert installation._current_installation().tracer_provider is other_provider
        make_agent(sync_client()).run(QuestionSchema(question="secret"))
        assert spans_of(other_exporter)[0].attributes["atomic_agents.input"] == "__REDACTED__"
        assert len(caplog.records) == 1
        assert "replaced" in caplog.records[0].message
    finally:
        other_provider.shutdown()


def test_reinstall_without_provider_preserves_provider(exporter, caplog):
    config = tracing.TracingConfig(hide_outputs=True)
    with caplog.at_level(logging.WARNING):
        AtomicAgentsInstrumentor().instrument(config=config)
    assert installation._current_installation().tracer_provider is exporter.provider
    caplog.clear()
    AtomicAgentsInstrumentor().instrument()
    assert caplog.records == []
    make_agent(sync_client()).run(QuestionSchema(question="?"))
    assert spans_of(exporter)[0].attributes["atomic_agents.output"] == "__REDACTED__"


@pytest.mark.asyncio
async def test_run_async_cancellation_leaves_span_unset(exporter):
    started = asyncio.Event()
    release = asyncio.Event()

    async def create(**kwargs):
        started.set()
        await release.wait()
        return AnswerSchema(answer="never")

    client = async_client()
    client.chat.completions.create = create
    task = asyncio.create_task(make_agent(client).run_async(QuestionSchema(question="?")))
    await asyncio.wait_for(started.wait(), timeout=2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    (span,) = spans_of(exporter)
    assert span.attributes.get("atomic_agents.cancelled") is True
    assert span.status.status_code == StatusCode.UNSET
    assert span.events == ()
    assert "error.type" not in span.attributes


@pytest.mark.parametrize("error_type", [GeneratorExit, asyncio.CancelledError])
def test_run_cancellation_leaves_span_unset(exporter, error_type):
    client = sync_client()
    client.chat.completions.create.side_effect = error_type
    with pytest.raises(error_type):
        make_agent(client).run(QuestionSchema(question="?"))
    (span,) = spans_of(exporter)
    assert span.attributes.get("atomic_agents.cancelled") is True
    assert span.status.status_code == StatusCode.UNSET
    assert span.events == ()


def test_run_stream_cancellation_marks_closed_early(exporter):
    def create_partial(**kwargs):
        yield AnswerSchema(answer="4")
        raise asyncio.CancelledError

    client = sync_client()
    client.chat.completions.create_partial = create_partial
    stream = make_agent(client).run_stream(QuestionSchema(question="?"))
    next(stream)
    with pytest.raises(asyncio.CancelledError):
        next(stream)
    (span,) = spans_of(exporter)
    assert span.attributes.get(span_attributes.ATTR_STREAM_CLOSED_EARLY) is True
    assert span.status.status_code == StatusCode.UNSET
    assert span.events == ()


@pytest.mark.parametrize("broken_name", [False, True])
def test_agent_name_falls_back_without_initialized_base(exporter, broken_name):
    class UninitializedAgent(AtomicAgent[QuestionSchema, AnswerSchema]):
        def __init__(self):
            self.client = sync_client()

    if broken_name:
        UninitializedAgent.name = property(lambda self: 1 / 0)
    agent = UninitializedAgent()
    try:
        display_name = span_attributes._agent_display_name(agent)
    except (AttributeError, ZeroDivisionError):
        display_name = None
    assert display_name == "UninitializedAgent"
    with agent_spans._agent_span(agent, (), {}):
        pass
    (span,) = spans_of(exporter)
    assert span.name == "invoke_agent UninitializedAgent"
    assert span.attributes["gen_ai.agent.name"] == "UninitializedAgent"
