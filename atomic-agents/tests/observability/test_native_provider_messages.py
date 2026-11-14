"""Native provider blocks keep their meaning in opt-in chat attempt messages."""

import json
from dataclasses import replace
from types import SimpleNamespace

import httpx
import instructor
import openai
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from atomic_agents import AgentConfig, AtomicAgent, BaseIOSchema
from atomic_agents.observability import TracingConfig, installation
from atomic_agents.observability.message_payloads import write_messages


def capture(value, direction="input", **options):
    exported = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exported))
    with provider.get_tracer(__name__).start_as_current_span("native") as span:
        write_messages(span, value, direction, replace(TracingConfig(capture_messages=True), **options))
    attributes = dict(exported.get_finished_spans()[0].attributes)
    provider.shutdown()
    return attributes


def decoded(value, direction="input"):
    return json.loads(capture(value, direction)[f"gen_ai.{direction}.messages"])


def tool_use():
    return {"type": "tool_use", "id": "call_lookup", "name": "lookup", "input": {"city": "東京", "count": 0}}


def canonical_call():
    return {"type": "tool_call", "id": "call_lookup", "name": "lookup", "arguments": {"city": "東京", "count": 0}}


class PrivateBlock:
    def __init__(self, kind):
        self.type = kind

    def __getattr__(self, name):
        raise AssertionError(f"private {self.type} block accessed: {name}")


@pytest.mark.parametrize("objects", [False, True])
def test_anthropic_output_and_replayed_input_keep_block_order(objects):
    blocks = [
        {"type": "text", "text": "Before"},
        tool_use(),
        PrivateBlock("thinking"),
        PrivateBlock("redacted_thinking"),
        PrivateBlock("image"),
        {"type": "text", "text": "After"},
    ]
    if objects:
        blocks = [SimpleNamespace(**block) if isinstance(block, dict) else block for block in blocks]
    response = {"type": "message", "role": "assistant", "content": blocks}
    expected = [
        {
            "role": "assistant",
            "parts": [
                {"type": "text", "content": "Before"},
                canonical_call(),
                {"type": "text", "content": "After"},
            ],
        }
    ]
    assert decoded(response, "output") == expected
    assert decoded([response]) == expected


@pytest.mark.parametrize(
    "result,expected",
    [
        ("sunny", "sunny"),
        ("", ""),
        (
            [{"type": "text", "text": "first"}, PrivateBlock("image"), {"type": "text", "text": "second"}],
            [{"type": "text", "content": "first"}, {"type": "text", "content": "second"}],
        ),
    ],
)
def test_anthropic_tool_results_keep_ids_and_only_text(result, expected):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_lookup", "content": result, "is_error": True},
                {"type": "text", "text": "Continue"},
            ],
        }
    ]
    assert decoded(messages) == [
        {
            "role": "user",
            "parts": [
                {"type": "tool_call_response", "id": "call_lookup", "response": expected},
                {"type": "text", "content": "Continue"},
            ],
        }
    ]


@pytest.mark.parametrize("objects", [False, True])
def test_responses_output_is_one_generation_with_ordered_text_and_calls(objects):
    items = [
        PrivateBlock("reasoning"),
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "Before"},
                PrivateBlock("refusal"),
            ],
        },
        {
            "type": "function_call",
            "id": "fc_internal",
            "call_id": "call_lookup",
            "name": "lookup",
            "arguments": '{"city":"東京","count":0}',
        },
        PrivateBlock("image_generation_call"),
        {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "After"}]},
    ]
    if objects:
        items = [SimpleNamespace(**item) if isinstance(item, dict) else item for item in items]
    value = {"object": "response", "output": items}
    if objects:
        value = SimpleNamespace(**value)
    assert decoded(value, "output") == [
        {
            "role": "assistant",
            "parts": [
                {"type": "text", "content": "Before"},
                canonical_call(),
                {"type": "text", "content": "After"},
            ],
        }
    ]


@pytest.mark.parametrize("kind", ["text", "input_text", "output_text"])
def test_native_text_block_spellings(kind):
    assert decoded([{"role": "user", "content": [{"type": kind, "text": "question"}]}]) == [
        {"role": "user", "parts": [{"type": "text", "content": "question"}]},
    ]


@pytest.mark.parametrize(
    "value,direction",
    [
        ({"type": "message", "role": "assistant", "content": [PrivateBlock("thinking")]}, "output"),
        ({"object": "response", "output": [PrivateBlock("reasoning")]}, "output"),
        ({"type": "message_delta", "role": "assistant", "content": [tool_use()]}, "output"),
        ([{"role": "user", "content": [{"type": "tool_result", "content": [PrivateBlock("image")]}]}], "input"),
        ([{"role": "assistant", "content": [{"type": "tool_use", "name": "", "input": {}}]}], "input"),
        ([{"role": "assistant", "content": [{"type": "tool_use", "name": "bad", "input": "not an object"}]}], "input"),
    ],
)
def test_unsupported_or_incomplete_native_blocks_are_omitted(value, direction):
    assert capture(value, direction) == {}


@pytest.mark.parametrize("direction", ["input", "output"])
def test_native_blocks_keep_complete_attribute_length_limit(direction):
    response = {"type": "message", "role": "assistant", "content": [tool_use()]}
    value = [response] if direction == "input" else response
    key = f"gen_ai.{direction}.messages"
    encoded = capture(value, direction)[key]
    assert capture(value, direction, max_attribute_length=len(encoded))[key] == encoded
    assert capture(value, direction, max_attribute_length=len(encoded) - 1) == {}


class Question(BaseIOSchema):
    """A native provider question."""

    question: str


class Answer(BaseIOSchema):
    """The validated provider answer."""

    answer: str


@pytest.fixture
def exporter():
    exported = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exported))
    installation.install(tracer_provider=provider, config=TracingConfig(capture_messages=True, max_attribute_length=20000))
    yield exported
    installation.uninstall()
    provider.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["anthropic", "responses"])
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("visibility", ["visible", "hide_inputs", "hide_outputs", "disabled"])
async def test_real_provider_retry_captures_native_messages(exporter, provider, asynchronous, visibility):
    config = TracingConfig(capture_messages=visibility != "disabled", max_attribute_length=20000)
    if visibility.startswith("hide_"):
        config = replace(config, **{visibility: True})
    installation.install(config=config)
    requests = []

    def respond(request):
        body = json.loads(request.content)
        number = len(requests)
        requests.append(body)
        answer = {"invalid": "retry"} if number == 0 else "42"
        if provider == "anthropic":
            assert request.url.path == "/v1/messages"
            content = [{"type": "tool_use", "id": f"call_{number}", "name": "Answer", "input": {"answer": answer}}]
            payload = {
                "id": f"msg_{number}",
                "type": "message",
                "role": "assistant",
                "model": "offline",
                "content": content,
                "stop_reason": "tool_use",
                "stop_sequence": None,
                "usage": {"input_tokens": 3, "output_tokens": 2},
            }
        else:
            assert request.url.path == "/v1/responses"
            content = [
                {
                    "type": "function_call",
                    "id": f"fc_{number}",
                    "call_id": f"call_{number}",
                    "name": "Answer",
                    "arguments": json.dumps({"answer": answer}),
                    "status": "completed",
                }
            ]
            payload = {
                "id": f"resp_{number}",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "offline",
                "output": content,
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            }
        return httpx.Response(200, json=payload)

    http_client = (httpx.AsyncClient if asynchronous else httpx.Client)(transport=httpx.MockTransport(respond))
    if provider == "anthropic":
        anthropic = pytest.importorskip("anthropic")
        sdk = (anthropic.AsyncAnthropic if asynchronous else anthropic.Anthropic)(api_key="test", http_client=http_client)
        client = instructor.from_anthropic(sdk, mode=instructor.Mode.ANTHROPIC_TOOLS)
    else:
        sdk = (openai.AsyncOpenAI if asynchronous else openai.OpenAI)(api_key="test", http_client=http_client)
        client = instructor.from_openai(sdk, mode=instructor.Mode.RESPONSES_TOOLS)
    agent = AtomicAgent[Question, Answer](
        AgentConfig(
            client=client,
            model="offline",
            model_api_parameters={"max_retries": 2, **({"max_tokens": 64} if provider == "anthropic" else {})},
        )
    )
    try:
        result = (
            await agent.run_async(Question(question="Native question"))
            if asynchronous
            else agent.run(Question(question="Native question"))
        )
        assert result.answer == "42"
    finally:
        if asynchronous:
            await sdk.close()
        else:
            sdk.close()
    first, second, parent = exporter.get_finished_spans()
    assert len(requests) == 2
    assert parent.attributes["atomic_agents.llm.attempts"] == 2
    for index, attempt in enumerate([first, second]):
        assert attempt.parent.span_id == parent.context.span_id
        input_key, output_key = "gen_ai.input.messages", "gen_ai.output.messages"
        assert (input_key in attempt.attributes) == (visibility == "visible")
        assert (output_key in attempt.attributes) == (visibility in {"visible", "hide_inputs"})
        if output_key in attempt.attributes:
            expected = {"invalid": "retry"} if index == 0 else "42"
            assert json.loads(attempt.attributes[output_key]) == [
                {
                    "role": "assistant",
                    "parts": [
                        {
                            "type": "tool_call",
                            "id": f"call_{index}",
                            "name": "Answer",
                            "arguments": {"answer": expected},
                        }
                    ],
                }
            ]
        if input_key in attempt.attributes:
            messages = json.loads(attempt.attributes[input_key])
            assert messages[0]["role"] == "system"
            if provider == "anthropic":
                assert messages[0]["parts"] == [
                    {"type": "text", "content": item["text"]} for item in requests[index]["system"]
                ]
                if index == 1:
                    parts = [part for message in messages for part in message["parts"]]
                    assert any(part.get("id") == "call_0" and part["type"] == "tool_call" for part in parts)
                    assert any(part.get("id") == "call_0" and part["type"] == "tool_call_response" for part in parts)
    assert "gen_ai.output.messages" not in parent.attributes
