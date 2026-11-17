"""Valid JSON whitespace may arrive before the first object token in a stream."""

import json

import httpx
import instructor
import openai
import pytest
from instructor.dsl.partial import Partial

from atomic_agents import AgentConfig, AtomicAgent, BasicChatInputSchema, BasicChatOutputSchema


def response_for_chunks(chunks):
    def respond(request):
        assert json.loads(request.content)["stream"] is True
        frames = []
        for piece in chunks:
            frames.append(
                "data: "
                + json.dumps(
                    {
                        "id": "whitespace-stream",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": "test",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call-1",
                                            "type": "function",
                                            "function": {"name": "BasicChatOutputSchema", "arguments": piece},
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                + "\n\n"
            )
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=("".join(frames) + "data: [DONE]\n\n").encode()
        )

    return respond


def assert_complete_history(agent, partials):
    assert partials[-1].chat_message == "hello  世界 "
    assert agent.history.history[-1].role == "assistant"
    assert type(agent.history.history[-1].content) is BasicChatOutputSchema
    assert agent.history.history[-1].content.chat_message == "hello  世界 "


@pytest.mark.parametrize("prefix", ["", " ", "\t", "\r\n", " \t\r\n"])
def test_sync_stream_accepts_json_whitespace_before_the_object(prefix):
    chunks = [prefix, "", '{"chat_message":"hello  ', "世界 ", '"}', "\n"]
    with openai.OpenAI(
        api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(response_for_chunks(chunks)))
    ) as raw_client:
        agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
            AgentConfig(client=instructor.from_openai(raw_client), model="test")
        )

        partials = list(agent.run_stream(BasicChatInputSchema(chat_message="hello")))

    assert_complete_history(agent, partials)


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["", " ", "\t", "\r\n", " \t\r\n"])
async def test_async_stream_accepts_json_whitespace_before_the_object(prefix):
    chunks = [prefix, "", '{"chat_message":"hello  ', "世界 ", '"}', "\n"]
    async with openai.AsyncOpenAI(
        api_key="test", http_client=httpx.AsyncClient(transport=httpx.MockTransport(response_for_chunks(chunks)))
    ) as raw_client:
        agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
            AgentConfig(client=instructor.from_openai(raw_client), model="test")
        )

        partials = [part async for part in agent.run_async_stream(BasicChatInputSchema(chat_message="hello"))]

    assert_complete_history(agent, partials)


@pytest.mark.parametrize("prefix", ["\u00a0", "\v", "\ufeff", "garbage"])
def test_non_json_whitespace_is_not_silently_accepted(prefix):
    with pytest.raises(ValueError):
        list(Partial[BasicChatOutputSchema].model_from_chunks([prefix, '{"chat_message":"hello"}']))


@pytest.mark.parametrize("strict", [None, True])
def test_separate_whitespace_chunks_preserve_partial_validation(strict):
    chunks = [" ", "\t", "\r\n", '{"chat_message":"hel', 'lo"}']

    partials = list(Partial[BasicChatOutputSchema].model_from_chunks(chunks, strict=strict))

    assert [part.chat_message for part in partials] == [None, None, None, "hel", "hello"]
