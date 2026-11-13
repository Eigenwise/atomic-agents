"""Preserve validation input types across stream and history round trips."""

import json

import httpx
import instructor
import openai
import pytest
from pydantic import BaseModel, Field, Json, computed_field

from atomic_agents import AgentConfig, AtomicAgent, BaseIOSchema, BasicChatInputSchema
from atomic_agents.context import ChatHistory


class JsonOutput(BaseIOSchema):
    """Return a JSON-encoded list as a typed field."""

    payload: Json[list[int]] = Field(alias="jsonPayload")


class NestedPayload(BaseModel):
    payload: Json[dict[str, list[int]]]


class NestedJsonOutput(BaseIOSchema):
    """Return JSON fields inside another model."""

    result: NestedPayload


class ComputedOutput(BaseIOSchema):
    """Return numbers with a derived total and reject unknown inputs."""

    model_config = {"extra": "forbid"}
    values: list[int]

    @computed_field
    @property
    def total(self) -> int:
        return sum(self.values)


class PlainOutput(BaseIOSchema):
    """Ordinary aliased output remains compatible."""

    message: str = Field(alias="reply")


CASES = [
    pytest.param(JsonOutput, {"jsonPayload": "[1,2,3]"}, id="json-field"),
    pytest.param(NestedJsonOutput, {"result": {"payload": '{"items":[1,2,3]}'}}, id="nested-json-field"),
    pytest.param(ComputedOutput, {"values": [1, 2, 3]}, id="computed-field"),
    pytest.param(PlainOutput, {"reply": "hello"}, id="plain-alias-control"),
]


def transport_for(arguments, requests):
    """Exercise actual OpenAI SSE decoding and Instructor partial construction."""

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        function = {"name": body["tools"][0]["function"]["name"], "arguments": json.dumps(arguments)}
        tool_call = {"id": "call-1", "type": "function", "function": function}
        if body.get("stream"):
            chunk = {
                "id": "chunk",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "test-model",
                "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, **tool_call}]}, "finish_reason": None}],
            }
            frames = "data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=frames.encode())
        return httpx.Response(
            200,
            json={
                "id": "completion",
                "object": "chat.completion",
                "created": 0,
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": None, "tool_calls": [tool_call]},
                        "finish_reason": "tool_calls",
                    }
                ],
            },
        )

    return httpx.MockTransport(respond)


def assert_complete_history(agent, schema, arguments):
    assert [message.role for message in agent.history.history] == ["user", "assistant"]
    complete = agent.history.history[-1].content
    assert isinstance(complete, schema)
    expected = schema.model_validate(arguments)
    for field_name in schema.model_fields:
        assert getattr(complete, field_name) == getattr(expected, field_name)
    if schema is ComputedOutput:
        assert complete.total == 6
    assert agent.messages[-1] == agent.history.get_history()[-1]
    return complete


@pytest.mark.parametrize("schema,arguments", CASES)
@pytest.mark.parametrize("streaming", [False, True], ids=["normal", "stream"])
def test_sync_output_roundtrip(schema, arguments, streaming):
    requests = []
    with openai.OpenAI(api_key="test", http_client=httpx.Client(transport=transport_for(arguments, requests))) as raw:
        agent = AtomicAgent[BasicChatInputSchema, schema](AgentConfig(client=instructor.from_openai(raw), model="test-model"))
        user_input = BasicChatInputSchema(chat_message="Return the requested structured result")
        if streaming:
            stream = agent.run_stream(user_input)
            partials = []
            while True:
                try:
                    partials.append(next(stream))
                except StopIteration as stop:
                    result = stop.value
                    break
            assert partials
        else:
            result = agent.run(user_input)
        complete = assert_complete_history(agent, schema, arguments)
        assert result == complete
        if streaming:
            assert type(result) is schema
        assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("schema,arguments", CASES)
@pytest.mark.parametrize("streaming", [False, True], ids=["normal", "stream"])
async def test_async_output_roundtrip(schema, arguments, streaming):
    requests = []
    async with openai.AsyncOpenAI(
        api_key="test", http_client=httpx.AsyncClient(transport=transport_for(arguments, requests))
    ) as raw:
        agent = AtomicAgent[BasicChatInputSchema, schema](AgentConfig(client=instructor.from_openai(raw), model="test-model"))
        user_input = BasicChatInputSchema(chat_message="Return the requested structured result")
        if streaming:
            partials = [partial async for partial in agent.run_async_stream(user_input)]
            assert partials
        else:
            result = await agent.run_async(user_input)
        complete = assert_complete_history(agent, schema, arguments)
        if streaming:
            assert type(complete) is schema
        if not streaming:
            assert result == complete
        assert len(requests) == 1


@pytest.mark.parametrize("schema,arguments", CASES)
def test_history_dump_load_roundtrip(schema, arguments):
    history = ChatHistory(max_messages=4)
    history.add_message("user", BasicChatInputSchema(chat_message="Keep this conversation"))
    expected = schema.model_validate(arguments)
    history.add_message("assistant", expected)

    serialized = history.dump()
    restored = ChatHistory()
    restored.load(serialized)

    assert restored.max_messages == 4
    assert restored.current_turn_id == history.current_turn_id
    assert [message.turn_id for message in restored.history] == [message.turn_id for message in history.history]
    assert [message.role for message in restored.history] == ["user", "assistant"]
    assert type(restored.history[-1].content) is schema
    assert restored.history[-1].content == expected
    assert restored.dump() == serialized
