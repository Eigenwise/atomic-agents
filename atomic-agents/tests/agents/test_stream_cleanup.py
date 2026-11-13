"""Closing an agent stream also closes the partial-response iterator it owns."""

from unittest.mock import Mock

import instructor
import pytest

from atomic_agents import AgentConfig, AtomicAgent, BasicChatInputSchema, BasicChatOutputSchema


def make_agent(response_stream, asynchronous=False):
    client = Mock(spec=instructor.AsyncInstructor if asynchronous else instructor.Instructor)
    client.mode = instructor.Mode.TOOLS
    client.chat.completions.create_partial.return_value = response_stream
    return AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](AgentConfig(client=client, model="test"))


@pytest.mark.parametrize("operation", ["close", "throw"])
def test_sync_stream_closes_partial_iterator_when_consumer_stops(operation):
    closed = []

    def responses():
        try:
            yield BasicChatOutputSchema(chat_message="partial")
            yield BasicChatOutputSchema(chat_message="complete")
        finally:
            closed.append(True)

    inner = responses()
    agent = make_agent(inner)
    stream = agent.run_stream(BasicChatInputSchema(chat_message="question"))
    assert next(stream).chat_message == "partial"
    if operation == "close":
        stream.close()
    else:
        error = RuntimeError("consumer stopped")
        with pytest.raises(RuntimeError) as caught:
            stream.throw(error)
        assert caught.value is error
    assert closed == [True]
    assert [message.role for message in agent.history.history] == ["user"]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["aclose", "athrow"])
async def test_async_stream_closes_partial_iterator_when_consumer_stops(operation):
    closed = []

    async def responses():
        try:
            yield BasicChatOutputSchema(chat_message="partial")
            yield BasicChatOutputSchema(chat_message="complete")
        finally:
            closed.append(True)

    inner = responses()
    agent = make_agent(inner, asynchronous=True)
    stream = agent.run_async_stream(BasicChatInputSchema(chat_message="question"))
    assert (await anext(stream)).chat_message == "partial"
    if operation == "aclose":
        await stream.aclose()
    else:
        error = RuntimeError("consumer stopped")
        with pytest.raises(RuntimeError) as caught:
            await stream.athrow(error)
        assert caught.value is error
    assert closed == [True]
    assert [message.role for message in agent.history.history] == ["user"]


def test_sync_stream_still_accepts_iterators_without_close():
    final = BasicChatOutputSchema(chat_message="complete")
    agent = make_agent(iter([final]))
    assert list(agent.run_stream()) == [final]
    assert agent.history.history[-1].content == final


@pytest.mark.asyncio
async def test_async_stream_finishes_and_records_complete_response():
    final = BasicChatOutputSchema(chat_message="complete")
    closed = []

    async def responses():
        try:
            yield final
        finally:
            closed.append(True)

    agent = make_agent(responses(), asynchronous=True)
    assert [response async for response in agent.run_async_stream()] == [final]
    assert closed == [True]
    assert agent.history.history[-1].content == final


@pytest.mark.asyncio
async def test_async_stream_still_accepts_iterators_without_aclose():
    final = BasicChatOutputSchema(chat_message="complete")

    class Responses:
        def __init__(self):
            self.items = iter([final])

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self.items)
            except StopIteration:
                raise StopAsyncIteration

    agent = make_agent(Responses(), asynchronous=True)
    assert [response async for response in agent.run_async_stream()] == [final]
    assert agent.history.history[-1].content == final
