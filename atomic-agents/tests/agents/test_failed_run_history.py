from unittest.mock import Mock

import instructor
import pytest

from atomic_agents import AgentConfig, AtomicAgent, BasicChatInputSchema, BasicChatOutputSchema
from atomic_agents.context import ChatHistory


class CompletionFailure(RuntimeError):
    pass


def _sync_client(create=None, create_partial=None):
    client = Mock(spec=instructor.Instructor)
    client.chat = Mock()
    client.chat.completions = Mock()
    client.chat.completions.create = Mock(side_effect=create)
    client.chat.completions.create_partial = Mock(side_effect=create_partial)
    return client


def _async_client(create=None, create_partial=None):
    client = Mock(spec=instructor.AsyncInstructor)
    client.chat = Mock()
    client.chat.completions = Mock()
    client.chat.completions.create = create
    client.chat.completions.create_partial = create_partial
    return client


def _agent(client, history):
    return AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
        AgentConfig(client=client, model="test-model", history=history)
    )


def _completed_history():
    history = ChatHistory()
    history.initialize_turn()
    history.add_message("user", BasicChatInputSchema(chat_message="earlier question"))
    history.add_message("assistant", BasicChatOutputSchema(chat_message="earlier answer"))
    return history


def _roles(history):
    return [message["role"] for message in history.get_history()]


def _raise_failure(*args, **kwargs):
    raise CompletionFailure("provider unavailable")


def _failing_stream(*args, **kwargs):
    yield BasicChatOutputSchema(chat_message="partial")
    raise CompletionFailure("stream dropped")


async def _raise_failure_async(*args, **kwargs):
    raise CompletionFailure("provider unavailable")


async def _failing_stream_async(*args, **kwargs):
    yield BasicChatOutputSchema(chat_message="partial")
    raise CompletionFailure("stream dropped")


def _run(agent, user_input):
    return agent.run(user_input)


def _run_stream(agent, user_input):
    return list(agent.run_stream(user_input))


SYNC_RUNS = [
    pytest.param(_run, {"create": _raise_failure}, id="run"),
    pytest.param(_run_stream, {"create_partial": _raise_failure}, id="run_stream-request"),
    pytest.param(_run_stream, {"create_partial": _failing_stream}, id="run_stream-mid-stream"),
]


@pytest.mark.parametrize("execute, client_kwargs", SYNC_RUNS)
def test_failed_sync_run_does_not_keep_the_user_turn(execute, client_kwargs):
    history = _completed_history()
    turn_before = history.get_current_turn_id()
    agent = _agent(_sync_client(**client_kwargs), history)

    with pytest.raises(CompletionFailure):
        execute(agent, BasicChatInputSchema(chat_message="new question"))

    assert _roles(history) == ["user", "assistant"]
    assert history.get_current_turn_id() == turn_before


@pytest.mark.parametrize("execute, client_kwargs", SYNC_RUNS)
def test_retry_after_failed_sync_run_sends_one_user_message(execute, client_kwargs):
    history = _completed_history()
    agent = _agent(_sync_client(**client_kwargs), history)
    user_input = BasicChatInputSchema(chat_message="new question")

    with pytest.raises(CompletionFailure):
        execute(agent, user_input)

    agent.client.chat.completions.create = Mock(return_value=BasicChatOutputSchema(chat_message="answer"))
    agent.run(user_input)

    sent_roles = [message["role"] for message in agent.client.chat.completions.create.call_args.kwargs["messages"]]
    assert sent_roles == ["system", "user", "assistant", "user"]
    assert _roles(history) == ["user", "assistant", "user", "assistant"]


def test_failed_run_on_empty_history_leaves_it_empty():
    history = ChatHistory()
    agent = _agent(_sync_client(create=_raise_failure), history)

    with pytest.raises(CompletionFailure):
        agent.run(BasicChatInputSchema(chat_message="first question"))

    assert history.get_history() == []
    assert history.get_current_turn_id() is None


def test_failed_run_without_input_keeps_the_existing_turn():
    history = _completed_history()
    history.initialize_turn()
    history.add_message("user", BasicChatInputSchema(chat_message="pending question"))
    agent = _agent(_sync_client(create=_raise_failure), history)

    with pytest.raises(CompletionFailure):
        agent.run()

    assert _roles(history) == ["user", "assistant", "user"]


ASYNC_RUNS = [
    pytest.param("run_async", {"create": _raise_failure_async}, id="run_async"),
    pytest.param("run_async_stream", {"create_partial": _failing_stream_async}, id="run_async_stream-mid-stream"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("method, client_kwargs", ASYNC_RUNS)
async def test_failed_async_run_does_not_keep_the_user_turn(method, client_kwargs):
    history = _completed_history()
    turn_before = history.get_current_turn_id()
    agent = _agent(_async_client(**client_kwargs), history)
    user_input = BasicChatInputSchema(chat_message="new question")

    with pytest.raises(CompletionFailure):
        if method == "run_async":
            await agent.run_async(user_input)
        else:
            async for _ in agent.run_async_stream(user_input):
                pass

    assert _roles(history) == ["user", "assistant"]
    assert history.get_current_turn_id() == turn_before
