"""Explicit empty collection models remain real inputs and final responses."""

import json

import httpx
import instructor
import openai
import pytest
from pydantic import Field

from atomic_agents import AgentConfig, AtomicAgent, BaseIOSchema


class QueueInput(BaseIOSchema):
    """Replace a queue with the requested jobs, including clearing it."""

    jobs: list[str] = Field(description="The complete new queue.")

    def __len__(self):
        return len(self.jobs)


class MatchesOutput(BaseIOSchema):
    """The matching jobs, with an empty result representing no matches."""

    matches: list[str] = Field(description="Matching job names.")

    def __bool__(self):
        return bool(self.matches)


def transport_for(matches, requests, empty_stream=False):
    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        arguments = json.dumps({"matches": matches})
        call = {"id": "call-1", "type": "function", "function": {"name": "MatchesOutput", "arguments": arguments}}
        response = {"id": "completion", "created": 0, "model": "test"}
        if body.get("stream"):
            response.update(
                object="chat.completion.chunk",
                choices=[{"index": 0, "delta": {"tool_calls": [{"index": 0, **call}]}, "finish_reason": None}],
            )
            frames = "" if empty_stream else "data: " + json.dumps(response) + "\n\n"
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=(frames + "data: [DONE]\n\n").encode()
            )
        response.update(
            object="chat.completion",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": None, "tool_calls": [call]},
                    "finish_reason": "tool_calls",
                }
            ],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )
        return httpx.Response(200, json=response)

    return httpx.MockTransport(respond)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["run", "run_async", "run_stream", "run_async_stream"])
@pytest.mark.parametrize("jobs,matches", [([], ["found"]), (["job"], []), (None, ["found"]), (["job"], ["found"])])
async def test_agent_preserves_explicit_models_regardless_of_truth_value(method, jobs, matches):
    requests = []
    asynchronous = "async" in method
    transport = transport_for(matches, requests)
    raw = (
        openai.AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=transport))
        if asynchronous
        else openai.OpenAI(api_key="test", http_client=httpx.Client(transport=transport))
    )
    try:
        agent = AtomicAgent[QueueInput, MatchesOutput](AgentConfig(client=instructor.from_openai(raw), model="test"))
        user_input = None if jobs is None else QueueInput(jobs=jobs)
        if method == "run":
            result = agent.run(user_input)
        elif method == "run_async":
            result = await agent.run_async(user_input)
        elif method == "run_stream":
            stream = agent.run_stream(user_input)
            partials = []
            while True:
                try:
                    partials.append(next(stream))
                except StopIteration as done:
                    result = done.value
                    break
            assert partials[-1].matches == matches
        else:
            partials = [part async for part in agent.run_async_stream(user_input)]
            assert partials[-1].matches == matches
            result = None  # Async generators return their final model through history.

        assert len(requests) == 1
        user_messages = [message for message in requests[0]["messages"] if message["role"] == "user"]
        assert [json.loads(message["content"]) for message in user_messages] == ([] if jobs is None else [{"jobs": jobs}])
        assert agent.current_user_input is user_input
        expected_roles = (["user"] if jobs is not None else []) + ["assistant"]
        assert [message.role for message in agent.history.history] == expected_roles
        assert agent.history.history[-1].content.matches == matches
        if jobs is not None:
            assert agent.history.history[0].content is user_input
            assert agent.history.history[0].turn_id == agent.history.history[-1].turn_id
        if method != "run_async_stream":
            assert isinstance(result, MatchesOutput)
            assert result.matches == matches
    finally:
        if asynchronous:
            await raw.close()
        else:
            raw.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_empty_response_stream_still_adds_no_assistant_message(asynchronous):
    requests = []
    transport = transport_for([], requests, empty_stream=True)
    raw = (
        openai.AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=transport))
        if asynchronous
        else openai.OpenAI(api_key="test", http_client=httpx.Client(transport=transport))
    )
    try:
        agent = AtomicAgent[QueueInput, MatchesOutput](AgentConfig(client=instructor.from_openai(raw), model="test"))
        user_input = QueueInput(jobs=["job"])
        partials = (
            [part async for part in agent.run_async_stream(user_input)] if asynchronous else list(agent.run_stream(user_input))
        )
        assert partials == []
        assert [message.role for message in agent.history.history] == ["user"]
    finally:
        if asynchronous:
            await raw.close()
        else:
            raw.close()
