"""A request must send the same dynamic system prompt that its budget measured."""

import json

import httpx
import instructor
import openai
import pytest

from atomic_agents.agents.atomic_agent import AgentConfig, AtomicAgent, BasicChatInputSchema, BasicChatOutputSchema
from atomic_agents.context import BaseDynamicContextProvider, ChatHistory, SystemPromptGenerator
from atomic_agents.utils.token_counter import get_token_counter


class ChangingContext(BaseDynamicContextProvider):
    def __init__(self):
        super().__init__("Changing context")
        self.calls = 0
        self.text = "short captured context"

    def get_info(self):
        self.calls += 1
        text = self.text
        self.text = "subsequent context " * 5000
        return text


def completion_transport(requests, provider):
    def handle(request):
        body = json.loads(request.content)
        requests.append((body, provider.calls))
        if body.get("stream"):
            chunks = [
                {
                    "id": "completion-1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": body["model"],
                    "choices": [{"index": 0, "delta": {"content": '{"chat_message":"ready"}'}, "finish_reason": None}],
                },
                {
                    "id": "completion-1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": body["model"],
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]
            stream = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream)
        return httpx.Response(
            200,
            json={
                "id": "completion-1",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"chat_message":"ready"}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    return httpx.MockTransport(handle)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["run", "run_stream", "run_async", "run_async_stream"])
async def test_dynamic_context_is_captured_before_budgeting_and_sending(method):
    provider = ChangingContext()
    requests = []
    transport = completion_transport(requests, provider)
    if "async" in method:
        sdk = openai.AsyncOpenAI(api_key="test-only", http_client=httpx.AsyncClient(transport=transport))
    else:
        sdk = openai.OpenAI(api_key="test-only", http_client=httpx.Client(transport=transport))
    client = instructor.from_openai(sdk, mode=instructor.Mode.JSON)
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
        AgentConfig(
            client=client,
            model="gpt-4o-mini",
            max_context_tokens=1000,
            system_prompt_generator=SystemPromptGenerator(context_providers={"changing": provider}),
        )
    )
    user_input = BasicChatInputSchema(chat_message="Please answer.")
    try:
        if method == "run":
            assert agent.run(user_input).chat_message == "ready"
        elif method == "run_stream":
            assert list(agent.run_stream(user_input))[-1].chat_message == "ready"
        elif method == "run_async":
            assert (await agent.run_async(user_input)).chat_message == "ready"
        else:
            responses = [response async for response in agent.run_async_stream(user_input)]
            assert responses[-1].chat_message == "ready"
        assert len(requests) == 1
        body, calls_at_send = requests[0]
        assert get_token_counter().count_messages(agent.model, body["messages"]) <= agent.max_context_tokens
        assert "short captured context" in body["messages"][0]["content"]
        assert calls_at_send == 1
    finally:
        if "async" in method:
            await sdk.close()
        else:
            sdk.close()


def test_trimming_old_turns_reuses_the_prompt_and_later_counts_refresh_it():
    provider = ChangingContext()
    requests = []
    history = ChatHistory()
    for index in range(6):
        history.initialize_turn()
        history.add_message("user", BasicChatInputSchema(chat_message=f"old-{index} " * 250))
    old_turns = {message.turn_id for message in history.history}
    sdk = openai.OpenAI(api_key="test-only", http_client=httpx.Client(transport=completion_transport(requests, provider)))
    try:
        agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
            AgentConfig(
                client=instructor.from_openai(sdk, mode=instructor.Mode.JSON),
                model="gpt-4o-mini",
                max_context_tokens=1000,
                history=history,
                system_prompt_generator=SystemPromptGenerator(context_providers={"changing": provider}),
            )
        )
        agent.run(BasicChatInputSchema(chat_message="newest request"))
        assert requests[0][1] == 1
        assert not old_turns.issubset({message.turn_id for message in history.history})
        assert "newest request" in requests[0][0]["messages"][-1]["content"]
        calls_before = provider.calls
        assert agent.get_context_token_count().total > agent.max_context_tokens
        assert provider.calls == calls_before + 1
    finally:
        sdk.close()


def test_a_rejected_budget_does_not_leave_a_stale_prompt_snapshot():
    provider = ChangingContext()
    provider.text = "oversized first context " * 5000
    requests = []
    sdk = openai.OpenAI(api_key="test-only", http_client=httpx.Client(transport=completion_transport(requests, provider)))
    try:
        agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
            AgentConfig(
                client=instructor.from_openai(sdk, mode=instructor.Mode.JSON),
                model="gpt-4o-mini",
                max_context_tokens=1000,
                system_prompt_generator=SystemPromptGenerator(context_providers={"changing": provider}),
            )
        )
        with pytest.raises(ValueError, match="max_context_tokens"):
            agent.run(BasicChatInputSchema(chat_message="current request"))
        assert requests == []
        provider.text = "small after rejection"
        assert agent.get_context_token_count().total < agent.max_context_tokens
        provider.text = "small next request"
        assert agent.run().chat_message == "ready"
        assert "small next request" in requests[0][0]["messages"][0]["content"]
    finally:
        sdk.close()


def test_standalone_counts_and_unbudgeted_requests_keep_fresh_context():
    provider = ChangingContext()
    requests = []
    sdk = openai.OpenAI(api_key="test-only", http_client=httpx.Client(transport=completion_transport(requests, provider)))
    try:
        agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
            AgentConfig(
                client=instructor.from_openai(sdk, mode=instructor.Mode.JSON),
                model="gpt-4o-mini",
                system_prompt_generator=SystemPromptGenerator(context_providers={"changing": provider}),
            )
        )
        assert agent.get_context_token_count().total < 1000
        assert agent.get_context_token_count().total > 1000
        assert provider.calls == 2
        provider.text = "fresh unbudgeted request"
        assert agent.run(BasicChatInputSchema(chat_message="hello")).chat_message == "ready"
        assert "fresh unbudgeted request" in requests[0][0]["messages"][0]["content"]
        assert requests[0][1] == 3
    finally:
        sdk.close()
