"""Provider failures must survive cleanup when history retains no messages."""

import httpx
import instructor
import openai
import pytest
from instructor.core import InstructorRetryException

from atomic_agents import AgentConfig, AtomicAgent, BasicChatInputSchema, BasicChatOutputSchema
from atomic_agents.context import ChatHistory


def _reject_request(status):
    def respond(request):
        return httpx.Response(
            status,
            json={"error": {"message": "Provider request rejected", "type": "request_error", "code": str(status)}},
        )

    return httpx.MockTransport(respond)


def _agent(raw_client, capacity):
    return AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
        AgentConfig(
            client=instructor.from_openai(raw_client),
            history=ChatHistory(max_messages=capacity),
            model_api_parameters={"max_retries": 1},
        )
    )


@pytest.mark.parametrize("capacity", [0, 1, None])
@pytest.mark.parametrize("status", [401, 429])
@pytest.mark.parametrize("stream", [False, True])
def test_sync_provider_failure_survives_history_cleanup(capacity, status, stream):
    with openai.OpenAI(
        api_key="test", max_retries=0, http_client=httpx.Client(transport=_reject_request(status))
    ) as raw_client:
        agent = _agent(raw_client, capacity)
        user_input = BasicChatInputSchema(chat_message="A request that the provider rejects")

        with pytest.raises(InstructorRetryException, match="Provider request rejected"):
            if stream:
                list(agent.run_stream(user_input))
            else:
                agent.run(user_input)

    assert agent.history.get_history() == []


@pytest.mark.asyncio
@pytest.mark.parametrize("capacity", [0, 1, None])
@pytest.mark.parametrize("status", [401, 429])
@pytest.mark.parametrize("stream", [False, True])
async def test_async_provider_failure_survives_history_cleanup(capacity, status, stream):
    async with openai.AsyncOpenAI(
        api_key="test", max_retries=0, http_client=httpx.AsyncClient(transport=_reject_request(status))
    ) as raw_client:
        agent = _agent(raw_client, capacity)
        user_input = BasicChatInputSchema(chat_message="A request that the provider rejects")

        with pytest.raises(InstructorRetryException, match="Provider request rejected"):
            if stream:
                async for _ in agent.run_async_stream(user_input):
                    pass
            else:
                await agent.run_async(user_input)

    assert agent.history.get_history() == []
