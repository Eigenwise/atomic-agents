"""JSON budgets include the exact schema and instructions prepared by Instructor."""

import json

import httpx
import instructor
import openai
import pytest
from instructor.processing.response import handle_response_model

from atomic_agents import AgentConfig, AtomicAgent, BasicChatInputSchema, BasicChatOutputSchema
from atomic_agents.utils.token_counter import get_token_counter


def make_agent(mode, role, requests):
    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
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
            },
        )

    sdk = openai.OpenAI(api_key="test-only", http_client=httpx.Client(transport=httpx.MockTransport(respond)))
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
        AgentConfig(client=instructor.from_openai(sdk, mode=mode), model="gpt-4o-mini", system_role=role)
    )
    return agent, sdk


@pytest.mark.parametrize("mode", [instructor.Mode.JSON, instructor.Mode.MD_JSON])
@pytest.mark.parametrize("role", ["system", "developer", None])
def test_count_matches_sdk_messages_without_changing_history(mode, role):
    requests = []
    agent, sdk = make_agent(mode, role, requests)
    try:
        agent.history.add_message("user", BasicChatInputSchema(chat_message="first part"))
        agent.history.add_message("user", BasicChatInputSchema(chat_message="第二部分"))
        before = agent.history.dump()
        counted = agent.get_context_token_count()
        assert agent.history.dump() == before
        assert agent.get_context_token_count() == counted
        assert agent.run().chat_message == "ready"
        wire_tokens = get_token_counter().count_messages(agent.model, requests[0]["messages"])
        assert counted.total == wire_tokens
        assert counted.system_prompt + counted.history + counted.tools == counted.total
        assert counted.utilization == counted.total / counted.max_tokens
    finally:
        sdk.close()


@pytest.mark.parametrize("mode", [instructor.Mode.JSON, instructor.Mode.MD_JSON])
@pytest.mark.parametrize("role", ["system", "developer", None])
def test_budget_rejects_a_request_one_token_below_its_prepared_size(mode, role):
    requests = []
    agent, sdk = make_agent(mode, role, requests)
    try:
        agent.history.add_message("user", BasicChatInputSchema(chat_message="current request"))
        agent._prepare_messages()
        _, prepared = handle_response_model(
            response_model=agent.output_schema, mode=mode, messages=agent.messages, model=agent.model
        )
        wire_tokens = get_token_counter().count_messages(agent.model, prepared["messages"])
        agent.max_context_tokens = wire_tokens - 1
        with pytest.raises(ValueError, match="max_context_tokens"):
            agent.run()
        assert requests == []
        agent.max_context_tokens = wire_tokens
        assert agent.run().chat_message == "ready"
        assert get_token_counter().count_messages(agent.model, requests[0]["messages"]) == wire_tokens
    finally:
        sdk.close()


@pytest.mark.parametrize("mode", [instructor.Mode.JSON, instructor.Mode.MD_JSON])
def test_empty_json_history_has_no_history_charge(mode):
    agent, sdk = make_agent(mode, None, [])
    try:
        counted = agent.get_context_token_count()
        assert counted.history == 0
        assert counted.tools == 0
        assert counted.system_prompt == counted.total
    finally:
        sdk.close()
