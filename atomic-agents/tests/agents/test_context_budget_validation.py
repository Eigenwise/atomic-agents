import httpx
import instructor
import openai
import pytest
from pydantic import ValidationError

from atomic_agents import AgentConfig, AtomicAgent, BaseIOSchema
from atomic_agents.context import ChatHistory


class Message(BaseIOSchema):
    """A message for checking context budget validation."""

    text: str


@pytest.fixture
def client():
    def reject_request(request):
        pytest.fail("Invalid budgets must be rejected before provider requests")

    with httpx.Client(transport=httpx.MockTransport(reject_request)) as http_client:
        yield instructor.from_openai(openai.OpenAI(api_key="test", http_client=http_client), mode=instructor.Mode.JSON)


@pytest.mark.parametrize("limit", [0, -1, -1000, "0", "-3", False])
def test_context_budget_rejects_nonpositive_values_during_configuration(client, limit):
    with pytest.raises(ValidationError, match="max_context_tokens"):
        AgentConfig(client=client, max_context_tokens=limit)


@pytest.mark.parametrize("limit", [None, 1, 1000, "2000"])
def test_disabled_and_positive_budgets_keep_existing_validation(client, limit):
    config = AgentConfig(client=client, max_context_tokens=limit)

    assert config.max_context_tokens == (int(limit) if limit is not None else None)


@pytest.mark.parametrize("limit", [0, -1])
def test_bad_budget_cannot_trim_existing_conversation_before_failing(client, limit):
    history = ChatHistory()
    for index in range(3):
        history.initialize_turn()
        history.add_message("user", Message(text=f"Saved question {index}"))
        history.add_message("assistant", Message(text=f"Saved answer {index}"))
    snapshot = history.dump()

    with pytest.raises(ValueError) as raised:
        agent = AtomicAgent[Message, Message](
            AgentConfig(client=client, history=history, model="gpt-4o-mini", max_context_tokens=limit)
        )
        agent.run(Message(text="A new question"))

    assert isinstance(raised.value, ValidationError)
    assert history.dump() == snapshot
