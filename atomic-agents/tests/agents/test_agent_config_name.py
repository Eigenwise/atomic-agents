import instructor
from openai import OpenAI

from atomic_agents import AgentConfig, AtomicAgent, BaseIOSchema


def test_agent_name_defaults_to_none() -> None:
    with OpenAI(api_key="synthetic-test-key") as provider_client:
        client = instructor.from_openai(provider_client)
        configuration = AgentConfig(client=client)
        agent = AtomicAgent[BaseIOSchema, BaseIOSchema](configuration)
    assert configuration.name is None
    assert agent.name is None


def test_explicit_agent_name_is_stored() -> None:
    with OpenAI(api_key="synthetic-test-key") as provider_client:
        client = instructor.from_openai(provider_client)
        configuration = AgentConfig(client=client, name="orchestrator_planning")
        agent = AtomicAgent[BaseIOSchema, BaseIOSchema](configuration)
    assert configuration.name == "orchestrator_planning"
    assert agent.name == "orchestrator_planning"
