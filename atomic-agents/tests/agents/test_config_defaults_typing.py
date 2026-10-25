from typing import TypedDict

import instructor
from openai import OpenAI

from atomic_agents import AgentConfig


class ModelSettings(TypedDict):
    model: str


def test_optional_agent_settings_have_defaults() -> None:
    settings: ModelSettings = {"model": "test-model"}
    with OpenAI(api_key="synthetic-test-key") as provider_client:
        client = instructor.from_openai(provider_client)
        direct_configuration = AgentConfig(client=client, model="test-model")
        unpacked_configuration = AgentConfig(client=client, **settings)

    assert (direct_configuration.model_api_parameters, direct_configuration.max_context_tokens) == (None, None)
    assert (unpacked_configuration.model_api_parameters, unpacked_configuration.max_context_tokens) == (None, None)
