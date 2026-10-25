"""Concrete schema bindings survive reusable generic agent subclasses."""

import json

import httpx
import instructor
import openai
import pytest

from atomic_agents import AgentConfig, AtomicAgent, BaseIOSchema, BasicChatInputSchema, BasicChatOutputSchema


class StageInput(BaseIOSchema):
    """Input to a composed stage."""

    text: str


class FirstOutput(BaseIOSchema):
    """Output from the first stage."""

    answer: str


class SecondOutput(BaseIOSchema):
    """Output from the second stage."""

    answer: str


class ConcreteAgent(AtomicAgent[StageInput, FirstOutput]):
    def __init__(self, config):
        super().__init__(config)


class ConcreteChild(ConcreteAgent):
    pass


class OutputTemplate[T: BaseIOSchema](AtomicAgent[StageInput, T]):
    def __init__(self, config):
        super().__init__(config)


class TwoSchemaTemplate[I: BaseIOSchema, O: BaseIOSchema](AtomicAgent[I, O]):
    pass


class ReorderedTemplate[O: BaseIOSchema, I: BaseIOSchema](AtomicAgent[I, O]):
    pass


class ForwardTemplate[T: BaseIOSchema](OutputTemplate[T]):
    pass


class FinalAgent(ForwardTemplate[FirstOutput]):
    pass


class UntypedAgent(AtomicAgent):
    pass


@pytest.fixture
def offline_client():
    requests = []

    def respond(request):
        body = json.loads(request.content)
        function = body["tools"][0]["function"]
        assert function["name"] in {"FirstOutput", "SecondOutput"}
        assert set(function["parameters"]["properties"]) == {"answer"}
        requests.append(body)
        return httpx.Response(
            200,
            json={
                "id": "schema-binding-test",
                "object": "chat.completion",
                "created": 0,
                "model": "offline-fixture",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_test",
                                    "type": "function",
                                    "function": {
                                        "name": function["name"],
                                        "arguments": json.dumps({"answer": "literal fixture result"}),
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
        )

    with openai.OpenAI(
        api_key="test-placeholder",
        base_url="http://localhost/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as provider:
        yield instructor.from_openai(provider), requests


@pytest.mark.parametrize(
    "factory, expected_output",
    [
        pytest.param(AtomicAgent[StageInput, FirstOutput], FirstOutput, id="direct"),
        pytest.param(ConcreteAgent, FirstOutput, id="concrete-custom-constructor"),
        pytest.param(ConcreteChild, FirstOutput, id="concrete-descendant"),
        pytest.param(OutputTemplate[FirstOutput], FirstOutput, id="single-output-parameter"),
        pytest.param(OutputTemplate[SecondOutput], SecondOutput, id="other-output-specialization"),
        pytest.param(TwoSchemaTemplate[StageInput, FirstOutput], FirstOutput, id="two-parameters"),
        pytest.param(ReorderedTemplate[FirstOutput, StageInput], FirstOutput, id="reordered-parameters"),
        pytest.param(ForwardTemplate[FirstOutput], FirstOutput, id="forwarded-parameter"),
        pytest.param(FinalAgent, FirstOutput, id="concrete-template-descendant"),
    ],
)
def test_specialized_agents_use_concrete_schemas(offline_client, factory, expected_output):
    client, requests = offline_client
    agent = factory(AgentConfig(client=client, model="offline-fixture"))

    assert agent.input_schema is StageInput
    assert agent.output_schema is expected_output
    result = agent.run(StageInput(text="ordinary stage input"))

    assert isinstance(result, expected_output)
    assert result.answer == "literal fixture result"
    assert len(requests) == 1
    assert agent.history.history[-1].content is result


def test_specializations_keep_their_own_bindings(offline_client):
    client, _ = offline_client
    config = AgentConfig(client=client)
    first = OutputTemplate[FirstOutput](config)
    second = OutputTemplate[SecondOutput](config)

    assert first.output_schema is FirstOutput
    assert second.output_schema is SecondOutput
    assert first.output_schema is FirstOutput
    assert FinalAgent(config).output_schema is FirstOutput
    assert second.output_schema is SecondOutput


@pytest.mark.parametrize("factory", [AtomicAgent, UntypedAgent])
def test_untyped_agents_keep_default_schemas(offline_client, factory):
    client, _ = offline_client
    agent = factory(AgentConfig(client=client))

    assert agent.input_schema is BasicChatInputSchema
    assert agent.output_schema is BasicChatOutputSchema
