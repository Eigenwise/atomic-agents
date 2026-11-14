"""Parametrizing a documented schema must keep its description and validation."""

import asyncio
import inspect
import json
from typing import Generic, TypeVar

import httpx
import instructor
import pytest
from openai import AsyncOpenAI, OpenAI
from pydantic import ConfigDict, Field, TypeAdapter, ValidationError

from atomic_agents import AgentConfig, AtomicAgent, BaseIOSchema


@pytest.mark.parametrize(
    "value_type,value,json_type", [(int, 42, "integer"), (str, "answer", "string"), (float, 2.5, "number")]
)
def test_pep695_specialization_keeps_description_and_concrete_type(value_type, value, json_type):
    class Result[T](BaseIOSchema):
        """A reusable typed result.

        Its payload carries the validated result.
        """

        payload: T

    specialized = Result[value_type]
    assert Result[value_type] is specialized
    assert specialized.__doc__ == Result.__doc__
    assert specialized.model_validate_json(json.dumps({"payload": value})).payload == value
    for mode in ("validation", "serialization"):
        schema = specialized.model_json_schema(mode=mode)
        assert schema["description"] == inspect.cleandoc(Result.__doc__)
        assert schema["properties"]["payload"]["type"] == json_type
        assert schema["title"] == specialized.__name__


def test_legacy_generic_syntax_and_validation_are_preserved():
    item_type = TypeVar("item_type")

    class Result(BaseIOSchema, Generic[item_type]):
        """A legacy generic result."""

        payload: item_type

    specialized = Result[int]
    assert specialized(payload="42").payload == 42
    with pytest.raises(ValidationError):
        specialized(payload="not an integer")
    assert specialized.model_json_schema()["description"] == "A legacy generic result."


def test_nested_specializations_keep_both_descriptions_and_shapes():
    class Result[T](BaseIOSchema):
        """A result with a typed payload."""

        payload: T

    class Batch[T](BaseIOSchema):
        """A batch of typed results."""

        items: list[Result[T]]

    model = Batch[int].model_validate({"items": [{"payload": 42}]})
    assert type(model.items[0]) is Result[int]
    assert model.model_dump() == {"items": [{"payload": 42}]}
    schema = Batch[int].model_json_schema()
    assert schema["description"] == "A batch of typed results."
    nested = schema["$defs"]["Result_int_"]
    assert nested["description"] == "A result with a typed payload."
    assert nested["properties"]["payload"]["type"] == "integer"


def test_partial_specialization_and_documented_generic_subclass():
    left = TypeVar("left")
    right = TypeVar("right")

    class Pair(BaseIOSchema, Generic[left, right]):
        """A pair of typed values."""

        first: left
        second: right

    partial = Pair[int, right]
    assert partial.__doc__ == Pair.__doc__
    assert partial[str](first=1, second="two").model_dump() == {"first": 1, "second": "two"}

    class NamedPair(partial, Generic[right]):
        """An integer paired with a named value."""

    named = NamedPair[str]
    assert named.__doc__ == NamedPair.__doc__
    assert named.model_json_schema()["description"] == "An integer paired with a named value."
    assert named(first=1, second="two").second == "two"


def test_recursive_generic_schema_retains_description_in_definitions():
    item_type = TypeVar("item_type")

    class Node(BaseIOSchema, Generic[item_type]):
        """A typed tree node."""

        value: item_type
        children: list["Node[item_type]"] = Field(default_factory=list)

    model = Node[int].model_validate({"value": 1, "children": [{"value": 2}]})
    assert model.children[0].value == 2
    schema = Node[int].model_json_schema()
    definition = schema["$defs"]["Node_int_"]
    assert definition["description"] == "A typed tree node."
    assert definition["properties"]["value"]["type"] == "integer"


def test_explicit_json_schema_description_and_title_still_win():
    class Result[T](BaseIOSchema):
        """The reusable Python description."""

        model_config = ConfigDict(title="Configured result", json_schema_extra={"description": "The wire description."})
        payload: T

    schema = Result[int].model_json_schema()
    assert schema["title"] == "Configured result"
    assert schema["description"] == "The wire description."
    assert Result[int].__doc__ == "The reusable Python description."


def test_custom_parametrized_name_still_has_the_origin_description():
    class Result[T](BaseIOSchema):
        """A named typed result."""

        payload: T

        @classmethod
        def model_parametrized_name(cls, params):
            return params[0].__name__.capitalize() + "Result"

    assert Result[int].model_json_schema()["title"] == "IntResult"
    assert Result[int].model_json_schema()["description"] == "A named typed result."


def test_type_adapter_includes_the_description_in_nested_definitions():
    class Result[T](BaseIOSchema):
        """A result nested in an external schema."""

        payload: T

    adapter = TypeAdapter(list[Result[int]])
    assert adapter.validate_python([{"payload": 42}])[0].payload == 42
    assert adapter.json_schema()["$defs"]["Result_int_"]["description"] == Result.__doc__


@pytest.mark.parametrize("description", [None, "", " \n  "])
def test_user_defined_undocumented_subclasses_still_fail(description):
    with pytest.raises(ValueError, match="Undocumented must have a non-empty docstring"):
        type("Undocumented", (BaseIOSchema,), {"__doc__": description, "__module__": __name__})


def test_subclassing_a_specialization_still_requires_a_new_docstring():
    class Result[T](BaseIOSchema):
        """An original generic schema."""

        payload: T

    specialized = Result[int]
    with pytest.raises(ValueError, match="Undocumented must have a non-empty docstring"):

        class Undocumented(specialized):
            pass

    class Documented(specialized):
        """A more specific documented schema."""

    assert Documented.model_json_schema()["description"] == "A more specific documented schema."


def test_undocumented_generic_template_is_not_exempt():
    with pytest.raises(ValueError, match="Undocumented must have a non-empty docstring"):

        class Undocumented[T](BaseIOSchema):
            payload: T


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("mode", [instructor.Mode.TOOLS, instructor.Mode.JSON], ids=["tools", "json"])
def test_specialized_agent_schema_reaches_real_sdk_and_validates_retries(asynchronous, mode):
    class Question[T](BaseIOSchema):
        """A typed request."""

        query: T

    class Result[T](BaseIOSchema):
        """A validated typed answer."""

        payload: T

        @classmethod
        def model_parametrized_name(cls, params):
            return params[0].__name__.capitalize() + "Result"

    incoming, outgoing = Question[str], Result[int]
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        if mode == instructor.Mode.TOOLS:
            function = body["tools"][0]["function"]
            assert function["name"] == "IntResult"
            assert function["description"] == "A validated typed answer."
            assert function["parameters"]["properties"]["payload"]["type"] == "integer"
        else:
            assert "A validated typed answer." in json.dumps(body["messages"])
        payload = json.dumps({"payload": "invalid" if len(requests) == 1 else 42})
        message = {"role": "assistant", "content": payload}
        finish_reason = "stop"
        if mode == instructor.Mode.TOOLS:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_result", "type": "function", "function": {"name": "IntResult", "arguments": payload}}
                ],
            }
            finish_reason = "tool_calls"
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-local",
                "object": "chat.completion",
                "created": 1,
                "model": "fixture-model",
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            },
        )

    def configure(native):
        client = instructor.from_openai(native, mode=mode)
        return AtomicAgent[incoming, outgoing](
            AgentConfig(client=client, model="fixture-model", model_api_parameters={"max_retries": 2})
        )

    if asynchronous:

        async def execute():
            async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
                async with AsyncOpenAI(api_key="local-fixture", http_client=http) as native:
                    agent = configure(native)
                    return agent, await agent.run_async(incoming(query="Return an integer"))

        agent, result = asyncio.run(execute())
    else:
        with httpx.Client(transport=httpx.MockTransport(respond)) as http:
            with OpenAI(api_key="local-fixture", http_client=http) as native:
                agent = configure(native)
                result = agent.run(incoming(query="Return an integer"))

    assert isinstance(result, outgoing)
    assert result.payload == 42
    assert len(requests) == 2
    assert [message.role for message in agent.history.history] == ["user", "assistant"]
    assert agent.history.history[-1].content is result
