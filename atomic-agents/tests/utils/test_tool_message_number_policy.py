"""Tool arguments must retain Pydantic's JSON representation of non-finite numbers."""

import json

import pytest
from pydantic import BaseModel, ConfigDict, field_serializer

from atomic_agents.utils import format_tool_message


def reject_json_constant(value):
    raise ValueError(f"Nonstandard JSON constant: {value}")


@pytest.mark.parametrize("policy", [None, "null", "strings", "constants"])
@pytest.mark.parametrize("value,token", [(float("inf"), "Infinity"), (float("-inf"), "-Infinity"), (float("nan"), "NaN")])
def test_nonfinite_argument_uses_the_configured_json_encoding(policy, value, token):
    class Arguments(BaseModel):
        model_config = ConfigDict(**({"ser_json_inf_nan": policy} if policy else {}))
        value: float

    result = format_tool_message(Arguments(value=value), "call-1")
    expected = token if policy == "constants" else f'"{token}"' if policy == "strings" else "null"

    assert result["function"]["arguments"] == f'{{"value": {expected}}}'
    assert result["id"] == "call-1"
    if policy != "constants":
        parsed = json.loads(result["function"]["arguments"], parse_constant=reject_json_constant)
        assert parsed == {"value": token if policy == "strings" else None}


@pytest.mark.parametrize("policy", [None, "null", "strings", "constants"])
def test_finite_arguments_keep_the_existing_format_and_value(policy):
    class Arguments(BaseModel):
        model_config = ConfigDict(**({"ser_json_inf_nan": policy} if policy else {}))
        value: float
        label: str

    result = format_tool_message(Arguments(value=1.25, label="雪"))

    assert result["function"]["arguments"] == '{"value": 1.25, "label": "\\u96ea"}'
    assert json.loads(result["function"]["arguments"], parse_constant=reject_json_constant) == {"value": 1.25, "label": "雪"}


def test_nested_models_keep_their_own_number_policy():
    class Child(BaseModel):
        model_config = ConfigDict(ser_json_inf_nan="strings")
        value: float

    class Arguments(BaseModel):
        child: Child
        values: list[float]

    arguments = Arguments(child=Child(value=float("inf")), values=[float("nan"), float("-inf")])
    result = format_tool_message(arguments)

    assert json.loads(result["function"]["arguments"], parse_constant=reject_json_constant) == {
        "child": {"value": "Infinity"},
        "values": [None, None],
    }


def test_nonfinite_values_from_json_only_serializers_follow_the_policy():
    class Arguments(BaseModel):
        model_config = ConfigDict(ser_json_inf_nan="strings")
        value: int

        @field_serializer("value", when_used="json")
        def serialize_value(self, value):
            return float("nan")

    result = format_tool_message(Arguments(value=1))

    assert json.loads(result["function"]["arguments"], parse_constant=reject_json_constant) == {"value": "NaN"}
