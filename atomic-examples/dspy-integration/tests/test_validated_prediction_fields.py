"""Read prediction data without confusing field names with DSPy methods."""

import importlib
from pathlib import Path

import dspy
from pydantic import Field, create_model
import pytest

from atomic_agents import BaseIOSchema


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("dspy_integration.bridge")


class Input(BaseIOSchema):
    """Input for a validated prediction."""

    text: str


class Emit(dspy.Module):
    def __init__(self, fields):
        super().__init__()
        self.fields = fields

    def forward(self, **kwargs):
        return dspy.Prediction(**self.fields)


@pytest.mark.parametrize(
    "name,annotation,value",
    [("items", list[str], ["a", "b"]), ("keys", str, "business keys"), ("get", int, 0), ("completions", str, "completed")],
)
def test_validated_output_reads_method_named_data(bridge, name, annotation, value):
    output_schema = create_model("Output", __base__=BaseIOSchema, __doc__="Prediction output.", **{name: (annotation, ...)})
    module = bridge.DSPyAtomicModule(Input, output_schema, use_chain_of_thought=False)
    module.predictor = Emit({name: value})

    output = module.run_validated(text="request")

    assert output.model_dump() == {name: value}


def test_missing_data_uses_the_schema_default_instead_of_a_method(bridge):
    class Output(BaseIOSchema):
        """A prediction may omit the optional items field."""

        items: list[str] = Field(default_factory=list)

    module = bridge.DSPyAtomicModule(Input, Output, use_chain_of_thought=False)
    module.predictor = Emit({})
    assert module.run_validated(text="request").items == []


def test_unrelated_prediction_data_remains_excluded(bridge):
    class Output(BaseIOSchema):
        """Only the requested output is validated."""

        answer: str

    module = bridge.DSPyAtomicModule(Input, Output, use_chain_of_thought=False)
    module.predictor = Emit({"answer": "kept", "reasoning": "internal"})
    assert module.run_validated(text="request").model_dump() == {"answer": "kept"}


def test_plain_predictions_remain_valid(bridge):
    class Output(BaseIOSchema):
        """A plain answer."""

        answer: str

    module = bridge.DSPyAtomicModule(Input, Output, use_chain_of_thought=False)
    module.predictor = Emit({"answer": "same result"})
    assert module.run_validated(text="request").answer == "same result"
