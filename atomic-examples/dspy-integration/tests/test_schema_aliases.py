"""Keep DSPy's canonical field names across Pydantic alias policies."""

import importlib
from inspect import signature
from pathlib import Path

import dspy
import pytest
from dspy.utils.dummies import DummyLM
from pydantic import AliasChoices, AliasPath, ConfigDict, Field, ValidationError

from atomic_agents import BaseIOSchema


def native_lm(answer):
    """Use each SDK version's native offline response fixture."""
    return DummyLM([answer] if "adapter" not in signature(DummyLM).parameters else [{"answer": answer}])


@pytest.fixture
def bridge(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("dspy_integration.bridge")


class PlainInput(BaseIOSchema):
    """Text to process."""

    text: str


class PlainOutput(BaseIOSchema):
    """The processed answer."""

    answer: str


@pytest.mark.parametrize(
    "field",
    [
        Field(alias="requestText"),
        Field(validation_alias=AliasChoices("requestText", "prompt")),
        Field(validation_alias=AliasPath("request", "text")),
    ],
    ids=["alias", "choices", "path"],
)
def test_signature_field_names_are_valid_inputs(bridge, field):
    class Input(BaseIOSchema):
        """Aliased input."""

        text: str = field

    module = bridge.DSPyAtomicModule(Input, PlainOutput, use_chain_of_thought=False)
    assert set(module.signature.input_fields) == {"text"}
    with dspy.context(lm=native_lm("accepted")):
        assert module(text="canonical input").answer == "accepted"


@pytest.mark.parametrize(
    "field",
    [
        Field(alias="answerText"),
        Field(validation_alias=AliasChoices("answerText", "result")),
        Field(validation_alias=AliasPath("response", "text")),
    ],
    ids=["alias", "choices", "path"],
)
def test_prediction_field_names_validate_as_output(bridge, field):
    class Output(BaseIOSchema):
        """Aliased output."""

        answer: str = field

    module = bridge.DSPyAtomicModule(PlainInput, Output, use_chain_of_thought=False)
    with dspy.context(lm=native_lm("validated")):
        output = module.run_validated(text="question")
    assert isinstance(output, Output)
    assert output.answer == "validated"


def test_output_alias_does_not_consume_another_prediction_field(bridge):
    class Output(BaseIOSchema):
        """Canonical fields must remain distinct from external aliases."""

        answer: str = Field(alias="label")
        label: str = Field(alias="wireLabel")

    answer = (
        "correct answer\nLabel: category"
        if "adapter" not in signature(DummyLM).parameters
        else {"answer": "correct answer", "label": "category"}
    )
    module = bridge.DSPyAtomicModule(PlainInput, Output, use_chain_of_thought=False)
    with dspy.context(lm=DummyLM([answer])):
        output = module.run_validated(text="question")
    assert output.answer == "correct answer"
    assert output.label == "category"


def test_serialization_aliases_do_not_rename_predictor_inputs(bridge):
    class Input(BaseIOSchema):
        """An external serialized request uses another field name."""

        model_config = ConfigDict(serialize_by_alias=True)
        text: str = Field(serialization_alias="requestText")

    module = bridge.DSPyAtomicModule(Input, PlainOutput, use_chain_of_thought=False)
    assert Input(text="question").model_dump() == {"requestText": "question"}
    lm = native_lm("retained")
    with dspy.context(lm=lm):
        assert module.run_validated(text="sentinel-input-alias-42").answer == "retained"
    assert "sentinel-input-alias-42" in str(lm.history)


@pytest.mark.parametrize("input_alias,output_alias", [(True, False), (False, True), (True, True)])
def test_training_examples_retain_signature_names(bridge, input_alias, output_alias):
    class Input(BaseIOSchema):
        """Training input."""

        model_config = ConfigDict(serialize_by_alias=input_alias)
        text: str = Field(serialization_alias="requestText")

    class Output(BaseIOSchema):
        """Training output."""

        model_config = ConfigDict(serialize_by_alias=output_alias)
        answer: str = Field(serialization_alias="answerText")

    example = bridge.create_dspy_example(Input, Output, {"text": "question"}, {"answer": "expected"})
    assert dict(example.inputs().items()) == {"text": "question"}
    assert dict(example.labels().items()) == {"answer": "expected"}


def test_explicit_input_alias_remains_accepted(bridge):
    class Input(BaseIOSchema):
        """Input accepted through its public alias."""

        text: str = Field(alias="requestText")

    module = bridge.DSPyAtomicModule(Input, PlainOutput, use_chain_of_thought=False)
    with dspy.context(lm=native_lm("alias accepted")):
        assert module.run_validated(requestText="question").answer == "alias accepted"


def test_plain_schemas_and_examples_remain_unchanged(bridge):
    module = bridge.DSPyAtomicModule(PlainInput, PlainOutput, use_chain_of_thought=False)
    with dspy.context(lm=native_lm("ordinary")):
        assert module.run_validated(text="question") == PlainOutput(answer="ordinary")
    example = bridge.create_dspy_example(PlainInput, PlainOutput, {"text": "question"}, {"answer": "ordinary"})
    assert dict(example.items()) == {"text": "question", "answer": "ordinary"}


def test_output_constraints_still_reject_invalid_predictions(bridge):
    class Output(BaseIOSchema):
        """A sufficiently long answer is required."""

        answer: str = Field(alias="answerText", min_length=10)

    module = bridge.DSPyAtomicModule(PlainInput, Output, use_chain_of_thought=False)
    with dspy.context(lm=native_lm("short")):
        with pytest.raises(ValidationError):
            module.run_validated(text="question")
