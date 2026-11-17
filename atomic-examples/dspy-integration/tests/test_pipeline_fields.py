"""Exercise pipeline handoffs with real DSPy modules and predictions."""

from pathlib import Path
import sys

import dspy
from dspy.utils import DummyLM
from pydantic import BaseModel
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atomic_agents import BaseIOSchema  # noqa: E402
from dspy_integration.bridge import DSPyAtomicModule, DSPyAtomicPipeline  # noqa: E402


class Emit(dspy.Module):
    def __init__(self, prediction):
        super().__init__()
        self.prediction = prediction

    def forward(self, **kwargs):
        return self.prediction


class Capture(dspy.Module):
    def __init__(self):
        super().__init__()
        self.received = None

    def forward(self, **kwargs):
        self.received = kwargs
        return dspy.Prediction(answer="done")


class Record(BaseModel):
    text: str


@pytest.mark.parametrize(
    "fields",
    [
        {"text": "Hello", "count": 2},
        {"zero": 0, "flag": False, "empty": "", "missing": None},
        {"records": [Record(text="nested")], "metadata": {"source": "document"}},
        {"items": ["a", "b"], "keys": "data", "completions": "business value"},
        {"answer": "kept", "dspy_internal": "excluded"},
    ],
)
def test_handoff_preserves_prediction_fields_and_values(fields):
    prediction = dspy.Prediction(**fields)
    capture = Capture()
    pipeline = DSPyAtomicPipeline([("produce", Emit(prediction)), ("consume", capture)])

    results = pipeline(seed="input")

    expected = {key: value for key, value in fields.items() if not key.startswith("dspy_")}
    assert capture.received == expected
    for key, value in expected.items():
        assert capture.received[key] is value
    assert results["produce"] is prediction
    assert results["consume"].answer == "done"


class Question(BaseIOSchema):
    """Question to answer."""

    question: str


class Answer(BaseIOSchema):
    """Answer to a question."""

    answer: str


class Summary(BaseIOSchema):
    """Summary of an answer."""

    summary: str


def test_native_predictors_receive_previous_validated_output():
    first = DSPyAtomicModule(Question, Answer, use_chain_of_thought=False)
    second = DSPyAtomicModule(Answer, Summary, use_chain_of_thought=False)
    pipeline = DSPyAtomicPipeline([("answer_question", first), ("summarize", second)])
    # DSPy 2.5's dummy speaks completion text; newer versions speak chat fields.
    answers = (
        ["Paris", "The answer is Paris."]
        if hasattr(DummyLM, "basic_request")
        else [
            {"answer": "Paris"},
            {"summary": "The answer is Paris."},
        ]
    )
    lm = DummyLM(answers)

    with dspy.context(lm=lm):
        results = pipeline(question="What is France's capital?")

    assert results["answer_question"].answer == "Paris"
    assert results["summarize"].summary == "The answer is Paris."
    request = lm.history[1]
    assert "Paris" in (request.get("prompt") or request["messages"][-1]["content"])


def test_each_handoff_uses_only_the_immediately_previous_prediction():
    first = Emit(dspy.Prediction(text="first"))
    second, third = Capture(), Capture()
    pipeline = DSPyAtomicPipeline([("first", first), ("second", second), ("third", third)])

    results = pipeline(initial="request")

    assert second.received == {"text": "first"}
    assert third.received == {"answer": "done"}
    assert list(results) == ["first", "second", "third"]


def test_single_step_preserves_the_original_prediction():
    prediction = dspy.Prediction(answer="single")
    assert DSPyAtomicPipeline([("only", Emit(prediction))])(question="input") == {"only": prediction}


def test_empty_pipeline_returns_no_results():
    assert DSPyAtomicPipeline([])(question="input") == {}
