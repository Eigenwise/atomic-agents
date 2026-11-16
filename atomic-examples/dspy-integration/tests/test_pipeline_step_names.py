"""Reject names that overwrite pipeline stages or framework methods."""

import importlib
from pathlib import Path

import dspy
import pytest


@pytest.fixture
def pipeline_type(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("dspy_integration.bridge").DSPyAtomicPipeline


class Step(dspy.Module):
    def __init__(self, label, calls):
        super().__init__()
        self.label = label
        self.calls = calls

    def forward(self, **kwargs):
        self.calls.append(self.label)
        return dspy.Prediction(answer=self.label)


def test_duplicate_names_cannot_silently_replace_a_stage(pipeline_type):
    calls = []
    with pytest.raises(ValueError, match="classify"):
        pipeline_type([("classify", Step("first", calls)), ("classify", Step("second", calls))])
    assert calls == []


@pytest.mark.parametrize("name", ["forward", "step_names", "predictors"])
def test_step_names_cannot_overwrite_pipeline_attributes(pipeline_type, name):
    with pytest.raises(ValueError, match=name):
        pipeline_type([(name, Step("stage", []))])


def test_distinct_steps_run_once_in_the_requested_order(pipeline_type):
    calls = []
    pipeline = pipeline_type([("first", Step("one", calls)), ("second", Step("two", calls))])
    result = pipeline(text="input")
    assert calls == ["one", "two"]
    assert list(result) == ["first", "second"]
    assert result["first"].answer == "one"
    assert result["second"].answer == "two"
    assert callable(pipeline.predictors)


def test_explicitly_reusing_a_module_under_distinct_names_is_allowed(pipeline_type):
    calls = []
    module = Step("shared", calls)
    pipeline = pipeline_type([("before", module), ("after", module)])
    assert list(pipeline(text="input")) == ["before", "after"]
    assert calls == ["shared", "shared"]


def test_single_step_remains_valid(pipeline_type):
    calls = []
    assert pipeline_type([("classify", Step("one", calls))])(text="input")["classify"].answer == "one"
    assert calls == ["one"]


def test_empty_pipeline_remains_valid(pipeline_type):
    assert pipeline_type([])(text="input") == {}
