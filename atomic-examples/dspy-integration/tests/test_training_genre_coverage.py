"""Check the training data actually supplied by both benchmark stages."""

import importlib
from collections import Counter
from pathlib import Path

import dspy
import pytest


@pytest.fixture
def example(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("dspy_integration")


def assert_balanced_examples(examples, genres, expected_size):
    assert len(examples) == expected_size
    assert all(isinstance(example, dspy.Example) for example in examples)
    assert all(set(example.inputs().keys()) == {"review"} for example in examples)
    counts = Counter(example.genre for example in examples)
    print("Training genre counts:", dict(counts))
    assert set(counts) == set(genres)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_stage1_optimizer_receives_all_six_genres(example, monkeypatch):
    stage = importlib.import_module("dspy_integration.stages.stage1_dspy")
    captured = []
    predictor = dspy.Predict("review -> genre")

    def capture_training_data(self, student, *, trainset):
        captured.extend(trainset)
        return student

    # Stop at the optimizer boundary; the stage builds real DSPy Examples.
    monkeypatch.setattr(dspy.BootstrapFewShot, "compile", capture_training_data)
    assert stage._run_optimization(None, predictor) is predictor
    assert_balanced_examples(captured, example.GENRES, 30)


def test_stage3_validated_examples_cover_all_six_genres(example):
    stage = importlib.import_module("dspy_integration.stages.stage3_combined")
    examples = stage._create_training_set()
    assert_balanced_examples(examples, example.GENRES, 40)
    for item in examples:
        example.MovieReviewInput.model_validate(dict(item.inputs()))
        example.MovieGenreOutput.model_validate(dict(item.labels()))


def test_reordering_preserves_the_complete_training_catalog(example):
    dataset = importlib.import_module("dspy_integration.domain.datasets")
    source = (
        dataset._ACTION_EXAMPLES
        + dataset._COMEDY_EXAMPLES
        + dataset._DRAMA_EXAMPLES
        + dataset._HORROR_EXAMPLES
        + dataset._SCIFI_EXAMPLES
        + dataset._ROMANCE_EXAMPLES
    )
    assert len(dataset.TRAINING_DATASET) == len(source) == 60
    assert Counter(map(id, dataset.TRAINING_DATASET)) == Counter(map(id, source))
    assert Counter(item["genre"] for item in dataset.TRAINING_DATASET) == dict.fromkeys(example.GENRES, 10)
