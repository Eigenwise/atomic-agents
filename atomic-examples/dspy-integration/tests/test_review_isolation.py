"""Check benchmark independence at the native Instructor/OpenAI request boundary."""

import importlib
import json
from contextlib import contextmanager
from pathlib import Path

import httpx
import instructor
import openai
import pytest

from atomic_agents import AgentConfig, AtomicAgent
from atomic_agents.context import ChatHistory


@pytest.fixture
def stage(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("dspy_integration.stages.stage2_atomic")


@contextmanager
def native_agent(stage, history=None, fail_first=False):
    requests = []

    def respond(request):
        assert request.url.path == "/v1/chat/completions"
        requests.append(json.loads(request.content))
        if fail_first and len(requests) == 1:
            return httpx.Response(500, json={"error": {"message": "offline failure", "type": "server_error"}})
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-offline",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call-offline",
                                    "type": "function",
                                    "function": {
                                        "name": "MovieGenreOutput",
                                        "arguments": '{"genre":"drama","confidence":0.75,"reasoning":"offline response"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        client = instructor.from_openai(openai.OpenAI(api_key="offline-test", http_client=http_client, max_retries=0))
        agent = AtomicAgent[stage.MovieReviewInput, stage.MovieGenreOutput](
            AgentConfig(
                client=client,
                model="gpt-4o-mini",
                model_api_parameters={"max_retries": 1},
                system_prompt_generator=stage._create_system_prompt(),
                history=history,
            )
        )
        yield agent, requests


def test_each_benchmark_review_has_its_own_request_context(stage):
    with native_agent(stage) as (agent, requests):
        result, predictions = stage._evaluate_agent(agent)

    assert result.total == len(predictions) == len(requests) == 30
    assert all(prediction["genre"] == "drama" for prediction in predictions)
    message_counts = [len(request["messages"]) for request in requests]
    print("Serialized request message counts:", message_counts, "total:", sum(message_counts))
    assert message_counts == [2] * len(requests)
    for request, example in zip(requests, stage.TEST_DATASET):
        assert request["messages"][1:] == [
            {"role": "user", "content": stage.MovieReviewInput(review=example["review"]).model_dump_json()}
        ]


def test_configured_initial_history_is_preserved_for_each_review(stage):
    history = ChatHistory()
    history.add_message("user", stage.MovieReviewInput(review="A fixed calibration example."))
    history.add_message("assistant", stage.MovieGenreOutput(genre="drama", confidence=1.0, reasoning="Calibration."))
    initial_messages = history.get_history()
    with native_agent(stage, history=history) as (agent, requests):
        for example in stage.TEST_DATASET[:2]:
            assert stage._get_single_prediction(agent, example)["genre"] == "drama"

    for request, example in zip(requests, stage.TEST_DATASET[:2]):
        assert request["messages"][1:] == initial_messages + [
            {"role": "user", "content": stage.MovieReviewInput(review=example["review"]).model_dump_json()}
        ]


def test_failed_review_does_not_leak_into_the_next_request(stage):
    with native_agent(stage, fail_first=True) as (agent, requests):
        assert stage._get_single_prediction(agent, stage.TEST_DATASET[0])["genre"] == "error"
        assert stage._get_single_prediction(agent, stage.TEST_DATASET[1])["genre"] == "drama"

    assert len(requests) == 2
    assert requests[1]["messages"][1:] == [
        {"role": "user", "content": stage.MovieReviewInput(review=stage.TEST_DATASET[1]["review"]).model_dump_json()}
    ]


def test_single_review_keeps_the_validated_prediction(stage):
    with native_agent(stage) as (agent, requests):
        prediction = stage._get_single_prediction(agent, stage.TEST_DATASET[0])
    assert prediction == {"genre": "drama", "confidence": 0.75, "reasoning": "offline response"}
    assert len(requests) == 1
