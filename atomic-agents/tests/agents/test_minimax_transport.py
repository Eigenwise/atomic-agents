"""Exercise MiniMax defaults and explicit models through the real SDK transport."""

import json
from pathlib import Path
import runpy

import httpx
import instructor
import openai
import pytest

from atomic_agents import AgentConfig, AtomicAgent, BaseIOSchema, BasicChatInputSchema, BasicChatOutputSchema, VideoURL


@pytest.fixture
def transport_requests():
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append((request, payload))
        return httpx.Response(
            200,
            json={
                "id": "offline-completion",
                "object": "chat.completion",
                "created": 0,
                "model": payload["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"chat_message":"Offline response"}'},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        yield client, requests


@pytest.mark.parametrize("provider", ["7", "minimax"])
def test_quickstart_default_reaches_http_transport(monkeypatch, transport_requests, provider):
    """Run the actual interactive example, including two complete agent turns."""
    import dotenv
    from rich.console import Console

    http_client, requests = transport_requests
    sdk_init = openai.OpenAI.__init__

    def offline_init(self, **kwargs):
        sdk_init(self, **kwargs, http_client=http_client)

    monkeypatch.setattr(openai.OpenAI, "__init__", offline_init)
    monkeypatch.setattr(dotenv, "load_dotenv", lambda: None)
    monkeypatch.setenv("MINIMAX_API_KEY", "offline-test-key")
    inputs = iter([provider, "Remember 42", "What number?", "/exit"])
    monkeypatch.setattr(Console, "input", lambda *args, **kwargs: next(inputs))
    example = (
        Path(__file__).resolve().parents[3] / "atomic-examples/quickstart/quickstart/4_basic_chatbot_different_providers.py"
    )
    namespace = runpy.run_path(str(example))

    assert len(requests) == 2
    for request, payload in requests:
        assert str(request.url) == "https://api.minimax.io/v1/chat/completions"
        assert payload["model"] == "MiniMax-M3.1-Flash"
        assert payload["max_tokens"] == 2048
        assert payload["response_format"] == {"type": "json_object"}
        assert "service_tier" not in payload
        assert "thinking" not in payload
        assert "reasoning_effort" not in payload
        assert "reasoning_split" not in payload
    second_messages = requests[1][1]["messages"]
    assert any("Remember 42" in str(message["content"]) for message in second_messages)
    assert any("Offline response" in str(message["content"]) for message in second_messages)
    assert namespace["agent"].model == "MiniMax-M3.1-Flash"

    # The existing non-MiniMax provider choice retains its model and parameters.
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    _, model, parameters, role = namespace["setup_client"]("openai")
    assert model == "gpt-5-mini"
    assert parameters == {"reasoning_effort": "low", "max_tokens": 2048}
    assert role == "assistant"


@pytest.mark.parametrize("base_url", ["https://api.minimax.io/v1", "https://api.minimaxi.com/v1"])
@pytest.mark.parametrize("model", ["MiniMax-M3.1-Flash", "MiniMax-M3", "MiniMax-M2.7", "custom-saved-model"])
def test_explicit_model_and_parameters_reach_http_transport(transport_requests, base_url, model):
    """Configured IDs and parameters remain caller-owned on either configured endpoint."""
    http_client, requests = transport_requests
    raw = openai.OpenAI(base_url=base_url, api_key="offline-test-key", http_client=http_client)
    client = instructor.from_openai(raw, mode=instructor.Mode.JSON)
    parameters = {"max_tokens": 2048}
    if model == "MiniMax-M3.1-Flash":
        parameters["reasoning_effort"] = "high"
    elif model == "MiniMax-M3":
        parameters["extra_body"] = {"thinking": {"type": "disabled"}}
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
        AgentConfig(client=client, model=model, mode=instructor.Mode.JSON, model_api_parameters=parameters)
    )

    assert agent.run(BasicChatInputSchema(chat_message="Hello")).chat_message == "Offline response"
    request, payload = requests[0]
    assert str(request.url) == base_url + "/chat/completions"
    assert payload["model"] == model
    assert payload["max_tokens"] == 2048
    assert "service_tier" not in payload
    if model == "MiniMax-M3.1-Flash":
        assert payload["reasoning_effort"] == "high"
    elif model == "MiniMax-M3":
        assert payload["thinking"] == {"type": "disabled"}


def test_video_history_preserved_at_http_boundary(transport_requests):
    """The default update does not change existing typed video serialization."""

    class VideoInput(BaseIOSchema):
        """A question about a video."""

        question: str
        video: VideoURL

    http_client, requests = transport_requests
    raw = openai.OpenAI(base_url="https://api.minimax.io/v1", api_key="offline-test-key", http_client=http_client)
    client = instructor.from_openai(raw, mode=instructor.Mode.JSON)
    agent = AtomicAgent[VideoInput, BasicChatOutputSchema](
        AgentConfig(client=client, model="MiniMax-M3.1-Flash", mode=instructor.Mode.JSON)
    )
    video = VideoURL(url="https://example.test/video.mp4", fps=1)
    agent.run(VideoInput(question="Summarize this video", video=video))
    user_message = next(message for message in requests[0][1]["messages"] if message["role"] == "user")
    assert video.to_openai() in user_message["content"]
