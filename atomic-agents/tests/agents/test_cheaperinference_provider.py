"""Tests for the Cheaper Inference branch of the multi-provider quickstart example.

All HTTP traffic is mocked. Real network transports raise if any test reaches them.
"""

import importlib.util
import io
import json
import os
from types import SimpleNamespace

import httpx
import instructor
import openai
import pytest
from rich.console import Console

from atomic_agents import AtomicAgent, AgentConfig, BasicChatInputSchema, BasicChatOutputSchema

EXAMPLE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "atomic-examples",
    "quickstart",
    "quickstart",
    "4_basic_chatbot_different_providers.py",
)

VENDOR_KEY = "ci-test-vendor-key"
UNRELATED_OPENAI_KEY = "sk-unrelated-dummy-openai-key"
BASE_URL = "https://api.cheaperinference.com/v1"


def _load_example():
    spec = importlib.util.spec_from_file_location("quickstart_providers_example", EXAMPLE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def example():
    return _load_example()


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    """Fail any request that reaches a real transport."""
    sent = []

    def _blocked(self, request):
        sent.append(request)
        raise AssertionError(f"Unexpected network request: {request.method} {request.url}")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", _blocked)
    return sent


@pytest.fixture
def unrelated_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", UNRELATED_OPENAI_KEY)
    monkeypatch.delenv("CHEAPER_INFERENCE_API_KEY", raising=False)


class TestSelector:
    def test_cheaperinference_is_menu_option_9(self, example):
        assert example.providers_list[8] == "cheaperinference"
        assert example.PROVIDER_NUMBERS["9"] == "cheaperinference"

    def test_existing_menu_numbers_unchanged(self, example):
        assert example.providers_list[:8] == [
            "openai",
            "anthropic",
            "groq",
            "ollama",
            "gemini",
            "openrouter",
            "minimax",
            "edenai",
        ]

    @pytest.mark.parametrize("choice", ["9", "cheaperinference"])
    def test_selector_dispatches_to_cheaperinference(self, example, monkeypatch, choice):
        sentinel = ("client", "model", {}, "assistant")
        monkeypatch.setitem(example.PROVIDER_SETUPS, "cheaperinference", lambda: sentinel)
        assert example.setup_client(choice) is sentinel

    @pytest.mark.parametrize("choice", ["0", "10", "cheaper", ""])
    def test_unsupported_provider_raises(self, example, choice):
        with pytest.raises(ValueError, match="Unsupported provider"):
            example.setup_client(choice)


class TestCredentials:
    @pytest.mark.parametrize("vendor_value", [None, "", "   "])
    @pytest.mark.parametrize("choice", ["9", "cheaperinference"])
    def test_missing_key_never_builds_client(self, example, monkeypatch, block_network, vendor_value, choice):
        monkeypatch.setenv("OPENAI_API_KEY", UNRELATED_OPENAI_KEY)
        if vendor_value is None:
            monkeypatch.delenv("CHEAPER_INFERENCE_API_KEY", raising=False)
        else:
            monkeypatch.setenv("CHEAPER_INFERENCE_API_KEY", vendor_value)

        constructed = []
        monkeypatch.setattr(openai, "OpenAI", lambda *a, **kw: constructed.append(kw))
        sends = []
        monkeypatch.setattr(httpx.Client, "send", lambda self, *a, **kw: sends.append(a))

        with pytest.raises(ValueError, match="CHEAPER_INFERENCE_API_KEY") as excinfo:
            example.setup_client(choice)

        assert constructed == []
        assert sends == []
        assert block_network == []
        assert UNRELATED_OPENAI_KEY not in str(excinfo.value)

    def test_vendor_key_is_passed_explicitly(self, example, monkeypatch, unrelated_openai_key):
        monkeypatch.setenv("CHEAPER_INFERENCE_API_KEY", VENDOR_KEY)
        constructed = []

        class _RecordingOpenAI(openai.OpenAI):
            def __init__(self, **kwargs):
                constructed.append(kwargs)
                super().__init__(**kwargs)

        monkeypatch.setattr(openai, "OpenAI", _RecordingOpenAI)

        client, _, _, _ = example.setup_client("cheaperinference")

        assert constructed == [{"base_url": BASE_URL, "api_key": VENDOR_KEY}]
        assert client.client.api_key == VENDOR_KEY


def _tool_call_response(arguments):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-5.4-mini",
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
                            "function": {"name": "BasicChatOutputSchema", "arguments": json.dumps(arguments)},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.fixture
def mocked_gateway(example, monkeypatch, unrelated_openai_key):
    """Build the Cheaper Inference client with an httpx MockTransport instead of the network."""
    monkeypatch.setenv("CHEAPER_INFERENCE_API_KEY", VENDOR_KEY)
    requests = []

    def _handler(request):
        requests.append(request)
        return httpx.Response(200, json=_tool_call_response({"chat_message": "Hello from the mock"}))

    class _MockedOpenAI(openai.OpenAI):
        def __init__(self, **kwargs):
            super().__init__(http_client=httpx.Client(transport=httpx.MockTransport(_handler)), max_retries=0, **kwargs)

    monkeypatch.setattr(openai, "OpenAI", _MockedOpenAI)
    return example.setup_client("cheaperinference"), requests


class TestRequestShape:
    def test_setup_returns_expected_configuration(self, mocked_gateway):
        (client, model, model_api_parameters, assistant_role), _ = mocked_gateway
        assert isinstance(client, instructor.Instructor)
        assert client.mode == instructor.Mode.TOOLS
        assert str(client.client.base_url) == BASE_URL + "/"
        assert model == "gpt-5.4-mini"
        assert model_api_parameters == {"max_tokens": 2048}
        assert assistant_role == "assistant"

    def test_request_sent_to_gateway_with_vendor_key(self, mocked_gateway):
        (client, model, model_api_parameters, assistant_role), requests = mocked_gateway
        agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
            AgentConfig(
                client=client,
                model=model,
                model_api_parameters=model_api_parameters,
                assistant_role=assistant_role,
            )
        )

        agent.run(BasicChatInputSchema(chat_message="Hi"))

        assert len(requests) == 1
        request = requests[0]
        assert request.method == "POST"
        assert str(request.url) == BASE_URL + "/chat/completions"
        assert request.headers["authorization"] == f"Bearer {VENDOR_KEY}"
        assert UNRELATED_OPENAI_KEY not in str(request.headers)

        body = json.loads(request.content)
        assert body["model"] == "gpt-5.4-mini"
        assert body["max_tokens"] == 2048
        assert body["tools"][0]["function"]["name"] == "BasicChatOutputSchema"
        assert body["tool_choice"] == {"type": "function", "function": {"name": "BasicChatOutputSchema"}}
        assert body["messages"][-1]["role"] == "user"
        assert json.loads(body["messages"][-1]["content"]) == {"chat_message": "Hi"}


class TestStructuredResponse:
    def test_tool_call_parsed_into_output_schema(self, mocked_gateway):
        (client, model, model_api_parameters, assistant_role), requests = mocked_gateway
        agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
            AgentConfig(
                client=client,
                model=model,
                model_api_parameters=model_api_parameters,
                assistant_role=assistant_role,
            )
        )

        response = agent.run(BasicChatInputSchema(chat_message="Hi"))

        assert isinstance(response, BasicChatOutputSchema)
        assert response.chat_message == "Hello from the mock"
        assert len(requests) == 1
        history = agent.history.get_history()
        assert [message["role"] for message in history] == ["user", "assistant"]


class TestOtherProvidersUnchanged:
    """The dispatch refactor keeps the OpenAI-SDK providers on the same settings."""

    @pytest.mark.parametrize(
        "choice, env_var, base_url, model, params, mode",
        [
            ("1", "OPENAI_API_KEY", None, "gpt-5-mini", {"reasoning_effort": "low", "max_tokens": 2048}, "TOOLS"),
            ("4", None, "http://localhost:11434/v1", "llama3", {"max_tokens": 2048}, "JSON"),
            ("6", "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "mistral/ministral-8b", {"max_tokens": 2048}, "TOOLS"),
            ("7", "MINIMAX_API_KEY", "https://api.minimax.io/v1", "MiniMax-M3", {"max_tokens": 2048}, "JSON"),
            ("8", "EDENAI_API_KEY", "https://api.edenai.run/v3", "openai/gpt-4o-mini", {"max_tokens": 2048}, "TOOLS"),
        ],
    )
    def test_openai_sdk_providers(self, example, monkeypatch, choice, env_var, base_url, model, params, mode):
        if env_var:
            monkeypatch.setenv(env_var, "dummy-key")
        client, got_model, got_params, role = example.setup_client(choice)
        assert got_model == model
        assert got_params == params
        assert role == "assistant"
        assert client.mode == getattr(instructor.Mode, mode)
        if base_url:
            assert str(client.client.base_url) == base_url + "/"


@pytest.fixture
def captured_console(example, monkeypatch):
    """Replace the example console with one that records output and reads scripted inputs."""
    console = Console(file=io.StringIO(), width=200, force_terminal=False)
    inputs = []
    prompts = []

    def _input(prompt=""):
        prompts.append(prompt)
        return inputs.pop(0)

    monkeypatch.setattr(console, "input", _input)
    monkeypatch.setattr(example, "console", console)
    return console, inputs, prompts


class _FakeAgent:
    def __init__(self, token_info):
        self.token_info = token_info
        self.runs = []

    def get_context_token_count(self):
        return self.token_info

    def run(self, input_schema):
        self.runs.append(input_schema)
        return BasicChatOutputSchema(chat_message=f"echo: {input_schema.chat_message}")


class TestInteractiveFlow:
    def test_provider_prompt_lists_numbered_providers(self, example):
        prompt = example._provider_prompt()
        for number, name in example.PROVIDER_NUMBERS.items():
            assert f"[[bold green]{number}[/bold green]]. [bold blue]{name}[/bold blue]" in prompt

    def test_choose_provider_lowercases_input(self, example, captured_console):
        _, inputs, prompts = captured_console
        inputs.append("CheaperInference")
        assert example._choose_provider() == "cheaperinference"
        assert prompts == [example._provider_prompt()]

    @pytest.mark.parametrize("command", ["/exit", "/quit", "/EXIT"])
    def test_exit_commands_stop_the_loop(self, example, captured_console, command):
        console, _, _ = captured_console
        agent = _FakeAgent(token_info=None)
        assert example._handle_turn(agent, "m", command) is False
        assert agent.runs == []
        assert "Exiting chat..." in console.file.getvalue()

    def test_tokens_command_prints_usage_without_model_call(self, example, captured_console):
        console, _, _ = captured_console
        token_info = SimpleNamespace(total=30, system_prompt=10, history=20, max_tokens=100, utilization=0.3)
        agent = _FakeAgent(token_info)
        assert example._handle_turn(agent, "gpt-5.4-mini", "/tokens") is True
        output = console.file.getvalue()
        assert agent.runs == []
        assert "Token Usage (gpt-5.4-mini):" in output
        assert "Total: 30 tokens" in output
        assert "Max context: 100 tokens" in output
        assert "Context utilization: 30.0%" in output

    def test_tokens_command_skips_unknown_limits(self, example, captured_console):
        console, _, _ = captured_console
        token_info = SimpleNamespace(total=30, system_prompt=10, history=20, max_tokens=None, utilization=None)
        example._handle_turn(_FakeAgent(token_info), "m", "/tokens")
        output = console.file.getvalue()
        assert "Max context" not in output
        assert "Context utilization" not in output

    def test_message_is_sent_to_agent_and_printed(self, example, captured_console):
        console, _, _ = captured_console
        agent = _FakeAgent(token_info=None)
        assert example._handle_turn(agent, "m", "Hi") is True
        assert [schema.chat_message for schema in agent.runs] == ["Hi"]
        assert "Agent: echo: Hi" in console.file.getvalue()

    def test_build_agent_starts_history_with_greeting(self, example, mocked_gateway):
        (client, model, model_api_parameters, assistant_role), requests = mocked_gateway
        agent, initial_message = example._build_agent(client, model, model_api_parameters, assistant_role)
        assert initial_message.chat_message == "Hello! How can I assist you today?"
        assert agent.model == "gpt-5.4-mini"
        assert agent.model_api_parameters == {"max_tokens": 2048}
        assert [message["role"] for message in agent.history.get_history()] == ["assistant"]
        assert requests == []

    def test_main_runs_one_turn_then_exits(self, example, monkeypatch, mocked_gateway, captured_console):
        setup, requests = mocked_gateway
        console, inputs, prompts = captured_console
        monkeypatch.setattr(example, "load_dotenv", lambda: None)
        monkeypatch.setitem(example.PROVIDER_SETUPS, "cheaperinference", lambda: setup)
        inputs.extend(["9", "Hi", "/quit"])

        example.main()

        output = console.file.getvalue()
        assert inputs == []
        assert len(prompts) == 3
        assert len(requests) == 1
        assert "Agent: Hello! How can I assist you today?" in output
        assert "Agent: Hello from the mock" in output
        assert output.rstrip().endswith("Exiting chat...")
