"""Tests for AgentConfig mode resolution and client/config mode consistency (issue #282).

The mode on the Instructor client decides the API call format, while
``AgentConfig.mode`` only drives token accounting. These tests cover deriving the
mode from the client, warning when an explicit mode disagrees with the client's
mode family, and keeping clients without an exposed mode safe.
"""

import logging
from unittest.mock import Mock

import instructor
from openai import OpenAI
from pydantic import Field

from atomic_agents import AgentConfig, AtomicAgent, BaseIOSchema, BasicChatInputSchema, BasicChatOutputSchema
from atomic_agents.context import ChatHistory

DUMMY_KEY = "not-a-real-key"
LOGGER_NAME = "atomic_agents.agents.atomic_agent"


class InputWithImage(BaseIOSchema):
    """Input with image."""

    chat_message: str = Field(..., description="Text input")
    image: instructor.Image = Field(..., description="Image to analyze")


def _client(mode):
    """Build a real Instructor client with the given mode (no request is ever made)."""
    return instructor.from_openai(OpenAI(api_key=DUMMY_KEY, base_url="https://example.invalid/v1"), mode=mode)


def _agent(config):
    return AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)


class TestModeDerivation:
    """AgentConfig.mode defaults to the client's mode."""

    def test_none_derives_client_mode(self):
        agent = _agent(AgentConfig(client=_client(instructor.Mode.JSON), model="test-model"))

        assert agent.mode == instructor.Mode.JSON

    def test_none_derives_tools_mode(self):
        agent = _agent(AgentConfig(client=_client(instructor.Mode.TOOLS), model="test-model"))

        assert agent.mode == instructor.Mode.TOOLS

    def test_derived_mode_drives_token_accounting(self):
        json_agent = _agent(AgentConfig(client=_client(instructor.Mode.JSON), model="test-model"))
        tools_agent = _agent(AgentConfig(client=_client(instructor.Mode.TOOLS), model="test-model"))

        assert json_agent._build_tools_definition() is None
        assert tools_agent._build_tools_definition() is not None


class TestClientWithoutMode:
    """Clients that expose no Mode attribute fall back to TOOLS without warnings."""

    def test_mock_client_falls_back_to_tools_mode(self, caplog):
        mock_client = Mock(spec=instructor.Instructor)  # spec exposes no instance attribute `mode`
        agent = _agent(AgentConfig(client=mock_client, model="test-model"))

        assert agent.mode == instructor.Mode.TOOLS
        assert not [r for r in caplog.records if r.name == LOGGER_NAME and r.levelno >= logging.WARNING]

    def test_explicit_mode_is_kept_for_mock_client(self, caplog):
        mock_client = Mock(spec=instructor.Instructor)
        agent = _agent(AgentConfig(client=mock_client, model="test-model", mode=instructor.Mode.JSON))

        assert agent.mode == instructor.Mode.JSON
        assert not [r for r in caplog.records if r.name == LOGGER_NAME and r.levelno >= logging.WARNING]


class TestModeConsistencyWarning:
    """Explicit modes are checked against the client's mode family."""

    def test_matching_family_does_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            agent = _agent(AgentConfig(client=_client(instructor.Mode.JSON), model="test-model", mode=instructor.Mode.JSON))

        assert agent.mode == instructor.Mode.JSON
        assert not [r for r in caplog.records if r.name == LOGGER_NAME]

    def test_mismatching_family_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            agent = _agent(AgentConfig(client=_client(instructor.Mode.JSON), model="test-model", mode=instructor.Mode.TOOLS))

        assert agent.mode == instructor.Mode.TOOLS
        warnings = [r for r in caplog.records if r.name == LOGGER_NAME and r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "TOOLS" in warnings[0].getMessage()
        assert "JSON" in warnings[0].getMessage()

    def test_same_family_different_mode_does_not_warn(self, caplog):
        # TOOLS_STRICT on the config and TOOLS on the client are the same family.
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            agent = _agent(
                AgentConfig(client=_client(instructor.Mode.TOOLS), model="test-model", mode=instructor.Mode.TOOLS_STRICT)
            )

        assert agent.mode == instructor.Mode.TOOLS_STRICT
        assert not [r for r in caplog.records if r.name == LOGGER_NAME]

    def test_warning_mentions_both_directions(self, caplog):
        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            _agent(AgentConfig(client=_client(instructor.Mode.TOOLS), model="test-model", mode=instructor.Mode.JSON))

        message = [r for r in caplog.records if r.name == LOGGER_NAME][0].getMessage()
        assert "AgentConfig.mode (JSON)" in message
        assert "client's mode (TOOLS)" in message


class TestMultimodalSerializationUsesEffectiveMode:
    """Multimodal history serialization uses the agent's effective mode, not a hardcoded JSON."""

    def test_to_openai_receives_agent_mode(self, monkeypatch):
        captured = {}
        original_to_openai = instructor.Image.to_openai

        def spy(self, mode):
            captured["mode"] = mode
            return original_to_openai(self, mode)

        monkeypatch.setattr(instructor.Image, "to_openai", spy)

        history = ChatHistory()
        history.add_message(
            "user", InputWithImage(chat_message="Look", image=instructor.Image.from_url("https://example.com/x.png"))
        )
        agent = _agent(
            AgentConfig(client=_client(instructor.Mode.TOOLS), model="test-model", history=history, mode=instructor.Mode.TOOLS)
        )

        serialized = agent._serialize_history_for_token_count()

        assert captured["mode"] == instructor.Mode.TOOLS
        assert [part.get("type") for part in serialized[0]["content"]] == ["text", "image_url"]
