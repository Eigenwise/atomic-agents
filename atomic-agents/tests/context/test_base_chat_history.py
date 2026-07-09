from unittest.mock import Mock

import instructor
import pytest
from pydantic import Field

from atomic_agents import (
    AgentConfig,
    AtomicAgent,
    BaseIOSchema,
    BasicChatInputSchema,
    BasicChatOutputSchema,
)
from atomic_agents.context import BaseChatHistory, ChatHistory, Message, SystemPromptGenerator


class InputSchema(BaseIOSchema):
    """Test Input Schema"""

    test_field: str = Field(..., description="A test field")


def test_base_chat_history_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseChatHistory()


def test_chat_history_is_subclass_and_instance_of_base():
    assert issubclass(ChatHistory, BaseChatHistory)
    assert isinstance(ChatHistory(), BaseChatHistory)


def test_subclass_missing_abstractmethod_cannot_be_instantiated():
    class IncompleteHistory(BaseChatHistory):
        """A subclass that forgets to implement `copy`."""

        def initialize_turn(self) -> None:
            pass

        def add_message(self, role, content):
            pass

        def get_history(self):
            return []

        def get_current_turn_id(self):
            return None

        def delete_turn_id(self, turn_id):
            pass

        def get_message_count(self):
            return 0

        def dump(self):
            return "{}"

        def load(self, serialized_data):
            pass

        # `copy` intentionally omitted

    with pytest.raises(TypeError):
        IncompleteHistory()


class RecordingChatHistory(ChatHistory):
    """A custom persistent-memory-style backend used to exercise the extension seam."""

    def __init__(self, max_messages=None):
        super().__init__(max_messages=max_messages)
        self.recorded_calls = 0

    def add_message(self, role, content):
        self.recorded_calls += 1
        super().add_message(role, content)


@pytest.fixture
def mock_instructor():
    mock = Mock(spec=instructor.Instructor)
    mock.chat = Mock()
    mock.chat.completions = Mock()
    mock.chat.completions.create = Mock(return_value=BasicChatOutputSchema(chat_message="Test output"))
    return mock


def test_custom_chat_history_subclass_used_by_agent(mock_instructor):
    custom_history = RecordingChatHistory()
    config = AgentConfig(
        client=mock_instructor,
        model="gpt-5-mini",
        history=custom_history,
        system_prompt_generator=SystemPromptGenerator(),
    )
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)

    assert isinstance(agent.history, BaseChatHistory)

    result = agent.run(BasicChatInputSchema(chat_message="Hello"))

    assert result.chat_message == "Test output"
    # One call for the user message, one for the assistant response.
    assert custom_history.recorded_calls == 2
    assert custom_history.get_message_count() == 2


class MinimalHistory(BaseChatHistory):
    """A from-scratch BaseChatHistory implementation that does NOT subclass ChatHistory.

    Exercises the "own everything" path the docs describe, to prove a backend built purely
    against the ABC drives AtomicAgent end to end.
    """

    def __init__(self, max_messages=None):
        self.history = []
        self.current_turn_id = None
        self.max_messages = max_messages

    def initialize_turn(self):
        self.current_turn_id = "turn"

    def add_message(self, role, content):
        if self.current_turn_id is None:
            self.initialize_turn()
        self.history.append(Message(role=role, content=content, turn_id=self.current_turn_id))

    def get_history(self):
        return [{"role": m.role, "content": m.content.model_dump_json()} for m in self.history]

    def get_current_turn_id(self):
        return self.current_turn_id

    def delete_turn_id(self, turn_id):
        self.history = [m for m in self.history if m.turn_id != turn_id]

    def get_message_count(self):
        return len(self.history)

    def dump(self):
        return "{}"

    def load(self, serialized_data):
        pass

    def copy(self):
        new = MinimalHistory(max_messages=self.max_messages)
        new.history = list(self.history)
        new.current_turn_id = self.current_turn_id
        return new


def test_from_scratch_base_history_drives_agent(mock_instructor):
    history = MinimalHistory()
    config = AgentConfig(
        client=mock_instructor,
        model="gpt-5-mini",
        history=history,
        system_prompt_generator=SystemPromptGenerator(),
    )
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)

    assert isinstance(agent.history, MinimalHistory)

    result = agent.run(BasicChatInputSchema(chat_message="Hello"))

    assert result.chat_message == "Test output"
    assert agent.history.get_message_count() == 2


class StatefulHistory(ChatHistory):
    """A ChatHistory subclass carrying extra state, with copy() overridden to preserve it."""

    def __init__(self, marker, max_messages=None):
        super().__init__(max_messages=max_messages)
        self.marker = marker

    def copy(self):
        new = StatefulHistory(self.marker, max_messages=self.max_messages)
        new.load(self.dump())
        new.current_turn_id = self.current_turn_id
        return new


def test_reset_history_preserves_backend_when_copy_overridden(mock_instructor):
    history = StatefulHistory(marker="db-handle")
    config = AgentConfig(
        client=mock_instructor,
        model="gpt-5-mini",
        history=history,
        system_prompt_generator=SystemPromptGenerator(),
    )
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)

    # The initial_history snapshot keeps the subclass type and its state.
    assert isinstance(agent.initial_history, StatefulHistory)
    assert agent.initial_history.marker == "db-handle"

    agent.run(BasicChatInputSchema(chat_message="Hi"))
    agent.reset_history()

    # After reset, the backend is still the custom type, not a plain ChatHistory.
    assert isinstance(agent.history, StatefulHistory)
    assert agent.history.marker == "db-handle"


def test_subclass_without_copy_override_downgrades_to_chat_history():
    """Pins the known sharp edge: an un-overridden copy() returns a plain ChatHistory.

    This is exactly why subclasses carrying extra state must override copy(); documented in
    the Memory guide and warned about in BaseChatHistory.copy's docstring.
    """

    class NaiveHistory(ChatHistory):
        pass

    copied = NaiveHistory().copy()
    assert type(copied) is ChatHistory


def test_falsy_custom_history_is_not_discarded(mock_instructor):
    """AgentConfig must keep a provided history even if it is falsy (e.g. defines __len__)."""

    class FalsyWhenEmpty(ChatHistory):
        def __len__(self):
            return self.get_message_count()

    history = FalsyWhenEmpty()
    assert not history  # empty -> falsy via __len__
    config = AgentConfig(
        client=mock_instructor,
        model="gpt-5-mini",
        history=history,
        system_prompt_generator=SystemPromptGenerator(),
    )
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)

    assert agent.history is history
