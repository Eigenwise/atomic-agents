import json

import pytest
from unittest.mock import AsyncMock, Mock, call, patch
from enum import Enum
from pydantic import BaseModel, Field
from pydantic import ValidationError
import instructor
from atomic_agents import (
    BaseIOSchema,
    AtomicAgent,
    AgentConfig,
    BasicChatInputSchema,
    BasicChatOutputSchema,
)
from atomic_agents.context import (
    ChatHistory,
    SystemPromptGenerator,
    BaseDynamicContextProvider,
    BaseSystemPromptGenerator,
)
from atomic_agents.utils.token_counter import TokenCountResult
from instructor.dsl.partial import PartialBase


@pytest.fixture
def mock_instructor():
    mock = Mock(spec=instructor.Instructor)
    # Set up the nested mock structure
    mock.chat = Mock()
    mock.chat.completions = Mock()
    mock.chat.completions.create = Mock(return_value=BasicChatOutputSchema(chat_message="Test output"))

    # Make create_partial return an iterable
    mock_response = BasicChatOutputSchema(chat_message="Test output")
    mock_iter = Mock()
    mock_iter.__iter__ = Mock(return_value=iter([mock_response]))
    mock.chat.completions.create_partial.return_value = mock_iter

    return mock


@pytest.fixture
def mock_instructor_async():
    # Changed spec from instructor.Instructor to instructor.core.client.AsyncInstructor
    mock = Mock(spec=instructor.core.client.AsyncInstructor)

    # Configure chat.completions structure
    mock.chat = Mock()
    mock.chat.completions = Mock()

    # Make create method awaitable by using an async function
    async def mock_create(*args, **kwargs):
        return BasicChatOutputSchema(chat_message="Test output")

    mock.chat.completions.create = mock_create

    # Mock the create_partial method to return an async generator
    async def mock_create_partial(*args, **kwargs):
        yield BasicChatOutputSchema(chat_message="Test output")

    mock.chat.completions.create_partial = mock_create_partial

    return mock


@pytest.fixture
def mock_history():
    mock = Mock(spec=ChatHistory)
    mock.get_history.return_value = []
    mock.add_message = Mock()
    mock.copy = Mock(return_value=Mock(spec=ChatHistory))
    mock.initialize_turn = Mock()
    return mock


@pytest.fixture
def mock_system_prompt_generator():
    mock = Mock(spec=SystemPromptGenerator)
    mock.generate_prompt.return_value = "Mocked system prompt"
    mock.context_providers = {}
    return mock


@pytest.fixture
def agent_config(mock_instructor, mock_history, mock_system_prompt_generator):
    return AgentConfig(
        client=mock_instructor,
        model="gpt-5-mini",
        history=mock_history,
        system_prompt_generator=mock_system_prompt_generator,
    )


@pytest.fixture
def agent(agent_config):
    return AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](agent_config)


@pytest.fixture
def agent_config_async(mock_instructor_async, mock_history, mock_system_prompt_generator):
    return AgentConfig(
        client=mock_instructor_async,
        model="gpt-5-mini",
        history=mock_history,
        system_prompt_generator=mock_system_prompt_generator,
    )


@pytest.fixture
def agent_async(agent_config_async):
    return AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](agent_config_async)


@pytest.fixture
def mock_custom_system_prompt_generator():
    class MockSystemPromptGenerator(BaseSystemPromptGenerator):
        def __init__(self, context_providers=None, system_prompt=""):
            super().__init__(context_providers=context_providers)
            self.system_prompt = system_prompt

        def generate_prompt(self) -> str:
            return self.system_prompt

    return MockSystemPromptGenerator(system_prompt="Custom Prompt")


@pytest.fixture
def mock_context_provider():
    class MockContextProvider(BaseDynamicContextProvider):
        def __init__(self, title: str, info: str):
            super().__init__(title)
            self._info = info

        def get_info(self) -> str:
            return self._info

    return MockContextProvider(title="Mock Provider", info="Test")


def test_initialization(agent, mock_instructor, mock_history, mock_system_prompt_generator):
    assert agent.client == mock_instructor
    assert agent.model == "gpt-5-mini"
    assert agent.history == mock_history
    assert agent.system_prompt_generator == mock_system_prompt_generator
    assert "max_tokens" not in agent.model_api_parameters


def test_initial_history_with_local_schema_can_be_reset(mock_instructor):
    class LocalInput(BaseIOSchema):
        """Input schema created inside an application factory."""

        chat_message: str

    history = ChatHistory()
    history.add_message("user", LocalInput(chat_message="Initial context"))
    agent = AtomicAgent[LocalInput, BasicChatOutputSchema](AgentConfig(client=mock_instructor, history=history))

    result = agent.run(LocalInput(chat_message="New question"))
    assert result.chat_message == "Test output"
    assert agent.history.get_message_count() == 3

    agent.reset_history()
    assert agent.history.get_message_count() == 1
    assert isinstance(agent.history.history[0].content, LocalInput)
    assert agent.history.history[0].content.chat_message == "Initial context"

    agent.history.history[0].content.chat_message = "Changed"
    agent.reset_history()
    assert agent.history.history[0].content.chat_message == "Initial context"


# model_api_parameters should have priority over other settings
def test_initialization_temperature_priority(mock_instructor, mock_history, mock_system_prompt_generator):
    config = AgentConfig(
        client=mock_instructor,
        model="gpt-5-mini",
        history=mock_history,
        system_prompt_generator=mock_system_prompt_generator,
        model_api_parameters={"temperature": 1.0},
    )
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)
    assert agent.model_api_parameters["temperature"] == 1.0


def test_initialization_without_temperature(mock_instructor, mock_history, mock_system_prompt_generator):
    config = AgentConfig(
        client=mock_instructor,
        model="gpt-5-mini",
        history=mock_history,
        system_prompt_generator=mock_system_prompt_generator,
        model_api_parameters={"temperature": 0.5},
    )
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)
    assert agent.model_api_parameters["temperature"] == 0.5


def test_initialization_without_max_tokens(mock_instructor, mock_history, mock_system_prompt_generator):
    config = AgentConfig(
        client=mock_instructor,
        model="gpt-5-mini",
        history=mock_history,
        system_prompt_generator=mock_system_prompt_generator,
        model_api_parameters={"max_tokens": 1024},
    )
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)
    assert agent.model_api_parameters["max_tokens"] == 1024


def test_run_uses_pydantic_default_strictness_for_enum_output(mock_history, mock_system_prompt_generator):
    class Topic(Enum):
        FOOD = "food"
        OTHER = "other"

    class EnumOutputSchema(BaseIOSchema):
        """Output schema with an enum field for validation tests."""

        topic: Topic = Field(...)

    enum_instructor = Mock(spec=instructor.Instructor)
    enum_instructor.chat = Mock()
    enum_instructor.chat.completions = Mock()

    def mock_create(*args, **kwargs):
        return kwargs["response_model"].model_validate({"topic": "food"}, strict=kwargs["strict"])

    enum_instructor.chat.completions.create.side_effect = mock_create

    config = AgentConfig(
        client=enum_instructor,
        model="gpt-5-mini",
        history=mock_history,
        system_prompt_generator=mock_system_prompt_generator,
    )
    agent = AtomicAgent[BasicChatInputSchema, EnumOutputSchema](config)

    result = agent.run()

    assert result.topic is Topic.FOOD
    assert enum_instructor.chat.completions.create.call_args.kwargs["strict"] is None


def test_run_respects_explicit_strict_override_for_enum_output(mock_history, mock_system_prompt_generator):
    class Topic(Enum):
        FOOD = "food"
        OTHER = "other"

    class EnumOutputSchema(BaseIOSchema):
        """Output schema with an enum field for validation tests."""

        topic: Topic = Field(...)

    enum_instructor = Mock(spec=instructor.Instructor)
    enum_instructor.chat = Mock()
    enum_instructor.chat.completions = Mock()

    def mock_create(*args, **kwargs):
        return kwargs["response_model"].model_validate({"topic": "food"}, strict=kwargs["strict"])

    enum_instructor.chat.completions.create.side_effect = mock_create

    config = AgentConfig(
        client=enum_instructor,
        model="gpt-5-mini",
        history=mock_history,
        system_prompt_generator=mock_system_prompt_generator,
        model_api_parameters={"strict": True},
    )
    agent = AtomicAgent[BasicChatInputSchema, EnumOutputSchema](config)

    with pytest.raises(ValidationError):
        agent.run()

    assert enum_instructor.chat.completions.create.call_args.kwargs["strict"] is True


def test_initialization_system_role_equals_developer(mock_instructor, mock_history, mock_system_prompt_generator):
    config = AgentConfig(
        client=mock_instructor,
        model="gpt-5-mini",
        history=mock_history,
        system_prompt_generator=mock_system_prompt_generator,
        system_role="developer",
        model_api_parameters={},  # No temperature specified
    )
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)
    _ = agent._prepare_messages()
    assert isinstance(agent.messages, list) and agent.messages[0]["role"] == "developer"


def test_initialization_system_role_equals_None(mock_instructor, mock_history, mock_system_prompt_generator):
    config = AgentConfig(
        client=mock_instructor,
        model="gpt-5-mini",
        history=mock_history,
        system_prompt_generator=mock_system_prompt_generator,
        system_role=None,
        model_api_parameters={},  # No temperature specified
    )
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)
    _ = agent._prepare_messages()
    assert isinstance(agent.messages, list) and len(agent.messages) == 0


def test_reset_history(agent, mock_history):
    initial_history = agent.initial_history
    agent.reset_history()
    assert agent.history != initial_history
    mock_history.copy.assert_called_once()


def test_get_context_provider(agent, mock_system_prompt_generator):
    mock_provider = Mock(spec=BaseDynamicContextProvider)
    mock_system_prompt_generator.context_providers = {"test_provider": mock_provider}

    result = agent.get_context_provider("test_provider")
    assert result == mock_provider

    with pytest.raises(KeyError):
        agent.get_context_provider("non_existent_provider")


def test_register_context_provider(agent, mock_system_prompt_generator):
    mock_provider = Mock(spec=BaseDynamicContextProvider)
    agent.register_context_provider("new_provider", mock_provider)

    assert "new_provider" in mock_system_prompt_generator.context_providers
    assert mock_system_prompt_generator.context_providers["new_provider"] == mock_provider


def test_unregister_context_provider(agent, mock_system_prompt_generator):
    mock_provider = Mock(spec=BaseDynamicContextProvider)
    mock_system_prompt_generator.context_providers = {"test_provider": mock_provider}

    agent.unregister_context_provider("test_provider")
    assert "test_provider" not in mock_system_prompt_generator.context_providers

    with pytest.raises(KeyError):
        agent.unregister_context_provider("non_existent_provider")


def test_no_type_parameters(mock_instructor):
    custom_config = AgentConfig(
        client=mock_instructor,
        model="gpt-5-mini",
    )

    custom_agent = AtomicAgent(custom_config)

    assert custom_agent.input_schema == BasicChatInputSchema
    assert custom_agent.output_schema == BasicChatOutputSchema


def test_custom_input_output_schemas(mock_instructor):
    class CustomInputSchema(BaseModel):
        custom_field: str

    class CustomOutputSchema(BaseModel):
        result: str

    custom_config = AgentConfig(
        client=mock_instructor,
        model="gpt-5-mini",
    )

    custom_agent = AtomicAgent[CustomInputSchema, CustomOutputSchema](custom_config)

    assert custom_agent.input_schema == CustomInputSchema
    assert custom_agent.output_schema == CustomOutputSchema


def test_subclass_with_custom_constructor(mock_instructor):
    """Test that generic types are preserved in subclasses with custom constructors."""

    class CustomInputSchema(BaseModel):
        custom_field: str

    class CustomOutputSchema(BaseModel):
        result: str

    class MyAgent(AtomicAgent[CustomInputSchema, CustomOutputSchema]):
        def __init__(self, extra_param: str):
            self.extra_param = extra_param
            config = AgentConfig(
                client=mock_instructor,
                model="gpt-5-mini",
            )
            super().__init__(config)

    agent = MyAgent("test_value")

    # These would fail without the __init_subclass__ fix
    assert agent.input_schema == CustomInputSchema
    assert agent.output_schema == CustomOutputSchema
    assert agent.extra_param == "test_value"


def test_base_agent_io_str_and_rich():
    class TestIO(BaseIOSchema):
        """TestIO docstring"""

        field: str

    test_io = TestIO(field="test")
    assert str(test_io) == '{"field":"test"}'
    assert test_io.__rich__() is not None  # Just check if it returns something, as we can't easily compare Rich objects


def test_base_io_schema_empty_docstring():
    with pytest.raises(ValueError, match="must have a non-empty docstring"):

        class EmptyDocStringSchema(BaseIOSchema):
            """"""

            pass


def test_base_io_schema_model_json_schema_no_description():
    class TestSchema(BaseIOSchema):
        """Test schema docstring."""

        field: str

    # Mock the superclass model_json_schema to return a schema without a description
    with patch("pydantic.BaseModel.model_json_schema", return_value={}):
        schema = TestSchema.model_json_schema()

    assert "description" in schema
    assert schema["description"] == "Test schema docstring."


def test_run(agent, mock_history):
    # Use the agent fixture that's already configured correctly
    mock_input = BasicChatInputSchema(chat_message="Test input")

    result = agent.run(mock_input)

    # Assertions
    assert result.chat_message == "Test output"
    assert agent.current_user_input == mock_input

    mock_history.add_message.assert_has_calls([call("user", mock_input), call("assistant", result)])


def test_messages_sync_after_run(mock_instructor, mock_system_prompt_generator):
    """
    Test that agent.messages includes the assistant response after run() completes.

    Regression test for GitHub issue #194:
    https://github.com/eigenwise/atomic-agents/issues/194

    The issue was that agent.messages only contained the system prompt and user message
    after run(), while agent.history.get_history() correctly included the assistant response.
    """
    # Use real ChatHistory instead of mock to verify actual message synchronization
    real_history = ChatHistory()

    config = AgentConfig(
        client=mock_instructor,
        model="gpt-5-mini",
        history=real_history,
        system_prompt_generator=mock_system_prompt_generator,
    )
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)

    mock_input = BasicChatInputSchema(chat_message="Test input")

    result = agent.run(mock_input)

    # Verify the result is returned correctly
    assert result.chat_message == "Test output"

    # Verify agent.messages is in sync with history.get_history()
    history_messages = agent.history.get_history()

    # agent.messages should contain: system prompt + history (user + assistant)
    assert len(agent.messages) == 3, f"Expected 3 messages (system + user + assistant), got {len(agent.messages)}"

    # First message should be the system prompt
    assert agent.messages[0]["role"] == "system"

    # Second message should be the user input
    assert agent.messages[1]["role"] == "user"

    # Third message should be the assistant response (the key fix for issue #194)
    assert agent.messages[2]["role"] == "assistant"

    # Verify consistency: agent.messages[-2:] should match history.get_history()
    assert len(history_messages) == 2, f"Expected 2 history messages, got {len(history_messages)}"
    assert agent.messages[1:] == history_messages


def test_run_stream(mock_instructor, mock_history):
    # Create a AgentConfig with system_role set to None
    config = AgentConfig(
        client=mock_instructor,
        model="gpt-5-mini",
        history=mock_history,
        system_prompt_generator=None,  # No system prompt generator
    )
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)

    mock_input = BasicChatInputSchema(chat_message="Test input")
    mock_output = BasicChatOutputSchema(chat_message="Test output")

    for result in agent.run_stream(mock_input):
        pass

    assert result == mock_output
    assert agent.current_user_input == mock_input

    mock_history.add_message.assert_has_calls([call("user", mock_input), call("assistant", mock_output)])


@pytest.mark.asyncio
async def test_run_async(agent_async, mock_history):
    # Create a mock input
    mock_input = BasicChatInputSchema(chat_message="Test input")
    mock_output = BasicChatOutputSchema(chat_message="Test output")

    # Get response from run_async method
    response = await agent_async.run_async(mock_input)

    # Assertions
    assert response == mock_output
    assert agent_async.current_user_input == mock_input
    mock_history.add_message.assert_has_calls([call("user", mock_input), call("assistant", mock_output)])


@pytest.mark.asyncio
async def test_run_async_stream(agent_async, mock_history):
    # Create a mock input
    mock_input = BasicChatInputSchema(chat_message="Test input")
    mock_output = BasicChatOutputSchema(chat_message="Test output")

    responses = []
    # Get response from run_async_stream method
    async for response in agent_async.run_async_stream(mock_input):
        responses.append(response)

    # Assertions
    assert len(responses) == 1
    assert responses[0] == mock_output
    assert agent_async.current_user_input == mock_input

    # Verify that both user input and assistant response were added to history
    mock_history.add_message.assert_any_call("user", mock_input)

    # Create the expected full response content to check
    full_response_content = agent_async.output_schema(**responses[0].model_dump())
    mock_history.add_message.assert_any_call("assistant", full_response_content)


def test_model_from_chunks_patched():
    class TestPartialModel(PartialBase):
        @classmethod
        def get_partial_model(cls):
            class PartialModel(BaseModel):
                field: str

            return PartialModel

    chunks = ['{"field": "hel', 'lo"}']
    expected_values = ["hel", "hello"]

    generator = TestPartialModel.model_from_chunks(chunks)
    results = [result.field for result in generator]

    assert results == expected_values


@pytest.mark.asyncio
async def test_model_from_chunks_async_patched():
    class TestPartialModel(PartialBase):
        @classmethod
        def get_partial_model(cls):
            class PartialModel(BaseModel):
                field: str

            return PartialModel

    async def async_gen():
        yield '{"field": "hel'
        yield 'lo"}'

    expected_values = ["hel", "hello"]

    generator = TestPartialModel.model_from_chunks_async(async_gen())
    results = []
    async for result in generator:
        results.append(result.field)

    assert results == expected_values


# Hook System Tests


def test_hook_initialization(agent):
    """Test that hook system is properly initialized."""
    # Verify hook attributes exist and are properly initialized
    assert hasattr(agent, "_hook_handlers")
    assert hasattr(agent, "_hooks_enabled")
    assert isinstance(agent._hook_handlers, dict)
    assert agent._hooks_enabled is True
    assert len(agent._hook_handlers) == 0


def test_hook_registration(agent):
    """Test hook registration and unregistration functionality."""
    # Test registration
    handler_called = []

    def test_handler(error):
        handler_called.append(error)

    agent.register_hook("parse:error", test_handler)

    # Verify internal storage
    assert "parse:error" in agent._hook_handlers
    assert test_handler in agent._hook_handlers["parse:error"]

    # Test unregistration
    agent.unregister_hook("parse:error", test_handler)
    assert test_handler not in agent._hook_handlers["parse:error"]


def test_hook_registration_with_instructor_client(mock_instructor):
    """Test that hooks are registered with instructor client when available."""
    # Add hook methods to mock instructor
    mock_instructor.on = Mock()
    mock_instructor.off = Mock()
    mock_instructor.clear = Mock()

    config = AgentConfig(client=mock_instructor, model="gpt-5-mini")
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)

    def test_handler(error):
        pass

    # Test registration delegates to instructor client
    agent.register_hook("parse:error", test_handler)
    mock_instructor.on.assert_called_once_with("parse:error", test_handler)

    # Test unregistration delegates to instructor client
    agent.unregister_hook("parse:error", test_handler)
    mock_instructor.off.assert_called_once_with("parse:error", test_handler)


def test_multiple_hook_handlers(agent):
    """Test multiple handlers for the same event."""
    handler1_calls = []
    handler2_calls = []

    def handler1(error):
        handler1_calls.append(error)

    def handler2(error):
        handler2_calls.append(error)

    # Register multiple handlers
    agent.register_hook("parse:error", handler1)
    agent.register_hook("parse:error", handler2)

    # Verify both are registered
    assert len(agent._hook_handlers["parse:error"]) == 2
    assert handler1 in agent._hook_handlers["parse:error"]
    assert handler2 in agent._hook_handlers["parse:error"]

    # Test dispatch to both handlers
    test_error = Exception("test error")
    agent._dispatch_hook("parse:error", test_error)

    assert len(handler1_calls) == 1
    assert len(handler2_calls) == 1
    assert handler1_calls[0] is test_error
    assert handler2_calls[0] is test_error


def test_hook_clear_specific_event(agent):
    """Test clearing hooks for a specific event."""

    def handler1():
        pass

    def handler2():
        pass

    # Register handlers for different events
    agent.register_hook("parse:error", handler1)
    agent.register_hook("completion:error", handler2)

    # Clear specific event
    agent.clear_hooks("parse:error")

    # Verify only parse:error was cleared
    assert len(agent._hook_handlers["parse:error"]) == 0
    assert handler2 in agent._hook_handlers["completion:error"]


def test_hook_clear_all_events(agent):
    """Test clearing all hooks."""

    def handler1():
        pass

    def handler2():
        pass

    # Register handlers for different events
    agent.register_hook("parse:error", handler1)
    agent.register_hook("completion:error", handler2)

    # Clear all hooks
    agent.clear_hooks()

    # Verify all hooks are cleared
    assert len(agent._hook_handlers) == 0


def test_hook_enable_disable(agent):
    """Test hook enable/disable functionality."""
    # Test initial state
    assert agent.hooks_enabled is True

    # Test disable
    agent.disable_hooks()
    assert agent.hooks_enabled is False
    assert agent._hooks_enabled is False

    # Test enable
    agent.enable_hooks()
    assert agent.hooks_enabled is True
    assert agent._hooks_enabled is True


def test_hook_dispatch_when_disabled(agent):
    """Test that hooks don't execute when disabled."""
    handler_called = []

    def test_handler(error):
        handler_called.append(error)

    agent.register_hook("parse:error", test_handler)

    # Disable hooks
    agent.disable_hooks()

    # Dispatch should not call handler
    agent._dispatch_hook("parse:error", Exception("test"))
    assert len(handler_called) == 0

    # Re-enable and test
    agent.enable_hooks()
    agent._dispatch_hook("parse:error", Exception("test"))
    assert len(handler_called) == 1


def test_hook_error_isolation(agent):
    """Test that hook handler errors don't interrupt main flow."""
    good_handler_called = []

    def bad_handler(error):
        raise RuntimeError("Handler error")

    def good_handler(error):
        good_handler_called.append(error)

    # Register both handlers
    agent.register_hook("test:event", bad_handler)
    agent.register_hook("test:event", good_handler)

    # Dispatch should not raise exception
    with patch("logging.getLogger") as mock_logger:
        mock_log = Mock()
        mock_logger.return_value = mock_log

        agent._dispatch_hook("test:event", Exception("test"))

        # Verify error was logged
        mock_log.warning.assert_called_once()

        # Verify good handler still executed
        assert len(good_handler_called) == 1


def test_hook_dispatch_nonexistent_event(agent):
    """Test dispatching to nonexistent event."""
    # Should not raise exception
    agent._dispatch_hook("nonexistent:event", Exception("test"))


def test_hook_unregister_nonexistent_handler(agent):
    """Test unregistering handler that doesn't exist."""

    def test_handler():
        pass

    # Should not raise exception
    agent.unregister_hook("parse:error", test_handler)


def test_agent_initialization_includes_hooks(mock_instructor, mock_history, mock_system_prompt_generator):
    """Test that agent initialization properly sets up hook system."""
    config = AgentConfig(
        client=mock_instructor,
        model="gpt-5-mini",
        history=mock_history,
        system_prompt_generator=mock_system_prompt_generator,
    )

    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)

    # Verify hook system is initialized
    assert hasattr(agent, "_hook_handlers")
    assert hasattr(agent, "_hooks_enabled")
    assert agent._hooks_enabled is True
    assert isinstance(agent._hook_handlers, dict)
    assert len(agent._hook_handlers) == 0

    # Verify hook management methods exist
    assert hasattr(agent, "register_hook")
    assert hasattr(agent, "unregister_hook")
    assert hasattr(agent, "clear_hooks")
    assert hasattr(agent, "enable_hooks")
    assert hasattr(agent, "disable_hooks")
    assert hasattr(agent, "hooks_enabled")
    assert hasattr(agent, "_dispatch_hook")


def test_backward_compatibility_no_breaking_changes(mock_instructor, mock_history, mock_system_prompt_generator):
    """Test that hook system addition doesn't break existing functionality."""
    # Ensure mock_history.get_history() returns an empty list
    mock_history.get_history.return_value = []

    # Ensure the copy method returns a properly configured mock
    copied_mock = Mock(spec=ChatHistory)
    copied_mock.get_history.return_value = []
    mock_history.copy.return_value = copied_mock

    config = AgentConfig(
        client=mock_instructor,
        model="gpt-5-mini",
        history=mock_history,
        system_prompt_generator=mock_system_prompt_generator,
    )

    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)

    # Test that all existing attributes still exist and work
    assert agent.client == mock_instructor
    assert agent.model == "gpt-5-mini"
    assert agent.history == mock_history
    assert agent.system_prompt_generator == mock_system_prompt_generator

    # Test that existing methods still work
    # Note: reset_history() changes the history object, so we skip it to focus on core functionality

    # Properties should work
    assert agent.input_schema == BasicChatInputSchema
    assert agent.output_schema == BasicChatOutputSchema

    # Run method should work (with hooks enabled by default)
    user_input = BasicChatInputSchema(chat_message="test")
    response = agent.run(user_input)

    # Verify the response is valid
    assert response is not None

    # Verify the call was made correctly
    mock_instructor.chat.completions.create.assert_called()

    # Test context provider methods still work
    from atomic_agents.context import BaseDynamicContextProvider

    class TestProvider(BaseDynamicContextProvider):
        def get_info(self):
            return "test"

    provider = TestProvider(title="Test")
    agent.register_context_provider("test", provider)

    retrieved = agent.get_context_provider("test")
    assert retrieved == provider

    agent.unregister_context_provider("test")

    # Should raise KeyError for non-existent provider
    with pytest.raises(KeyError):
        agent.get_context_provider("test")


# Token Counting Tests


@patch("atomic_agents.agents.atomic_agent.get_token_counter")
def test_get_context_token_count_basic(mock_get_token_counter, mock_instructor, mock_history, mock_system_prompt_generator):
    """Test basic token counting functionality."""
    mock_history.get_history.return_value = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    mock_system_prompt_generator.generate_prompt.return_value = "You are a helpful assistant."

    mock_counter_instance = Mock()
    mock_get_token_counter.return_value = mock_counter_instance
    mock_counter_instance.count_context.return_value = TokenCountResult(
        total=100, system_prompt=30, history=70, tools=0, model="gpt-5-mini", max_tokens=8192, utilization=0.0122
    )

    config = AgentConfig(
        client=mock_instructor,
        model="gpt-5-mini",
        history=mock_history,
        system_prompt_generator=mock_system_prompt_generator,
    )
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)

    result = agent.get_context_token_count()

    assert result.total == 100
    assert result.system_prompt == 30
    assert result.history == 70
    assert result.tools == 0
    assert result.model == "gpt-5-mini"
    assert result.max_tokens == 8192
    assert result.utilization is not None  # Should have utilization since max_tokens is known
    mock_counter_instance.count_context.assert_called_once()


@patch("atomic_agents.agents.atomic_agent.get_token_counter")
def test_get_context_token_count_includes_schema_overhead(
    mock_get_token_counter, mock_instructor, mock_history, mock_system_prompt_generator
):
    """Test that token counting includes the output schema overhead for JSON mode."""
    from instructor import Mode

    mock_history.get_history.return_value = []
    mock_system_prompt_generator.generate_prompt.return_value = "System prompt"

    mock_counter_instance = Mock()
    mock_get_token_counter.return_value = mock_counter_instance
    mock_counter_instance.count_context.return_value = TokenCountResult(
        total=50, system_prompt=50, history=0, tools=0, model="gpt-5-mini"
    )

    config = AgentConfig(
        client=mock_instructor,
        model="gpt-5-mini",
        history=mock_history,
        system_prompt_generator=mock_system_prompt_generator,
        mode=Mode.JSON,  # JSON mode appends schema to system message
    )
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)

    agent.get_context_token_count()

    # Verify count_context was called with system message that includes schema
    call_args = mock_counter_instance.count_context.call_args
    system_messages = call_args.kwargs["system_messages"]
    assert len(system_messages) == 1
    # System message should contain both the prompt AND the schema (for JSON mode)
    assert "System prompt" in system_messages[0]["content"]
    assert "chat_message" in system_messages[0]["content"]  # Schema field from BasicChatOutputSchema


@patch("atomic_agents.agents.atomic_agent.get_token_counter")
def test_get_context_token_count_no_system_prompt(mock_get_token_counter, mock_instructor, mock_history):
    """Test token counting when system_role is None (schema still included for JSON mode)."""
    from instructor import Mode

    mock_history.get_history.return_value = [{"role": "user", "content": "Hello"}]

    mock_counter_instance = Mock()
    mock_get_token_counter.return_value = mock_counter_instance
    mock_counter_instance.count_context.return_value = TokenCountResult(
        total=25, system_prompt=20, history=5, tools=0, model="gpt-5-mini"
    )

    config = AgentConfig(
        client=mock_instructor,
        model="gpt-5-mini",
        history=mock_history,
        system_role=None,  # No system prompt
        mode=Mode.JSON,  # JSON mode to test schema in system message
    )
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)

    agent.get_context_token_count()

    # Even without system prompt, schema should be included (for JSON mode)
    call_args = mock_counter_instance.count_context.call_args
    system_messages = call_args.kwargs["system_messages"]
    assert len(system_messages) == 1  # Schema is added as system message
    assert "chat_message" in system_messages[0]["content"]  # Schema content


@patch("atomic_agents.agents.atomic_agent.get_token_counter")
def test_get_context_token_count_uses_configured_model(
    mock_get_token_counter, mock_instructor, mock_history, mock_system_prompt_generator
):
    """Test that token counting uses the agent's configured model."""
    mock_history.get_history.return_value = []

    mock_counter_instance = Mock()
    mock_get_token_counter.return_value = mock_counter_instance
    mock_counter_instance.count_context.return_value = TokenCountResult(
        total=15, system_prompt=15, history=0, tools=0, model="claude-3-opus-20240229"
    )

    config = AgentConfig(
        client=mock_instructor,
        model="claude-3-opus-20240229",  # Different model
        history=mock_history,
        system_prompt_generator=mock_system_prompt_generator,
    )
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)

    agent.get_context_token_count()

    call_args = mock_counter_instance.count_context.call_args
    assert call_args.kwargs["model"] == "claude-3-opus-20240229"


@patch("atomic_agents.agents.atomic_agent.get_token_counter")
def test_get_context_token_count_returns_token_count_result(mock_get_token_counter, agent):
    """Test that get_context_token_count returns a TokenCountResult."""
    mock_counter_instance = Mock()
    mock_get_token_counter.return_value = mock_counter_instance
    mock_counter_instance.count_context.return_value = TokenCountResult(
        total=50, system_prompt=20, history=30, tools=0, model="gpt-5-mini"
    )

    result = agent.get_context_token_count()

    assert isinstance(result, TokenCountResult)
    assert hasattr(result, "total")
    assert hasattr(result, "system_prompt")
    assert hasattr(result, "history")
    assert hasattr(result, "tools")
    assert hasattr(result, "model")
    assert hasattr(result, "max_tokens")
    assert hasattr(result, "utilization")


@patch("atomic_agents.agents.atomic_agent.get_token_counter")
def test_get_context_token_count_hook_dispatch(mock_get_token_counter, agent):
    """Test that token:counted hook is dispatched."""
    mock_counter_instance = Mock()
    mock_get_token_counter.return_value = mock_counter_instance
    expected_result = TokenCountResult(total=100, system_prompt=30, history=70, tools=0, model="gpt-5-mini")
    mock_counter_instance.count_context.return_value = expected_result

    hook_called = []

    def hook_handler(result):
        hook_called.append(result)

    agent.register_hook("token:counted", hook_handler)

    agent.get_context_token_count()

    assert len(hook_called) == 1
    assert hook_called[0] == expected_result
    assert hook_called[0].total == 100


@patch("atomic_agents.agents.atomic_agent.get_token_counter")
def test_get_context_token_count_hook_not_called_when_disabled(mock_get_token_counter, agent):
    """Test that token:counted hook is not called when hooks are disabled."""
    mock_counter_instance = Mock()
    mock_get_token_counter.return_value = mock_counter_instance
    mock_counter_instance.count_context.return_value = TokenCountResult(
        total=100, system_prompt=30, history=70, tools=0, model="gpt-5-mini"
    )

    hook_called = []

    def hook_handler(result):
        hook_called.append(result)

    agent.register_hook("token:counted", hook_handler)
    agent.disable_hooks()

    agent.get_context_token_count()

    assert len(hook_called) == 0


@patch("atomic_agents.agents.atomic_agent.get_token_counter")
def test_get_context_token_count_multimodal_content(mock_get_token_counter, mock_instructor):
    """Test that multimodal content is properly serialized for token counting."""
    from instructor.processing.multimodal import Image

    mock_counter_instance = Mock()
    mock_get_token_counter.return_value = mock_counter_instance
    mock_counter_instance.count_context.return_value = TokenCountResult(
        total=150, system_prompt=50, history=100, tools=0, model="gpt-4-vision-preview"
    )

    # Create a multimodal input schema
    class MultimodalInputSchema(BaseIOSchema):
        """Input with image."""

        text: str = Field(..., description="Text input")
        image: instructor.Image = Field(..., description="Image to analyze")

    # Create agent with multimodal schema
    config = AgentConfig(
        client=mock_instructor,
        model="gpt-4-vision-preview",
    )
    agent = AtomicAgent[MultimodalInputSchema, BasicChatOutputSchema](config)

    # Add multimodal message to history
    test_image = Image(source="https://example.com/test.png", media_type="image/png")
    multimodal_input = MultimodalInputSchema(text="Describe this image", image=test_image)
    agent.history.add_message("user", multimodal_input)

    # Get token count
    agent.get_context_token_count()

    # Verify count_context was called
    assert mock_counter_instance.count_context.called

    # Get the history_messages passed to count_context
    call_args = mock_counter_instance.count_context.call_args
    history_messages = call_args.kwargs["history_messages"]

    # Verify multimodal content was serialized properly
    assert len(history_messages) == 1
    assert history_messages[0]["role"] == "user"

    # Content should be a list with text and image
    content = history_messages[0]["content"]
    assert isinstance(content, list)

    # Should have text and image entries
    content_types = [item.get("type") for item in content]
    assert "text" in content_types
    assert "image_url" in content_types

    # Verify image was converted to OpenAI format
    image_entry = next(item for item in content if item.get("type") == "image_url")
    assert "image_url" in image_entry
    assert image_entry["image_url"]["url"] == "https://example.com/test.png"


def test_serialize_history_for_token_count_video_placeholder(agent, mock_history):
    video_part = {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}}
    mock_history.get_history.return_value = [{"role": "user", "content": ['{"prompt":"Summarize the video"}', video_part]}]

    serialized = agent._serialize_history_for_token_count()

    assert serialized[0]["content"] == [
        {"type": "text", "text": '{"prompt":"Summarize the video"}'},
        {"type": "text", "text": "[video_url content]"},
    ]


# --- Tests for tool_result_role and Gemini system message remapping (issue #221) ---


def test_tool_result_role_defaults_to_system_for_openai(mock_instructor, mock_system_prompt_generator):
    config = AgentConfig(
        client=mock_instructor,
        model="gpt-5-mini",
        system_prompt_generator=mock_system_prompt_generator,
        assistant_role="assistant",
    )
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)
    assert agent.tool_result_role == "system"


def test_tool_result_role_defaults_to_user_for_gemini(mock_instructor, mock_system_prompt_generator):
    config = AgentConfig(
        client=mock_instructor,
        model="gemini-2.0-flash",
        system_prompt_generator=mock_system_prompt_generator,
        assistant_role="model",
    )
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)
    assert agent.tool_result_role == "user"


def test_tool_result_role_explicit_override(mock_instructor, mock_system_prompt_generator):
    config = AgentConfig(
        client=mock_instructor,
        model="gemini-2.0-flash",
        system_prompt_generator=mock_system_prompt_generator,
        assistant_role="model",
        tool_result_role="system",
    )
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)
    assert agent.tool_result_role == "system"


def test_add_tool_result_uses_correct_role(mock_instructor, mock_system_prompt_generator):
    history = ChatHistory()
    config = AgentConfig(
        client=mock_instructor,
        model="gemini-2.0-flash",
        system_prompt_generator=mock_system_prompt_generator,
        assistant_role="model",
        history=history,
    )
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)
    content = BasicChatInputSchema(chat_message="Tool result data")
    agent.add_tool_result(content)

    messages = history.get_history()
    assert len(messages) == 1
    assert messages[0]["role"] == "user"


def test_add_tool_result_uses_system_for_openai(mock_instructor, mock_system_prompt_generator):
    history = ChatHistory()
    config = AgentConfig(
        client=mock_instructor,
        model="gpt-5-mini",
        system_prompt_generator=mock_system_prompt_generator,
        history=history,
    )
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)
    content = BasicChatInputSchema(chat_message="Tool result data")
    agent.add_tool_result(content)

    messages = history.get_history()
    assert len(messages) == 1
    assert messages[0]["role"] == "system"


def test_prepare_messages_remaps_system_to_user_for_gemini(mock_instructor, mock_system_prompt_generator):
    history = ChatHistory()
    config = AgentConfig(
        client=mock_instructor,
        model="gemini-2.0-flash",
        system_prompt_generator=mock_system_prompt_generator,
        assistant_role="model",
        history=history,
    )
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)

    # Simulate legacy pattern: add_message("system", ...) directly on history
    history.add_message("system", BasicChatInputSchema(chat_message="Tool output"))
    agent._prepare_messages()

    # The system prompt message (first) should keep its role,
    # but mid-conversation "system" messages in history should be remapped to "user"
    history_messages = [m for m in agent.messages if m.get("content") != mock_system_prompt_generator.generate_prompt()]
    assert len(history_messages) == 1
    assert history_messages[0]["role"] == "user"


def test_prepare_messages_keeps_system_for_openai(mock_instructor, mock_system_prompt_generator):
    history = ChatHistory()
    config = AgentConfig(
        client=mock_instructor,
        model="gpt-5-mini",
        system_prompt_generator=mock_system_prompt_generator,
        history=history,
    )
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)

    history.add_message("system", BasicChatInputSchema(chat_message="Tool output"))
    agent._prepare_messages()

    history_messages = [m for m in agent.messages if m.get("content") != mock_system_prompt_generator.generate_prompt()]
    assert len(history_messages) == 1
    assert history_messages[0]["role"] == "system"


# --- max_context_tokens and _trim_context tests ---


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["run", "run_stream", "run_async", "run_async_stream"])
@pytest.mark.parametrize("input_size", [15, 40])
@patch("atomic_agents.agents.atomic_agent.get_token_counter")
async def test_context_budget_includes_new_input(
    mock_get_token_counter, method, input_size, mock_instructor, mock_instructor_async, mock_system_prompt_generator
):
    """All completion APIs budget the actual request and preserve the latest input."""
    history = ChatHistory()
    history.add_message("user", BasicChatInputSchema(chat_message="o" * 20))
    client = mock_instructor_async if "async" in method else mock_instructor
    response = BasicChatOutputSchema(chat_message="Response")
    if "async" in method:
        client.chat.completions.create = AsyncMock(return_value=response)

        async def partials(**kwargs):
            yield response

        client.chat.completions.create_partial = Mock(side_effect=partials)

    def count_context(model, system_messages, history_messages, tools):
        history_tokens = sum(len(json.loads(message["content"])["chat_message"]) for message in history_messages)
        return TokenCountResult(total=10 + history_tokens, system_prompt=10, history=history_tokens, tools=0, model=model)

    mock_get_token_counter.return_value.count_context.side_effect = count_context
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
        AgentConfig(
            client=client,
            model="gpt-5-mini",
            history=history,
            system_prompt_generator=mock_system_prompt_generator,
            max_context_tokens=30,
        )
    )
    user_input = BasicChatInputSchema(chat_message="n" * input_size)

    async def invoke():
        if method == "run":
            agent.run(user_input)
        elif method == "run_stream":
            list(agent.run_stream(user_input))
        elif method == "run_async":
            await agent.run_async(user_input)
        else:
            async for _ in agent.run_async_stream(user_input):
                pass

    if input_size == 40:
        with pytest.raises(ValueError, match="max_context_tokens"):
            await invoke()
        client.chat.completions.create.assert_not_called()
        client.chat.completions.create_partial.assert_not_called()
        assert history.history[-1].content == user_input
    else:
        await invoke()
        completion = client.chat.completions.create_partial if "stream" in method else client.chat.completions.create
        request_history = completion.call_args.kwargs["messages"][1:]
        assert request_history == [{"role": "user", "content": user_input.model_dump_json()}]
        assert count_context(agent.model, [], request_history, None).total <= agent.max_context_tokens


def test_trim_context_no_op_when_max_context_tokens_unset(mock_instructor, mock_system_prompt_generator):
    """When max_context_tokens is None, _trim_context returns without any action."""
    history = ChatHistory()
    history.add_message("user", BasicChatInputSchema(chat_message="Hello"))
    config = AgentConfig(
        client=mock_instructor,
        model="gpt-5-mini",
        history=history,
        system_prompt_generator=mock_system_prompt_generator,
        # max_context_tokens not set — default None
    )
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)

    # Should not raise, should not modify history
    agent._trim_context()
    assert history.get_message_count() == 1


@patch("atomic_agents.agents.atomic_agent.get_token_counter")
def test_trim_context_no_op_when_within_limit(mock_get_token_counter, mock_instructor, mock_system_prompt_generator):
    """When context is within max_context_tokens, history is not modified."""
    history = ChatHistory()
    history.add_message("user", BasicChatInputSchema(chat_message="Hello"))

    mock_counter_instance = Mock()
    mock_get_token_counter.return_value = mock_counter_instance
    mock_counter_instance.count_context.return_value = TokenCountResult(
        total=100, system_prompt=30, history=70, tools=0, model="gpt-5-mini", max_tokens=8192, utilization=0.01
    )

    config = AgentConfig(
        client=mock_instructor,
        model="gpt-5-mini",
        history=history,
        system_prompt_generator=mock_system_prompt_generator,
        max_context_tokens=200,  # well above current 100
    )
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)
    agent._trim_context()

    # History untouched
    assert history.get_message_count() == 1
    # delete_turn_id never called
    mock_counter_instance.count_context.assert_called_once()


@patch("atomic_agents.agents.atomic_agent.get_token_counter")
def test_trim_context_trims_oldest_turn_when_over_limit(mock_get_token_counter, mock_instructor, mock_system_prompt_generator):
    """When context exceeds max_context_tokens, oldest turn is removed."""
    history = ChatHistory()

    # Turn 1 (user message)
    history.initialize_turn()
    history.add_message("user", BasicChatInputSchema(chat_message="First request"))

    # Turn 2 (user message)
    history.initialize_turn()
    history.add_message("user", BasicChatInputSchema(chat_message="Second request"))

    turn1_id = history.history[0].turn_id

    mock_counter_instance = Mock()
    mock_get_token_counter.return_value = mock_counter_instance
    # First call: over limit; Second call: within limit after one turn removed
    mock_counter_instance.count_context.side_effect = [
        TokenCountResult(
            total=500, system_prompt=100, history=400, tools=0, model="gpt-5-mini", max_tokens=8192, utilization=0.06
        ),
        TokenCountResult(
            total=150, system_prompt=100, history=50, tools=0, model="gpt-5-mini", max_tokens=8192, utilization=0.018
        ),
    ]

    config = AgentConfig(
        client=mock_instructor,
        model="gpt-5-mini",
        history=history,
        system_prompt_generator=mock_system_prompt_generator,
        max_context_tokens=200,
    )
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)
    agent._trim_context()

    # Turn 1 should be gone; Turn 2 remains
    assert history.get_message_count() == 1
    remaining = history.history[0]
    assert remaining.content.chat_message == "Second request"
    assert remaining.turn_id != turn1_id


@patch("atomic_agents.agents.atomic_agent.get_token_counter")
def test_trim_context_raises_when_single_turn_exceeds_limit(
    mock_get_token_counter, mock_instructor, mock_system_prompt_generator
):
    """When even one turn exceeds max_context_tokens, ValueError is raised."""
    history = ChatHistory()
    history.initialize_turn()
    history.add_message("user", BasicChatInputSchema(chat_message="Very long message that exceeds the limit"))

    mock_counter_instance = Mock()
    mock_get_token_counter.return_value = mock_counter_instance
    # Even after trimming all turns, still over limit
    mock_counter_instance.count_context.return_value = TokenCountResult(
        total=500, system_prompt=400, history=100, tools=0, model="gpt-5-mini", max_tokens=8192, utilization=0.06
    )

    config = AgentConfig(
        client=mock_instructor,
        model="gpt-5-mini",
        history=history,
        system_prompt_generator=mock_system_prompt_generator,
        max_context_tokens=100,  # artificially low
    )
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)

    with pytest.raises(ValueError, match="max_context_tokens.*smaller than the minimum"):
        agent._trim_context()


@patch("atomic_agents.agents.atomic_agent.get_token_counter")
def test_trim_context_preserves_new_user_message_in_run(mock_get_token_counter, mock_instructor, mock_system_prompt_generator):
    """Old turns can be trimmed while the new user input is preserved."""
    history = ChatHistory()

    # Existing turn in history
    history.initialize_turn()
    history.add_message("user", BasicChatInputSchema(chat_message="Old message"))

    mock_counter_instance = Mock()
    mock_get_token_counter.return_value = mock_counter_instance
    mock_counter_instance.count_context.side_effect = [
        TokenCountResult(
            total=500, system_prompt=100, history=400, tools=0, model="gpt-5-mini", max_tokens=8192, utilization=0.06
        ),
        TokenCountResult(
            total=50, system_prompt=100, history=0, tools=0, model="gpt-5-mini", max_tokens=8192, utilization=0.006
        ),
    ]

    config = AgentConfig(
        client=mock_instructor,
        model="gpt-5-mini",
        history=history,
        system_prompt_generator=mock_system_prompt_generator,
        max_context_tokens=200,
    )
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](config)

    # Mock chat completion
    mock_instructor.chat.completions.create.return_value = BasicChatOutputSchema(chat_message="Response")

    agent.run(BasicChatInputSchema(chat_message="New message"))

    # The old turn was trimmed and the new message was preserved.
    assert history.get_message_count() == 2  # assistant response + new user message
    assert history.history[0].content.chat_message == "New message"  # new message is first


# --- Test BaseSystemPromptGenerator integration ---


def test_custom_system_prompt_generator_reaches_agent(
    mock_instructor, mock_custom_system_prompt_generator, mock_context_provider
):

    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
        AgentConfig(
            client=mock_instructor,
            model="gpt-5-mini",
            system_prompt_generator=mock_custom_system_prompt_generator,
        )
    )

    agent.register_context_provider("test_provider", mock_context_provider)

    assert agent.system_prompt_generator == mock_custom_system_prompt_generator
    assert agent._build_system_messages() == [{"content": "Custom Prompt", "role": "system"}]
    assert agent.system_prompt_generator.context_providers == {"test_provider": mock_context_provider}
