import instructor
import pytest
from unittest.mock import Mock

from atomic_agents import AgentConfig, AtomicAgent


@pytest.mark.parametrize("event", [None, "completion:kwargs"])
def test_clear_hooks_preserves_foreign_handlers(event):
    client = instructor.Instructor(client=Mock(), create=Mock())
    agent = AtomicAgent(AgentConfig(client=client))
    foreign = Mock()
    owned = Mock()
    other_owned = Mock()
    client.on("completion:kwargs", foreign)
    agent.register_hook("completion:kwargs", owned)
    agent.register_hook("completion:response", other_owned)

    agent.clear_hooks(event)
    client.hooks.emit_completion_arguments(model="test-model")
    client.hooks.emit_completion_response(Mock())

    foreign.assert_called_once_with(model="test-model")
    owned.assert_not_called()
    assert other_owned.call_count == (0 if event is None else 1)
    assert agent._hook_handlers.get("completion:kwargs", []) == []


def test_instructor_hooks_stop_and_resume_without_duplicates():
    client = instructor.Instructor(client=Mock(), create=Mock())
    agent = AtomicAgent(AgentConfig(client=client))
    foreign = Mock()
    owned = Mock()
    client.on("completion:kwargs", foreign)
    agent.register_hook("completion:kwargs", owned)

    client.hooks.emit_completion_arguments(model="before")
    owned.assert_called_once_with(model="before")

    agent.disable_hooks()
    agent.disable_hooks()
    client.hooks.emit_completion_arguments(model="disabled")
    assert owned.call_count == 1
    assert foreign.call_count == 2

    agent.enable_hooks()
    agent.enable_hooks()
    client.hooks.emit_completion_arguments(model="resumed")
    assert owned.call_count == 2
    owned.assert_called_with(model="resumed")
    assert foreign.call_count == 3


def test_instructor_hooks_registered_while_disabled_wait_for_enable():
    client = instructor.Instructor(client=Mock(), create=Mock())
    agent = AtomicAgent(AgentConfig(client=client))
    handler = Mock()
    agent.disable_hooks()
    agent.register_hook("completion:kwargs", handler)

    client.hooks.emit_completion_arguments(model="disabled")
    handler.assert_not_called()

    agent.enable_hooks()
    client.hooks.emit_completion_arguments(model="resumed")
    handler.assert_called_once_with(model="resumed")


@pytest.mark.parametrize("remove", ["unregister", "clear_event", "clear_all"])
def test_removing_disabled_instructor_hooks_does_not_restore_them(remove):
    client = instructor.Instructor(client=Mock(), create=Mock())
    agent = AtomicAgent(AgentConfig(client=client))
    removed = Mock()
    retained = Mock()
    agent.register_hook("completion:kwargs", removed)
    agent.register_hook("completion:response", retained)
    agent.disable_hooks()

    if remove == "unregister":
        agent.unregister_hook("completion:kwargs", removed)
    elif remove == "clear_event":
        agent.clear_hooks("completion:kwargs")
    else:
        agent.clear_hooks()

    agent.enable_hooks()
    client.hooks.emit_completion_arguments(model="resumed")
    client.hooks.emit_completion_response("response")
    removed.assert_not_called()
    assert retained.call_count == (0 if remove == "clear_all" else 1)


@pytest.fixture
def token_hook_agent(monkeypatch):
    """Use real Instructor hook validation without making provider/tokenizer calls."""
    from atomic_agents.utils.token_counter import TokenCountResult

    client = instructor.Instructor(client=Mock(), create=Mock())
    agent = AtomicAgent(AgentConfig(client=client))
    result = TokenCountResult(total=100, system_prompt=30, history=70, tools=0, model="gpt-5-mini")
    counter = Mock()
    counter.count_context.return_value = result
    monkeypatch.setattr("atomic_agents.agents.atomic_agent.get_token_counter", lambda: counter)
    return agent, result


@pytest.mark.parametrize("initially_disabled", [False, True])
def test_local_token_hook_lifecycle_with_real_instructor(token_hook_agent, initially_disabled):
    agent, result = token_hook_agent
    local = Mock()
    native = Mock()
    foreign = Mock()
    agent.client.on("completion:kwargs", foreign)
    if initially_disabled:
        agent.disable_hooks()
    agent.register_hook("token:counted", local)
    agent.register_hook("completion:kwargs", native)
    if initially_disabled:
        agent.get_context_token_count()
        local.assert_not_called()
        agent.enable_hooks()

    assert agent.get_context_token_count() == result
    local.assert_called_once_with(result)
    agent.client.hooks.emit_completion_arguments(model="before")
    native.assert_called_once_with(model="before")

    agent.disable_hooks()
    agent.disable_hooks()
    agent.get_context_token_count()
    agent.client.hooks.emit_completion_arguments(model="disabled")
    assert local.call_count == native.call_count == 1
    assert foreign.call_count == 2

    agent.enable_hooks()
    agent.enable_hooks()
    agent.get_context_token_count()
    agent.client.hooks.emit_completion_arguments(model="resumed")
    assert local.call_count == native.call_count == 2
    assert foreign.call_count == 3


@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize("remove", ["unregister", "clear_event", "clear_all"])
def test_remove_local_token_hook_with_real_instructor(token_hook_agent, disabled, remove):
    agent, _ = token_hook_agent
    local = Mock()
    native = Mock()
    foreign = Mock()
    agent.client.on("completion:kwargs", foreign)
    agent.register_hook("token:counted", local)
    agent.register_hook("completion:kwargs", native)
    if disabled:
        agent.disable_hooks()
    if remove == "unregister":
        agent.unregister_hook("token:counted", local)
    elif remove == "clear_event":
        agent.clear_hooks("token:counted")
    else:
        agent.clear_hooks()
    if disabled:
        agent.enable_hooks()
    agent.get_context_token_count()
    agent.client.hooks.emit_completion_arguments(model="after")
    local.assert_not_called()
    foreign.assert_called_once_with(model="after")
    assert native.call_count == (0 if remove == "clear_all" else 1)


def test_duplicate_local_token_hooks_with_real_instructor(token_hook_agent):
    agent, result = token_hook_agent
    local = Mock()
    agent.register_hook("token:counted", local)
    agent.register_hook("token:counted", local)
    agent.disable_hooks()
    agent.enable_hooks()
    agent.get_context_token_count()
    assert local.call_count == 2
    local.assert_called_with(result)
    agent.unregister_hook("token:counted", local)
    agent.get_context_token_count()
    assert local.call_count == 3
    agent.clear_hooks("token:counted")
    agent.get_context_token_count()
    assert local.call_count == 3


@pytest.mark.parametrize("raises", [False, True])
def test_local_token_hook_can_unregister_itself_without_skipping_other_handlers(token_hook_agent, raises):
    agent, result = token_hook_agent
    calls = []

    def once(value):
        calls.append(("once", value))
        agent.unregister_hook("token:counted", once)
        if raises:
            raise ValueError("one-shot callback failed")

    def retained(value):
        calls.append(("retained", value))

    agent.register_hook("token:counted", once)
    agent.register_hook("token:counted", retained)

    assert agent.get_context_token_count() == result
    assert calls == [("once", result), ("retained", result)]
    assert agent.get_context_token_count() == result
    assert calls == [("once", result), ("retained", result), ("retained", result)]


def test_local_token_hook_added_by_callback_waits_until_next_event(token_hook_agent):
    agent, result = token_hook_agent
    calls = []
    added = False

    def late(value):
        calls.append(("late", value))

    def register_late(value):
        nonlocal added
        calls.append(("register", value))
        if not added:
            added = True
            agent.register_hook("token:counted", late)

    agent.register_hook("token:counted", register_late)

    assert agent.get_context_token_count() == result
    assert calls == [("register", result)]
    assert agent.get_context_token_count() == result
    assert calls == [("register", result), ("register", result), ("late", result)]


@pytest.mark.parametrize("event", [None, "token:counted"])
def test_clearing_local_token_hooks_during_dispatch_takes_effect_on_next_event(token_hook_agent, event):
    agent, result = token_hook_agent
    retained = Mock()

    def clear(value):
        agent.clear_hooks(event)

    agent.register_hook("token:counted", clear)
    agent.register_hook("token:counted", retained)

    assert agent.get_context_token_count() == result
    retained.assert_called_once_with(result)
    assert agent.get_context_token_count() == result
    retained.assert_called_once_with(result)
