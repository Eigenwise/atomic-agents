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
