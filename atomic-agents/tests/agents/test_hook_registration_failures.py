"""Exercise failed registrations through Instructor's native hook registry."""

from unittest.mock import Mock

import instructor
import pytest

from atomic_agents import AgentConfig, AtomicAgent


@pytest.fixture(params=[instructor.Instructor, instructor.AsyncInstructor], ids=["sync", "async"])
def agent(request):
    client = request.param(client=Mock(), create=Mock())
    return AtomicAgent(AgentConfig(client=client))


def test_rejected_registration_does_not_poison_later_hook_operations(agent):
    rejected = Mock()
    with pytest.raises(ValueError, match="Invalid hook name"):
        agent.register_hook("invalid:event", rejected)

    assert "invalid:event" not in agent._hook_handlers
    owned = Mock()
    agent.register_hook("completion:kwargs", owned)
    agent.client.hooks.emit_completion_arguments(model="before")
    owned.assert_called_once_with(model="before")

    agent.disable_hooks()
    agent.client.hooks.emit_completion_arguments(model="paused")
    assert owned.call_count == 1
    agent.clear_hooks()
    agent.enable_hooks()
    agent.client.hooks.emit_completion_arguments(model="cleared")
    assert owned.call_count == 1
    rejected.assert_not_called()


@pytest.mark.parametrize("invalid_first", [False, True], ids=["rollback", "nothing-attached"])
def test_failed_resume_remains_disabled_and_can_be_retried(agent, invalid_first):
    foreign = Mock()
    owned = Mock()
    local = Mock()
    rejected = Mock()
    agent.client.on("completion:kwargs", foreign)
    agent.disable_hooks()
    agent.register_hook("token:counted", local)
    if invalid_first:
        agent.register_hook("invalid:event", rejected)
    # Duplicate callbacks must each be detached after a failed resume.
    agent.register_hook("completion:kwargs", owned)
    agent.register_hook("completion:kwargs", owned)
    if not invalid_first:
        agent.register_hook("invalid:event", rejected)

    with pytest.raises(ValueError, match="Invalid hook name"):
        agent.enable_hooks()

    assert not agent.hooks_enabled
    agent.client.hooks.emit_completion_arguments(model="failed-resume")
    agent._dispatch_hook("token:counted", "failed-resume")
    owned.assert_not_called()
    local.assert_not_called()
    foreign.assert_called_once_with(model="failed-resume")

    agent.unregister_hook("invalid:event", rejected)
    agent.enable_hooks()
    agent.enable_hooks()
    assert agent.hooks_enabled
    agent.client.hooks.emit_completion_arguments(model="recovered")
    agent._dispatch_hook("token:counted", "recovered")
    assert owned.call_count == 2
    owned.assert_called_with(model="recovered")
    local.assert_called_once_with("recovered")
    assert foreign.call_count == 2
    rejected.assert_not_called()
