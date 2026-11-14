from pathlib import Path
import runpy

import pytest

from atomic_agents import BasicChatInputSchema


@pytest.fixture
def sqlite_history_type(monkeypatch):
    # Import the example without starting its interactive client or requiring a real key.
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")
    example = Path(__file__).resolve().parents[3] / "atomic-examples/persistent-memory/persistent_memory/main.py"
    return runpy.run_path(str(example))["SQLiteChatHistory"]


@pytest.mark.parametrize("turn_count, deleted_index", [(2, 0), (2, 1), (1, 0)])
def test_deleted_turn_stays_deleted_after_reload(sqlite_history_type, tmp_path, turn_count, deleted_index):
    db_path = str(tmp_path / "history.db")
    history = sqlite_history_type("session", db_path=db_path)
    turns = []
    for i in range(turn_count):
        history.initialize_turn()
        turns.append(history.get_current_turn_id())
        history.add_message("user", BasicChatInputSchema(chat_message=f"Message {i}"))

    history.delete_turn_id(turns[deleted_index])
    restored = sqlite_history_type("session", db_path=db_path)

    assert restored.dump() == history.dump()
    assert all(message.turn_id != turns[deleted_index] for message in restored.history)
    assert restored.get_message_count() == turn_count - 1


def test_failed_turn_deletion_preserves_saved_history(sqlite_history_type, tmp_path):
    db_path = str(tmp_path / "history.db")
    history = sqlite_history_type("session", db_path=db_path)
    history.add_message("user", BasicChatInputSchema(chat_message="Keep this message"))
    saved = history.dump()

    with pytest.raises(ValueError, match="not found"):
        history.delete_turn_id("missing-turn")

    assert history.dump() == saved
    assert sqlite_history_type("session", db_path=db_path).dump() == saved


def test_turn_deletion_does_not_change_another_session(sqlite_history_type, tmp_path):
    db_path = str(tmp_path / "history.db")
    first = sqlite_history_type("first", db_path=db_path)
    second = sqlite_history_type("second", db_path=db_path)
    first.add_message("user", BasicChatInputSchema(chat_message="First session"))
    second.add_message("user", BasicChatInputSchema(chat_message="Second session"))
    saved_second = second.dump()

    first.delete_turn_id(first.get_current_turn_id())

    assert sqlite_history_type("first", db_path=db_path).get_message_count() == 0
    assert sqlite_history_type("second", db_path=db_path).dump() == saved_second
