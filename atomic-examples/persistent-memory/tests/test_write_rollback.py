"""Use SQLite's own write and commit locks to check in-memory rollback."""

import importlib.util
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from atomic_agents import BasicChatInputSchema


@pytest.fixture
def history_class(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test")
    path = Path(__file__).resolve().parents[1] / "persistent_memory" / "main.py"
    spec = importlib.util.spec_from_file_location("sqlite_write_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SQLiteChatHistory


@pytest.fixture
def immediate_sqlite(monkeypatch):
    """Keep real connections; shorten the busy timeout and record native statements."""
    connect = sqlite3.connect
    statements = []

    def immediate(*args, **kwargs):
        kwargs["timeout"] = 0
        connection = connect(*args, **kwargs)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(sqlite3, "connect", immediate)
    return statements


@pytest.mark.parametrize("capacity", [None, 1], ids=["unbounded", "evicts-oldest"])
@pytest.mark.parametrize("lock_stage", ["write", "commit"])
def test_failed_database_write_restores_history_and_allows_retry(
    history_class, immediate_sqlite, tmp_path, capacity, lock_stage
):
    path = tmp_path / "history.sqlite"
    history = history_class("main", str(path), max_messages=capacity)
    history.add_message("user", BasicChatInputSchema(chat_message="saved"))
    before = history.dump()
    original_messages = history.history
    original_message = history.history[0]
    blocker = sqlite3.connect(path)
    try:
        if lock_stage == "write":
            blocker.execute("BEGIN IMMEDIATE")
        else:
            blocker.execute("BEGIN")
            blocker.execute("SELECT * FROM sessions").fetchall()
        immediate_sqlite.clear()
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            history.add_message("user", BasicChatInputSchema(chat_message="retry once"))
        assert ("COMMIT" in immediate_sqlite) == (lock_stage == "commit")
        assert history.dump() == before
        assert history.history is original_messages
        assert history.history[0] is original_message
    finally:
        blocker.rollback()
        blocker.close()

    assert history_class("main", str(path)).dump() == before
    history.add_message("user", BasicChatInputSchema(chat_message="retry once"))
    restored = history_class("main", str(path))
    assert restored.dump() == history.dump()
    assert [message.content.chat_message for message in restored.history].count("retry once") == 1


def test_first_failed_write_does_not_leave_a_phantom_turn(history_class, immediate_sqlite, tmp_path):
    path = tmp_path / "history.sqlite"
    history = history_class("empty", str(path))
    blocker = sqlite3.connect(path)
    try:
        blocker.execute("BEGIN IMMEDIATE")
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            history.add_message("user", BasicChatInputSchema(chat_message="not saved"))
        assert history.history == []
        assert history.current_turn_id is None
    finally:
        blocker.rollback()
        blocker.close()
    assert history_class("empty", str(path)).get_message_count() == 0


def test_concurrent_failed_writes_do_not_restore_each_others_unsaved_messages(history_class, immediate_sqlite, tmp_path):
    path = tmp_path / "history.sqlite"
    history = history_class("shared", str(path))
    history.add_message("user", BasicChatInputSchema(chat_message="saved"))
    before = history.dump()
    ready = threading.Barrier(2)

    def fail_write(text):
        ready.wait(timeout=5)
        with pytest.raises(sqlite3.OperationalError, match="locked"):
            history.add_message("user", BasicChatInputSchema(chat_message=text))

    blocker = sqlite3.connect(path)
    try:
        blocker.execute("BEGIN IMMEDIATE")
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(fail_write, text) for text in ("failed A", "failed B")]
            for future in futures:
                future.result(timeout=5)
        assert history.dump() == before
    finally:
        blocker.rollback()
        blocker.close()
    assert history_class("shared", str(path)).dump() == before


def test_successful_writes_still_persist_and_isolate_sessions(history_class, tmp_path):
    path = tmp_path / "history.sqlite"
    first = history_class("first", str(path))
    second = history_class("second", str(path))
    first.add_message("user", BasicChatInputSchema(chat_message="first message"))
    second.add_message("user", BasicChatInputSchema(chat_message="second message"))
    assert history_class("first", str(path)).dump() == first.dump()
    assert history_class("second", str(path)).dump() == second.dump()
