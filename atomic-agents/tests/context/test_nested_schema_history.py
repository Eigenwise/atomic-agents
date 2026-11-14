"""History persistence retains importable nested schema identities."""

import importlib
import json
import sys

import pytest

from atomic_agents import BaseIOSchema
from atomic_agents.context import ChatHistory


class Content(BaseIOSchema):
    """A top-level schema sharing the nested schemas' simple class name."""

    text: str


class First:
    class Content(BaseIOSchema):
        """Content owned by the first component."""

        text: str


class Second:
    class Content(BaseIOSchema):
        """Content owned by the second component."""

        text: str


class Deep:
    class Inner:
        class Content(BaseIOSchema):
            """Content nested more than one level deep."""

            text: str


@pytest.mark.parametrize("schema", [Content, First.Content, Second.Content, Deep.Inner.Content])
def test_round_trip_preserves_exact_schema_class(schema):
    original = ChatHistory(max_messages=4)
    original.add_message("user", schema(text="saved conversation"))
    saved = original.dump()
    restored = ChatHistory()
    restored.load(saved)

    assert type(restored.history[0].content) is schema
    assert restored.get_history() == original.get_history()
    assert restored.current_turn_id == original.current_turn_id
    assert restored.max_messages == 4
    assert json.loads(saved)["history"][0]["content"]["class_name"] == f"{schema.__module__}.{schema.__qualname__}"


def test_nested_schema_module_can_be_reimported_after_saving(tmp_path, monkeypatch):
    module_name = "nested_history_fixture"
    (tmp_path / f"{module_name}.py").write_text(
        "from atomic_agents import BaseIOSchema\n"
        "class Component:\n"
        "    class Message(BaseIOSchema):\n"
        '        """A durable component message."""\n'
        "        text: str\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        module = importlib.import_module(module_name)
        original = ChatHistory()
        original.add_message("assistant", module.Component.Message(text="persisted"))
        saved = original.dump()
        del sys.modules[module_name]

        restored = ChatHistory()
        restored.load(saved)
        reloaded = importlib.import_module(module_name)
        assert type(restored.history[0].content) is reloaded.Component.Message
        assert restored.history[0].content.text == "persisted"
    finally:
        sys.modules.pop(module_name, None)


def test_exported_local_schema_keeps_legacy_path(monkeypatch):
    class ExportedContent(BaseIOSchema):
        """A local schema explicitly exported by its defining module."""

        text: str

    monkeypatch.setattr(sys.modules[__name__], "ExportedContent", ExportedContent, raising=False)
    original = ChatHistory()
    original.add_message("user", ExportedContent(text="legacy export"))
    saved = original.dump()
    restored = ChatHistory()
    restored.load(saved)

    assert type(restored.history[0].content) is ExportedContent
    assert json.loads(saved)["history"][0]["content"]["class_name"] == f"{__name__}.ExportedContent"


def test_missing_nested_class_keeps_existing_history():
    history = ChatHistory()
    history.add_message("user", Content(text="keep me"))
    original = history.dump()
    broken = json.loads(original)
    broken["history"][0]["content"]["class_name"] = f"{__name__}.First.Missing"

    with pytest.raises(ValueError, match="Invalid serialized data"):
        history.load(json.dumps(broken))
    assert history.dump() == original


def test_schema_module_dependency_failure_is_not_hidden(tmp_path, monkeypatch):
    (tmp_path / "broken_history_fixture.py").write_text("import missing_history_dependency\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ModuleNotFoundError) as error:
        ChatHistory._get_class_from_string("broken_history_fixture.Component.Message")
    assert error.value.name == "missing_history_dependency"
