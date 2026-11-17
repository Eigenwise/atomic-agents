"""History persistence retains importable nested schema identities."""

import importlib
import json
import sys
from types import SimpleNamespace

import pytest
from pydantic import Json

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


class NonSchema:
    value = object()

    class Content:
        pass


class Typed:
    class Content(BaseIOSchema):
        """A nested schema requiring round-trip JSON serialization."""

        values: Json[list[int]]


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


@pytest.mark.parametrize("name", ["NonSchema.value", "NonSchema.Content"])
def test_nested_non_schema_objects_cannot_replace_existing_history(name):
    history = ChatHistory()
    history.add_message("user", Content(text="keep me"))
    original = history.dump()
    broken = json.loads(original)
    broken["history"][0]["content"]["class_name"] = f"{__name__}.{name}"

    with pytest.raises(ValueError, match="is not a BaseIOSchema subclass"):
        history.load(json.dumps(broken))
    assert history.dump() == original


@pytest.mark.parametrize("nested", [False, True], ids=["module-attribute", "class-attribute"])
@pytest.mark.parametrize("dynamic", [False, True], ids=["stored-attribute", "dynamic-attribute"])
def test_loading_does_not_traverse_non_class_objects(monkeypatch, nested, dynamic):
    visited = []

    class DynamicNamespace:
        def __getattr__(self, name):
            visited.append(name)
            return Content

    namespace = DynamicNamespace() if dynamic else SimpleNamespace(Content=Content)
    parent = First if nested else sys.modules[__name__]
    monkeypatch.setattr(parent, "namespace", namespace, raising=False)
    path = f"{__name__}." + ("First." if nested else "") + "namespace.Content"
    history = ChatHistory()
    history.add_message("user", Content(text="keep me"))
    original = history.dump()
    broken = json.loads(original)
    broken["history"][0]["content"]["class_name"] = path

    with pytest.raises(ValueError, match="Invalid serialized data"):
        history.load(json.dumps(broken))
    assert visited == []
    assert history.dump() == original


def test_nested_schema_preserves_round_trip_json_fields():
    original = ChatHistory()
    original.add_message("user", Typed.Content(values="[1, 2, 3]"))
    restored = ChatHistory()
    restored.load(original.dump())
    assert type(restored.history[0].content) is Typed.Content
    assert restored.history[0].content.values == [1, 2, 3]
