import json

import pytest
from instructor.processing.multimodal import Image
from pydantic import RootModel

from atomic_agents.base.base_io_schema import BaseIOSchema
from atomic_agents.context.chat_history import ChatHistory


class MediaRoot(BaseIOSchema, RootModel[object]):
    """A root payload containing media and optional text."""


class MediaMessage(BaseIOSchema):
    """A message with a nested root payload."""

    instruction: str = "Describe"
    media: MediaRoot


@pytest.mark.parametrize("nested", [False, True], ids=["message-root", "nested-root"])
@pytest.mark.parametrize("shape", ["single", "list", "dict", "nested"], ids=str)
def test_root_models_exclude_extracted_media_from_text(nested, shape):
    image = Image(source="data:image/png;base64,aW1hZ2U=", media_type="image/png")
    if shape == "single":
        payload, remaining = image, None
    elif shape == "list":
        payload, remaining = [image, "caption"], ["caption"]
    elif shape == "dict":
        payload, remaining = {"image": image, "caption": "caption"}, {"caption": "caption"}
    else:
        payload, remaining = MediaRoot([image, "caption"]), ["caption"]
    root = MediaRoot(payload)
    message = MediaMessage(media=root) if nested else root
    history = ChatHistory()
    history.add_message("user", message)

    parts = history.get_history()[0]["content"]
    assert parts[-1] is image
    if nested:
        expected = {"instruction": "Describe"}
        if remaining is not None:
            expected["media"] = remaining
        assert len(parts) == 2
        assert json.loads(parts[0]) == expected
    elif remaining is None:
        assert parts == [image]
    else:
        assert len(parts) == 2
        assert json.loads(parts[0]) == remaining


def test_root_model_without_media_keeps_its_json_payload():
    history = ChatHistory()
    message = MediaRoot(["caption", {"count": 2}])
    history.add_message("user", message)

    assert history.get_history() == [{"role": "user", "content": message.model_dump_json()}]
