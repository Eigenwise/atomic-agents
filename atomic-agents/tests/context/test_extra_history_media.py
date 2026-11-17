import json
from pathlib import Path
from typing import Any

import pytest
from instructor import Image, Mode
from instructor.processing.multimodal import Audio, PDF, convert_messages
from pydantic import ConfigDict, Field

from atomic_agents import BaseIOSchema, VideoURL
from atomic_agents.context import ChatHistory


class ExtensibleResult(BaseIOSchema):
    """A tool result that accepts provider-specific extension fields."""

    model_config = ConfigDict(extra="allow")
    caption: str = "Result"


class ExtraImages(BaseIOSchema):
    """A persisted collection of named image attachments."""

    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, Image] = Field(init=False)


class ExtraDocuments(BaseIOSchema):
    """A persisted collection of named PDF attachments."""

    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, PDF] = Field(init=False)


def media_for(kind):
    if kind == "image":
        return Image(source="https://example.test/chart.png", media_type="image/png", data="aGVsbG8=")
    if kind == "audio":
        return Audio(source="https://example.test/audio.wav", media_type="audio/wav", data="aGVsbG8=")
    if kind == "pdf":
        return PDF(source="https://example.test/report.pdf", media_type="application/pdf", data="aGVsbG8=")
    return VideoURL(url="https://example.test/video.mp4", fps=1.0)


def history_for(content):
    history = ChatHistory()
    history.add_message("user", content)
    return history


@pytest.mark.parametrize("kind", ["image", "audio", "pdf", "video"])
def test_extra_media_reaches_instructor_as_a_provider_content_part(kind):
    media = media_for(kind)
    content = ExtensibleResult(attachment=media, count=3)
    original_json = content.model_dump_json()

    messages = convert_messages(history_for(content).get_history(), Mode.TOOLS)

    expected = media.to_openai() if isinstance(media, VideoURL) else media.to_openai(Mode.TOOLS)
    assert messages == [{"role": "user", "content": [{"type": "text", "text": '{"caption":"Result","count":3}'}, expected]}]
    assert content.model_dump_json() == original_json
    assert content.model_extra["attachment"] is media


@pytest.mark.parametrize("container", ["list", "tuple", "dict", "model"])
def test_nested_extra_media_preserves_nonmedia_extension_values(container):
    image = media_for("image")
    values = {
        "list": [image, "retained"],
        "tuple": (image, "retained"),
        "dict": {"image": image, "label": "retained"},
        "model": ExtensibleResult(caption="Inner", image=image),
    }
    expected = {
        "list": ["retained"],
        "tuple": ["retained"],
        "dict": {"label": "retained"},
        "model": {"caption": "Inner"},
    }

    parts = history_for(ExtensibleResult(payload=values[container])).get_history()[0]["content"]

    assert isinstance(parts, list)
    assert json.loads(parts[0]) == {"caption": "Result", "payload": expected[container]}
    assert parts[1:] == [image]


def test_declared_and_extra_media_keep_their_order():
    class CombinedResult(ExtensibleResult):
        """A declared attachment followed by extension attachments."""

        image: Image

    declared, extra = media_for("image"), media_for("pdf")
    parts = history_for(CombinedResult(image=declared, extra=extra)).get_history()[0]["content"]

    assert parts[1:] == [declared, extra]


def test_excluded_parent_does_not_expose_extra_media():
    class PrivateResult(BaseIOSchema):
        """A public status containing an excluded internal result."""

        status: str = "ready"
        internal: ExtensibleResult = Field(exclude=True)

    content = PrivateResult(internal=ExtensibleResult(image=media_for("image")))

    assert history_for(content).get_history() == [{"role": "user", "content": '{"status":"ready"}'}]


@pytest.mark.parametrize("extra", [None, 0, {"nested": ["text", 1]}])
def test_nonmedia_extras_keep_native_json_serialization(extra: Any):
    content = ExtensibleResult(extra=extra)

    assert history_for(content).get_history() == [{"role": "user", "content": content.model_dump_json()}]


@pytest.mark.parametrize(
    "schema,media_type,kind", [(ExtraImages, "image/png", Image), (ExtraDocuments, "application/pdf", PDF)]
)
def test_typed_extra_media_survives_history_persistence(schema, media_type, kind):
    attachment = kind(source=Path("local-attachment"), media_type=media_type, data="aGVsbG8=")
    original = history_for(schema(attachment=attachment))
    restored = ChatHistory()

    restored.load(original.dump())

    media = restored.history[0].content.model_extra["attachment"]
    assert media.source == Path("local-attachment")
    assert isinstance(media.source, Path)
    assert restored.get_history() == original.get_history()
