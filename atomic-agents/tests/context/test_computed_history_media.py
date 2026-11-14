import json
from functools import cached_property
from typing import Any

import pytest
from instructor import Image, Mode
from instructor.processing.multimodal import Audio, PDF, convert_messages
from pydantic import Field, PrivateAttr, computed_field

from atomic_agents import BaseIOSchema, VideoURL
from atomic_agents.context import ChatHistory


def make_media(kind):
    if kind == "image":
        return Image(source="https://example.test/chart.png", media_type="image/png", data="aGVsbG8=")
    if kind == "audio":
        return Audio(source="https://example.test/audio.wav", media_type="audio/wav", data="aGVsbG8=")
    if kind == "pdf":
        return PDF(source="https://example.test/report.pdf", media_type="application/pdf", data="aGVsbG8=")
    return VideoURL(url="https://example.test/video.mp4", fps=1.0)


class DerivedAttachment(BaseIOSchema):
    """A tool result that derives its attachment from an internal source."""

    caption: str = "Results"
    source: Any = Field(exclude=True)
    _reads: int = PrivateAttr(default=0)

    @computed_field(alias="asset")
    @cached_property
    def attachment(self) -> Any:
        self._reads += 1
        return self.source


def rendered(content):
    history = ChatHistory()
    history.add_message("user", content)
    return history.get_history()


@pytest.mark.parametrize("kind", ["image", "audio", "pdf", "video"])
def test_computed_media_is_a_content_part(kind):
    media = make_media(kind)

    parts = rendered(DerivedAttachment(source=media))[0]["content"]

    assert isinstance(parts, list)
    assert json.loads(parts[0]) == {"caption": "Results"}
    assert parts[1:] == [media.to_openai() if isinstance(media, VideoURL) else media]


@pytest.mark.parametrize("container", ["list", "tuple", "dict"])
def test_computed_containers_keep_their_nonmedia_values(container):
    media = make_media("image")
    source = {"image": media, "caption": "retained"} if container == "dict" else [media, "retained"]
    if container == "tuple":
        source = tuple(source)

    parts = rendered(DerivedAttachment(source=source))[0]["content"]

    assert isinstance(parts, list)
    expected = {"caption": "retained"} if container == "dict" else ["retained"]
    assert json.loads(parts[0]) == {"caption": "Results", "attachment": expected}
    assert parts[1:] == [media]


@pytest.mark.parametrize("derived_parent", [False, True], ids=["declared-parent", "computed-parent"])
def test_nested_computed_media_is_extracted(derived_parent):
    class Envelope(BaseIOSchema):
        """A result that contains a nested tool response."""

        payload: DerivedAttachment

    media = make_media("image")
    inner = DerivedAttachment(caption="Inner", source=media)
    outer = DerivedAttachment(source=inner) if derived_parent else Envelope(payload=inner)

    parts = rendered(outer)[0]["content"]

    assert isinstance(parts, list)
    expected = (
        {"caption": "Results", "attachment": {"caption": "Inner"}} if derived_parent else {"payload": {"caption": "Inner"}}
    )
    assert json.loads(parts[0]) == expected
    assert parts[1:] == [media]


def test_computed_scalar_stays_in_the_json_message():
    assert json.loads(rendered(DerivedAttachment(source=42))[0]["content"]) == {"caption": "Results", "attachment": 42}


def test_excluded_parent_does_not_evaluate_or_expose_computed_media():
    class PrivateResult(BaseIOSchema):
        """A public status with an internal result that must not be sent."""

        status: str = "ready"
        internal: DerivedAttachment = Field(exclude=True)

    internal = DerivedAttachment(source=make_media("image"))

    assert rendered(PrivateResult(internal=internal)) == [{"role": "user", "content": '{"status":"ready"}'}]
    assert internal._reads == 0


def test_computed_media_getter_is_evaluated_once():
    content = DerivedAttachment(source=make_media("image"))

    rendered(content)

    assert content._reads == 1


@pytest.mark.parametrize("container", [False, True], ids=["scalar", "changing-container"])
def test_ordinary_computed_properties_keep_single_serialization_evaluation(container):
    class LiveResult(BaseIOSchema):
        """A changing computed value which must keep its normal serialization path."""

        _reads: int = PrivateAttr(default=0)

        @computed_field
        @property
        def value(self) -> Any:
            self._reads += 1
            if container:
                media = make_media("image")
                return [self._reads, media] if self._reads == 1 else [media, self._reads]
            return self._reads

    content = LiveResult()

    result = rendered(content)[0]["content"]

    assert isinstance(result, str)
    expected = [1, make_media("image").model_dump(mode="json")] if container else 1
    assert json.loads(result) == {"value": expected}
    assert content._reads == 1


def test_cached_none_stays_in_the_json_message_without_recomputation():
    content = DerivedAttachment(source=None)

    assert json.loads(rendered(content)[0]["content"]) == {"caption": "Results", "attachment": None}
    assert content._reads == 1


@pytest.mark.parametrize(
    "kind,part_type", [("image", "image_url"), ("audio", "input_audio"), ("pdf", "file"), ("video", "video_url")]
)
def test_instructor_receives_computed_media_in_provider_format(kind, part_type):
    media = make_media(kind)
    messages = rendered(DerivedAttachment(source=media))

    request_messages = convert_messages(messages, Mode.TOOLS)

    assert request_messages[0]["role"] == "user"
    parts = request_messages[0]["content"]
    assert isinstance(parts, list)
    assert parts[0] == {"type": "text", "text": '{"caption":"Results"}'}
    expected = media.to_openai() if isinstance(media, VideoURL) else media.to_openai(Mode.TOOLS)
    assert parts[1] == expected
    assert parts[1]["type"] == part_type
    assert len(parts) == 2
