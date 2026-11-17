import json

import pytest
from instructor.processing.multimodal import Audio, Image, PDF
from pydantic import Field

from atomic_agents import BaseIOSchema, VideoURL
from atomic_agents.context import ChatHistory


pytest.importorskip("pydantic", minversion="2.12", reason="Field.exclude_if requires Pydantic 2.12")


def is_private(media: Image | Audio | PDF | VideoURL) -> bool:
    source = media.url if isinstance(media, VideoURL) else str(media.source)
    return "private" in source


class ConditionalMedia(BaseIOSchema):
    """Message whose attachment visibility follows its current value."""

    caption: str = "public caption"
    media: Image | Audio | PDF | VideoURL = Field(exclude_if=is_private)


MEDIA_FACTORIES = [
    pytest.param(lambda source: Image(source=source, media_type="image/jpeg"), id="image"),
    pytest.param(lambda source: Audio(source=source, media_type="audio/mp3"), id="audio"),
    pytest.param(lambda source: PDF(source=source, media_type="application/pdf"), id="pdf"),
    pytest.param(lambda source: VideoURL(url=f"https://example.com/{source}.mp4"), id="video"),
]


@pytest.mark.parametrize("factory", MEDIA_FACTORIES)
def test_conditional_media_visibility_tracks_the_current_value(factory):
    content = ConditionalMedia(media=factory("private"))
    history = ChatHistory()
    history.add_message("user", content)

    assert content.model_dump() == {"caption": "public caption"}
    assert history.get_history() == [{"role": "user", "content": content.model_dump_json()}]

    content.media = factory("public")
    parts = history.get_history()[0]["content"]
    assert json.loads(parts[0]) == {"caption": "public caption"}
    assert parts[1:] == [content.media.to_openai() if isinstance(content.media, VideoURL) else content.media]

    content.media = factory("private-again")
    assert history.get_history() == [{"role": "user", "content": content.model_dump_json()}]


@pytest.mark.parametrize("factory", MEDIA_FACTORIES)
def test_false_predicate_keeps_visible_media(factory):
    content = ConditionalMedia(media=factory("public"))
    history = ChatHistory()
    history.add_message("user", content)

    parts = history.get_history()[0]["content"]
    assert json.loads(parts[0]) == {"caption": "public caption"}
    assert parts[1:] == [content.media.to_openai() if isinstance(content.media, VideoURL) else content.media]


@pytest.mark.parametrize("container", [list, tuple, lambda values: dict(enumerate(values))], ids=["list", "tuple", "dict"])
def test_conditionally_excluded_container_is_not_traversed(container):
    class Message(BaseIOSchema):
        """Message with a conditionally hidden attachment collection."""

        visible: Image
        originals: object = Field(exclude_if=lambda value: len(value) > 1)

    visible = Image(source="visible.jpg", media_type="image/jpeg")
    private = Image(source="private.jpg", media_type="image/jpeg")
    content = Message(visible=visible, originals=container([private, private]))
    history = ChatHistory()
    history.add_message("user", content)

    assert "originals" not in content.model_dump()
    assert history.get_history() == [{"role": "user", "content": [visible]}]


def test_nested_conditional_media_keeps_other_attachments_in_order():
    class Documents(BaseIOSchema):
        """Documents with independently evaluated attachment visibility."""

        documents: list[ConditionalMedia]

    first = Image(source="first.jpg", media_type="image/jpeg")
    last = Image(source="last.jpg", media_type="image/jpeg")
    content = Documents(
        documents=[
            ConditionalMedia(caption="first", media=first),
            ConditionalMedia(caption="summary only", media=Image(source="private.jpg", media_type="image/jpeg")),
            ConditionalMedia(caption="last", media=last),
        ]
    )
    history = ChatHistory()
    history.add_message("user", content)

    parts = history.get_history()[0]["content"]
    assert json.loads(parts[0]) == {"documents": [{"caption": "first"}, {"caption": "summary only"}, {"caption": "last"}]}
    assert parts[1:] == [first, last]


def test_unconditional_exclusion_takes_precedence_over_predicate():
    def unexpected_predicate(_value):
        raise AssertionError("exclude=True must skip the predicate")

    class Message(BaseIOSchema):
        """Message with an unconditional exclusion and an unused predicate."""

        caption: str = "public"
        original: Image = Field(exclude=True, exclude_if=unexpected_predicate)

    content = Message(original=Image(source="private.jpg", media_type="image/jpeg"))
    history = ChatHistory()
    history.add_message("user", content)

    assert history.get_history() == [{"role": "user", "content": content.model_dump_json()}]
