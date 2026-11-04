from pathlib import Path

from instructor.processing.multimodal import PDF, Image

from atomic_agents import BaseIOSchema
from atomic_agents.context import ChatHistory


class TupleAttachments(BaseIOSchema):
    """A message with typed attachments held in tuples."""

    attachments: tuple[Image, PDF]
    nested: dict[str, tuple[tuple[Image, ...], ...]]


def test_tuple_attachment_paths_survive_history_round_trip():
    original = ChatHistory()
    original.add_message(
        "user",
        TupleAttachments(
            attachments=(
                Image(source=Path("image.png"), media_type="image/png"),
                PDF(source=Path("document.pdf"), media_type="application/pdf"),
            ),
            nested={"images": ((Image(source=Path("nested.png"), media_type="image/png"),),)},
        ),
    )

    restored = ChatHistory()
    restored.load(original.dump())
    content = restored.history[0].content

    assert content.attachments[0].source == Path("image.png")
    assert content.attachments[1].source == Path("document.pdf")
    assert content.nested["images"][0][0].source == Path("nested.png")
    assert isinstance(content.attachments, tuple)
    assert restored.get_current_turn_id() == original.get_current_turn_id()
    parts = restored.get_history()[0]["content"]
    assert len(parts) == 3
    assert all(isinstance(part.source, Path) for part in parts)


def test_tuple_url_and_data_sources_remain_strings():
    history = ChatHistory()
    attachments = (
        Image(source="https://example.com/image.png", media_type="image/png"),
        PDF(source="data:application/pdf;base64,JVBERi0=", media_type="application/pdf"),
    )

    history._process_multimodal_paths(attachments)

    assert attachments[0].source == "https://example.com/image.png"
    assert attachments[1].source == "data:application/pdf;base64,JVBERi0="
