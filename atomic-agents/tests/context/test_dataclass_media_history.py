import json
from dataclasses import InitVar, dataclass, field
from typing import Annotated, Any, ClassVar, Generic, TypeVar

import httpx
import instructor
import openai
import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    RootModel,
    computed_field,
    field_serializer,
    model_serializer,
)
from pydantic.dataclasses import dataclass as pydantic_dataclass

from atomic_agents import AgentConfig, AtomicAgent, BaseIOSchema, VideoURL
from atomic_agents.context import ChatHistory


@dataclass
class StandardAttachment:
    image: instructor.Image
    caption: str


@pydantic_dataclass
class ValidatedAttachment:
    image: instructor.Image
    caption: str


class StandardInput(BaseIOSchema):
    """A camera frame supplied through an application dataclass."""

    attachment: StandardAttachment


class ValidatedInput(BaseIOSchema):
    """A camera frame supplied through a validated dataclass."""

    attachment: ValidatedAttachment


class Output(BaseIOSchema):
    """The description returned by the model."""

    answer: str


def image(name="visible"):
    return instructor.Image(source=f"https://example.test/{name}.png", media_type="image/png")


def history_content(content):
    history = ChatHistory()
    history.add_message("user", content)
    return history.get_history()[0]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "input_type,attachment_type", [(StandardInput, StandardAttachment), (ValidatedInput, ValidatedAttachment)]
)
@pytest.mark.parametrize("from_mapping", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_dataclass_image_reaches_real_instructor_request(input_type, attachment_type, from_mapping, asynchronous):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-dataclass",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": '{"answer":"ok"}'}, "finish_reason": "stop"}
                ],
            },
        )

    attachment = {"image": image(), "caption": "A camera frame"}
    content = input_type(attachment=attachment if from_mapping else attachment_type(**attachment))
    if asynchronous:
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond), trust_env=False) as http_client:
            provider = openai.AsyncOpenAI(api_key="test-key", base_url="http://atomic.test/v1", http_client=http_client)
            client = instructor.from_openai(provider, mode=instructor.Mode.JSON)
            agent = AtomicAgent[input_type, Output](config=AgentConfig(client=client, model="gpt-4o"))
            assert (await agent.run_async(content)).answer == "ok"
    else:
        with httpx.Client(transport=httpx.MockTransport(respond), trust_env=False) as http_client:
            provider = openai.OpenAI(api_key="test-key", base_url="http://atomic.test/v1", http_client=http_client)
            client = instructor.from_openai(provider, mode=instructor.Mode.JSON)
            agent = AtomicAgent[input_type, Output](config=AgentConfig(client=client, model="gpt-4o"))
            assert agent.run(content).answer == "ok"

    assert len(requests) == 1
    user = next(message for message in requests[0]["messages"] if message["role"] == "user")
    assert isinstance(user["content"], list)
    assert user["content"][1:] == [{"type": "image_url", "image_url": {"url": "https://example.test/visible.png"}}]
    assert json.loads(user["content"][0]["text"]) == {"attachment": {"caption": "A camera frame"}}
    assert content.attachment.image.source == "https://example.test/visible.png"


@pytest.mark.parametrize("decorate", [dataclass, pydantic_dataclass])
def test_nested_dataclasses_and_containers_keep_field_order(decorate):
    @decorate(frozen=True, slots=True)
    class Attachment:
        media: instructor.Image
        caption: str

    @decorate
    class Collection:
        single: Attachment | None
        repeated: tuple[Attachment, ...]
        fixed: tuple[str, Attachment]
        mapping: dict[str, list[Attachment]]

    class Input(BaseIOSchema):
        """Attachments held in nested dataclasses and ordinary containers."""

        collection: Collection

    attachments = [Attachment(image(str(i)), f"caption {i}") for i in range(5)]
    content = Input(
        collection=Collection(
            attachments[0], (attachments[1], attachments[2]), ("label", attachments[3]), {"a": [attachments[4]]}
        )
    )
    parts = history_content(content)
    assert parts[1:] == [attachment.media for attachment in attachments]
    assert json.loads(parts[0]) == {
        "collection": {
            "single": {"caption": "caption 0"},
            "repeated": [{"caption": "caption 1"}, {"caption": "caption 2"}],
            "fixed": ["label", {"caption": "caption 3"}],
            "mapping": {"a": [{"caption": "caption 4"}]},
        }
    }


@pytest.mark.parametrize("decorate", [dataclass, pydantic_dataclass])
def test_dataclass_exclusions_use_resolved_annotations_and_metadata(decorate):
    HiddenImage = Annotated[instructor.Image, Field(exclude=True)]
    LocalImage = instructor.Image

    @decorate
    class Attachment:
        visible: instructor.Image
        annotated: HiddenImage
        metadata: "LocalImage" = field(metadata={"exclude": True})
        pydantic_field: instructor.Image = Field(exclude=True)

    class Input(BaseIOSchema):
        """A dataclass with several forms of private field metadata."""

        attachment: Attachment

    Input.model_rebuild(_types_namespace={"LocalImage": LocalImage})
    visible, private = image(), image("private")
    content = Input(attachment=Attachment(visible, private, private, private))
    assert history_content(content) == ['{"attachment":{}}', visible]
    assert "private" not in content.model_dump_json()


@pytest.mark.skipif(not hasattr(Field(), "exclude_if"), reason="Requires Pydantic conditional field exclusion")
@pytest.mark.parametrize("decorate", [dataclass, pydantic_dataclass])
@pytest.mark.parametrize("hidden", [False, True])
def test_dataclass_conditional_exclusion(decorate, hidden):
    @decorate
    class Attachment:
        image: instructor.Image = Field(exclude_if=lambda value: "private" in str(value.source))

    class Input(BaseIOSchema):
        """A dataclass attachment with conditional visibility."""

        attachment: Attachment

    content = Input(attachment=Attachment(image("private" if hidden else "visible")))
    parts = history_content(content)
    assert parts == '{"attachment":{}}' if hidden else parts == ['{"attachment":{}}', content.attachment.image]


def test_dataclass_declared_base_does_not_expose_subclass_or_init_only_fields():
    @dataclass
    class Attachment:
        image: instructor.Image
        private_class: ClassVar[instructor.Image] = image("class-private")
        private_init: InitVar[instructor.Image] = image("init-private")

    @dataclass
    class PrivateAttachment(Attachment):
        private_subclass: instructor.Image = field(default_factory=lambda: image("subclass-private"))

    class Input(BaseIOSchema):
        """An attachment serialized using its declared base dataclass."""

        attachment: Attachment

    content = Input(attachment=PrivateAttachment(image()))
    assert history_content(content) == ['{"attachment":{}}', content.attachment.image]


@pytest.mark.parametrize(
    "media",
    [
        instructor.processing.multimodal.Audio(source="sound.wav", media_type="audio/wav"),
        instructor.processing.multimodal.PDF(source="document.pdf", media_type="application/pdf"),
        VideoURL(url="https://example.test/video.mp4"),
    ],
)
def test_dataclass_supports_existing_media_types(media):
    @dataclass
    class Attachment:
        value: type(media)

    class Input(BaseIOSchema):
        """A dataclass carrying a supported non-image media type."""

        attachment: Attachment

    parts = history_content(Input(attachment=Attachment(media)))
    expected = media.to_openai() if isinstance(media, VideoURL) else media
    assert parts == ['{"attachment":{}}', expected]


def test_generic_dataclass_fields_use_their_own_compiled_serialization():
    T = TypeVar("T")

    @dataclass
    class Attachment(Generic[T]):
        value: T

    class Input(BaseIOSchema):
        """The same dataclass with different field serialization policies."""

        visible: Attachment[instructor.Image]
        private: Attachment[Annotated[instructor.Image, PlainSerializer(lambda value: "redacted", return_type=str)]]

    visible, private = image(), image("private")
    parts = history_content(Input(visible=Attachment(visible), private=Attachment(private)))
    assert parts == ['{"visible":{},"private":{"value":"redacted"}}', visible]


@pytest.mark.parametrize("serializer_location", ["outer", "field", "dataclass", "nested_model"])
@pytest.mark.parametrize("serialized", ["redacted", {"safe": "redacted"}, None])
def test_opaque_serializers_do_not_expose_dataclass_media_or_run_twice(serializer_location, serialized):
    calls = []

    class Nested(BaseModel):
        image: instructor.Image

        @model_serializer
        def redact(self):
            calls.append("nested_model")
            return serialized

    value_type = Nested if serializer_location == "nested_model" else instructor.Image

    @pydantic_dataclass
    class Attachment:
        value: value_type

        if serializer_location == "dataclass":

            @model_serializer
            def redact(self):
                calls.append("dataclass")
                return serialized

    class Input(BaseIOSchema):
        """An application controls the public representation of its attachment."""

        attachment: Attachment

        if serializer_location == "field":

            @field_serializer("attachment")
            def redact_field(self, value):
                calls.append("field")
                return serialized

        if serializer_location == "outer":

            @model_serializer
            def redact(self):
                calls.append("outer")
                return serialized

    value = Nested(image=image("private")) if serializer_location == "nested_model" else image("private")
    parts = history_content(Input(attachment=Attachment(value)))
    assert isinstance(parts, str)
    assert "private" not in parts
    assert calls == [serializer_location]


def test_default_dataclass_traversal_does_not_repeat_computed_fields():
    calls = []

    class Input(BaseIOSchema):
        """A normal computed field must still be serialized only once."""

        attachment: StandardAttachment

        @computed_field
        @property
        def summary(self) -> str:
            calls.append("summary")
            return "one frame"

    parts = history_content(Input(attachment=StandardAttachment(image(), "caption")))
    assert parts[1:] == [image()]
    assert json.loads(parts[0]) == {"attachment": {"caption": "caption"}, "summary": "one frame"}
    assert calls == ["summary"]


def test_dataclass_field_aliases_keep_caption_and_exclude_media_by_field_name():
    @pydantic_dataclass(config=ConfigDict(serialize_by_alias=True))
    class Attachment:
        image: instructor.Image = Field(serialization_alias="frame")
        caption: str = Field(serialization_alias="description")

    class Input(BaseIOSchema):
        """Dataclass aliases affect JSON names, not the exclusion path."""

        attachment: Attachment

    content = Input(attachment=Attachment(image(), "caption"))
    assert history_content(content) == ['{"attachment":{"description":"caption"}}', content.attachment.image]


def test_recursive_dataclass_uses_compiled_definition_references():
    @dataclass
    class Attachment:
        image: instructor.Image
        child: "Attachment | None" = None

    class Input(BaseIOSchema):
        """A finite tree of recursive dataclass attachments."""

        attachment: Attachment

    content = Input(attachment=Attachment(image("first"), Attachment(image("second"))))
    parts = history_content(content)
    assert parts == ['{"attachment":{"child":{"child":null}}}', image("first"), image("second")]


@pytest.mark.parametrize("dynamic_type", [False, True])
def test_unresolved_serialization_layout_keeps_media_as_text(dynamic_type):
    class PrivateImage(instructor.Image):
        @model_serializer
        def redact(self):
            return "redacted"

    media_type = (
        Any if dynamic_type else Annotated[instructor.Image, PlainSerializer(lambda value: "redacted", return_type=str)] | str
    )

    @dataclass
    class Attachment:
        media: media_type

    class Input(BaseIOSchema):
        """Opaque union or dynamic serializers must not be bypassed."""

        attachment: Attachment

    media = (
        PrivateImage(source="https://example.test/private.png", media_type="image/png") if dynamic_type else image("private")
    )
    content = Input(attachment=Attachment(media))
    assert content.model_dump_json() == '{"attachment":{"media":"redacted"}}'
    assert history_content(content) == content.model_dump_json()


@pytest.mark.parametrize("inside_dataclass", [False, True])
def test_root_model_dataclass_layout_keeps_existing_text_behavior(inside_dataclass):
    wrapped_type = RootModel[StandardAttachment]

    @dataclass
    class Container:
        value: wrapped_type

    attachment_type = Container if inside_dataclass else wrapped_type

    class Input(BaseIOSchema):
        """A root model does not have the normal model-fields layout."""

        attachment: attachment_type

    wrapped = wrapped_type(StandardAttachment(image(), "caption"))
    content = Input(attachment=Container(wrapped) if inside_dataclass else wrapped)
    assert history_content(content) == content.model_dump_json()
