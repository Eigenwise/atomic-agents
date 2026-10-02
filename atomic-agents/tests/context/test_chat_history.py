from enum import Enum

import pytest
import json
from typing import List, Dict, Union
from pathlib import Path
from pydantic import Field
from atomic_agents.context import ChatHistory, Message
from atomic_agents import BaseIOSchema, VideoURL
import instructor


class InputSchema(BaseIOSchema):
    """Test Input Schema"""

    test_field: str = Field(..., description="A test field")


class MockOutputSchema(BaseIOSchema):
    """Test Output Schema"""

    test_field: str = Field(..., description="A test field")


class MockNestedSchema(BaseIOSchema):
    """Test Nested Schema"""

    nested_field: str = Field(..., description="A nested field")
    nested_int: int = Field(..., description="A nested integer")


class MockComplexInputSchema(BaseIOSchema):
    """Test Complex Input Schema"""

    text_field: str = Field(..., description="A text field")
    number_field: float = Field(..., description="A number field")
    list_field: List[str] = Field(..., description="A list of strings")
    nested_field: MockNestedSchema = Field(..., description="A nested schema")


class MockComplexOutputSchema(BaseIOSchema):
    """Test Complex Output Schema"""

    response_text: str = Field(..., description="A response text")
    calculated_value: int = Field(..., description="A calculated value")
    data_dict: Dict[str, MockNestedSchema] = Field(..., description="A dictionary of nested schemas")


class MockMultimodalSchema(BaseIOSchema):
    """Test schema for multimodal content"""

    instruction_text: str = Field(..., description="The instruction text")
    images: List[instructor.Image] = Field(..., description="The images to analyze")
    pdfs: List[instructor.processing.multimodal.PDF] = Field(..., description="The PDFs to analyze")
    audio: instructor.processing.multimodal.Audio = Field(..., description="The audio to analyze")


class ColorEnum(str, Enum):
    BLUE = "blue"
    RED = "red"


class MockEnumSchema(BaseIOSchema):
    """Test Input Schema with Enum."""

    color: ColorEnum = Field(..., description="Some color.")


@pytest.fixture
def history():
    return ChatHistory(max_messages=5)


def test_initialization(history):
    assert history.history == []
    assert history.max_messages == 5
    assert history.current_turn_id is None


def test_initialize_turn(history):
    history.initialize_turn()
    assert history.current_turn_id is not None


def test_add_message(history):
    history.add_message("user", InputSchema(test_field="Hello"))
    assert len(history.history) == 1
    assert history.history[0].role == "user"
    assert isinstance(history.history[0].content, InputSchema)
    assert history.history[0].turn_id is not None


def test_manage_overflow(history):
    for i in range(7):
        history.add_message("user", InputSchema(test_field=f"Message {i}"))
    assert len(history.history) == 5
    assert history.history[0].content.test_field == "Message 2"


def test_get_history(history):
    """
    Ensure non-ASCII characters are serialized without Unicode escaping, because
    it can cause issue with some OpenAI models like GPT-4.1.

    Reference ticket: https://github.com/eigenwise/atomic-agents/issues/138.
    """
    history.add_message("user", InputSchema(test_field="Hello"))
    history.add_message("assistant", MockOutputSchema(test_field="Hi there"))
    history = history.get_history()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert json.loads(history[0]["content"]) == {"test_field": "Hello"}
    assert json.loads(history[1]["content"]) == {"test_field": "Hi there"}


def test_get_history_allow_unicode(history):
    history.add_message("user", InputSchema(test_field="àéèï"))
    history.add_message("assistant", MockOutputSchema(test_field="â"))
    history = history.get_history()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == '{"test_field":"àéèï"}'
    assert history[1]["content"] == '{"test_field":"â"}'
    assert json.loads(history[0]["content"]) == {"test_field": "àéèï"}
    assert json.loads(history[1]["content"]) == {"test_field": "â"}


def test_copy(history):
    history.add_message("user", InputSchema(test_field="Hello"))
    copied_history = history.copy()
    assert copied_history.max_messages == history.max_messages
    assert copied_history.current_turn_id == history.current_turn_id
    assert len(copied_history.history) == len(history.history)
    assert copied_history.history[0].role == history.history[0].role
    assert copied_history.history[0].content.test_field == history.history[0].content.test_field


def test_copy_with_local_schema_is_independent(history):
    class LocalSchema(BaseIOSchema):
        """Message schema defined by an application factory."""

        items: List[Dict[str, List[str]]]

    history.add_message("user", LocalSchema(items=[{"values": ["original"]}]))
    copied_history = history.copy()

    assert copied_history.max_messages == history.max_messages
    assert copied_history.current_turn_id == history.current_turn_id
    assert copied_history.get_history() == history.get_history()
    assert isinstance(copied_history.history[0].content, LocalSchema)

    copied_history.history[0].content.items[0]["values"].append("changed")
    copied_history.history[0].role = "assistant"
    copied_history.add_message("user", LocalSchema(items=[]))

    assert history.history[0].content.items == [{"values": ["original"]}]
    assert history.history[0].role == "user"
    assert history.get_message_count() == 1


def test_get_current_turn_id(history):
    assert history.get_current_turn_id() is None
    history.initialize_turn()
    assert history.get_current_turn_id() is not None


def test_get_message_count(history):
    assert history.get_message_count() == 0
    history.add_message("user", InputSchema(test_field="Hello"))
    assert history.get_message_count() == 1


def test_dump_and_load_comprehensive(history):
    """Comprehensive test for dump/load functionality with complex nested data"""
    # Test complex nested schemas
    history.add_message(
        "user",
        MockComplexInputSchema(
            text_field="Complex input",
            number_field=2.718,
            list_field=["a", "b", "c"],
            nested_field=MockNestedSchema(nested_field="Nested input", nested_int=99),
        ),
    )

    history.add_message(
        "assistant",
        MockComplexOutputSchema(
            response_text="Complex output",
            calculated_value=200,
            data_dict={
                "key1": MockNestedSchema(nested_field="Nested output 1", nested_int=10),
                "key2": MockNestedSchema(nested_field="Nested output 2", nested_int=20),
            },
        ),
    )

    # Test get_history format with nested models
    history_output = history.get_history()
    assert len(history_output) == 2
    assert history_output[0]["role"] == "user"
    assert history_output[1]["role"] == "assistant"
    expected_input_content = (
        '{"text_field":"Complex input","number_field":2.718,"list_field":["a","b","c"],'
        '"nested_field":{"nested_field":"Nested input","nested_int":99}}'
    )
    expected_output_content = (
        '{"response_text":"Complex output","calculated_value":200,'
        '"data_dict":{"key1":{"nested_field":"Nested output 1","nested_int":10},'
        '"key2":{"nested_field":"Nested output 2","nested_int":20}}}'
    )
    assert history_output[0]["content"] == expected_input_content
    assert history_output[1]["content"] == expected_output_content

    # Test dump and load
    dumped_data = history.dump()
    new_history = ChatHistory()
    new_history.load(dumped_data)

    # Verify all properties are preserved
    assert new_history.max_messages == history.max_messages
    assert new_history.current_turn_id == history.current_turn_id
    assert len(new_history.history) == len(history.history)
    assert isinstance(new_history.history[0].content, MockComplexInputSchema)
    assert isinstance(new_history.history[1].content, MockComplexOutputSchema)

    # Verify detailed content
    assert new_history.history[0].content.text_field == "Complex input"
    assert new_history.history[0].content.nested_field.nested_int == 99
    assert new_history.history[1].content.response_text == "Complex output"
    assert new_history.history[1].content.data_dict["key1"].nested_field == "Nested output 1"

    # Test adding new messages to loaded history still works
    new_history.add_message("user", InputSchema(test_field="New message"))
    assert len(new_history.history) == 3
    assert new_history.history[2].content.test_field == "New message"


def test_dump_and_load_multimodal_data(history):
    import os

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    test_image = instructor.Image.from_path(path=os.path.join(base_path, "files/image_sample.jpg"))
    test_pdf = instructor.processing.multimodal.PDF.from_path(path=os.path.join(base_path, "files/pdf_sample.pdf"))
    test_audio = instructor.processing.multimodal.Audio.from_path(path=os.path.join(base_path, "files/audio_sample.mp3"))

    # multimodal message
    history.add_message(
        role="user",
        content=MockMultimodalSchema(
            instruction_text="Analyze this image", images=[test_image], pdfs=[test_pdf], audio=test_audio
        ),
    )

    dumped_data = history.dump()
    new_history = ChatHistory()
    new_history.load(dumped_data)

    assert new_history.max_messages == history.max_messages
    assert new_history.current_turn_id == history.current_turn_id
    assert len(new_history.history) == len(history.history)
    assert isinstance(new_history.history[0].content, MockMultimodalSchema)
    assert new_history.history[0].content.instruction_text == history.history[0].content.instruction_text
    assert new_history.history[0].content.images == history.history[0].content.images
    assert new_history.history[0].content.pdfs == history.history[0].content.pdfs
    assert new_history.history[0].content.audio == history.history[0].content.audio


def test_dump_and_load_with_enum(history):
    """Test that get_history works with Enum."""

    history.add_message(
        "user",
        MockEnumSchema(
            color=ColorEnum.RED,
        ),
    )

    dumped_data = history.dump()
    new_history = ChatHistory()
    new_history.load(dumped_data)

    assert new_history.max_messages == history.max_messages
    assert new_history.current_turn_id == history.current_turn_id
    assert len(new_history.history) == len(history.history)


def test_load_invalid_data(history):
    with pytest.raises(ValueError):
        history.load("invalid json")


@pytest.mark.parametrize(
    "failure",
    [
        "invalid_json",
        "missing_max_messages",
        "missing_current_turn_id",
        "missing_history",
        "invalid_history",
        "missing_content",
        "unknown_class",
        "invalid_content_json",
        "invalid_content",
        "later_invalid_content",
        "invalid_role",
        "invalid_turn_id",
    ],
)
def test_load_failure_preserves_existing_history(history, failure):
    history.add_message("user", InputSchema(test_field="Keep this message"))
    original_history = history.history
    original_message = history.history[0]
    original_dump = history.dump()
    original_turn_id = history.current_turn_id

    replacement = ChatHistory(max_messages=2)
    replacement.current_turn_id = "replacement-turn"
    replacement.add_message("user", InputSchema(test_field="Replacement user message"))
    replacement.add_message("assistant", MockOutputSchema(test_field="Replacement assistant message"))
    replacement_data = json.loads(replacement.dump())
    first_message = replacement_data["history"][0]

    if failure.startswith("missing_"):
        field = failure.removeprefix("missing_")
        del (first_message if field == "content" else replacement_data)[field]
    elif failure == "invalid_history":
        replacement_data["history"] = None
    elif failure == "unknown_class":
        first_message["content"]["class_name"] = f"{InputSchema.__module__}.NonexistentSchema"
    elif failure == "invalid_content_json":
        first_message["content"]["data"] = "invalid json"
    elif failure == "invalid_content":
        first_message["content"]["data"] = "{}"
    elif failure == "later_invalid_content":
        replacement_data["history"][1]["content"]["data"] = "{}"
    elif failure == "invalid_role":
        first_message["role"] = None
    elif failure == "invalid_turn_id":
        first_message["turn_id"] = []

    serialized_data = "invalid json" if failure == "invalid_json" else json.dumps(replacement_data)
    with pytest.raises(ValueError):
        history.load(serialized_data)

    assert history.history is original_history
    assert history.history[0] is original_message
    assert history.dump() == original_dump
    assert history.max_messages == 5
    assert history.current_turn_id == original_turn_id


def test_load_replaces_existing_history(history):
    history.add_message("user", InputSchema(test_field="Old message"))
    original_history = history.history
    replacement = ChatHistory(max_messages=2)
    replacement.current_turn_id = "replacement-turn"
    replacement.add_message("assistant", MockOutputSchema(test_field="Replacement message"))

    history.load(replacement.dump())

    assert history.history is not original_history
    assert history.dump() == replacement.dump()
    assert isinstance(history.history[0].content, MockOutputSchema)
    assert history.max_messages == 2
    assert history.current_turn_id == "replacement-turn"


@pytest.mark.parametrize("max_messages, current_turn_id", [(None, None), (0, "empty-turn")])
def test_load_empty_history_replaces_existing_history(history, max_messages, current_turn_id):
    history.add_message("user", InputSchema(test_field="Old message"))
    replacement = ChatHistory(max_messages=max_messages)
    replacement.current_turn_id = current_turn_id

    history.load(replacement.dump())

    assert history.history == []
    assert history.dump() == replacement.dump()
    assert history.max_messages == max_messages
    assert history.current_turn_id == current_turn_id


def test_get_class_from_string():
    class_string = "tests.context.test_chat_history.InputSchema"
    cls = ChatHistory._get_class_from_string(class_string)
    assert cls.__name__ == InputSchema.__name__
    assert cls.__module__.endswith("test_chat_history")
    assert issubclass(cls, BaseIOSchema)


def test_get_class_from_string_invalid():
    with pytest.raises((ImportError, AttributeError)):
        ChatHistory._get_class_from_string("invalid.module.Class")


def test_message_model():
    message = Message(role="user", content=InputSchema(test_field="Test"), turn_id="123")
    assert message.role == "user"
    assert isinstance(message.content, InputSchema)
    assert message.turn_id == "123"


def test_history_with_no_max_messages():
    unlimited_history = ChatHistory()
    for i in range(100):
        unlimited_history.add_message("user", InputSchema(test_field=f"Message {i}"))
    assert len(unlimited_history.history) == 100


def test_history_with_zero_max_messages():
    zero_max_history = ChatHistory(max_messages=0)
    for i in range(10):
        zero_max_history.add_message("user", InputSchema(test_field=f"Message {i}"))
    assert len(zero_max_history.history) == 0


def test_history_turn_consistency():
    history = ChatHistory()
    history.initialize_turn()
    turn_id = history.get_current_turn_id()
    history.add_message("user", InputSchema(test_field="Hello"))
    history.add_message("assistant", MockOutputSchema(test_field="Hi"))
    assert history.history[0].turn_id == turn_id
    assert history.history[1].turn_id == turn_id

    history.initialize_turn()
    new_turn_id = history.get_current_turn_id()
    assert new_turn_id != turn_id
    history.add_message("user", InputSchema(test_field="Next turn"))
    assert history.history[2].turn_id == new_turn_id


def test_chat_history_delete_turn_id(history):
    mock_input = InputSchema(test_field="Test input")
    mock_output = InputSchema(test_field="Test output")

    history = ChatHistory()
    initial_turn_id = "123-456"
    history.current_turn_id = initial_turn_id

    # Add a message with a specific turn ID
    history.add_message(
        "user",
        mock_input,
    )
    history.history[-1].turn_id = initial_turn_id

    # Add another message with a different turn ID
    other_turn_id = "789-012"
    history.add_message(
        "assistant",
        mock_output,
    )
    history.history[-1].turn_id = other_turn_id

    # Act & Assert: Delete the message with initial_turn_id and verify
    history.delete_turn_id(initial_turn_id)

    # The remaining message in history should have the other_turn_id
    assert len(history.history) == 1
    assert history.history[0].turn_id == other_turn_id

    # If we delete the last message, current_turn_id should become None
    history.delete_turn_id(other_turn_id)
    assert history.current_turn_id is None
    assert len(history.history) == 0

    # Assert: Trying to delete a non-existing turn ID should raise a ValueError
    with pytest.raises(ValueError, match="Turn ID non-existent-id not found in history."):
        history.delete_turn_id("non-existent-id")


def test_get_history_with_multimodal_content(history):
    """Test that get_history correctly handles multimodal content"""
    # Create mock multimodal objects
    mock_image = instructor.Image(source="test_url", media_type="image/jpeg", detail="low")
    mock_pdf = instructor.processing.multimodal.PDF(source="test_pdf_url", media_type="application/pdf", detail="low")
    mock_audio = instructor.processing.multimodal.Audio(source="test_audio_url", media_type="audio/mp3", detail="low")

    # Add a multimodal message
    history.add_message(
        "user",
        MockMultimodalSchema(instruction_text="Analyze this image", images=[mock_image], pdfs=[mock_pdf], audio=mock_audio),
    )

    # Get history and verify format
    history = history.get_history()
    assert len(history) == 1
    assert history[0]["role"] == "user"
    assert isinstance(history[0]["content"], list)
    assert json.loads(history[0]["content"][0]) == {"instruction_text": "Analyze this image"}
    assert history[0]["content"][1] == mock_image


def test_get_history_with_multiple_images_multimodal_content(history):
    """Test that get_history correctly handles multimodal content"""

    class MockMultimodalSchemaArbitraryKeys(BaseIOSchema):
        """Test schema for multimodal content"""

        instruction_text: str = Field(..., description="The instruction text")
        some_key_for_images: List[instructor.Image] = Field(..., description="The images to analyze")
        some_other_key_with_image: instructor.Image = Field(..., description="The images to analyze")

    # Create a mock image
    mock_image = instructor.Image(source="test_url", media_type="image/jpeg", detail="low")
    mock_image_2 = instructor.Image(source="test_url_2", media_type="image/jpeg", detail="low")
    mock_image_3 = instructor.Image(source="test_url_3", media_type="image/jpeg", detail="low")

    # Add a multimodal message
    history.add_message(
        "user",
        MockMultimodalSchemaArbitraryKeys(
            instruction_text="Analyze this image",
            some_other_key_with_image=mock_image,
            some_key_for_images=[mock_image_2, mock_image_3],
        ),
    )

    # Get history and verify format
    history = history.get_history()
    assert len(history) == 1
    assert history[0]["role"] == "user"
    assert isinstance(history[0]["content"], list)
    assert json.loads(history[0]["content"][0]) == {"instruction_text": "Analyze this image"}
    assert mock_image in history[0]["content"]
    assert mock_image_2 in history[0]["content"]
    assert mock_image_3 in history[0]["content"]


def test_get_history_with_mixed_content(history):
    """Test that get_history correctly handles mixed multimodal and non-multimodal items in lists"""

    # Create a schema with a list that can contain both multimodal and non-multimodal items
    class MixedContentSchema(BaseIOSchema):
        """Schema for testing mixed multimodal and non-multimodal content"""

        instruction_text: str = Field(..., description="The instruction text")
        mixed_items: List[Union[str, instructor.Image]] = Field(..., description="Mix of strings and images")

    mock_image = instructor.Image(source="test_url", media_type="image/jpeg", detail="low")

    # Add a message with mixed content
    history.add_message(
        "user",
        MixedContentSchema(instruction_text="Analyze this", mixed_items=["text_item1", mock_image, "text_item2"]),
    )

    # Get history and verify format
    result = history.get_history()
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert isinstance(result[0]["content"], list)

    # Should have JSON for non-multimodal items and the image separately
    json_content = json.loads(result[0]["content"][0])
    assert json_content["instruction_text"] == "Analyze this"
    assert json_content["mixed_items"] == ["text_item1", "text_item2"]
    assert result[0]["content"][1] == mock_image


def test_process_multimodal_paths_comprehensive():
    """Comprehensive test for _process_multimodal_paths and load functionality"""
    history = ChatHistory()

    # Test 1: Direct Image/PDF objects with file paths vs URLs
    image_file = instructor.Image(source="test/image.jpg", media_type="image/jpeg")
    image_url = instructor.Image(source="https://example.com/image.jpg", media_type="image/jpeg")
    image_data = instructor.Image(source="data:image/jpeg;base64,xyz", media_type="image/jpeg")
    pdf_file = instructor.processing.multimodal.PDF(source="test/doc.pdf", media_type="application/pdf")

    history._process_multimodal_paths(image_file)
    history._process_multimodal_paths(image_url)
    history._process_multimodal_paths(image_data)
    history._process_multimodal_paths(pdf_file)

    assert isinstance(image_file.source, Path) and image_file.source == Path("test/image.jpg")
    assert isinstance(image_url.source, str) and image_url.source == "https://example.com/image.jpg"
    assert isinstance(image_data.source, str) and image_data.source == "data:image/jpeg;base64,xyz"
    assert isinstance(pdf_file.source, Path) and pdf_file.source == Path("test/doc.pdf")

    # Test 2: Lists with mixed content
    test_list = [
        "regular_string",
        instructor.Image(source="test/list_image.jpg", media_type="image/jpeg"),
        instructor.Image(source="https://example.com/url_image.jpg", media_type="image/jpeg"),
    ]
    history._process_multimodal_paths(test_list)
    assert isinstance(test_list[1].source, Path) and test_list[1].source == Path("test/list_image.jpg")
    assert isinstance(test_list[2].source, str) and test_list[2].source == "https://example.com/url_image.jpg"

    # Test 3: Dictionaries
    test_dict = {"image": instructor.Image(source="test/dict_image.jpg", media_type="image/jpeg"), "regular": "text_content"}
    history._process_multimodal_paths(test_dict)
    assert isinstance(test_dict["image"].source, Path) and test_dict["image"].source == Path("test/dict_image.jpg")

    # Test 4: Pydantic model
    class TestModel(BaseIOSchema):
        """Test model for multimodal path processing"""

        image_field: instructor.Image = Field(..., description="Image field")
        text_field: str = Field(..., description="Text field")

    model_instance = TestModel(
        image_field=instructor.Image(source="test/model_image.jpg", media_type="image/jpeg"), text_field="test text"
    )
    history._process_multimodal_paths(model_instance)
    assert isinstance(model_instance.image_field.source, Path)
    assert model_instance.image_field.source == Path("test/model_image.jpg")

    # Test 5: Object with __dict__
    class SimpleObject:
        def __init__(self):
            self.image = instructor.Image(source="test/obj_image.jpg", media_type="image/jpeg")
            self.__pydantic_fields_set__ = {"should_be_skipped"}

    obj = SimpleObject()
    history._process_multimodal_paths(obj)
    assert isinstance(obj.image.source, Path) and obj.image.source == Path("test/obj_image.jpg")

    # Test 6: Enum (should not process __dict__)
    from enum import Enum

    class TestEnum(Enum):
        VALUE1 = "value1"

    history._process_multimodal_paths(TestEnum.VALUE1)  # Should not raise errors
    assert TestEnum.VALUE1.value == "value1"

    # Test 7: Load functionality with multimodal file paths
    original_history = ChatHistory()
    original_history.add_message(
        "user",
        MockMultimodalSchema(
            instruction_text="Process this file",
            images=[instructor.Image(source="test/sample.jpg", media_type="image/jpeg")],
            pdfs=[instructor.processing.multimodal.PDF(source="test/doc.pdf", media_type="application/pdf")],
            audio=instructor.processing.multimodal.Audio(source="test/audio.mp3", media_type="audio/mp3"),
        ),
    )

    # Dump and reload
    dumped = original_history.dump()
    loaded_history = ChatHistory()
    loaded_history.load(dumped)

    # Verify that the loaded images and PDFs have Path objects for file-like sources
    loaded_message = loaded_history.history[0]
    loaded_content = loaded_message.content
    assert isinstance(loaded_content.images[0].source, Path)
    assert loaded_content.images[0].source == Path("test/sample.jpg")
    assert isinstance(loaded_content.pdfs[0].source, Path)
    assert loaded_content.pdfs[0].source == Path("test/doc.pdf")


def test_get_history_nested_pydantic_with_toplevel_multimodal(history):
    """Issue #208: nested Pydantic model + top-level multimodal causes json.dumps TypeError"""

    class ContextInfo(BaseIOSchema):
        """Nested context info."""

        label: str = Field(..., description="Label")
        value: str = Field(..., description="Value")

    class AgentInput(BaseIOSchema):
        """Input with multimodal and nested schema."""

        instruction: str = Field(..., description="Instruction text")
        images: List[instructor.Image] = Field(..., description="Images")
        context: ContextInfo = Field(..., description="Nested context")

    mock_image = instructor.Image(source="test_url", media_type="image/jpeg", detail="low")
    context = ContextInfo(label="example", value="nested")
    content = AgentInput(instruction="Do something", images=[mock_image], context=context)

    history.add_message("user", content)
    result = history.get_history()

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert isinstance(result[0]["content"], list)

    json_part = json.loads(result[0]["content"][0])
    assert json_part["instruction"] == "Do something"
    assert json_part["context"] == {"label": "example", "value": "nested"}
    assert "images" not in json_part
    assert result[0]["content"][1] == mock_image


def test_get_history_deeply_nested_multimodal_only(history):
    """Issue #141: multimodal inside nested schema with no top-level multimodal"""

    class Document(BaseIOSchema):
        """PDF document with owner."""

        pdf: instructor.processing.multimodal.PDF = Field(..., description="The PDF data")
        owner: str = Field(..., description="The PDF owner")

    class InputSchema(BaseIOSchema):
        """A list of documents to analyze."""

        documents: List[Document] = Field(..., description="List of documents")
        instruction: str = Field(..., description="What to do")

    mock_pdf = instructor.processing.multimodal.PDF(source="test_pdf_url", media_type="application/pdf", detail="low")
    content = InputSchema(
        documents=[Document(pdf=mock_pdf, owner="Alice")],
        instruction="Analyze these",
    )

    history.add_message("user", content)
    result = history.get_history()

    assert len(result) == 1
    assert isinstance(result[0]["content"], list)

    json_part = json.loads(result[0]["content"][0])
    assert json_part["instruction"] == "Analyze these"
    assert json_part["documents"] == [{"owner": "Alice"}]
    assert result[0]["content"][1] == mock_pdf


def test_get_history_mixed_nested_and_toplevel_multimodal(history):
    """Both nested and top-level multimodal content"""

    class Attachment(BaseIOSchema):
        """An attachment with an image."""

        image: instructor.Image = Field(..., description="Attached image")
        caption: str = Field(..., description="Caption")

    class MessageInput(BaseIOSchema):
        """Message with both nested and top-level multimodal."""

        text: str = Field(..., description="Message text")
        inline_image: instructor.Image = Field(..., description="Inline image")
        attachment: Attachment = Field(..., description="An attachment")

    img1 = instructor.Image(source="inline_url", media_type="image/jpeg", detail="low")
    img2 = instructor.Image(source="attached_url", media_type="image/png", detail="low")
    content = MessageInput(
        text="Check this out",
        inline_image=img1,
        attachment=Attachment(image=img2, caption="See here"),
    )

    history.add_message("user", content)
    result = history.get_history()

    assert len(result) == 1
    assert isinstance(result[0]["content"], list)

    assert len(result[0]["content"]) == 3  # JSON + 2 images
    json_part = json.loads(result[0]["content"][0])
    assert json_part["text"] == "Check this out"
    assert json_part["attachment"] == {"caption": "See here"}
    assert "inline_image" not in json_part
    assert img1 in result[0]["content"]
    assert img2 in result[0]["content"]


def test_get_history_list_of_nested_schemas_with_multimodal(history):
    """Multiple nested schemas each containing multimodal objects"""

    class Document(BaseIOSchema):
        """A document with PDF."""

        pdf: instructor.processing.multimodal.PDF = Field(..., description="The PDF")
        title: str = Field(..., description="Document title")

    class BatchInput(BaseIOSchema):
        """Batch of documents."""

        documents: List[Document] = Field(..., description="Documents to process")

    pdf1 = instructor.processing.multimodal.PDF(source="doc1.pdf", media_type="application/pdf", detail="low")
    pdf2 = instructor.processing.multimodal.PDF(source="doc2.pdf", media_type="application/pdf", detail="low")
    content = BatchInput(
        documents=[
            Document(pdf=pdf1, title="First"),
            Document(pdf=pdf2, title="Second"),
        ]
    )

    history.add_message("user", content)
    result = history.get_history()

    assert len(result) == 1
    assert isinstance(result[0]["content"], list)

    assert len(result[0]["content"]) == 3  # JSON + 2 PDFs
    json_part = json.loads(result[0]["content"][0])
    assert json_part["documents"] == [{"title": "First"}, {"title": "Second"}]
    assert pdf1 in result[0]["content"]
    assert pdf2 in result[0]["content"]


def test_get_history_only_multimodal_fields(history):
    """Schema where ALL fields are multimodal - JSON should be omitted"""

    class ImagesOnly(BaseIOSchema):
        """Only images."""

        images: List[instructor.Image] = Field(..., description="Images")

    img1 = instructor.Image(source="url1", media_type="image/jpeg", detail="low")
    img2 = instructor.Image(source="url2", media_type="image/jpeg", detail="low")
    content = ImagesOnly(images=[img1, img2])

    history.add_message("user", content)
    result = history.get_history()

    assert len(result) == 1
    assert isinstance(result[0]["content"], list)
    # No JSON string should be present since all fields are multimodal
    assert len(result[0]["content"]) == 2
    assert all(not isinstance(item, str) for item in result[0]["content"])
    assert img1 in result[0]["content"]
    assert img2 in result[0]["content"]


def test_get_history_no_multimodal_unchanged(history):
    """Non-multimodal schemas should work exactly as before"""

    class SimpleInput(BaseIOSchema):
        """Simple input."""

        text: str = Field(..., description="Text")
        count: int = Field(..., description="Count")

    content = SimpleInput(text="hello", count=42)
    history.add_message("user", content)
    result = history.get_history()

    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert isinstance(result[0]["content"], str)
    assert json.loads(result[0]["content"]) == {"text": "hello", "count": 42}


# ---------------------------------------------------------------------------
# Direct unit tests for _extract_multimodal_info
# ---------------------------------------------------------------------------


def test_extract_multimodal_info_plain_schema():
    """No multimodal content returns empty list and None exclude spec"""

    class Plain(BaseIOSchema):
        """Plain schema."""

        text: str = Field(..., description="Text")

    objs, spec = ChatHistory._extract_multimodal_info(Plain(text="hello"))
    assert objs == []
    assert spec is None


def test_extract_multimodal_info_toplevel_image():
    """Top-level Image returns the object and True exclude spec"""
    img = instructor.Image(source="url", media_type="image/jpeg", detail="low")
    objs, spec = ChatHistory._extract_multimodal_info(img)
    assert objs == [img]
    assert spec is True


def test_extract_multimodal_info_nested_schema():
    """Nested schema with multimodal returns correct exclude spec shape"""

    class Inner(BaseIOSchema):
        """Inner schema."""

        image: instructor.Image = Field(..., description="Image")
        label: str = Field(..., description="Label")

    class Outer(BaseIOSchema):
        """Outer schema."""

        inner: Inner = Field(..., description="Inner")
        text: str = Field(..., description="Text")

    img = instructor.Image(source="url", media_type="image/jpeg", detail="low")
    objs, spec = ChatHistory._extract_multimodal_info(Outer(inner=Inner(image=img, label="test"), text="hello"))

    assert objs == [img]
    # Exclude spec should be {"inner": {"image": True}}
    assert spec == {"inner": {"image": True}}


def test_extract_multimodal_info_list_all_multimodal():
    """List where every item is multimodal collapses to True"""

    class Schema(BaseIOSchema):
        """Schema with all-multimodal list."""

        images: List[instructor.Image] = Field(..., description="Images")

    img1 = instructor.Image(source="url1", media_type="image/jpeg", detail="low")
    img2 = instructor.Image(source="url2", media_type="image/jpeg", detail="low")
    objs, spec = ChatHistory._extract_multimodal_info(Schema(images=[img1, img2]))

    assert objs == [img1, img2]
    # All items multimodal → field excluded entirely
    assert spec == {"images": True}


def test_extract_multimodal_info_list_partial_multimodal():
    """List with mixed content returns index-based exclude spec"""

    class Schema(BaseIOSchema):
        """Schema with mixed list."""

        items: List[Union[str, instructor.Image]] = Field(..., description="Items")

    img = instructor.Image(source="url", media_type="image/jpeg", detail="low")
    objs, spec = ChatHistory._extract_multimodal_info(Schema(items=["text", img]))

    assert objs == [img]
    # Only index 1 is multimodal
    assert spec == {"items": {1: True}}


def test_extract_multimodal_info_tuple_support():
    """Tuples are handled identically to lists"""
    img = instructor.Image(source="url", media_type="image/jpeg", detail="low")
    objs, spec = ChatHistory._extract_multimodal_info((img, "text"))
    assert objs == [img]
    assert spec == {0: True}


def test_extract_multimodal_info_dict_with_multimodal():
    """Dict values containing multimodal objects return correct exclude spec"""
    img = instructor.Image(source="url", media_type="image/jpeg", detail="low")
    objs, spec = ChatHistory._extract_multimodal_info({"key1": img, "key2": "text"})

    assert objs == [img]
    assert spec == {"key1": True}


def test_extract_multimodal_info_dict_all_multimodal():
    """Dict where all values are multimodal collapses to True"""
    img1 = instructor.Image(source="url1", media_type="image/jpeg", detail="low")
    img2 = instructor.Image(source="url2", media_type="image/jpeg", detail="low")
    objs, spec = ChatHistory._extract_multimodal_info({"a": img1, "b": img2})

    assert objs == [img1, img2]
    assert spec is True


def test_extract_multimodal_info_dict_no_multimodal():
    """Dict with no multimodal returns empty list and None"""
    objs, spec = ChatHistory._extract_multimodal_info({"key1": "text", "key2": 42})
    assert objs == []
    assert spec is None


def test_get_history_dict_of_images(history):
    """Dict[str, Image] field exercises the dict code path in get_history"""

    class DictImageSchema(BaseIOSchema):
        """Schema with dict of images."""

        image_map: Dict[str, instructor.Image] = Field(..., description="Named images")
        note: str = Field(..., description="A note")

    img_a = instructor.Image(source="url_a", media_type="image/jpeg", detail="low")
    img_b = instructor.Image(source="url_b", media_type="image/jpeg", detail="low")
    content = DictImageSchema(image_map={"front": img_a, "back": img_b}, note="Two views")

    history.add_message("user", content)
    result = history.get_history()

    assert len(result) == 1
    assert isinstance(result[0]["content"], list)
    assert len(result[0]["content"]) == 3  # JSON + 2 images
    json_part = json.loads(result[0]["content"][0])
    assert json_part["note"] == "Two views"
    assert "image_map" not in json_part
    assert img_a in result[0]["content"]
    assert img_b in result[0]["content"]


class VideoMessageSchema(BaseIOSchema):
    """Test schema with a video field"""

    prompt: str = Field(..., description="Instruction for the video")
    video: VideoURL = Field(..., description="The video to analyze")


def test_get_history_video_url_becomes_content_part(history):
    """VideoURL fields are separated from the JSON text as video_url content parts"""
    content = VideoMessageSchema(
        prompt="Summarize the video",
        video=VideoURL(url="https://example.com/clip.mp4", fps=1.0),
    )

    history.add_message("user", content)
    result = history.get_history()

    assert len(result) == 1
    assert json.loads(result[0]["content"][0]) == {"prompt": "Summarize the video"}
    assert result[0]["content"][1] == {
        "type": "video_url",
        "video_url": {"url": "https://example.com/clip.mp4", "fps": 1.0},
    }


def test_dump_and_load_video_url(history):
    """VideoURL round-trips through dump() and load() as a typed object"""
    history.add_message("user", VideoMessageSchema(prompt="Summarize", video=VideoURL(url="https://example.com/clip.mp4")))

    loaded_history = ChatHistory()
    loaded_history.load(history.dump())

    loaded_content = loaded_history.history[0].content
    assert isinstance(loaded_content.video, VideoURL)
    assert loaded_content.video.url == "https://example.com/clip.mp4"
