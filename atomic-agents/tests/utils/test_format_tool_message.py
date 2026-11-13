import uuid
import json
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_serializer
from instructor.processing.schema import generate_openai_schema
import pytest
from atomic_agents import BaseIOSchema
from atomic_agents.utils import format_tool_message


# Mock classes for testing
class MockToolCall(BaseModel):
    """Mock class for testing"""

    param1: str
    param2: int


def test_format_tool_message_with_provided_tool_id():
    tool_call = MockToolCall(param1="test", param2=42)
    tool_id = "test-tool-id"

    result = format_tool_message(tool_call, tool_id)

    assert result == {
        "id": "test-tool-id",
        "type": "function",
        "function": {"name": "MockToolCall", "arguments": '{"param1": "test", "param2": 42}'},
    }


def test_format_tool_message_without_tool_id():
    tool_call = MockToolCall(param1="test", param2=42)

    result = format_tool_message(tool_call)

    assert isinstance(result["id"], str)
    assert len(result["id"]) == 36  # UUID length
    assert result["type"] == "function"
    assert result["function"]["name"] == "MockToolCall"
    assert result["function"]["arguments"] == '{"param1": "test", "param2": 42}'


def test_format_tool_message_with_different_tool():
    class AnotherToolCall(BaseModel):
        """Another tool schema"""

        field1: bool
        field2: float

    tool_call = AnotherToolCall(field1=True, field2=3.14)

    result = format_tool_message(tool_call)

    assert result["type"] == "function"
    assert result["function"]["name"] == "AnotherToolCall"
    assert result["function"]["arguments"] == '{"field1": true, "field2": 3.14}'


def test_format_tool_message_id_is_valid_uuid():
    tool_call = MockToolCall(param1="test", param2=42)

    result = format_tool_message(tool_call)

    try:
        uuid.UUID(result["id"])
    except ValueError:
        pytest.fail("The generated tool_id is not a valid UUID")


def test_format_tool_message_consistent_output():
    tool_call = MockToolCall(param1="test", param2=42)
    tool_id = "fixed-id"

    result1 = format_tool_message(tool_call, tool_id)
    result2 = format_tool_message(tool_call, tool_id)

    assert result1 == result2


def test_format_tool_message_with_complex_model():
    class ComplexToolCall(BaseIOSchema):
        """Mock complex tool call schema"""

        nested: dict
        list_field: list

    tool_call = ComplexToolCall(nested={"key": "value"}, list_field=[1, 2, 3])

    result = format_tool_message(tool_call)

    assert result["function"]["name"] == "ComplexToolCall"
    assert result["function"]["arguments"] == '{"nested": {"key": "value"}, "list_field": [1, 2, 3]}'


def test_format_tool_message_serializes_pydantic_json_types():
    class Priority(Enum):
        HIGH = "high"

    class Details(BaseModel):
        created: datetime
        due: date
        request_id: uuid.UUID
        amount: Decimal
        path: Path
        priority: Priority

    class RichToolCall(BaseModel):
        details: Details

    tool_call = RichToolCall(
        details=Details(
            created=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            due=date(2026, 1, 3),
            request_id=uuid.UUID("12345678-1234-5678-1234-567812345678"),
            amount=Decimal("12.50"),
            path=Path("report.txt"),
            priority=Priority.HIGH,
        )
    )

    result = format_tool_message(tool_call, "request-1")

    assert result["function"]["name"] == "RichToolCall"
    assert json.loads(result["function"]["arguments"]) == {
        "details": {
            "created": "2026-01-02T03:04:05Z",
            "due": "2026-01-03",
            "request_id": "12345678-1234-5678-1234-567812345678",
            "amount": "12.50",
            "path": "report.txt",
            "priority": "high",
        }
    }


def test_format_tool_message_honors_json_field_serializers():
    class SerializedToolCall(BaseModel):
        value: int

        @field_serializer("value", when_used="json")
        def serialize_value(self, value):
            return f"item-{value}"

    result = format_tool_message(SerializedToolCall(value=42))

    assert json.loads(result["function"]["arguments"]) == {"value": "item-42"}


@pytest.mark.parametrize(
    "config",
    [
        ConfigDict(title="search_documents"),
        ConfigDict(model_title_generator=lambda cls: "generated_search"),
        ConfigDict(json_schema_extra={"title": "custom_search"}),
    ],
)
def test_format_tool_message_name_matches_advertised_schema(config):
    class SearchArguments(BaseIOSchema):
        """Search the available documents."""

        model_config = config
        query: str

    call = SearchArguments(query="example")
    result = format_tool_message(call, "search-1")

    assert result["function"]["name"] == generate_openai_schema(SearchArguments)["name"]
    assert json.loads(result["function"]["arguments"]) == {"query": "example"}
    assert result["id"] == "search-1"


if __name__ == "__main__":
    pytest.main()
