import json
import uuid
from pydantic import BaseModel
from typing import Dict, Optional, Type


def format_tool_message(tool_call: Type[BaseModel], tool_id: Optional[str] = None) -> Dict:
    """
    Formats a message for a tool call, using Pydantic's JSON representations for arguments.

    Args:
        tool_call (Type[BaseModel]): The Pydantic model instance representing the tool call.
        tool_id (str, optional): The unique identifier for the tool call. If not provided, a random UUID will be generated.

    Returns:
        Dict: A formatted message dictionary for the tool call.
    """
    if tool_id is None:
        tool_id = str(uuid.uuid4())

    # Use the same schema title that Instructor advertises as the function name.
    tool_name = tool_call.__class__.model_json_schema().get("title", tool_call.__class__.__name__)
    return {
        "id": tool_id,
        "type": "function",
        "function": {
            "name": tool_name,
            "arguments": json.dumps(tool_call.model_dump(mode="json"), separators=(", ", ": ")),
        },
    }
