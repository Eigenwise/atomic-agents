from unittest.mock import AsyncMock

import pytest
from mcp import ClientSession


@pytest.fixture
def paginated_session():
    """Feed the same catalog pages through either supported SDK request path."""
    session = AsyncMock(spec=ClientSession)
    methods = {
        "tools/list": "list_tools",
        "resources/list": "list_resources",
        "resources/templates/list": "list_resource_templates",
        "prompts/list": "list_prompts",
    }

    async def send_request(request, result_type):
        payload = request.model_dump(by_alias=True, mode="json", exclude_none=True)
        return await getattr(session, methods[payload["method"]])(cursor=payload["params"]["cursor"])

    session.send_request.side_effect = send_request
    return session
