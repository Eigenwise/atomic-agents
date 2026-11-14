from unittest.mock import call

import pytest
from mcp import types

from atomic_agents.connectors.mcp import MCPDefinitionService


CATALOGS = [
    (
        "list_tools",
        types.ListToolsResult,
        "tools",
        MCPDefinitionService.fetch_tool_definitions_from_session,
    ),
    (
        "list_prompts",
        types.ListPromptsResult,
        "prompts",
        MCPDefinitionService.fetch_prompt_definitions_from_session,
    ),
    (
        "list_resources",
        types.ListResourcesResult,
        "resources",
        MCPDefinitionService.fetch_resource_definitions_from_session,
    ),
    (
        "list_resource_templates",
        types.ListResourceTemplatesResult,
        "resourceTemplates",
        MCPDefinitionService.fetch_resource_definitions_from_session,
    ),
]


@pytest.fixture
def session(paginated_session):
    client = paginated_session
    client.list_resources.return_value = types.ListResourcesResult(resources=[])
    client.list_resource_templates.return_value = types.ListResourceTemplatesResult(resourceTemplates=[])
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize("method,result_type,field,discover", CATALOGS)
@pytest.mark.parametrize("cursors", [("same", "same"), ("a", "b", "a"), ("", "")])
async def test_discovery_rejects_cyclic_pagination(session, method, result_type, field, discover, cursors):
    list_pages = getattr(session, method)
    list_pages.side_effect = [result_type(**{field: [], "nextCursor": cursor}) for cursor in cursors] + [
        AssertionError("The repeated cursor must not trigger another request")
    ]

    with pytest.raises(RuntimeError, match="MCP pagination returned a repeated cursor"):
        await discover(session)

    assert list_pages.await_args_list == [call()] + [call(cursor=cursor) for cursor in cursors[:-1]]


@pytest.mark.asyncio
@pytest.mark.parametrize("method,result_type,field,discover", CATALOGS)
async def test_empty_string_cursor_is_followed_once(session, method, result_type, field, discover):
    list_pages = getattr(session, method)
    list_pages.side_effect = [result_type(**{field: [], "nextCursor": ""}), result_type(**{field: []})]

    assert await discover(session) == []
    assert list_pages.await_args_list == [call(), call(cursor="")]


@pytest.mark.asyncio
async def test_resource_and_template_cursors_are_independent(session):
    session.list_resources.side_effect = [
        types.ListResourcesResult(resources=[], nextCursor="next"),
        types.ListResourcesResult(resources=[types.Resource(name="file", uri="file:///data")]),
    ]
    session.list_resource_templates.side_effect = [
        types.ListResourceTemplatesResult(resourceTemplates=[], nextCursor="next"),
        types.ListResourceTemplatesResult(
            resourceTemplates=[types.ResourceTemplate(name="template", uriTemplate="file:///{id}")]
        ),
    ]

    definitions = await MCPDefinitionService.fetch_resource_definitions_from_session(session)

    assert [definition.name for definition in definitions] == ["file", "template"]
    assert session.list_resources.await_args_list == [call(), call(cursor="next")]
    assert session.list_resource_templates.await_args_list == [call(), call(cursor="next")]
