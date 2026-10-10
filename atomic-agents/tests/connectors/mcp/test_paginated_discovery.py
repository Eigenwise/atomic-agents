from unittest.mock import AsyncMock, call

import pytest
from mcp import ClientSession
from mcp import types

from atomic_agents.connectors.mcp import MCPDefinitionService


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_first_page", [False, True])
async def test_discover_all_tool_pages(empty_first_page):
    first = types.Tool(name="first", inputSchema={"type": "object"})
    last = types.Tool(name="last", inputSchema={"type": "object"}, outputSchema={"type": "string"})
    session = AsyncMock(spec=ClientSession)
    session.list_tools.side_effect = [
        types.ListToolsResult(tools=[] if empty_first_page else [first], nextCursor="tools-next"),
        types.ListToolsResult(tools=[last]),
    ]

    definitions = await MCPDefinitionService.fetch_tool_definitions_from_session(session)

    assert [definition.name for definition in definitions] == ([] if empty_first_page else ["first"]) + ["last"]
    assert definitions[-1].output_schema == {"type": "string"}
    assert session.list_tools.await_args_list == [call(), call(cursor="tools-next")]
    session.initialize.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_first_page", [False, True])
async def test_discover_all_prompt_pages(empty_first_page):
    session = AsyncMock(spec=ClientSession)
    session.list_prompts.side_effect = [
        types.ListPromptsResult(prompts=[] if empty_first_page else [types.Prompt(name="first")], nextCursor="prompts-next"),
        types.ListPromptsResult(
            prompts=[types.Prompt(name="last", arguments=[types.PromptArgument(name="topic", required=True)])]
        ),
    ]

    definitions = await MCPDefinitionService.fetch_prompt_definitions_from_session(session)

    assert [definition.name for definition in definitions] == ([] if empty_first_page else ["first"]) + ["last"]
    assert definitions[-1].input_schema["required"] == ["topic"]
    assert session.list_prompts.await_args_list == [call(), call(cursor="prompts-next")]


@pytest.mark.asyncio
@pytest.mark.parametrize("empty_first_page", [False, True])
async def test_discover_all_resource_and_template_pages(empty_first_page):
    session = AsyncMock(spec=ClientSession)
    session.list_resources.side_effect = [
        types.ListResourcesResult(
            resources=[] if empty_first_page else [types.Resource(name="first", uri="file:///first")],
            nextCursor="resources-next",
        ),
        types.ListResourcesResult(resources=[types.Resource(name="last", uri="file:///last", mimeType="text/plain")]),
    ]
    session.list_resource_templates.side_effect = [
        types.ListResourceTemplatesResult(
            resourceTemplates=(
                [] if empty_first_page else [types.ResourceTemplate(name="template-first", uriTemplate="file:///{id}")]
            ),
            nextCursor="templates-next",
        ),
        types.ListResourceTemplatesResult(
            resourceTemplates=[types.ResourceTemplate(name="template-last", uriTemplate="data:///{key}")]
        ),
    ]

    definitions = await MCPDefinitionService.fetch_resource_definitions_from_session(session)

    expected = ([] if empty_first_page else ["first"]) + ["last"]
    expected += ([] if empty_first_page else ["template-first"]) + ["template-last"]
    assert [definition.name for definition in definitions] == expected
    assert next(definition for definition in definitions if definition.name == "last").mime_type == "text/plain"
    assert definitions[-1].input_schema["required"] == ["key"]
    assert session.list_resources.await_args_list == [call(), call(cursor="resources-next")]
    assert session.list_resource_templates.await_args_list == [call(), call(cursor="templates-next")]
