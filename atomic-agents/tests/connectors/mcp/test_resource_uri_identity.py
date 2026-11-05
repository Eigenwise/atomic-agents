"""Preserve MCP resource identifiers across catalog discovery and reads."""

from unittest.mock import AsyncMock

import pytest
from mcp import types

from atomic_agents.connectors.mcp.mcp_definition_service import MCPDefinitionService
from atomic_agents.connectors.mcp.mcp_factory import fetch_mcp_resources_async


ENCODED_URIS = [
    "resource://bucket/key%2Fvalue",
    "resource://bucket/a%23b",
    "resource://bucket/a%3Fb",
    "resource://bucket/%7Bid%7D",
    "resource://bucket/%252F",
    "resource://bucket/caf%C3%A9",
    "resource://bucket/plain",
]


def resource_session(uri: str) -> AsyncMock:
    session = AsyncMock()
    session.list_resources.return_value = types.ListResourcesResult(resources=[types.Resource(name="document", uri=uri)])
    session.list_resource_templates.return_value = types.ListResourceTemplatesResult(resourceTemplates=[])
    session.read_resource.return_value = types.ReadResourceResult(contents=[])
    return session


@pytest.mark.asyncio
@pytest.mark.parametrize("uri", ENCODED_URIS)
async def test_discovery_preserves_literal_resource_uri(uri: str) -> None:
    session = resource_session(uri)
    definition = (await MCPDefinitionService.fetch_resource_definitions_from_session(session))[0]

    assert definition.uri == uri
    assert definition.input_schema == {"type": "object", "properties": {}, "required": []}


@pytest.mark.asyncio
@pytest.mark.parametrize("uri", ENCODED_URIS)
async def test_generated_resource_reads_the_advertised_identifier(uri: str) -> None:
    session = resource_session(uri)
    resource_class = (await fetch_mcp_resources_async(client_session=session))[0]
    resource = resource_class()

    await resource.aread(resource.input_schema(resource_name="document"))

    session.read_resource.assert_awaited_once_with(uri=uri)
    request = types.ReadResourceRequest(
        method="resources/read", params=types.ReadResourceRequestParams(uri=session.read_resource.call_args.kwargs["uri"])
    )
    assert request.model_dump(mode="json")["params"]["uri"] == uri


@pytest.mark.asyncio
@pytest.mark.parametrize("literal", ["key%2Fvalue", "a%23b", "%7Bliteral%7D", "%252F", "plain"])
async def test_template_expands_only_the_declared_parameter(literal: str) -> None:
    uri_template = f"resource://bucket/{literal}/{{item}}"
    session = resource_session("resource://bucket/static")
    session.list_resources.return_value = types.ListResourcesResult(resources=[])
    session.list_resource_templates.return_value = types.ListResourceTemplatesResult(
        resourceTemplates=[types.ResourceTemplate(name="document", uriTemplate=uri_template)]
    )
    definition = (await MCPDefinitionService.fetch_resource_definitions_from_session(session))[0]

    assert definition.uri == uri_template
    assert definition.input_schema["required"] == ["item"]
    assert set(definition.input_schema["properties"]) == {"item"}

    resource_class = (await fetch_mcp_resources_async(client_session=session))[0]
    resource = resource_class()
    await resource.aread(resource.input_schema(resource_name="document", item="entry"))

    session.read_resource.assert_awaited_once_with(uri=f"resource://bucket/{literal}/entry")
