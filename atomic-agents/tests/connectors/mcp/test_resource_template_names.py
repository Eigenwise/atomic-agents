"""Read discovered resource templates with literal, non-Python variable names."""

import asyncio

from mcp import types as mt
from mcp.server import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
import pytest

from atomic_agents.connectors.mcp.mcp_factory import fetch_mcp_resources_async
from test_generated_class_session import connected


@pytest.mark.parametrize(
    "template,values,expected",
    [
        ("{document.id}", {"document.id": "report"}, "report"),
        ("{0}", {"0": "report"}, "report"),
        ("{01}", {"01": "report"}, "report"),
        ("{document.id}/{revision.no}", {"document.id": "report", "revision.no": "2"}, "report/2"),
        ("{name}/{name.part}", {"name": "report", "name.part": "summary"}, "report/summary"),
        ("{item.id}/{0}/{item.id}", {"item.id": "report", "0": "part"}, "report/part/report"),
        ("{item}/{item}", {"item": "report"}, "report/report"),
        ("{item_id}", {"item_id": "report"}, "report"),
        ("{item%2Eid}", {"item%2Eid": "report"}, "report"),
        ("literal%2Fpath/{item}", {"item": "report"}, "literal%2Fpath/report"),
    ],
)
def test_literal_template_names_reach_the_native_server(template, values, expected):
    async def scenario():
        server = Server("literal-resource-variables")
        frames = []
        expected_uri = "memory://documents/" + expected

        @server.list_resources()
        async def resources():
            return []

        @server.list_resource_templates()
        async def templates():
            return [mt.ResourceTemplate(name="document", uriTemplate="memory://documents/" + template)]

        @server.read_resource()
        async def read_resource(uri):
            if str(uri) != expected_uri:
                raise ValueError(f"No resource at {uri}")
            return [ReadResourceContents(content="requested document", mime_type="text/plain")]

        async with connected(server, frames) as session:
            resource_type = (await fetch_mcp_resources_async(client_session=session))[0]
            resource = resource_type()
            assert set(resource.input_schema.model_json_schema()["properties"]) == {*values, "resource_name"}
            result = await resource.aread(resource.input_schema(resource_name="document", **values))
            assert result.content[0].text == "requested document"
            assert str(result.content[0].uri) == expected_uri
            await session.send_ping()

        requests = [frame["params"]["uri"] for frame in frames if frame.get("method") == "resources/read"]
        assert requests == [expected_uri]

    asyncio.run(scenario())
