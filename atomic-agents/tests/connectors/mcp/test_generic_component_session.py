"""Exercise generated MCP schemas through reusable generic components.

Real client/server sessions exchange JSON-round-tripped messages over in-memory
streams. This does not exercise a network or stdio transport.
"""

import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import timedelta
import json
import types

import anyio
from mcp import ClientSession
from mcp import types as mt
from mcp.server import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.shared.message import SessionMessage
import pytest

from atomic_agents.base import BaseIOSchema, BaseTool, BaseResource, BasePrompt
from atomic_agents.connectors.mcp.mcp_factory import fetch_mcp_tools_async, fetch_mcp_resources_async, fetch_mcp_prompts_async


@asynccontextmanager
async def connected(server, frames):
    """Own both sessions and the JSON relays for the duration of one test."""
    async with AsyncExitStack() as stack:
        channels = [anyio.create_memory_object_stream(1) for _ in range(4)]
        for pair in channels:
            for stream in pair:
                await stack.enter_async_context(stream)
        client_send, client_out = channels[0]
        server_send, server_out = channels[1]
        to_server, server_read = channels[2]
        to_client, client_read = channels[3]

        async def relay(source, target):
            async for message in source:
                encoded = message.message.model_dump_json(by_alias=True, exclude_none=True)
                frames.append(json.loads(encoded))
                await target.send(SessionMessage(mt.JSONRPCMessage.model_validate_json(encoded)))

        async with anyio.create_task_group() as group:
            group.start_soon(relay, client_out, to_server)
            group.start_soon(relay, server_out, to_client)
            group.start_soon(server.run, server_read, server_send, server.create_initialization_options())
            try:
                async with ClientSession(client_read, client_send, read_timeout_seconds=timedelta(seconds=3)) as session:
                    await session.initialize()
                    yield session
            finally:
                group.cancel_scope.cancel()


class Operations:
    """A reusable generic adapter forwards operations to a generated component."""

    def run(self, params):
        return self.inner.run(params)

    def read(self, params):
        return self.inner.read(params)

    def generate(self, params):
        return self.inner.generate(params)

    async def invoke(self, params, method):
        return await getattr(self.inner, method)(params)


@pytest.mark.parametrize("kind", ["tool", "resource", "prompt"])
@pytest.mark.parametrize("generic", [False, True], ids=["generated-control", "generic-adapter"])
def test_generated_components_roundtrip_through_generic_adapter(kind, generic):
    async def scenario():
        server = Server("generic-component-test")
        calls = []
        frames = []

        @server.list_tools()
        async def tools():
            return [
                mt.Tool(
                    name="echo",
                    description="Echo tool",
                    inputSchema={"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
                )
            ]

        @server.call_tool()
        async def call_tool(name, arguments):
            calls.append((name, arguments))
            return [mt.TextContent(type="text", text=arguments["text"])]

        @server.list_resources()
        async def resources():
            return [mt.Resource(name="echo", description="Echo resource", uri="memory://echo/value", mimeType="text/plain")]

        @server.list_resource_templates()
        async def templates():
            return []

        @server.read_resource()
        async def read_resource(uri):
            calls.append(str(uri))
            return [ReadResourceContents(content="hello 世界", mime_type="text/plain")]

        @server.list_prompts()
        async def prompts():
            return [
                mt.Prompt(name="echo", description="Echo prompt", arguments=[mt.PromptArgument(name="text", required=True)])
            ]

        @server.get_prompt()
        async def get_prompt(name, arguments):
            calls.append((name, arguments))
            return mt.GetPromptResult(
                messages=[mt.PromptMessage(role="user", content=mt.TextContent(type="text", text=arguments["text"]))]
            )

        fetch, base, method = {
            "tool": (fetch_mcp_tools_async, BaseTool, "arun"),
            "resource": (fetch_mcp_resources_async, BaseResource, "aread"),
            "prompt": (fetch_mcp_prompts_async, BasePrompt, "agenerate"),
        }[kind]
        async with connected(server, frames) as session:
            (generated,) = await fetch(client_session=session)
            instance = generated()
            if generic:

                class Reusable[I: BaseIOSchema, O: BaseIOSchema](Operations, base[I, O]):
                    pass

                concrete = types.new_class("Concrete", (Reusable[generated.input_schema, generated.output_schema],))
                instance = concrete()
                instance.inner = generated()
            assert instance.input_schema is generated.input_schema
            assert instance.output_schema is generated.output_schema
            params = instance.input_schema(
                **{f"{kind}_name": "echo", **({} if kind == "resource" else {"text": "hello 世界"})}
            )
            output = await instance.invoke(params, method) if generic else await getattr(instance, method)(params)
            assert type(output) is instance.output_schema
            if kind == "tool":
                assert output.result[0].text == "hello 世界"
            elif kind == "resource":
                assert output.content[0].text == "hello 世界"
                assert output.mime_type == "text/plain"
            else:
                assert "hello 世界" in output.content
            assert calls == (["memory://echo/value"] if kind == "resource" else [("echo", {"text": "hello 世界"})])
            assert any(
                frame.get("method") == {"tool": "tools/call", "resource": "resources/read", "prompt": "prompts/get"}[kind]
                for frame in frames
            )
            await session.send_ping()

    asyncio.run(scenario())
