"""Verify multiple generated MCP classes across real JSON-round-tripped sessions.

In-memory streams exercise discovery and the client/server protocol, not network
or stdio transports. Each session is borrowed by every generated component.
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import timedelta
import json

import anyio
from mcp import ClientSession
from mcp import types as mt
from mcp.server import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.shared.message import SessionMessage
import pytest

from atomic_agents.connectors.mcp.mcp_factory import fetch_mcp_resources_async, fetch_mcp_prompts_async


@asynccontextmanager
async def connected(server: Server, frames: list[dict]) -> AsyncIterator[ClientSession]:
    """Relay serialized messages while owning both sessions and their streams."""
    async with AsyncExitStack() as stack:
        channels = [anyio.create_memory_object_stream(1) for _ in range(4)]
        for pair in channels:
            for stream in pair:
                await stack.enter_async_context(stream)
        client_send, client_out = channels[0]
        server_send, server_out = channels[1]
        to_server, server_read = channels[2]
        to_client, client_read = channels[3]

        async def relay(source, target) -> None:
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


@pytest.mark.parametrize("empty", [True, False], ids=["empty-fallback", "explicit-content-mime"])
def test_discovered_resources_keep_output_binding_and_mime(empty: bool) -> None:
    async def scenario() -> None:
        server = Server("resource-binding-test")
        frames = []
        names = ["first", "last"]
        uris = {name: f"memory://binding/{name}" for name in names}
        fallback = dict(zip(names, ["text/plain", "application/json"]))
        calls = []

        @server.list_resources()
        async def resources() -> list[mt.Resource]:
            return [mt.Resource(name=name, uri=uris[name], mimeType=fallback[name]) for name in names]

        @server.list_resource_templates()
        async def templates() -> list[mt.ResourceTemplate]:
            return []

        @server.read_resource()
        async def read_resource(uri) -> list[ReadResourceContents]:
            calls.append(str(uri))
            name = str(uri).rsplit("/", 1)[1]
            return [] if empty else [ReadResourceContents(content=f"{name}: 世界", mime_type=f"application/x-{name}")]

        async with connected(server, frames) as session:
            classes = await fetch_mcp_resources_async(client_session=session)
            assert [cls.mcp_resource_name for cls in classes] == names
            instances = {cls.mcp_resource_name: cls() for cls in classes}
            assert instances["first"].output_schema is not instances["last"].output_schema
            order = ["last", "first", "last", "first"]
            for name in order:
                resource = instances[name]
                result = await resource.aread(resource.input_schema(resource_name=name))
                assert result.mime_type == (fallback[name] if empty else f"application/x-{name}")
                assert type(result) is resource.output_schema
                if empty:
                    assert result.content == []
                else:
                    assert len(result.content) == 1
                    assert str(result.content[0].uri) == uris[name]
                    assert result.content[0].text == f"{name}: 世界"
                await session.send_ping()
            assert calls == [uris[name] for name in order]
            requests = [frame["params"]["uri"] for frame in frames if frame.get("method") == "resources/read"]
            assert requests == calls
            assert any(frame.get("method") == "resources/list" for frame in frames)
            assert sum(frame.get("method") == "ping" for frame in frames) == len(order)

    asyncio.run(scenario())


def test_discovered_prompts_keep_output_binding_and_arguments() -> None:
    async def scenario() -> None:
        server = Server("prompt-binding-test")
        frames = []
        names = ["first", "last"]
        calls = []

        @server.list_prompts()
        async def prompts() -> list[mt.Prompt]:
            return [mt.Prompt(name=name, arguments=[mt.PromptArgument(name="text", required=True)]) for name in names]

        @server.get_prompt()
        async def get_prompt(name: str, arguments: dict[str, str] | None) -> mt.GetPromptResult:
            assert arguments is not None
            calls.append((name, arguments))
            return mt.GetPromptResult(
                messages=[
                    mt.PromptMessage(role="user", content=mt.TextContent(type="text", text=f"{name}: {arguments['text']}")),
                    mt.PromptMessage(role="assistant", content=mt.TextContent(type="text", text=f"second: {name}")),
                ]
            )

        async with connected(server, frames) as session:
            classes = await fetch_mcp_prompts_async(client_session=session)
            assert [cls.mcp_prompt_name for cls in classes] == names
            instances = {cls.mcp_prompt_name: cls() for cls in classes}
            assert instances["first"].output_schema is not instances["last"].output_schema
            order = ["last", "first", "last", "first"]
            for index, name in enumerate(order):
                prompt = instances[name]
                text = f"世界 {index}"
                result = await prompt.agenerate(prompt.input_schema(prompt_name=name, text=text))
                assert type(result) is prompt.output_schema
                assert result.content == f"{name}: {text}\n\nsecond: {name}"
                await session.send_ping()
            expected = [(name, {"text": f"世界 {index}"}) for index, name in enumerate(order)]
            assert calls == expected
            requests = [frame["params"] for frame in frames if frame.get("method") == "prompts/get"]
            assert requests == [{"name": name, "arguments": arguments} for name, arguments in expected]
            assert any(frame.get("method") == "prompts/list" for frame in frames)
            assert sum(frame.get("method") == "ping" for frame in frames) == len(order)

    asyncio.run(scenario())
