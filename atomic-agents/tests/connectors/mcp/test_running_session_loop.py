"""A borrowed MCP session must execute on its owning, continuously running loop."""

import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
import gc
from threading import Thread

import pytest
from mcp import types
from mcp.server import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.shared.memory import create_connected_server_and_client_session

from atomic_agents.connectors.mcp import fetch_mcp_prompts, fetch_mcp_resources, fetch_mcp_tools


@contextmanager
def running_session():
    ready = Future()
    stopped = Future()
    server = Server("running-session-regression")

    @server.list_tools()
    async def list_tools():
        return [
            types.Tool(
                name="echo",
                inputSchema={
                    "type": "object",
                    "properties": {"value": {"type": "string"}, "pair": {"type": "boolean", "default": False}},
                    "required": ["value"],
                },
            )
        ]

    @server.list_resources()
    async def list_resources():
        return [types.Resource(name="document", uri="memory://document")]

    @server.list_resource_templates()
    async def list_templates():
        return []

    @server.read_resource()
    async def read_resource(uri):
        return [ReadResourceContents("resource content", "text/plain")]

    @server.list_prompts()
    async def list_prompts():
        return [types.Prompt(name="greeting")]

    @server.get_prompt()
    async def get_prompt(name, arguments):
        return types.GetPromptResult(
            messages=[types.PromptMessage(role="user", content=types.TextContent(type="text", text="hello"))]
        )

    async def owner():
        both_started = asyncio.Event()
        pair_calls = 0

        @server.call_tool()
        async def call_tool(name, arguments):
            nonlocal pair_calls
            if arguments.get("pair"):
                pair_calls += 1
                if pair_calls == 2:
                    both_started.set()
                await asyncio.wait_for(both_started.wait(), timeout=3)
            return [types.TextContent(type="text", text=arguments["value"])]

        async with create_connected_server_and_client_session(server) as session:
            stop = asyncio.Event()
            ready.set_result((session, asyncio.get_running_loop(), stop))
            await stop.wait()

    def run():
        try:
            asyncio.run(owner())
        except BaseException as error:
            if not ready.done():
                ready.set_exception(error)
            stopped.set_exception(error)
        else:
            stopped.set_result(None)

    thread = Thread(target=run, name="mcp-test-owner")
    thread.start()
    session, loop, stop = ready.result(timeout=5)
    try:
        yield session, loop
    finally:
        loop.call_soon_threadsafe(stop.set)
        thread.join(timeout=5)
        assert not thread.is_alive()
        stopped.result(timeout=1)
        assert loop.is_closed()


@pytest.mark.parametrize("kind", ["tool", "resource", "prompt"])
def test_sync_discovery_and_invocation_use_the_borrowed_running_loop(kind):
    with running_session() as (session, loop):
        fetch = {"tool": fetch_mcp_tools, "resource": fetch_mcp_resources, "prompt": fetch_mcp_prompts}[kind]
        generated = fetch(client_session=session, event_loop=loop)[0]
        instance = generated()
        if kind == "tool":
            result = instance.run(generated.input_schema(tool_name="echo", value="tool content"))
            assert result.result[0].text == "tool content"
        elif kind == "resource":
            result = instance.read(generated.input_schema(resource_name="document"))
            assert result.content[0].text == "resource content"
        else:
            result = instance.generate(generated.input_schema(prompt_name="greeting"))
            assert result.content == "hello"


def test_two_sync_workers_can_run_tools_concurrently_on_one_session():
    with running_session() as (session, loop):
        generated = fetch_mcp_tools(client_session=session, event_loop=loop)[0]
        with ThreadPoolExecutor(max_workers=2) as workers:
            pending = [
                workers.submit(generated().run, generated.input_schema(tool_name="echo", value=value, pair=True))
                for value in ("first", "second")
            ]
            assert [task.result(timeout=5).result[0].text for task in pending] == ["first", "second"]


def test_sync_api_rejects_its_own_running_loop_without_leaking_a_coroutine(recwarn):
    with running_session() as (session, loop):

        async def call_from_owner():
            with pytest.raises(RuntimeError, match="asynchronous"):
                fetch_mcp_tools(client_session=session, event_loop=loop)

        asyncio.run_coroutine_threadsafe(call_from_owner(), loop).result(timeout=5)
    gc.collect()
    assert not [warning for warning in recwarn if issubclass(warning.category, RuntimeWarning)]
