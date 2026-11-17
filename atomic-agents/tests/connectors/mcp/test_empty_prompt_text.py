import asyncio
from contextlib import asynccontextmanager

import anyio
import pytest
from mcp import ClientSession, types
from mcp.server.lowlevel import Server
from mcp.shared.memory import create_client_server_memory_streams

from atomic_agents.connectors.mcp import fetch_mcp_prompts_async


@pytest.fixture
def prompt_server(monkeypatch):
    server = Server("empty-prompt-text-test")
    messages = []
    calls = []

    @server.list_prompts()
    async def list_prompts():
        return [types.Prompt(name="render", description="Render the supplied messages")]

    @server.get_prompt()
    async def get_prompt(name, arguments):
        calls.append((name, arguments))
        return types.GetPromptResult(messages=messages)

    @asynccontextmanager
    async def transport(endpoint=None):
        async with create_client_server_memory_streams() as (client_streams, server_streams):
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(server.run, *server_streams, server.create_initialization_options())
                try:
                    yield (*client_streams, None)
                finally:
                    tasks.cancel_scope.cancel()

    monkeypatch.setattr("atomic_agents.connectors.mcp.mcp_factory.streamablehttp_client", transport)
    monkeypatch.setattr("atomic_agents.connectors.mcp.mcp_definition_service.streamablehttp_client", transport)
    return messages, calls, transport


@pytest.mark.asyncio
@pytest.mark.parametrize("borrowed", [False, True], ids=["owned-session", "borrowed-session"])
@pytest.mark.parametrize(
    "texts",
    [("",), ("first", "", "last"), (" \n ",), ("first", "last"), ()],
    ids=["empty", "empty-middle", "whitespace", "ordinary-text", "no-messages"],
)
async def test_sdk_prompt_text_is_preserved_through_discovery_and_generation(prompt_server, borrowed, texts):
    messages, calls, transport = prompt_server
    messages.extend(types.PromptMessage(role="user", content=types.TextContent(type="text", text=text)) for text in texts)

    async def exercise(session=None):
        prompt_type = (await fetch_mcp_prompts_async("http://prompt.test", client_session=session))[0]
        prompt = prompt_type()
        result = await prompt.agenerate(prompt.input_schema(prompt_name="render"))
        assert result.content == "\n\n".join(texts)
        assert calls == [("render", {})]
        if session is not None:
            await session.send_ping()

    if borrowed:
        async with transport() as (reader, writer, _):
            async with ClientSession(reader, writer) as session:
                await session.initialize()
                await exercise(session)
    else:
        await exercise()


def test_synchronous_generation_keeps_empty_sdk_text(prompt_server):
    messages, calls, _ = prompt_server
    messages.append(types.PromptMessage(role="assistant", content=types.TextContent(type="text", text="")))
    prompt_type = asyncio.run(fetch_mcp_prompts_async("http://prompt.test"))[0]
    prompt = prompt_type()

    result = prompt.generate(prompt.input_schema(prompt_name="render"))

    assert result.content == ""
    assert calls == [("render", {})]
