import asyncio
import os
import shlex
import sys
import textwrap

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from atomic_agents.connectors.mcp import (
    MCPDefinitionService,
    MCPFactory,
    MCPTransportType,
    fetch_mcp_attributes_with_schema,
    fetch_mcp_prompts,
    fetch_mcp_prompts_async,
    fetch_mcp_resources,
    fetch_mcp_resources_async,
    fetch_mcp_tools,
    fetch_mcp_tools_async,
)


SETTING = "ATOMIC_TEST_CHILD_SETTING"


@pytest.fixture
def stdio_server(tmp_path):
    script = tmp_path / "environment_server.py"
    script.write_text(
        textwrap.dedent(
            """
            import os
            from mcp.server.fastmcp import FastMCP

            value = os.environ.get("ATOMIC_TEST_CHILD_SETTING", "unset")
            server = FastMCP("environment-fixture")

            @server.tool(description=value)
            def setting() -> str:
                return value

            @server.resource("config://setting", name="setting", description=value)
            def setting_resource() -> str:
                return value

            @server.prompt(name="setting", description=value)
            def setting_prompt() -> str:
                return value

            server.run()
            """
        ),
        encoding="utf-8",
    )
    return shlex.join([sys.executable, str(script)])


def invoke(component, kind):
    instance = component()
    params = instance.input_schema(**{kind + "_name": "setting"})
    return getattr(instance, {"tool": "run", "resource": "read", "prompt": "generate"}[kind])(params)


async def invoke_async(component, kind):
    instance = component()
    params = instance.input_schema(**{kind + "_name": "setting"})
    return await getattr(instance, {"tool": "arun", "resource": "aread", "prompt": "agenerate"}[kind])(params)


def assert_value(result, kind, value):
    if kind == "tool":
        # Newer FastMCP versions advertise a typed string result; older versions
        # return the same value as a text content block.
        actual = result.result
        if not isinstance(actual, str):
            block = actual[0]
            actual = block["text"] if isinstance(block, dict) else block.text
        assert actual == value
    elif kind == "resource":
        assert result.content[0].text == value
    else:
        assert result.content == value


@pytest.mark.parametrize("kind", ["tool", "resource", "prompt"])
def test_definition_service_passes_an_owned_environment_snapshot(stdio_server, kind):
    environment = {SETTING: "discovery-value"}
    service = MCPDefinitionService(stdio_server, MCPTransportType.STDIO, stdio_env=environment)
    environment[SETTING] = "changed-by-caller"

    definitions = asyncio.run(getattr(service, f"fetch_{kind}_definitions")())

    assert len(definitions) == 1
    assert definitions[0].description == "discovery-value"


@pytest.mark.parametrize(
    "kind,fetch", [("tool", fetch_mcp_tools), ("resource", fetch_mcp_resources), ("prompt", fetch_mcp_prompts)]
)
def test_sync_helpers_preserve_environment_for_discovery_and_later_calls(stdio_server, kind, fetch, monkeypatch):
    monkeypatch.setenv(SETTING, "parent-value")
    environment = {SETTING: "configured-value"}

    components = fetch(stdio_server, MCPTransportType.STDIO, stdio_env=environment)
    environment[SETTING] = "changed-by-caller"

    assert components[0].__doc__ == "configured-value"
    assert_value(invoke(components[0], kind), kind, "configured-value")
    assert os.environ[SETTING] == "parent-value"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,fetch",
    [("tool", fetch_mcp_tools_async), ("resource", fetch_mcp_resources_async), ("prompt", fetch_mcp_prompts_async)],
)
async def test_async_helpers_preserve_environment_for_discovery_and_later_calls(stdio_server, kind, fetch):
    components = await fetch(stdio_server, MCPTransportType.STDIO, stdio_env={SETTING: "async-value"})

    assert components[0].__doc__ == "async-value"
    assert_value(await invoke_async(components[0], kind), kind, "async-value")


def test_factory_copies_environment_before_discovery(stdio_server):
    environment = {SETTING: "factory-value"}
    factory = MCPFactory(stdio_server, MCPTransportType.STDIO, stdio_env=environment)
    environment[SETTING] = "changed-by-caller"

    for kind in ("tool", "resource", "prompt"):
        component = getattr(factory, f"create_{kind}s")()[0]
        assert component.__doc__ == "factory-value"
        assert_value(invoke(component, kind), kind, "factory-value")


def test_combined_schema_helper_forwards_environment_to_all_catalogs(stdio_server):
    tools, resources, prompts, schema = fetch_mcp_attributes_with_schema(
        stdio_server, MCPTransportType.STDIO, stdio_env={SETTING: "combined-value"}
    )

    assert schema is not None
    for kind, components in zip(("tool", "resource", "prompt"), (tools, resources, prompts)):
        assert components[0].__doc__ == "combined-value"
        assert_value(invoke(components[0], kind), kind, "combined-value")


@pytest.mark.parametrize("environment", [None, {}])
def test_default_environment_does_not_copy_arbitrary_parent_variables(stdio_server, monkeypatch, environment):
    monkeypatch.setenv(SETTING, "parent-only")
    kwargs = {} if environment is None else {"stdio_env": environment}

    components = fetch_mcp_tools(stdio_server, MCPTransportType.STDIO, **kwargs)

    assert components[0].__doc__ == "unset"
    assert_value(invoke(components[0], "tool"), "tool", "unset")


@pytest.mark.asyncio
async def test_borrowed_session_keeps_its_existing_process_environment(stdio_server):
    command, *args = shlex.split(stdio_server)
    parameters = StdioServerParameters(command=command, args=args, env={SETTING: "borrowed-value"})
    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            components = await fetch_mcp_tools_async(client_session=session, stdio_env={SETTING: "unused"})

            assert components[0].__doc__ == "borrowed-value"
            assert_value(await invoke_async(components[0], "tool"), "tool", "borrowed-value")
