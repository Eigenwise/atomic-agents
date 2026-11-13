"""Exercise the demo's persistent sessions with real local MCP child processes."""

from concurrent.futures import ThreadPoolExecutor
import io
import os
from pathlib import Path
import shlex
import sys

import pytest
from rich.console import Console


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from progressive_disclosure.main import MCPServerManager, ServerConfig  # noqa: E402


@pytest.fixture
def local_server(tmp_path):
    script = tmp_path / "server.py"
    script.write_text(
        """import asyncio
import os
from mcp.server.fastmcp import FastMCP

server = FastMCP("session-owner-test")

@server.tool()
async def delayed_add(a: int, b: int) -> int:
    '''Add two numbers after allowing another request to start.'''
    await asyncio.sleep(0.1)
    return a + b

@server.tool()
def server_pid() -> int:
    '''Return the child process identifier for the cleanup assertion.'''
    return os.getpid()

server.run(transport="stdio")
""",
        encoding="utf-8",
    )
    return ServerConfig("local", shlex.join([sys.executable, str(script)]), "math")


def test_normal_close_exits_the_owner_task_and_reaps_the_server(local_server):
    manager = MCPServerManager([local_server])
    output = io.StringIO()
    console = Console(file=output, force_terminal=False)
    try:
        manager.connect_all(console)
        loops = list(manager.loops.values())
        tool = next(tool for tool in manager.all_tools if tool.mcp_tool_name == "server_pid")
        pid = tool().run(tool.input_schema(tool_name="server_pid")).result
    finally:
        manager.close_all(console)

    assert all(loop.is_closed() for loop in loops)
    assert "Error closing" not in output.getvalue()
    if os.name == "posix":
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    # Repeated cleanup must not try to close an already closed event loop.
    manager.close_all(console)
    assert "Error closing" not in output.getvalue()


def test_same_server_tools_can_be_called_from_parallel_workers(local_server):
    manager = MCPServerManager([local_server])
    output = io.StringIO()
    console = Console(file=output, force_terminal=False)
    try:
        manager.connect_all(console)
        tool = next(tool for tool in manager.all_tools if tool.mcp_tool_name == "delayed_add")
        with ThreadPoolExecutor(max_workers=2) as workers:
            pending = [
                workers.submit(tool().run, tool.input_schema(tool_name="delayed_add", a=value, b=1)) for value in (1, 2)
            ]
            assert [task.result(timeout=5).result for task in pending] == [2, 3]
    finally:
        manager.close_all(console)

    assert "Error closing" not in output.getvalue()


def test_failed_later_connection_also_closes_earlier_and_partial_sessions(local_server):
    missing = ServerConfig("closed", shlex.join([sys.executable, "-c", "pass"]), "test")
    manager = MCPServerManager([local_server, missing])
    console = Console(file=io.StringIO(), force_terminal=False)
    try:
        with pytest.raises(Exception):
            manager.connect_all(console)
        loops = list(manager.loops.values())
    finally:
        manager.close_all(console)

    assert all(loop.is_closed() for loop in loops)
