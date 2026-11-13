"""Measure tracing setup/teardown for a generated MCP tool catalog, without I/O.

Run from the repository root with ``PYTHONPATH=atomic-agents python
atomic-agents/benchmarks/benchmark_tool_tracing.py --tools 4000 --repeat 3``.
Model/class creation is outside the timed region. Each timed iteration installs
and removes instrumentation and verifies every generated method is restored.
"""

import argparse
import hashlib
import json
import time

from opentelemetry.trace import NoOpTracerProvider

from atomic_agents.connectors.mcp import MCPFactory, MCPToolDefinition
from atomic_agents.observability import installation
from atomic_agents.observability.tracing_config import TracingConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools", type=int, default=4000)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    schema = {"type": "object", "properties": {"value": {"type": "string"}}, "required": ["value"]}
    definitions = [MCPToolDefinition(f"tool_{i}", "Echo a value", schema, schema) for i in range(args.tools)]
    tools = MCPFactory(mcp_endpoint="http://unused.invalid")._create_tool_classes(definitions)
    assert len(tools) == args.tools
    originals = [(tool.run, tool.arun) for tool in tools]
    provider = NoOpTracerProvider()
    config = TracingConfig(openinference_attributes=False)

    # Warm up the optional integration loader before collecting measurements.
    installation.install(tracer_provider=provider, config=config)
    installation.uninstall()
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    for _ in range(args.repeat):
        installation.install(tracer_provider=provider, config=config)
        assert all(getattr(tool.run, installation.TRACED_MARKER, False) for tool in tools)
        assert all(getattr(tool.arun, installation.TRACED_MARKER, False) for tool in tools)
        installation.uninstall()
        assert all(tool.run is run and tool.arun is arun for tool, (run, arun) in zip(tools, originals))
    wall_ms = (time.perf_counter() - wall_start) * 1000
    cpu_ms = (time.process_time() - cpu_start) * 1000
    digest = hashlib.sha256("\n".join(tool.__name__ for tool in tools).encode()).hexdigest()
    print(json.dumps({"tools": args.tools, "repeat": args.repeat, "wall_ms": wall_ms, "cpu_ms": cpu_ms, "sha256": digest}))


if __name__ == "__main__":
    main()
