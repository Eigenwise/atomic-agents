# Phoenix Tracing Example

Trace an Atomic Agents turn in [Arize Phoenix](https://github.com/Arize-ai/phoenix): a planner agent writes
an arithmetic expression, a calculator tool evaluates it, and an answer agent explains the result. One run
produces this tree in Phoenix:

```
CHAIN  answer_question
├── AGENT  invoke_agent {agent name}
│   ├── LLM  chat {model}              (Atomic Agents attempt span, one per Instructor attempt)
│   └── LLM  ChatCompletion            (from openinference-instrumentation-openai)
├── TOOL   execute_tool {tool name}
└── AGENT  invoke_agent {agent name}
    ├── LLM  chat {model}
    └── LLM  ChatCompletion
```

Running the OpenAI instrumentor next to the Atomic Agents attempt spans gives two LLM layers. To keep one, pass
`TracingConfig(llm_spans=False)` when you call `AtomicAgentsInstrumentor().instrument(...)` yourself.

The AGENT and TOOL spans come from `atomic_agents.observability`, which Phoenix activates automatically
through `register(auto_instrument=True)`. The LLM spans come from the OpenAI instrumentor. The CHAIN span
is the only manual tracing in `main.py`; without it every agent and tool call would be its own trace.

## Install

Inside this repository `uv sync` installs everything. In your own project:

```bash
pip install "atomic-agents[phoenix]" arize-phoenix-otel openinference-instrumentation-openai
```

## Run it

1. Start Phoenix locally (or point `PHOENIX_COLLECTOR_ENDPOINT` at a hosted instance):

   ```bash
   uv run --with arize-phoenix phoenix serve
   ```

   Phoenix listens on http://localhost:6006.

2. Put `OPENAI_API_KEY=...` in a `.env` file here, then run the example from this directory:

   ```bash
   uv sync
   uv run python phoenix_tracing/main.py
   ```

3. Open http://localhost:6006, pick the `atomic-agents-example` project, and open the latest trace.

## Environment

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Required. |
| `OPENAI_MODEL` | Optional, defaults to `gpt-5-mini`. |
| `PHOENIX_COLLECTOR_ENDPOINT` | Where spans go. Unset, `register()` uses gRPC to `localhost:4317`; `http://localhost:6006` sends HTTP to the UI port. |
| `PHOENIX_API_KEY` | Only for Phoenix Cloud or an authenticated server. |
| `OPENINFERENCE_HIDE_INPUTS` / `OPENINFERENCE_HIDE_OUTPUTS` | Set to `true` to redact agent and tool payloads. |

The full setup guide, including manual OpenTelemetry setup and privacy controls, is in
`docs/guides/observability.md`.
