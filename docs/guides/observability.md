# Observability with OpenTelemetry

Atomic Agents can turn every agent run, tool run and LLM attempt into an OpenTelemetry span. The spans follow the
OpenTelemetry GenAI semantic conventions, so they show up on any OpenTelemetry backend: a plain OTLP collector,
Grafana Tempo, Sentry, Jaeger, Honeycomb, Datadog, Arize Phoenix, Langfuse and so on. Only `opentelemetry-api` is
required. Nothing changes for applications that do not opt in.

You get three kinds of spans:

- **AGENT**: one per `AtomicAgent` run (`run`, `run_async`, `run_stream`, `run_async_stream`).
- **TOOL**: one per `BaseTool` run, including MCP tools.
- **LLM attempt**: one per call Instructor makes to the model, so retries after a validation error are visible.

## Install

```bash
pip install "atomic-agents[observability]" opentelemetry-sdk opentelemetry-exporter-otlp-proto-http
```

The `observability` extra installs `opentelemetry-api` and nothing else. You bring the SDK and an exporter for your
backend. The `phoenix` extra adds the OpenInference packages:

```bash
pip install "atomic-agents[phoenix]"
```

With those installed, Atomic Agents also writes the OpenInference attribute family (turned on automatically) and the
OpenInference context managers `using_session`, `using_user` and `using_metadata` start to work. Backends that read
OpenInference attributes, such as Arize Phoenix, use them for their own views. Every other backend works from the GenAI
attributes alone.

## Setup with any OpenTelemetry backend (manual)

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from atomic_agents.observability import AtomicAgentsInstrumentor, TracingConfig

tracer_provider = TracerProvider()
tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))

AtomicAgentsInstrumentor().instrument(tracer_provider=tracer_provider, config=TracingConfig())
```

`OTLPSpanExporter()` reads the standard `OTEL_EXPORTER_OTLP_*` environment variables, so this code stays the same for
every backend below. Only the environment changes. `tracer_provider` and `config` are both optional: without a provider,
the global one is used, and without a config you get the defaults described under "Privacy and size".

Call `instrument()` once, at import time or in your service's startup hook, before any agent runs.

## Backends

The GenAI attributes are the OpenTelemetry standard, and backends with LLM views (Datadog LLM Observability, Langfuse,
Phoenix, Sentry AI monitoring, PostHog) read them. Every recipe below uses the setup above. Only the exporter
configuration differs.

What has actually been run: the span output is tested offline against an in-memory exporter, and the Phoenix recipe is
the [phoenix-tracing example](https://github.com/eigenwise/atomic-agents/tree/main/atomic-examples/phoenix-tracing).
Everything marked "untested" follows the vendor's documented OTLP settings but has not been run against a live account.
Check the vendor's page before relying on it.

**Plain OTLP collector** (untested against a live collector):

```bash
export OTEL_SERVICE_NAME=my-app
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
```

**Grafana Tempo** (untested): send OTLP to Tempo directly or through a collector or Grafana Alloy. Self-hosted, point the
endpoint at Tempo's OTLP HTTP port (4318 by default). Grafana Cloud uses its OTLP gateway plus a basic-auth header built
from your instance ID and an access token:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp-gateway-<zone>.grafana.net/otlp
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic%20<base64 of instance_id:token>"
```

**Jaeger** (untested): current Jaeger releases accept OTLP directly.

```bash
docker run --rm -p 16686:16686 -p 4318:4318 jaegertracing/all-in-one
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

**Sentry** (follows Sentry's Python OTLP integration docs, untested here). The integration sets up the exporter itself,
so you do not build a `TracerProvider` and do not set `traces_sample_rate`:

```bash
pip install "atomic-agents[observability]" "sentry-sdk[opentelemetry-otlp]"
```

```python
import sentry_sdk
from sentry_sdk.integrations.otlp import OTLPIntegration

from atomic_agents.observability import AtomicAgentsInstrumentor

sentry_sdk.init(dsn="https://<key>@o<orgId>.ingest.sentry.io/<projectId>", integrations=[OTLPIntegration()])
AtomicAgentsInstrumentor().instrument()
```

Sentry's OTLP ingest drops span events. The `validation_error` event and recorded exceptions do not arrive there, while
`error.type`, the span status and `atomic_agents.llm.last_validation_error` do. Whether Sentry's AI Agents dashboard
fills from these spans is unverified.

**Honeycomb** (untested): use `https://api.eu1.honeycomb.io` for the EU region. Honeycomb's Agent Timeline groups by
`gen_ai.conversation.id`, so wrap calls in `using_conversation` (see "Naming and conversations").

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://api.honeycomb.io
export OTEL_EXPORTER_OTLP_HEADERS="x-honeycomb-team=<api key>"
```

**Datadog** (untested): send OTLP to a Datadog Agent with OTLP ingestion enabled (HTTP on 4318 by default) and set
`OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318`. Datadog LLM Observability maps `invoke_agent` spans to agents,
`execute_tool` to tools and `chat` to LLM calls, and reads the `gen_ai.usage.*` tokens and `gen_ai.conversation.id`.
It reads prompt and response text only from `gen_ai.input.messages` and `gen_ai.output.messages`, which Atomic Agents
does not emit, so you see spans, tokens and sessions but not the agent text. PostHog behaves the same way.

**Arize Phoenix**: see "Setup with Phoenix" below.

**Langfuse** (untested): Langfuse takes OTLP at `/api/public/otel` with basic auth built from your public and secret
key. Its default filter keeps spans that carry `gen_ai.*` attributes, which these spans do.

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://cloud.langfuse.com/api/public/otel
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic%20<base64 of public_key:secret_key>"
```

**PostHog** (untested): PostHog AI Observability ingests OTLP over HTTP only (protobuf or JSON, no gRPC). Use the
`TRACES` variables, because the generic endpoint variable appends `/v1/traces` and gets a 404. Use your project's client
API host in the URL, see [PostHog's OpenTelemetry page](https://posthog.com/docs/ai-observability/installation/opentelemetry).
`chat` spans become `$ai_generation` events and the others become `$ai_span`.

```bash
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="<client api host>/i/v0/ai/otel"
export OTEL_EXPORTER_OTLP_TRACES_HEADERS="Authorization=Bearer%20<project token>"
```

## Setup with the OpenTelemetry launcher (automatic)

Install the tracing adapter, launcher, SDK and exporter explicitly:

```bash
pip install "atomic-agents[observability]" opentelemetry-distro opentelemetry-exporter-otlp-proto-http
```

Run your existing application with environment-only configuration:

```bash
export OTEL_SERVICE_NAME=my-app
export OTEL_TRACES_EXPORTER=otlp
export OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
export OTEL_METRICS_EXPORTER=none
export OTEL_LOGS_EXPORTER=none
export ATOMIC_AGENTS_TRACE_HIDE_INPUTS=true
export ATOMIC_AGENTS_TRACE_HIDE_OUTPUTS=true
export ATOMIC_AGENTS_TRACE_LLM_SPANS=false
opentelemetry-instrument python app.py
```

The launcher discovers `atomic_agents` through the `opentelemetry_instrumentor` entry point and activates the same
adapter as the manual setup. Without `opentelemetry-api` installed, the loader skips the adapter.
`opentelemetry-bootstrap` does not automatically install Atomic Agents or its extras.

To disable the adapter, set `OTEL_PYTHON_DISABLED_INSTRUMENTATIONS=atomic_agents` (or add `atomic_agents` to an
existing comma-separated list). Ordinary imports without a launcher or explicit setup stay uninstrumented.

Choose one provider configuration path: the launcher, Phoenix registration, or manual setup. The discovery entry points
share one installation, so activating several does not duplicate spans. OpenTelemetry itself keeps its first global
provider, so calling Phoenix registration after launcher startup does not switch where spans are exported. See
"Re-instrumentation" for what a second `instrument()` call does.

The launcher adds AGENT, TOOL and LLM-attempt spans and no parent span. The example above disables Atomic Agents' LLM attempt
spans for applications that also use a provider instrumentor. Leave `ATOMIC_AGENTS_TRACE_LLM_SPANS` unset to keep
attempt spans enabled. Use the grouping example below when you want several agent and tool calls in one trace.

## Setup with Phoenix (automatic)

`register(auto_instrument=True)` also activates the OpenAI instrumentor, so set
`ATOMIC_AGENTS_TRACE_LLM_SPANS=false` (before import) to avoid two LLM layers:

```python
import os

os.environ.setdefault("ATOMIC_AGENTS_TRACE_LLM_SPANS", "false")

from phoenix.otel import register

tracer_provider = register(project_name="my-app", auto_instrument=True)
```

`register(auto_instrument=True)` discovers the Atomic Agents instrumentor through the `openinference_instrumentor` entry
point and activates it together with every other OpenInference instrumentor in the environment. Install
`atomic-agents[phoenix]` (or `arize-phoenix-otel`, which brings the same packages) and do this once, at import time or in
your startup hook, before any agent runs. Phoenix reads `PHOENIX_COLLECTOR_ENDPOINT`, `PHOENIX_PROJECT` (alias
`PHOENIX_PROJECT_NAME`) and `PHOENIX_API_KEY` from the environment. With no endpoint set, `register()` exports over gRPC
to `localhost:4317`; set `PHOENIX_COLLECTOR_ENDPOINT=http://localhost:6006` to send HTTP to the port the Phoenix UI
serves on.

Two things to know about the automatic path:

- Phoenix imports `atomic_agents` to load the entry point. That import is the same as any `import atomic_agents` in
  your application, including the Instructor streaming patch the framework applies at import.
- Existing `auto_instrument=True` users start exporting agent and tool payloads when they upgrade to a version with the
  instrumentor. Use the privacy controls below if that is not wanted.

Langfuse and other OpenInference-aware backends use the same OpenInference attributes, so the Phoenix-style setup, or
the manual setup with the `phoenix` extra installed, works for them too.

## What you get

| Call | Span name | Kind |
|------|-----------|------|
| `AtomicAgent.run`, `run_async`, `run_stream`, `run_async_stream` | `invoke_agent {agent name}` | AGENT |
| `run`, `run_async`, `arun` on any `BaseTool` subclass | `execute_tool {tool name}` | TOOL |
| Each model call Instructor makes during an agent run | `chat {request model}` | LLM attempt |

The agent name is `AgentConfig.name`, or the class name when you do not set one. The tool name is the name the model
sees (`tool_name` on the tool). Spans attach to the current context and never start a new trace when a parent span
exists, so your own request or job span becomes their parent. An agent span is the parent of its LLM attempt spans, and
a tool span nests under whatever span is current when the tool runs. A span that fails gets
status `ERROR`, the exception recorded, and `error.type` set to the exception class name.

Attributes, GenAI keys first. All of them are written whether or not the OpenInference packages are installed.

| Key | On | Value |
|-----|----|-------|
| `gen_ai.operation.name` | all | `invoke_agent`, `execute_tool` or `chat` |
| `gen_ai.agent.name` | AGENT | Agent name |
| `gen_ai.tool.name` | TOOL | Tool name the model sees |
| `gen_ai.tool.description` | TOOL | Tool description |
| `gen_ai.tool.type` | TOOL | `function` |
| `gen_ai.tool.call.arguments` | TOOL | Tool input as JSON (hidden by `hide_inputs`, size-capped) |
| `gen_ai.tool.call.result` | TOOL | Tool output as JSON (hidden by `hide_outputs`, size-capped) |
| `gen_ai.request.model` | AGENT, LLM | Model the agent was configured with, or the model in the completion call |
| `gen_ai.provider.name` | AGENT, LLM | Instructor provider mapped to the OpenTelemetry value, for example `azure.ai.openai`, `gcp.vertex_ai`, `aws.bedrock`; `openai` and `anthropic` pass through |
| `gen_ai.response.model` | AGENT, LLM | Model the provider reported; on the AGENT span, the last response's model |
| `gen_ai.usage.input_tokens` | AGENT, LLM | Input tokens (AGENT: total across attempts) |
| `gen_ai.usage.output_tokens` | AGENT, LLM | Output tokens |
| `gen_ai.usage.cache_read.input_tokens` | AGENT, LLM | Cached input tokens, when the provider reports them |
| `gen_ai.usage.reasoning.output_tokens` | AGENT, LLM | Reasoning tokens, when the provider reports them |
| `gen_ai.conversation.id` | all | Set by `using_conversation` |
| `error.type` | failed spans | Exception class name |

Atomic Agents' own keys:

| Key | On | Value |
|-----|----|-------|
| `atomic_agents.agent.class` | AGENT | For example `AtomicAgent[InputSchema, OutputSchema]` (a subclass reports its own class) |
| `atomic_agents.tool.class` | TOOL | Tool class name |
| `atomic_agents.model` | AGENT | Same as the request model |
| `atomic_agents.mode` | AGENT | Instructor mode |
| `atomic_agents.input_schema`, `atomic_agents.output_schema` | AGENT | Schema class names |
| `atomic_agents.history.message_count` | AGENT | History size entering the turn |
| `atomic_agents.context_providers` | AGENT | Registered context provider names |
| `atomic_agents.system_prompt.sha256` | AGENT | Hash of the static prompt sections (background, steps, output instructions). Context provider output is not included, so it identifies the prompt version even when inputs are hidden |
| `atomic_agents.input`, `atomic_agents.output` | AGENT | Agent input and output as JSON (hidden and size-capped as described under "Privacy and size"). A stream records the final response, or the last partial one |
| `atomic_agents.stream.closed_early` | AGENT | `true` when a stream consumer stopped early or the task was cancelled (not reported as an error) |
| `atomic_agents.llm.attempts` | AGENT | Number of model calls in the run |
| `atomic_agents.llm.last_validation_error` | AGENT | Text of the last validation error |
| `atomic_agents.llm.attempt` | LLM | 1-based attempt number |

OpenInference keys, optional. These are written only when the OpenInference packages are installed (the `phoenix`
extra), or when you force them with `TracingConfig(openinference_attributes=True)`; set it to `False` to drop them even
when the packages are present.

| Key | On | Value |
|-----|----|-------|
| `openinference.span.kind` | all | `AGENT`, `TOOL` or `LLM` |
| `input.value`, `input.mime_type` | AGENT, TOOL | Same payload as `atomic_agents.input` or `gen_ai.tool.call.arguments` |
| `output.value`, `output.mime_type` | AGENT, TOOL | Same payload as `atomic_agents.output` or `gen_ai.tool.call.result` |
| `agent.name` | AGENT | Agent name |
| `tool.name`, `tool.description` | TOOL | Same as the GenAI tool keys |
| `tool.parameters` | TOOL | The input JSON schema (size-capped, not hidden) |
| `session.id` | all | The conversation id, plus anything from `using_session`, `using_user` and `using_metadata` |

## LLM attempt spans

Instructor drives the model call and its retry loop, so the detail of each attempt only exists at the Atomic Agents
layer. The instrumentor registers hooks on the agent's Instructor client (once per client, on the first instrumented
run) and removes them again in `uninstrument()`:

- `completion:kwargs` starts the `chat {model}` span as a child of the agent span and counts the attempt.
- `completion:response` sets `gen_ai.response.model` and the token usage on that span and ends it with status OK.
- `completion:error` records the exception, sets `error.type` and ends the span with status `ERROR`.
- `parse:error` adds a `validation_error` event (with `atomic_agents.llm.attempt`) to the AGENT span and sets
  `atomic_agents.llm.last_validation_error`. Instructor then retries, which opens the next attempt span.
- `completion:usage` writes the running totals to the AGENT span: the four `gen_ai.usage.*` keys and
  `atomic_agents.llm.attempts`.

An attempt is one call to the model. A retry is any attempt after the first, normally because the response failed
validation, so two attempt spans under one agent span means one retry. Hooks you registered on the same client yourself
keep running next to these, because Instructor keeps a list of handlers per event. If the agent span ends while an
attempt is still open, the attempt span is ended with an unset status. The hooks never raise into your code.

OpenTelemetry's GenAI guidance says an agent framework should not emit model-call spans by default when the LLM library
already does. Atomic Agents emits them by default because the attempt and validation detail comes from Instructor, which
no provider instrumentor sees. If you also run a provider instrumentor (`openinference-instrumentation-openai` and
friends), turn the attempt spans off or you get two LLM layers:

```python
AtomicAgentsInstrumentor().instrument(tracer_provider=tracer_provider, config=TracingConfig(llm_spans=False))
```

With `llm_spans=False` the usage totals and attempt count still land on the AGENT span, and the `validation_error` event
still fires.

## Naming and conversations

Give each agent a stable name, so spans group the same way across deployments and class renames:

```python
agent = AtomicAgent[InputSchema, OutputSchema](
    config=AgentConfig(client=client, model="gpt-5-mini", name="orchestrator_planning")
)
```

The span becomes `invoke_agent orchestrator_planning`. Without a name, the class name is used.

Group related runs into a conversation with `using_conversation`:

```python
from atomic_agents.observability import using_conversation

with using_conversation(conversation_id):
    response = agent.run(user_message)
```

Every AGENT, TOOL and LLM attempt span started inside the block gets `gen_ai.conversation.id` (and `session.id` when
OpenInference keys are on). With the `phoenix` extra installed, `using_session`, `using_user` and `using_metadata` from
`openinference.instrumentation` keep working, and a `using_session` value is copied to `gen_ai.conversation.id` when you
have not set one with `using_conversation`.

For a streaming endpoint, put the context manager inside the generator that yields the chunks, not around the endpoint
function. The generator runs after the endpoint returned, so a context manager around the endpoint body no longer
applies.

## Privacy and size

By default, agent and tool payloads are exported. `TracingConfig` controls that:

```python
TracingConfig(hide_inputs=True, hide_outputs=True, max_attribute_length=2048)
```

| Field | Default | Effect |
|-------|---------|--------|
| `hide_inputs` | `False` | Input payload values become `__REDACTED__` |
| `hide_outputs` | `False` | Output payload values become `__REDACTED__` |
| `max_attribute_length` | `4096` | Longer values are cut to this many characters and end in `...[truncated]` |
| `openinference_attributes` | auto | See "What you get" |
| `llm_spans` | `True` | See "LLM attempt spans" |

When no config is passed to `instrument()`, the settings come from the environment. `ATOMIC_AGENTS_TRACE_HIDE_INPUTS`
and `ATOMIC_AGENTS_TRACE_HIDE_OUTPUTS` are read first, then `OPENINFERENCE_HIDE_INPUTS` and `OPENINFERENCE_HIDE_OUTPUTS`
as a fallback (`true` or `1`). `ATOMIC_AGENTS_TRACE_MAX_ATTRIBUTE_LENGTH` sets the size cap.
`ATOMIC_AGENTS_TRACE_LLM_SPANS` disables attempt spans for `false` or `0` (case-insensitive); any other value or unset
keeps them enabled. If you pass your own
`TracingConfig(...)`, the environment is not read; use `TracingConfig.from_environment()` to start from it. An
OpenInference `TraceConfig` is also accepted for `config`; its `hide_inputs` and `hide_outputs` override those
environment settings, while the other settings still come from the environment.

A truncated value gets a companion attribute with the original size, for example `atomic_agents.input.original_length`.
Hiding wins over truncation, and a hidden payload is never serialized in the first place.

Covered by hiding and truncation: `atomic_agents.input`, `atomic_agents.output`, `input.value`, `output.value`,
`gen_ai.tool.call.arguments`, `gen_ai.tool.call.result` and `atomic_agents.llm.last_validation_error`. `tool.parameters`
is truncated but never hidden. Status descriptions and recorded exception messages are truncated too, and under
`hide_outputs` a Pydantic validation error message is replaced by `__REDACTED__` plus the error count, because the raw
model output is quoted inside it.

Instructor's own logging is separate from Atomic Agents' tracing: in the installed Instructor 1.17.0 source,
`instructor.v2.retry` (re-exported through `instructor.core.retry`) logs per-attempt validation failures at DEBUG
and retry exhaustion's raw last error, including model output, at ERROR regardless of `hide_outputs`.
Setting its level to ERROR does not silence exhaustion. To keep model output out of logs, use CRITICAL below
or a filter/handler that drops those records. The logger name and levels may change in later Instructor versions.
Anything that forwards ERROR log records off the host carries that text with it: Sentry's `LoggingIntegration`
(`event_level=logging.ERROR` by default) turns the exhaustion record into a Sentry event, and OpenTelemetry or Datadog
log handlers export it the same way. If `hide_outputs` matters to you, silence the logger before those handlers attach.

```python
import logging
logging.getLogger("instructor.v2.retry").setLevel(logging.CRITICAL)
```

Not covered: tool names and descriptions, the system prompt hash, and whatever you put in `using_metadata`. A provider
instrumentor and your own tracer apply their own settings, so configure them the same way if you want matching masking.
Multimodal inputs (images, audio, PDFs) are serialized into the input payload in full, so `max_attribute_length` is the
control for those.

## Grouping agent and tool calls into one trace

Atomic Agents has no tool-calling loop: your code calls `agent.run(...)`, then `tool.run(...)`, then another agent.
Each of those is a root span, so without a parent they become separate traces. Open one parent span per unit of work
with a plain OpenTelemetry tracer. It also answers "where do the seconds go": add child spans for retrieval or database
work next to the agent and tool calls.

```python
tracer = tracer_provider.get_tracer(__name__)

with tracer.start_as_current_span("answer_question"):
    with tracer.start_as_current_span("retrieve_documents"):
        documents = search_index(question)
    plan = planner.run(BasicChatInputSchema(chat_message=question))
    calculation = calculator.run(CalculatorToolInputSchema(expression=plan.expression))
    answer = answerer.run(CalculationResultSchema(question=question, expression=plan.expression, result=calculation.result))
```

The agent, tool and attempt spans become children of `answer_question`, and the slowest step stands out in the
waterfall. The [phoenix-tracing example](https://github.com/eigenwise/atomic-agents/tree/main/atomic-examples/phoenix-tracing)
is this flow. On the Phoenix path the provider's tracers are OpenInference `OITracer`s, which also take
`openinference_span_kind="chain"` and `set_input`/`set_output` if you want Phoenix to label the parent as a CHAIN.

## Re-instrumentation

`instrument()` with the same provider and config as the active installation does nothing. With a different provider or
config it replaces them in place and logs a WARNING that says so. Spans already in flight finish on the provider they
started with; new spans use the new one. `uninstrument()` followed by `instrument()` is the clean way to change
settings, and `uninstrument()` restores the original methods on `AtomicAgent` and on every tool class.

In a test suite, call `instrument(tracer_provider=provider)` with a fresh in-memory provider in each test. The replace
behavior means you do not need to uninstrument between tests, and the warning is expected there:

```python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
AtomicAgentsInstrumentor().instrument(tracer_provider=provider)
agent.run(user_input)
assert [span.name for span in exporter.get_finished_spans()][-1] == "invoke_agent my_agent"
```

## Services and async

- Call `instrument(...)` (or `register(...)`) at import or in the startup hook so every worker process is instrumented.
- Use a `BatchSpanProcessor` (or `register(batch=True)`) in services so span export does not block request handling.
- Concurrent `run_async` calls, including `asyncio.gather` over several agents sharing one client, get independent
  spans and their own attempt spans.
- Streaming token counts need `model_api_parameters={"stream_options": {"include_usage": True}}` for OpenAI.

## Limitations

- Methods bound before instrumentation stay untraced: `callback = agent.run` taken before `instrument()` keeps
  calling the original.
- An `AtomicAgent` subclass that overrides `run` (or a stream method) without calling `super()` bypasses the wrapper
  and emits no span. Open a span in the override yourself, with `tracer.start_as_current_span(...)` or `@tracer.agent`
  on an `OITracer`.
- Tool methods that are async generators are left untraced. Open a span inside the generator body with
  `tracer.start_as_current_span(...)`, or use `@tracer.tool` on an `OITracer`.
- A plain `ThreadPoolExecutor` starts its worker with an empty context, so a tool that submits work to one loses the
  current span and the re-entrancy guard inside the pool. Submit through a copied context instead, as the forge tools
  do: `executor.submit(contextvars.copy_context().run, asyncio.run, self.run_async(params))`. `asyncio.to_thread` and
  `asyncio.run` already copy the context.
- The AGENT span records the final (or last partial) response of a stream, not every chunk.
- The system prompt text is not on any span. The AGENT span carries its SHA-256 instead.
- Datadog and PostHog read prompt and response text from `gen_ai.input.messages` and `gen_ai.output.messages`, which
  are not emitted.
