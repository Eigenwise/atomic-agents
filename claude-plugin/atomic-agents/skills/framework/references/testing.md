# Testing Agents and Tools

## Contents
- Test layout
- Testing schemas
- Testing tools (unit)
- Testing agents (without hitting a real LLM)
- Integration tests against a real provider
- Async and streaming
- Fixtures that keep tests honest

## Test layout

```
tests/
├── conftest.py
├── test_schemas.py
├── test_tools/
│   ├── test_calculator.py
│   └── test_weather.py
├── test_agents/
│   ├── test_router.py
│   └── test_summarizer.py
└── test_orchestration.py
```

`pytest` + `pytest-asyncio` covers everything. No extra runtime is needed.

## Testing schemas

Schemas are the cheapest thing to test — they validate at construction. Focus on the edge cases that matter:

```python
import pytest
from pydantic import ValidationError
from my_app.schemas import SearchQuery

def test_search_query_rejects_empty_string():
    with pytest.raises(ValidationError):
        SearchQuery(query="", limit=10)

def test_search_query_caps_limit():
    with pytest.raises(ValidationError):
        SearchQuery(query="x", limit=101)
```

Don't test the trivial cases (correct type accepted, missing required field rejected) — Pydantic already does that.

## Testing tools (unit)

Tools are plain classes. Stub any external I/O, assert on the output schema:

```python
import httpx, pytest
from my_app.tools.weather_tool import WeatherTool, WeatherConfig, WeatherInput

def test_weather_returns_error_without_key():
    tool = WeatherTool(WeatherConfig(api_key=""))
    out = tool.run(WeatherInput(city="Ghent"))
    assert out.status == "error"
    assert "WEATHER_API_KEY" in out.error

def test_weather_parses_ok_response(monkeypatch):
    def fake_get(*a, **kw):
        class R:
            def raise_for_status(self): pass
            def json(self): return {"temp_c": 12.3, "summary": "rain"}
        return R()
    monkeypatch.setattr(httpx, "get", fake_get)

    tool = WeatherTool(WeatherConfig(api_key="x"))
    out = tool.run(WeatherInput(city="Ghent"))
    assert out.status == "ok" and out.temperature_c == 12.3
```

## Testing agents (without hitting a real LLM)

Swap the Instructor client with one that returns a canned schema instance. The cleanest way is a fake `client.chat.completions.create` that returns the structured output directly. For simple cases, bypass the client entirely by stubbing `agent.run`:

```python
from unittest.mock import MagicMock
from my_app.schemas import UserQuery, Answer
from my_app.agents.qa_agent import create_qa_agent

def test_qa_agent_calls_llm_with_query():
    client = MagicMock()
    # Instructor clients route through .chat.completions.create(...)
    client.chat.completions.create.return_value = Answer(text="42")

    agent = create_qa_agent(client=client, model="gpt-5.6-terra")
    out = agent.run(UserQuery(question="meaning of life"))
    assert out.text == "42"
```

For agents that exercise validators or hooks, lean on Instructor's test helpers rather than hand-rolling client fakes. `instructor.Instructor` implementations can be constructed for testing — consult the Instructor docs for the exact factory.

## Integration tests against a real provider

Keep these opt-in behind an environment flag so they don't run in CI by default:

```python
import os, pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_PROVIDER_TESTS"),
    reason="set RUN_PROVIDER_TESTS=1 to hit the real provider",
)

def test_agent_answers_real_question():
    ...
```

Add `RUN_PROVIDER_TESTS=1` to a nightly or on-demand CI job. Pin a cheap model (`gpt-5.6-terra`, `claude-haiku-4-5`) and a tight `max_tokens`.

## Async and streaming

`pytest-asyncio` handles async agents:

```python
import pytest

@pytest.mark.asyncio
async def test_async_agent(fake_async_client):
    agent = AtomicAgent[In, Out](config=AgentConfig(client=fake_async_client, model="m"))
    out = await agent.run_async(In(...))
    assert out == expected
```

For streaming, collect the partials and assert on the final one:

```python
@pytest.mark.asyncio
async def test_streaming_agent(fake_streaming_client):
    partials = [p async for p in agent.run_async_stream(In(...))]
    assert partials[-1] == expected_final_output
```

## Fixtures that keep tests honest

- **Freeze time** in tests involving `TimeCtx` or timestamps.
- **Don't share `ChatHistory` across tests** — each test gets a fresh one, otherwise state bleeds.
- **Seed randomness** if the agent or a tool uses it.
- **Assert on hook invocations** to ensure error paths fire — register a spy hook before `run()`.

## Common mistakes

- Tests that silently call the real OpenAI API because the mock wasn't wired up — add a guard that fails loudly when `OPENAI_API_KEY` is set during unit tests.
- Asserting on free-text LLM output in unit tests — assert on schema fields or hook events instead.
- Forgetting to `reset_history()` between tests that share an agent fixture.
- Integration tests with no `max_tokens` cap — costs spiral.
