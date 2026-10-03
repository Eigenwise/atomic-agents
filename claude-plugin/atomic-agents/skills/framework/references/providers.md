# LLM Providers

Atomic Agents is provider-agnostic: any Instructor-supported client works. This reference collects the wiring per provider, the correct `Mode`, roles, and example models. Read every key from environment variables.

Model IDs in this file are illustrative. Provider model names change often — check the provider's own docs or `atomic-examples/quickstart/quickstart/4_basic_chatbot_different_providers.py` for the ones the repo currently tests against.

## Contents
- Provider matrix
- OpenAI
- Anthropic
- Groq
- Ollama (local)
- Gemini
- OpenRouter
- MiniMax
- Picking a model
- `model_api_parameters` per provider
- Installation

## Provider matrix

| Provider | Instructor factory | `Mode` | `assistant_role` | Transport |
|---|---|---|---|---|
| OpenAI | `instructor.from_openai(OpenAI(...))` | `Mode.TOOLS` (default) | `"assistant"` | Native |
| Anthropic | `instructor.from_anthropic(Anthropic(...))` | `Mode.TOOLS` | `"assistant"` | Native |
| Groq | `instructor.from_groq(Groq(...), mode=Mode.JSON)` | `Mode.JSON` | `"assistant"` | Native |
| Ollama | `instructor.from_openai(OpenAI(base_url=..., api_key="ollama"), mode=Mode.JSON)` | `Mode.JSON` | `"assistant"` | OpenAI-compatible |
| Gemini | `instructor.from_genai(google.genai.Client(...), mode=Mode.GENAI_TOOLS)` | `Mode.GENAI_TOOLS` | `"model"` | Native |
| OpenRouter | `instructor.from_openai(OpenAI(base_url=..., api_key=...))` | `Mode.TOOLS` | `"assistant"` | OpenAI-compatible |
| MiniMax | `instructor.from_openai(OpenAI(base_url=..., api_key=...), mode=Mode.JSON)` | `Mode.JSON` | `"assistant"` | OpenAI-compatible |

The `mode` value is passed to `AgentConfig(mode=...)` **and** sometimes to the Instructor factory itself. These are two different knobs:

- The **factory** mode (e.g. `instructor.from_openai(..., mode=Mode.JSON)`) decides the API call format.
- `AgentConfig(mode=...)` decides how the output schema is accounted for when counting tokens, and how multimodal content is serialized. It defaults to following the client.

They must use the same mode family. `AgentConfig(mode=...)` is optional — omit it and the client's mode is used, so setting the factory mode alone is enough. Setting it explicitly to a mode from a different family than the client's logs a warning.

This matters most for OpenAI-compatible hosts that do not speak the tools protocol: the factory needs `Mode.JSON`, and the accounting follows once `AgentConfig.mode` is left at its default or set to the same mode.

## OpenAI

```python
import os, instructor
from openai import OpenAI
from instructor import Mode

client = instructor.from_openai(OpenAI(api_key=os.environ["OPENAI_API_KEY"]))
model = "gpt-5-mini"   # also: "gpt-5", "gpt-5-nano", and reasoning variants
api_params = {"reasoning_effort": "low", "max_tokens": 2048}
```

Reasoning models (o-series, GPT-5 reasoning variants) often prefer `system_role=None` on `AgentConfig`. Pass `reasoning_effort` through `model_api_parameters`.

## Anthropic

```python
import os, instructor
from anthropic import Anthropic

client = instructor.from_anthropic(Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"]))
model = "claude-opus-4-6"        # or "claude-sonnet-4-6", "claude-haiku-4-5"
api_params = {"max_tokens": 4096}
```

Anthropic requires `max_tokens` for every call — always set it in `model_api_parameters`.

## Groq

```python
import os, instructor
from groq import Groq
from instructor import Mode

client = instructor.from_groq(Groq(api_key=os.environ["GROQ_API_KEY"]), mode=Mode.JSON)
model = "llama-3.3-70b-versatile"
api_params = {"max_tokens": 2048}
# In AgentConfig: mode=Mode.JSON
```

Groq does not support tool-calling for all models — stick with `Mode.JSON`.

## Ollama (local)

```python
import instructor
from openai import OpenAI
from instructor import Mode

client = instructor.from_openai(
    OpenAI(base_url="http://localhost:11434/v1", api_key="ollama"),
    mode=Mode.JSON,
)
model = "llama3.1"      # any model pulled with `ollama pull`
api_params = {"max_tokens": 2048}
```

No API key is needed. The literal `"ollama"` string satisfies the OpenAI SDK. Use `Mode.JSON` — Ollama's OpenAI adapter does not implement tool-calling reliably.

## Gemini

```python
import os, instructor
import google.genai
from instructor import Mode

client = instructor.from_genai(
    google.genai.Client(api_key=os.environ["GEMINI_API_KEY"]),
    mode=Mode.GENAI_TOOLS,
)
model = "gemini-2.5-flash"
api_params = {}
# In AgentConfig: assistant_role="model", mode=Mode.GENAI_TOOLS
```

Gemini is the only provider where `assistant_role` must be `"model"`. Forgetting this causes silent role confusion in multi-turn chats.

## OpenRouter

```python
import os, instructor
from openai import OpenAI

client = instructor.from_openai(OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
))
model = "anthropic/claude-opus-4-6"   # OpenRouter's model id syntax
api_params = {"max_tokens": 2048}
```

OpenRouter is a gateway to many providers via OpenAI-compatible protocol. Model ids use `provider/model` form.

## MiniMax

```python
import os, instructor
from openai import OpenAI
from instructor import Mode

client = instructor.from_openai(
    OpenAI(base_url="https://api.minimax.io/v1", api_key=os.environ["MINIMAX_API_KEY"]),
    mode=Mode.JSON,
)
model = "MiniMax-M3"   # current default; "MiniMax-M2.7" remains available as a legacy option
api_params = {"max_tokens": 2048}
```

## Picking a model

- **Drafting / simple chat** — `gpt-5-mini`, `claude-haiku-4-5`, or a Groq Llama for cost.
- **Tool use, multi-agent routing** — `gpt-5` or `claude-sonnet-4-6`.
- **Hardest reasoning** — `claude-opus-4-6` or reasoning-tier OpenAI models.
- **Local / offline** — Ollama with `llama3.1` or a code-specialized variant.

The framework is indifferent to provider choice — schemas and hooks behave the same. Swap `model` and the provider wiring, leave the rest alone.

## `model_api_parameters` per provider

Common keys:

- `max_tokens` — Anthropic requires it. Safe to set elsewhere too.
- `temperature` — all providers.
- `top_p` — most providers.
- `reasoning_effort` — OpenAI reasoning models only (`"low"`, `"medium"`, `"high"`).
- `frequency_penalty`, `presence_penalty` — OpenAI/OpenRouter.

Unknown keys are forwarded verbatim; the provider raises if it dislikes them.

## Installation

The repo uses Instructor extras to pull provider SDKs:

```toml
# pyproject.toml
[project]
dependencies = [
    "atomic-agents>=2.7",
    "instructor[openai]>=1.14",     # or [anthropic], [groq], [google-genai]
]
```

For Ollama use the `openai` extra (Ollama exposes an OpenAI-compatible endpoint). For Gemini use `instructor[google-genai]`.
