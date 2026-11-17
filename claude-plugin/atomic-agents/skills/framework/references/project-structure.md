# Project Structure

## Contents
- Size-based layouts
- `pyproject.toml` template
- Configuration pattern
- Environment and `.gitignore`
- Naming conventions

## Size-based layouts

Pick the smallest layout that fits. Grow into the next tier when the current one feels cramped.

### Single-agent script (prototyping)

```
my_app/
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
└── my_app/
    ├── __init__.py
    ├── main.py           # entry + agent setup
    └── schemas.py        # input/output schemas
```

### Medium app (a few agents, one or two tools)

```
my_app/
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
└── my_app/
    ├── __init__.py
    ├── main.py
    ├── config.py
    ├── schemas.py
    ├── agents/
    │   ├── __init__.py
    │   ├── router_agent.py
    │   └── worker_agent.py
    └── tools/
        ├── __init__.py
        └── search_tool.py
```

### Large app (several agents, services, providers)

```
my_app/
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
├── tests/
│   ├── test_agents.py
│   └── test_tools.py
└── my_app/
    ├── __init__.py
    ├── main.py
    ├── config.py
    ├── agents/
    ├── tools/
    ├── schemas/
    │   ├── __init__.py
    │   ├── inputs.py
    │   └── outputs.py
    ├── context_providers/
    ├── services/
    └── presentation/
```

Split schemas by direction (`inputs.py`, `outputs.py`) only when they start to collide in a single file. Until then, one `schemas.py` is easier to navigate.

## `pyproject.toml` template

```toml
[project]
name = "my-app"
version = "0.1.0"
description = "…"
readme = "README.md"
requires-python = ">=3.12"
license = { text = "MIT" }
authors = [{ name = "Your Name", email = "you@example.com" }]

dependencies = [
    "atomic-agents>=2.7",
    "instructor[openai]>=1.14",   # swap extras per provider
    "pydantic>=2",
    "python-dotenv>=1",
    "rich>=13",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-asyncio>=0.23",
    "ruff>=0.5",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["my_app"]
```

`instructor[openai]` pulls in the `openai` package. Use `[anthropic]`, `[groq]`, `[google-genai]` for other providers. The monorepo depends on instructor extras specifically to keep provider SDK versions consistent.

## Configuration pattern

```python
# my_app/config.py
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Config:
    openai_api_key: str
    model: str = "gpt-5.6-terra"
    max_tokens: int = 2048
    temperature: float = 0.2

    @classmethod
    def from_env(cls) -> "Config":
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        return cls(
            openai_api_key=key,
            model=os.environ.get("MY_APP_MODEL", cls.model),
        )

config = Config.from_env()
```

Freeze config objects so they are safe to pass around. Load them at process start — never inside a per-request handler.

## Environment and `.gitignore`

`.env.example` (commit this):

```bash
OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=...
# GEMINI_API_KEY=...
# GROQ_API_KEY=...
MY_APP_MODEL=gpt-5.6-terra
```

`.gitignore` (add these):

```gitignore
.env
.env.*.local
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
.ruff_cache/
.coverage
htmlcov/
dist/
build/
*.egg-info/
```

Commit `.env.example`, never `.env`.

## Naming conventions

| Component | File | Class |
|---|---|---|
| Agent module | `agents/<purpose>_agent.py` | `<Purpose>Agent` (or factory `create_<purpose>_agent`) |
| Tool module | `tools/<name>_tool.py` | `<Name>Tool` |
| Schema | `schemas.py` or `schemas/*.py` | `<Purpose>Input`, `<Purpose>Output` |
| Context provider | `context_providers/<name>_ctx.py` | `<Name>Ctx` |
| Service | `services/<name>.py` | `<Name>Service` |

Factory functions (`create_*_agent(client, model) -> AtomicAgent[...]`) keep tests simple and make dependency wiring explicit.
