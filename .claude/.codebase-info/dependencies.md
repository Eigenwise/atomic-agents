# Dependencies

*Last Updated: 2026-10-21*

Declared in `pyproject.toml` (Hatchling build, `uv` workspace; versions locked in `uv.lock`).

## Runtime — core
| Package | Purpose |
|---------|---------|
| instructor (>=1.17.0,<2) | Structured (Pydantic) LLM outputs across providers — the core dependency |
| pydantic (≥2.11.4,<3) | Schemas / data validation for all I/O; 2.11.4 is the first version whose `create_model` accepts `__config__` together with `__base__` (pydantic#11714), which the MCP schema transformer relies on |
| litellm (≥1.50,<2) | Provider-agnostic token counting & model metadata |
| rich (≥13.7.1,<14) | Console formatting |

## Optional extras — `observability` and `phoenix`
| Extra | Package | Purpose |
|-------|---------|---------|
| `observability` | opentelemetry-api (≥1.30,<2) | Tracer / context API; the only runtime dependency of `atomic_agents.observability` |
| `phoenix` | opentelemetry-api, openinference-instrumentation (≥0.1.38,<0.2), openinference-semantic-conventions (≥0.1.20,<0.2) | Adds the OpenInference attribute family and `using_session` / `using_user` / `using_metadata`; needed for Arize Phoenix views |

Users bring their own SDK and exporter (`opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-http`, ...). The dev
group carries the OpenInference packages plus `opentelemetry-sdk` so the tests run; `arize-phoenix-otel` already
brings the OpenInference packages, so the Phoenix auto path works without the `phoenix` extra.

## Runtime — CLI / tooling
| Package | Purpose |
|---------|---------|
| textual (≥5.3,<6) | TUI framework for the `atomic` assembler |
| mcp[cli] (≥1.8,<2) | Model Context Protocol client + CLI; `mcp.client.streamable_http` first shipped in 1.8.0 and was renamed in 2.0.0, so 2.x waits for a migration |
| requests (≥2.32,<3) | HTTP |
| gitpython (≥3.1.43,<4) | Clone the repo to fetch forge tools |
| pyyaml (≥6,<7) | Read tool `config.yaml` metadata |
| pyfiglet (≥1,<2) | ASCII-art banners in the TUI |

## Development
| Package | Purpose |
|---------|---------|
| black (≥24.8,<25) | Formatter (line length 127) |
| flake8 (≥7.1,<8) | Linter |
| pre-commit (≥4,<5) | Git hook runner |
| pytest / pytest-cov / pytest-asyncio | Tests, coverage, async tests |
| openai (≥2,<3) | Example/test provider client |
| opentelemetry-instrumentation (≥0.51b0,<1) | Real auto-loader and launcher tests; dev-only, not a runtime dependency |

## Docs
`sphinx`, `sphinx-rtd-theme`, `myst-parser`, `sphinx-copybutton`, `sphinx-design`,
`sphinx-autobuild`, `sphinxcontrib-mermaid`, `pdoc3`, `beautifulsoup4`, `markdownify`.

## Console script
`atomic = atomic_assembler.main:main` (from `[project.scripts]`).

## Notes
- `requirements.txt` / `setup.py` are legacy shims; `pyproject.toml` + `uv.lock` are the source of truth.
- Instructor was an exact pin (`==1.14.5`) through 2.12.1 because patch releases had removed import paths (567-labs/instructor#1796); it is a range now, guarded by two CI jobs: `fresh-install` (build the wheel, install it without the lockfile, import every module) and `lowest-versions` (run the suite with every runtime dependency at its declared floor). The framework references `instructor.Instructor` / `instructor.AsyncInstructor` from the package root; `instructor.core` is not an attribute path on 1.17.0.
- Every example project and the legacy `requirements.txt` carry the same Instructor range.
- Each `atomic-forge` tool and each `atomic-examples` project declares its **own** dependencies.
