# Wikipedia Search Tool

## Overview
Searches Wikipedia and returns matching articles with titles, URLs, summaries, and (optionally) the full plain-text extract for the top hit. Uses Wikipedia's free public APIs (MediaWiki + REST) — no API key required. Supports any language edition (`en`, `fr`, `de`, `nl`, `ja`, …).

## Prerequisites and Dependencies
- Python 3.12 or later
- `atomic-agents`
- `pydantic`
- `aiohttp`

## Installation
1. Use the Atomic Assembler CLI: run `atomic` and pick `wikipedia_search`.
2. Or copy the `tool/` folder directly into your project.

## Input & Output Structure

### Input Schema
- `queries` (list[str]): one or more search queries — each is searched in parallel.
- `language` (str): Wikipedia edition code (default `en`), such as `simple` or `zh-min-nan`. Supply one ASCII hostname label, up to 63 characters, with letters/digits at either end and optional interior hyphens. URLs, hostnames containing dots, paths and query strings are rejected before opening a network session.
- `max_results_per_query` (int): how many articles to return per query (1–20, default 3).
- `full_text` (bool): when true, the top result for each query also includes the full plain-text extract.

### Output Schema
- `results`: a list of `WikipediaArticle` items. Each item has `query`, `title`, `page_url`, `summary`, `description`, `full_text` (optional), and `thumbnail_url` (optional).

## Usage

```python
from tool.wikipedia_search import WikipediaSearchTool, WikipediaSearchToolInputSchema

tool = WikipediaSearchTool()
output = tool.run(WikipediaSearchToolInputSchema(
    queries=["Atomic Agents", "Pydantic"],
    language="en",
    max_results_per_query=2,
))

for article in output.results:
    print(article.title, "->", article.page_url)
    print(article.summary)
```

## Contributing
PRs welcome — see the main repo `CONTRIBUTING.md`.

## License
Same as the main Atomic Agents project.
