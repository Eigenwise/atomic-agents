# Outcrawl Search Tool

## Overview
Searches the web through the [Outcrawl](https://outcrawl.ai) API and returns each result's title, URL, snippet, and rank. With `scrape_content` set, it also fetches every result page and returns its main content as Markdown. Requires an Outcrawl API key (the free tier includes 1,000 credits a month).

## Prerequisites and Dependencies
- Python 3.12 or later
- `atomic-agents`
- `pydantic`
- `aiohttp`
- An Outcrawl API key from [outcrawl.ai](https://outcrawl.ai)

## Installation
1. Use the Atomic Assembler CLI: run `atomic` and pick `outcrawl_search`.
2. Or copy the `tool/` folder directly into your project.

## Configuration
- `api_key` (str): Outcrawl API key. Falls back to the `OUTCRAWL_API_KEY` environment variable when empty.
- `base_url` (str): API base URL (default `https://api.outcrawl.ai/v1`).
- `user_agent` (str): user agent sent with each request.
- `timeout` (float): HTTP timeout in seconds (default 60, since page fetches can take a while).

## Input & Output Structure

### Input Schema
- `queries` (list[str]): search queries to run.
- `max_results_per_query` (int): 1-20 (default 10).
- `scrape_content` (bool): also fetch each result page as Markdown (default `false`). Each page fetch is a separate API call.

### Output Schema
A list of `OutcrawlSearchResultItem` items. Each has `query`, `title`, `url`, and optional `snippet`, `rank`, and `content` (Markdown, only when `scrape_content` is set and the page fetch succeeded).

`failures` lists unsuccessful queries and their error messages when other queries returned results. It is empty when every query succeeded.

## Usage

```python
from tool.outcrawl_search import OutcrawlSearchTool, OutcrawlSearchToolConfig, OutcrawlSearchToolInputSchema

tool = OutcrawlSearchTool(config=OutcrawlSearchToolConfig(api_key="your-outcrawl-api-key"))

output = tool.run(OutcrawlSearchToolInputSchema(
    queries=["retrieval augmented generation"],
    max_results_per_query=3,
    scrape_content=True,
))

for item in output.results:
    print(item.rank, item.title, "-", item.url)
    if item.content:
        print(item.content[:200])
```

A query that fails is logged and recorded in `failures`, so the other queries still return results. If at least one query failed and none returned results, the tool raises `RuntimeError` with the first failure. This keeps authentication, exhausted-credit, rate-limit and transport failures distinguishable from a successful search with no matches. Provider error messages have the configured API key redacted before logging or reporting.

A page that fails to fetch keeps its search result with `content` left empty; optional page-fetch failures do not turn a successful search into a failed query.

## Contributing
PRs welcome. See the main repo `CONTRIBUTING.md`.

## License
Same as the main Atomic Agents project.
