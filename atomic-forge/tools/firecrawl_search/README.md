# Firecrawl Search Tool

## Overview
Searches the web or recent news through the [Firecrawl Search](https://www.firecrawl.dev/search) API and returns each result's title, URL, snippet, and rank, plus the publish date for news. With `include_content=True` the main content of each result page also comes back as Markdown in the same call, so an agent does not need a separate fetch step to read the pages. Requires a Firecrawl API key.

## Prerequisites and Dependencies
- Python 3.12 or later
- `atomic-agents`
- `pydantic`
- `aiohttp`
- A Firecrawl API key from [firecrawl.dev/app/api-keys](https://www.firecrawl.dev/app/api-keys), set as `FIRECRAWL_API_KEY` or passed in the config

## Installation
1. Use the Atomic Assembler CLI: run `atomic` and pick `firecrawl_search`.
2. Or copy the `tool/` folder directly into your project.

## Configuration
- `api_key` (str): Firecrawl API key. Falls back to the `FIRECRAWL_API_KEY` environment variable when empty.
- `base_url` (str): API base URL (default `https://api.firecrawl.dev/v2`).
- `location` (str, optional): location to localise results, e.g. `Germany`.
- `time_range` (str, optional): `day`, `week`, `month`, or `year`. Only applies to web results.
- `max_content_chars` (int, optional): truncate each result's `content` to this many characters. No limit by default.
- `timeout` (float): HTTP timeout in seconds (default 75, since requests with `include_content` fetch each page).

## Input & Output Structure

### Input Schema
- `queries` (list[str]): search queries to run. Operators such as `site:`, `"exact phrase"`, `-term`, and `filetype:pdf` are supported.
- `search_type` (str): `web` or `news`. Default `web`.
- `max_results_per_query` (int): 1-100 (default 5, kept low because each result can carry a full page with `include_content`).
- `include_content` (bool): also return each result page as Markdown (default `False`).

### Output Schema
A list of `FirecrawlSearchResultItem` items. Each has `query`, `title`, `url`, and optional `description` (a query-relevant excerpt or snippet), `position`, `published` (news), and `content` (Markdown, when `include_content` is set).

## Usage
If every query fails (for example a wrong API key), `run` raises the error. If only some fail, the others are returned and the failures are logged.

```python
from tool.firecrawl_search import FirecrawlSearchTool, FirecrawlSearchToolConfig, FirecrawlSearchToolInputSchema

tool = FirecrawlSearchTool(config=FirecrawlSearchToolConfig(api_key="your-firecrawl-api-key"))

output = tool.run(FirecrawlSearchToolInputSchema(
    queries=["retrieval augmented generation"],
    max_results_per_query=3,
    include_content=True,
))

for item in output.results:
    print(item.title, "-", item.url)
    print(item.content[:500] if item.content else item.description)
```

The request and response formats are documented at [docs.firecrawl.dev/features/search](https://docs.firecrawl.dev/features/search).

## Contributing
PRs welcome. See the main repo `CONTRIBUTING.md`.

## License
Same as the main Atomic Agents project.
