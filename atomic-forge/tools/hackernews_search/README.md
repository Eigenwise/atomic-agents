# Hacker News Search Tool

## Overview
Searches Hacker News stories, comments, Show HN, Ask HN, and polls via the free Algolia HN search API. Supports relevance and date sorting and arbitrary numeric filters (`points>100`, `num_comments>50`, etc.). No authentication required.

## Prerequisites and Dependencies
- Python 3.12 or later
- `atomic-agents`
- `pydantic`
- `aiohttp`

## Installation
1. Use the Atomic Assembler CLI: run `atomic` and pick `hackernews_search`.
2. Or copy the `tool/` folder directly into your project.

## Input & Output Structure

### Input Schema
- `queries` (list[str]): one or more search queries.
- `tags` (str): `story`, `comment`, `show_hn`, `ask_hn`, `poll`, or `front_page` (default `story`).
- `sort_by` (str): `relevance` or `date` (default `relevance`).
- `max_results_per_query` (int): 1–50 (default 10).
- `numeric_filters` (str, optional): Algolia numeric filter expression (e.g. `points>100`).

### Output Schema
A list of `HackerNewsItem` items, each with `object_id`, `title`, `url`, `hn_url`, `author`, `points`, `num_comments`, `created_at`, `story_text`, `comment_text`, `parent_id`, `story_id`.

`failures` lists each unsuccessful query and its error. Successful queries retain their results when another query fails. HTTP errors, transport errors, and malformed responses are reported here, so an outage or invalid filter does not look like a successful search with no matches. A successful empty search returns both `results=[]` and `failures=[]`.

## Usage

```python
from tool.hackernews_search import HackerNewsSearchTool, HackerNewsSearchToolInputSchema

tool = HackerNewsSearchTool()
output = tool.run(HackerNewsSearchToolInputSchema(
    queries=["MCP server", "atomic agents"],
    sort_by="date",
    max_results_per_query=5,
    numeric_filters="points>10",
))

for item in output.results:
    print(item.title, "-", item.points, "points")
    print(item.hn_url)
```

## Contributing
PRs welcome — see the main repo `CONTRIBUTING.md`.

## License
Same as the main Atomic Agents project.
