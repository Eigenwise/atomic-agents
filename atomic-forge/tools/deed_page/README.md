# deed.page Publisher Tool

## Overview
Publishes static files (usually a single `index.html`) to a public HTTPS URL through the [deed.page](https://deed.page) API and returns the URL. Use it when an agent's output should be shared as a link: a report, a summary page, a small demo site. No account or API key is needed to start. Published pages are public, so don't publish secrets or private data.

## Prerequisites and Dependencies
- Python 3.12 or later
- `atomic-agents`
- `pydantic`
- `aiohttp`

## Installation
1. Use the Atomic Assembler CLI: run `atomic` and pick `deed_page`.
2. Or copy the `tool/` folder directly into your project.

## Configuration
- `api_token` (str): optional owner token. Falls back to the `DEEDPAGE_TOKEN` environment variable when empty. Without one, the first publish creates a token and the tool keeps it on `tool.token` for later updates and deletes.
- `base_url` (str): API base URL (default `https://deed.page`).
- `user_agent` (str): user agent sent with each request.
- `timeout` (float): HTTP timeout in seconds (default 30).

The token is never part of the tool output, so it doesn't end up in the model's context. If you want to update or delete pages in a later session, save `tool.token` somewhere private (for example as `DEEDPAGE_TOKEN`).

## Input & Output Structure

### Input Schema
- `files` (dict[str, str]): file path to content, e.g. `{"index.html": "<!doctype html>..."}`.
- `slug` (str, optional): subdomain to publish to (lowercase letters, digits, hyphens). Empty gives a random one.
- `ttl_seconds` (int, optional): remove the page after this many seconds (minimum 60). Empty keeps it.

### Output Schema
- `url` (str): public URL of the page.
- `slug` (str): the site's slug.
- `version` (str, optional): version of this publish, e.g. `v1`.
- `expires_at` (str, optional): expiry time when a TTL was set.

## Usage

```python
from tool.deed_page import DeedPageTool, DeedPageToolInputSchema

tool = DeedPageTool()

output = tool.run(DeedPageToolInputSchema(
    files={"index.html": "<!doctype html><title>Report</title><h1>Weekly report</h1>"},
    ttl_seconds=3600,
))
print(output.url)

# Delete it again with the token the tool kept (this method is not exposed to the agent).
tool.delete(output.slug)
```

To let an agent publish what it writes, register the tool like any other Forge tool and have the agent produce `DeedPageToolInputSchema`. Running `python tool/deed_page.py` publishes a sample page with a 10-minute TTL, prints its URL and status, then deletes it.

The API is documented at [deed.page/docs](https://deed.page/docs).

## Contributing
PRs welcome. See the main repo `CONTRIBUTING.md`.

## License
Same as the main Atomic Agents project.
