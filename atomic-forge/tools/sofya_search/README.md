# Sofya Search Tool

## Overview
Searches the web or recent news through the [Sofya](https://sofya.co) Search API and returns each result's title, URL, snippet, and publish date when known. With the default `search_depth="basic"` each result also carries the text of its page, so an agent does not need a separate fetch step to read the pages. Requires a Sofya API key.

## Prerequisites and Dependencies
- Python 3.12 or later
- `atomic-agents`
- `pydantic`
- `aiohttp`
- A Sofya API key from [sofya.co](https://sofya.co), set as `SOFYA_API_KEY` or passed in the config

## Installation
1. Use the Atomic Assembler CLI: run `atomic` and pick `sofya_search`.
2. Or copy the `tool/` folder directly into your project.

## Configuration
- `api_key` (`SecretStr`, a plain string is accepted): Sofya API key. Falls back to the `SOFYA_API_KEY` environment variable when empty. It is only sent in the `Authorization` header. Raised errors, log messages and `failures` redact the configured key (when it is at least 8 characters) and anything in Sofya's key formats (`sofya_live_`, `sofya_test_`, `ay_live_`). In tracebacks that capture local variables, the key does not appear in the tool's own state or in anything derived from a response or a transport error; the inputs you pass in, such as a query, appear there as given, the same as in your own code's frames.
- `base_url` (str): API base URL (default `https://sofya.co/v1`).
- `search_depth` (str): `basic` (default) returns each result's page text in `content`; `snippets` returns snippets only and costs fewer credits. See [sofya.co/pricing](https://sofya.co/pricing).
- `freshness` (str, optional): `day`, `week`, `month`, `year`, or a date range `YYYY-MM-DD:YYYY-MM-DD`.
- `include_domains` / `exclude_domains` (list[str], up to 10 each): bare host names such as `github.com` to restrict results to, or leave out.
- `max_content_chars` (int, optional): truncate each result's `content` to this many characters. No limit by default.
- `timeout` (float): HTTP timeout in seconds (default 30).

## Input & Output Structure

### Input Schema
- `queries` (list[str], at least one): search queries to run. Duplicates are sent once. Operators such as `site:`, `"exact phrase"`, and `-term` are supported.
- `topic` (str): `general` or `news`. Default `general`.
- `max_results_per_query` (int): 1-20 (default 5, kept low because each result can carry a full page).

### Output Schema
- `results`: a list of `SofyaSearchResultItem` items. Each has `query`, `title`, `url`, and optional `description` (the snippet), `published` (`YYYY-MM-DD`), and `content` (the page text, with `search_depth="basic"` when the page could be read in time). Results without a well-formed http(s) URL (a valid host name, IPv4 or IPv6 address, a valid port, no whitespace or control characters; Unicode host names are checked in their IDNA 2003 encoded form, as Python's `idna` codec produces it) are skipped, and badly typed optional fields are left empty.
- `failures`: queries that failed while others returned results, each with `query` and `error` (the HTTP status and the API's error message; for 402 the top-up link, for 429 the `Retry-After` seconds). Keys are redacted from both as described for `api_key`. They are also logged as warnings.

## Usage
If no query returns results and at least one failed (for example a wrong API key, no credits, or a malformed response), `run` raises the first error instead of returning an empty list. Redirects are not followed; they are reported as errors.

```python
from tool.sofya_search import SofyaSearchTool, SofyaSearchToolConfig, SofyaSearchToolInputSchema

tool = SofyaSearchTool(config=SofyaSearchToolConfig(api_key="your-sofya-api-key"))

output = tool.run(SofyaSearchToolInputSchema(
    queries=["retrieval augmented generation"],
    max_results_per_query=3,
))

for item in output.results:
    print(item.title, "-", item.url)
    print(item.content[:500] if item.content else item.description)
```

The request and response formats are documented at [sofya.co/docs](https://sofya.co/docs).

## Contributing
PRs welcome. See the main repo `CONTRIBUTING.md`.

## License
Same as the main Atomic Agents project.
