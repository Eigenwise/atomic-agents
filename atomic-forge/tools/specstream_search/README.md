# SpecStream Search Tool

## Overview
Searches the web or recent news through the [SpecStream](https://specstream.ai) Search API (`POST /v1/web-search`) and returns each result's title, URL, site, snippet, and publication date when known, along with a sourced answer on the default tier. With `include_passages=True` on web searches, each result also carries the text excerpt of its page closest to the query, so an agent does not need a separate fetch step. Requires a SpecStream API key.

## Prerequisites and Dependencies
- Python 3.12 or later
- `atomic-agents`
- `pydantic`
- `aiohttp`
- A SpecStream API key from [specstream.ai/workspace](https://specstream.ai/workspace), set as `SPECSTREAM_API_KEY` or passed in the config

## Installation
1. Use the Atomic Assembler CLI: run `atomic` and pick `specstream_search`.
2. Or copy the `tool/` folder directly into your project.

## Configuration
- `api_key` (`SecretStr`, a plain string is accepted): SpecStream API key. Falls back to the `SPECSTREAM_API_KEY` environment variable when empty. It is only sent in the `Authorization` header. Raised errors, log messages, and `failures` redact the configured key (when it is at least 8 characters) and anything in SpecStream's key formats (`ss_live_`, `ss_test_`, `specstream_live_`, `specstream_test_`). In tracebacks that capture local variables, the key does not appear in the tool's own state or in anything derived from a response or a transport error; the inputs you pass in, such as a query, appear there as given, the same as in your own code's frames.
- `base_url` (str): API base URL (default `https://api.specstream.ai/v1`).
- `speed` (str): `default` ($5 per 1,000 searches of up to 10 results) returns a sourced answer when available alongside search results; `fast` ($1 per 1,000 searches of up to 10 results) returns search results and snippets faster without answer extraction.
- `language` (str, optional): two-letter ISO 639-1 language code, e.g. `en` or `de`.
- `country` (str, optional): two-letter ISO 3166-1 country code, e.g. `US` or `DE`.
- `safe` (str, optional): safe search level (`off`, `moderate`, or `strict`).
- `engines` (str, optional): comma-separated engine names or country filters, e.g. `-CN` or `bing,duckduckgo`.
- `pages_wait_ms` (int, optional): with `include_passages=True`, milliseconds to wait for page text (`0` to `3500`).
- `max_content_chars` (int, optional): truncate each result's `content` to this many characters. No limit by default.
- `timeout` (float): HTTP timeout in seconds (default `30`).

## Input & Output Structure

### Input Schema
- `queries` (list[str], at least one): search queries to run (1 to 2,000 characters each). Duplicates are sent once. Supports operators such as `site:`, `-site:`, `-word`, `-"phrase"`, `"phrase"`, `filetype:`, `after:`, and `before:`.
- `topic` (str): `web` or `news`. Default `web`.
- `max_results_per_query` (int): 1-100 (default `10`).
- `include_passages` (bool): when `True` on `web` searches, include page text excerpts closest to the query in `content`. Cannot be used when `topic` is `news`. Default `False`.

### Output Schema
- `results`: a list of `SpecstreamSearchResultItem` items. Each has `query`, `title`, `url`, and optional `description` (the snippet), `site`, `published` (`YYYY-MM-DD`), `source` (news publisher), `cited` (whether the extracted answer or rows cited it), and `content` (page excerpt when `include_passages=True`). Results without a well-formed http(s) URL (a valid host name, IPv4 or IPv6 address, a valid port, no whitespace or control characters; Unicode host names are checked in their IDNA 2003 encoded form) are skipped, and badly typed optional fields are left empty.
- `answers`: a list of `SpecstreamSearchAnswer` items (`query`, `value`, `page`, optional `seen_at`) when `speed="default"` extracts a direct sourced answer.
- `failures`: queries that failed while others returned results, each with `query` and `error` (the HTTP status and the API's error code/message; `Retry-After` seconds on `503`/`429`). Keys are redacted from both as described for `api_key`. They are also logged as warnings.

## Usage
If no query returns results and at least one failed (for example a wrong API key, insufficient balance, or a malformed response), `run` raises the first error instead of returning an empty list. Redirects are not followed; they are reported as errors.

```python
from tool.specstream_search import (
    SpecstreamSearchTool,
    SpecstreamSearchToolConfig,
    SpecstreamSearchToolInputSchema,
)

tool = SpecstreamSearchTool(config=SpecstreamSearchToolConfig(api_key="your-specstream-api-key"))

output = tool.run(
    SpecstreamSearchToolInputSchema(
        queries=["retrieval augmented generation"],
        max_results_per_query=5,
        include_passages=True,
    )
)

for answer in output.answers:
    print(f"Answer: {answer.value} ({answer.page})")

for item in output.results:
    print(item.title, "-", item.url)
    print(item.content[:500] if item.content else item.description)
```

The request and response formats are documented at [specstream.ai/docs/search-api](https://specstream.ai/docs/search-api).

## Contributing
PRs welcome. See the main repo `CONTRIBUTING.md`.

## License
Same as the main Atomic Agents project.
