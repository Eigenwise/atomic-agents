# FXMacroData Tool

## Overview
Queries the [FXMacroData](https://fxmacrodata.com/?utm_source=github&utm_medium=referral&utm_campaign=atomic-agents&utm_content=readme) API for official-source macroeconomic data: release history with the official announcement timestamp of each print (CPI, GDP, unemployment, payrolls, policy rates, bond yields and more), the latest value of every indicator for a currency, upcoming release dates, the indicator catalogue, FX spot history, CFTC Commitments of Traders positioning, and commodity prices.

USD releases, the USD release calendar and the indicator catalogue work without an API key. Other currencies, FX rates, COT and commodities need a key from [fxmacrodata.com/subscribe](https://fxmacrodata.com/subscribe?utm_source=github&utm_medium=referral&utm_campaign=atomic-agents&utm_content=readme).

## Prerequisites and Dependencies
- Python 3.12 or later
- `atomic-agents`
- `pydantic`
- `requests`
- Optional: an FXMacroData API key

## Installation
1. Use the Atomic Assembler CLI: run `atomic` and pick `fxmacrodata`.
2. Or copy the `tool/` folder directly into your project.

## Configuration
- `api_key` (str): FXMacroData API key, sent in the `X-API-Key` header. Falls back to the `FXMACRODATA_API_KEY` environment variable. Leave empty for keyless USD access.
- `base_url` (str): API base URL (default `https://api.fxmacrodata.com/v1`).
- `timeout` (float): HTTP timeout in seconds (default 30).

## Input & Output Structure

### Input Schema
- `endpoint` (str): one of
  - `announcements`: release history for one indicator (`currency` and `indicator` required)
  - `latest`: the most recent value of every indicator for a currency
  - `calendar`: upcoming release dates for a currency, optionally filtered by `indicator`
  - `catalogue`: indicator slugs, units and coverage available for a currency
  - `forex`: FX spot history (`currency` is the base, `quote` required)
  - `cot`: CFTC positioning for a currency
  - `commodities`: price history for a commodity (`indicator`, e.g. `gold`)
- `currency` (str, optional): three-letter code such as `usd`, `eur`, `jpy`.
- `indicator` (str, optional): indicator slug such as `inflation`, `policy_rate`, `gdp`.
- `quote` (str, optional): quote currency for `forex`.
- `start_date`, `end_date` (str, optional): `YYYY-MM-DD`.
- `limit` (int, optional): rows per page, 1-100. History endpoints return 20 rows by default.
- `offset` (int, optional): row offset for paging.

### Output Schema
- `endpoint` (str): the endpoint that was called.
- `url` (str): the request URL (the key is never included).
- `data`: the rows returned by the API, with the API's own field names. Announcement rows carry `date` (reference period), `val`, and `announcement_datetime` (Unix seconds of the official release). For `catalogue` this is a mapping of indicator slug to metadata.
- `pagination` (dict, optional): `has_more`, `next_offset` and `total_count` for paged endpoints.
- `metadata` (dict): the remaining top-level response fields, such as source, provenance and data quality.
- `error` (str, optional): set instead of raising when the input is incomplete or the request fails, so an agent can read the reason (for example a missing API key for a non-USD currency) and adjust.

## Usage

```python
from tool.fxmacrodata import FXMacroDataTool, FXMacroDataToolInputSchema

tool = FXMacroDataTool()  # reads FXMACRODATA_API_KEY if set

cpi = tool.run(FXMacroDataToolInputSchema(endpoint="announcements", currency="usd", indicator="inflation", limit=3))
for row in cpi.data:
    print(row["date"], row["val"], row["announcement_datetime"])

upcoming = tool.run(FXMacroDataToolInputSchema(endpoint="calendar", currency="usd"))
for row in upcoming.data[:5]:
    print(row["announcement_datetime_utc"], row["name"])
```

To page through a long history, repeat the call with `offset=output.pagination["next_offset"]` until `has_more` is false.

The full list of endpoints and response fields is in the [API reference](https://fxmacrodata.com/documentation/reference?utm_source=github&utm_medium=referral&utm_campaign=atomic-agents&utm_content=readme).

## Contributing
PRs welcome. See the main repo `CONTRIBUTING.md`.

## License
Same as the main Atomic Agents project.
