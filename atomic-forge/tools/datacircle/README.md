# Datacircle Tool

## Overview
Finds the best GTM API plans on [Datacircle](https://datacircle.dev). Datacircle benchmarks GTM APIs (email finders, LinkedIn profiles, company data, phone finders, SERP, web scraping and more) and scores each plan out of 100. The tool lists every API, gives the best plans of one, searches every plan and shows one provider's rank on each API.

It calls Datacircle's open API at `https://api.datacircle.dev/v1`, with no API key and no account.

## Prerequisites and Dependencies
- Python 3.12 or later
- `atomic-agents`
- `pydantic`
- `requests`

## Installation
1. Use the Atomic Assembler CLI: run `atomic` and pick `datacircle`.
2. Or copy the `tool/` folder directly into your project.

## Configuration
- `base_url` (str): API base URL (default `https://api.datacircle.dev/v1`). Must be an `https://` URL with a host and a valid port, checked when the config is created. Redirects are not followed.
- `timeout` (float): HTTP timeout in seconds (default 30).
- `plan_fields` (list[str] or None): the fields kept on each plan, in this order. The default keeps `rank`, `provider`, `plan`, `api`, `score`, `perThousand`, `notFoundPerThousand`, `billing`, `pay`, `currency`, `strengths` and `page`. `None` keeps every field of the API's answer. Each field is described at [datacircle.dev/fields](https://datacircle.dev/fields).

## Input & Output Structure

### Input Schema
- `call` (str): one of these, each one call of the API:

  | `call` | API call | Gives |
  |---|---|---|
  | `list_apis` | `GET /apis` | every API, with its slug, its number of plans and providers and its best plan |
  | `best_plans` | `GET /apis/{slug}/best` | one API's plans best first by score (`api` required) |
  | `search_plans` | `GET /plans` | the plans of every API or of one `api`, whose provider or plan name holds every word of `query` |
  | `provider` | `GET /providers/{slug}` | one provider's rank and plans on each API it sells (`provider` required) |

- `api` (str, optional): the API's slug as `list_apis` gives it, such as `email-finder`. For `search_plans`, its slug or its name (`Employee email`).
- `provider` (str, optional): the provider's slug, its name in lowercase letters and digits joined by hyphens, such as `bright-data`.
- `query` (str, optional): words that must all appear in the provider's or plan's name, for `search_plans`.
- `sort` (str, optional): for `search_plans`, `score` (best first, the default), `provider` (A to Z) or `price` (lowest $ per 1,000 first).
- `limit` (int, optional): how many plans to give, 1 to 50. The API gives 10 when left out. For `provider`, how many of each API's plans to keep, best first (every plan when left out).
- `offset` (int, optional): how many found plans to skip first, for `search_plans`.

A call ignores the fields it does not take. Slugs are checked before any request, and a blank string fails validation: leave the field out instead.

### Output Schema
- `call` (str): the call that was made.
- `url` (str): the request URL.
- `data`: the API's answer, with the API's own field names.
  - `list_apis`: a list of APIs, each with `api`, `slug`, `plans`, `providers`, `best`, `sheetPage` and `bestPage`.
  - `best_plans` and `search_plans`: an object whose `plans` are the plans found and `total` how many there are.
  - `provider`: an object whose `apis` give the provider's `rank` and `plans` on each API.
  - Each plan has its `rank` on its API, its `score` out of 100, `perThousand` (what 1,000 results cost, in dollars) and its `page` on datacircle.dev.
- `error` (str, optional): set instead of raising when the input is invalid, the request fails or the answer is not the documented shape. An unknown API slug gives every valid slug, so an agent can correct itself.

## Usage

```python
from tool.datacircle import DatacircleTool, DatacircleToolInputSchema

tool = DatacircleTool()

best = tool.run(DatacircleToolInputSchema(call="best_plans", api="email-finder", limit=3))
for plan in best.data["plans"]:
    print(plan["rank"], plan["provider"], plan["plan"], plan["score"], plan["perThousand"])

cheapest = tool.run(DatacircleToolInputSchema(call="search_plans", api="serp", sort="price", limit=5))
for plan in cheapest.data["plans"]:
    print(plan["provider"], plan["plan"], plan["perThousand"])
```

To page through a search, repeat the call with `offset` raised by `limit` until it reaches `total`.

The API allows 60 calls a minute from one address. Its spec is at [api.datacircle.dev/v1/openapi.json](https://api.datacircle.dev/v1/openapi.json).

## Contributing
PRs welcome. See the main repo `CONTRIBUTING.md`.

## License
Same as the main Atomic Agents project.
