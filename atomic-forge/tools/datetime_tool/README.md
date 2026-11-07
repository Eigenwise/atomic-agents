# DateTime Tool

## Overview
A timezone-aware date and time tool. Use it to:

- get the current time in any IANA timezone (`now`)
- parse an ISO 8601 datetime string into structured fields (`parse`)
- convert a datetime from one timezone to another (`convert`)
- shift a datetime by days/hours/minutes/seconds (`shift`)
- compute the difference between two datetimes (`diff`)

It uses the standard library only (`datetime`, `zoneinfo`) plus `tzdata` so the tool also works on Windows where the OS does not ship the IANA timezone database.

## Prerequisites and Dependencies
- Python 3.12 or later
- `atomic-agents`
- `pydantic`
- `tzdata`

## Installation
1. Using the Atomic Assembler CLI: run `atomic` and select `datetime_tool`.
2. Or copy the `tool/` folder directly into your project.

## Input & Output Structure

### Input Schema
- `operation` (Literal): one of `now`, `parse`, `convert`, `shift`, `diff`.
- `timezone` (str): IANA timezone name. Defaults to `UTC`. For `convert` this is the source timezone of `input`.
- `target_timezone` (str, optional): required for `convert`.
- `input` (str, optional): ISO 8601 datetime string for non-`now` operations.
- `input_2` (str, optional): second datetime, required for `diff` (computes `input_2 - input`).
- `days` (int): calendar days for `shift`, applied in the input timezone first (negative values subtract).
- `hours`, `minutes`, `seconds` (int): elapsed time for `shift`, applied after calendar days (negative values subtract).

Across daylight-saving transitions, `days=1` preserves the local clock time, while
`hours=24` or `seconds=86400` advances by exactly 24 elapsed hours. For example,
from `2026-03-08T00:00:00` in `America/New_York`, `days=1` gives the following
midnight, but `hours=24` gives `2026-03-09T01:00:00-04:00`. Calendar results in a
spring-forward gap are normalized through UTC (for example, 02:30 becomes 03:30).
An explicit UTC offset in `input` remains authoritative; it is a fixed offset,
not an IANA timezone with daylight-saving rules.

### Output Schema
- `operation` (str): operation that was performed.
- `iso`, `timezone`, `unix_timestamp`, `year`, `month`, `day`, `hour`, `minute`, `second`, `weekday`, `human`: populated for `now`/`parse`/`convert`/`shift`.
- `diff_seconds`, `diff_human`: populated for `diff`.
- `error` (str, optional): set when something went wrong.

## Usage

```python
from tool.datetime_tool import DateTimeTool, DateTimeToolInputSchema

tool = DateTimeTool()

# Current time in Tokyo
print(tool.run(DateTimeToolInputSchema(operation="now", timezone="Asia/Tokyo")))

# Convert a Brussels meeting time to New York
print(tool.run(DateTimeToolInputSchema(
    operation="convert",
    input="2026-04-29T15:30:00",
    timezone="Europe/Brussels",
    target_timezone="America/New_York",
)))
```

## Contributing
PRs welcome — see the main repo `CONTRIBUTING.md`.

## License
Same as the main Atomic Agents project.
