# Weather Tool

## Overview
Looks up current conditions and a daily/hourly forecast for any location using the free [Open-Meteo](https://open-meteo.com) APIs. Accepts city names (resolved via Open-Meteo's geocoding API) or `lat,lon` pairs. No API key required.

## Prerequisites and Dependencies
- Python 3.12 or later
- `atomic-agents`
- `pydantic`
- `requests`

## Installation
1. Use the Atomic Assembler CLI: run `atomic` and pick `weather`.
2. Or copy the `tool/` folder directly into your project.

## Input & Output Structure

### Input Schema
- `location` (str): city name, `'City, Region'`, or `'lat,lon'`.
- `country_code` (str, optional): two-letter ISO 3166-1 country code for city lookup, such as `GB` or `US`. Case-insensitive; omitted by default. Coordinates bypass geocoding, so the filter does not apply to `lat,lon`.
- `units` (str): `metric` (Celsius / km/h / mm) or `imperial` (Fahrenheit / mph / inches). Default `metric`.
- `forecast_days` (int): 1–16, default 3.
- `include_hourly` (bool): include hourly forecast data, default `False`.
- `language` (str): geocoder language hint (default `en`).

### Output Schema
- `location_name`, `country`, `latitude`, `longitude`, `timezone`, `units` (dict).
- `current` (`WeatherCurrent`): now-cast fields.
- `daily` (list[`WeatherDay`]): forecast buckets.
- `hourly` (list[`WeatherHour`]): only populated when `include_hourly=True`.
- `error` (str, optional): set when the operation failed.

## Usage

```python
from tool.weather import WeatherTool, WeatherToolInputSchema

tool = WeatherTool()
out = tool.run(WeatherToolInputSchema(location="Brussels", units="metric", forecast_days=3))

if out.error:
    print("Error:", out.error)
else:
    print(out.location_name, out.country)
    print("Now:", out.current.temperature, out.units["temperature"], out.current.weather_description)
    for day in out.daily:
        print(day.date, day.temperature_min, "→", day.temperature_max, day.weather_description)
```

For an ambiguous city name, use an explicit country filter:

```python
out = tool.run(WeatherToolInputSchema(location="London", country_code="GB"))
```

The filter uses Open-Meteo's [`countryCode` parameter](https://open-meteo.com/en/docs/geocoding-api). If no city matches in that country, the tool returns a lookup error without fetching a forecast for another country.

## Notes
- Weather codes are translated to human-readable descriptions via the WMO interpretation table (Open-Meteo).
- The tool fails gracefully and returns `error` set on the output rather than raising.

## Contributing
PRs welcome — see the main repo `CONTRIBUTING.md`.

## License
Same as the main Atomic Agents project.
