from unittest.mock import MagicMock, patch

import pytest

from tool.weather import (
    WeatherTool,
    WeatherToolConfig,
    WeatherToolInputSchema,
    WeatherToolOutputSchema,
    describe_weather_code,
)


@pytest.fixture
def tool():
    return WeatherTool(config=WeatherToolConfig())


# ----------- helpers -----------


def test_describe_weather_code_known():
    assert describe_weather_code(0) == "Clear sky"
    assert describe_weather_code(95) == "Thunderstorm"


def test_describe_weather_code_unknown_falls_back():
    assert describe_weather_code(7777) == "Code 7777"


def test_describe_weather_code_none():
    assert describe_weather_code(None) is None


def test_parse_latlon_valid():
    assert WeatherTool.parse_latlon("48.85, 2.35") == (48.85, 2.35)
    assert WeatherTool.parse_latlon("-33.8688,151.2093") == (-33.8688, 151.2093)


def test_parse_latlon_rejects_non_numeric():
    assert WeatherTool.parse_latlon("Brussels") is None


def test_parse_latlon_rejects_out_of_range():
    assert WeatherTool.parse_latlon("100,200") is None


# ----------- run() with mocked HTTP -----------


SAMPLE_GEOCODE = {
    "name": "Brussels",
    "admin1": "Brussels Capital",
    "country": "Belgium",
    "latitude": 50.8505,
    "longitude": 4.3488,
}

SAMPLE_FORECAST = {
    "timezone": "Europe/Brussels",
    "current": {
        "time": "2026-04-29T15:00",
        "temperature_2m": 14.5,
        "apparent_temperature": 13.0,
        "relative_humidity_2m": 60,
        "wind_speed_10m": 11.3,
        "wind_direction_10m": 220,
        "precipitation": 0.0,
        "weather_code": 3,
        "is_day": 1,
    },
    "daily": {
        "time": ["2026-04-29", "2026-04-30", "2026-05-01"],
        "temperature_2m_max": [16.0, 17.5, 19.0],
        "temperature_2m_min": [9.0, 10.0, 11.5],
        "apparent_temperature_max": [15.0, 16.0, 18.0],
        "apparent_temperature_min": [8.0, 9.0, 10.0],
        "precipitation_sum": [1.2, 0.0, 3.5],
        "precipitation_probability_max": [60, 10, 80],
        "weather_code": [3, 2, 65],
        "sunrise": ["2026-04-29T06:30", "2026-04-30T06:28", "2026-05-01T06:26"],
        "sunset": ["2026-04-29T20:30", "2026-04-30T20:32", "2026-05-01T20:34"],
        "wind_speed_10m_max": [18.0, 12.0, 22.0],
    },
    "hourly": {
        "time": ["2026-04-29T15:00", "2026-04-29T16:00"],
        "temperature_2m": [14.5, 14.0],
        "precipitation_probability": [40, 50],
        "weather_code": [3, 61],
    },
}


def _mock_response(payload: dict):
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json = MagicMock(return_value=payload)
    return response


def test_run_resolves_city_and_returns_full_payload(tool):
    geocode_response = _mock_response({"results": [SAMPLE_GEOCODE]})
    forecast_response = _mock_response(SAMPLE_FORECAST)
    with patch.object(tool._session, "get", side_effect=[geocode_response, forecast_response]) as mock_get:
        out = tool.run(WeatherToolInputSchema(location="Brussels", units="metric", forecast_days=3))

    assert isinstance(out, WeatherToolOutputSchema)
    assert out.error is None
    assert out.location_name == "Brussels, Brussels Capital"
    assert out.country == "Belgium"
    assert out.timezone == "Europe/Brussels"
    assert out.units["temperature"] == "°C"
    assert out.current.temperature == 14.5
    assert out.current.weather_description == "Overcast"
    assert len(out.daily) == 3
    assert out.daily[0].weather_description == "Overcast"
    assert out.daily[2].weather_description == "Heavy rain"
    # hourly only when requested
    assert out.hourly == []

    # Verify the forecast call asked for the right things
    forecast_call = mock_get.call_args_list[1]
    params = forecast_call.kwargs["params"]
    assert params["temperature_unit"] == "celsius"
    assert params["wind_speed_unit"] == "kmh"
    assert params["forecast_days"] == "3"


def test_run_includes_hourly_when_requested(tool):
    with patch.object(
        tool._session,
        "get",
        side_effect=[_mock_response({"results": [SAMPLE_GEOCODE]}), _mock_response(SAMPLE_FORECAST)],
    ):
        out = tool.run(WeatherToolInputSchema(location="Brussels", include_hourly=True))
    assert len(out.hourly) == 2
    assert out.hourly[1].weather_description == "Slight rain"


def test_run_imperial_units(tool):
    with patch.object(
        tool._session,
        "get",
        side_effect=[_mock_response({"results": [SAMPLE_GEOCODE]}), _mock_response(SAMPLE_FORECAST)],
    ) as mock_get:
        out = tool.run(WeatherToolInputSchema(location="Brussels", units="imperial"))
    params = mock_get.call_args_list[1].kwargs["params"]
    assert params["temperature_unit"] == "fahrenheit"
    assert params["wind_speed_unit"] == "mph"
    assert params["precipitation_unit"] == "inch"
    assert out.units["temperature"] == "°F"


def test_run_with_lat_lon_skips_geocoding(tool):
    with patch.object(tool._session, "get", side_effect=[_mock_response(SAMPLE_FORECAST)]) as mock_get:
        out = tool.run(WeatherToolInputSchema(location="50.85,4.35"))
    # Only one HTTP call (the forecast) — no geocoding
    assert mock_get.call_count == 1
    assert out.latitude == 50.85
    assert out.longitude == 4.35
    assert out.country is None  # no country lookup performed


@pytest.mark.parametrize("location, expected", [("+48.85,+2.35", (48.85, 2.35)), (".5,-.25", (0.5, -0.25)), ("9e1,-1.8e2", (90.0, -180.0))])
def test_run_accepts_decimal_coordinate_notation(tool, location, expected):
    with patch.object(tool._session, "get", return_value=_mock_response(SAMPLE_FORECAST)) as mock_get:
        out = tool.run(WeatherToolInputSchema(location=location))
    assert out.error is None
    assert (out.latitude, out.longitude) == expected
    assert mock_get.call_count == 1
    assert mock_get.call_args.args[0] == tool.forecast_url


@pytest.mark.parametrize("location", ["91,0", "0,-181", "1e999,0"])
def test_invalid_numeric_coordinate_never_falls_back_to_city_search(tool, location):
    with patch.object(tool._session, "get") as mock_get:
        out = tool.run(WeatherToolInputSchema(location=location))
    assert out.error is not None
    assert "Latitude must be between" in out.error
    mock_get.assert_not_called()


def test_run_unknown_location_returns_error(tool):
    with patch.object(tool._session, "get", side_effect=[_mock_response({"results": []})]):
        out = tool.run(WeatherToolInputSchema(location="Nowheresville XYZ"))
    assert out.error is not None
    assert "not found" in out.error.lower()


def test_run_http_error_returns_error(tool):
    bad_response = MagicMock()
    bad_response.raise_for_status = MagicMock(side_effect=Exception("Server is sad"))
    with patch.object(tool._session, "get", return_value=bad_response):
        out = tool.run(WeatherToolInputSchema(location="Brussels"))
    assert out.error is not None
    assert "sad" in out.error.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
