"""A missing current reading must not discard an otherwise usable forecast."""

from contextlib import contextmanager
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from threading import Thread
from urllib.parse import parse_qs, urlsplit

import pytest

from tool.weather import WeatherTool, WeatherToolConfig, WeatherToolInputSchema


FORECAST = {
    "timezone": "Europe/Paris",
    "current": {"time": "2026-01-01T12:00", "temperature_2m": 0.0, "relative_humidity_2m": 73, "weather_code": 0},
    "daily": {"time": ["2026-01-01"], "temperature_2m_max": [12.5]},
    "hourly": {"time": ["2026-01-01T13:00"], "temperature_2m": [11.8]},
}


@contextmanager
def forecast_server(payload):
    requests = []
    body = json.dumps(payload).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/forecast", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        assert not thread.is_alive()


@pytest.mark.parametrize("missing", ["null", "omitted"])
@pytest.mark.parametrize("units", ["metric", "imperial"])
def test_missing_temperature_preserves_forecast_over_http(missing, units):
    payload = deepcopy(FORECAST)
    if missing == "null":
        payload["current"]["temperature_2m"] = None
    else:
        del payload["current"]["temperature_2m"]
    with forecast_server(payload) as (url, requests):
        tool = WeatherTool(WeatherToolConfig(forecast_url=url))
        tool._session.trust_env = False
        try:
            result = tool.run(WeatherToolInputSchema(location="48.8566,2.3522", units=units, include_hourly=True))
        finally:
            tool._session.close()

    assert result.error is None
    assert (result.latitude, result.longitude) == (48.8566, 2.3522)
    assert result.timezone == "Europe/Paris"
    assert result.units["temperature"] == ("°C" if units == "metric" else "°F")
    assert result.current.temperature is None
    assert result.current.humidity == 73
    assert result.current.weather_description == "Clear sky"
    assert result.daily[0].temperature_max == 12.5
    assert result.hourly[0].temperature == 11.8
    assert len(result.daily) == len(result.hourly) == 1
    assert len(requests) == 1
    query = parse_qs(urlsplit(requests[0]).query)
    assert "temperature_2m" in query["current"][0].split(",")
    assert query["temperature_unit"] == ["celsius" if units == "metric" else "fahrenheit"]
    assert json.loads(result.model_dump_json())["current"]["temperature"] is None


def test_zero_temperature_is_a_real_reading():
    current = WeatherTool._build_current(FORECAST)
    assert current.temperature == 0.0
    assert current.humidity == 73


def test_absent_current_section_is_still_optional():
    payload = deepcopy(FORECAST)
    del payload["current"]
    assert WeatherTool._build_current(payload) is None
    assert WeatherTool._build_daily(payload)[0].temperature_max == 12.5


def test_nonnumeric_current_temperature_is_still_invalid():
    payload = deepcopy(FORECAST)
    payload["current"]["temperature_2m"] = "invalid"
    with pytest.raises(ValueError):
        WeatherTool._build_current(payload)
