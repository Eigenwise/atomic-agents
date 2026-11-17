import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import pytest
from pydantic import ValidationError

from tool.weather import WeatherTool, WeatherToolConfig, WeatherToolInputSchema


@pytest.fixture
def service():
    requests = []
    locations = {
        "GB": {"name": "London", "country": "United Kingdom", "latitude": 51.5085, "longitude": -0.1257},
        "US": {"name": "London", "country": "United States", "latitude": 39.8865, "longitude": -83.4483},
    }

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            params = parse_qs(url.query)
            requests.append((url.path, params))
            if url.path == "/geocode":
                location = locations.get(params.get("countryCode", ["US"])[0])
                body = {"results": [location] if location else []}
            else:
                body = {"current": {"time": "2026-01-01T00:00", "temperature_2m": 12.5}}
            encoded = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    tool = WeatherTool(WeatherToolConfig(geocoding_url=origin + "/geocode", forecast_url=origin + "/forecast"))
    try:
        yield tool, requests, locations
    finally:
        tool._session.close()
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize("country_code", ["gb", "GB", "US"])
def test_ambiguous_city_uses_requested_country_coordinates_for_forecast(service, country_code):
    tool, requests, locations = service
    result = tool.run(WeatherToolInputSchema(location="London", country_code=country_code, language="fr"))
    expected = locations[country_code.upper()]

    assert result.error is None
    assert result.country == expected["country"]
    assert (result.latitude, result.longitude) == (expected["latitude"], expected["longitude"])
    assert result.current.temperature == 12.5
    assert requests[0] == (
        "/geocode",
        {"name": ["London"], "count": ["1"], "language": ["fr"], "format": ["json"], "countryCode": [country_code.upper()]},
    )
    assert requests[1][0] == "/forecast"
    assert requests[1][1]["latitude"] == [str(expected["latitude"])]
    assert requests[1][1]["longitude"] == [str(expected["longitude"])]
    assert "countryCode" not in requests[1][1]


def test_omitting_country_keeps_unfiltered_geocoding(service):
    tool, requests, _ = service
    result = tool.run(WeatherToolInputSchema(location="London"))

    assert result.error is None
    assert result.country == "United States"
    assert "countryCode" not in requests[0][1]


def test_country_without_a_matching_city_does_not_fetch_another_countrys_forecast(service):
    tool, requests, _ = service
    result = tool.run(WeatherToolInputSchema(location="London", country_code="JP"))

    assert result.error == "Location 'London' not found."
    assert result.current is None
    assert [path for path, _ in requests] == ["/geocode"]


def test_explicit_coordinates_bypass_country_geocoding(service):
    tool, requests, _ = service
    result = tool.run(WeatherToolInputSchema(location="48.8566,2.3522", country_code="GB"))

    assert result.error is None
    assert (result.latitude, result.longitude) == (48.8566, 2.3522)
    assert result.country is None
    assert [path for path, _ in requests] == ["/forecast"]
    assert "countryCode" not in requests[0][1]


@pytest.mark.parametrize("country_code", ["", "G", "GBR", "G1", " GB", "GB ", "éé", "US,GB"])
def test_malformed_country_filters_are_rejected_before_lookup(country_code):
    with pytest.raises(ValidationError, match="country_code"):
        WeatherToolInputSchema(location="London", country_code=country_code)
