import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tool.fxmacrodata import (  # noqa: E402
    FXMacroDataTool,
    FXMacroDataToolConfig,
    FXMacroDataToolInputSchema,
    FXMacroDataToolOutputSchema,
)


ANNOUNCEMENTS_PAYLOAD = {
    "currency": "USD",
    "indicator": "inflation",
    "name": "Inflation (CPI)",
    "source": "BLS",
    "data": [
        {"date": "2026-08-31", "val": 3.4, "announcement_datetime": 1789129800},
        {"date": "2026-07-31", "val": 3.4, "announcement_datetime": 1786537800},
    ],
    "pagination": {"limit": 2, "offset": 0, "has_more": True, "next_offset": 2, "total_count": 3},
}

CALENDAR_PAYLOAD = {
    "currency": "USD",
    "timezone": "America/New_York",
    "data": [
        {
            "release": "trade_balance",
            "name": "Trade Balance",
            "announcement_datetime": 1791289800,
            "announcement_datetime_utc": "2026-10-06T12:30:00+00:00",
        }
    ],
}

CATALOGUE_PAYLOAD = {
    "gdp": {"name": "GDP", "unit": "USD bn", "frequency": "Quarterly"},
    "inflation": {"name": "Inflation (CPI)", "unit": "%", "frequency": "Monthly"},
}


def _response(status: int, payload, reason: str = "OK") -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.reason = reason
    response.json.return_value = payload
    return response


@pytest.fixture
def tool(monkeypatch):
    monkeypatch.delenv("FXMACRODATA_API_KEY", raising=False)
    return FXMacroDataTool(FXMacroDataToolConfig())


@pytest.mark.parametrize(
    "fields, path, query",
    [
        (
            {"endpoint": "announcements", "currency": "USD", "indicator": "inflation", "limit": 5},
            "/announcements/usd/inflation",
            {"limit": "5"},
        ),
        ({"endpoint": "latest", "currency": "eur"}, "/announcements/eur/latest", {}),
        (
            {"endpoint": "calendar", "currency": "gbp", "indicator": "policy_rate"},
            "/calendar/gbp",
            {"indicator": "policy_rate"},
        ),
        ({"endpoint": "catalogue", "currency": "jpy"}, "/data_catalogue/jpy", {}),
        (
            {"endpoint": "forex", "currency": "eur", "quote": "USD", "start_date": "2026-01-01", "end_date": "2026-02-01"},
            "/forex/eur/usd",
            {"start_date": "2026-01-01", "end_date": "2026-02-01"},
        ),
        ({"endpoint": "cot", "currency": "aud", "offset": 20}, "/cot/aud", {"offset": "20"}),
        ({"endpoint": "commodities", "indicator": "gold"}, "/commodities/gold", {}),
    ],
)
def test_build_request(fields, path, query):
    assert FXMacroDataTool.build_request(FXMacroDataToolInputSchema(**fields)) == (path, query)


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"endpoint": "announcements", "indicator": "inflation"}, "'currency'"),
        ({"endpoint": "announcements", "currency": "usd"}, "'indicator'"),
        ({"endpoint": "forex", "currency": "eur"}, "'quote'"),
        ({"endpoint": "latest", "currency": "dollars"}, "'currency'"),
        ({"endpoint": "announcements", "currency": "usd", "indicator": "../ping"}, "'indicator'"),
        ({"endpoint": "calendar", "currency": "usd", "start_date": "01/02/2026"}, "'start_date'"),
        ({"endpoint": "commodities"}, "'indicator'"),
    ],
)
def test_invalid_input_returns_error_without_request(tool, fields, message):
    with patch("tool.fxmacrodata.requests.get") as mock_get:
        output = tool.run(FXMacroDataToolInputSchema(**fields))
    mock_get.assert_not_called()
    assert output.error and message in output.error
    assert output.data == []


def test_announcements_split_rows_pagination_and_metadata(tool):
    with patch("tool.fxmacrodata.requests.get", return_value=_response(200, ANNOUNCEMENTS_PAYLOAD)) as mock_get:
        output = tool.run(FXMacroDataToolInputSchema(endpoint="announcements", currency="usd", indicator="inflation", limit=2))

    assert isinstance(output, FXMacroDataToolOutputSchema)
    assert output.error is None
    assert [row["val"] for row in output.data] == [3.4, 3.4]
    assert output.data[0]["announcement_datetime"] == 1789129800
    assert output.pagination["next_offset"] == 2
    assert output.metadata == {"currency": "USD", "indicator": "inflation", "name": "Inflation (CPI)", "source": "BLS"}
    assert output.url == "https://api.fxmacrodata.com/v1/announcements/usd/inflation?limit=2"

    args, kwargs = mock_get.call_args
    assert args[0] == "https://api.fxmacrodata.com/v1/announcements/usd/inflation"
    assert kwargs["params"] == {"limit": "2"}
    assert kwargs["timeout"] == 30.0
    assert "X-API-Key" not in kwargs["headers"]


def test_calendar_without_pagination(tool):
    with patch("tool.fxmacrodata.requests.get", return_value=_response(200, CALENDAR_PAYLOAD)):
        output = tool.run(FXMacroDataToolInputSchema(endpoint="calendar", currency="usd"))

    assert output.data[0]["release"] == "trade_balance"
    assert output.pagination is None
    assert output.metadata == {"currency": "USD", "timezone": "America/New_York"}


def test_catalogue_returns_mapping(tool):
    with patch("tool.fxmacrodata.requests.get", return_value=_response(200, CATALOGUE_PAYLOAD)):
        output = tool.run(FXMacroDataToolInputSchema(endpoint="catalogue", currency="usd"))

    assert output.data == CATALOGUE_PAYLOAD
    assert output.metadata == {}


def test_api_key_from_config_is_sent_as_header():
    tool = FXMacroDataTool(FXMacroDataToolConfig(api_key="test-key", base_url="https://example.test/v1/", timeout=5))
    with patch("tool.fxmacrodata.requests.get", return_value=_response(200, CALENDAR_PAYLOAD)) as mock_get:
        output = tool.run(FXMacroDataToolInputSchema(endpoint="calendar", currency="eur"))

    args, kwargs = mock_get.call_args
    assert args[0] == "https://example.test/v1/calendar/eur"
    assert kwargs["headers"]["X-API-Key"] == "test-key"
    assert kwargs["timeout"] == 5
    assert "test-key" not in output.url


def test_api_key_falls_back_to_environment(monkeypatch):
    monkeypatch.setenv("FXMACRODATA_API_KEY", "env-key")
    tool = FXMacroDataTool()
    with patch("tool.fxmacrodata.requests.get", return_value=_response(200, CALENDAR_PAYLOAD)) as mock_get:
        tool.run(FXMacroDataToolInputSchema(endpoint="calendar", currency="eur"))
    assert mock_get.call_args[1]["headers"]["X-API-Key"] == "env-key"


def test_http_error_surfaces_api_detail(tool):
    body = {"detail": "This endpoint requires an Individual or Business API key.", "code": "api_key_required"}
    with patch("tool.fxmacrodata.requests.get", return_value=_response(401, body, "Unauthorized")):
        output = tool.run(FXMacroDataToolInputSchema(endpoint="forex", currency="eur", quote="usd"))

    assert output.error == "FXMacroData returned 401: This endpoint requires an Individual or Business API key."
    assert output.url == "https://api.fxmacrodata.com/v1/forex/eur/usd"
    assert output.data == []


def test_http_error_without_json_body(tool):
    response = _response(502, None, "Bad Gateway")
    response.json.side_effect = ValueError("not json")
    with patch("tool.fxmacrodata.requests.get", return_value=response):
        output = tool.run(FXMacroDataToolInputSchema(endpoint="latest", currency="usd"))
    assert output.error == "FXMacroData returned 502 Bad Gateway"


def test_network_error_is_reported(tool):
    with patch("tool.fxmacrodata.requests.get", side_effect=requests.ConnectionError("connection refused")):
        output = tool.run(FXMacroDataToolInputSchema(endpoint="latest", currency="usd"))
    assert "connection refused" in output.error


@pytest.mark.asyncio
async def test_run_async(tool):
    with patch("tool.fxmacrodata.requests.get", return_value=_response(200, ANNOUNCEMENTS_PAYLOAD)):
        output = await tool.run_async(
            FXMacroDataToolInputSchema(endpoint="announcements", currency="usd", indicator="inflation")
        )
    assert len(output.data) == 2
