import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests
from pydantic import ValidationError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tool.fxmacrodata import (  # noqa: E402
    FXMacroDataTool,
    FXMacroDataToolConfig,
    FXMacroDataToolInputSchema,
    FXMacroDataToolOutputSchema,
    build_request,
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

SECRET = "fxmd-dummy-secret-0123456789"


def _response(status: int, payload, reason: str = "OK") -> MagicMock:
    response = MagicMock()
    response.status_code = status
    response.reason = reason
    response.json.return_value = payload
    return response


def _real_response(request: requests.PreparedRequest, status: int, body: str, headers=None) -> requests.Response:
    """A genuine requests.Response, so redirect handling and header checks run through Requests itself."""
    response = requests.Response()
    response.status_code = status
    response._content = body.encode()
    response.headers.update(headers or {})
    response.url = request.url
    response.request = request
    response.reason = "Found" if status == 302 else "OK"
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
    assert build_request(FXMacroDataToolInputSchema(**fields)) == (path, query)


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


@pytest.mark.parametrize(
    "dates, message",
    [
        ({"start_date": "2026-02-30"}, "'start_date' must be a real date"),
        ({"end_date": "2026-13-01"}, "'end_date' must be a real date"),
        ({"start_date": ""}, "'start_date' must be a real date"),
        ({"start_date": "20260101"}, "'start_date' must be a real date"),
        ({"start_date": "2026-03-01", "end_date": "2026-02-01"}, "'start_date' must not be after 'end_date'"),
    ],
)
def test_impossible_empty_and_reversed_dates_are_rejected(tool, dates, message):
    with patch("tool.fxmacrodata.requests.get") as mock_get:
        output = tool.run(FXMacroDataToolInputSchema(endpoint="forex", currency="eur", quote="usd", **dates))
    mock_get.assert_not_called()
    assert message in output.error


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
    assert kwargs["allow_redirects"] is False
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
    assert output.data == []


def test_http_error_without_json_body(tool):
    response = _response(502, None, "Bad Gateway")
    response.json.side_effect = ValueError("not json")
    with patch("tool.fxmacrodata.requests.get", return_value=response):
        output = tool.run(FXMacroDataToolInputSchema(endpoint="latest", currency="usd"))
    assert output.error == "FXMacroData returned 502 Bad Gateway"


# Regression: base_url problems used to surface mid-request (InvalidURL/ValueError); they now fail when the
# config is built, so they can never reach a request.
@pytest.mark.parametrize(
    "base_url",
    ["https://", "https:///v1", "http://api.fxmacrodata.com/v1", "api.fxmacrodata.com/v1", "ftp://api.fxmacrodata.com", ""],
)
def test_invalid_base_url_is_rejected_by_config(base_url):
    with pytest.raises(ValidationError, match="base_url must be an https:// URL with a host"):
        FXMacroDataToolConfig(api_key=SECRET, base_url=base_url)


def test_base_url_is_normalised_by_config():
    assert FXMacroDataToolConfig(base_url=" https://example.test/v1/ ").base_url == "https://example.test/v1"


# Regression: the full key appeared in the config's repr() and str().
def test_config_repr_and_str_hide_the_key():
    config = FXMacroDataToolConfig(api_key=SECRET)
    assert SECRET not in repr(config)
    assert SECRET not in str(config)
    assert SECRET not in config.model_dump_json()


# Regression: a successful response echoing the key copied it into metadata and the output repr.
def test_key_echoed_in_a_successful_response_is_redacted():
    payload = {"data": [{"note": f"row {SECRET}"}], "detail": "echo " + SECRET, f"field-{SECRET}": 1}
    tool = FXMacroDataTool(FXMacroDataToolConfig(api_key=SECRET))
    with patch("tool.fxmacrodata.requests.get", return_value=_response(200, payload)):
        output = tool.run(FXMacroDataToolInputSchema(endpoint="latest", currency="usd"))

    assert output.error is None
    assert SECRET not in repr(output)
    assert SECRET not in output.model_dump_json()
    assert output.metadata["detail"] == "echo [redacted]"
    assert output.data == [{"note": "row [redacted]"}]


def test_key_echoed_in_a_catalogue_response_is_redacted():
    payload = {"gdp": {"name": "GDP", "note": SECRET}}
    tool = FXMacroDataTool(FXMacroDataToolConfig(api_key=SECRET))
    with patch("tool.fxmacrodata.requests.get", return_value=_response(200, payload)):
        output = tool.run(FXMacroDataToolInputSchema(endpoint="catalogue", currency="usd"))
    assert output.data == {"gdp": {"name": "GDP", "note": "[redacted]"}}


# Regression: a 302 from the HTTPS API to an unrelated HTTP host must not carry X-API-Key there.
def test_redirect_to_other_http_host_does_not_forward_api_key():
    sent = []

    def send(adapter, request, **kwargs):
        sent.append((request.url, dict(request.headers)))
        if request.url.startswith("https://api.fxmacrodata.com/"):
            return _real_response(request, 302, "", {"Location": "http://collector.example/steal"})
        return _real_response(request, 200, json.dumps(CALENDAR_PAYLOAD))

    tool = FXMacroDataTool(FXMacroDataToolConfig(api_key=SECRET))
    with patch("requests.adapters.HTTPAdapter.send", send):
        output = tool.run(FXMacroDataToolInputSchema(endpoint="calendar", currency="usd"))

    assert [url for url, _ in sent] == ["https://api.fxmacrodata.com/v1/calendar/usd"]
    assert all(not url.startswith("http://collector.example") for url, _ in sent)
    assert output.error == "FXMacroData answered with a redirect (302); redirects are not followed."
    assert output.data == []


# Regression: a key with leading whitespace raised InvalidHeader, whose text carried the full key.
def test_key_with_leading_whitespace_is_stripped_and_never_echoed():
    sent = []

    def send(adapter, request, **kwargs):
        sent.append(dict(request.headers))
        return _real_response(request, 200, json.dumps(CALENDAR_PAYLOAD))

    tool = FXMacroDataTool(FXMacroDataToolConfig(api_key=f"  {SECRET}"))
    with patch("requests.adapters.HTTPAdapter.send", send):
        output = tool.run(FXMacroDataToolInputSchema(endpoint="calendar", currency="usd"))

    assert output.error is None
    assert sent[0]["X-API-Key"] == SECRET


def test_key_that_cannot_be_a_header_is_reported_without_its_value():
    tool = FXMacroDataTool(FXMacroDataToolConfig(api_key=f"{SECRET}\nInjected: 1"))
    with patch("requests.adapters.HTTPAdapter.send") as send:
        output = tool.run(FXMacroDataToolInputSchema(endpoint="calendar", currency="usd"))

    send.assert_not_called()
    assert SECRET not in output.error
    assert "cannot be sent in an HTTP header" in output.error


@pytest.mark.parametrize(
    "error",
    [requests.ConnectionError(f"refused for X-API-Key: {SECRET}"), requests.exceptions.InvalidHeader(f"bad {SECRET}")],
)
def test_request_exception_text_never_reaches_the_output(error):
    tool = FXMacroDataTool(FXMacroDataToolConfig(api_key=SECRET))
    with patch("requests.adapters.HTTPAdapter.send", side_effect=error):
        output = tool.run(FXMacroDataToolInputSchema(endpoint="latest", currency="usd"))

    assert SECRET not in output.error
    assert output.error == f"Could not reach FXMacroData ({type(error).__name__})."


def test_upstream_detail_echoing_the_key_is_redacted():
    tool = FXMacroDataTool(FXMacroDataToolConfig(api_key=SECRET))
    with patch("tool.fxmacrodata.requests.get", return_value=_response(403, {"detail": f"key {SECRET} revoked"})):
        output = tool.run(FXMacroDataToolInputSchema(endpoint="latest", currency="usd"))
    assert output.error == "FXMacroData returned 403: key [redacted] revoked"


# Regression: HTTP 200 bodies that are not the documented shape must come back as an error, not as data=[] or a
# ValidationError.
@pytest.mark.parametrize(
    "endpoint, payload, message",
    [
        ("latest", {"detail": "upstream failure"}, "unexpected response body: upstream failure"),
        ("latest", {"data": None}, "unexpected response body"),
        ("latest", {"data": "not rows"}, "unexpected response body"),
        ("latest", {"data": [1, 2, 3]}, "unexpected response body"),
        ("latest", {"data": [], "pagination": "page 1"}, "unexpected response body"),
        ("latest", ["not", "an", "object"], "unexpected response body"),
        ("catalogue", {"detail": "upstream failure"}, "unexpected response body: upstream failure"),
        ("catalogue", {"gdp": "GDP"}, "unexpected response body"),
        ("catalogue", [], "unexpected response body"),
    ],
)
def test_malformed_200_responses_return_an_error(tool, endpoint, payload, message):
    with patch("tool.fxmacrodata.requests.get", return_value=_response(200, payload)):
        output = tool.run(FXMacroDataToolInputSchema(endpoint=endpoint, currency="usd"))

    assert isinstance(output, FXMacroDataToolOutputSchema)
    assert message in output.error
    assert output.data == []


# Regression: a present but malformed pagination value was normalised away and returned error=None.
@pytest.mark.parametrize(
    "pagination",
    [[], False, 0, "", "page 1", {"has_more": "yes"}, {"total_count": True}, {"next_offset": "2"}, {"limit": 1.5}],
)
def test_present_malformed_pagination_is_an_error(tool, pagination):
    with patch("tool.fxmacrodata.requests.get", return_value=_response(200, {"data": [], "pagination": pagination})):
        output = tool.run(FXMacroDataToolInputSchema(endpoint="latest", currency="usd"))
    assert "unexpected response body" in output.error
    assert output.pagination is None


def test_null_pagination_is_treated_as_absent(tool):
    with patch("tool.fxmacrodata.requests.get", return_value=_response(200, {"data": [], "pagination": None})):
        output = tool.run(FXMacroDataToolInputSchema(endpoint="latest", currency="usd"))
    assert output.error is None
    assert output.pagination is None


def test_non_json_200_response_returns_an_error(tool):
    response = _response(200, None)
    response.json.side_effect = ValueError("Expecting value")
    with patch("tool.fxmacrodata.requests.get", return_value=response):
        output = tool.run(FXMacroDataToolInputSchema(endpoint="latest", currency="usd"))
    assert output.error == "FXMacroData returned a response that is not JSON."


@pytest.mark.asyncio
async def test_run_async(tool):
    with patch("tool.fxmacrodata.requests.get", return_value=_response(200, ANNOUNCEMENTS_PAYLOAD)):
        output = await tool.run_async(
            FXMacroDataToolInputSchema(endpoint="announcements", currency="usd", indicator="inflation")
        )
    assert len(output.data) == 2
