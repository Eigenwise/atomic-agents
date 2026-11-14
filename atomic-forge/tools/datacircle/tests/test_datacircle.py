import json
import os
import sys
from unittest.mock import patch

import pytest
import requests
from pydantic import ValidationError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tool.datacircle import (  # noqa: E402
    DEFAULT_PLAN_FIELDS,
    DatacircleTool,
    DatacircleToolConfig,
    DatacircleToolInputSchema,
    DatacircleToolOutputSchema,
    build_request,
)

BASE = "https://api.datacircle.dev/v1"

PLAN = {
    "provider": "Emelia",
    "api": "Employee email",
    "plan": "100,000 credits per month",
    "rank": 1,
    "score": 76.2,
    "perThousand": 4.99,
    "notFoundPerThousand": 0,
    "billing": "monthly",
    "pay": 499,
    "currency": "USD",
    "strengths": ["under $5 per 1,000"],
    "page": "https://datacircle.dev/find-email-apis/emelia/19-monthly",
    "website": "https://emelia.io",
    "pitch": "Find verified professional emails.",
    "latencyMs": None,
}
SECOND_PLAN = {**PLAN, "provider": "Bridgly", "plan": "Scale", "rank": 2, "score": 70.1}

APIS_PAYLOAD = [
    {
        "api": "Employee email",
        "slug": "email-finder",
        "plans": 5404,
        "providers": 300,
        "sheetPage": "https://datacircle.dev/email-finder-api-pricing",
        "bestPage": "https://datacircle.dev/best-email-finder-api",
        "best": {"provider": "Emelia", "plan": "100,000 credits per month", "score": 76.2, "perThousand": 4.99},
    }
]
BEST_PAYLOAD = {
    "api": "Employee email",
    "slug": "email-finder",
    "sheetPage": "https://datacircle.dev/email-finder-api-pricing",
    "bestPage": "https://datacircle.dev/best-email-finder-api",
    "total": 5404,
    "plans": [PLAN, SECOND_PLAN],
}
PLANS_PAYLOAD = {"total": 313, "offset": 0, "plans": [PLAN]}
PROVIDER_PAYLOAD = {
    "provider": "Emelia",
    "slug": "emelia",
    "website": "https://emelia.io",
    "apis": [
        {
            "api": "Employee email",
            "slug": "email-finder",
            "rank": 1,
            "sheetPage": "https://datacircle.dev/email-finder-api-pricing",
            "bestPage": "https://datacircle.dev/best-email-finder-api",
            "plans": [PLAN, SECOND_PLAN],
        }
    ],
}
SLUG_404 = {"error": "no API has the slug nope", "slugs": ["linkedin-profile", "email-finder"]}

TRIMMED_PLAN = {name: PLAN[name] for name in DEFAULT_PLAN_FIELDS}


def _real_response(request: requests.PreparedRequest, status: int, body: str, headers=None) -> requests.Response:
    """A genuine requests.Response, so redirects, headers and JSON decoding run through Requests itself."""
    response = requests.Response()
    response.status_code = status
    response._content = body.encode()
    response.headers.update(headers or {})
    response.url = request.url
    response.request = request
    response.reason = {200: "OK", 302: "Found", 404: "Not Found", 502: "Bad Gateway"}.get(status, "Error")
    return response


class FakeApi:
    """Stands in for the network: records every request sent and answers with one canned response."""

    def __init__(self, status=200, payload=None, body=None, headers=None):
        self.status, self.headers, self.sent = status, headers, []
        self.body = body if body is not None else json.dumps(payload)

    def send(self, request, **kwargs):
        """Patched in for HTTPAdapter.send: a bound method is not rebound, so the adapter is not passed."""
        self.sent.append(request)
        return _real_response(request, self.status, self.body, self.headers)

    def run(self, tool, **fields):
        with patch("requests.adapters.HTTPAdapter.send", self.send):
            return tool.run(DatacircleToolInputSchema(**fields))


@pytest.fixture
def tool():
    return DatacircleTool(DatacircleToolConfig())


@pytest.mark.parametrize(
    "fields, path, query",
    [
        ({"call": "list_apis", "limit": 5, "query": "ignored"}, "/apis", {}),
        ({"call": "best_plans", "api": "email-finder"}, "/apis/email-finder/best", {}),
        ({"call": "best_plans", "api": " Email-Finder ", "limit": 3}, "/apis/email-finder/best", {"limit": "3"}),
        (
            {"call": "search_plans", "api": "Employee email", "query": "apollo", "sort": "price", "limit": 50, "offset": 0},
            "/plans",
            {"api": "Employee email", "query": "apollo", "sort": "price", "limit": "50", "offset": "0"},
        ),
        ({"call": "search_plans"}, "/plans", {}),
        ({"call": "provider", "provider": "bright-data", "limit": 2}, "/providers/bright-data", {}),
    ],
)
def test_build_request(fields, path, query):
    assert build_request(DatacircleToolInputSchema(**fields)) == (path, query)


@pytest.mark.parametrize(
    "fields, message",
    [
        ({"call": "best_plans"}, "'best_plans' needs 'api' as a slug"),
        ({"call": "best_plans", "api": "Employee email"}, "'best_plans' needs 'api' as a slug"),
        ({"call": "best_plans", "api": "../providers/x"}, "'best_plans' needs 'api' as a slug"),
        ({"call": "best_plans", "api": "email--finder"}, "'best_plans' needs 'api' as a slug"),
        ({"call": "provider"}, "'provider' needs 'provider' as a slug"),
        ({"call": "provider", "provider": "bright data"}, "'provider' needs 'provider' as a slug"),
        ({"call": "provider", "provider": "bright-data?x=1"}, "'provider' needs 'provider' as a slug"),
    ],
)
def test_invalid_slugs_return_an_error_without_a_request(tool, fields, message):
    api = FakeApi(payload=BEST_PAYLOAD)
    output = api.run(tool, **fields)
    assert api.sent == []
    assert message in output.error
    assert output.data == {} and output.url == ""


@pytest.mark.parametrize(
    "fields",
    [
        {"call": "search_plans", "query": ""},
        {"call": "search_plans", "query": "   "},
        {"call": "search_plans", "api": ""},
        {"call": "provider", "provider": " "},
        {"call": "best_plans", "api": "email-finder", "limit": 0},
        {"call": "best_plans", "api": "email-finder", "limit": 51},
        {"call": "search_plans", "offset": -1},
        {"call": "search_plans", "sort": "cheapest"},
        {"call": "plans"},
    ],
)
def test_out_of_range_and_blank_inputs_fail_validation(fields):
    with pytest.raises(ValidationError):
        DatacircleToolInputSchema(**fields)


def test_list_apis(tool):
    api = FakeApi(payload=APIS_PAYLOAD)
    output = api.run(tool, call="list_apis")
    assert [request.url for request in api.sent] == [f"{BASE}/apis"]
    assert api.sent[0].headers["Accept"] == "application/json"
    assert output == DatacircleToolOutputSchema(call="list_apis", url=f"{BASE}/apis", data=APIS_PAYLOAD)


def test_best_plans_keeps_the_default_plan_fields(tool):
    api = FakeApi(payload=BEST_PAYLOAD)
    output = api.run(tool, call="best_plans", api="email-finder", limit=2)
    assert output.error is None
    assert output.url == f"{BASE}/apis/email-finder/best?limit=2"
    assert api.sent[0].url == output.url
    assert output.data["plans"][0] == TRIMMED_PLAN
    assert list(output.data["plans"][0]) == DEFAULT_PLAN_FIELDS
    assert {name: output.data[name] for name in ("api", "slug", "total", "bestPage")} == {
        "api": "Employee email",
        "slug": "email-finder",
        "total": 5404,
        "bestPage": "https://datacircle.dev/best-email-finder-api",
    }


def test_search_plans_sends_every_given_parameter(tool):
    api = FakeApi(payload=PLANS_PAYLOAD)
    output = api.run(tool, call="search_plans", api="Employee email", query="emelia", sort="price", limit=1, offset=4)
    assert api.sent[0].url == f"{BASE}/plans?api=Employee+email&query=emelia&sort=price&limit=1&offset=4"
    assert output.data == {"total": 313, "offset": 0, "plans": [TRIMMED_PLAN]}


def test_provider_keeps_limit_plans_of_each_api(tool):
    api = FakeApi(payload=PROVIDER_PAYLOAD)
    output = api.run(tool, call="provider", provider="Emelia", limit=1)
    assert api.sent[0].url == f"{BASE}/providers/emelia"
    assert output.data["provider"] == "Emelia"
    assert output.data["apis"][0]["rank"] == 1
    assert output.data["apis"][0]["plans"] == [TRIMMED_PLAN]


def test_provider_keeps_every_plan_without_limit(tool):
    output = FakeApi(payload=PROVIDER_PAYLOAD).run(tool, call="provider", provider="emelia")
    assert [plan["provider"] for plan in output.data["apis"][0]["plans"]] == ["Emelia", "Bridgly"]


def test_plan_fields_none_keeps_every_field():
    tool = DatacircleTool(DatacircleToolConfig(plan_fields=None))
    output = FakeApi(payload=BEST_PAYLOAD).run(tool, call="best_plans", api="email-finder")
    assert output.data["plans"] == [PLAN, SECOND_PLAN]


def test_plan_fields_choose_the_fields_and_skip_missing_ones():
    tool = DatacircleTool(DatacircleToolConfig(plan_fields=["provider", "latencyMs", "notAField"]))
    output = FakeApi(payload=PLANS_PAYLOAD).run(tool, call="search_plans")
    assert output.data["plans"] == [{"provider": "Emelia", "latencyMs": None}]


def test_default_plan_fields_are_not_shared_between_configs():
    first = DatacircleToolConfig()
    first.plan_fields.append("pitch")
    assert DatacircleToolConfig().plan_fields == DEFAULT_PLAN_FIELDS


def test_unknown_slug_error_lists_the_valid_slugs(tool):
    output = FakeApi(404, SLUG_404).run(tool, call="best_plans", api="nope")
    assert output.error == "Datacircle returned 404: no API has the slug nope. Valid slugs: linkedin-profile, email-finder"
    assert output.data == {}


@pytest.mark.parametrize(
    "status, body, message",
    [
        (
            400,
            json.dumps({"error": "api must be one of: Employee email"}),
            "Datacircle returned 400: api must be one of: Employee email",
        ),
        (404, json.dumps({"error": "no provider has the slug x"}), "Datacircle returned 404: no provider has the slug x"),
        (404, json.dumps({"error": "", "slugs": "email-finder"}), "Datacircle returned 404: Not Found"),
        (404, json.dumps({"error": 5, "slugs": ["email-finder", 3]}), "Datacircle returned 404: Not Found"),
        (404, json.dumps(["no", "object"]), "Datacircle returned 404: Not Found"),
        (502, "<html>Bad Gateway</html>", "Datacircle returned 502: Bad Gateway"),
    ],
)
def test_http_errors_carry_the_api_text(tool, status, body, message):
    output = FakeApi(status, body=body).run(tool, call="provider", provider="nope")
    assert output.error == message


def test_rate_limit_says_how_long_to_wait(tool):
    output = FakeApi(429, {"error": "too many calls"}, headers={"Retry-After": "17"}).run(tool, call="list_apis")
    assert output.error == "Datacircle allows 60 calls a minute from one address: wait 17 seconds and call again."


def test_rate_limit_without_retry_after(tool):
    output = FakeApi(429, {"error": "too many calls"}).run(tool, call="list_apis")
    assert "wait 60 seconds" in output.error


def test_redirects_are_not_followed(tool):
    api = FakeApi(302, body="", headers={"Location": "http://elsewhere.example/apis"})
    output = api.run(tool, call="list_apis")
    assert [request.url for request in api.sent] == [f"{BASE}/apis"]
    assert output.error == "Datacircle answered with a redirect (302); redirects are not followed."


@pytest.mark.parametrize("error", [requests.ConnectionError("secret detail"), requests.Timeout("slow")])
def test_request_exceptions_are_reported_by_type_only(tool, error):
    with patch("requests.adapters.HTTPAdapter.send", side_effect=error):
        output = tool.run(DatacircleToolInputSchema(call="list_apis"))
    assert output.error == f"Could not reach Datacircle ({type(error).__name__})."


def test_non_json_200_response_returns_an_error(tool):
    output = FakeApi(body="<html>maintenance</html>").run(tool, call="list_apis")
    assert output.error == "Datacircle returned a response that is not JSON."


@pytest.mark.parametrize(
    "fields, payload",
    [
        ({"call": "list_apis"}, {"apis": []}),
        ({"call": "list_apis"}, ["email-finder"]),
        ({"call": "best_plans", "api": "email-finder"}, []),
        ({"call": "best_plans", "api": "email-finder"}, {"total": 1}),
        ({"call": "best_plans", "api": "email-finder"}, {"total": 1, "plans": {"provider": "Emelia"}}),
        ({"call": "best_plans", "api": "email-finder"}, {"total": 1, "plans": ["Emelia"]}),
        ({"call": "best_plans", "api": "email-finder"}, {"total": "1", "plans": []}),
        ({"call": "best_plans", "api": "email-finder"}, {"total": True, "plans": []}),
        ({"call": "best_plans", "api": "email-finder"}, {"plans": []}),
        ({"call": "search_plans"}, {"total": 1, "offset": None, "plans": []}),
        ({"call": "search_plans"}, {"total": -1, "offset": 0, "plans": []}),
        ({"call": "search_plans"}, {"error": "upstream failure"}),
        ({"call": "provider", "provider": "emelia"}, {"provider": "Emelia"}),
        ({"call": "provider", "provider": "emelia"}, {"apis": [{"api": "Employee email"}]}),
        ({"call": "provider", "provider": "emelia"}, {"apis": [{"plans": [1]}]}),
        ({"call": "provider", "provider": "emelia"}, "Emelia"),
    ],
)
def test_malformed_200_responses_return_an_error(tool, fields, payload):
    output = FakeApi(payload=payload).run(tool, **fields)
    assert output.error == "Datacircle returned an unexpected response body."
    assert output.data == {}


@pytest.mark.parametrize(
    "base_url",
    [
        "http://api.datacircle.dev/v1",
        "https://",
        "api.datacircle.dev/v1",
        "ftp://api.datacircle.dev/v1",
        "https://api.datacircle.dev:abc/v1",
        "https://api.datacircle.dev:65536/v1",
        "https://[invalid/v1",
    ],
)
def test_invalid_base_url_is_rejected_by_config(base_url):
    with pytest.raises(ValidationError, match="base_url must be an https:// URL with a host"):
        DatacircleToolConfig(base_url=base_url)


def test_base_url_is_normalised_and_used():
    tool = DatacircleTool(DatacircleToolConfig(base_url=" https://localhost:8443/v1/ "))
    api = FakeApi(payload=APIS_PAYLOAD)
    output = api.run(tool, call="list_apis")
    assert api.sent[0].url == "https://localhost:8443/v1/apis" == output.url


def test_timeout_is_passed_to_requests():
    seen = {}

    def send(adapter, request, **kwargs):
        seen.update(kwargs)
        return _real_response(request, 200, json.dumps(APIS_PAYLOAD))

    with patch("requests.adapters.HTTPAdapter.send", send):
        DatacircleTool(DatacircleToolConfig(timeout=5)).run(DatacircleToolInputSchema(call="list_apis"))
    assert seen["timeout"] == 5


@pytest.mark.asyncio
async def test_run_async(tool):
    with patch("requests.adapters.HTTPAdapter.send", FakeApi(payload=BEST_PAYLOAD).send):
        output = await tool.run_async(DatacircleToolInputSchema(call="best_plans", api="email-finder"))
    assert [plan["rank"] for plan in output.data["plans"]] == [1, 2]
