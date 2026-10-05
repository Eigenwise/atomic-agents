import asyncio
import os
import re
from datetime import date
from typing import Dict, List, Literal, Optional, Tuple, Union
from urllib.parse import urlsplit

import requests
from pydantic import Field, JsonValue, SecretStr, field_validator

from atomic_agents import BaseIOSchema, BaseTool, BaseToolConfig


CURRENCY_PATTERN = re.compile(r"^[a-z]{3}$")
INDICATOR_PATTERN = re.compile(r"^[a-z0-9_]+$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# Printable ASCII without spaces: anything else cannot travel in an HTTP header.
API_KEY_PATTERN = re.compile(r"^[\x21-\x7e]+$")

ENDPOINT_PATHS = {
    "announcements": "/announcements/{currency}/{indicator}",
    "latest": "/announcements/{currency}/latest",
    "calendar": "/calendar/{currency}",
    "catalogue": "/data_catalogue/{currency}",
    "forex": "/forex/{currency}/{quote}",
    "cot": "/cot/{currency}",
    "commodities": "/commodities/{indicator}",
}

FIELD_RULES = {
    "currency": (CURRENCY_PATTERN, "a three-letter currency code such as 'usd' or 'eur'"),
    "quote": (CURRENCY_PATTERN, "a three-letter currency code such as 'usd' or 'eur'"),
    "indicator": (INDICATOR_PATTERN, "an indicator slug such as 'inflation' or 'policy_rate'"),
}

JsonObject = Dict[str, JsonValue]

# Documented pagination fields and the exact JSON types each may hold. bool is listed
# separately because it is a subclass of int and must not pass as a count.
PAGINATION_TYPES = {
    "limit": {int},
    "offset": {int},
    "returned_count": {int},
    "total_count": {int},
    "next_offset": {int, type(None)},
    "has_more": {bool},
    "page_includes_latest_available": {bool},
}


class FXMacroDataError(Exception):
    """A request or response problem, worded so it is safe to hand back to an agent."""


################
# INPUT SCHEMA #
################
class FXMacroDataToolInputSchema(BaseIOSchema):
    """
    Look up official-source macroeconomic data from FXMacroData: economic releases with their
    announcement timestamps (CPI, GDP, unemployment, payrolls, policy rates, bond yields and more),
    the latest value of every indicator for a currency, upcoming release dates, the indicator
    catalogue, FX spot history, CFTC Commitments of Traders positioning, and commodity prices.
    USD releases work without an API key; other currencies, FX, COT and commodities need one.
    """

    endpoint: Literal["announcements", "latest", "calendar", "catalogue", "forex", "cot", "commodities"] = Field(
        ...,
        description=(
            "'announcements' = release history for one indicator of a currency; 'latest' = most recent value of every"
            " indicator for a currency; 'calendar' = upcoming release dates for a currency; 'catalogue' = available"
            " indicator slugs for a currency (call this first if unsure of a slug); 'forex' = FX spot history for"
            " currency/quote; 'cot' = CFTC positioning for a currency; 'commodities' = price history for a commodity"
            " indicator."
        ),
    )
    currency: Optional[str] = Field(
        None,
        description="Three-letter currency code, e.g. 'usd', 'eur', 'jpy'. Required for every endpoint except commodities."
        " For 'forex' this is the base currency.",
    )
    indicator: Optional[str] = Field(
        None,
        description="Indicator slug, e.g. 'inflation', 'policy_rate', 'gdp', 'unemployment', 'non_farm_payrolls'."
        " Required for 'announcements' and 'commodities' (e.g. 'gold'); optional filter for 'calendar'.",
    )
    quote: Optional[str] = Field(None, description="Quote currency for the 'forex' endpoint, e.g. 'usd' for EUR/USD.")
    start_date: Optional[str] = Field(None, description="Start of the date range, YYYY-MM-DD.")
    end_date: Optional[str] = Field(None, description="End of the date range, YYYY-MM-DD.")
    limit: Optional[int] = Field(None, ge=1, le=100, description="Rows per page for history endpoints (API default 20).")
    offset: Optional[int] = Field(None, ge=0, description="Row offset for paging through history endpoints.")


####################
# OUTPUT SCHEMA(S) #
####################
class FXMacroDataToolOutputSchema(BaseIOSchema):
    """Output of the FXMacroData tool. Field names inside `data` are the API's own."""

    endpoint: str = Field(..., description="The endpoint that was called.")
    url: str = Field(..., description="The request URL, without the API key.")
    data: Union[List[JsonObject], JsonObject] = Field(
        default_factory=list,
        description=(
            "Rows returned by the API. Announcement rows carry 'date' (reference period), 'val', and"
            " 'announcement_datetime' (Unix seconds of the official release). For 'catalogue' this is a mapping"
            " of indicator slug to its metadata."
        ),
    )
    pagination: Optional[JsonObject] = Field(
        None, description="Paging info ('has_more', 'next_offset', 'total_count') when the endpoint pages."
    )
    metadata: JsonObject = Field(
        default_factory=dict,
        description="Remaining top-level response fields, such as source, units, provenance and data quality.",
    )
    error: Optional[str] = Field(None, description="Error message when the request failed.")


#################
# CONFIGURATION #
#################
class FXMacroDataToolConfig(BaseToolConfig):
    """Configuration for the FXMacroDataTool."""

    api_key: SecretStr = Field(
        default=SecretStr(""),
        description="FXMacroData API key. Falls back to the FXMACRODATA_API_KEY environment variable. Optional for USD data.",
    )
    base_url: str = Field(default="https://api.fxmacrodata.com/v1", description="FXMacroData API base URL (HTTPS only).")
    timeout: float = Field(default=30.0, ge=1.0, le=120.0, description="HTTP request timeout in seconds.")

    @field_validator("base_url")
    @classmethod
    def require_https_host(cls, value: str) -> str:
        """Reject anything but an https:// URL with a host, so a bad value fails here rather than mid-request."""
        base_url = value.strip().rstrip("/")
        parts = urlsplit(base_url)
        if parts.scheme != "https" or not parts.hostname:
            raise ValueError("base_url must be an https:// URL with a host, e.g. https://api.fxmacrodata.com/v1")
        return base_url


####################
# REQUEST BUILDING #
####################
def validated_field(params: FXMacroDataToolInputSchema, name: str) -> str:
    """Return a path field lower-cased, or raise if it is missing or malformed."""
    pattern, expected = FIELD_RULES[name]
    value = (getattr(params, name) or "").strip().lower()
    if not pattern.match(value):
        raise FXMacroDataError(f"'{name}' must be {expected}.")
    return value


def parse_date(name: str, value: str) -> date:
    """Parse a YYYY-MM-DD string into a real calendar date."""
    try:
        if DATE_PATTERN.match(value):
            return date.fromisoformat(value)
    except ValueError:
        pass
    raise FXMacroDataError(f"'{name}' must be a real date formatted YYYY-MM-DD.")


def optional_date(name: str, value: Optional[str]) -> Optional[date]:
    return None if value is None else parse_date(name, value)


def check_order(start: Optional[date], end: Optional[date]) -> None:
    if start is not None and end is not None and start > end:
        raise FXMacroDataError("'start_date' must not be after 'end_date'.")


def date_query(params: FXMacroDataToolInputSchema) -> Dict[str, str]:
    """Validate the date range and return it as query parameters."""
    dates = {name: optional_date(name, getattr(params, name)) for name in ("start_date", "end_date")}
    check_order(dates["start_date"], dates["end_date"])
    return {name: day.isoformat() for name, day in dates.items() if day is not None}


def build_path(params: FXMacroDataToolInputSchema) -> str:
    template = ENDPOINT_PATHS[params.endpoint]
    names = re.findall(r"{(\w+)}", template)
    return template.format(**{name: validated_field(params, name) for name in names})


def build_query(params: FXMacroDataToolInputSchema) -> Dict[str, str]:
    query = date_query(params)
    paging = {"limit": params.limit, "offset": params.offset}
    query.update({name: str(value) for name, value in paging.items() if value is not None})
    if params.endpoint == "calendar" and params.indicator is not None:
        query["indicator"] = validated_field(params, "indicator")
    return query


def build_request(params: FXMacroDataToolInputSchema) -> Tuple[str, Dict[str, str]]:
    """Return the (path, query) pair for an input, validating the fields the endpoint needs."""
    return build_path(params), build_query(params)


####################
# RESPONSE PARSING #
####################
def is_object_list(value: JsonValue) -> bool:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def upstream_detail(payload: JsonValue) -> str:
    detail = payload.get("detail") if isinstance(payload, dict) else None
    return f": {detail}" if isinstance(detail, str) and detail else ""


def unexpected_body(payload: JsonValue) -> FXMacroDataError:
    return FXMacroDataError(f"FXMacroData returned an unexpected response body{upstream_detail(payload)}")


def response_object(payload: JsonValue) -> JsonObject:
    if not isinstance(payload, dict):
        raise unexpected_body(payload)
    return payload


def valid_pagination(value: JsonValue) -> bool:
    """Absent or null is fine; anything else must be an object whose documented fields have their documented types."""
    if value is None:
        return True
    if not isinstance(value, dict):
        return False
    return all(type(value[name]) in PAGINATION_TYPES[name] for name in PAGINATION_TYPES.keys() & value.keys())


def metadata_fields(body: JsonObject) -> JsonObject:
    return {name: value for name, value in body.items() if name not in ("data", "pagination")}


def parse_rows(payload: JsonValue) -> Tuple[List[JsonObject], Optional[JsonObject], JsonObject]:
    """Split a row endpoint's body into rows, pagination and the remaining metadata."""
    body = response_object(payload)
    rows, pagination = body.get("data"), body.get("pagination")
    if not is_object_list(rows) or not valid_pagination(pagination):
        raise unexpected_body(body)
    return rows, pagination, metadata_fields(body)


def parse_catalogue(payload: JsonValue) -> JsonObject:
    """The catalogue is a mapping of indicator slug to an object describing it."""
    body = response_object(payload)
    if not body or not all(isinstance(entry, dict) for entry in body.values()):
        raise unexpected_body(body)
    return body


#####################
# MAIN TOOL & LOGIC #
#####################
class FXMacroDataTool(BaseTool[FXMacroDataToolInputSchema, FXMacroDataToolOutputSchema]):
    """Tool for querying macroeconomic releases, release calendars, FX and positioning data from FXMacroData."""

    def __init__(self, config: FXMacroDataToolConfig = FXMacroDataToolConfig()):
        super().__init__(config)
        self._api_key = (config.api_key.get_secret_value() or os.getenv("FXMACRODATA_API_KEY", "")).strip()
        self.base_url = config.base_url
        self.timeout = config.timeout

    def redact(self, message: str) -> str:
        """Remove the API key from any text that leaves the tool."""
        return message.replace(self._api_key, "[redacted]") if self._api_key else message

    def redact_json(self, value: JsonValue) -> JsonValue:
        """Redact the API key from every string in a response body, keys included."""
        if isinstance(value, list):
            return [self.redact_json(item) for item in value]
        if isinstance(value, dict):
            return self.redact_object(value)
        return self.redact(value) if isinstance(value, str) else value

    def redact_object(self, value: JsonObject) -> JsonObject:
        return {self.redact(name): self.redact_json(item) for name, item in value.items()}

    def headers(self) -> Dict[str, str]:
        if self._api_key and not API_KEY_PATTERN.match(self._api_key):
            raise FXMacroDataError("The FXMacroData API key contains characters that cannot be sent in an HTTP header.")
        return {"Accept": "application/json", **({"X-API-Key": self._api_key} if self._api_key else {})}

    def fetch(self, url: str, query: Dict[str, str]) -> JsonValue:
        """GET one URL. Redirects are never followed, so the key only goes to the configured HTTPS origin."""
        try:
            response = requests.get(url, params=query, headers=self.headers(), timeout=self.timeout, allow_redirects=False)
        except requests.RequestException as error:
            raise FXMacroDataError(f"Could not reach FXMacroData ({type(error).__name__}).") from None
        if response.status_code != 200:
            raise FXMacroDataError(self.status_message(response))
        try:
            return response.json()
        except ValueError:
            raise FXMacroDataError("FXMacroData returned a response that is not JSON.") from None

    @staticmethod
    def status_message(response: requests.Response) -> str:
        if 300 <= response.status_code < 400:
            return f"FXMacroData answered with a redirect ({response.status_code}); redirects are not followed."
        try:
            detail = upstream_detail(response.json())
        except ValueError:
            detail = ""
        return f"FXMacroData returned {response.status_code}{detail or ' ' + str(response.reason)}"

    def query_endpoint(self, params: FXMacroDataToolInputSchema) -> FXMacroDataToolOutputSchema:
        path, query = build_request(params)
        url = f"{self.base_url}{path}"
        display_url = requests.Request("GET", url, params=query).prepare().url
        payload = self.redact_json(self.fetch(url, query))
        if params.endpoint == "catalogue":
            return FXMacroDataToolOutputSchema(endpoint=params.endpoint, url=display_url, data=parse_catalogue(payload))
        rows, pagination, metadata = parse_rows(payload)
        return FXMacroDataToolOutputSchema(
            endpoint=params.endpoint, url=display_url, data=rows, pagination=pagination, metadata=metadata
        )

    def run(self, params: FXMacroDataToolInputSchema) -> FXMacroDataToolOutputSchema:
        try:
            return self.query_endpoint(params)
        except FXMacroDataError as error:
            return FXMacroDataToolOutputSchema(endpoint=params.endpoint, url="", error=self.redact(str(error)))

    async def run_async(self, params: FXMacroDataToolInputSchema) -> FXMacroDataToolOutputSchema:
        return await asyncio.to_thread(self.run, params)


#################
# EXAMPLE USAGE #
#################
if __name__ == "__main__":  # pragma: no cover
    from dotenv import load_dotenv
    from rich.console import Console

    load_dotenv()
    console = Console()
    tool = FXMacroDataTool()

    output = tool.run(FXMacroDataToolInputSchema(endpoint="announcements", currency="usd", indicator="inflation", limit=3))
    console.rule("[bold cyan]USD CPI")
    console.print(output.error or [(row["date"], row["val"]) for row in output.data])

    output = tool.run(FXMacroDataToolInputSchema(endpoint="calendar", currency="usd"))
    console.rule("[bold cyan]Upcoming USD releases")
    for row in output.data[:5]:
        console.print(row.get("announcement_datetime_utc"), row.get("name"))
