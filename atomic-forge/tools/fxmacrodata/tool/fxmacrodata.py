import asyncio
import os
import re
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import requests
from pydantic import Field

from atomic_agents import BaseIOSchema, BaseTool, BaseToolConfig


_CURRENCY_RE = re.compile(r"^[A-Za-z]{3}$")
_SLUG_RE = re.compile(r"^[A-Za-z0-9_]+$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Endpoints whose responses wrap rows in a "data" list next to metadata.
_ROW_ENDPOINTS = {"announcements", "latest", "calendar", "forex", "cot", "commodities"}


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
    data: Union[List[Dict[str, Any]], Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Rows returned by the API. Announcement rows carry 'date' (reference period), 'val', and"
            " 'announcement_datetime' (Unix seconds of the official release). For 'catalogue' this is a mapping"
            " of indicator slug to its metadata."
        ),
    )
    pagination: Optional[Dict[str, Any]] = Field(
        None, description="Paging info ('has_more', 'next_offset', 'total_count') when the endpoint pages."
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Remaining top-level response fields, such as source, units, provenance and data quality.",
    )
    error: Optional[str] = Field(None, description="Error message when the request failed.")


#################
# CONFIGURATION #
#################
class FXMacroDataToolConfig(BaseToolConfig):
    """Configuration for the FXMacroDataTool."""

    api_key: str = Field(
        default="",
        description="FXMacroData API key. Falls back to the FXMACRODATA_API_KEY environment variable. Optional for USD data.",
    )
    base_url: str = Field(default="https://api.fxmacrodata.com/v1", description="FXMacroData API base URL.")
    timeout: float = Field(default=30.0, ge=1.0, le=120.0, description="HTTP request timeout in seconds.")


#####################
# MAIN TOOL & LOGIC #
#####################
class FXMacroDataTool(BaseTool[FXMacroDataToolInputSchema, FXMacroDataToolOutputSchema]):
    """Tool for querying macroeconomic releases, release calendars, FX and positioning data from FXMacroData."""

    def __init__(self, config: FXMacroDataToolConfig = FXMacroDataToolConfig()):
        super().__init__(config)
        self.api_key = config.api_key or os.getenv("FXMACRODATA_API_KEY", "")
        self.base_url = config.base_url.rstrip("/")
        self.timeout = config.timeout

    @staticmethod
    def _currency(value: Optional[str], name: str) -> str:
        if not value or not _CURRENCY_RE.match(value.strip()):
            raise ValueError(f"'{name}' must be a three-letter currency code such as 'usd' or 'eur'.")
        return value.strip().lower()

    @staticmethod
    def _slug(value: Optional[str], name: str) -> str:
        if not value or not _SLUG_RE.match(value.strip()):
            raise ValueError(f"'{name}' must be an indicator slug such as 'inflation' or 'policy_rate'.")
        return value.strip().lower()

    @classmethod
    def build_request(cls, params: FXMacroDataToolInputSchema) -> Tuple[str, Dict[str, str]]:
        """Return the (path, query) pair for an input, validating the fields the endpoint needs."""
        endpoint = params.endpoint
        if endpoint == "commodities":
            path = f"/commodities/{cls._slug(params.indicator, 'indicator')}"
        else:
            currency = cls._currency(params.currency, "currency")
            if endpoint == "announcements":
                path = f"/announcements/{currency}/{cls._slug(params.indicator, 'indicator')}"
            elif endpoint == "latest":
                path = f"/announcements/{currency}/latest"
            elif endpoint == "calendar":
                path = f"/calendar/{currency}"
            elif endpoint == "catalogue":
                path = f"/data_catalogue/{currency}"
            elif endpoint == "forex":
                path = f"/forex/{currency}/{cls._currency(params.quote, 'quote')}"
            else:
                path = f"/cot/{currency}"

        query: Dict[str, str] = {}
        for name in ("start_date", "end_date"):
            value = getattr(params, name)
            if value:
                if not _DATE_RE.match(value):
                    raise ValueError(f"'{name}' must be formatted YYYY-MM-DD.")
                query[name] = value
        if params.limit is not None:
            query["limit"] = str(params.limit)
        if params.offset is not None:
            query["offset"] = str(params.offset)
        if endpoint == "calendar" and params.indicator:
            query["indicator"] = cls._slug(params.indicator, "indicator")
        return path, query

    @staticmethod
    def _error_message(response: requests.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            body = None
        detail = body.get("detail") if isinstance(body, dict) else None
        if isinstance(detail, str) and detail:
            return f"FXMacroData returned {response.status_code}: {detail}"
        return f"FXMacroData returned {response.status_code} {response.reason}"

    def _get(self, path: str, query: Dict[str, str]) -> Union[Dict[str, Any], List[Any]]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        response = requests.get(f"{self.base_url}{path}", params=query, headers=headers, timeout=self.timeout)
        if response.status_code != 200:
            raise RuntimeError(self._error_message(response))
        return response.json()

    def run(self, params: FXMacroDataToolInputSchema) -> FXMacroDataToolOutputSchema:
        try:
            path, query = self.build_request(params)
        except ValueError as exc:
            return FXMacroDataToolOutputSchema(endpoint=params.endpoint, url="", error=str(exc))

        url = requests.Request("GET", f"{self.base_url}{path}", params=query).prepare().url
        try:
            payload = self._get(path, query)
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            return FXMacroDataToolOutputSchema(endpoint=params.endpoint, url=url, error=str(exc))

        if params.endpoint in _ROW_ENDPOINTS and isinstance(payload, dict):
            metadata = {k: v for k, v in payload.items() if k not in ("data", "pagination")}
            return FXMacroDataToolOutputSchema(
                endpoint=params.endpoint,
                url=url,
                data=payload.get("data") or [],
                pagination=payload.get("pagination"),
                metadata=metadata,
            )
        return FXMacroDataToolOutputSchema(endpoint=params.endpoint, url=url, data=payload)

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
