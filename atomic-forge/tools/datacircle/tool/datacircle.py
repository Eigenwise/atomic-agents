import asyncio
import re
from typing import Dict, List, Literal, Optional, Union
from urllib.parse import urlsplit

import requests
from pydantic import Field, JsonValue, field_validator

from atomic_agents import BaseIOSchema, BaseTool, BaseToolConfig

# A slug is lowercase letters and digits, words joined by single hyphens, as in 'email-finder' or 'bright-data'.
SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# Each call's path, the input field its path takes as a slug, and the input fields it sends as query parameters.
CALLS = {
    "list_apis": ("/apis", None, ()),
    "ranked_plans": ("/apis/{}/best", "api", ("limit",)),
    "search_plans": ("/plans", None, ("api", "query", "sort", "limit", "offset")),
    "provider": ("/providers/{}", "provider", ()),
}

# The plan fields an agent needs to compare plans. Every plan has 80+ fields, each described at https://datacircle.dev/fields.
DEFAULT_PLAN_FIELDS = (
    "rank provider plan api score perThousand notFoundPerThousand billing pay currency strengths page".split()
)

JsonObject = Dict[str, JsonValue]


class DatacircleError(Exception):
    """A request or response problem, worded so it can be handed back to an agent."""


################
# INPUT SCHEMA #
################
class DatacircleToolInputSchema(BaseIOSchema):
    """
    Lists Datacircle's GTM API rankings and prices: the GTM APIs it covers (email finder, LinkedIn profiles, company
    data, phone finder, SERP, web scraping and more), each API's plans in Datacircle's ranking with Datacircle's score
    out of 100 and the price per 1,000 results, a search of every plan, and one provider's ranks. No API key.
    """

    call: Literal["list_apis", "ranked_plans", "search_plans", "provider"] = Field(
        ...,
        description=(
            "'list_apis' = every API with its slug and its top-ranked plan (call this first if unsure of a slug);"
            " 'ranked_plans' = one API's plans in Datacircle's ranking, highest score first (needs 'api');"
            " 'search_plans' = plans of every API or one 'api', whose provider or plan name holds every word of 'query';"
            " 'provider' = one provider's rank and plans on each API it sells (needs 'provider')."
        ),
    )
    api: Optional[str] = Field(
        None,
        description="The API's slug as 'list_apis' gives it, e.g. 'email-finder' or 'linkedin-profile'. Required for"
        " 'ranked_plans'. For 'search_plans', an optional filter: the API's slug or its name, e.g. 'Employee email'.",
    )
    provider: Optional[str] = Field(
        None,
        description="The provider's slug, its name in lowercase letters and digits joined by hyphens, e.g. 'bright-data'."
        " Required for 'provider'.",
    )
    query: Optional[str] = Field(
        None, description="For 'search_plans': words that must all appear in the provider's or plan's name, e.g. 'apollo'."
    )
    sort: Optional[Literal["score", "provider", "price"]] = Field(
        None,
        description="For 'search_plans': 'score' highest score first (the default), 'provider' A to Z,"
        " 'price' lowest $ per 1,000 first.",
    )
    limit: Optional[int] = Field(
        None,
        ge=1,
        le=50,
        description="How many plans to give, 1 to 50 (the API gives 10 when left out). For 'provider', how many of each"
        " API's plans to keep, highest score first (every plan when left out).",
    )
    offset: Optional[int] = Field(None, ge=0, description="For 'search_plans': how many found plans to skip first.")

    @field_validator("api", "provider", "query")
    @classmethod
    def reject_blank(cls, value: Optional[str]) -> Optional[str]:
        """The API refuses an empty string, so a blank value fails here rather than as a 400 later."""
        if value is not None and not value.strip():
            raise ValueError("must not be blank: leave it out instead")
        return value if value is None else value.strip()


####################
# OUTPUT SCHEMA(S) #
####################
class DatacircleToolOutputSchema(BaseIOSchema):
    """Output of the Datacircle tool. Field names inside `data` are the API's own."""

    call: str = Field(..., description="The call that was made.")
    url: str = Field(..., description="The request URL.")
    source: str = Field(
        "Datacircle", description="Whose scores and ranks these are. Name Datacircle when reporting a score or a rank."
    )
    data: Union[List[JsonObject], JsonObject] = Field(
        default_factory=dict,
        description=(
            "The answer of Datacircle's API. 'list_apis': a list of APIs, each with 'api', 'slug', 'plans', 'providers',"
            " 'best' (its top plan in Datacircle's ranking) and its pages. 'ranked_plans' and 'search_plans': an object"
            " whose 'plans' are the plans found and 'total' how many there are. 'provider': an object whose 'apis' give"
            " the provider's 'rank' in Datacircle's ranking and 'plans' on each API. Each plan has 'rank' (its place in"
            " Datacircle's ranking of its API), 'score' (Datacircle's own score out of 100, not the provider's),"
            " 'perThousand' ($ per 1,000 results) and 'page' on datacircle.dev."
        ),
    )
    error: Optional[str] = Field(None, description="Error message when the call failed.")


#################
# CONFIGURATION #
#################
class DatacircleToolConfig(BaseToolConfig):
    """Configuration for the DatacircleTool."""

    base_url: str = Field(default="https://api.datacircle.dev/v1", description="Datacircle API base URL (HTTPS only).")
    timeout: float = Field(default=30.0, ge=1.0, le=120.0, description="HTTP request timeout in seconds.")
    plan_fields: Optional[List[str]] = Field(
        default_factory=lambda: list(DEFAULT_PLAN_FIELDS),
        description="The fields kept on each plan, in this order. None keeps every field of the API's answer.",
    )

    @field_validator("base_url")
    @classmethod
    def require_https_host(cls, value: str) -> str:
        """Reject anything but an https:// URL with a host and a valid port, so a bad value fails here."""
        base_url = value.strip().rstrip("/")
        try:
            parts = urlsplit(base_url)
            if parts.scheme == "https" and parts.hostname:
                _ = parts.port  # urlsplit defers malformed or out-of-range port checks until this access.
                return base_url
        except ValueError:
            pass
        raise ValueError("base_url must be an https:// URL with a host, e.g. https://api.datacircle.dev/v1")


###########
# HELPERS #
###########
def is_object_list(value: JsonValue) -> bool:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def has_shape(call: str, payload: JsonValue) -> bool:
    """Whether an answer has the shape its call documents, so a changed API fails loudly instead of quietly."""
    if call == "list_apis":
        return is_object_list(payload)
    if not isinstance(payload, dict):
        return False
    if call == "provider":
        apis = payload.get("apis")
        return is_object_list(apis) and all(is_object_list(api.get("plans")) for api in apis)
    counts = [payload.get("total")] + ([payload["offset"]] if "offset" in payload else [])
    # bool is a subclass of int and must not pass as a count.
    return is_object_list(payload.get("plans")) and all(type(count) is int and count >= 0 for count in counts)


def keep_fields(plans: List[JsonObject], fields: Optional[List[str]], limit: Optional[int] = None) -> List[JsonObject]:
    """Keep the first `limit` plans (all when None) and, on each, only `fields` (all when None)."""
    kept = plans[:limit]
    if fields is None:
        return kept
    return [{name: plan[name] for name in fields if name in plan} for plan in kept]


def status_message(response: requests.Response) -> str:
    """A non-200 answer as an agent can act on it: the wait for a 429, the API's own error and valid slugs otherwise."""
    if response.status_code == 429:
        wait = response.headers.get("Retry-After", "60")
        return f"Datacircle allows 60 calls a minute from one address: wait {wait} seconds and call again."
    if 300 <= response.status_code < 400:
        return f"Datacircle answered with a redirect ({response.status_code}); redirects are not followed."
    try:
        body = response.json()
    except ValueError:
        body = None
    body = body if isinstance(body, dict) else {}
    error = body.get("error")
    message = error if isinstance(error, str) and error else str(response.reason)
    slugs = body.get("slugs")
    if isinstance(slugs, list) and all(isinstance(slug, str) for slug in slugs):
        message += ". Valid slugs: " + ", ".join(slugs)
    return f"Datacircle returned {response.status_code}: {message}"


#####################
# MAIN TOOL & LOGIC #
#####################
class DatacircleTool(BaseTool[DatacircleToolInputSchema, DatacircleToolOutputSchema]):
    """Tool that lists Datacircle's GTM API rankings and prices from its open API."""

    def __init__(self, config: DatacircleToolConfig = DatacircleToolConfig()):
        super().__init__(config)
        self.base_url = config.base_url
        self.timeout = config.timeout
        self.plan_fields = config.plan_fields

    def fetch(self, url: str) -> JsonValue:
        """GET one URL. Redirects are never followed, so every call stays on the configured HTTPS origin."""
        try:
            response = requests.get(url, headers={"Accept": "application/json"}, timeout=self.timeout, allow_redirects=False)
        except requests.RequestException as error:
            raise DatacircleError(f"Could not reach Datacircle ({type(error).__name__}).") from None
        if response.status_code != 200:
            raise DatacircleError(status_message(response))
        try:
            return response.json()
        except ValueError:
            raise DatacircleError("Datacircle returned a response that is not JSON.") from None

    def call_api(self, params: DatacircleToolInputSchema) -> DatacircleToolOutputSchema:
        path, slug_field, query_fields = CALLS[params.call]
        if slug_field:
            slug = (getattr(params, slug_field) or "").lower()
            if not SLUG_PATTERN.match(slug):
                raise DatacircleError(f"'{params.call}' needs '{slug_field}' as a slug, e.g. 'email-finder' or 'bright-data'.")
            path = path.format(slug)
        query = {name: str(getattr(params, name)) for name in query_fields if getattr(params, name) is not None}
        url = requests.Request("GET", self.base_url + path, params=query).prepare().url
        payload = self.fetch(url)
        if not has_shape(params.call, payload):
            raise DatacircleError("Datacircle returned an unexpected response body.")
        if params.call == "provider":
            apis = [{**api, "plans": keep_fields(api["plans"], self.plan_fields, params.limit)} for api in payload["apis"]]
            payload = {**payload, "apis": apis}
        elif params.call != "list_apis":
            payload = {**payload, "plans": keep_fields(payload["plans"], self.plan_fields)}
        return DatacircleToolOutputSchema(call=params.call, url=url, data=payload)

    def run(self, params: DatacircleToolInputSchema) -> DatacircleToolOutputSchema:
        try:
            return self.call_api(params)
        except DatacircleError as error:
            return DatacircleToolOutputSchema(call=params.call, url="", error=str(error))

    async def run_async(self, params: DatacircleToolInputSchema) -> DatacircleToolOutputSchema:
        return await asyncio.to_thread(self.run, params)


#################
# EXAMPLE USAGE #
#################
if __name__ == "__main__":  # pragma: no cover
    from rich.console import Console

    console = Console()
    tool = DatacircleTool()

    output = tool.run(DatacircleToolInputSchema(call="ranked_plans", api="email-finder", limit=3))
    console.rule("[bold cyan]Email finder API plans in Datacircle's ranking")
    console.print(output.error or [(plan["rank"], plan["provider"], plan["score"]) for plan in output.data["plans"]])

    output = tool.run(DatacircleToolInputSchema(call="search_plans", api="serp", sort="price", limit=3))
    console.rule("[bold cyan]SERP API plans by price")
    console.print(output.error or [(plan["provider"], plan["plan"], plan["perThousand"]) for plan in output.data["plans"]])
