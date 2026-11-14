import asyncio
import re
from typing import Dict, List, Literal, Optional, Tuple, Union
from urllib.parse import urlsplit

import requests
from pydantic import Field, JsonValue, field_validator

from atomic_agents import BaseIOSchema, BaseTool, BaseToolConfig

# A slug is lowercase letters and digits, words joined by single hyphens, as in 'email-finder' or 'bright-data'.
SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

CALL_PATHS = {
    "list_apis": "/apis",
    "best_plans": "/apis/{api}/best",
    "search_plans": "/plans",
    "provider": "/providers/{provider}",
}

# The plan fields an agent needs to compare plans. Every plan has 80+ fields; each one is described at
# https://datacircle.dev/fields and the plan's own page links them all.
DEFAULT_PLAN_FIELDS = [
    "rank",
    "provider",
    "plan",
    "api",
    "score",
    "perThousand",
    "notFoundPerThousand",
    "billing",
    "pay",
    "currency",
    "strengths",
    "page",
]

JsonObject = Dict[str, JsonValue]


class DatacircleError(Exception):
    """A request or response problem, worded so it can be handed back to an agent."""


################
# INPUT SCHEMA #
################
class DatacircleToolInputSchema(BaseIOSchema):
    """
    Find the best GTM API for a job (email finder, LinkedIn profiles, company data, phone finder, SERP, web scraping
    and more) on Datacircle, which benchmarks GTM APIs and scores each provider's plans out of 100 on price, accuracy,
    cost to get started, bundled calls and performance. No API key.
    """

    call: Literal["list_apis", "best_plans", "search_plans", "provider"] = Field(
        ...,
        description=(
            "'list_apis' = every API with its slug and best plan (call this first if unsure of a slug);"
            " 'best_plans' = one API's plans best first by score (needs 'api');"
            " 'search_plans' = plans of every API or one 'api', whose provider or plan name holds every word of 'query';"
            " 'provider' = one provider's rank and plans on each API it sells (needs 'provider')."
        ),
    )
    api: Optional[str] = Field(
        None,
        description="The API's slug as 'list_apis' gives it, e.g. 'email-finder' or 'linkedin-profile'. Required for"
        " 'best_plans'. For 'search_plans', an optional filter: the API's slug or its name, e.g. 'Employee email'.",
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
        description="For 'search_plans': 'score' best first (the default), 'provider' A to Z,"
        " 'price' lowest $ per 1,000 first.",
    )
    limit: Optional[int] = Field(
        None,
        ge=1,
        le=50,
        description="How many plans to give, 1 to 50 (the API gives 10 when left out). For 'provider', how many of each"
        " API's plans to keep, best first (every plan when left out).",
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
    data: Union[List[JsonObject], JsonObject] = Field(
        default_factory=dict,
        description=(
            "The API's answer. 'list_apis': a list of APIs, each with 'api', 'slug', 'plans', 'providers', 'best' and"
            " its pages. 'best_plans' and 'search_plans': an object whose 'plans' are the plans found and 'total' how"
            " many there are. 'provider': an object whose 'apis' give the provider's 'rank' and 'plans' on each API."
            " Each plan has 'rank', 'score' out of 100, 'perThousand' ($ per 1,000) and 'page' on datacircle.dev."
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


####################
# REQUEST BUILDING #
####################
def slug_field(params: DatacircleToolInputSchema, name: str) -> str:
    """Return a slug field lower-cased, or raise if it is missing or not a slug."""
    value = (getattr(params, name) or "").lower()
    if not SLUG_PATTERN.match(value):
        raise DatacircleError(f"'{params.call}' needs '{name}' as a slug, e.g. 'email-finder' or 'bright-data'.")
    return value


def build_path(params: DatacircleToolInputSchema) -> str:
    template = CALL_PATHS[params.call]
    names = re.findall(r"{(\w+)}", template)
    return template.format(**{name: slug_field(params, name) for name in names})


def build_query(params: DatacircleToolInputSchema) -> Dict[str, str]:
    """The query parameters of a call: the API's own for 'best_plans' and 'search_plans', none for the others."""
    names = {"best_plans": ("limit",), "search_plans": ("api", "query", "sort", "limit", "offset")}.get(params.call, ())
    values = {name: getattr(params, name) for name in names}
    return {name: str(value) for name, value in values.items() if value is not None}


def build_request(params: DatacircleToolInputSchema) -> Tuple[str, Dict[str, str]]:
    """Return the (path, query) pair for an input, validating the fields the call needs."""
    return build_path(params), build_query(params)


####################
# RESPONSE PARSING #
####################
def is_object_list(value: JsonValue) -> bool:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def is_count(value: JsonValue) -> bool:
    """bool is a subclass of int and must not pass as a count."""
    return type(value) is int and value >= 0


def unexpected_body() -> DatacircleError:
    return DatacircleError("Datacircle returned an unexpected response body.")


def response_object(payload: JsonValue) -> JsonObject:
    if not isinstance(payload, dict):
        raise unexpected_body()
    return payload


def check_apis(payload: JsonValue) -> List[JsonObject]:
    if not is_object_list(payload):
        raise unexpected_body()
    return payload


def check_plans(payload: JsonValue) -> JsonObject:
    """A plans answer: its 'plans' a list of objects, its 'total' and any 'offset' counts."""
    body = response_object(payload)
    counts = [body.get("total")] + ([body["offset"]] if "offset" in body else [])
    if not is_object_list(body.get("plans")) or not all(is_count(count) for count in counts):
        raise unexpected_body()
    return body


def check_provider(payload: JsonValue) -> JsonObject:
    """A provider answer: its 'apis' a list of objects, each with 'plans' a list of objects."""
    body = response_object(payload)
    apis = body.get("apis")
    if not is_object_list(apis) or not all(is_object_list(api.get("plans")) for api in apis):
        raise unexpected_body()
    return body


def keep_fields(plans: List[JsonObject], fields: Optional[List[str]], limit: Optional[int] = None) -> List[JsonObject]:
    """Keep the first `limit` plans (all when None) and, on each, only `fields` (all when None)."""
    kept = plans[:limit]
    if fields is None:
        return kept
    return [{name: plan[name] for name in fields if name in plan} for plan in kept]


def parse_answer(
    params: DatacircleToolInputSchema, payload: JsonValue, fields: Optional[List[str]]
) -> Union[List[JsonObject], JsonObject]:
    """Validate the answer's shape for its call, then trim each plan to `fields`."""
    if params.call == "list_apis":
        return check_apis(payload)
    if params.call == "provider":
        body = check_provider(payload)
        apis = [{**api, "plans": keep_fields(api["plans"], fields, params.limit)} for api in body["apis"]]
        return {**body, "apis": apis}
    body = check_plans(payload)
    return {**body, "plans": keep_fields(body["plans"], fields)}


def error_body(response: requests.Response) -> JsonObject:
    """An error answer's JSON object, or an empty one when the body is not a JSON object."""
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def valid_slugs(body: JsonObject) -> str:
    """The valid slugs a 404 lists, so an agent can correct an unknown one."""
    slugs = body.get("slugs")
    if isinstance(slugs, list) and all(isinstance(slug, str) for slug in slugs):
        return ". Valid slugs: " + ", ".join(slugs)
    return ""


def upstream_error(response: requests.Response) -> str:
    """The API's own 'error' text, with the valid slugs when it lists them."""
    body = error_body(response)
    error = body.get("error")
    detail = error if isinstance(error, str) and error else str(response.reason)
    return detail + valid_slugs(body)


def status_message(response: requests.Response) -> str:
    if response.status_code == 429:
        wait = response.headers.get("Retry-After", "60")
        return f"Datacircle allows 60 calls a minute from one address: wait {wait} seconds and call again."
    if 300 <= response.status_code < 400:
        return f"Datacircle answered with a redirect ({response.status_code}); redirects are not followed."
    return f"Datacircle returned {response.status_code}: {upstream_error(response)}"


#####################
# MAIN TOOL & LOGIC #
#####################
class DatacircleTool(BaseTool[DatacircleToolInputSchema, DatacircleToolOutputSchema]):
    """Tool for finding the best GTM API plans, their scores and prices on Datacircle's open API."""

    def __init__(self, config: DatacircleToolConfig = DatacircleToolConfig()):
        super().__init__(config)
        self.base_url = config.base_url
        self.timeout = config.timeout
        self.plan_fields = config.plan_fields

    def fetch(self, url: str, query: Dict[str, str]) -> JsonValue:
        """GET one URL. Redirects are never followed, so every call stays on the configured HTTPS origin."""
        try:
            response = requests.get(
                url, params=query, headers={"Accept": "application/json"}, timeout=self.timeout, allow_redirects=False
            )
        except requests.RequestException as error:
            raise DatacircleError(f"Could not reach Datacircle ({type(error).__name__}).") from None
        if response.status_code != 200:
            raise DatacircleError(status_message(response))
        try:
            return response.json()
        except ValueError:
            raise DatacircleError("Datacircle returned a response that is not JSON.") from None

    def call_api(self, params: DatacircleToolInputSchema) -> DatacircleToolOutputSchema:
        path, query = build_request(params)
        url = f"{self.base_url}{path}"
        display_url = requests.Request("GET", url, params=query).prepare().url
        data = parse_answer(params, self.fetch(url, query), self.plan_fields)
        return DatacircleToolOutputSchema(call=params.call, url=display_url, data=data)

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

    output = tool.run(DatacircleToolInputSchema(call="best_plans", api="email-finder", limit=3))
    console.rule("[bold cyan]Best email finder API plans")
    console.print(output.error or [(plan["rank"], plan["provider"], plan["score"]) for plan in output.data["plans"]])

    output = tool.run(DatacircleToolInputSchema(call="search_plans", api="serp", sort="price", limit=3))
    console.rule("[bold cyan]Cheapest SERP API plans")
    console.print(output.error or [(plan["provider"], plan["plan"], plan["perThousand"]) for plan in output.data["plans"]])
