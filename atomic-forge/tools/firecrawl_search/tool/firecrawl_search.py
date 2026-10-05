import asyncio
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Literal, Optional, Tuple

import aiohttp
from pydantic import Field, SecretStr, field_validator

from atomic_agents import BaseIOSchema, BaseTool, BaseToolConfig


logger = logging.getLogger(__name__)


################
# INPUT SCHEMA #
################
class FirecrawlSearchToolInputSchema(BaseIOSchema):
    """
    Search the web or recent news through the Firecrawl Search API.
    Use this to find current information on a topic. Set include_content to true to also get
    each result page as Markdown in the same call. Requires a Firecrawl API key.
    """

    queries: List[str] = Field(
        ...,
        min_length=1,
        description='Search queries to run. Supports operators such as site:, "exact phrase", -term, and filetype:pdf.',
    )
    search_type: Literal["web", "news"] = Field(
        default="web",
        description="Which results to return. 'web' = web pages; 'news' = news articles.",
    )
    max_results_per_query: int = Field(default=5, ge=1, le=100, description="Maximum results per query.")
    include_content: bool = Field(
        default=False,
        description="Also return the main content of each result page as Markdown. Slower than a plain search.",
    )


####################
# OUTPUT SCHEMA(S) #
####################
class FirecrawlSearchResultItem(BaseIOSchema):
    """A single Firecrawl search result (web page or news article)."""

    query: str = Field(..., description="The query that produced this result.")
    title: str = Field(..., description="Title of the result.")
    url: str = Field(..., description="URL of the result.")
    description: Optional[str] = Field(None, description="Query-relevant excerpt or snippet for the result.")
    position: Optional[int] = Field(None, description="Rank of the result.")
    published: Optional[str] = Field(None, description="Publication date string (news only).")
    content: Optional[str] = Field(None, description="Page content as Markdown (only when include_content is set).")

    @field_validator("position", mode="before")
    @classmethod
    def _positive_rank_or_none(cls, value: Any) -> Optional[int]:
        # Ranks come from the provider; anything other than a positive int (bool, float, "N/A") is dropped.
        return value if type(value) is int and value > 0 else None


class FirecrawlSearchFailure(BaseIOSchema):
    """A query that failed while other queries returned results."""

    query: str = Field(..., description="The query that failed.")
    error: str = Field(..., description="Why it failed, e.g. an HTTP status and the API's error message.")


class FirecrawlSearchToolOutputSchema(BaseIOSchema):
    """Output of the Firecrawl search tool."""

    results: List[FirecrawlSearchResultItem] = Field(..., description="Matching results across all queries.")
    failures: List[FirecrawlSearchFailure] = Field(
        default_factory=list, description="Queries that failed while others returned results, with the reason."
    )


#################
# CONFIGURATION #
#################
class FirecrawlSearchToolConfig(BaseToolConfig):
    """Configuration for the FirecrawlSearchTool."""

    api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Firecrawl API key. Falls back to the FIRECRAWL_API_KEY environment variable. Hidden in repr.",
    )
    base_url: str = Field(default="https://api.firecrawl.dev/v2", description="Firecrawl API base URL.")
    location: Optional[str] = Field(
        default=None, description="Location to localise results, e.g. 'Germany' or 'San Francisco,California,United States'."
    )
    time_range: Optional[Literal["day", "week", "month", "year"]] = Field(
        default=None, description="Only return web results from this time window. Does not apply to news."
    )
    max_content_chars: Optional[int] = Field(
        default=None, ge=1, description="Truncate each result's content to this many characters. No limit when unset."
    )
    timeout: float = Field(default=75.0, ge=1.0, le=120.0, description="HTTP request timeout in seconds.")


#####################
# MAIN TOOL & LOGIC #
#####################
class FirecrawlSearchTool(BaseTool[FirecrawlSearchToolInputSchema, FirecrawlSearchToolOutputSchema]):
    """Tool for searching the web and news via the Firecrawl Search API."""

    TIME_RANGES = {"day": "qdr:d", "week": "qdr:w", "month": "qdr:m", "year": "qdr:y"}
    # Anything in the Firecrawl key format is redacted from error and log text, plus the configured key itself
    # when it is long enough that replacing it cannot mangle ordinary words.
    KEY_PATTERN = re.compile(r"\bfc-[0-9a-f]{32}\b", re.IGNORECASE)
    MIN_REDACTED_KEY_LENGTH = 8

    def __init__(self, config: FirecrawlSearchToolConfig = FirecrawlSearchToolConfig()):
        super().__init__(config)
        self._api_key = SecretStr(config.api_key.get_secret_value() or os.getenv("FIRECRAWL_API_KEY", ""))
        self.base_url = config.base_url.rstrip("/")
        self.location = config.location
        self.time_range = config.time_range
        self.max_content_chars = config.max_content_chars
        self.timeout = config.timeout

    @staticmethod
    def _first_text(*values: Any) -> Optional[str]:
        """Return the first non-empty string; result fields are provider data and may have any type."""
        return next((value for value in values if isinstance(value, str) and value), None)

    @classmethod
    def _to_item(cls, hit: Any, query: str, max_content_chars: Optional[int] = None) -> Optional[FirecrawlSearchResultItem]:
        """Normalise one result, or return None when it has no http(s) URL."""
        url = hit.get("url") if isinstance(hit, dict) else None
        if not str(url).startswith(("http://", "https://")):
            return None
        content = cls._first_text(hit.get("markdown"))
        return FirecrawlSearchResultItem(
            query=query,
            title=cls._first_text(hit.get("title"), url),
            url=url,
            description=cls._first_text(hit.get("description"), hit.get("snippet")),
            position=hit.get("position"),
            published=cls._first_text(hit.get("date")),
            content=content and content[:max_content_chars],
        )

    def _build_body(self, query: str, params: FirecrawlSearchToolInputSchema) -> dict:
        body = {"query": query, "limit": params.max_results_per_query, "sources": [params.search_type]}
        if self.location:
            body["location"] = self.location
        if self.time_range and params.search_type == "web":
            body["tbs"] = self.TIME_RANGES[self.time_range]
        if params.include_content:
            body["scrapeOptions"] = {"formats": ["markdown"], "onlyMainContent": True}
        return body

    def _redact(self, text: str) -> str:
        key = self._api_key.get_secret_value()
        if len(key) >= self.MIN_REDACTED_KEY_LENGTH:
            text = re.sub(re.escape(key), "[redacted]", text, flags=re.IGNORECASE)
        return self.KEY_PATTERN.sub("[redacted]", text)

    @staticmethod
    def _error(query: str, detail: str) -> Exception:
        return Exception(f"Firecrawl search failed for '{query}': {detail}")

    @staticmethod
    async def _read_json(resp: aiohttp.ClientResponse) -> Any:
        try:
            return await resp.json(content_type=None)
        except ValueError:
            return None

    def _failure(self, status: int, reason: Optional[str], data: Any) -> Optional[str]:
        """Return why a response is not a successful search (only a 200 whose body says success is true is),
        or None when it is. Provider text is redacted here, where it enters the tool."""
        payload = data if isinstance(data, dict) else {}
        if status == 200 and payload.get("success") is True:
            return None
        return self._redact(f"{status} {payload.get('error') or reason}")

    async def _exchange(
        self, session: aiohttp.ClientSession, query: str, params: FirecrawlSearchToolInputSchema
    ) -> Tuple[Any, Optional[str]]:
        """Send one search request and return the parsed body plus why it failed (None on success).

        Transport errors, timeouts and invalid URLs are turned into a failure reason here instead of propagating,
        because aiohttp's own frames and errors hold the request headers, Authorization included."""
        try:
            async with session.post(
                f"{self.base_url}/search", json=self._build_body(query, params), allow_redirects=False
            ) as resp:
                payload = await self._read_json(resp)
                reason = resp.reason if resp.status != 200 else "response did not report success"
                return payload, self._failure(resp.status, reason, payload)
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as error:
            return None, self._redact(f"request failed: {type(error).__name__} {str(error) or f'after {self.timeout:g}s'}")

    async def _post(self, session: aiohttp.ClientSession, query: str, params: FirecrawlSearchToolInputSchema) -> dict:
        """Return the body of a successful search response, or raise.

        The raise happens in this frame, which holds neither the response nor an aiohttp error, and the body of a
        failed response is dropped first, so a traceback that captures locals cannot show a key echoed back in an
        error response."""
        payload, failure = await self._exchange(session, query, params)
        if failure:
            payload = None
            raise self._error(query, failure)
        return payload

    def _extract_hits(self, query: str, payload: dict, search_type: str) -> list:
        # The key for the requested source is absent when there are no results.
        groups = payload.get("data")
        hits = groups.get(search_type, []) if isinstance(groups, dict) else None
        if not isinstance(hits, list):
            raise self._error(query, f"200 unexpected response shape for '{search_type}' results")
        return hits

    async def _fetch(
        self,
        session: aiohttp.ClientSession,
        query: str,
        params: FirecrawlSearchToolInputSchema,
    ) -> List[FirecrawlSearchResultItem]:
        payload = await self._post(session, query, params)
        hits = self._extract_hits(query, payload, params.search_type)
        items = list(filter(None, (self._to_item(hit, query, self.max_content_chars) for hit in hits)))
        if hits and not items:
            raise self._error(query, f"200 none of the {len(hits)} '{params.search_type}' results had a usable url")
        return items[: params.max_results_per_query]

    def _collect(self, queries: List[str], grouped: list) -> FirecrawlSearchToolOutputSchema:
        """Merge per-query results. Raise when no query produced results and at least one failed, so a bad key,
        exhausted credits or a malformed response never looks like an empty search."""
        results: List[FirecrawlSearchResultItem] = []
        failures: List[FirecrawlSearchFailure] = []
        errors: List[Exception] = []
        for query, group in zip(queries, grouped):
            if isinstance(group, Exception):
                errors.append(group)
                failures.append(FirecrawlSearchFailure(query=query, error=str(group)))
                logger.warning("Firecrawl query '%s' failed: %s", query, failures[-1].error)
                continue
            results.extend(group)

        if errors and not results:
            raise errors[0]
        return FirecrawlSearchToolOutputSchema(results=results, failures=failures)

    async def run_async(self, params: FirecrawlSearchToolInputSchema) -> FirecrawlSearchToolOutputSchema:
        if not self._api_key.get_secret_value():
            raise ValueError(
                "Firecrawl API key is missing. Set FirecrawlSearchToolConfig.api_key or the FIRECRAWL_API_KEY "
                "environment variable."
            )

        # Duplicate queries would be billed twice and return the same results.
        queries = list(dict.fromkeys(params.queries))
        # The Authorization header is built inline so no local variable holds the key.
        async with aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self._api_key.get_secret_value()}", "Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        ) as session:
            grouped = await asyncio.gather(*(self._fetch(session, q, params) for q in queries), return_exceptions=True)

        return self._collect(queries, grouped)

    def run(self, params: FirecrawlSearchToolInputSchema) -> FirecrawlSearchToolOutputSchema:
        with ThreadPoolExecutor() as executor:
            return executor.submit(asyncio.run, self.run_async(params)).result()


#################
# EXAMPLE USAGE #
#################
if __name__ == "__main__":  # pragma: no cover
    from dotenv import load_dotenv
    from rich.console import Console

    load_dotenv()
    console = Console()
    tool = FirecrawlSearchTool(config=FirecrawlSearchToolConfig(api_key=os.getenv("FIRECRAWL_API_KEY", "")))

    examples = [
        ("web", False, "atomic agents framework"),
        ("news", False, "AI agents"),
        ("web", True, "atomic agents framework"),
    ]
    for search_type, include_content, query in examples:
        output = tool.run(
            FirecrawlSearchToolInputSchema(
                queries=[query],
                search_type=search_type,
                max_results_per_query=3,
                include_content=include_content,
            )
        )
        console.rule(f"[bold cyan]{search_type}{' + content' if include_content else ''}")
        for item in output.results:
            console.print(f"[bold]{item.title}[/bold]")
            console.print(item.url)
            if item.description:
                console.print(item.description[:200])
            if item.published:
                console.print(f"[bold]Published:[/bold] {item.published}")
            if item.content:
                console.print(f"[bold]Content:[/bold] {len(item.content)} chars")
                console.print(item.content[:300])
            console.print()
