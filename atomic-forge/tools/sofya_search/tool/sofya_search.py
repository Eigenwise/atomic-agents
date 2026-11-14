import asyncio
import contextvars
import ipaddress
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import Dict, List, Literal, Optional, Tuple, Union
from urllib.parse import urlsplit

import aiohttp
from pydantic import Field, SecretStr, field_validator

from atomic_agents import BaseIOSchema, BaseTool, BaseToolConfig


logger = logging.getLogger(__name__)


################
# INPUT SCHEMA #
################
class SofyaSearchToolInputSchema(BaseIOSchema):
    """
    Search the web or recent news through the Sofya Search API.
    Use this to find current information on a topic. Each result carries the page's text as well as a snippet,
    so the pages can be read without a separate fetch step. Requires a Sofya API key.
    """

    queries: List[str] = Field(
        ...,
        min_length=1,
        description='Search queries to run. Supports operators such as site:, "exact phrase", and -term.',
    )
    topic: Literal["general", "news"] = Field(
        default="general",
        description="Which results to return. 'general' = web pages; 'news' = news articles with publication dates.",
    )
    max_results_per_query: int = Field(default=5, ge=1, le=20, description="Maximum results per query.")


####################
# OUTPUT SCHEMA(S) #
####################
class SofyaSearchResultItem(BaseIOSchema):
    """A single Sofya search result (web page or news article)."""

    query: str = Field(..., description="The query that produced this result.")
    title: str = Field(..., description="Title of the result.")
    url: str = Field(..., description="URL of the result.")
    description: Optional[str] = Field(default=None, description="Snippet for the result.")
    published: Optional[str] = Field(default=None, description="Publication date as YYYY-MM-DD, when known.")
    content: Optional[str] = Field(
        default=None, description="Text of the result page (with search_depth 'basic', when the page could be read)."
    )


class SofyaSearchFailure(BaseIOSchema):
    """A query that failed while other queries returned results."""

    query: str = Field(..., description="The query that failed.")
    error: str = Field(..., description="Why it failed, e.g. an HTTP status and the API's error message.")


class SofyaSearchToolOutputSchema(BaseIOSchema):
    """Output of the Sofya search tool."""

    results: List[SofyaSearchResultItem] = Field(..., description="Matching results across all queries.")
    failures: List[SofyaSearchFailure] = Field(
        default_factory=list, description="Queries that failed while others returned results, with the reason."
    )


#################
# CONFIGURATION #
#################
class SofyaSearchToolConfig(BaseToolConfig):
    """Configuration for the SofyaSearchTool."""

    api_key: SecretStr = Field(
        default=SecretStr(""),
        description="Sofya API key. Falls back to the SOFYA_API_KEY environment variable. Hidden in repr.",
    )
    base_url: str = Field(default="https://sofya.co/v1", description="Sofya API base URL.")
    search_depth: Literal["snippets", "basic"] = Field(
        default="basic",
        description="'basic' returns each result's page text in content; 'snippets' returns snippets only and costs less.",
    )
    freshness: Optional[str] = Field(
        default=None,
        description="Only return results from this window: 'day', 'week', 'month', 'year', or 'YYYY-MM-DD:YYYY-MM-DD'.",
    )
    include_domains: List[str] = Field(
        default_factory=list, max_length=10, description="Only return results from these hosts, e.g. 'github.com'."
    )
    exclude_domains: List[str] = Field(
        default_factory=list, max_length=10, description="Leave out results from these hosts, e.g. 'pinterest.com'."
    )
    max_content_chars: Optional[int] = Field(
        default=None, ge=1, description="Truncate each result's content to this many characters. No limit when unset."
    )
    timeout: float = Field(default=30.0, ge=1.0, le=120.0, description="HTTP request timeout in seconds.")

    @field_validator("freshness")
    @classmethod
    def _valid_freshness(cls, value: Optional[str]) -> Optional[str]:
        if value is None or value in ("day", "week", "month", "year"):
            return value
        start, sep, end = value.partition(":")
        try:
            if sep and date.fromisoformat(start) <= date.fromisoformat(end):
                return value
        except ValueError:
            pass
        raise ValueError("freshness must be 'day', 'week', 'month', 'year', or 'YYYY-MM-DD:YYYY-MM-DD' (start <= end)")


#####################
# MAIN TOOL & LOGIC #
#####################
class SofyaSearchTool(BaseTool[SofyaSearchToolInputSchema, SofyaSearchToolOutputSchema]):
    """Tool for searching the web and news via the Sofya Search API."""

    # Anything in a Sofya key format is redacted from error and log text, plus the configured key itself when it is
    # long enough that replacing it cannot mangle ordinary words.
    KEY_PATTERN = re.compile(r"\b(?:sofya_(?:live|test)_[0-9a-z]{38}\b|ay_live_[0-9a-z_-]{16,})", re.IGNORECASE)
    MIN_REDACTED_KEY_LENGTH = 8
    # Shape of a usable result URL: http(s), optional userinfo, a host or IPv6 literal, an optional port, and no
    # whitespace anywhere. No backslash or extra "@" before the path, since parsers disagree on what host
    # "a.com\@b.com" or "a.com:1\@b.com" means. urlsplit then validates the port and the IPv6 literal, and
    # _valid_host the host name.
    RESULT_URL_PATTERN = re.compile(
        r"https?://(?:[^\s/?#@\\]*@)?(?:[^\s/?#@:\[\]]+|\[[0-9a-f:.]+\])(?::[^\s/?#@\\]*)?(?:[/?#]\S*)?", re.IGNORECASE
    )

    # One label of a host name in its ASCII (IDNA) form: letters, digits, hyphens and underscores, starting and ending
    # with a letter or digit. The IDNA codec enforces the 63-byte label limit. No DNS lookup happens; this only
    # rejects host names that cannot be valid, such as "example..com", "a%.com" or "aא.com".
    HOST_LABEL = r"[a-z0-9](?:[a-z0-9_-]*[a-z0-9])?"
    HOST_NAME_PATTERN = re.compile(rf"(?:{HOST_LABEL}\.)*{HOST_LABEL}", re.IGNORECASE)
    MAX_HOST_NAME_LENGTH = 253

    def __init__(self, config: SofyaSearchToolConfig = SofyaSearchToolConfig()):
        super().__init__(config)
        self._api_key = SecretStr(config.api_key.get_secret_value() or os.getenv("SOFYA_API_KEY", ""))
        self.base_url = config.base_url.rstrip("/")
        self.search_depth = config.search_depth
        self.freshness = config.freshness
        self.include_domains = list(config.include_domains)
        self.exclude_domains = list(config.exclude_domains)
        self.max_content_chars = config.max_content_chars
        self.timeout = config.timeout

    @staticmethod
    def _first_text(*values: object) -> Optional[str]:
        """Return the first non-empty string; result fields are provider data and may have any type."""
        return next((value for value in values if isinstance(value, str) and value), None)

    @classmethod
    def _valid_host(cls, host: Optional[str]) -> bool:
        """True for an IPv6 literal (urlsplit has already validated it), an IPv4 address, or a host name whose ASCII
        (IDNA) form is at most 253 characters of labels that start and end with a letter or digit, with one optional
        trailing dot. Unicode names are checked after encoding, which is what DNS sees."""
        name = host or ""
        if ":" in name:
            return True
        ascii_name = cls._idna(name)
        if ascii_name is None:
            return False
        # A numeric last label is never a TLD, so only a real IPv4 address is usable (not 256.0.0.1 or 1.2.3). Both
        # forms are checked: encoding turns dots such as "。" into ".", and leaves digits such as "١٢٣" non-ASCII.
        if cls._numeric_last_label(name.removesuffix("."), ascii_name):
            return cls._is_ipv4(ascii_name)
        return cls.HOST_NAME_PATTERN.fullmatch(ascii_name) is not None

    @staticmethod
    def _numeric_last_label(*names: str) -> bool:
        return any(name.rsplit(".", 1)[-1].isdigit() for name in names)

    @classmethod
    def _idna(cls, name: str) -> Optional[str]:
        """The ASCII (IDNA 2003, as Python's codec implements it) form of a host name without its trailing dot, or
        None when it is longer than 253 characters or cannot be encoded: the codec rejects labels that are empty or
        over 63 bytes once encoded, and Unicode labels that break the IDNA rules. Encoding also turns dots such as
        "。" into "."."""
        try:
            ascii_name = name.encode("idna").decode("ascii").removesuffix(".")
        except UnicodeError:
            return None
        return ascii_name if len(ascii_name) <= cls.MAX_HOST_NAME_LENGTH else None

    @staticmethod
    def _is_ipv4(name: str) -> bool:
        try:
            ipaddress.IPv4Address(name)
        except ValueError:
            return False
        return True

    @classmethod
    def _usable_url(cls, url: str) -> bool:
        """True for a printable http(s) URL with a valid host and an in-range, non-zero port."""
        if not (url.isprintable() and cls.RESULT_URL_PATTERN.fullmatch(url)):
            return False
        try:
            # urlsplit raises ValueError for an out-of-range port or an invalid IPv6 literal.
            parts = urlsplit(url)
            return parts.port != 0 and cls._valid_host(parts.hostname)
        except ValueError:
            return False

    @classmethod
    def _to_item(cls, hit: object, query: str, max_content_chars: Optional[int] = None) -> Optional[SofyaSearchResultItem]:
        """Normalise one result, or return None when it has no usable http(s) URL.

        The API sets fetched to true when content is the page text; otherwise content repeats the snippet, so it is
        only used as a fallback description."""
        fields: Dict[str, object] = hit if isinstance(hit, dict) else {}
        url = str(fields.get("url"))
        if not cls._usable_url(url):
            return None
        fetched = fields.get("fetched") is True
        content = cls._first_text(fields.get("content")) if fetched else None
        return SofyaSearchResultItem(
            query=query,
            title=cls._first_text(fields.get("title")) or url,
            url=url,
            description=cls._first_text(fields.get("description"), None if fetched else fields.get("content")),
            published=cls._first_text(fields.get("published_date")),
            content=content[:max_content_chars] if content else None,
        )

    def _build_body(self, query: str, params: SofyaSearchToolInputSchema) -> Dict[str, object]:
        body: Dict[str, object] = {
            "query": query,
            "search_depth": self.search_depth,
            "max_results": params.max_results_per_query,
            "topic": params.topic,
        }
        if self.freshness:
            body["freshness"] = self.freshness
        if self.include_domains:
            body["include_domains"] = self.include_domains
        if self.exclude_domains:
            body["exclude_domains"] = self.exclude_domains
        return body

    def _redact(self, text: str) -> str:
        key = self._api_key.get_secret_value()
        if len(key) >= self.MIN_REDACTED_KEY_LENGTH:
            text = re.sub(re.escape(key), "[redacted]", text, flags=re.IGNORECASE)
        return self.KEY_PATTERN.sub("[redacted]", text)

    def _error(self, query: str, detail: str) -> Exception:
        # Redacted after the query is interpolated, since the query text can contain a key too.
        return Exception(self._redact(f"Sofya search failed for '{query}': {detail}"))

    @staticmethod
    async def _read_json(resp: aiohttp.ClientResponse) -> object:
        try:
            return await resp.json(content_type=None)
        except (ValueError, RecursionError):
            # Invalid JSON, or JSON nested too deeply for the parser.
            return None

    @staticmethod
    def _error_detail(payload: Dict[str, object], reason: Optional[str], retry_after: Optional[str]) -> str:
        """The API's error message: `detail` is a string, or a list of validation errors for a 422. A 402 adds where to
        top up, and a 429 says how long to wait."""
        detail = payload.get("detail")
        if isinstance(detail, list):
            detail = "; ".join(str(entry.get("msg")) for entry in detail if isinstance(entry, dict)) or None
        text = str(detail) if detail else str(reason)
        top_up_url = payload.get("top_up_url")
        if isinstance(top_up_url, str) and top_up_url:
            text += f" (top up at {top_up_url})"
        if retry_after:
            text += f" (retry after {retry_after}s)"
        return text

    def _check_response(
        self, status: int, reason: Optional[str], data: object, retry_after: Optional[str] = None
    ) -> Union[Dict[str, object], str]:
        """Return the body of a successful search (a 200 whose body is a JSON object), or why the response is not
        one. Provider text is redacted here, where it enters the tool."""
        payload: Dict[str, object] = data if isinstance(data, dict) else {}
        if status == 200 and payload:
            return payload
        return self._redact(f"{status} {self._error_detail(payload, reason, retry_after)}")

    async def _exchange(
        self, session: aiohttp.ClientSession, query: str, params: SofyaSearchToolInputSchema
    ) -> Union[Dict[str, object], str]:
        """Send one search request and return the body of a successful response, or why it failed.

        Transport errors, timeouts and invalid URLs are turned into a failure reason here instead of propagating,
        because aiohttp's own frames and errors hold the request headers, Authorization included."""
        try:
            async with session.post(
                f"{self.base_url}/search", json=self._build_body(query, params), allow_redirects=False
            ) as resp:
                payload = await self._read_json(resp)
                reason = resp.reason if resp.status != 200 else "response was not a JSON object"
                return self._check_response(resp.status, reason, payload, resp.headers.get("Retry-After"))
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as error:
            return self._redact(f"request failed: {type(error).__name__} {str(error) or f'after {self.timeout:g}s'}")

    async def _post(self, session: aiohttp.ClientSession, query: str, params: SofyaSearchToolInputSchema) -> Dict[str, object]:
        """Return the body of a successful search response, or raise.

        The raise happens in this frame, which only ever holds the redacted failure reason: no response, no aiohttp
        error and no failed body, so a traceback that captures locals cannot show a key echoed back. The caller's
        own inputs (the query and params) are in these frames as given, as they are in the caller's frames."""
        outcome = await self._exchange(session, query, params)
        if isinstance(outcome, str):
            raise self._error(query, outcome)
        return outcome

    def _parse(self, payload: Dict[str, object], query: str) -> Tuple[List[SofyaSearchResultItem], Optional[str]]:
        """Turn a successful response body into result items, or say why it cannot be used. It never raises, so
        the rejected body is not left in the locals of a raising frame."""
        hits = payload.get("results")
        if not isinstance(hits, list):
            return [], "200 unexpected response shape: no results list"
        items = list(filter(None, (self._to_item(hit, query, self.max_content_chars) for hit in hits)))
        if hits and not items:
            return [], f"200 none of the {len(hits)} results had a usable url"
        return items, None

    async def _fetch(
        self,
        session: aiohttp.ClientSession,
        query: str,
        params: SofyaSearchToolInputSchema,
    ) -> List[SofyaSearchResultItem]:
        items, problem = self._parse(await self._post(session, query, params), query)
        if problem:
            raise self._error(query, problem)
        return items[: params.max_results_per_query]

    def _collect(
        self, queries: List[str], grouped: List[Union[List[SofyaSearchResultItem], BaseException]]
    ) -> SofyaSearchToolOutputSchema:
        """Merge per-query results. Raise when no query produced results and at least one failed, so a bad key,
        exhausted credits or a malformed response never looks like an empty search."""
        results: List[SofyaSearchResultItem] = []
        failures: List[SofyaSearchFailure] = []
        errors: List[BaseException] = []
        for query, group in zip(queries, grouped):
            if isinstance(group, BaseException):
                errors.append(group)
                failures.append(SofyaSearchFailure(query=self._redact(query), error=self._redact(str(group))))
                logger.warning("%s", failures[-1].error)
                continue
            results.extend(group)

        if errors and not results:
            raise errors[0]
        return SofyaSearchToolOutputSchema(results=results, failures=failures)

    async def run_async(self, params: SofyaSearchToolInputSchema) -> SofyaSearchToolOutputSchema:
        if not self._api_key.get_secret_value():
            raise ValueError(
                "Sofya API key is missing. Set SofyaSearchToolConfig.api_key or the SOFYA_API_KEY environment variable."
            )

        # Duplicate queries would be billed twice and return the same results.
        queries = list(dict.fromkeys(params.queries))
        # The Authorization header is built inline so no local variable holds the key.
        async with aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self._api_key.get_secret_value()}", "Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        ) as session:
            grouped: List[Union[List[SofyaSearchResultItem], BaseException]] = await asyncio.gather(
                *(self._fetch(session, q, params) for q in queries), return_exceptions=True
            )

        return self._collect(queries, grouped)

    def run(self, params: SofyaSearchToolInputSchema) -> SofyaSearchToolOutputSchema:
        with ThreadPoolExecutor() as executor:
            return executor.submit(contextvars.copy_context().run, asyncio.run, self.run_async(params)).result()


#################
# EXAMPLE USAGE #
#################
if __name__ == "__main__":  # pragma: no cover
    from dotenv import load_dotenv
    from rich.console import Console

    load_dotenv()
    console = Console()
    tool = SofyaSearchTool(config=SofyaSearchToolConfig(api_key=SecretStr(os.getenv("SOFYA_API_KEY", ""))))

    examples: List[Tuple[Literal["general", "news"], str]] = [
        ("general", "atomic agents framework"),
        ("news", "AI agents"),
    ]
    for topic, query in examples:
        output = tool.run(SofyaSearchToolInputSchema(queries=[query], topic=topic, max_results_per_query=3))
        console.rule(f"[bold cyan]{topic}")
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
