import asyncio
import contextvars
import ipaddress
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Literal, Optional, Tuple, Union
from urllib.parse import urlsplit

import aiohttp
from pydantic import Field, SecretStr, field_validator, model_validator

from atomic_agents import BaseIOSchema, BaseTool, BaseToolConfig


logger = logging.getLogger(__name__)


################
# INPUT SCHEMA #
################
class SpecstreamSearchToolInputSchema(BaseIOSchema):
    """
    Search the web or recent news through the SpecStream Search API.
    Use this to find current information on a topic, with sourced direct answers on the default tier
    and optional page excerpts when include_passages is true. Requires a SpecStream API key.
    """

    queries: List[str] = Field(
        ...,
        min_length=1,
        description=(
            "Search queries to run (1 to 2,000 characters each). "
            'Supports operators such as site:, -site:, -word, -"phrase", "phrase", filetype:, after:, and before:.'
        ),
    )
    topic: Literal["web", "news"] = Field(
        default="web",
        description="Which results to return. 'web' = web search results; 'news' = recent news articles.",
    )
    max_results_per_query: int = Field(default=10, ge=1, le=100, description="Maximum results per query.")
    include_passages: bool = Field(
        default=False,
        description="When true on web searches, also return page text excerpts closest to the query in content.",
    )

    @field_validator("queries")
    @classmethod
    def _valid_queries(cls, queries: List[str]) -> List[str]:
        for query in queries:
            if not query.strip() or len(query) > 2000:
                raise ValueError("each query must be non-empty and at most 2000 characters")
        return queries

    @model_validator(mode="after")
    def _no_passages_with_news(self) -> "SpecstreamSearchToolInputSchema":
        if self.topic == "news" and self.include_passages:
            raise ValueError("include_passages cannot be used when topic is 'news'")
        return self


####################
# OUTPUT SCHEMA(S) #
####################
class SpecstreamSearchResultItem(BaseIOSchema):
    """A single SpecStream search result (web page or news article)."""

    query: str = Field(..., description="The query that produced this result.")
    title: str = Field(..., description="Title of the result.")
    url: str = Field(..., description="URL of the result.")
    description: Optional[str] = Field(default=None, description="Snippet for the result.")
    site: Optional[str] = Field(default=None, description="Host or domain name of the result.")
    published: Optional[str] = Field(default=None, description="Publication date as YYYY-MM-DD, when known.")
    source: Optional[str] = Field(default=None, description="Publisher name (news results only).")
    cited: Optional[bool] = Field(default=None, description="Whether this source was cited by the extracted answer or rows.")
    content: Optional[str] = Field(
        default=None, description="Page text excerpt closest to the query (when include_passages is True)."
    )


class SpecstreamSearchAnswer(BaseIOSchema):
    """A direct answer extracted from search results for a query."""

    query: str = Field(..., description="The query that produced this answer.")
    value: str = Field(..., description="Answer value as printed on the cited page.")
    page: str = Field(..., description="URL of the page the answer was read from.")
    seen_at: Optional[str] = Field(default=None, description="ISO 8601 timestamp when the cited page was read.")


class SpecstreamSearchFailure(BaseIOSchema):
    """A query that failed while other queries returned results."""

    query: str = Field(..., description="The query that failed.")
    error: str = Field(..., description="Why it failed, e.g. an HTTP status and the API's error message.")


class SpecstreamSearchToolOutputSchema(BaseIOSchema):
    """Output of the SpecStream search tool."""

    results: List[SpecstreamSearchResultItem] = Field(..., description="Matching results across all queries.")
    answers: List[SpecstreamSearchAnswer] = Field(
        default_factory=list, description="Direct answers extracted for queries when available."
    )
    failures: List[SpecstreamSearchFailure] = Field(
        default_factory=list, description="Queries that failed while others returned results, with the reason."
    )


#################
# CONFIGURATION #
#################
class SpecstreamSearchToolConfig(BaseToolConfig):
    """Configuration for the SpecstreamSearchTool."""

    api_key: SecretStr = Field(
        default=SecretStr(""),
        description="SpecStream API key. Falls back to the SPECSTREAM_API_KEY environment variable. Hidden in repr.",
    )
    base_url: str = Field(default="https://api.specstream.ai/v1", description="SpecStream API base URL.")
    speed: Literal["default", "fast"] = Field(
        default="default",
        description="'default' returns a sourced answer and results; 'fast' returns search results and snippets faster.",
    )
    language: Optional[str] = Field(
        default=None, pattern=r"^[A-Za-z]{2}$", description="Two-letter ISO 639-1 language code, e.g. 'en' or 'de'."
    )
    country: Optional[str] = Field(
        default=None, pattern=r"^[A-Za-z]{2}$", description="Two-letter ISO 3166-1 country code, e.g. 'US' or 'DE'."
    )
    safe: Optional[Literal["off", "moderate", "strict"]] = Field(
        default=None, description="Safe search level: 'off', 'moderate', or 'strict'."
    )
    engines: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Comma-separated engine names or country filters, e.g. '-CN' or 'bing,duckduckgo'.",
    )
    pages_wait_ms: Optional[int] = Field(
        default=None,
        ge=0,
        le=3500,
        description="With include_passages=True, milliseconds to wait for page text (0-3500).",
    )
    max_content_chars: Optional[int] = Field(
        default=None, ge=1, description="Truncate each result's content to this many characters. No limit when unset."
    )
    timeout: float = Field(default=30.0, ge=1.0, le=120.0, description="HTTP request timeout in seconds.")


#####################
# MAIN TOOL & LOGIC #
#####################
class SpecstreamSearchTool(BaseTool[SpecstreamSearchToolInputSchema, SpecstreamSearchToolOutputSchema]):
    """Tool for searching the web and news via the SpecStream Search API."""

    # Anything in a SpecStream key format is redacted from error and log text, plus the configured key itself when it
    # is long enough that replacing it cannot mangle ordinary words.
    KEY_PATTERN = re.compile(r"\b(?:ss|specstream)_(?:live|test)_[0-9a-z_-]{16,}\b", re.IGNORECASE)
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

    def __init__(self, config: SpecstreamSearchToolConfig = SpecstreamSearchToolConfig()):
        super().__init__(config)
        self._api_key = SecretStr(config.api_key.get_secret_value() or os.getenv("SPECSTREAM_API_KEY", ""))
        self.base_url = config.base_url.rstrip("/")
        self.speed = config.speed
        self.language = config.language
        self.country = config.country
        self.safe = config.safe
        self.engines = config.engines
        self.pages_wait_ms = config.pages_wait_ms
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
    def _to_item(
        cls,
        hit: object,
        query: str,
        max_content_chars: Optional[int] = None,
        passage_by_url: Optional[Dict[str, str]] = None,
    ) -> Optional[SpecstreamSearchResultItem]:
        """Normalise one source, or return None when it has no usable http(s) URL."""
        fields: Dict[str, object] = hit if isinstance(hit, dict) else {}
        url = str(fields.get("url"))
        if not cls._usable_url(url):
            return None
        cited = fields.get("cited") if isinstance(fields.get("cited"), bool) else None
        content = cls._first_text(fields.get("excerpt"), (passage_by_url or {}).get(url))
        return SpecstreamSearchResultItem(
            query=query,
            title=cls._first_text(fields.get("title")) or url,
            url=url,
            description=cls._first_text(fields.get("snippet")),
            site=cls._first_text(fields.get("site")),
            published=cls._first_text(fields.get("published")),
            source=cls._first_text(fields.get("source")),
            cited=cited,
            content=content[:max_content_chars] if content else None,
        )

    @classmethod
    def _to_answer(cls, raw_answer: object, query: str) -> Optional[SpecstreamSearchAnswer]:
        if not isinstance(raw_answer, dict):
            return None
        value = cls._first_text(raw_answer.get("value"))
        page = str(raw_answer.get("page"))
        if not value or not cls._usable_url(page):
            return None
        return SpecstreamSearchAnswer(
            query=query,
            value=value,
            page=page,
            seen_at=cls._first_text(raw_answer.get("seen_at")),
        )

    def _build_body(self, query: str, params: SpecstreamSearchToolInputSchema) -> Dict[str, object]:
        body: Dict[str, object] = {
            "q": query,
            "num": params.max_results_per_query,
            "speed": self.speed,
        }
        if params.topic == "news":
            body["news"] = True
        if params.include_passages:
            body["passages"] = True
            if self.pages_wait_ms is not None:
                body["pages_wait_ms"] = self.pages_wait_ms
        if self.language:
            body["language"] = self.language
        if self.country:
            body["country"] = self.country
        if self.safe:
            body["safe"] = self.safe
        if self.engines:
            body["engines"] = self.engines
        return body

    def _redact(self, text: str) -> str:
        key = self._api_key.get_secret_value()
        if len(key) >= self.MIN_REDACTED_KEY_LENGTH:
            text = re.sub(re.escape(key), "[redacted]", text, flags=re.IGNORECASE)
        return self.KEY_PATTERN.sub("[redacted]", text)

    def _error(self, query: str, detail: str) -> Exception:
        # Redacted after the query is interpolated, since the query text can contain a key too.
        return Exception(self._redact(f"SpecStream search failed for '{query}': {detail}"))

    @staticmethod
    async def _read_json(resp: aiohttp.ClientResponse) -> object:
        try:
            return await resp.json(content_type=None)
        except (ValueError, RecursionError):
            # Invalid JSON, or JSON nested too deeply for the parser.
            return None

    @staticmethod
    def _error_detail(payload: Dict[str, object], reason: Optional[str], retry_after: Optional[str]) -> str:
        """Format SpecStream's ErrorEnvelope (`error.code` and `error.message`), or FastAPI's `detail` fallback."""
        err = payload.get("error")
        text: Optional[str] = None
        if isinstance(err, dict):
            msg = err.get("message")
            code = err.get("code")
            if isinstance(msg, str) and msg:
                text = f"{code}: {msg}" if isinstance(code, str) and code else msg
        if not text:
            detail = payload.get("detail")
            if isinstance(detail, list):
                detail = "; ".join(str(entry.get("msg")) for entry in detail if isinstance(entry, dict)) or None
            text = str(detail) if detail else str(reason)
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
        self, session: aiohttp.ClientSession, query: str, params: SpecstreamSearchToolInputSchema
    ) -> Union[Dict[str, object], str]:
        """Send one search request and return the body of a successful response, or why it failed.

        Transport errors, timeouts and invalid URLs are turned into a failure reason here instead of propagating,
        because aiohttp's own frames and errors hold the request headers, Authorization included."""
        try:
            async with session.post(
                f"{self.base_url}/web-search", json=self._build_body(query, params), allow_redirects=False
            ) as resp:
                payload = await self._read_json(resp)
                reason = resp.reason if resp.status != 200 else "response was not a JSON object"
                return self._check_response(resp.status, reason, payload, resp.headers.get("Retry-After"))
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as error:
            return self._redact(f"request failed: {type(error).__name__} {str(error) or f'after {self.timeout:g}s'}")

    async def _post(
        self, session: aiohttp.ClientSession, query: str, params: SpecstreamSearchToolInputSchema
    ) -> Dict[str, object]:
        """Return the body of a successful search response, or raise.

        The raise happens in this frame, which only ever holds the redacted failure reason: no response, no aiohttp
        error and no failed body, so a traceback that captures locals cannot show a key echoed back."""
        outcome = await self._exchange(session, query, params)
        if isinstance(outcome, str):
            raise self._error(query, outcome)
        return outcome

    @classmethod
    def _passage_map(cls, data: Dict[str, object]) -> Dict[str, str]:
        raw_passages = data.get("passages")
        passage_by_url: Dict[str, str] = {}
        if isinstance(raw_passages, list):
            for entry in raw_passages:
                if isinstance(entry, dict):
                    p_url = cls._first_text(entry.get("url"))
                    p_text = cls._first_text(entry.get("text"))
                    if p_url and p_text and p_url not in passage_by_url:
                        passage_by_url[p_url] = p_text
        return passage_by_url

    def _parse(
        self, payload: Dict[str, object], query: str
    ) -> Tuple[List[SpecstreamSearchResultItem], Optional[SpecstreamSearchAnswer], Optional[str]]:
        """Turn a successful response body into result items and optional answer, or say why it cannot be used.
        Never raises, so the rejected body is not left in the locals of a raising frame."""
        data = payload.get("data")
        if not isinstance(data, dict):
            return [], None, "200 unexpected response shape: no data object"
        hits = data.get("sources")
        if not isinstance(hits, list):
            return [], None, "200 unexpected response shape: no sources list"
        passage_by_url = self._passage_map(data)
        items = list(filter(None, (self._to_item(hit, query, self.max_content_chars, passage_by_url) for hit in hits)))
        if hits and not items:
            return [], None, f"200 none of the {len(hits)} results had a usable url"
        answer = self._to_answer(data.get("answer"), query)
        return items, answer, None

    async def _fetch(
        self,
        session: aiohttp.ClientSession,
        query: str,
        params: SpecstreamSearchToolInputSchema,
    ) -> Tuple[List[SpecstreamSearchResultItem], Optional[SpecstreamSearchAnswer]]:
        items, answer, problem = self._parse(await self._post(session, query, params), query)
        if problem:
            raise self._error(query, problem)
        return items[: params.max_results_per_query], answer

    def _collect(
        self,
        queries: List[str],
        grouped: List[Union[Tuple[List[SpecstreamSearchResultItem], Optional[SpecstreamSearchAnswer]], BaseException]],
    ) -> SpecstreamSearchToolOutputSchema:
        """Merge per-query results. Raise when no query produced results and at least one failed, so a bad key,
        exhausted balance or a malformed response never looks like an empty search."""
        results: List[SpecstreamSearchResultItem] = []
        answers: List[SpecstreamSearchAnswer] = []
        failures: List[SpecstreamSearchFailure] = []
        errors: List[BaseException] = []
        for query, group in zip(queries, grouped):
            if isinstance(group, BaseException):
                errors.append(group)
                failures.append(SpecstreamSearchFailure(query=self._redact(query), error=self._redact(str(group))))
                logger.warning("%s", failures[-1].error)
                continue
            items, answer = group
            results.extend(items)
            if answer is not None:
                answers.append(answer)

        if errors and not results:
            raise errors[0]
        return SpecstreamSearchToolOutputSchema(results=results, answers=answers, failures=failures)

    async def run_async(self, params: SpecstreamSearchToolInputSchema) -> SpecstreamSearchToolOutputSchema:
        if not self._api_key.get_secret_value():
            raise ValueError(
                "SpecStream API key is missing. Set SpecstreamSearchToolConfig.api_key or the SPECSTREAM_API_KEY "
                "environment variable."
            )

        # Duplicate queries would be billed twice and return the same results.
        queries = list(dict.fromkeys(params.queries))
        # The Authorization header is built inline so no local variable holds the key.
        async with aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self._api_key.get_secret_value()}", "Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        ) as session:
            grouped: List[Union[Tuple[List[SpecstreamSearchResultItem], Optional[SpecstreamSearchAnswer]], BaseException]] = (
                await asyncio.gather(*(self._fetch(session, q, params) for q in queries), return_exceptions=True)
            )

        return self._collect(queries, grouped)

    def run(self, params: SpecstreamSearchToolInputSchema) -> SpecstreamSearchToolOutputSchema:
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
    tool = SpecstreamSearchTool(config=SpecstreamSearchToolConfig(api_key=SecretStr(os.getenv("SPECSTREAM_API_KEY", ""))))

    examples: List[Tuple[Literal["web", "news"], str]] = [
        ("web", "atomic agents framework"),
        ("news", "AI agents"),
    ]
    for topic, query in examples:
        output = tool.run(SpecstreamSearchToolInputSchema(queries=[query], topic=topic, max_results_per_query=3))
        console.rule(f"[bold cyan]{topic}")
        for answer in output.answers:
            console.print(f"[bold green]Answer:[/bold green] {answer.value} ({answer.page})")
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
