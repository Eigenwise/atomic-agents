import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Literal, Optional

import aiohttp
from pydantic import Field

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


class FirecrawlSearchToolOutputSchema(BaseIOSchema):
    """Output of the Firecrawl search tool."""

    results: List[FirecrawlSearchResultItem] = Field(..., description="Matching results across all queries.")


#################
# CONFIGURATION #
#################
class FirecrawlSearchToolConfig(BaseToolConfig):
    """Configuration for the FirecrawlSearchTool."""

    api_key: str = Field(
        default="", description="Firecrawl API key. Falls back to the FIRECRAWL_API_KEY environment variable."
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

    def __init__(self, config: FirecrawlSearchToolConfig = FirecrawlSearchToolConfig()):
        super().__init__(config)
        self.api_key = config.api_key or os.getenv("FIRECRAWL_API_KEY", "")
        self.base_url = config.base_url.rstrip("/")
        self.location = config.location
        self.time_range = config.time_range
        self.max_content_chars = config.max_content_chars
        self.timeout = config.timeout

    @classmethod
    def _to_item(cls, hit: dict, query: str, max_content_chars: Optional[int] = None) -> FirecrawlSearchResultItem:
        metadata = hit.get("metadata") if isinstance(hit.get("metadata"), dict) else {}
        content = hit.get("markdown")
        if content and max_content_chars:
            content = content[:max_content_chars]
        return FirecrawlSearchResultItem(
            query=query,
            title=hit.get("title") or metadata.get("title") or hit["url"],
            url=hit["url"],
            description=hit.get("description") or hit.get("snippet") or metadata.get("description"),
            position=hit.get("position"),
            published=hit.get("date"),
            content=content,
        )

    def _build_body(self, query: str, params: FirecrawlSearchToolInputSchema) -> dict:
        body = {
            "query": query,
            "limit": params.max_results_per_query,
            "sources": [params.search_type],
            "origin": "atomic-agents",
        }
        if self.location:
            body["location"] = self.location
        if self.time_range and params.search_type == "web":
            body["tbs"] = self.TIME_RANGES[self.time_range]
        if params.include_content:
            body["scrapeOptions"] = {"formats": ["markdown"], "onlyMainContent": True}
        return body

    async def _fetch(
        self,
        session: aiohttp.ClientSession,
        query: str,
        params: FirecrawlSearchToolInputSchema,
    ) -> List[FirecrawlSearchResultItem]:
        async with session.post(f"{self.base_url}/search", json=self._build_body(query, params)) as resp:
            try:
                data = await resp.json(content_type=None)
            except ValueError:
                data = None
            if resp.status != 200 or not isinstance(data, dict) or not data.get("success"):
                error = data.get("error") if isinstance(data, dict) else None
                raise Exception(f"Firecrawl search failed for '{query}': {resp.status} {error or resp.reason}")

        # The key for the requested source is absent when there are no results.
        groups = data.get("data") if isinstance(data.get("data"), dict) else {}
        hits = groups.get(params.search_type) or []
        items = [self._to_item(hit, query, self.max_content_chars) for hit in hits if isinstance(hit, dict) and hit.get("url")]
        return items[: params.max_results_per_query]

    async def run_async(self, params: FirecrawlSearchToolInputSchema) -> FirecrawlSearchToolOutputSchema:
        if not self.api_key:
            raise ValueError(
                "Firecrawl API key is missing. Set FirecrawlSearchToolConfig.api_key or the FIRECRAWL_API_KEY "
                "environment variable."
            )

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            tasks = [self._fetch(session, q, params) for q in params.queries]
            grouped = await asyncio.gather(*tasks, return_exceptions=True)

        # A bad key or exhausted credits fails every query; raise instead of returning an empty list.
        failures = [group for group in grouped if isinstance(group, Exception)]
        if failures and len(failures) == len(grouped):
            raise failures[0]

        results: List[FirecrawlSearchResultItem] = []
        for query, group in zip(params.queries, grouped):
            if isinstance(group, Exception):
                logger.warning("Firecrawl query '%s' failed: %s", query, str(group) or repr(group))
                continue
            results.extend(group)
        return FirecrawlSearchToolOutputSchema(results=results)

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
