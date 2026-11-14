import asyncio
import contextvars
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import aiohttp
from pydantic import Field

from atomic_agents import BaseIOSchema, BaseTool, BaseToolConfig


logger = logging.getLogger(__name__)


################
# INPUT SCHEMA #
################
class OutcrawlSearchToolInputSchema(BaseIOSchema):
    """
    Search the web through the Outcrawl API, optionally fetching each result page as Markdown.
    Use this to find current information on a topic and, when needed, read the full pages.
    Requires an Outcrawl API key.
    """

    queries: List[str] = Field(..., description="Search queries to run.")
    max_results_per_query: int = Field(default=10, ge=1, le=20, description="Maximum results per query.")
    scrape_content: bool = Field(
        default=False,
        description="When true, each result page is also fetched and returned as Markdown in `content`.",
    )


####################
# OUTPUT SCHEMA(S) #
####################
class OutcrawlSearchResultItem(BaseIOSchema):
    """A single Outcrawl search result."""

    query: str = Field(..., description="The query that produced this result.")
    title: str = Field(..., description="Title of the result.")
    url: str = Field(..., description="URL of the result.")
    snippet: Optional[str] = Field(None, description="Snippet text from the results page.")
    rank: Optional[int] = Field(None, description="Rank of the result on the results page.")
    content: Optional[str] = Field(None, description="Main page content as Markdown (only when scrape_content is set).")


class OutcrawlSearchToolOutputSchema(BaseIOSchema):
    """Output of the Outcrawl search tool."""

    results: List[OutcrawlSearchResultItem] = Field(..., description="Matching results across all queries.")


#################
# CONFIGURATION #
#################
class OutcrawlSearchToolConfig(BaseToolConfig):
    """Configuration for the OutcrawlSearchTool."""

    api_key: str = Field(default="", description="Outcrawl API key. Falls back to the OUTCRAWL_API_KEY environment variable.")
    base_url: str = Field(default="https://api.outcrawl.ai/v1", description="Outcrawl API base URL.")
    user_agent: str = Field(
        default="atomic-agents-outcrawl-tool/1.0 (+https://github.com/eigenwise/atomic-agents)",
        description="User agent for HTTP requests.",
    )
    timeout: float = Field(default=60.0, ge=1.0, le=300.0, description="HTTP request timeout in seconds.")


#####################
# MAIN TOOL & LOGIC #
#####################
class OutcrawlSearchTool(BaseTool[OutcrawlSearchToolInputSchema, OutcrawlSearchToolOutputSchema]):
    """Tool for searching the web, and optionally reading result pages, via the Outcrawl API."""

    def __init__(self, config: OutcrawlSearchToolConfig = OutcrawlSearchToolConfig()):
        super().__init__(config)
        self.api_key = config.api_key or os.getenv("OUTCRAWL_API_KEY", "")
        self.base_url = config.base_url.rstrip("/")
        self.user_agent = config.user_agent
        self.timeout = config.timeout

    @staticmethod
    def _to_item(hit: dict, query: str) -> OutcrawlSearchResultItem:
        return OutcrawlSearchResultItem(
            query=query,
            title=hit["title"],
            url=hit["url"],
            snippet=hit.get("snippet") or None,
            rank=hit.get("rank"),
        )

    @staticmethod
    async def _post(session: aiohttp.ClientSession, url: str, payload: dict, what: str) -> dict:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                detail = (await resp.text())[:200]
                raise Exception(f"Outcrawl {what}: {resp.status} {resp.reason} {detail}".rstrip())
            return await resp.json()

    async def _scrape(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        payload = {"url": url, "formats": ["markdown"], "onlyMainContent": True}
        data = await self._post(session, f"{self.base_url}/scrape", payload, f"scrape failed for '{url}'")
        return data.get("markdown") or None

    async def _fetch(
        self,
        session: aiohttp.ClientSession,
        query: str,
        max_results: int,
        scrape_content: bool,
    ) -> List[OutcrawlSearchResultItem]:
        payload = {"query": query, "limit": max_results}
        data = await self._post(session, f"{self.base_url}/search", payload, f"search failed for '{query}'")

        hits = data.get("hits") or []
        items = [self._to_item(hit, query) for hit in hits if hit.get("title") and hit.get("url")][:max_results]

        if scrape_content and items:
            pages = await asyncio.gather(*(self._scrape(session, item.url) for item in items), return_exceptions=True)
            for item, page in zip(items, pages):
                if isinstance(page, Exception):
                    logger.warning("Outcrawl scrape of '%s' failed: %s", item.url, page)
                    continue
                item.content = page
        return items

    async def run_async(self, params: OutcrawlSearchToolInputSchema) -> OutcrawlSearchToolOutputSchema:
        if not self.api_key:
            raise ValueError(
                "Outcrawl API key is missing. Set OutcrawlSearchToolConfig.api_key or the OUTCRAWL_API_KEY environment variable."
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": self.user_agent,
            "Accept": "application/json",
        }
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            tasks = [self._fetch(session, q, params.max_results_per_query, params.scrape_content) for q in params.queries]
            grouped = await asyncio.gather(*tasks, return_exceptions=True)

        results: List[OutcrawlSearchResultItem] = []
        for query, group in zip(params.queries, grouped):
            if isinstance(group, Exception):
                logger.warning("Outcrawl query '%s' failed: %s", query, group)
                continue
            results.extend(group)
        return OutcrawlSearchToolOutputSchema(results=results)

    def run(self, params: OutcrawlSearchToolInputSchema) -> OutcrawlSearchToolOutputSchema:
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
    tool = OutcrawlSearchTool(config=OutcrawlSearchToolConfig(api_key=os.getenv("OUTCRAWL_API_KEY", "")))

    output = tool.run(
        OutcrawlSearchToolInputSchema(
            queries=["atomic agents framework"],
            max_results_per_query=3,
            scrape_content=True,
        )
    )
    for item in output.results:
        console.print(f"[bold]{item.title}[/bold]")
        console.print(item.url)
        if item.snippet:
            console.print(item.snippet[:200])
        if item.content:
            console.print(f"[bold]Content:[/bold] {item.content[:200]}")
        console.print()
