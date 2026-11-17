import asyncio
import contextvars
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
class SerplySearchToolInputSchema(BaseIOSchema):
    """
    Search Google web results, Google News, or Google Scholar through the Serply API.
    Use this to find current information, news coverage, or academic papers on a topic.
    Requires a Serply API key.
    """

    queries: List[str] = Field(..., description="Search queries to run.")
    search_type: Literal["search", "news", "scholar"] = Field(
        default="search",
        description=(
            "Which Serply endpoint to query. 'search' = Google web results; 'news' = Google News"
            " articles; 'scholar' = Google Scholar papers."
        ),
    )
    max_results_per_query: int = Field(default=10, ge=1, le=100, description="Maximum results per query.")


####################
# OUTPUT SCHEMA(S) #
####################
class SerplySearchResultItem(BaseIOSchema):
    """A single Serply search result (web page, news article, or paper)."""

    query: str = Field(..., description="The query that produced this result.")
    title: str = Field(..., description="Title of the result.")
    url: str = Field(..., description="URL of the result.")
    description: Optional[str] = Field(None, description="Snippet, summary, or abstract text.")
    position: Optional[int] = Field(None, description="Rank of the result on the results page (web search only).")
    published: Optional[str] = Field(None, description="Publication date string (news only).")
    source: Optional[str] = Field(None, description="Publisher name (news only).")
    authors: Optional[str] = Field(None, description="Comma-separated author names (scholar only).")
    citations: Optional[int] = Field(None, description="Citation count (scholar only).")


class SerplySearchToolOutputSchema(BaseIOSchema):
    """Output of the Serply search tool."""

    results: List[SerplySearchResultItem] = Field(..., description="Matching results across all queries.")


#################
# CONFIGURATION #
#################
class SerplySearchToolConfig(BaseToolConfig):
    """Configuration for the SerplySearchTool."""

    api_key: str = Field(default="", description="Serply API key. Falls back to the SERPLY_API_KEY environment variable.")
    base_url: str = Field(default="https://api.serply.io/v1", description="Serply API base URL.")
    hl: str = Field(default="en", description="Interface language code passed to Google, e.g. 'en' or 'de'.")
    gl: str = Field(default="us", description="Country code for the search, e.g. 'us' or 'gb'.")
    user_agent: str = Field(
        default="atomic-agents-serply-tool/1.0 (+https://github.com/eigenwise/atomic-agents)",
        description="User agent for HTTP requests.",
    )
    timeout: float = Field(default=30.0, ge=1.0, le=120.0, description="HTTP request timeout in seconds.")


#####################
# MAIN TOOL & LOGIC #
#####################
class SerplySearchTool(BaseTool[SerplySearchToolInputSchema, SerplySearchToolOutputSchema]):
    """Tool for searching Google web, news, and scholar results via the Serply API."""

    RESULT_KEYS = {"search": "results", "news": "entries", "scholar": "articles"}

    def __init__(self, config: SerplySearchToolConfig = SerplySearchToolConfig()):
        super().__init__(config)
        self.api_key = config.api_key or os.getenv("SERPLY_API_KEY", "")
        self.base_url = config.base_url.rstrip("/")
        self.hl = config.hl
        self.gl = config.gl
        self.user_agent = config.user_agent
        self.timeout = config.timeout

    @classmethod
    def _to_item(cls, hit: dict, query: str) -> SerplySearchResultItem:
        source = hit.get("source")
        if isinstance(source, dict):
            source = source.get("title") or source.get("href")

        author = hit.get("author")
        authors = author.get("names") if isinstance(author, dict) else author

        citations = None
        extras = hit.get("extras")
        if isinstance(extras, dict) and isinstance(extras.get("citations"), dict):
            citations = extras["citations"].get("count")

        return SerplySearchResultItem(
            query=query,
            title=hit["title"],
            url=hit["link"],
            description=hit.get("description") or hit.get("summary"),
            position=hit.get("position"),
            published=hit.get("published"),
            source=str(source) if source else None,
            authors=str(authors) if authors else None,
            citations=citations,
        )

    async def _fetch(
        self,
        session: aiohttp.ClientSession,
        query: str,
        search_type: str,
        max_results: int,
    ) -> List[SerplySearchResultItem]:
        params = {"q": query, "num": str(max_results), "hl": self.hl, "gl": self.gl}

        async with session.get(f"{self.base_url}/{search_type}/", params=params, allow_redirects=False) as resp:
            if resp.status != 200:
                raise Exception(f"Serply {search_type} failed for '{query}': {resp.status} {resp.reason}")
            data = await resp.json()

        hits = data.get(self.RESULT_KEYS[search_type], [])
        items = [self._to_item(hit, query) for hit in hits if hit.get("title") and hit.get("link")]
        return items[:max_results]

    async def run_async(self, params: SerplySearchToolInputSchema) -> SerplySearchToolOutputSchema:
        if not self.api_key:
            raise ValueError(
                "Serply API key is missing. Set SerplySearchToolConfig.api_key or the SERPLY_API_KEY environment variable."
            )

        headers = {"X-Api-Key": self.api_key, "User-Agent": self.user_agent, "Accept": "application/json"}
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            tasks = [self._fetch(session, q, params.search_type, params.max_results_per_query) for q in params.queries]
            grouped = await asyncio.gather(*tasks, return_exceptions=True)

        results: List[SerplySearchResultItem] = []
        for query, group in zip(params.queries, grouped):
            if isinstance(group, Exception):
                logger.warning("Serply query '%s' failed: %s", query, group)
                continue
            results.extend(group)
        return SerplySearchToolOutputSchema(results=results)

    def run(self, params: SerplySearchToolInputSchema) -> SerplySearchToolOutputSchema:
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
    tool = SerplySearchTool(config=SerplySearchToolConfig(api_key=os.getenv("SERPLY_API_KEY", "")))

    for search_type in ("search", "news", "scholar"):
        output = tool.run(
            SerplySearchToolInputSchema(
                queries=["atomic agents framework"],
                search_type=search_type,
                max_results_per_query=3,
            )
        )
        console.rule(f"[bold cyan]{search_type}")
        for item in output.results:
            console.print(f"[bold]{item.title}[/bold]")
            console.print(item.url)
            if item.description:
                console.print(item.description[:200])
            if item.published:
                console.print(f"[bold]Published:[/bold] {item.published}  [bold]Source:[/bold] {item.source}")
            if item.authors:
                console.print(f"[bold]Authors:[/bold] {item.authors[:120]}  [bold]Citations:[/bold] {item.citations}")
            console.print()
