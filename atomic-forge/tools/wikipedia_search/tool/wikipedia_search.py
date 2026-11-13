import asyncio
import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import aiohttp
from pydantic import Field

from atomic_agents import BaseIOSchema, BaseTool, BaseToolConfig


logger = logging.getLogger(__name__)


################
# INPUT SCHEMA #
################
class WikipediaSearchToolInputSchema(BaseIOSchema):
    """
    Schema for searching Wikipedia. Returns matching articles with titles, URLs,
    and plain-text summaries. Use this whenever you need encyclopedic background on a
    person, place, concept, or event. Free public API; no key required.
    """

    queries: List[str] = Field(..., description="One or more search queries to look up on Wikipedia.")
    language: str = Field(
        default="en",
        description="Wikipedia language edition code (e.g. 'en', 'fr', 'de', 'nl', 'ja').",
    )
    max_results_per_query: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum number of articles to return per query.",
    )
    full_text: bool = Field(
        default=False,
        description=(
            "When true, fetch the full plain-text extract for the top returned result of each"
            " query in addition to the short summary. Costs an extra API call per"
            " query, so leave this off unless you need full article content."
        ),
    )


####################
# OUTPUT SCHEMA(S) #
####################
class WikipediaArticle(BaseIOSchema):
    """A single Wikipedia article result."""

    query: str = Field(..., description="The query that produced this result.")
    title: str = Field(..., description="The article title.")
    page_url: str = Field(..., description="Canonical URL of the article on Wikipedia.")
    summary: str = Field(..., description="Plain-text lead summary of the article.")
    description: Optional[str] = Field(None, description="Short description (one-liner) for the article, when available.")
    full_text: Optional[str] = Field(
        None, description="Full plain-text article content, populated only when `full_text` was requested."
    )
    thumbnail_url: Optional[str] = Field(None, description="URL of a thumbnail image for the article, when available.")


class WikipediaSearchToolOutputSchema(BaseIOSchema):
    """Output of the Wikipedia search tool."""

    results: List[WikipediaArticle] = Field(
        ..., description="Articles returned across all queries, in best-match order per query."
    )


#################
# CONFIGURATION #
#################
class WikipediaSearchToolConfig(BaseToolConfig):
    """Configuration for the WikipediaSearchTool."""

    user_agent: str = Field(
        default="atomic-agents-wikipedia-tool/1.0 (+https://github.com/eigenwise/atomic-agents)",
        description="User agent for Wikipedia API requests. Wikipedia asks for a descriptive UA.",
    )
    timeout: float = Field(default=15.0, ge=1.0, le=120.0, description="HTTP request timeout in seconds.")


#####################
# MAIN TOOL & LOGIC #
#####################
class WikipediaSearchTool(BaseTool[WikipediaSearchToolInputSchema, WikipediaSearchToolOutputSchema]):
    """Tool for searching Wikipedia articles via the public MediaWiki and REST APIs."""

    def __init__(self, config: WikipediaSearchToolConfig = WikipediaSearchToolConfig()):
        super().__init__(config)
        self.user_agent = config.user_agent
        self.timeout = config.timeout

    @staticmethod
    def _api_base(language: str) -> str:
        return f"https://{language}.wikipedia.org"

    async def _search_titles(self, session: aiohttp.ClientSession, query: str, language: str, limit: int) -> List[str]:
        """Call MediaWiki search API and return a list of matching article titles."""
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": str(limit),
            "format": "json",
            "utf8": "",
        }
        async with session.get(f"{self._api_base(language)}/w/api.php", params=params) as resp:
            if resp.status != 200:
                raise Exception(f"Wikipedia search failed for '{query}': {resp.status} {resp.reason}")
            data = await resp.json()
        return [hit["title"] for hit in data.get("query", {}).get("search", [])]

    async def _fetch_summary(self, session: aiohttp.ClientSession, title: str, language: str) -> Optional[dict]:
        """Fetch the page summary via the REST API. Returns None if not found."""
        url = f"{self._api_base(language)}/api/rest_v1/page/summary/{aiohttp.helpers.quote(title, safe='')}"
        async with session.get(url) as resp:
            if resp.status == 404:
                return None
            if resp.status != 200:
                raise Exception(f"Wikipedia summary failed for '{title}': {resp.status} {resp.reason}")
            return await resp.json()

    async def _fetch_full_extract(self, session: aiohttp.ClientSession, title: str, language: str) -> Optional[str]:
        """Fetch the full plain-text extract for an article."""
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": "true",
            "titles": title,
            "format": "json",
            "redirects": "1",
            "utf8": "",
        }
        async with session.get(f"{self._api_base(language)}/w/api.php", params=params) as resp:
            if resp.status != 200:
                raise Exception(f"Wikipedia extract failed for '{title}': {resp.status} {resp.reason}")
            data = await resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract")
            if extract:
                return extract
        return None

    async def _process_query(
        self,
        session: aiohttp.ClientSession,
        query: str,
        language: str,
        limit: int,
        full_text: bool,
    ) -> List[WikipediaArticle]:
        titles = await self._search_titles(session, query, language, limit)
        articles: List[WikipediaArticle] = []
        for title in titles:
            try:
                summary_data = await self._fetch_summary(session, title, language)
            except Exception as e:
                logger.warning("Failed to fetch summary for '%s': %s", title, e)
                continue
            if not summary_data:
                continue
            full = None
            if full_text and not articles:
                try:
                    full = await self._fetch_full_extract(session, title, language)
                except Exception as e:
                    logger.warning("Failed to fetch full extract for '%s': %s", title, e)
            page_url = (
                summary_data.get("content_urls", {}).get("desktop", {}).get("page")
                or f"{self._api_base(language)}/wiki/{title.replace(' ', '_')}"
            )
            articles.append(
                WikipediaArticle(
                    query=query,
                    title=summary_data.get("title", title),
                    page_url=page_url,
                    summary=summary_data.get("extract", ""),
                    description=summary_data.get("description"),
                    full_text=full,
                    thumbnail_url=(summary_data.get("thumbnail") or {}).get("source"),
                )
            )
        return articles

    async def run_async(self, params: WikipediaSearchToolInputSchema) -> WikipediaSearchToolOutputSchema:
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            tasks = [
                self._process_query(session, q, params.language, params.max_results_per_query, params.full_text)
                for q in params.queries
            ]
            grouped = await asyncio.gather(*tasks, return_exceptions=True)

        results: List[WikipediaArticle] = []
        for query, group in zip(params.queries, grouped):
            if isinstance(group, Exception):
                logger.warning("Query '%s' failed: %s", query, group)
                continue
            results.extend(group)
        return WikipediaSearchToolOutputSchema(results=results)

    def run(self, params: WikipediaSearchToolInputSchema) -> WikipediaSearchToolOutputSchema:
        with ThreadPoolExecutor() as executor:
            return executor.submit(contextvars.copy_context().run, asyncio.run, self.run_async(params)).result()


#################
# EXAMPLE USAGE #
#################
if __name__ == "__main__":  # pragma: no cover
    from rich.console import Console

    console = Console()
    tool = WikipediaSearchTool()

    output = tool.run(
        WikipediaSearchToolInputSchema(
            queries=["Atomic Agents", "Pydantic"],
            language="en",
            max_results_per_query=2,
            full_text=False,
        )
    )

    for article in output.results:
        console.rule(f"[bold cyan]{article.query} -> {article.title}")
        console.print(f"[bold]URL:[/bold] {article.page_url}")
        if article.description:
            console.print(f"[bold]Description:[/bold] {article.description}")
        console.print(article.summary)
