import asyncio
import contextvars
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Literal, Optional

import aiohttp
from pydantic import Field

from atomic_agents import BaseIOSchema, BaseTool, BaseToolConfig


logger = logging.getLogger(__name__)


################
# INPUT SCHEMA #
################
class HackerNewsSearchToolInputSchema(BaseIOSchema):
    """
    Search Hacker News stories, comments, Show HN posts, Ask HN posts, and polls
    via the free Algolia HN search API. No authentication required. Use this to
    find tech news, discussions, and threads on a topic.
    """

    queries: List[str] = Field(..., description="Search queries to run against Hacker News.")
    tags: Literal["story", "comment", "show_hn", "ask_hn", "poll", "front_page"] = Field(
        default="story",
        description=(
            "Type of item to search for. 'story' = regular submissions; 'comment' ="
            " HN comments; 'show_hn'/'ask_hn' = labelled posts; 'poll' = polls;"
            " 'front_page' = stories that reached the front page."
        ),
    )
    sort_by: Literal["relevance", "date"] = Field(
        default="relevance",
        description=(
            "'relevance' uses Algolia's relevance ranking; 'date' returns the newest"
            " items first (uses the search_by_date endpoint)."
        ),
    )
    max_results_per_query: int = Field(default=10, ge=1, le=50, description="Maximum results per query.")
    numeric_filters: Optional[str] = Field(
        default=None,
        description=(
            "Optional Algolia numeric filter expression to narrow results, e.g."
            " 'points>100', 'created_at_i>1700000000', or"
            " 'num_comments>50,points>20'."
        ),
    )


####################
# OUTPUT SCHEMA(S) #
####################
class HackerNewsItem(BaseIOSchema):
    """A single Hacker News item (story, comment, or other)."""

    query: str = Field(..., description="The query that produced this item.")
    object_id: str = Field(..., description="Hacker News item id.")
    title: Optional[str] = Field(None, description="Title (stories only; absent for comments).")
    url: Optional[str] = Field(None, description="External URL submitted to HN (absent for Ask HN posts and comments).")
    hn_url: str = Field(..., description="Direct URL to the item on news.ycombinator.com.")
    author: Optional[str] = Field(None, description="Username of the poster.")
    points: Optional[int] = Field(None, description="Score of the item, if applicable.")
    num_comments: Optional[int] = Field(None, description="Number of comments on the story, if applicable.")
    created_at: Optional[str] = Field(None, description="ISO 8601 timestamp the item was posted.")
    story_text: Optional[str] = Field(None, description="Body text for Ask HN posts and self-posts (HTML).")
    comment_text: Optional[str] = Field(None, description="Body text for comments (HTML).")
    parent_id: Optional[str] = Field(None, description="Parent item id (for comments).")
    story_id: Optional[str] = Field(None, description="Story this comment belongs to (for comments).")


class HackerNewsSearchFailure(BaseIOSchema):
    """A query that could not be completed, distinct from a successful empty search."""

    query: str = Field(..., description="The query that failed.")
    error: str = Field(..., description="Reason the query could not be completed.")


class HackerNewsSearchToolOutputSchema(BaseIOSchema):
    """Output of the Hacker News search tool."""

    results: List[HackerNewsItem] = Field(..., description="Matching items across all queries.")
    failures: List[HackerNewsSearchFailure] = Field(
        default_factory=list, description="Queries that failed; an empty results list is conclusive only when this is empty."
    )


#################
# CONFIGURATION #
#################
class HackerNewsSearchToolConfig(BaseToolConfig):
    """Configuration for the HackerNewsSearchTool."""

    base_url: str = Field(default="https://hn.algolia.com/api/v1", description="Algolia HN search base URL.")
    user_agent: str = Field(
        default="atomic-agents-hackernews-tool/1.0 (+https://github.com/eigenwise/atomic-agents)",
        description="User agent for HTTP requests.",
    )
    timeout: float = Field(default=15.0, ge=1.0, le=120.0, description="HTTP request timeout in seconds.")


#####################
# MAIN TOOL & LOGIC #
#####################
class HackerNewsSearchTool(BaseTool[HackerNewsSearchToolInputSchema, HackerNewsSearchToolOutputSchema]):
    """Tool for searching Hacker News via the public Algolia API."""

    HN_ITEM_BASE = "https://news.ycombinator.com/item?id="

    def __init__(self, config: HackerNewsSearchToolConfig = HackerNewsSearchToolConfig()):
        super().__init__(config)
        self.base_url = config.base_url
        self.user_agent = config.user_agent
        self.timeout = config.timeout

    @classmethod
    def _to_item(cls, hit: dict, query: str) -> HackerNewsItem:
        object_id = str(hit.get("objectID") or hit.get("story_id") or "")
        return HackerNewsItem(
            query=query,
            object_id=object_id,
            title=hit.get("title") or hit.get("story_title"),
            url=hit.get("url") or hit.get("story_url"),
            hn_url=f"{cls.HN_ITEM_BASE}{object_id}" if object_id else "https://news.ycombinator.com/",
            author=hit.get("author"),
            points=hit.get("points"),
            num_comments=hit.get("num_comments"),
            created_at=hit.get("created_at"),
            story_text=hit.get("story_text"),
            comment_text=hit.get("comment_text"),
            parent_id=str(hit["parent_id"]) if hit.get("parent_id") is not None else None,
            story_id=str(hit["story_id"]) if hit.get("story_id") is not None else None,
        )

    async def _fetch(
        self,
        session: aiohttp.ClientSession,
        query: str,
        tags: str,
        sort_by: str,
        max_results: int,
        numeric_filters: Optional[str],
    ) -> List[HackerNewsItem]:
        endpoint = "search_by_date" if sort_by == "date" else "search"
        params = {
            "query": query,
            "tags": tags,
            "hitsPerPage": str(max_results),
        }
        if numeric_filters:
            params["numericFilters"] = numeric_filters

        async with session.get(f"{self.base_url}/{endpoint}", params=params) as resp:
            if resp.status != 200:
                raise Exception(f"HN search failed for '{query}': {resp.status} {resp.reason}")
            data = await resp.json()
        if not isinstance(data, dict) or not isinstance(data.get("hits"), list):
            raise ValueError("HN search returned an invalid hits response")
        return [self._to_item(hit, query) for hit in data["hits"]]

    async def run_async(self, params: HackerNewsSearchToolInputSchema) -> HackerNewsSearchToolOutputSchema:
        headers = {"User-Agent": self.user_agent, "Accept": "application/json"}
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            tasks = [
                self._fetch(
                    session,
                    q,
                    params.tags,
                    params.sort_by,
                    params.max_results_per_query,
                    params.numeric_filters,
                )
                for q in params.queries
            ]
            grouped = await asyncio.gather(*tasks, return_exceptions=True)

        results: List[HackerNewsItem] = []
        failures: List[HackerNewsSearchFailure] = []
        for query, group in zip(params.queries, grouped):
            if isinstance(group, Exception):
                logger.warning("HN query '%s' failed: %s", query, group)
                failures.append(HackerNewsSearchFailure(query=query, error=str(group) or type(group).__name__))
                continue
            results.extend(group)
        return HackerNewsSearchToolOutputSchema(results=results, failures=failures)

    def run(self, params: HackerNewsSearchToolInputSchema) -> HackerNewsSearchToolOutputSchema:
        with ThreadPoolExecutor() as executor:
            return executor.submit(contextvars.copy_context().run, asyncio.run, self.run_async(params)).result()


#################
# EXAMPLE USAGE #
#################
if __name__ == "__main__":  # pragma: no cover
    from rich.console import Console

    console = Console()
    tool = HackerNewsSearchTool()

    output = tool.run(
        HackerNewsSearchToolInputSchema(
            queries=["atomic agents", "MCP server"],
            tags="story",
            sort_by="relevance",
            max_results_per_query=3,
            numeric_filters="points>10",
        )
    )

    for item in output.results:
        console.rule(f"[bold cyan]{item.title or item.object_id}")
        console.print(
            f"[bold]Author:[/bold] {item.author}  [bold]Points:[/bold] {item.points}  "
            f"[bold]Comments:[/bold] {item.num_comments}"
        )
        console.print(f"[bold]Posted:[/bold] {item.created_at}")
        console.print(f"[bold]HN URL:[/bold] {item.hn_url}")
        if item.url:
            console.print(f"[bold]Submitted URL:[/bold] {item.url}")
