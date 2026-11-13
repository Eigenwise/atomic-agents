import os
from typing import List, Literal, Optional
import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor

import aiohttp
from pydantic import Field

from atomic_agents import BaseIOSchema, BaseTool, BaseToolConfig


################
# INPUT SCHEMA #
################
class SearXNGSearchToolInputSchema(BaseIOSchema):
    """
    Schema for input to a tool for searching for information, news, references, and other content using SearXNG.
    Returns a list of search results with a short description or content snippet and URLs for further exploration
    """

    queries: List[str] = Field(..., description="List of search queries.")
    category: Optional[Literal["general", "news", "social_media"]] = Field(
        "general", description="Category of the search queries."
    )


####################
# OUTPUT SCHEMA(S) #
####################
class SearXNGSearchResultItemSchema(BaseIOSchema):
    """This schema represents a single search result item"""

    url: str = Field(..., description="The URL of the search result")
    title: str = Field(..., description="The title of the search result")
    content: Optional[str] = Field(None, description="The content snippet of the search result")
    query: str = Field(..., description="The query used to obtain this search result")


class SearXNGSearchToolOutputSchema(BaseIOSchema):
    """This schema represents the output of the SearXNG search tool."""

    results: List[SearXNGSearchResultItemSchema] = Field(..., description="List of search result items")
    category: Optional[str] = Field(None, description="The category of the search results")


##############
# TOOL LOGIC #
##############
class SearXNGSearchToolConfig(BaseToolConfig):
    base_url: str = ""
    max_results: int = 10


class SearXNGSearchTool(BaseTool[SearXNGSearchToolInputSchema, SearXNGSearchToolOutputSchema]):
    """
    Tool for performing searches on SearXNG based on the provided queries and category.

    Attributes:
        input_schema (SearXNGSearchToolInputSchema): The schema for the input data.
        output_schema (SearXNGSearchToolOutputSchema): The schema for the output data.
        max_results (int): The maximum number of search results to return.
        base_url (str): The base URL for the SearXNG instance to use.
    """

    def __init__(self, config: SearXNGSearchToolConfig = SearXNGSearchToolConfig()):
        """
        Initializes the SearXNGTool.

        Args:
            config (SearXNGSearchToolConfig):
                Configuration for the tool, including base URL, max results, and optional title and description overrides.
        """
        super().__init__(config)
        self.base_url = config.base_url
        self.max_results = config.max_results

    async def _fetch_search_results(self, session: aiohttp.ClientSession, query: str, category: Optional[str]) -> List[dict]:
        """
        Fetches search results for a single query asynchronously.

        Args:
            session (aiohttp.ClientSession): The aiohttp session to use for the request.
            query (str): The search query.
            category (Optional[str]): The category of the search query.

        Returns:
            List[dict]: A list of search result dictionaries.

        Raises:
            Exception: If the request to SearXNG fails.
        """
        query_params = {
            "q": query,
            "safesearch": "0",
            "format": "json",
            "language": "en",
            "engines": "bing,duckduckgo,google,startpage,yandex",
        }

        if category:
            query_params["categories"] = category

        async with session.get(f"{self.base_url}/search", params=query_params) as response:
            if response.status != 200:
                raise Exception(f"Failed to fetch search results for query '{query}': {response.status} {response.reason}")
            data = await response.json()
            results = data.get("results", [])

            # Add the query to each result
            for result in results:
                result["query"] = query

            return results

    async def run_async(
        self, params: SearXNGSearchToolInputSchema, max_results: Optional[int] = None
    ) -> SearXNGSearchToolOutputSchema:
        """
        Runs the SearXNGTool asynchronously with the given parameters.

        Args:
            params (SearXNGSearchToolInputSchema): The input parameters for the tool, adhering to the input schema.
            max_results (Optional[int]): The maximum number of search results to return.

        Returns:
            SearXNGSearchToolOutputSchema: The output of the tool, adhering to the output schema.

        Raises:
            ValueError: If the base URL is not provided.
            Exception: If the request to SearXNG fails.
        """
        async with aiohttp.ClientSession() as session:
            tasks = [
                asyncio.create_task(self._fetch_search_results(session, query, params.category)) for query in params.queries
            ]
            try:
                results = await asyncio.gather(*tasks)
            except BaseException:
                # Finish every request before its shared session is closed.
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        all_results = [item for sublist in results for item in sublist]

        # Sort the combined results by score in descending order
        sorted_results = sorted(all_results, key=lambda x: x.get("score", 0), reverse=True)

        # Remove duplicates while preserving order
        seen_urls = set()
        unique_results = []
        for result in sorted_results:
            if "content" not in result or "title" not in result or "url" not in result or "query" not in result:
                continue
            if result["url"] not in seen_urls:
                unique_results.append(result)
                if "metadata" in result:
                    result["title"] = f"{result['title']} - (Published {result['metadata']})"
                if "publishedDate" in result and result["publishedDate"]:
                    result["title"] = f"{result['title']} - (Published {result['publishedDate']})"
                seen_urls.add(result["url"])

        # Filter results to include only those with the correct category if it is set
        if params.category:
            filtered_results = [result for result in unique_results if result.get("category") == params.category]
        else:
            filtered_results = unique_results

        filtered_results = filtered_results[: max_results or self.max_results]

        return SearXNGSearchToolOutputSchema(
            results=[
                SearXNGSearchResultItemSchema(
                    url=result["url"], title=result["title"], content=result.get("content"), query=result["query"]
                )
                for result in filtered_results
            ],
            category=params.category,
        )

    def run(self, params: SearXNGSearchToolInputSchema, max_results: Optional[int] = None) -> SearXNGSearchToolOutputSchema:
        """
        Runs the SearXNGTool synchronously with the given parameters.

        This method creates an event loop in a separate thread to run the asynchronous operations.

        Args:
            params (SearXNGSearchToolInputSchema): The input parameters for the tool, adhering to the input schema.
            max_results (Optional[int]): The maximum number of search results to return.

        Returns:
            SearXNGSearchToolOutputSchema: The output of the tool, adhering to the output schema.

        Raises:
            ValueError: If the base URL is not provided.
            Exception: If the request to SearXNG fails.
        """
        with ThreadPoolExecutor() as executor:
            return executor.submit(contextvars.copy_context().run, asyncio.run, self.run_async(params, max_results)).result()


#################
# EXAMPLE USAGE #
#################
if __name__ == "__main__":
    from rich.console import Console
    from dotenv import load_dotenv

    load_dotenv()
    rich_console = Console()

    search_tool_instance = SearXNGSearchTool(
        config=SearXNGSearchToolConfig(base_url=os.getenv("SEARXNG_BASE_URL"), max_results=5)
    )

    search_input = SearXNGSearchToolInputSchema(
        queries=["Python programming", "Machine learning", "Artificial intelligence"],
        category="news",
    )

    output = search_tool_instance.run(search_input)

    rich_console.print(output)
