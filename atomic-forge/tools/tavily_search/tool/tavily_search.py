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
class TavilySearchToolInputSchema(BaseIOSchema):
    """
    Schema for input to a tool for searching for information, news, references, and other content using Tavily.
    Returns a list of search results with a short description or content snippet and URLs for further exploration
    """

    queries: List[str] = Field(..., description="List of search queries.")


####################
# OUTPUT SCHEMA(S) #
####################
class TavilySearchResultItemSchema(BaseIOSchema):
    """This schema represents a single search result item"""

    title: str = Field(..., description="The title of the search result")
    url: str = Field(..., description="The URL of the search result")
    content: str = Field(None, description="The content snippet of the search result")
    score: float = Field(..., description="The score of the search result")
    raw_content: Optional[str] = Field(None, description="The raw content of the search result")
    query: Optional[str] = Field(..., description="The query used to obtain this search result")
    answer: Optional[str] = Field(..., description="The answer to the query provided by Tavily")


class TavilySearchToolOutputSchema(BaseIOSchema):
    """This schema represents the output of the Tavily search tool."""

    results: List[TavilySearchResultItemSchema] = Field(..., description="List of search result items")


##############
# TOOL LOGIC #
##############
class TavilySearchToolConfig(BaseToolConfig):
    api_key: str = ""
    max_results: int = 5
    search_depth: Literal["basic", "advanced"] = "basic"
    include_domains: Optional[List[str]] = None
    exclude_domains: Optional[List[str]] = None


class TavilySearchTool(BaseTool[TavilySearchToolInputSchema, TavilySearchToolOutputSchema]):
    """
    Tool for performing searches using the Tavily search API.

    Attributes:
        input_schema (TavilySearchToolInputSchema): The schema for the input data.
        output_schema (TavilySearchToolOutputSchema): The schema for the output data.
        max_results (int): The maximum number of search results to return.
        api_key (str): The API key for the Tavily API.
    """

    def __init__(self, config: TavilySearchToolConfig = TavilySearchToolConfig()):
        """
        Initializes the TavilySearchTool.

        Args:
            config (TavilySearchToolConfig):
                Configuration for the tool, including API key, max results, and optional title and description overrides.
        """
        super().__init__(config)
        self.api_key = config.api_key or os.getenv("TAVILY_API_KEY", "")
        self.max_results = config.max_results
        self.search_depth = config.search_depth
        self.include_domains = config.include_domains
        self.exclude_domains = config.exclude_domains
        self.include_answer = False  # Add this property to control whether to include the answer

    async def _fetch_search_results(self, session: aiohttp.ClientSession, query: str, max_results: int) -> dict:
        headers = {
            "accept": "/",
            "content-type": "application/json",
            "origin": "https://app.tavily.com/",
            "referer": "https://app.tavily.com/",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
        }

        json_data = {
            "query": query,
            "api_key": self.api_key,
            "search_depth": self.search_depth,
            "include_domains": self.include_domains,
            "exclude_domains": self.exclude_domains,
            "max_results": max_results,
            "include_answer": self.include_answer,  # Add the include_answer flag to the API request
        }

        async with session.post("https://api.tavily.com/search", headers=headers, json=json_data) as response:
            if response.status != 200:
                error_message = await response.text()
                raise Exception(
                    f"Failed to fetch search results for query '{query}': {response.status} {response.reason}. Details: {error_message}"
                )
            data = await response.json()

            results = data.get("results", [])
            answer = data.get("answer", "")  # Get the answer from the response

            # Add query information to each result
            for result in results:
                result["query"] = query

            return {"results": results, "answer": answer}  # Return both results and answer

    async def run_async(
        self, params: TavilySearchToolInputSchema, max_results: Optional[int] = None
    ) -> TavilySearchToolOutputSchema:
        effective_max_results = self.max_results if max_results is None else max_results

        async with aiohttp.ClientSession() as session:
            # Fetch results for all queries
            tasks = [
                asyncio.create_task(self._fetch_search_results(session, query, effective_max_results))
                for query in params.queries
            ]
            try:
                raw_responses = await asyncio.gather(*tasks)
            except BaseException:
                # Finish every request before its shared session is closed.
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        # Process results for each query
        processed_results = []
        for response in raw_responses:
            query_results = response["results"]
            answer = response["answer"]  # Get the answer for this query

            query_processed = []
            for result in query_results:
                if all(key in result for key in ["title", "url", "content", "score"]):
                    query_processed.append(
                        TavilySearchResultItemSchema(
                            title=result["title"],
                            url=result["url"],
                            content=result.get("content", ""),
                            score=result.get("score", 0),
                            raw_content=result.get("raw_content"),
                            query=result.get("query"),
                            answer=answer,  # Use the answer from the API response
                        )
                    )
                else:
                    print(f"Skipping result due to missing keys: {result}")

            # Limit results per query
            query_processed = query_processed[:effective_max_results]
            processed_results.extend(query_processed)

        return TavilySearchToolOutputSchema(results=processed_results)

    def run(self, params: TavilySearchToolInputSchema, max_results: Optional[int] = None) -> TavilySearchToolOutputSchema:
        """
        Runs the TavilyTool synchronously with the given parameters.

        This method creates an event loop in a separate thread to run the asynchronous operations.

        Args:
            params (TavilySearchToolInputSchema): The input parameters for the tool, adhering to the input schema.
            max_results (Optional[int]): The maximum number of search results to return.

        Returns:
            TavilySearchToolOutputSchema: The output of the tool, adhering to the output schema.

        Raises:
            ValueError: If the base URL is not provided.
            Exception: If the request to Tavily fails.
        """
        with ThreadPoolExecutor() as executor:
            result = executor.submit(
                contextvars.copy_context().run,
                asyncio.run,
                self.run_async(
                    params,
                    max_results,
                ),
            ).result()

        return result


####
# Main entry point for testing
if __name__ == "__main__":
    from rich.console import Console
    from dotenv import load_dotenv

    load_dotenv()
    rich_console = Console()

    search_tool_instance = TavilySearchTool(config=TavilySearchToolConfig(api_key=os.getenv("TAVILY_API_KEY"), max_results=2))

    search_input = TavilySearchToolInputSchema(queries=["Python programming", "Machine learning", "Artificial intelligence"])

    output = search_tool_instance.run(search_input)

    rich_console.print(output)
