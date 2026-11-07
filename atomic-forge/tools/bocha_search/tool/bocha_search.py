import os
import re
from datetime import date
from typing import List, Optional
import asyncio
import contextvars
from concurrent.futures import ThreadPoolExecutor
import logging
import aiohttp
from pydantic import Field, field_validator

from atomic_agents import BaseIOSchema, BaseTool, BaseToolConfig


################
# INPUT SCHEMA #
################
class BoChaSearchToolInputSchema(BaseIOSchema):
    """
    Schema for input to a tool for searching for information, news, references, and other content using BoCha.
    Returns a list of search results with a short description or content snippet and URLs for further exploration
    """

    queries: List[str] = Field(..., description="List of search queries.")


####################
# OUTPUT SCHEMA(S) #
####################
class BoChaSearchResultItemSchema(BaseIOSchema):
    """This schema represents a single search result item"""

    name: str = Field(..., description="The title of the search result")
    url: str = Field(..., description="The URL of the search result")
    snippet: Optional[str] = Field(None, description="The content snippet of the search result")
    query: Optional[str] = Field(None, description="The query used to obtain this search result")


class BoChaSearchToolOutputSchema(BaseIOSchema):
    """This schema represents the output of the BoCha search tool."""

    results: List[BoChaSearchResultItemSchema] = Field(..., description="List of search result items")


##############
# TOOL LOGIC #
##############
class BoChaSearchToolConfig(BaseToolConfig):

    api_key: str = ""
    freshness: str = Field(
        default="noLimit",
        description="Time filter: noLimit, oneDay, oneWeek, oneMonth, oneYear, YYYY-MM-DD, or YYYY-MM-DD..YYYY-MM-DD.",
    )
    include: Optional[str] = None
    exclude: Optional[str] = None
    count: int = 10

    @field_validator("freshness")
    @classmethod
    def validate_freshness(cls, value: str) -> str:
        """Accept supported presets or real, chronologically ordered ISO dates."""
        if value in {"noLimit", "oneDay", "oneWeek", "oneMonth", "oneYear"}:
            return value
        message = "freshness must be a supported preset, a YYYY-MM-DD date, or an ordered YYYY-MM-DD..YYYY-MM-DD range"
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}(?:\.\.[0-9]{4}-[0-9]{2}-[0-9]{2})?", value):
            raise ValueError(message)
        try:
            dates = [date.fromisoformat(part) for part in value.split("..")]
        except ValueError as exc:
            raise ValueError(message) from exc
        if len(dates) == 2 and dates[0] > dates[1]:
            raise ValueError(message)
        return value


class BoChaSearchTool(BaseTool[BoChaSearchToolInputSchema, BoChaSearchToolOutputSchema]):
    """
    Tool for performing searches using the BoCha search API.

    Attributes:
        input_schema (BoChaSearchToolInputSchema): The schema for the input data.
        output_schema (BoChaSearchToolOutputSchema): The schema for the output data.
        count (int): The number of search results to return.
        api_key (str): The API key for the BoCha API.
    """

    def __init__(self, config: BoChaSearchToolConfig = BoChaSearchToolConfig()):
        """
        Initializes the BoChaSearchTool.

        Args:
            config (BoChaSearchToolConfig):
                Configuration for the tool, including API key, max results, and optional title and description overrides.
        """
        super().__init__(config)
        self.api_key = config.api_key or os.getenv("BOCHA_API_KEY", "")
        self.count = config.count
        self.freshness = config.freshness
        self.include = config.include
        self.exclude = config.exclude

    async def _fetch_search_results(self, session: aiohttp.ClientSession, query: str) -> dict:
        headers = {
            "content-type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        json_data = {
            "query": query,
            "count": self.count,
            "freshness": self.freshness,
        }
        if self.include is not None:
            json_data["include"] = self.include
        if self.exclude is not None:
            json_data["exclude"] = self.exclude

        async with session.post("https://api.bochaai.com/v1/web-search", headers=headers, json=json_data) as response:
            if response.status != 200:
                error_message = await response.text()
                raise Exception(
                    f"Failed to fetch search results for query '{query}': {response.status} {response.reason}. Details: {error_message}"
                )
            data = await response.json()

            results = []
            webpages = data.get("data", {}).get("webPages", {})
            if isinstance(webpages, dict):
                results = webpages.get("value", [])

            # Add query information to each result
            for result in results:
                result["query"] = query

            return {"results": results}

    async def run_async(self, params: BoChaSearchToolInputSchema, count: Optional[int] = None) -> BoChaSearchToolOutputSchema:
        async with aiohttp.ClientSession() as session:
            # Fetch results for all queries
            tasks = [self._fetch_search_results(session, query) for query in params.queries]
            raw_responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results for each query
        processed_results = []
        for query, response in zip(params.queries, raw_responses):
            if isinstance(response, Exception):
                # Log the failure, skip or handle as needed
                logging.warning(f"Query '{query}' failed: {response}")
                continue
            query_results = response["results"]
            query_processed = []
            for result in query_results:
                if all(key in result for key in ["name", "url", "snippet"]):
                    query_processed.append(
                        BoChaSearchResultItemSchema(
                            name=result["name"],
                            url=result["url"],
                            snippet=result.get("snippet", ""),
                            query=result.get("query"),
                        )
                    )
                else:
                    print(f"Skipping result due to missing keys: {result}")

            # Limit results per query
            query_processed = query_processed[: count or self.count]
            processed_results.extend(query_processed)

        return BoChaSearchToolOutputSchema(results=processed_results)

    def run(self, params: BoChaSearchToolInputSchema, count: Optional[int] = None) -> BoChaSearchToolOutputSchema:
        """
        Runs the BoChaTool synchronously with the given parameters.

        This method creates an event loop in a separate thread to run the asynchronous operations.

        Args:
            params (BoChaSearchToolInputSchema): The input parameters for the tool, adhering to the input schema.
            count (Optional[int]): The maximum number of search results to return.

        Returns:
            BoChaSearchToolOutputSchema: The output of the tool, adhering to the output schema.

        Raises:
            ValueError: If the base URL is not provided.
            Exception: If the request to BOCha fails.
        """
        with ThreadPoolExecutor() as executor:
            result = executor.submit(
                contextvars.copy_context().run,
                asyncio.run,
                self.run_async(
                    params,
                    count,
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

    search_tool_instance = BoChaSearchTool(config=BoChaSearchToolConfig(api_key=os.getenv("BOCHA_API_KEY"), count=2))

    # search_input = BoChaSearchToolInputSchema(queries=["北京天气怎么样？", "天津天气怎么样？", "杭州天气怎么样？"])
    search_input = BoChaSearchToolInputSchema(queries=["Python programming", "Machine learning"])

    output = search_tool_instance.run(search_input)

    rich_console.print(output)
