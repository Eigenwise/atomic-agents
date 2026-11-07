# BoCha Search Tool

## Overview
The BoCha Search Tool is a powerful utility within the Atomic Agents ecosystem that allows you to perform searches using BoCha, a search engine built for AI Agents. This tool enables you to fetch search results from BoCha.

## Prerequisites and Dependencies
- Python 3.9 or later
- atomic-agents (See [here](/README.md) for installation instructions)
- pydantic
- aiohttp

## Installation
You can install the tool using any of the following options:

1. Using the CLI tool that comes with Atomic Agents. Simply run `atomic` and select the tool from the list of available tools. After doing so you will be asked for a target directory to download the tool into.
2. Good old fashioned copy/paste: Just like any other tool inside the Atomic Forge, you can copy the code from this repo directly into your own project, provided you already have atomic-agents installed according to the instructions in the main [README](/README.md).

## Configuration

### Parameters

- `api_key` (str): The api key of the BoCha user.
- `freshness` (str, optional): `noLimit` (default), `oneDay`, `oneWeek`, `oneMonth`, `oneYear`, a real date such as `2026-11-07`, or an inclusive date range such as `2026-11-01..2026-11-07`. Invalid calendar dates, format placeholders, and reversed ranges are rejected during configuration.
- `count` (int, optional): The maximum number of search results to return.

### Example

```python
config = BoChaSearchToolConfig(
    api_key="my-api-key",
    count=5
)
```

## Input & Output Structure

### Input Schema
- `queries` (List[str]): List of search queries.

### Output Schema
- `results` (List[BoChaSearchResultItemSchema]): List of search result items.

Each `BoChaSearchResultItemSchema` contains:
- `name` (str): The title of the search result.
- `url` (str): The URL of the search result.
- `snippet` (str): The content snippet of the search result.
- `query` (str): The query used to obtain this search result.


## Usage

Here's an example of how to use the BoCha Search Tool:


```python
import os
from tool.bocha_search import BoChaSearchTool, BoChaSearchToolConfig, BoChaSearchToolInputSchema

# Initialize the tool with your BoCha instance URL
config = BoChaSearchToolConfig(api_key=os.getenv("BOCHA_API_KEY"), count=5)
search_tool = BoChaSearchTool(config=config)

# Define input data
input_data = BoChaSearchToolInputSchema(
    queries=["Python programming", "Machine learning"],
)

# Perform the search
result = search_tool.run(input_data)
print(result)
```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes with clear messages.
4. Open a pull request detailing your changes.

Please ensure you follow the project's coding standards and include tests for any new features or bug fixes.

## License

This project is licensed under the same license as the main Atomic Agents project. See the [LICENSE](/LICENSE) file in the repository root for more details.
