# Webpage Scraper Tool

## Overview
The Webpage Scraper Tool is a utility within the Atomic Agents ecosystem designed for scraping web content and converting it to markdown format. It includes features for extracting metadata and cleaning up the content for better readability.

## Prerequisites and Dependencies
- Python 3.9 or later
- atomic-agents (See [here](/README.md) for installation instructions)
- pydantic
- requests
- beautifulsoup4
- markdownify
- readability-lxml

## Installation
You can install the tool using any of the following options:

1. Using the CLI tool that comes with Atomic Agents. Simply run `atomic` and select the tool from the list of available tools. After doing so you will be asked for a target directory to download the tool into.
2. Good old fashioned copy/paste: Just like any other tool inside the Atomic Forge, you can copy the code from this repo directly into your own project, provided you already have atomic-agents installed according to the instructions in the main [README](/README.md).

## Configuration

### Parameters

- `user_agent` (str): User agent string to use for requests. Defaults to Chrome/Windows user agent.
- `timeout` (int): Timeout in seconds for HTTP requests. Defaults to 30.
- `max_content_length` (int): Maximum response body length in bytes after HTTP decompression. Defaults to 1,000,000. The body is read incrementally and the response is closed as soon as this limit is exceeded, even when `Content-Length` is absent or inaccurate. Character decoding happens after the size check.

### Example

```python
config = WebpageScraperToolConfig(
    user_agent="Custom User Agent String",
    timeout=60,
    max_content_length=2_000_000
)
```

## Input & Output Structure

### Input Schema
- `url` (HttpUrl): URL of the webpage to scrape.
- `include_links` (bool): Whether to preserve hyperlinks in the markdown output. Defaults to True.

### Output Schema
- `content` (str): The scraped content in markdown format.
- `metadata` (WebpageMetadata): Metadata about the scraped webpage, including:
  - `title` (str): The title of the webpage
  - `author` (Optional[str]): The author of the webpage content
  - `description` (Optional[str]): Meta description of the webpage
  - `site_name` (Optional[str]): Name of the website
  - `domain` (str): Domain name of the website

## Usage

Here's an example of how to use the Webpage Scraper Tool:

```python
from tool.webpage_scraper import WebpageScraperTool, WebpageScraperToolConfig

# Initialize the tool
scraper = WebpageScraperTool(config=WebpageScraperToolConfig())

# Define input data
input_data = WebpageScraperTool.input_schema(
    url="https://example.com/article",
    include_links=True
)

# Perform the scraping
result = scraper.run(input_data)
print(f"Title: {result.metadata.title}")
print(f"Content: {result.content[:200]}...")  # Preview first 200 chars
```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes with clear messages.
4. Open a pull request detailing your changes.

Please ensure you follow the project's coding standards and include tests for any new features or bug fixes.

## License

This project is licensed under the same license as the main Atomic Agents project. See the [LICENSE](../../../LICENSE) file in the repository root for more details.
