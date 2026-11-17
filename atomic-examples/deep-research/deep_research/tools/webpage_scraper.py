from email.message import Message
from typing import Optional, Dict
import re
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup, UnicodeDammit
from markdownify import markdownify
from pydantic import Field, HttpUrl
from readability import Document
from atomic_agents import BaseIOSchema, BaseTool, BaseToolConfig


################
# INPUT SCHEMA #
################
class WebpageScraperToolInputSchema(BaseIOSchema):
    """
    Input schema for the WebpageScraperTool.
    """

    url: HttpUrl = Field(
        ...,
        description="URL of the webpage to scrape.",
    )
    include_links: bool = Field(
        default=True,
        description="Whether to preserve hyperlinks in the markdown output.",
    )


#################
# OUTPUT SCHEMA #
#################
class WebpageMetadata(BaseIOSchema):
    """Schema for webpage metadata."""

    title: str = Field(..., description="The title of the webpage.")
    author: Optional[str] = Field(None, description="The author of the webpage content.")
    description: Optional[str] = Field(None, description="Meta description of the webpage.")
    site_name: Optional[str] = Field(None, description="Name of the website.")
    domain: str = Field(..., description="Domain name of the website.")


class WebpageScraperToolOutputSchema(BaseIOSchema):
    """Schema for the output of the WebpageScraperTool."""

    content: str = Field(..., description="The scraped content in markdown format.")
    metadata: WebpageMetadata = Field(..., description="Metadata about the scraped webpage.")
    error: Optional[str] = Field(None, description="Error message if the scraping failed.")


#################
# CONFIGURATION #
#################
class WebpageScraperToolConfig(BaseToolConfig):
    """
    Configuration for the WebpageScraperTool.

    Attributes:
        timeout (int): Timeout for the HTTP request in seconds.
        headers (Dict[str, str]): HTTP headers to use for the request.
        min_text_length (int): Minimum length of text to consider the webpage valid.
        use_trafilatura (bool): Whether to use trafilatura for webpage parsing.
    """

    timeout: int = 30
    headers: Dict[str, str] = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Accept": "text/html,application/xhtml+xml,application/xml",
        "Accept-Language": "en-US,en;q=0.9",
    }
    min_text_length: int = 200
    max_content_length: int = 10 * 1024 * 1024  # 10 MB
    use_trafilatura: bool = True


#####################
# MAIN TOOL & LOGIC #
#####################
class WebpageScraperTool(BaseTool[WebpageScraperToolInputSchema, WebpageScraperToolOutputSchema]):
    """
    Tool for scraping and extracting information from a webpage.

    Attributes:
        input_schema (WebpageScraperToolInputSchema): The schema for the input data.
        output_schema (WebpageScraperToolOutputSchema): The schema for the output data.
        timeout (int): Timeout for the HTTP request in seconds.
        headers (Dict[str, str]): HTTP headers to use for the request.
        min_text_length (int): Minimum length of text to consider the webpage valid.
        use_trafilatura (bool): Whether to use trafilatura for webpage parsing.
    """

    def __init__(self, config: WebpageScraperToolConfig = WebpageScraperToolConfig()):
        """
        Initializes the WebpageScraperTool.

        Args:
            config (WebpageScraperToolConfig): Configuration for the WebpageScraperTool.
        """
        super().__init__(config)
        self.timeout = config.timeout
        self.headers = config.headers
        self.min_text_length = config.min_text_length
        self.use_trafilatura = config.use_trafilatura

    def _fetch_webpage(self, url: str) -> str:
        """
        Fetches the webpage content with custom headers.

        Args:
            url (str): The URL to fetch.

        Returns:
            str: The HTML content of the webpage.
        """
        response = requests.get(url, headers=self.headers, timeout=self.timeout)

        if len(response.content) > self.config.max_content_length:
            raise ValueError(f"Content length exceeds maximum of {self.config.max_content_length} bytes")

        content_type = Message()
        content_type["Content-Type"] = response.headers.get("Content-Type", "")
        charset = content_type.get_content_charset()
        # Decode HTML from its bytes: a BOM, an explicit HTTP charset, and an
        # HTML declaration take precedence over heuristic encoding detection.
        decoded = UnicodeDammit(response.content, is_html=True, user_encodings=[charset] if charset else []).unicode_markup
        return decoded if decoded is not None else response.text

    def _extract_metadata(self, soup: BeautifulSoup, doc: Document, url: str) -> WebpageMetadata:
        """
        Extracts metadata from the webpage.

        Args:
            soup (BeautifulSoup): The parsed HTML content.
            doc (Document): The readability document.
            url (str): The URL of the webpage.

        Returns:
            WebpageMetadata: The extracted metadata.
        """
        domain = urlparse(url).netloc

        # Extract metadata from meta tags
        metadata = {
            "title": doc.title(),
            "domain": domain,
            "author": None,
            "description": None,
            "site_name": None,
        }

        author_tag = soup.find("meta", attrs={"name": "author"})
        if author_tag:
            metadata["author"] = author_tag.get("content")

        description_tag = soup.find("meta", attrs={"name": "description"})
        if description_tag:
            metadata["description"] = description_tag.get("content")

        site_name_tag = soup.find("meta", attrs={"property": "og:site_name"})
        if site_name_tag:
            metadata["site_name"] = site_name_tag.get("content")

        return WebpageMetadata(**metadata)

    def _clean_markdown(self, markdown: str) -> str:
        """
        Cleans up the markdown content by removing excessive whitespace and normalizing formatting.

        Args:
            markdown (str): Raw markdown content.

        Returns:
            str: Cleaned markdown content.
        """
        # Remove multiple blank lines
        markdown = re.sub(r"\n\s*\n\s*\n", "\n\n", markdown)
        # Remove trailing whitespace
        markdown = "\n".join(line.rstrip() for line in markdown.splitlines())
        # Ensure content ends with single newline
        markdown = markdown.strip() + "\n"
        return markdown

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Extracts the main content from the webpage using custom heuristics.

        Args:
            soup (BeautifulSoup): Parsed HTML content.

        Returns:
            str: Main content HTML.
        """
        # Remove unwanted elements
        for element in soup.find_all(["script", "style", "nav", "header", "footer"]):
            element.decompose()

        # Try to find main content container
        content_candidates = [
            soup.find("main"),
            soup.find(id=re.compile(r"content|main", re.I)),
            soup.find(class_=re.compile(r"content|main", re.I)),
            soup.find("article"),
        ]

        main_content = next((candidate for candidate in content_candidates if candidate), None)

        if not main_content:
            main_content = soup.find("body")

        return str(main_content) if main_content else str(soup)

    def run(self, params: WebpageScraperToolInputSchema) -> WebpageScraperToolOutputSchema:
        """
        Runs the WebpageScraperTool with the given parameters.

        Args:
            params (WebpageScraperToolInputSchema): The input parameters for the tool.

        Returns:
            WebpageScraperToolOutputSchema: The output containing the markdown content and metadata.
        """

        try:
            # Fetch webpage content
            html_content = self._fetch_webpage(str(params.url))

            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")

            # Extract main content using custom extraction
            main_content = self._extract_main_content(soup)

            # Convert to markdown
            markdown_options = {
                "strip": ["script", "style"],
                "heading_style": "ATX",
                "bullets": "-",
                "wrap": True,
            }

            if not params.include_links:
                markdown_options["strip"].append("a")

            markdown_content = markdownify(main_content, **markdown_options)

            # Clean up the markdown
            markdown_content = self._clean_markdown(markdown_content)

            # Extract metadata
            metadata = self._extract_metadata(soup, Document(html_content), str(params.url))

            return WebpageScraperToolOutputSchema(
                content=markdown_content,
                metadata=metadata,
            )
        except Exception as e:
            # Create empty/minimal metadata with at least the domain
            domain = urlparse(str(params.url)).netloc
            minimal_metadata = WebpageMetadata(title="Error retrieving page", domain=domain)

            # Return with error message in the error field
            return WebpageScraperToolOutputSchema(content="", metadata=minimal_metadata, error=str(e))


#################
# EXAMPLE USAGE #
#################
if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown

    console = Console()
    scraper = WebpageScraperTool()

    try:
        result = scraper.run(
            WebpageScraperToolInputSchema(
                url="https://github.com/eigenwise/atomic-agents",
                include_links=True,
            )
        )

        # Check if there was an error during scraping, otherwise print the results
        if result.error:
            console.print(Panel.fit("Error", style="bold red"))
            console.print(f"[red]{result.error}[/red]")
        else:
            console.print(Panel.fit("Metadata", style="bold green"))
            console.print(result.metadata.model_dump_json(indent=2))

            console.print(Panel.fit("Content Preview (first 500 chars)", style="bold green"))
            # To show as markdown with proper formatting
            console.print(Panel.fit("Content as Markdown", style="bold green"))
            console.print(Markdown(result.content[:500]))

    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
