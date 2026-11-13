import unittest
from unittest.mock import patch, Mock
import requests
import sys
import os
from bs4 import BeautifulSoup
from readability import Document

# Add the parent directory to sys.path to find the tool module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tool.webpage_scraper import (  # noqa: E402
    WebpageScraperTool,
    WebpageScraperToolConfig,
    WebpageScraperToolInputSchema,
    WebpageScraperToolOutputSchema,
    WebpageMetadata,
)


class TestWebpageScraperTool(unittest.TestCase):
    """Test cases for the WebpageScraperTool."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = WebpageScraperToolConfig(
            user_agent="Test User Agent",
            timeout=10,
            max_content_length=500000,
        )
        self.tool = WebpageScraperTool(config=self.config)

        # Sample HTML content for testing
        self.sample_html = """<!DOCTYPE html>
        <html>
        <head>
            <title>Test Page</title>
            <meta name="author" content="Test Author">
            <meta name="description" content="Test Description">
            <meta property="og:site_name" content="Test Site">
        </head>
        <body>
            <header>
                <nav>Navigation</nav>
            </header>
            <main>
                <h1>Main Content</h1>
                <p>This is a test paragraph with a <a href="https://example.com">link</a>.</p>
                <ul>
                    <li>Item 1</li>
                    <li>Item 2</li>
                </ul>
            </main>
            <footer>Footer content</footer>
            <script>console.log('test');</script>
        </body>
        </html>
        """
        self.mock_url = "https://example.com/test"

    @patch("requests.get")
    def test_fetch_webpage(self, mock_get):
        """Test fetching a webpage."""
        # Configure the mock
        mock_response = Mock()
        mock_response.text = self.sample_html
        mock_response.content = self.sample_html.encode("utf-8")
        mock_get.return_value = mock_response

        # Test the method
        result = self.tool._fetch_webpage(self.mock_url)

        # Assertions
        mock_get.assert_called_once()
        self.assertEqual(result, self.sample_html)

        # Check headers
        call_args = mock_get.call_args
        headers = call_args[1]["headers"]
        self.assertEqual(headers["User-Agent"], "Test User Agent")

    @patch("requests.get")
    def test_fetch_webpage_content_too_large(self, mock_get):
        """Test handling of content that exceeds max length."""
        # Configure the mock
        mock_response = Mock()
        mock_response.text = "X" * (self.config.max_content_length + 1)
        mock_response.content = ("X" * (self.config.max_content_length + 1)).encode("utf-8")
        mock_get.return_value = mock_response

        # Test the method
        with self.assertRaises(ValueError) as context:
            self.tool._fetch_webpage(self.mock_url)

        self.assertIn("Content length exceeds maximum", str(context.exception))

    def test_extract_metadata(self):
        """Test extracting metadata from a webpage."""
        # Create BeautifulSoup and Document objects
        soup = BeautifulSoup(self.sample_html, "html.parser")
        doc = Document(self.sample_html)

        # Test the method
        metadata = self.tool._extract_metadata(soup, doc, self.mock_url)

        # Assertions
        self.assertIsInstance(metadata, WebpageMetadata)
        self.assertEqual(metadata.title, "Test Page")
        self.assertEqual(metadata.author, "Test Author")
        self.assertEqual(metadata.description, "Test Description")
        self.assertEqual(metadata.site_name, "Test Site")
        self.assertEqual(metadata.domain, "example.com")

    def test_clean_markdown(self):
        """Test cleaning up markdown content."""
        markdown = "# Title\n\n  \n\nSome content\n  with weird spacing\n  \n\n\nMore content\n"

        expected = "# Title\n\nSome content\n  with weird spacing\n\nMore content\n"

        # Test the method
        result = self.tool._clean_markdown(markdown)

        # Assertions
        self.assertEqual(result, expected)

    def test_clean_markdown_preserves_arbitrary_content(self):
        markdown = "# Title of a real article\n\nArticle with weird spacing\n\nA unique conclusion."

        self.assertEqual(self.tool._clean_markdown(markdown), markdown + "\n")

    def test_clean_markdown_preserves_indented_code(self):
        markdown = "Example:\n\n    if ready:\n        run()\n\n\nNext paragraph."

        self.assertEqual(
            self.tool._clean_markdown(markdown),
            "Example:\n\n    if ready:\n        run()\n\nNext paragraph.\n",
        )

    def test_clean_markdown_normalizes_whitespace_only_blank_lines(self):
        markdown = "First paragraph.\n \n\t\n\nSecond paragraph."

        self.assertEqual(self.tool._clean_markdown(markdown), "First paragraph.\n\nSecond paragraph.\n")

    def test_clean_markdown_triple_newlines(self):
        """Test cleaning up markdown content with triple newlines."""
        markdown = "Test\n\n\nMultiple\n\n\nNewlines"

        # Test the method
        result = self.tool._clean_markdown(markdown)

        # Verify that triple newlines are replaced with double newlines
        self.assertEqual(result, "Test\n\nMultiple\n\nNewlines\n")
        self.assertNotIn("\n\n\n", result)

        # Test another case to ensure the function works with other content patterns
        complex_markdown = """Regular content

          

        with triple newlines above"""

        complex_result = self.tool._clean_markdown(complex_markdown)
        self.assertNotIn("\n\n\n", complex_result)

    def test_extract_main_content(self):
        """Test extracting main content from HTML."""
        soup = BeautifulSoup(self.sample_html, "html.parser")

        # Test the method
        result = self.tool._extract_main_content(soup)

        # Assertions
        # The result should contain the main content but not header/footer/scripts
        self.assertIn("Main Content", result)
        self.assertIn("test paragraph", result)
        self.assertIn("Item 1", result)
        self.assertNotIn("Navigation", result)
        self.assertNotIn("Footer content", result)
        self.assertNotIn("console.log", result)

    def test_extract_main_content_fallbacks(self):
        """Test extracting main content with various fallback mechanisms."""
        # Test with HTML that has no main tag but has content/main in id
        html_with_id = """
        <html><body>
            <div id="content"><p>Content in div with ID</p></div>
        </body></html>
        """
        soup = BeautifulSoup(html_with_id, "html.parser")
        result = self.tool._extract_main_content(soup)
        self.assertIn("Content in div with ID", result)

        # Test with HTML that has no main or ID but has content in class
        html_with_class = """
        <html><body>
            <div class="main-content"><p>Content in div with class</p></div>
        </body></html>
        """
        soup = BeautifulSoup(html_with_class, "html.parser")
        result = self.tool._extract_main_content(soup)
        self.assertIn("Content in div with class", result)

        # Test with HTML that has article
        html_with_article = """
        <html><body>
            <article><p>Content in article</p></article>
        </body></html>
        """
        soup = BeautifulSoup(html_with_article, "html.parser")
        result = self.tool._extract_main_content(soup)
        self.assertIn("Content in article", result)

        # Test with HTML that has none of the above, should fallback to body
        html_with_body_only = """
        <html><body>
            <p>Content in body</p>
        </body></html>
        """
        soup = BeautifulSoup(html_with_body_only, "html.parser")
        result = self.tool._extract_main_content(soup)
        self.assertIn("Content in body", result)

        # Test with completely empty HTML
        empty_html = "<html></html>"
        soup = BeautifulSoup(empty_html, "html.parser")
        result = self.tool._extract_main_content(soup)
        self.assertEqual(result, str(soup))

    @patch.object(WebpageScraperTool, "_fetch_webpage")
    def test_run_with_links(self, mock_fetch):
        """Test running the tool with links included."""
        # Configure the mock
        mock_fetch.return_value = self.sample_html

        # Create input parameters
        params = WebpageScraperToolInputSchema(
            url=self.mock_url,
            include_links=True,
        )

        # Test the method
        result = self.tool.run(params)

        # Assertions
        self.assertIsInstance(result, WebpageScraperToolOutputSchema)
        self.assertIn("Main Content", result.content)
        self.assertIn("link", result.content)
        self.assertIn("https://example.com", result.content)
        self.assertEqual(result.metadata.title, "Test Page")

    @patch.object(WebpageScraperTool, "_fetch_webpage")
    def test_run_without_links(self, mock_fetch):
        """Test running the tool with links excluded."""
        # Configure the mock
        mock_fetch.return_value = self.sample_html

        # Create input parameters
        params = WebpageScraperToolInputSchema(
            url=self.mock_url,
            include_links=False,
        )

        # Test the method
        result = self.tool.run(params)

        # Assertions
        self.assertIsInstance(result, WebpageScraperToolOutputSchema)
        self.assertIn("Main Content", result.content)
        self.assertIn("link", result.content)  # The text "link" should still be there
        self.assertNotIn("https://example.com", result.content)  # But the URL should be gone
        self.assertEqual(result.metadata.title, "Test Page")

    @patch("requests.get")
    def test_http_errors(self, mock_get):
        """Test handling of HTTP errors."""
        # Configure the mock to raise an exception
        mock_get.side_effect = requests.exceptions.HTTPError("404 Client Error")

        # Create input parameters
        params = WebpageScraperToolInputSchema(
            url=self.mock_url,
            include_links=True,
        )

        # Test the method
        result = self.tool.run(params)

        # Check that the error is captured in the result
        self.assertIsNotNone(result.error)
        self.assertIn("404 Client Error", result.error)

    @patch("requests.get")
    def test_connection_timeout(self, mock_get):
        """Test handling of connection timeouts."""
        # Configure the mock to raise a timeout exception
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

        # Create input parameters
        params = WebpageScraperToolInputSchema(
            url=self.mock_url,
            include_links=True,
        )

        # Test the method
        result = self.tool.run(params)

        # Check that the error is captured in the result
        self.assertIsNotNone(result.error)
        self.assertIn("Connection timed out", result.error)

    def test_tool_config(self):
        """Test tool configuration."""
        # Test default config
        default_config = WebpageScraperToolConfig()
        self.assertEqual(default_config.timeout, 30)
        self.assertIn("Mozilla", default_config.user_agent)
        self.assertEqual(default_config.max_content_length, 1_000_000)

        # Test custom config
        custom_config = WebpageScraperToolConfig(
            user_agent="Custom Agent",
            timeout=45,
            max_content_length=2_000_000,
        )
        self.assertEqual(custom_config.timeout, 45)
        self.assertEqual(custom_config.user_agent, "Custom Agent")
        self.assertEqual(custom_config.max_content_length, 2_000_000)

    def test_clean_markdown_empty_content(self):
        """Test cleaning up empty markdown content."""
        # Test with empty string
        result = self.tool._clean_markdown("")
        self.assertEqual(result, "\n")

        # Test with whitespace only
        result = self.tool._clean_markdown("   \n\n   \n   ")
        self.assertEqual(result, "\n")

    def test_clean_markdown_with_triple_newlines(self):
        """Test cleaning up markdown content with forced triple newlines."""
        # Create input with triple newlines specifically for regex coverage
        markdown = "Line1\n\n\nLine2"

        # Call the method directly to ensure coverage
        result = self.tool._clean_markdown(markdown)

        # Verify it collapsed triple newlines to double
        self.assertEqual(result, "Line1\n\nLine2\n")


if __name__ == "__main__":
    unittest.main()
