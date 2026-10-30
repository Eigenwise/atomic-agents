import io
import re
from pathlib import Path
from typing import List, Optional, Set

import requests
from pydantic import Field
from pypdf import PdfReader

from atomic_agents import BaseIOSchema, BaseTool, BaseToolConfig


################
# INPUT SCHEMA #
################
class PdfReaderToolInputSchema(BaseIOSchema):
    """
    Read a PDF file and extract its text. The source can be a local file path or
    an HTTP(S) URL. An optional page range limits extraction to specific pages
    (1-based, e.g. '1', '3-5', '1,3,5-7'). Use this whenever the agent needs the
    contents of a PDF — papers, reports, datasheets, etc.
    """

    source: str = Field(
        ...,
        min_length=1,
        description="Local file path (absolute or relative) or HTTP(S) URL of the PDF to read.",
    )
    page_range: Optional[str] = Field(
        default=None,
        description=(
            "Optional 1-based page-range expression to limit extraction. Single"
            " pages (`'3'`), ranges (`'1-5'`), and comma-separated combinations"
            " (`'1,3,5-7'`) are supported. When omitted, all pages are read."
        ),
    )
    include_metadata: bool = Field(default=True, description="Whether to include document metadata in the output.")


####################
# OUTPUT SCHEMA(S) #
####################
class PdfPage(BaseIOSchema):
    """A single extracted PDF page."""

    page_number: int = Field(..., description="1-based page number in the source document.")
    text: str = Field(..., description="Extracted text for this page.")


class PdfMetadata(BaseIOSchema):
    """Metadata fields read from the PDF document info dictionary."""

    title: Optional[str] = Field(None, description="Document title.")
    author: Optional[str] = Field(None, description="Document author.")
    subject: Optional[str] = Field(None, description="Document subject.")
    creator: Optional[str] = Field(None, description="Application that created the original document.")
    producer: Optional[str] = Field(None, description="Application that produced the PDF.")
    creation_date: Optional[str] = Field(None, description="Raw creation date string from the PDF.")
    modification_date: Optional[str] = Field(None, description="Raw modification date string from the PDF.")


class PdfReaderToolOutputSchema(BaseIOSchema):
    """Output of the PdfReaderTool."""

    source: str = Field(..., description="The source path or URL that was read (echoed back).")
    text: str = Field(..., description="Concatenated text across all extracted pages, separated by form feeds.")
    pages: List[PdfPage] = Field(..., description="Per-page extracted text.")
    page_count: int = Field(..., description="Number of pages extracted (after page_range filtering).")
    total_page_count: int = Field(..., description="Total number of pages in the source document.")
    metadata: Optional[PdfMetadata] = Field(None, description="Document metadata, when requested.")
    error: Optional[str] = Field(None, description="Error message when the operation failed.")


#################
# CONFIGURATION #
#################
class PdfReaderToolConfig(BaseToolConfig):
    """Configuration for the PdfReaderTool."""

    user_agent: str = Field(
        default=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            " (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        ),
        description="User agent for HTTP requests when the source is a URL.",
    )
    timeout: float = Field(default=30.0, ge=1.0, le=600.0, description="HTTP timeout in seconds for URL sources.")
    max_size_bytes: int = Field(
        default=50 * 1024 * 1024,
        ge=1,
        description="Maximum allowed PDF size in bytes when downloading from a URL (default 50 MB).",
    )


#####################
# MAIN TOOL & LOGIC #
#####################
class PdfReaderTool(BaseTool[PdfReaderToolInputSchema, PdfReaderToolOutputSchema]):
    """Tool for extracting text and metadata from local or remote PDF files."""

    def __init__(self, config: PdfReaderToolConfig = PdfReaderToolConfig()):
        super().__init__(config)
        self.user_agent = config.user_agent
        self.timeout = config.timeout
        self.max_size_bytes = config.max_size_bytes

    @staticmethod
    def _is_url(source: str) -> bool:
        return bool(re.match(r"^https?://", source, re.IGNORECASE))

    def _load_bytes(self, source: str) -> bytes:
        if self._is_url(source):
            headers = {"User-Agent": self.user_agent, "Accept": "application/pdf"}
            response = requests.get(source, headers=headers, timeout=self.timeout, stream=True)
            response.raise_for_status()
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > self.max_size_bytes:
                raise ValueError(f"PDF too large: declared {content_length} bytes (limit {self.max_size_bytes}).")
            data = bytearray()
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if chunk:
                    data.extend(chunk)
                    if len(data) > self.max_size_bytes:
                        raise ValueError(f"PDF too large: exceeded {self.max_size_bytes} bytes during download.")
            return bytes(data)

        path = Path(source).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"PDF not found at '{source}'.")
        if not path.is_file():
            raise ValueError(f"'{source}' is not a file.")
        return path.read_bytes()

    @staticmethod
    def parse_page_range(expression: str, total_pages: int) -> List[int]:
        """
        Parse a 1-based page-range expression into a sorted list of 1-based page
        numbers, clipped to ``total_pages``. Raises ValueError on malformed input.
        """
        text = expression.strip()
        if not text:
            raise ValueError("page_range cannot be empty when provided.")
        result: Set[int] = set()
        for part in text.split(","):
            chunk = part.strip()
            if not chunk:
                continue
            if "-" in chunk:
                start_str, end_str = chunk.split("-", 1)
                try:
                    start = int(start_str)
                    end = int(end_str)
                except ValueError as e:
                    raise ValueError(f"Invalid page range segment '{chunk}'.") from e
                if start < 1 or end < 1 or end < start:
                    raise ValueError(f"Invalid page range segment '{chunk}'.")
                for page in range(start, min(end, total_pages) + 1):
                    result.add(page)
            else:
                try:
                    page = int(chunk)
                except ValueError as e:
                    raise ValueError(f"Invalid page number '{chunk}'.") from e
                if page < 1:
                    raise ValueError(f"Page numbers must be >= 1 (got '{chunk}').")
                if page <= total_pages:
                    result.add(page)
        if not result:
            raise ValueError(f"No pages selected by '{expression}' (document has {total_pages} pages).")
        return sorted(result)

    @staticmethod
    def _format_metadata(reader: PdfReader) -> PdfMetadata:
        info = reader.metadata or {}

        def _g(key: str) -> Optional[str]:
            value = info.get(key) if hasattr(info, "get") else None
            return str(value) if value not in (None, "") else None

        return PdfMetadata(
            title=_g("/Title"),
            author=_g("/Author"),
            subject=_g("/Subject"),
            creator=_g("/Creator"),
            producer=_g("/Producer"),
            creation_date=_g("/CreationDate"),
            modification_date=_g("/ModDate"),
        )

    def run(self, params: PdfReaderToolInputSchema) -> PdfReaderToolOutputSchema:
        try:
            data = self._load_bytes(params.source)
            reader = PdfReader(io.BytesIO(data))
            total_pages = len(reader.pages)

            if params.page_range:
                page_numbers = self.parse_page_range(params.page_range, total_pages)
            else:
                page_numbers = list(range(1, total_pages + 1))

            pages: List[PdfPage] = []
            for page_number in page_numbers:
                page = reader.pages[page_number - 1]
                pages.append(PdfPage(page_number=page_number, text=page.extract_text() or ""))

            text = "\n\f\n".join(page.text for page in pages)
            metadata = self._format_metadata(reader) if params.include_metadata else None

            return PdfReaderToolOutputSchema(
                source=params.source,
                text=text,
                pages=pages,
                page_count=len(pages),
                total_page_count=total_pages,
                metadata=metadata,
            )
        except Exception as e:
            return PdfReaderToolOutputSchema(
                source=params.source,
                text="",
                pages=[],
                page_count=0,
                total_page_count=0,
                metadata=None,
                error=str(e),
            )


#################
# EXAMPLE USAGE #
#################
if __name__ == "__main__":  # pragma: no cover
    import sys

    from rich.console import Console

    # Force UTF-8 output so PDFs containing non-ASCII characters render on
    # Windows consoles with a default cp1252 codepage.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    console = Console()
    tool = PdfReaderTool()

    # The classic "Attention Is All You Need" paper from arXiv — a stable test PDF.
    output = tool.run(PdfReaderToolInputSchema(source="https://arxiv.org/pdf/1706.03762", page_range="1-2"))

    if output.error:
        console.print(f"[red]Error:[/red] {output.error}")
    else:
        if output.metadata:
            console.rule("[bold cyan]Metadata")
            console.print(output.metadata.model_dump())
        console.rule(f"[bold cyan]Pages {output.page_count} of {output.total_page_count}")
        snippet = output.text[:1500] + ("..." if len(output.text) > 1500 else "")
        # Print via stdout (UTF-8 reconfigured) to bypass rich's legacy-Windows code path.
        print(snippet)
