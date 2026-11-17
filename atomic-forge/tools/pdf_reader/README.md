# PDF Reader Tool

## Overview
Extracts text and metadata from a PDF, given either a local file path or an HTTP(S) URL. Supports page-range filtering (e.g. `1-5`, `3`, `1,3,5-7`) and exposes per-page text plus the concatenated full text. Backed by `pypdf` — pure Python, no native binaries needed.

## Prerequisites and Dependencies
- Python 3.12 or later
- `atomic-agents`
- `pydantic`
- `pypdf`
- `requests`

## Installation
1. Use the Atomic Assembler CLI: run `atomic` and pick `pdf_reader`.
2. Or copy the `tool/` folder directly into your project.

## Input & Output Structure

### Input Schema
- `source` (str): a local file path or HTTP(S) URL pointing at a PDF.
- `page_range` (str, optional): `'1'`, `'1-5'`, `'1,3,5-7'`, etc. (1-based). Out-of-range pages are silently skipped.
- `include_metadata` (bool): default `True`.
- `extraction_mode` (`"plain"` or `"layout"`): defaults to `"plain"`. Choose `"layout"` to preserve rendered row and column positions with fixed-width spacing, useful for reports and tables whose PDF text operators are stored column by column.
- `include_links` (bool): extract embedded URI annotations for the selected pages; default `False`.

### Output Schema
- `source` (str): echo of the input source.
- `text` (str): concatenated text for the extracted pages, separated by `\f` (form feed).
- `pages` (list[PdfPage]): per-page extracted text with 1-based `page_number` and `links` (list[str]). Links are empty unless requested.
- `page_count` (int): number of pages extracted (after page_range filter).
- `total_page_count` (int): total pages in the source document.
- `metadata` (PdfMetadata, optional): document info dictionary.
- `error` (str, optional): set when the operation failed.

## Usage

```python
from tool.pdf_reader import PdfReaderTool, PdfReaderToolInputSchema

tool = PdfReaderTool()

# Read remote PDF, only first two pages
out = tool.run(PdfReaderToolInputSchema(
    source="https://arxiv.org/pdf/1706.03762",
    page_range="1-2",
))
print(out.text[:1000])

# Read a local PDF, every page
out = tool.run(PdfReaderToolInputSchema(source="./paper.pdf"))

# Preserve row associations in a text-based inventory report
out = tool.run(PdfReaderToolInputSchema(
    source="./inventory.pdf",
    extraction_mode="layout",
))
```

To keep citation targets that are not printed as text:

```python
out = tool.run(PdfReaderToolInputSchema(source="./paper.pdf", include_links=True))
for page in out.pages:
    print(page.page_number, page.links)
```

URI targets are returned as stored in the PDF, in annotation order, including repeated and relative targets. Internal page jumps and other annotation types are omitted. Targets are never fetched or executed, and extracted page text stays unchanged.

## Notes
- The tool fails gracefully and returns `error` set in the output rather than raising.
- For URL sources, the download is hard-capped by `max_size_bytes` in the config (default 50 MB).
- Encrypted PDFs are not currently supported.
- Layout extraction uses [pypdf's layout mode](https://pypdf.readthedocs.io/en/5.1.0/user/extract-text.html). It retains spacing in the returned text; it does not produce structured table cells or perform OCR. Complex layouts and rotated text have the same limitations as the underlying pypdf mode.

## Contributing
PRs welcome — see the main repo `CONTRIBUTING.md`.

## License
Same as the main Atomic Agents project.
