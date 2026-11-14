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

### Output Schema
- `source` (str): echo of the input source.
- `text` (str): concatenated text for the extracted pages, separated by `\f` (form feed).
- `pages` (list[PdfPage]): per-page extracted text with 1-based `page_number`.
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
```

### Password-protected PDFs

Set a known user or owner password in the tool configuration. The password is a
Pydantic `SecretStr`, so configuration representations and JSON serialization
mask its value. It is not part of the model-facing input schema or sent as an
HTTP request header.

```python
import os

from pydantic import SecretStr
from tool.pdf_reader import PdfReaderTool, PdfReaderToolConfig, PdfReaderToolInputSchema

tool = PdfReaderTool(PdfReaderToolConfig(password=SecretStr(os.environ["PDF_PASSWORD"])))
out = tool.run(PdfReaderToolInputSchema(source="./protected-report.pdf", page_range="1-2"))
```

This works with local files and downloaded PDFs. An incorrect password returns
the usual `error` result without extracted text or metadata. Omitting the option
keeps pypdf's default behavior, including opening documents with an empty user
password. A configured password is ignored for an unencrypted document.

For AES-encrypted PDFs, install pypdf's optional cryptography dependencies with
`pip install 'pypdf[crypto]>=5.1.0,<7.0.0'`. The supported encryption algorithms are
those provided by the installed pypdf version; see its
[encryption documentation](https://pypdf.readthedocs.io/en/stable/user/encryption-decryption.html).

## Notes
- The tool fails gracefully and returns `error` set in the output rather than raising.
- For URL sources, the download is hard-capped by `max_size_bytes` in the config (default 50 MB).

## Contributing
PRs welcome — see the main repo `CONTRIBUTING.md`.

## License
Same as the main Atomic Agents project.
