"""Exercise both scrapers against actual HTTP response bytes and charset headers."""

import codecs
import gzip
import importlib.util
import sys
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest


@dataclass(frozen=True)
class Page:
    name: str
    text: str
    body: bytes
    content_type: str | None = "text/html"
    compressed: bool = False


def page(name, text, encoding="utf-8", declaration="", content_type="text/html", bom=b"", compressed=False):
    markup = (
        f"<!doctype html><html><head>{declaration}<title>{text}</title>"
        f'<meta name="author" content="{text}"><meta name="description" content="{text}">'
        f'<meta property="og:site_name" content="{text}"></head>'
        f"<body><main><h1>{text}</h1><p>{text}</p></main></body></html>"
    )
    body = bom + markup.encode(encoding)
    return Page(name, text, gzip.compress(body) if compressed else body, content_type, compressed)


PAGES = [
    page("utf8-meta", "中文研究报告", declaration='<meta charset="utf-8">'),
    page(
        "utf8-http-equiv",
        "日本語の研究報告",
        declaration='<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">',
    ),
    page("cyrillic-meta", "Научные исследования", "cp1251", '<meta charset="windows-1251">'),
    page("japanese-meta", "日本語の研究報告", "shift_jis", '<meta charset="shift_jis">'),
    page("windows-meta", "Étude française résultat €", "cp1252", '<meta charset="windows-1252">'),
    page("utf8-bom", "中文与日本語研究", bom=codecs.BOM_UTF8),
    page("utf16-le-bom", "中文与日本語研究", "utf-16le", bom=codecs.BOM_UTF16_LE),
    page("utf16-be-bom", "中文与日本語研究", "utf-16be", bom=codecs.BOM_UTF16_BE),
    page(
        "bom-before-header",
        "中文与日本語研究",
        "utf-16le",
        content_type="text/html; charset=windows-1251",
        bom=codecs.BOM_UTF16_LE,
    ),
    page(
        "header-before-meta", "Научные исследования", "cp1251", '<meta charset="utf-8">', 'text/html; CHARSET="WINDOWS-1251"'
    ),
    page("explicit-utf8", "中文研究报告", content_type="text/html; charset=utf-8"),
    page("explicit-latin1", "Étude française", "latin1", content_type="text/html; charset=iso-8859-1"),
    page(
        "unknown-header", "中文研究报告", declaration='<meta charset="utf-8">', content_type="text/html; charset=not-a-charset"
    ),
    page("unknown-meta", "中文研究报告", declaration='<meta charset="not-a-charset">'),
    page("absent-header", "中文研究报告", content_type=None),
    page("ascii", "Ordinary page content"),
    page("gzip-meta", "中文研究报告", declaration='<meta charset="utf-8">', compressed=True),
]


@pytest.fixture(scope="module")
def http_pages():
    pages = {f"/{entry.name}": entry for entry in PAGES}
    pages["/empty"] = Page("empty", "", b"")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            entry = pages[self.path]
            self.send_response(200)
            if entry.content_type is not None:
                self.send_header("Content-Type", entry.content_type)
            if entry.compressed:
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(entry.body)))
            self.end_headers()
            self.wfile.write(entry.body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.fixture(scope="module", params=["forge", "deep_research"])
def scraper_module(request):
    root = Path(__file__).resolve().parents[4]
    source = (
        root / "atomic-forge/tools/webpage_scraper/tool/webpage_scraper.py"
        if request.param == "forge"
        else root / "atomic-examples/deep-research/deep_research/tools/webpage_scraper.py"
    )
    name = f"test_encoding_{request.param}_scraper"
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("entry", PAGES, ids=lambda entry: entry.name)
def test_html_encoding_reaches_content_and_metadata(scraper_module, http_pages, entry):
    tool = scraper_module.WebpageScraperTool()
    result = tool.run(scraper_module.WebpageScraperToolInputSchema(url=f"{http_pages}/{entry.name}"))
    assert result.error is None
    assert entry.text in result.content
    assert result.metadata.title == entry.text
    assert result.metadata.author == entry.text
    assert result.metadata.description == entry.text
    assert result.metadata.site_name == entry.text
    assert "\ufffd" not in result.content


def test_fetch_empty_body_remains_an_empty_string(scraper_module, http_pages):
    assert scraper_module.WebpageScraperTool()._fetch_webpage(f"{http_pages}/empty") == ""


def test_content_limit_still_counts_bytes_before_html_decoding(scraper_module, http_pages):
    entry = PAGES[0]
    config = scraper_module.WebpageScraperToolConfig(max_content_length=len(entry.body) - 1)
    tool = scraper_module.WebpageScraperTool(config)
    with pytest.raises(ValueError, match="Content length exceeds"):
        tool._fetch_webpage(f"{http_pages}/{entry.name}")
    config.max_content_length = len(entry.body)
    tool = scraper_module.WebpageScraperTool(config)
    assert entry.text in tool._fetch_webpage(f"{http_pages}/{entry.name}")
