"""Exercise content selection through both scrapers and real HTTP responses."""

import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest
from bs4 import BeautifulSoup


@pytest.fixture(params=["forge", "deep-research"])
def scraper_module(request):
    root = Path(__file__).resolve().parents[4]
    if request.param == "forge":
        path = root / "atomic-forge/tools/webpage_scraper/tool/webpage_scraper.py"
    else:
        path = root / "atomic-examples/deep-research/deep_research/tools/webpage_scraper.py"
    spec = importlib.util.spec_from_file_location(f"candidate_scraper_{request.param}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def page_server():
    pages = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            content = pages[self.path].encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield pages, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize(
    "body",
    [
        # Drupal Bootstrap's page template puts this skip target before page.content.
        '<div role="main" class="main-container"><section><a id="main-content"></a>'
        "<article><h1>Research report</h1><p>Evidence survives.</p></article></section></div>",
        '<div id="content-placeholder"> \n\t </div><section id="main-content">Evidence survives.</section>',
        '<div class="content-placeholder"><!-- No article here. --></div>'
        '<section class="main-content">Evidence survives.</section>',
        "<main> \n </main><main>Evidence survives.</main>",
        "<article></article><article>Evidence survives.</article>",
        '<div id="content"></div><p>Evidence survives.</p>',
    ],
    ids=["skip-target", "second-id", "second-class", "second-main", "second-article", "body-fallback"],
)
def test_empty_candidates_do_not_discard_the_article(scraper_module, page_server, body):
    pages, url = page_server
    pages["/report"] = f"<html><head><title>Research report</title></head><body>{body}</body></html>"
    result = scraper_module.WebpageScraperTool().run(scraper_module.WebpageScraperToolInputSchema(url=f"{url}/report"))
    assert result.error is None
    assert "Evidence survives." in result.content
    assert result.metadata.title == "Research report"


@pytest.mark.parametrize(
    "body",
    [
        '<main>Selected article.</main><div id="content">Unrelated material.</div>',
        '<div id="content">Selected article.</div><article>Unrelated material.</article>',
        '<section class="content">Selected article.</section><article>Unrelated material.</article>',
    ],
)
def test_nonempty_candidates_keep_their_existing_priority(scraper_module, page_server, body):
    pages, url = page_server
    pages["/report"] = f"<html><head><title>Report</title></head><body>{body}</body></html>"
    result = scraper_module.WebpageScraperTool().run(scraper_module.WebpageScraperToolInputSchema(url=f"{url}/report"))
    assert result.error is None
    assert "Selected article." in result.content
    assert "Unrelated material." not in result.content


@pytest.mark.parametrize("media_tag", ["img", "audio", "video", "iframe", "object", "embed", "svg", "canvas"])
def test_media_only_candidates_are_not_empty(scraper_module, media_tag):
    html = f'<div id="content"><{media_tag} src="/asset"></{media_tag}></div>' "<article>Unrelated material.</article>"
    selected = scraper_module.WebpageScraperTool()._extract_main_content(BeautifulSoup(html, "html.parser"))
    assert f"<{media_tag}" in selected
    assert "Unrelated material." not in selected


def test_image_only_article_survives_the_public_http_path(scraper_module, page_server):
    pages, url = page_server
    pages["/chart"] = (
        '<html><head><title>Chart</title></head><body><img id="main-content" src="/chart.png" alt="">'
        "<article>Unrelated material.</article></body></html>"
    )
    result = scraper_module.WebpageScraperTool().run(scraper_module.WebpageScraperToolInputSchema(url=f"{url}/chart"))
    assert result.error is None
    assert "![](/chart.png)" in result.content
    assert "Unrelated material." not in result.content
