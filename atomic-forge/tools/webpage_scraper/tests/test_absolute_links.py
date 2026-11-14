import importlib.util
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tool import webpage_scraper  # noqa: E402


@pytest.fixture(scope="module", params=["forge", "deep-research"])
def scrape(request):
    module = webpage_scraper
    if request.param == "deep-research":
        path = Path(__file__).resolve().parents[4] / "atomic-examples/deep-research/deep_research/tools/webpage_scraper.py"
        spec = importlib.util.spec_from_file_location("deep_research_webpage_scraper", path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

    def run(**params):
        return module.WebpageScraperTool().run(module.WebpageScraperToolInputSchema(**params))

    return run


@pytest.fixture
def pages(monkeypatch):
    bodies = {}
    redirects = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path in redirects:
                self.send_response(302)
                self.send_header("Location", redirects[self.path])
                self.end_headers()
                return
            content = bodies[self.path].encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def log_message(self, *args):
            pass

    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", bodies, redirects
        finally:
            server.shutdown()
            worker.join()


def document(body, head=""):
    return f"<html><head><title>Linked article</title>{head}</head><body><main>{body}</main></body></html>"


@pytest.mark.parametrize(
    "reference,suffix",
    [
        ("next.html", "/docs/current/next.html"),
        ("../guide.html?x=1#part", "/docs/guide.html?x=1#part"),
        ("/assets/diagram.png", "/assets/diagram.png"),
        ("?edition=2", "/docs/current/page.html?edition=2"),
        ("#section", "/docs/current/page.html?edition=1#section"),
        ("//cdn.example.test/image.png", "http://cdn.example.test/image.png"),
    ],
)
def test_relative_links_and_images_keep_the_document_context(pages, scrape, reference, suffix):
    origin, bodies, _ = pages
    path = "/docs/current/page.html?edition=1"
    bodies[path] = document(f'<p><a href="{reference}">Guide</a><img alt="Diagram" src="{reference}"></p>')

    result = scrape(url=origin + path)

    assert result.error is None
    expected = suffix if urlsplit(suffix).scheme else origin + suffix
    assert f"[Guide]({expected})" in result.content
    assert f"![Diagram]({expected})" in result.content


@pytest.mark.parametrize("other_host", [False, True], ids=["same-host", "other-host"])
def test_redirected_pages_use_the_final_response_url(pages, scrape, other_host):
    origin, bodies, redirects = pages
    final_origin = origin.replace("127.0.0.1", "localhost") if other_host else origin
    redirects["/start"] = final_origin + "/docs/page.html"
    bodies["/docs/page.html"] = document('<p><a href="next.html">Next</a><img alt="Picture" src="picture.png"></p>')

    result = scrape(url=origin + "/start")

    assert result.error is None
    assert f"[Next]({final_origin}/docs/next.html)" in result.content
    assert f"![Picture]({final_origin}/docs/picture.png)" in result.content
    assert result.metadata.domain == urlsplit(final_origin).netloc


@pytest.mark.parametrize(
    "head,base",
    [
        ('<base href="/assets/">', "/assets/"),
        ('<base href="../assets/">', "/docs/assets/"),
        ('<base href="/first/"><base href="/ignored/">', "/first/"),
        ('<base target="_blank"><base href="/assets/">', "/assets/"),
        ('<base href="http://[broken">', "/docs/current/"),
        ('<base href="data:text/plain,ignored"><base href="/ignored/">', "/docs/current/"),
        ('<base href="javascript:ignored">', "/docs/current/"),
        ('<base href="https://assets.example.test/docs/">', "https://assets.example.test/docs/"),
    ],
    ids=[
        "root-base",
        "relative-base",
        "first-base",
        "first-base-with-href",
        "invalid-base",
        "data-base",
        "script-base",
        "absolute-base",
    ],
)
def test_document_base_href_is_applied_before_main_content_extraction(pages, scrape, head, base):
    origin, bodies, _ = pages
    path = "/docs/current/page.html"
    bodies[path] = document('<p><a href="guide.html">Guide</a><img alt="Diagram" src="diagram.png"></p>', head)

    result = scrape(url=origin + path)

    assert result.error is None
    expected = base if urlsplit(base).scheme else origin + base
    assert f"[Guide]({expected}guide.html)" in result.content
    assert f"![Diagram]({expected}diagram.png)" in result.content


def test_excluding_links_keeps_labels_and_resolved_images(pages, scrape):
    origin, bodies, _ = pages
    bodies["/article/page.html"] = document('<p><a href="guide.html">Guide</a><img alt="Diagram" src="diagram.png"></p>')

    result = scrape(url=origin + "/article/page.html", include_links=False)

    assert result.error is None
    assert "Guide" in result.content
    assert "guide.html" not in result.content
    assert f"![Diagram]({origin}/article/diagram.png)" in result.content


def test_absolute_links_and_image_data_keep_their_values(pages, scrape):
    origin, bodies, _ = pages
    bodies["/article"] = document(
        '<p><a href="https://example.test/path?q=1#part">External</a>'
        '<a href="mailto:owner@example.test">Mail</a>'
        '<img alt="Pixel" src="data:image/png;base64,YQ=="></p>'
    )

    result = scrape(url=origin + "/article")

    assert result.error is None
    assert "[External](https://example.test/path?q=1#part)" in result.content
    assert "[Mail](mailto:owner@example.test)" in result.content
    assert "![Pixel](data:image/png;base64,YQ==)" in result.content


def test_one_malformed_link_does_not_discard_the_article(pages, scrape):
    origin, bodies, _ = pages
    bodies["/article"] = document('<p>Retained article.<a href="http://[broken">Bad link</a></p>')

    result = scrape(url=origin + "/article")

    assert result.error is None
    assert "Retained article." in result.content
    assert "[Bad link](http://[broken)" in result.content
