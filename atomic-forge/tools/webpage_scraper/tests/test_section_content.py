"""Exercise both scraper implementations with section-scoped HTML content."""

import importlib.util
from pathlib import Path

import pytest
from bs4 import BeautifulSoup


REPOSITORY = Path(__file__).resolve().parents[4]
IMPLEMENTATIONS = [
    "atomic-forge/tools/webpage_scraper/tool/webpage_scraper.py",
    "atomic-examples/deep-research/deep_research/tools/webpage_scraper.py",
]


@pytest.fixture(params=IMPLEMENTATIONS, ids=["forge", "deep-research"])
def scraper(request):
    spec = importlib.util.spec_from_file_location("scraper_section_regression", REPOSITORY / request.param)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("scope", ["main", "article", "section"])
def test_preserves_section_headings_and_attribution(scraper, monkeypatch, scope):
    html = f"""<!doctype html><html><head><title>Research report</title></head><body>
    <header>Site branding<nav>Global navigation</nav></header>
    <{scope}>
      <header id="content"><h1>Measured results</h1><p>By Ada Researcher</p>
        <nav>Article navigation</nav><script>hidden_script()</script>
      </header>
      <p>The measured value was 42.</p>
      <footer class="main"><p>Source: original experiment.</p></footer>
    </{scope}>
    <footer>Site copyright</footer>
    </body></html>"""
    tool = scraper.WebpageScraperTool()
    monkeypatch.setattr(tool, "_fetch_webpage", lambda url: html)

    result = tool.run(scraper.WebpageScraperToolInputSchema(url="https://example.com/report"))

    assert result.error is None
    for text in ["# Measured results", "By Ada Researcher", "The measured value was 42.", "Source: original experiment."]:
        assert text in result.content
    for text in ["Site branding", "Global navigation", "Article navigation", "hidden_script", "Site copyright"]:
        assert text not in result.content


def test_preserves_nested_section_headings(scraper):
    soup = BeautifulSoup(
        """<main><article><header><h1>Study</h1></header><p>Introduction.</p>
        <section><header><h2>Results</h2></header><p>Results body.</p>
        <footer><p>Results attribution.</p></footer></section>
        <footer><p>Study attribution.</p></footer></article></main>""",
        "html.parser",
    )

    extracted = BeautifulSoup(scraper.WebpageScraperTool()._extract_main_content(soup), "html.parser")

    assert extracted.h1.get_text() == "Study"
    assert extracted.h2.get_text() == "Results"
    assert "Results body." in extracted.get_text()
    assert "Results attribution." in extracted.get_text()
    assert "Study attribution." in extracted.get_text()


def test_site_headers_and_footers_remain_excluded(scraper):
    soup = BeautifulSoup(
        """<html><body><header id="content">Site branding<nav>Site navigation</nav></header>
        <p>Body content.</p><footer class="main">Site footer</footer></body></html>""",
        "html.parser",
    )

    extracted = scraper.WebpageScraperTool()._extract_main_content(soup)

    assert "Body content." in extracted
    assert "Site branding" not in extracted
    assert "Site navigation" not in extracted
    assert "Site footer" not in extracted


def test_plain_article_content_stays_intact(scraper):
    soup = BeautifulSoup("<article><h1>Title</h1><p>Paragraph.</p></article>", "html.parser")

    assert scraper.WebpageScraperTool()._extract_main_content(soup) == str(soup.article)
