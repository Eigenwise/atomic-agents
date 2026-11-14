from unittest.mock import AsyncMock, patch

import pytest

from tool.arxiv_search import (
    ArxivPaper,
    ArxivSearchTool,
    ArxivSearchToolConfig,
    ArxivSearchToolInputSchema,
    ArxivSearchToolOutputSchema,
)


SAMPLE_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.12345v2</id>
    <updated>2024-02-01T10:00:00Z</updated>
    <published>2024-01-20T09:00:00Z</published>
    <title>A Cool Paper About RAG</title>
    <summary>
      We propose a new method for retrieval-augmented generation
      that improves accuracy across many tasks.
    </summary>
    <author><name>Alice Researcher</name></author>
    <author><name>Bob Engineer</name></author>
    <link href="http://arxiv.org/abs/2401.12345v2" rel="alternate" type="text/html"/>
    <link href="http://arxiv.org/pdf/2401.12345v2.pdf" rel="related" title="pdf" type="application/pdf"/>
    <arxiv:primary_category term="cs.AI"/>
    <category term="cs.AI"/>
    <category term="cs.CL"/>
    <arxiv:comment>Accepted at ACL 2024</arxiv:comment>
    <arxiv:journal_ref>ACL 2024</arxiv:journal_ref>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2402.99999v1</id>
    <updated>2024-03-15T11:00:00Z</updated>
    <published>2024-03-10T08:00:00Z</published>
    <title>Another Cool Paper</title>
    <summary>Short abstract.</summary>
    <author><name>Carol Scholar</name></author>
    <link href="http://arxiv.org/abs/2402.99999v1" rel="alternate" type="text/html"/>
    <arxiv:primary_category term="cs.LG"/>
    <category term="cs.LG"/>
  </entry>
</feed>
"""


@pytest.fixture
def tool():
    return ArxivSearchTool(config=ArxivSearchToolConfig())


def test_build_search_query_plain(tool):
    assert tool._build_search_query("agentic LLM", None) == "all:agentic LLM"


def test_build_search_query_with_field_prefix(tool):
    assert tool._build_search_query("ti:RAG", None) == "ti:RAG"


def test_build_search_query_with_category(tool):
    assert tool._build_search_query("RAG", "cs.AI") == "(all:RAG) AND cat:cs.AI"


def test_build_search_query_groups_a_multi_word_query_before_the_category(tool):
    assert tool._build_search_query("diffusion model", "cs.CV") == "(all:diffusion model) AND cat:cs.CV"
    assert tool._build_search_query("ti:RAG OR abs:agents", "cs.AI") == "(ti:RAG OR abs:agents) AND cat:cs.AI"


def test_build_search_query_field_prefix_and_category(tool):
    assert tool._build_search_query("ti:RAG", "cs.AI") == "(ti:RAG) AND cat:cs.AI"


def test_extract_id_strips_version_and_url(tool):
    assert tool._extract_id("http://arxiv.org/abs/2401.12345v2") == "2401.12345"
    assert tool._extract_id("http://arxiv.org/abs/2401.12345") == "2401.12345"


@pytest.mark.parametrize(
    "entry_id, expected",
    [
        ("http://arxiv.org/abs/solv-int/9912010", "solv-int/9912010"),
        ("https://arxiv.org/abs/solv-int/9912010v2", "solv-int/9912010"),
        ("http://arxiv.org/abs/solv-int/9702004v12", "solv-int/9702004"),
        ("http://arxiv.org/abs/hep-th/9901001v2", "hep-th/9901001"),
        ("http://arxiv.org/abs/2401.12345v12", "2401.12345"),
        ("http://arxiv.org/abs/2401.12345", "2401.12345"),
    ],
)
def test_extract_id_removes_only_terminal_version(entry_id, expected):
    assert ArxivSearchTool._extract_id(entry_id) == expected


def test_parse_feed_preserves_distinct_legacy_identifiers():
    feed = SAMPLE_FEED.replace("2401.12345v2", "solv-int/9912010v2").replace("2402.99999v1", "solv-int/9702004v1")
    papers = ArxivSearchTool.parse_feed(feed, "integrable systems")
    assert [paper.arxiv_id for paper in papers] == ["solv-int/9912010", "solv-int/9702004"]
    assert papers[0].abs_url == "http://arxiv.org/abs/solv-int/9912010v2"
    assert papers[0].pdf_url == "http://arxiv.org/pdf/solv-int/9912010v2.pdf"


def test_parse_feed_returns_papers(tool):
    papers = ArxivSearchTool.parse_feed(SAMPLE_FEED, "RAG")
    assert len(papers) == 2

    paper = papers[0]
    assert isinstance(paper, ArxivPaper)
    assert paper.query == "RAG"
    assert paper.arxiv_id == "2401.12345"
    assert paper.title == "A Cool Paper About RAG"
    assert paper.summary.startswith("We propose a new method")
    assert paper.authors == ["Alice Researcher", "Bob Engineer"]
    assert paper.categories == ["cs.AI", "cs.CL"]
    assert paper.primary_category == "cs.AI"
    assert paper.pdf_url == "http://arxiv.org/pdf/2401.12345v2.pdf"
    assert paper.abs_url == "http://arxiv.org/abs/2401.12345v2"
    assert paper.comment == "Accepted at ACL 2024"
    assert paper.journal_ref == "ACL 2024"

    second = papers[1]
    # PDF url is derived when not explicitly present
    assert second.pdf_url == "http://arxiv.org/pdf/2402.99999v1"
    assert second.comment is None


@pytest.mark.asyncio
async def test_run_async_uses_parse_feed(tool):
    """Bypass the network: stub _fetch with parsed sample feed."""

    async def fake_fetch(self, session, query, max_results, sort_by, sort_order, category):
        return ArxivSearchTool.parse_feed(SAMPLE_FEED, query)

    with patch.object(ArxivSearchTool, "_fetch", fake_fetch):
        out = await tool.run_async(ArxivSearchToolInputSchema(queries=["RAG", "agents"]))

    assert isinstance(out, ArxivSearchToolOutputSchema)
    assert len(out.results) == 4  # 2 sample papers x 2 queries
    assert {p.query for p in out.results} == {"RAG", "agents"}


@pytest.mark.asyncio
async def test_run_async_skips_failed_query(tool):
    async def fake_fetch(self, session, query, *_):
        if query == "bad":
            raise Exception("network down")
        return ArxivSearchTool.parse_feed(SAMPLE_FEED, query)

    with patch.object(ArxivSearchTool, "_fetch", fake_fetch):
        out = await tool.run_async(ArxivSearchToolInputSchema(queries=["good", "bad"]))

    assert len(out.results) == 2
    assert all(p.query == "good" for p in out.results)


def test_run_invokes_run_async(tool):
    sentinel = ArxivSearchToolOutputSchema(results=[])
    with patch.object(ArxivSearchTool, "run_async", AsyncMock(return_value=sentinel)):
        out = tool.run(ArxivSearchToolInputSchema(queries=["q"]))
    assert out is sentinel


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
