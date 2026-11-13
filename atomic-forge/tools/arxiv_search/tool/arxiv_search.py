import asyncio
import contextvars
import logging
import re
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from typing import List, Literal, Optional

import aiohttp
from pydantic import Field

from atomic_agents import BaseIOSchema, BaseTool, BaseToolConfig


logger = logging.getLogger(__name__)


ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


################
# INPUT SCHEMA #
################
class ArxivSearchToolInputSchema(BaseIOSchema):
    """
    Search arXiv for academic papers. Each query is sent to the public arXiv API
    (no key needed) and the matching papers are returned with title, authors,
    abstract, dates, categories, and links to the abstract page and PDF.
    """

    queries: List[str] = Field(
        ...,
        description=(
            "Search queries to run. Use plain keywords (e.g. 'retrieval augmented"
            " generation') or arXiv field-prefixed expressions (e.g. 'ti:LLM AND"
            " abs:agents')."
        ),
    )
    max_results_per_query: int = Field(default=5, ge=1, le=50, description="Maximum number of papers per query.")
    sort_by: Literal["relevance", "lastUpdatedDate", "submittedDate"] = Field(
        default="relevance", description="Sort order for results."
    )
    sort_order: Literal["ascending", "descending"] = Field(default="descending", description="Direction for the chosen sort.")
    category: Optional[str] = Field(
        default=None,
        description=(
            "Optional arXiv category to filter to (e.g. 'cs.AI', 'cs.CL',"
            " 'stat.ML'). Combined with the query as `cat:<value>`."
        ),
    )


####################
# OUTPUT SCHEMA(S) #
####################
class ArxivPaper(BaseIOSchema):
    """A single paper returned by arXiv."""

    query: str = Field(..., description="The query that produced this paper.")
    arxiv_id: str = Field(..., description="arXiv identifier (e.g. '2401.12345').")
    title: str = Field(..., description="Paper title.")
    summary: str = Field(..., description="Abstract / summary.")
    authors: List[str] = Field(..., description="Author names.")
    published: str = Field(..., description="First publication timestamp (ISO 8601).")
    updated: str = Field(..., description="Most recent update timestamp (ISO 8601).")
    categories: List[str] = Field(..., description="arXiv categories (e.g. ['cs.AI']).")
    primary_category: Optional[str] = Field(None, description="Primary arXiv category.")
    pdf_url: str = Field(..., description="Direct URL to the PDF.")
    abs_url: str = Field(..., description="URL to the abstract page.")
    comment: Optional[str] = Field(None, description="Author-supplied comment, if any (e.g. 'Accepted at NeurIPS 2024').")
    journal_ref: Optional[str] = Field(None, description="Journal reference, if provided.")


class ArxivSearchToolOutputSchema(BaseIOSchema):
    """Output of the arXiv search tool."""

    results: List[ArxivPaper] = Field(..., description="Matching papers across all queries.")


#################
# CONFIGURATION #
#################
class ArxivSearchToolConfig(BaseToolConfig):
    """Configuration for the ArxivSearchTool."""

    base_url: str = Field(default="https://export.arxiv.org/api/query", description="Base URL of the arXiv API.")
    user_agent: str = Field(
        default="atomic-agents-arxiv-tool/1.0 (+https://github.com/eigenwise/atomic-agents)",
        description="User agent string for arXiv requests.",
    )
    timeout: float = Field(default=20.0, ge=1.0, le=120.0, description="HTTP request timeout in seconds.")


#####################
# MAIN TOOL & LOGIC #
#####################
class ArxivSearchTool(BaseTool[ArxivSearchToolInputSchema, ArxivSearchToolOutputSchema]):
    """Tool for searching arXiv (free public API)."""

    def __init__(self, config: ArxivSearchToolConfig = ArxivSearchToolConfig()):
        super().__init__(config)
        self.base_url = config.base_url
        self.user_agent = config.user_agent
        self.timeout = config.timeout

    @staticmethod
    def _build_search_query(query: str, category: Optional[str]) -> str:
        # If the user already used field prefixes, leave them alone; otherwise wrap as `all:`
        if any(prefix in query for prefix in ("ti:", "au:", "abs:", "cat:", "all:", "id:")):
            base = query
        else:
            base = f"all:{query}"
        if category:
            # Group the query: arXiv binds `all:` to the first word only and gives AND
            # precedence, so "all:diffusion model AND cat:cs.CV" also matches any paper
            # mentioning "diffusion" in another category.
            base = f"({base}) AND cat:{category}"
        return base

    @staticmethod
    def _extract_id(entry_id: str) -> str:
        """Strip the version suffix and the URL prefix from an arXiv entry id."""
        # entry id looks like: http://arxiv.org/abs/2401.12345v2
        tail = entry_id.rsplit("/abs/", 1)[-1]
        return re.sub(r"v[0-9]+$", "", tail)

    @classmethod
    def _parse_entry(cls, entry: ET.Element, query: str) -> ArxivPaper:
        def _text(tag: str) -> str:
            el = entry.find(tag, ATOM_NS)
            return (el.text or "").strip() if el is not None else ""

        entry_id = _text("atom:id")
        if re.match(r"^https?://arxiv\.org/api/errors(?:#|$)", entry_id):
            raise ValueError(f"arXiv API error: {_text('atom:summary') or 'unspecified API error'}")
        arxiv_id = cls._extract_id(entry_id)

        title = " ".join(_text("atom:title").split())
        summary = " ".join(_text("atom:summary").split())

        authors = [
            (a.findtext("atom:name", default="", namespaces=ATOM_NS) or "").strip()
            for a in entry.findall("atom:author", ATOM_NS)
        ]
        authors = [name for name in authors if name]

        categories = [c.attrib.get("term", "") for c in entry.findall("atom:category", ATOM_NS) if c.attrib.get("term")]

        primary_el = entry.find("arxiv:primary_category", ATOM_NS)
        primary_category = primary_el.attrib.get("term") if primary_el is not None else None

        pdf_url = ""
        abs_url = entry_id  # default to the id URL
        for link in entry.findall("atom:link", ATOM_NS):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href", "")
            elif link.attrib.get("rel") == "alternate":
                abs_url = link.attrib.get("href", abs_url)
        if not pdf_url and abs_url:
            pdf_url = abs_url.replace("/abs/", "/pdf/")

        comment_el = entry.find("arxiv:comment", ATOM_NS)
        journal_el = entry.find("arxiv:journal_ref", ATOM_NS)

        return ArxivPaper(
            query=query,
            arxiv_id=arxiv_id,
            title=title,
            summary=summary,
            authors=authors,
            published=_text("atom:published"),
            updated=_text("atom:updated"),
            categories=categories,
            primary_category=primary_category,
            pdf_url=pdf_url,
            abs_url=abs_url,
            comment=(comment_el.text or "").strip() if comment_el is not None and comment_el.text else None,
            journal_ref=(journal_el.text or "").strip() if journal_el is not None and journal_el.text else None,
        )

    @classmethod
    def parse_feed(cls, xml_text: str, query: str) -> List[ArxivPaper]:
        """Parse the Atom feed XML returned by arXiv into a list of ArxivPaper."""
        root = ET.fromstring(xml_text)
        return [cls._parse_entry(entry, query) for entry in root.findall("atom:entry", ATOM_NS)]

    async def _fetch(
        self,
        session: aiohttp.ClientSession,
        query: str,
        max_results: int,
        sort_by: str,
        sort_order: str,
        category: Optional[str],
    ) -> List[ArxivPaper]:
        params = {
            "search_query": self._build_search_query(query, category),
            "start": "0",
            "max_results": str(max_results),
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        async with session.get(self.base_url, params=params) as resp:
            if resp.status != 200:
                raise Exception(f"arXiv search failed for '{query}': {resp.status} {resp.reason}")
            text = await resp.text()
        return self.parse_feed(text, query)

    async def run_async(self, params: ArxivSearchToolInputSchema) -> ArxivSearchToolOutputSchema:
        headers = {"User-Agent": self.user_agent, "Accept": "application/atom+xml"}
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        async with aiohttp.ClientSession(headers=headers, timeout=timeout) as session:
            tasks = [
                self._fetch(
                    session,
                    q,
                    params.max_results_per_query,
                    params.sort_by,
                    params.sort_order,
                    params.category,
                )
                for q in params.queries
            ]
            grouped = await asyncio.gather(*tasks, return_exceptions=True)

        results: List[ArxivPaper] = []
        for query, group in zip(params.queries, grouped):
            if isinstance(group, Exception):
                logger.warning("arXiv query '%s' failed: %s", query, group)
                continue
            results.extend(group)
        return ArxivSearchToolOutputSchema(results=results)

    def run(self, params: ArxivSearchToolInputSchema) -> ArxivSearchToolOutputSchema:
        with ThreadPoolExecutor() as executor:
            return executor.submit(contextvars.copy_context().run, asyncio.run, self.run_async(params)).result()


#################
# EXAMPLE USAGE #
#################
if __name__ == "__main__":  # pragma: no cover
    from rich.console import Console

    console = Console()
    tool = ArxivSearchTool()

    output = tool.run(
        ArxivSearchToolInputSchema(
            queries=["retrieval augmented generation", "agentic LLM tool use"],
            max_results_per_query=2,
            category="cs.AI",
        )
    )
    for paper in output.results:
        console.rule(f"[bold cyan]{paper.arxiv_id} — {paper.title}")
        console.print(f"[bold]Authors:[/bold] {', '.join(paper.authors)}")
        console.print(f"[bold]Published:[/bold] {paper.published}  [bold]Updated:[/bold] {paper.updated}")
        console.print(f"[bold]Categories:[/bold] {', '.join(paper.categories)}")
        console.print(f"[bold]Abstract URL:[/bold] {paper.abs_url}")
        console.print(f"[bold]PDF URL:[/bold] {paper.pdf_url}")
        console.print(paper.summary)
