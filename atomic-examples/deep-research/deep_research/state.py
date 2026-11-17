"""
Shared state for the deep-research pipeline.

Every agent in the pipeline reads from — and contributes to — a single
`ResearchState` object. Passing it explicitly through function arguments
(instead of hiding it in globals or on an agent) makes the data flow
inspectable and each pipeline stage easy to reason about in isolation.

The state holds three kinds of data:

- The plan: durable sub-topics the planner produced.
- Accumulated findings: sources we've seen and learnings extracted from them.
- Deduplication sets: queries and URLs already touched, so the search
  loop and the planner don't re-do work on follow-up turns.

Source IDs (``S1``, ``S2``, ...) are assigned when a source is first
registered and are used throughout the pipeline as citation anchors.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Source:
    """A web page we've scraped. ``id`` is referenced by learnings and the final report."""

    id: str
    url: str
    title: str


@dataclass
class Learning:
    """One atomic claim extracted from a single source."""

    text: str
    source_id: str  # must match some Source.id
    sub_topic: str  # the sub-topic this was gathered under


@dataclass
class SubTopic:
    """One durable branch of the research plan. Queries iterate; sub-topics don't."""

    name: str
    initial_queries: list[str]
    sufficient: bool = False  # set by the reflector when further research is unnecessary


@dataclass
class ResearchState:
    question: str
    plan: list[SubTopic] = field(default_factory=list)
    learnings: list[Learning] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)

    # Dedup sets — keep the search loop and the planner from repeating themselves.
    queries_seen: set[str] = field(default_factory=set)
    urls_seen: set[str] = field(default_factory=set)

    # Agent call attempts, including failures — see ResearchBudget.hard_call_cap.
    agent_calls: int = 0

    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def learnings_for(self, sub_topic: str) -> list[Learning]:
        return [learning for learning in self.learnings if learning.sub_topic == sub_topic]

    def register_source(self, url: str, title: str) -> Source:
        """Register a source if new, return the (new or existing) record.

        IDs are stable within a run — once a URL has an ID, it keeps it even
        if the source is looked up again later.
        """
        for s in self.sources:
            if s.url == url:
                return s
        source = Source(id=f"S{len(self.sources) + 1}", url=url, title=title)
        self.sources.append(source)
        self.urls_seen.add(url)
        return source
