"""
Context providers for the deep-research pipeline.

Context providers are how runtime state reaches an agent's system prompt.
We use one provider that renders the shared ``ResearchState`` (see
``state.py``) so every agent sees a consistent, up-to-date picture without
having to plumb data through its input schema.

All six agents register the same ``ResearchStateProvider``. The planner
uses it on follow-up turns to extend coverage instead of duplicating it;
on the very first turn the state is empty and the provider renders a
short "no research yet" stub.
"""

from datetime import datetime, timezone

from atomic_agents.context import BaseDynamicContextProvider

from deep_research.state import ResearchState


class ResearchStateProvider(BaseDynamicContextProvider):
    """Renders the current plan, sources, and learnings for agents that need full context."""

    def __init__(self, title: str, state: ResearchState):
        super().__init__(title=title)
        self.state = state

    def get_info(self) -> str:
        if not self.state.plan and not self.state.sources and not self.state.learnings:
            return "No research has been done yet."

        lines: list[str] = []

        if self.state.plan:
            lines.append("### Research plan")
            for sub_topic in self.state.plan:
                status = "sufficient" if sub_topic.sufficient else "needs research"
                lines.append(f"- {sub_topic.name} ({status})")
                lines.extend(f"  - {query}" for query in sub_topic.initial_queries)

        if self.state.sources:
            if lines:
                lines.append("")
            lines.append("### Sources")
            for s in self.state.sources:
                lines.append(f"[{s.id}] {s.title}")
                lines.append(f"      {s.url}")

        if self.state.learnings:
            lines.append("")
            lines.append("### Learnings so far (grouped by sub-topic)")
            seen_topics: list[str] = []
            for learning in self.state.learnings:
                if learning.sub_topic not in seen_topics:
                    seen_topics.append(learning.sub_topic)
            for sub_topic in seen_topics:
                lines.append(f"**{sub_topic}**")
                for learning in self.state.learnings_for(sub_topic):
                    lines.append(f"- {learning.text} [{learning.source_id}]")

        return "\n".join(lines)


class CurrentDateProvider(BaseDynamicContextProvider):
    """So agents don't get confused about what counts as 'recent'."""

    def __init__(self, title: str):
        super().__init__(title=title)

    def get_info(self) -> str:
        return datetime.now(timezone.utc).strftime("Today is %A, %B %d, %Y.")
