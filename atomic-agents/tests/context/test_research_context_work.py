"""The example context renderer preserves text without rescanning every topic."""

import importlib
from pathlib import Path

import pytest

from atomic_agents.context import SystemPromptGenerator


@pytest.fixture
def research(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    monkeypatch.syspath_prepend(str(root / "atomic-examples" / "deep-research"))
    context = importlib.import_module("deep_research.context_providers")
    state = importlib.import_module("deep_research.state")
    return context, state


def original_render(state):
    """Independent transcription of the established text contract."""
    if not state.sources and not state.learnings:
        return "No research has been done yet."
    lines = []
    if state.sources:
        lines.append("### Sources")
        for source in state.sources:
            lines.extend([f"[{source.id}] {source.title}", f"      {source.url}"])
    if state.learnings:
        lines.extend(["", "### Learnings so far (grouped by sub-topic)"])
        topics = dict.fromkeys(learning.sub_topic for learning in state.learnings)
        for topic in topics:
            lines.append(f"**{topic}**")
            lines.extend(
                f"- {learning.text} [{learning.source_id}]" for learning in state.learnings if learning.sub_topic == topic
            )
    return "\n".join(lines)


@pytest.mark.parametrize("with_sources,with_learnings", [(False, False), (True, False), (False, True), (True, True)])
def test_research_context_preserves_complete_text(research, with_sources, with_learnings):
    context, types = research
    state = types.ResearchState(question="Question")
    if with_sources:
        state.register_source("https://example.test/one", "First source")
        state.register_source("https://example.test/two", "Second source")
    if with_learnings:
        state.learnings = [
            types.Learning("First B", "S2", "B"),
            types.Learning("First A", "S1", "A"),
            types.Learning("Second B", "S1", "B"),
            types.Learning("Empty topic", "S2", ""),
            types.Learning("漢字\ncontinued", "S1", "A"),
        ]
    before = repr(state)
    assert context.ResearchStateProvider("Research", state).get_info() == original_render(state)
    assert repr(state) == before


def test_research_context_rebuilds_groups_after_state_changes(research):
    context, types = research
    state = types.ResearchState(question="Question")
    provider = context.ResearchStateProvider("Research", state)
    for index in range(6):
        state.learnings.append(types.Learning(f"Finding {index}", "S1", f"Topic {index % 2}"))
        assert provider.get_info() == original_render(state)
    state.learnings[0].sub_topic = "Reassigned"
    state.learnings[1].text = "Revised claim"
    state.learnings.pop(3)
    assert provider.get_info() == original_render(state)
    state.learnings.reverse()
    assert provider.get_info() == original_render(state)
    state.learnings.clear()
    assert provider.get_info() == "No research has been done yet."


def test_shared_research_provider_keeps_each_agents_prompt_current(research):
    context, types = research
    state = types.ResearchState(question="Question", learnings=[types.Learning("One", "S1", "Topic")])
    provider = context.ResearchStateProvider("Research", state)
    prompts = [SystemPromptGenerator(context_providers={"research": provider}) for _ in range(3)]
    for prompt in prompts:
        assert prompt.generate_prompt().endswith("## Research\n" + original_render(state))
    state.learnings.append(types.Learning("Two", "S2", "Another"))
    for prompt in prompts:
        assert prompt.generate_prompt().endswith("## Research\n" + original_render(state))


def test_research_context_visits_findings_linearly(research):
    context, types = research

    class CountedFindings(list):
        visits = 0

        def __iter__(self):
            for learning in super().__iter__():
                self.visits += 1
                yield learning

    findings = CountedFindings(types.Learning(f"Finding {index}", "S1", f"Topic {index % 600}") for index in range(1200))
    state = types.ResearchState(question="Question", learnings=findings)
    expected = original_render(state)
    findings.visits = 0
    assert context.ResearchStateProvider("Research", state).get_info() == expected
    assert findings.visits <= 2 * len(findings), f"renderer visited {findings.visits} findings"
