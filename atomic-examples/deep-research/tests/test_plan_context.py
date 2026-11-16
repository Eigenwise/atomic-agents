"""The shared prompt must expose the durable research plan as current state."""

from copy import deepcopy
from pathlib import Path

import pytest

from atomic_agents.context import SystemPromptGenerator


@pytest.fixture
def research_types(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from deep_research.context_providers import ResearchStateProvider
    from deep_research.state import Learning, ResearchState, Source, SubTopic

    return ResearchStateProvider, ResearchState, SubTopic, Source, Learning


def _plan(types):
    _, _, sub_topic, _, _ = types
    return [
        sub_topic("Field measurements", ["irrigation field trial", "water use measurements"]),
        sub_topic("Installation costs", ["irrigation capital cost"], sufficient=True),
    ]


PLAN_TEXT = (
    "### Research plan\n"
    "- Field measurements (needs research)\n"
    "  - irrigation field trial\n"
    "  - water use measurements\n"
    "- Installation costs (sufficient)\n"
    "  - irrigation capital cost"
)
SOURCES_TEXT = "### Sources\n[S1] Trial results\n      https://example.org/trial"
LEARNINGS_TEXT = "### Learnings so far (grouped by sub-topic)\n**Field measurements**\n- Water use declined. [S1]"


@pytest.mark.parametrize("sources", [False, True])
@pytest.mark.parametrize("learnings", [False, True])
def test_plan_reaches_the_real_shared_prompt(research_types, sources, learnings):
    provider_type, state_type, _, source_type, learning_type = research_types
    state = state_type(question="A user question stays in its existing input schema.", plan=_plan(research_types))
    if sources:
        state.sources.append(source_type("S1", "https://example.org/trial", "Trial results"))
    if learnings:
        state.learnings.append(learning_type("Water use declined.", "S1", "Field measurements"))
    before = deepcopy(state)
    provider = provider_type("Research State", state)
    expected = PLAN_TEXT
    if sources:
        expected += "\n\n" + SOURCES_TEXT
    if learnings:
        expected += "\n\n" + LEARNINGS_TEXT
    assert provider.get_info() == expected
    prompt = SystemPromptGenerator(context_providers={"research_state": provider}).generate_prompt()
    assert prompt.endswith("## Research State\n" + expected)
    assert state.question not in prompt
    assert state == before


@pytest.mark.parametrize(
    "sources,learnings,expected",
    [
        (False, False, "No research has been done yet."),
        (True, False, SOURCES_TEXT),
        (False, True, "\n" + LEARNINGS_TEXT),
        (True, True, SOURCES_TEXT + "\n\n" + LEARNINGS_TEXT),
    ],
)
def test_empty_plan_preserves_existing_context(research_types, sources, learnings, expected):
    provider_type, state_type, _, source_type, learning_type = research_types
    state = state_type(question="Water use")
    if sources:
        state.sources.append(source_type("S1", "https://example.org/trial", "Trial results"))
    if learnings:
        state.learnings.append(learning_type("Water use declined.", "S1", "Field measurements"))
    assert provider_type("Research State", state).get_info() == expected


def test_live_plan_updates_reach_every_shared_prompt(research_types):
    provider_type, state_type, sub_topic, _, _ = research_types
    state = state_type(question="Water use", plan=_plan(research_types))
    provider = provider_type("Research State", state)
    generators = [SystemPromptGenerator(context_providers={"research_state": provider}) for _ in range(2)]
    for generator in generators:
        assert "Field measurements (needs research)" in generator.generate_prompt()

    state.plan[0].sufficient = True
    state.plan[0].initial_queries.append("long term trial results")
    state.plan.append(sub_topic("Maintenance", []))
    for generator in generators:
        prompt = generator.generate_prompt()
        assert "Field measurements (sufficient)" in prompt
        assert "Field measurements (needs research)" not in prompt
        assert "long term trial results" in prompt
        assert prompt.endswith("- Maintenance (needs research)")
        assert prompt.index("Field measurements") < prompt.index("Installation costs") < prompt.index("Maintenance")

    state.plan.clear()
    for generator in generators:
        prompt = generator.generate_prompt()
        assert "### Research plan" not in prompt
        assert prompt.endswith("No research has been done yet.")
