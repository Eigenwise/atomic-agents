import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def pipeline(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    module = importlib.import_module("deep_research.main")
    monkeypatch.setattr(module, "console", Mock())
    for name in ("planner", "extractor", "reflector", "writer", "decider", "qa"):
        monkeypatch.setattr(module, f"{name}_agent", Mock())
    return module


@pytest.fixture
def state(pipeline):
    value = pipeline.ResearchState(question="offline research")
    value.register_source("https://example.com/first", "First")
    value.register_source("https://example.com/second", "Second")
    return value


def test_extraction_stops_at_the_last_available_call(pipeline, state, monkeypatch):
    monkeypatch.setattr(pipeline.ResearchBudget, "hard_call_cap", 1)
    pipeline.extractor_agent.run.return_value = SimpleNamespace(claims=["Supported claim"])
    topic = pipeline.SubTopic("Topic", ["query"])

    with pytest.raises(RuntimeError, match="Hard call cap"):
        pipeline.extract_claims(topic, [("S1", "first page"), ("S2", "second page")], state)

    assert pipeline.extractor_agent.run.call_count == state.agent_calls == 1
    assert [(learning.text, learning.source_id) for learning in state.learnings] == [("Supported claim", "S1")]


def test_writer_cannot_start_verification_after_budget_is_used(pipeline, state, monkeypatch):
    monkeypatch.setattr(pipeline.ResearchBudget, "hard_call_cap", 1)
    pipeline.writer_agent.run.return_value = SimpleNamespace(headline="Draft", report="Draft report")

    with pytest.raises(RuntimeError, match="Hard call cap"):
        pipeline.write_report(state)

    assert pipeline.writer_agent.run.call_count == state.agent_calls == 1


@pytest.mark.parametrize("stage", ["plan", "reflect", "qa"])
def test_exhausted_budget_blocks_every_direct_stage(pipeline, state, monkeypatch, stage):
    monkeypatch.setattr(pipeline.ResearchBudget, "hard_call_cap", 1)
    state.agent_calls = 1
    pipeline.planner_agent.run.return_value = SimpleNamespace(sub_topics=[])
    pipeline.reflector_agent.run.return_value = SimpleNamespace(reasoning="Done", sufficient=True, next_queries=[])
    pipeline.qa_agent.run.return_value = SimpleNamespace(answer="Answer", follow_up_questions=[])

    with pytest.raises(RuntimeError, match="Hard call cap"):
        if stage == "plan":
            pipeline.plan_research(state)
        elif stage == "reflect":
            pipeline.reflect(pipeline.SubTopic("Topic", ["query"]), 1, state)
        else:
            pipeline.answer_from_state("follow-up", state)

    assert state.agent_calls == 1
    pipeline.planner_agent.run.assert_not_called()
    pipeline.reflector_agent.run.assert_not_called()
    pipeline.qa_agent.run.assert_not_called()


@pytest.mark.parametrize("needs_research,cap", [(False, 1), (True, 2)])
def test_follow_up_routes_share_the_remaining_budget(pipeline, state, monkeypatch, needs_research, cap):
    monkeypatch.setattr(pipeline.ResearchBudget, "hard_call_cap", cap)
    pipeline.decider_agent.run.return_value = SimpleNamespace(reasoning="Route", needs_research=needs_research)
    pipeline.planner_agent.run.return_value = SimpleNamespace(sub_topics=[])
    pipeline.qa_agent.run.return_value = SimpleNamespace(answer="Answer", follow_up_questions=[])

    with pytest.raises(RuntimeError, match="Hard call cap"):
        pipeline.handle_follow_up("follow-up", state, Mock(), Mock())

    assert state.agent_calls == cap
    pipeline.decider_agent.run.assert_called_once()
    assert pipeline.planner_agent.run.call_count == int(needs_research)
    pipeline.qa_agent.run.assert_not_called()


def test_failed_agent_attempt_consumes_its_slot(pipeline, state, monkeypatch):
    monkeypatch.setattr(pipeline.ResearchBudget, "hard_call_cap", 1)
    pipeline.planner_agent.run.side_effect = RuntimeError("provider failed")

    with pytest.raises(RuntimeError, match="provider failed"):
        pipeline.plan_research(state)
    assert state.agent_calls == 1

    with pytest.raises(RuntimeError, match="Hard call cap"):
        pipeline.plan_research(state)
    pipeline.planner_agent.run.assert_called_once()


@pytest.mark.parametrize("cap", [3, 6])
def test_pipeline_respects_budget_between_stages(pipeline, state, monkeypatch, cap):
    monkeypatch.setattr(pipeline.ResearchBudget, "hard_call_cap", cap)
    pipeline.planner_agent.run.return_value = SimpleNamespace(
        sub_topics=[SimpleNamespace(name="Topic", initial_queries=["query"])]
    )
    pipeline.extractor_agent.run.return_value = SimpleNamespace(claims=["Supported claim"])
    pipeline.reflector_agent.run.return_value = SimpleNamespace(reasoning="Done", sufficient=True, next_queries=[])
    pipeline.writer_agent.run.return_value = SimpleNamespace(headline="Report", report="Report [S1] [S2]")
    monkeypatch.setattr(pipeline, "search_and_scrape", Mock(return_value=[("S1", "first"), ("S2", "second")]))

    if cap == 3:
        with pytest.raises(RuntimeError, match="Hard call cap"):
            pipeline.run_initial_pipeline("question", state, Mock(), Mock())
        pipeline.reflector_agent.run.assert_not_called()
        pipeline.writer_agent.run.assert_not_called()
    else:
        pipeline.run_initial_pipeline("question", state, Mock(), Mock())
        assert pipeline.writer_agent.run.call_count == 2

    assert state.agent_calls == cap
    assert len(state.learnings) == 2
