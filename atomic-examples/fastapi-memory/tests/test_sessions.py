"""Offline API regressions; only provider completions are mocked."""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import instructor
import openai
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi_memory import main as api  # noqa: E402
from fastapi_memory.lib.schemas import ChatRequest, ChatResponse  # noqa: E402
from atomic_agents import AgentConfig, AtomicAgent  # noqa: E402


@pytest.fixture
def agents(monkeypatch):
    created = []

    def make_agent(is_async):
        provider = (
            openai.AsyncOpenAI(api_key="offline-placeholder") if is_async else openai.OpenAI(api_key="offline-placeholder")
        )
        client = instructor.from_openai(provider)
        agent = AtomicAgent[ChatRequest, ChatResponse](AgentConfig(client=client))
        answer = ChatResponse(response="offline answer", session_id="unused")
        create = AsyncMock(return_value=answer) if is_async else Mock(return_value=answer)
        monkeypatch.setattr(client, "create", create)

        async def chunks(**kwargs):
            yield answer

        partial = Mock(side_effect=chunks)
        monkeypatch.setattr(client, "create_partial", partial)
        created.append((agent, create, partial))
        return agent

    # The old implementation uses a separate synchronous factory for /chat.
    monkeypatch.setattr(api, "create_chat_agent", lambda: make_agent(False), raising=False)
    monkeypatch.setattr(api, "create_async_chat_agent", lambda: make_agent(True))
    return created


@pytest.fixture
def client(agents):
    with TestClient(api.app) as client:
        yield client


def create_session(client, user="alice"):
    response = client.post(f"/users/{user}/sessions")
    assert response.status_code == 200
    return response.json()["session_id"]


@pytest.mark.parametrize("endpoint", ["/chat", "/chat/stream"])
def test_missing_session_id_returns_400_without_creating_an_agent(client, agents, endpoint):
    response = client.post(endpoint, json={"user_id": "alice", "message": "hello"})
    assert response.status_code == 400
    assert not agents


def test_empty_session_is_visible_and_deletable_without_an_agent(client, agents):
    sid = create_session(client)
    assert [item["session_id"] for item in client.get("/users/alice/sessions").json()["sessions"]] == [sid]
    assert client.get(f"/users/alice/sessions/{sid}/history").json()["messages"] == []
    assert client.delete(f"/users/alice/sessions/{sid}").status_code == 200
    assert client.get("/users/alice/sessions").json()["sessions"] == []
    assert client.delete(f"/users/alice/sessions/{sid}").status_code == 404
    assert not agents


@pytest.mark.parametrize("endpoint", ["/chat", "/chat/stream"])
def test_unknown_session_does_not_get_created_by_chat(client, agents, endpoint):
    response = client.post(endpoint, json={"user_id": "alice", "session_id": "unknown", "message": "hello"})
    assert response.status_code == 404
    assert not agents


def test_history_of_unknown_session_is_404_and_has_no_side_effects(client):
    assert client.get("/users/alice/sessions/unknown/history").status_code == 404
    assert api.conversation_history.get("alice", {}) == {}


@pytest.mark.parametrize("first,second", [("/chat", "/chat/stream"), ("/chat/stream", "/chat")])
def test_response_modes_share_model_history(client, agents, first, second):
    sid = create_session(client)
    payload = {"user_id": "alice", "session_id": sid, "message": "Remember ALPHA42"}
    assert client.post(first, json=payload).status_code == 200
    assert client.post(second, json={**payload, "message": "What was the code?"}).status_code == 200
    history = client.get(f"/users/alice/sessions/{sid}/history").json()["messages"]
    assert [message["role"] for message in history] == ["user", "assistant", "user", "assistant"]
    # Inspect the actual request assembled by AtomicAgent, not just the UI history.
    last_agent, create, partial = agents[-1]
    call = partial.call_args if second.endswith("stream") else create.call_args
    assert "ALPHA42" in str(call.kwargs["messages"])
    assert last_agent.history.get_message_count() == 4


def test_sessions_are_isolated_and_deleted_sessions_cannot_be_reused(client, agents):
    first = create_session(client)
    second = create_session(client)
    client.post("/chat", json={"user_id": "alice", "session_id": first, "message": "private ALPHA42"})
    client.post("/chat", json={"user_id": "alice", "session_id": second, "message": "new conversation"})
    assert "ALPHA42" not in str(agents[-1][1].call_args.kwargs["messages"])
    assert client.get(f"/users/bob/sessions/{first}/history").status_code == 404
    assert client.delete(f"/users/alice/sessions/{first}").status_code == 200
    assert client.post("/chat", json={"user_id": "alice", "session_id": first, "message": "reuse"}).status_code == 404


@pytest.mark.asyncio
async def test_same_session_serializes_chat_and_stream_and_releases_lock_on_close(agents):
    async with api.lifespan(api.app):
        sid = (await api.create_session("alice")).session_id
        request = ChatRequest(user_id="alice", session_id=sid, message="stream first")
        stream = await api.chat_stream(request)
        iterator = stream.body_iterator
        await anext(iterator)
        pending = asyncio.create_task(api.chat(ChatRequest(user_id="alice", session_id=sid, message="queued chat")))
        await asyncio.sleep(0)
        assert not pending.done()
        await iterator.aclose()
        result = await asyncio.wait_for(pending, timeout=2)
        assert result.response == "offline answer"
        assert [m["content"] for m in api.conversation_history["alice"][sid]] == ["queued chat", "offline answer"]
        assert "stream first" not in str(agents[-1][1].call_args.kwargs["messages"])


@pytest.mark.asyncio
async def test_delete_waits_for_stream_and_does_not_resurrect_session(agents):
    async with api.lifespan(api.app):
        sid = (await api.create_session("alice")).session_id
        stream = await api.chat_stream(ChatRequest(user_id="alice", session_id=sid, message="hello"))
        iterator = stream.body_iterator
        await anext(iterator)
        deletion = asyncio.create_task(api.delete_session("alice", sid))
        await asyncio.sleep(0)
        assert not deletion.done()
        await iterator.aclose()
        await asyncio.wait_for(deletion, timeout=2)
        assert sid not in api.session_metadata["alice"]
        assert sid not in api.async_sessions["alice"]


@pytest.mark.asyncio
async def test_stream_failure_restores_history_and_allows_next_turn(agents, monkeypatch):
    async with api.lifespan(api.app):
        sid = (await api.create_session("alice")).session_id
        agent = api.get_or_create_async_agent("alice", sid)

        async def broken_chunks(**kwargs):
            raise RuntimeError("offline failure")
            yield  # Make this an async iterator.

        monkeypatch.setattr(agent.client, "create_partial", broken_chunks)
        stream = await api.chat_stream(ChatRequest(user_id="alice", session_id=sid, message="failed message"))
        events = [part async for part in stream.body_iterator]
        assert json.loads(events[-1].removeprefix("data: "))["error"] == "offline failure"
        assert agent.history.get_message_count() == 0
        assert api.conversation_history["alice"][sid] == []
        await api.chat(ChatRequest(user_id="alice", session_id=sid, message="retry"))
        assert "failed message" not in str(agents[-1][1].call_args.kwargs["messages"])


@pytest.mark.asyncio
async def test_other_sessions_can_progress_while_same_session_turns_wait(agents, monkeypatch):
    async with api.lifespan(api.app):
        sid = (await api.create_session("alice")).session_id
        other = (await api.create_session("alice")).session_id
        agent = api.get_or_create_async_agent("alice", sid)
        started, release = asyncio.Event(), asyncio.Event()

        async def complete(**kwargs):
            if not started.is_set():
                started.set()
                await release.wait()
            return ChatResponse(response="offline answer", session_id=sid)

        monkeypatch.setattr(agent.client, "create", AsyncMock(side_effect=complete))
        first = asyncio.create_task(api.chat(ChatRequest(user_id="alice", session_id=sid, message="FIRST42")))
        await asyncio.wait_for(started.wait(), timeout=2)
        second = asyncio.create_task(api.chat(ChatRequest(user_id="alice", session_id=sid, message="second")))
        try:
            await asyncio.sleep(0)
            assert not second.done()
            independent = await asyncio.wait_for(
                api.chat(ChatRequest(user_id="alice", session_id=other, message="other")), timeout=2
            )
            assert independent.response == "offline answer"
        finally:
            release.set()
            await asyncio.gather(first, second)
        assert "FIRST42" in str(agent.client.create.call_args.kwargs["messages"])
        assert [message["content"] for message in api.conversation_history["alice"][sid]] == [
            "FIRST42",
            "offline answer",
            "second",
            "offline answer",
        ]


@pytest.mark.asyncio
async def test_nonstreaming_failure_restores_history_and_releases_lock(agents, monkeypatch):
    from fastapi import HTTPException

    async with api.lifespan(api.app):
        sid = (await api.create_session("alice")).session_id
        agent = api.get_or_create_async_agent("alice", sid)
        create = agent.client.create
        monkeypatch.setattr(agent.client, "create", AsyncMock(side_effect=RuntimeError("offline failure")))
        with pytest.raises(HTTPException) as failure:
            await api.chat(ChatRequest(user_id="alice", session_id=sid, message="failed message"))
        assert failure.value.status_code == 500
        assert agent.history.get_message_count() == 0
        assert api.conversation_history["alice"][sid] == []
        monkeypatch.setattr(agent.client, "create", create)
        await asyncio.wait_for(api.chat(ChatRequest(user_id="alice", session_id=sid, message="retry")), timeout=2)
        assert "failed message" not in str(create.call_args.kwargs["messages"])
