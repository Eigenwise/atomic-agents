"""A document batch must never silently truncate before Qdrant upsert."""

import importlib
import json
from pathlib import Path
import sys
import uuid

import httpx
import openai
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def database(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unused-test-key")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    service_type = importlib.import_module("rag_chatbot.services.qdrant_db").QdrantDBService
    service = service_type("documents", persist_directory=str(tmp_path / "vectors"))
    calls = []
    response = {"extra_embeddings": 0}

    def embed(request):
        body = json.loads(request.content)
        calls.append(body)
        count = len(body["input"]) + response["extra_embeddings"]
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "text-embedding-3-small",
                "data": [{"object": "embedding", "index": index, "embedding": [1.0] * 1536} for index in range(count)],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    service.openai_client.close()
    service.openai_client = openai.OpenAI(
        api_key="unused-test-key", http_client=httpx.Client(transport=httpx.MockTransport(embed))
    )
    try:
        yield service, calls, response
    finally:
        service.client.close()
        service.openai_client.close()


@pytest.mark.parametrize("field", ["ids", "metadatas"])
@pytest.mark.parametrize("length", [0, 1, 3])
def test_mismatched_inputs_fail_before_embedding_or_upsert(database, field, length):
    service, calls, _ = database
    values = [str(uuid.uuid4()) for _ in range(length)] if field == "ids" else [{"source": str(i)} for i in range(length)]

    with pytest.raises(ValueError, match="same length"):
        service.add_documents(["first", "second"], **{field: values})

    assert calls == []
    assert service.client.count("documents").count == 0


@pytest.mark.parametrize("extra", [-1, 1])
def test_incomplete_or_extra_embedding_response_does_not_partially_write(database, extra):
    service, calls, response = database
    response["extra_embeddings"] = extra

    with pytest.raises(ValueError):
        service.add_documents(["first", "second"])

    assert len(calls) == 1
    assert service.client.count("documents").count == 0


@pytest.mark.parametrize("provided", [False, True])
def test_matching_documents_ids_and_metadata_are_stored_together(database, provided):
    service, calls, _ = database
    ids = [str(uuid.uuid4()), str(uuid.uuid4())] if provided else None
    metadatas = [{"source": "one"}, {"source": "two"}] if provided else None

    added = service.add_documents(["first", "second"], ids=ids, metadatas=metadatas)

    assert len(added) == 2
    if provided:
        assert added == ids
    assert len(calls) == 1
    assert calls[0]["input"] == ["first", "second"]
    points = {point.id: point.payload for point in service.client.retrieve("documents", ids=added)}
    assert points[added[0]] == {"text": "first", "metadata": {"source": "one"} if provided else {}}
    assert points[added[1]] == {"text": "second", "metadata": {"source": "two"} if provided else {}}
