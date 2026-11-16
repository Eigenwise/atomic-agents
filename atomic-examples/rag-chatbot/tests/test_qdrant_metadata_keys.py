"""Query actual local Qdrant records using literal metadata field names."""

import json
from pathlib import Path
import sys
from uuid import UUID

import httpx
import openai
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_chatbot.services import qdrant_db  # noqa: E402


@pytest.fixture
def service(tmp_path, monkeypatch):
    def embed(request):
        assert request.url.path == "/v1/embeddings"
        texts = json.loads(request.content)["input"]
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "text-embedding-3-small",
                "data": [{"object": "embedding", "index": i, "embedding": [1.0] + [0.0] * 1535} for i in range(len(texts))],
                "usage": {"prompt_tokens": len(texts), "total_tokens": len(texts)},
            },
        )

    client = openai.OpenAI(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(embed)))
    monkeypatch.setattr(qdrant_db.openai, "OpenAI", lambda **kwargs: client)
    database = qdrant_db.QdrantDBService("metadata", persist_directory=str(tmp_path / "qdrant"))
    try:
        yield database
    finally:
        database.client.close()
        client.close()


@pytest.mark.parametrize("key", ["source", "file.name", "tags[0]", "tags[]", "document title", "日本語", ""])
def test_query_preserves_literal_metadata_keys(service, key):
    ids = [str(UUID(int=1)), str(UUID(int=2))]
    metadata = [{key: "wanted"}, {key: "other"}]
    service.add_documents(["Matching document", "Other document"], metadatas=metadata, ids=ids)

    result = service.query("query", where={key: "wanted"})

    assert result["ids"] == ids[:1]
    assert result["documents"] == ["Matching document"]
    assert result["metadatas"] == metadata[:1]
    # Quoting a filter path does not rename stored metadata keys.
    stored = service.client.retrieve(service.collection_name, ids=ids)
    assert [point.payload["metadata"] for point in stored] == metadata


def test_multiple_literal_keys_preserve_and_filtering(service):
    ids = [str(UUID(int=i)) for i in range(1, 4)]
    service.add_documents(
        ["Both predicates", "Only source", "Only status"],
        metadatas=[
            {"file.name": "paper", "review status": "approved"},
            {"file.name": "paper", "review status": "pending"},
            {"file.name": "other", "review status": "approved"},
        ],
        ids=ids,
    )

    result = service.query("query", where={"file.name": "paper", "review status": "approved"})

    assert result["ids"] == ids[:1]
    assert result["documents"] == ["Both predicates"]


@pytest.mark.parametrize("where", [None, {}])
def test_queries_without_filters_still_return_documents(service, where):
    ids = [str(UUID(int=1))]
    service.add_documents(["Document"], metadatas=[{"file.name": "paper"}], ids=ids)

    assert service.query("query", where=where)["ids"] == ids


@pytest.mark.parametrize("key", ['a"b', 'a"."b', "a\\b"])
def test_unrepresentable_keys_are_rejected(service, key):
    service.add_documents(["Document"], metadatas=[{key: "wanted"}], ids=[str(UUID(int=1))])

    with pytest.raises(ValueError):
        service.query("query", where={key: "wanted"})
