"""The interchangeable RAG backends must report the same cosine distances."""

import importlib
import json
from pathlib import Path
import sys
import uuid

import httpx
import openai
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(params=["qdrant", "chroma"])
def database(request, tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unused-test-key")
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)

    vectors = {
        "query": [2.0, 0.0],
        "same": [3.0, 0.0],
        "near": [4.0, 3.0],
        "orthogonal": [0.0, 2.0],
        "distant": [-4.0, 3.0],
        "opposite": [-3.0, 0.0],
    }

    def embed(http_request):
        body = json.loads(http_request.content)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "text-embedding-3-small",
                "data": [
                    {"object": "embedding", "index": index, "embedding": vectors[text] + [0.0] * 1534}
                    for index, text in enumerate(body["input"])
                ],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )

    client = openai.OpenAI(api_key="unused-test-key", http_client=httpx.Client(transport=httpx.MockTransport(embed)))
    if request.param == "qdrant":
        service_type = importlib.import_module("rag_chatbot.services.qdrant_db").QdrantDBService
        service = service_type("documents", persist_directory=str(tmp_path / "vectors"))
        service.openai_client.close()
        service.openai_client = client
    else:
        service_type = importlib.import_module("rag_chatbot.services.chroma_db").ChromaDBService
        service = service_type("documents", persist_directory=str(tmp_path / "vectors"))
        service.embedding_function.client.close()
        service.embedding_function.client = client

    documents = ["opposite", "near", "distant", "same", "orthogonal"]
    ids = {document: str(uuid.uuid4()) for document in documents}
    metadatas = {
        document: {"source": document, "group": "edge" if document in ("same", "opposite") else "middle"}
        for document in documents
    }
    service.add_documents(documents, metadatas=[metadatas[document] for document in documents], ids=list(ids.values()))
    try:
        yield service, ids, metadatas
    finally:
        client.close()
        if request.param == "qdrant":
            service.client.close()


def test_query_returns_cosine_distances_in_nearest_first_order(database):
    service, ids, metadatas = database

    result = service.query("query", n_results=5)

    ordered = ["same", "near", "orthogonal", "distant", "opposite"]
    assert result["documents"] == ordered
    assert result["ids"] == [ids[document] for document in ordered]
    assert result["metadatas"] == [metadatas[document] for document in ordered]
    assert result["distances"] == pytest.approx([0.0, 0.2, 1.0, 1.8, 2.0], abs=1e-6)


def test_filtered_top_result_keeps_distance_semantics(database):
    service, ids, metadatas = database

    result = service.query("query", n_results=1, where={"group": "middle"})

    assert result["documents"] == ["near"]
    assert result["ids"] == [ids["near"]]
    assert result["metadatas"] == [metadatas["near"]]
    assert result["distances"] == pytest.approx([0.2], abs=1e-6)


def test_filter_without_matches_returns_empty_parallel_lists(database):
    service, _, _ = database

    result = service.query("query", where={"group": "missing"})

    assert result == {"documents": [], "ids": [], "metadatas": [], "distances": []}
