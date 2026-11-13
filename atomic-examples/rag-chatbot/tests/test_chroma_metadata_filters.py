"""The service's flat metadata predicates must work with the actual Chroma query API."""

import importlib
import json
from pathlib import Path
import sys

import httpx
import openai
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def database(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "unused-test-key")
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    service_type = importlib.import_module("rag_chatbot.services.chroma_db").ChromaDBService
    service = service_type("documents", persist_directory=str(tmp_path / "vectors"))

    def embed(request):
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "text-embedding-3-small",
                "data": [
                    {"object": "embedding", "index": index, "embedding": [1.0] * 1536} for index, _ in enumerate(body["input"])
                ],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    service.embedding_function.client.close()
    service.embedding_function.client = openai.OpenAI(
        api_key="unused-test-key", http_client=httpx.Client(transport=httpx.MockTransport(embed))
    )
    records = [
        ("guide-en", {"source": "guide", "language": "en", "edition": "current"}),
        ("guide-zh", {"source": "guide", "language": "zh", "edition": "current"}),
        ("blog-en", {"source": "blog", "language": "en", "edition": "current"}),
        ("blog-zh", {"source": "blog", "language": "zh", "edition": "old"}),
    ]
    try:
        service.add_documents(
            documents=[identifier for identifier, _ in records],
            ids=[identifier for identifier, _ in records],
            metadatas=[metadata for _, metadata in records],
        )
        yield service
    finally:
        service.embedding_function.client.close()
        service.client._system.stop()


@pytest.mark.parametrize(
    "where,expected",
    [
        (None, {"guide-en", "guide-zh", "blog-en", "blog-zh"}),
        ({}, {"guide-en", "guide-zh", "blog-en", "blog-zh"}),
        ({"source": "guide"}, {"guide-en", "guide-zh"}),
        ({"language": "zh"}, {"guide-zh", "blog-zh"}),
        ({"source": "guide", "language": "zh"}, {"guide-zh"}),
        ({"source": "missing", "language": "zh"}, set()),
        ({"source": "guide", "language": "zh", "edition": "current"}, {"guide-zh"}),
    ],
)
def test_query_applies_every_flat_metadata_predicate(database, where, expected):
    original = where.copy() if where is not None else None

    results = database.query("documentation", n_results=4, where=where)

    assert set(results["ids"]) == expected
    assert set(results["documents"]) == expected
    assert len(results["metadatas"]) == len(expected)
    assert where == original
