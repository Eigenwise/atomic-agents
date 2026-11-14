"""Keep document/vector associations through the real SDK and a local Qdrant store."""

import base64
import importlib
import json
from pathlib import Path
import struct
import uuid

import httpx
import pytest
from openai import OpenAI

pytest.importorskip("qdrant_client")


DOCUMENTS = ["alpha document", "beta document", "gamma document"]
IDS = [str(uuid.UUID(int=number + 1)) for number in range(3)]
METADATA = [{"source": f"source-{number}"} for number in range(3)]


def vector(index):
    return [1.0 if coordinate == index else 0.0 for coordinate in range(1536)]


def embedding(index, values, encoded=False):
    wire_values = base64.b64encode(struct.pack("<1536f", *values)).decode("ascii") if encoded else values
    return {"object": "embedding", "index": index, "embedding": wire_values}


@pytest.fixture
def service_factory(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    module = importlib.import_module("rag_chatbot.services.qdrant_db")
    opened = []

    def create(batch_indices, encoded=False, query_index=0, identical=False):
        requests = []

        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            assert body["model"] == "text-embedding-3-small"
            if len(body["input"]) == 1:
                document_index = DOCUMENTS.index(body["input"][0])
                entries = [embedding(query_index, vector(0 if identical else document_index), encoded)]
            else:
                assert body["input"] == DOCUMENTS
                entries = [embedding(index, vector(0 if identical else index % 3), encoded) for index in batch_indices]
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": entries,
                    "model": body["model"],
                    "usage": {"prompt_tokens": len(body["input"]), "total_tokens": len(body["input"])},
                },
            )

        http = httpx.Client(transport=httpx.MockTransport(respond))
        native = OpenAI(api_key="local-embedding-fixture", http_client=http)
        # Only supply the transport; embeddings.create and response decoding are native SDK code.
        monkeypatch.setattr(module.openai, "OpenAI", lambda **kwargs: native)
        service = module.QdrantDBService("documents", persist_directory=str(tmp_path / f"qdrant-{len(opened)}"))
        opened.append(service)
        return service, requests

    yield create
    for service in opened:
        service.openai_client.close()
        service.client.close()


@pytest.mark.parametrize("indices", [[0, 1, 2], [2, 0, 1], [1, 0, 2], [2, 1, 0]])
@pytest.mark.parametrize("encoded", [False, True], ids=["float", "base64"])
def test_reordered_response_preserves_stored_payload_and_nearest_document(service_factory, indices, encoded):
    service, requests = service_factory(indices, encoded)

    assert service.add_documents(DOCUMENTS, METADATA, IDS) == IDS

    stored = {point.id: point for point in service.client.retrieve("documents", IDS, with_vectors=True)}
    for index, document_id in enumerate(IDS):
        result = service.query(DOCUMENTS[index], n_results=1)
        assert result["ids"] == [document_id]
        assert result["documents"] == [DOCUMENTS[index]]
        assert result["metadatas"] == [METADATA[index]]
        assert stored[document_id].payload == {"text": DOCUMENTS[index], "metadata": METADATA[index]}
        assert stored[document_id].vector == vector(index)
    assert len(requests) == 4
    assert requests[0]["input"] == DOCUMENTS


@pytest.mark.parametrize("indices", [[0, 0, 2], [2, 2, 0], [-1, 1, 2], [0, 1, 3]])
def test_invalid_same_length_index_set_cannot_write_a_batch(service_factory, indices):
    from qdrant_client.models import PointStruct

    service, requests = service_factory(indices)
    sentinel_id = str(uuid.UUID(int=10))
    sentinel = PointStruct(id=sentinel_id, vector=vector(0), payload={"text": "existing", "metadata": {"source": "kept"}})
    service.client.upsert("documents", [sentinel])

    with pytest.raises(ValueError, match="indices"):
        service.add_documents(DOCUMENTS, METADATA, IDS)

    assert len(requests) == 1
    assert service.client.count("documents").count == 1
    remaining = service.client.retrieve("documents", [sentinel_id], with_vectors=True)
    assert remaining[0].payload == sentinel.payload
    assert remaining[0].vector == sentinel.vector
    assert service.client.retrieve("documents", IDS) == []


@pytest.mark.parametrize("query_index", [0, 1])
def test_query_embedding_index_must_belong_to_the_single_input(service_factory, query_index):
    from qdrant_client.models import PointStruct

    service, requests = service_factory([0, 1, 2], query_index=query_index)
    service.client.upsert(
        "documents", [PointStruct(id=IDS[0], vector=vector(0), payload={"text": DOCUMENTS[0], "metadata": {}})]
    )

    if query_index == 0:
        assert service.query(DOCUMENTS[0], n_results=1)["ids"] == [IDS[0]]
    else:
        with pytest.raises(ValueError, match="indices"):
            service.query(DOCUMENTS[0], n_results=1)
    assert len(requests) == 1
    assert service.client.count("documents").count == 1


def test_identical_vectors_are_kept_for_distinct_valid_indices(service_factory):
    service, _ = service_factory([2, 0, 1], identical=True)
    assert service.add_documents(DOCUMENTS, METADATA, IDS) == IDS
    points = {point.id: point for point in service.client.retrieve("documents", IDS, with_vectors=True)}
    assert len(points) == 3
    for index, document_id in enumerate(IDS):
        assert points[document_id].vector == vector(0)
        assert points[document_id].payload == {"text": DOCUMENTS[index], "metadata": METADATA[index]}
