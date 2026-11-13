"""Collection recreation must preserve the rest of a persistent vector database."""

import importlib
import os
from pathlib import Path
import subprocess
import sys
import uuid

import pytest


EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXAMPLE_ROOT))


def service_type(backend):
    module = importlib.import_module(f"rag_chatbot.services.{backend}_db")
    return getattr(module, "ChromaDBService" if backend == "chroma" else "QdrantDBService")


def seed_store(backend, directory, target_exists):
    """Seed in a separate process so all persistence handles are closed on return."""
    service = service_type(backend)("untouched", persist_directory=str(directory))
    if backend == "chroma":
        service.collection.add(ids=["kept"], embeddings=[[1.0] * 1536], documents=["keep me"])
        if target_exists:
            target = service.client.create_collection("target", embedding_function=service.embedding_function)
            target.add(ids=["old"], embeddings=[[1.0] * 1536], documents=["replace me"])
    else:
        from qdrant_client.models import PointStruct, VectorParams, Distance

        point = PointStruct(id=str(uuid.uuid4()), vector=[1.0] * 1536, payload={"text": "keep me"})
        service.client.upsert("untouched", points=[point])
        if target_exists:
            service.client.create_collection("target", vectors_config=VectorParams(size=1536, distance=Distance.COSINE))
            service.client.upsert("target", points=[point])
        service.client.close()
        service.openai_client.close()
    (directory / "keep.txt").write_text("unrelated file", encoding="utf-8")


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch):
    # Explicit vectors avoid embedding requests; disable Chroma's usage telemetry.
    monkeypatch.setenv("OPENAI_API_KEY", "unused-test-key")
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(params=["chroma", "qdrant"])
def backend(request):
    return request.param


def seed_in_process(backend, directory, target_exists):
    subprocess.run(
        [sys.executable, __file__, backend, str(directory), str(int(target_exists))],
        check=True,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )


def counts(service, backend):
    if backend == "chroma":
        return service.client.get_collection("untouched").count(), service.collection.count()
    return service.client.count("untouched").count, service.client.count("target").count


@pytest.mark.parametrize("target_exists", [False, True])
def test_recreation_keeps_other_collections_and_files(tmp_path, backend, target_exists):
    directory = tmp_path / "vectors"
    seed_in_process(backend, directory, target_exists)

    service = service_type(backend)("target", persist_directory=str(directory), recreate_collection=True)
    try:
        assert counts(service, backend) == (1, 0)
        assert (directory / "keep.txt").read_text(encoding="utf-8") == "unrelated file"
    finally:
        if backend == "qdrant":
            service.client.close()
            service.openai_client.close()


def test_default_keeps_existing_target(tmp_path, backend):
    directory = tmp_path / "vectors"
    seed_in_process(backend, directory, True)

    service = service_type(backend)("target", persist_directory=str(directory))
    try:
        assert counts(service, backend) == (1, 1)
    finally:
        if backend == "qdrant":
            service.client.close()
            service.openai_client.close()


def test_recreation_can_create_new_database(tmp_path, backend):
    service = service_type(backend)("target", persist_directory=str(tmp_path / "new"), recreate_collection=True)
    try:
        if backend == "chroma":
            assert service.collection.count() == 0
        else:
            assert service.client.count("target").count == 0
    finally:
        if backend == "qdrant":
            service.client.close()
            service.openai_client.close()


if __name__ == "__main__":
    seed_store(sys.argv[1], Path(sys.argv[2]), bool(int(sys.argv[3])))
