import os
import shutil
import uuid
from typing import Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
import openai

from .base import BaseVectorDBService, QueryResult


class QdrantDBService(BaseVectorDBService):
    """Service for interacting with Qdrant using OpenAI embeddings."""

    def __init__(
        self,
        collection_name: str,
        persist_directory: str = "./qdrant_db",
        recreate_collection: bool = False,
    ) -> None:
        """Initialize Qdrant service with OpenAI embeddings.

        Args:
            collection_name: Name of the collection to use
            persist_directory: Directory to persist Qdrant data
            recreate_collection: If True, deletes the collection if it exists before creating
        """
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = "text-embedding-3-small"

        if recreate_collection and os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
            os.makedirs(persist_directory)

        self.client = QdrantClient(path=persist_directory)
        self.collection_name = collection_name
        self._ensure_collection_exists(recreate_collection)

    def _ensure_collection_exists(self, recreate_collection: bool = False) -> None:
        collection_exists = self.client.collection_exists(self.collection_name)

        if recreate_collection and collection_exists:
            self.client.delete_collection(self.collection_name)
            collection_exists = False

        if not collection_exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1536,  # OpenAI text-embedding-3-small dimension
                    distance=Distance.COSINE,
                ),
            )

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        response = self.openai_client.embeddings.create(model=self.embedding_model, input=texts)

        return [embedding.embedding for embedding in response.data]

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, str]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        ids = ids or [str(uuid.uuid4()) for _ in documents]
        metadatas = metadatas or [{} for _ in documents]

        embeddings = self._get_embeddings(documents)

        points = []
        for doc_id, doc, embedding, metadata in zip(ids, documents, embeddings, metadatas):
            point = PointStruct(id=doc_id, vector=embedding, payload={"text": doc, "metadata": metadata})
            points.append(point)

        self.client.upsert(collection_name=self.collection_name, points=points)

        return ids

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, str]] = None,
    ) -> QueryResult:
        query_embedding = self._get_embeddings([query_text])[0]

        filter_condition = None
        if where:
            conditions = []
            for key, value in where.items():
                conditions.append(FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value)))
            if conditions:
                filter_condition = Filter(must=conditions)

        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=n_results,
            query_filter=filter_condition,
            with_payload=True,
        ).points

        # Extract results
        documents = []
        metadatas = []
        distances = []
        ids = []

        for result in search_results:
            documents.append(result.payload["text"])
            metadatas.append(result.payload["metadata"])
            # Qdrant returns cosine similarity; expose the same distance as Chroma.
            distances.append(1.0 - result.score)
            ids.append(result.id)

        return {
            "documents": documents,
            "metadatas": metadatas,
            "distances": distances,
            "ids": ids,
        }

    def delete_collection(self, collection_name: Optional[str] = None) -> None:
        name_to_delete = collection_name if collection_name is not None else self.collection_name
        self.client.delete_collection(name_to_delete)

    def delete_by_ids(self, ids: List[str]) -> None:
        self.client.delete(collection_name=self.collection_name, points_selector=ids)


if __name__ == "__main__":
    qdrant_db_service = QdrantDBService(collection_name="test", recreate_collection=True)
    added_ids = qdrant_db_service.add_documents(
        documents=["Hello, world!", "This is a test document."],
        metadatas=[{"source": "test"}, {"source": "test"}],
    )
    print("Added documents with IDs:", added_ids)

    results = qdrant_db_service.query(query_text="Hello, world!")
    print("Query results:", results)

    qdrant_db_service.delete_by_ids([added_ids[0]])
    print("Deleted document with ID:", added_ids[0])

    updated_results = qdrant_db_service.query(query_text="Hello, world!")
    print("Updated results after deletion:", updated_results)
