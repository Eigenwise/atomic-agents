import os
import shutil
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from typing import Dict, List, Optional
import uuid

from .base import BaseVectorDBService, QueryResult


class ChromaDBService(BaseVectorDBService):
    """Service for interacting with ChromaDB using OpenAI embeddings."""

    def __init__(
        self,
        collection_name: str,
        persist_directory: str = "./chroma_db",
        recreate_collection: bool = False,
    ) -> None:
        """Initialize ChromaDB service with OpenAI embeddings.

        Args:
            collection_name: Name of the collection to use
            persist_directory: Directory to persist ChromaDB data
            recreate_collection: If True, deletes the collection if it exists before creating
        """
        # Initialize embedding function with OpenAI
        self.embedding_function = OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
        )

        # If recreating, delete the entire persist directory
        if recreate_collection and os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
            os.makedirs(persist_directory)

        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=persist_directory)

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},  # Explicitly set distance metric
        )

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, str]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add documents to the collection.

        Args:
            documents: List of text documents to add
            metadatas: Optional list of metadata dicts for each document
            ids: Optional list of IDs for each document. If not provided, UUIDs will be generated.

        Returns:
            List[str]: The IDs of the added documents
        """
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]

        self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        return ids

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, str]] = None,
    ) -> QueryResult:
        """Query the collection for similar documents.

        Args:
            query_text: Text to find similar documents for
            n_results: Number of results to return
            where: Metadata equality filters; all supplied fields must match.

        Returns:
            QueryResult containing documents, metadata, distances and IDs
        """
        chroma_filter = None
        if where:
            predicates = [{key: value} for key, value in where.items()]
            chroma_filter = predicates[0] if len(predicates) == 1 else {"$and": predicates}

        results = self.collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=chroma_filter,
            include=["documents", "metadatas", "distances"],
        )

        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0],
            "ids": results["ids"][0],
        }

    def delete_collection(self, collection_name: Optional[str] = None) -> None:
        """Delete a collection by name.

        Args:
            collection_name: Name of the collection to delete. If None, deletes the current collection.
        """
        name_to_delete = collection_name if collection_name is not None else self.collection.name
        self.client.delete_collection(name_to_delete)

    def delete_by_ids(self, ids: List[str]) -> None:
        """Delete documents from the collection by their IDs.

        Args:
            ids: List of IDs to delete
        """
        self.collection.delete(ids=ids)


if __name__ == "__main__":
    chroma_db_service = ChromaDBService(collection_name="test", recreate_collection=True)
    added_ids = chroma_db_service.add_documents(
        documents=["Hello, world!", "This is a test document."],
        metadatas=[{"source": "test"}, {"source": "test"}],
    )
    print("Added documents with IDs:", added_ids)

    results = chroma_db_service.query(query_text="Hello, world!")
    print("Query results:", results)

    chroma_db_service.delete_by_ids([added_ids[0]])
    print("Deleted document with ID:", added_ids[0])

    updated_results = chroma_db_service.query(query_text="Hello, world!")
    print("Updated results after deletion:", updated_results)
