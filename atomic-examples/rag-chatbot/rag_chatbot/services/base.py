from abc import ABC, abstractmethod
from typing import Dict, List, Optional, TypedDict


class QueryResult(TypedDict):
    """Nearest-first results with cosine distances (smaller means more similar)."""

    documents: List[str]
    metadatas: List[Dict[str, str]]
    distances: List[float]
    ids: List[str]


class BaseVectorDBService(ABC):
    """Abstract base class for vector database services."""

    @abstractmethod
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
        pass

    @abstractmethod
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
            where: Optional filter criteria

        Returns:
            QueryResult containing documents, metadata, distances and IDs
        """
        pass

    @abstractmethod
    def delete_collection(self, collection_name: Optional[str] = None) -> None:
        """Delete a collection by name.

        Args:
            collection_name: Name of the collection to delete. If None, deletes the current collection.
        """
        pass

    @abstractmethod
    def delete_by_ids(self, ids: List[str]) -> None:
        """Delete documents from the collection by their IDs.

        Args:
            ids: List of IDs to delete
        """
        pass
