"""
Vector store abstractions with in-memory (numpy) and ChromaDB backends.
"""

from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single result returned by a vector store search."""

    chunk_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    score: float = 0.0


class BaseVectorStore(ABC):
    """Abstract base class for all vector store backends."""

    @abstractmethod
    def add_documents(
        self,
        chunk_ids: List[str],
        embeddings: List[List[float]],
        contents: List[str],
        metadatas: Optional[List[Dict]] = None,
    ) -> None:
        """Add documents (chunks) with their embeddings to the store."""
        ...

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        """Return the *top_k* most similar chunks for *query_embedding*."""
        ...

    @abstractmethod
    def delete(self, chunk_ids: List[str]) -> None:
        """Remove chunks by id."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Remove all chunks from the store."""
        ...

    @property
    @abstractmethod
    def size(self) -> int:
        """Return the number of stored chunks."""
        ...


class InMemoryVectorStore(BaseVectorStore):
    """
    Pure-numpy in-memory vector store using cosine similarity.

    Suitable for testing and small corpora.  Not thread-safe.
    """

    def __init__(self) -> None:
        self._ids: List[str] = []
        self._embeddings: List[np.ndarray] = []
        self._contents: List[str] = []
        self._metadatas: List[Dict] = []

    def add_documents(
        self,
        chunk_ids: List[str],
        embeddings: List[List[float]],
        contents: List[str],
        metadatas: Optional[List[Dict]] = None,
    ) -> None:
        metadatas = metadatas or [{} for _ in chunk_ids]
        for cid, emb, content, meta in zip(chunk_ids, embeddings, contents, metadatas):
            self._ids.append(cid)
            self._embeddings.append(np.array(emb, dtype=np.float32))
            self._contents.append(content)
            self._metadatas.append(meta)
        logger.debug("Added %d chunks; store size = %d", len(chunk_ids), self.size)

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        if not self._embeddings:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-10)

        matrix = np.vstack(self._embeddings)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        normed = matrix / norms
        scores = normed @ q_norm  # cosine similarity

        top_k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [
            SearchResult(
                chunk_id=self._ids[i],
                content=self._contents[i],
                metadata=self._metadatas[i],
                score=float(scores[i]),
            )
            for i in top_indices
        ]

    def delete(self, chunk_ids: List[str]) -> None:
        id_set = set(chunk_ids)
        keep = [i for i, cid in enumerate(self._ids) if cid not in id_set]
        self._ids = [self._ids[i] for i in keep]
        self._embeddings = [self._embeddings[i] for i in keep]
        self._contents = [self._contents[i] for i in keep]
        self._metadatas = [self._metadatas[i] for i in keep]

    def clear(self) -> None:
        self._ids = []
        self._embeddings = []
        self._contents = []
        self._metadatas = []

    @property
    def size(self) -> int:
        return len(self._ids)


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB-backed vector store.

    Parameters
    ----------
    collection_name:
        Name of the Chroma collection.
    persist_directory:
        If provided, Chroma will persist to this directory.
    """

    def __init__(
        self,
        collection_name: str = "hallucination_mitigation",
        persist_directory: Optional[str] = None,
    ) -> None:
        try:
            import chromadb  # noqa: PLC0415

            if persist_directory:
                self._client = chromadb.PersistentClient(path=persist_directory)
            else:
                self._client = chromadb.EphemeralClient()
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except ImportError as exc:
            raise ImportError("chromadb package is required for ChromaVectorStore") from exc

    def add_documents(
        self,
        chunk_ids: List[str],
        embeddings: List[List[float]],
        contents: List[str],
        metadatas: Optional[List[Dict]] = None,
    ) -> None:
        metadatas = metadatas or [{} for _ in chunk_ids]
        self._collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
        )

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[SearchResult]:
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.size or 1),
            include=["documents", "metadatas", "distances"],
        )
        search_results: List[SearchResult] = []
        for cid, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            search_results.append(
                SearchResult(chunk_id=cid, content=doc, metadata=meta, score=1.0 - dist)
            )
        return search_results

    def delete(self, chunk_ids: List[str]) -> None:
        self._collection.delete(ids=chunk_ids)

    def clear(self) -> None:
        self._collection.delete(where={"$exists": True})  # type: ignore[arg-type]

    @property
    def size(self) -> int:
        return self._collection.count()


class VectorStoreFactory:
    """Factory for creating vector store instances by backend name."""

    @staticmethod
    def create(backend: str, **kwargs: Any) -> BaseVectorStore:
        """
        Create and return a vector store.

        Supported backends: ``"memory"``, ``"chroma"``.
        """
        backend = backend.lower()
        if backend == "memory":
            return InMemoryVectorStore()
        if backend == "chroma":
            return ChromaVectorStore(**kwargs)
        raise ValueError(f"Unknown vector store backend: '{backend}'. Choose from: memory, chroma")
