"""
Unit tests for llm_system.vector_store module.
"""

import math
import uuid

import pytest

from llm_system.embeddings import MockEmbedder
from llm_system.vector_store import InMemoryVectorStore, SearchResult, VectorStoreFactory


def make_chunks(n: int, dim: int = 384):
    """Helper to create n chunk entries with mock embeddings."""
    embedder = MockEmbedder(dimension=dim)
    ids = [str(uuid.uuid4()) for _ in range(n)]
    contents = [f"Chunk content number {i}" for i in range(n)]
    embeddings = embedder.embed_batch(contents)
    return ids, embeddings, contents


class TestInMemoryVectorStore:
    def test_add_and_search(self):
        store = InMemoryVectorStore()
        ids, embeddings, contents = make_chunks(5)
        store.add_documents(ids, embeddings, contents)
        assert store.size == 5

        # Query with first embedding should return first chunk as top result
        results = store.search(embeddings[0], top_k=1)
        assert len(results) == 1
        assert isinstance(results[0], SearchResult)

    def test_cosine_similarity_ordering(self):
        """Results should be ordered by similarity score descending."""
        store = InMemoryVectorStore()
        ids, embeddings, contents = make_chunks(10)
        store.add_documents(ids, embeddings, contents)

        results = store.search(embeddings[0], top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), "Results must be sorted by score descending"

    def test_top_k_respected(self):
        store = InMemoryVectorStore()
        ids, embeddings, contents = make_chunks(20)
        store.add_documents(ids, embeddings, contents)

        for k in [1, 3, 5, 10]:
            results = store.search(embeddings[0], top_k=k)
            assert len(results) == k

    def test_top_k_capped_at_store_size(self):
        store = InMemoryVectorStore()
        ids, embeddings, contents = make_chunks(3)
        store.add_documents(ids, embeddings, contents)

        results = store.search(embeddings[0], top_k=100)
        assert len(results) == 3

    def test_scores_in_valid_range(self):
        store = InMemoryVectorStore()
        ids, embeddings, contents = make_chunks(5)
        store.add_documents(ids, embeddings, contents)

        results = store.search(embeddings[0], top_k=5)
        for r in results:
            assert -1.0 <= r.score <= 1.0 + 1e-6

    def test_search_returns_correct_content(self):
        store = InMemoryVectorStore()
        embedder = MockEmbedder()
        text = "unique specific content for search test"
        emb = embedder.embed_text(text)
        cid = "test_chunk_id"
        store.add_documents([cid], [emb], [text])

        results = store.search(emb, top_k=1)
        assert results[0].chunk_id == cid
        assert results[0].content == text

    def test_delete(self):
        store = InMemoryVectorStore()
        ids, embeddings, contents = make_chunks(5)
        store.add_documents(ids, embeddings, contents)
        assert store.size == 5

        store.delete([ids[0], ids[1]])
        assert store.size == 3

    def test_clear(self):
        store = InMemoryVectorStore()
        ids, embeddings, contents = make_chunks(5)
        store.add_documents(ids, embeddings, contents)
        store.clear()
        assert store.size == 0

    def test_empty_store_returns_empty(self):
        store = InMemoryVectorStore()
        embedder = MockEmbedder()
        results = store.search(embedder.embed_text("query"), top_k=5)
        assert results == []

    def test_metadata_preserved(self):
        store = InMemoryVectorStore()
        embedder = MockEmbedder()
        text = "metadata test"
        emb = embedder.embed_text(text)
        meta = {"source": "doc_001", "page": 1}
        store.add_documents(["meta_chunk"], [emb], [text], [meta])
        results = store.search(emb, top_k=1)
        assert results[0].metadata == meta


class TestVectorStoreFactory:
    def test_create_memory_backend(self):
        store = VectorStoreFactory.create("memory")
        assert isinstance(store, InMemoryVectorStore)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown vector store backend"):
            VectorStoreFactory.create("nonexistent_backend")

    def test_case_insensitive(self):
        store = VectorStoreFactory.create("MEMORY")
        assert isinstance(store, InMemoryVectorStore)
