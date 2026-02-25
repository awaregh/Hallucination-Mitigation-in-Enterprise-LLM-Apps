"""
Unit tests for llm_system.embeddings module.
"""

import pytest

from llm_system.embeddings import MockEmbedder, get_embedder


class TestMockEmbedder:
    def test_correct_dimension_default(self):
        embedder = MockEmbedder()
        vec = embedder.embed_text("test text")
        assert len(vec) == 384

    def test_correct_dimension_custom(self):
        embedder = MockEmbedder(dimension=128)
        vec = embedder.embed_text("test text")
        assert len(vec) == 128

    def test_returns_list_of_floats(self):
        embedder = MockEmbedder()
        vec = embedder.embed_text("hello")
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)

    def test_reproducible_for_same_text(self):
        """Same text must always produce the same embedding vector."""
        embedder = MockEmbedder()
        text = "deterministic embedding test"
        vec1 = embedder.embed_text(text)
        vec2 = embedder.embed_text(text)
        assert vec1 == vec2

    def test_different_texts_produce_different_embeddings(self):
        embedder = MockEmbedder()
        vec1 = embedder.embed_text("apple")
        vec2 = embedder.embed_text("orange")
        assert vec1 != vec2

    def test_unit_normalized(self):
        """Embeddings should be approximately unit-normalized."""
        import math  # noqa: PLC0415

        embedder = MockEmbedder()
        vec = embedder.embed_text("normalization test")
        norm = math.sqrt(sum(v ** 2 for v in vec))
        assert abs(norm - 1.0) < 1e-4

    def test_dimension_property(self):
        embedder = MockEmbedder(dimension=256)
        assert embedder.dimension == 256


class TestMockEmbedderBatch:
    def test_embed_batch_returns_list(self):
        embedder = MockEmbedder()
        texts = ["text one", "text two", "text three"]
        result = embedder.embed_batch(texts)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_embed_batch_correct_dimensions(self):
        embedder = MockEmbedder(dimension=64)
        texts = ["a", "b", "c"]
        result = embedder.embed_batch(texts)
        for vec in result:
            assert len(vec) == 64

    def test_embed_batch_consistent_with_single(self):
        """embed_batch should return the same results as individual embed_text calls."""
        embedder = MockEmbedder()
        texts = ["hello world", "foo bar"]
        batch_result = embedder.embed_batch(texts)
        for i, text in enumerate(texts):
            single = embedder.embed_text(text)
            assert batch_result[i] == single

    def test_embed_empty_batch(self):
        embedder = MockEmbedder()
        result = embedder.embed_batch([])
        assert result == []


class TestGetEmbedderFactory:
    def test_get_mock_embedder(self):
        embedder = get_embedder("mock")
        assert isinstance(embedder, MockEmbedder)

    def test_get_mock_with_config(self):
        embedder = get_embedder("mock", {"dimension": 512})
        assert embedder.dimension == 512

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown embedder provider"):
            get_embedder("nonexistent_provider")

    def test_case_insensitive(self):
        embedder = get_embedder("MOCK")
        assert isinstance(embedder, MockEmbedder)
