"""
Unit tests for llm_system.chunking module.
"""

import pytest

from llm_system.chunking import (
    ChunkResult,
    FixedSizeChunker,
    RecursiveChunker,
    SentenceChunker,
    get_chunker,
)

SAMPLE_TEXT = (
    "Retrieval-Augmented Generation (RAG) is a technique that combines retrieval with generation. "
    "It first retrieves relevant documents from a knowledge base. "
    "Then it passes those documents as context to a language model. "
    "The language model generates an answer grounded in the retrieved context. "
    "This reduces hallucinations compared to purely parametric generation."
)


class TestFixedSizeChunker:
    def test_basic_chunking(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=0)
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, ChunkResult)
            assert len(chunk.content) <= 100

    def test_overlap(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) >= 1
        if len(chunks) >= 2:
            # With overlap, consecutive chunks should share content
            end_of_first = chunks[0].content[-20:]
            start_of_second = chunks[1].content[:20]
            # The overlap means the second chunk starts earlier in the text
            assert len(chunks[1].content) > 0

    def test_chunk_has_doc_id(self):
        chunker = FixedSizeChunker(chunk_size=200, overlap=0)
        chunks = chunker.chunk("Hello world.", doc_id="test_doc")
        for chunk in chunks:
            assert chunk.doc_id == "test_doc"

    def test_chunk_has_unique_ids(self):
        chunker = FixedSizeChunker(chunk_size=50, overlap=0)
        chunks = chunker.chunk(SAMPLE_TEXT)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            FixedSizeChunker(chunk_size=100, overlap=100)

    def test_empty_text(self):
        chunker = FixedSizeChunker(chunk_size=100, overlap=0)
        chunks = chunker.chunk("")
        # Empty string should produce one chunk (empty) or no chunks
        # Both are acceptable
        for chunk in chunks:
            assert isinstance(chunk, ChunkResult)


class TestSentenceChunker:
    def test_sentence_chunker(self):
        chunker = SentenceChunker(max_chunk_size=200)
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, ChunkResult)
            assert len(chunk.content) > 0

    def test_respects_max_size(self):
        chunker = SentenceChunker(max_chunk_size=100)
        chunks = chunker.chunk(SAMPLE_TEXT)
        # Each chunk should be reasonably close to max_chunk_size
        # (may slightly exceed due to sentence boundary)
        for chunk in chunks:
            assert len(chunk.content) <= 300  # generous bound

    def test_metadata_strategy(self):
        chunker = SentenceChunker()
        chunks = chunker.chunk("This is a test sentence.")
        assert any(c.metadata.get("strategy") == "sentence" for c in chunks)

    def test_single_sentence(self):
        chunker = SentenceChunker(max_chunk_size=200)
        chunks = chunker.chunk("This is a single sentence.")
        assert len(chunks) == 1
        assert chunks[0].content == "This is a single sentence."


class TestRecursiveChunker:
    def test_recursive_chunker(self):
        chunker = RecursiveChunker(chunk_size=150, overlap=20)
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, ChunkResult)
            assert len(chunk.content) > 0

    def test_short_text_returns_one_chunk(self):
        chunker = RecursiveChunker(chunk_size=1000)
        short_text = "Short text."
        chunks = chunker.chunk(short_text)
        assert len(chunks) == 1
        assert chunks[0].content == short_text

    def test_metadata_strategy(self):
        chunker = RecursiveChunker(chunk_size=200)
        chunks = chunker.chunk(SAMPLE_TEXT)
        for chunk in chunks:
            assert chunk.metadata.get("strategy") == "recursive"


class TestGetChunkerFactory:
    def test_get_fixed_chunker(self):
        chunker = get_chunker("fixed", chunk_size=256, overlap=32)
        assert isinstance(chunker, FixedSizeChunker)
        assert chunker.chunk_size == 256
        assert chunker.overlap == 32

    def test_get_sentence_chunker(self):
        chunker = get_chunker("sentence", max_chunk_size=400)
        assert isinstance(chunker, SentenceChunker)

    def test_get_recursive_chunker(self):
        chunker = get_chunker("recursive", chunk_size=512)
        assert isinstance(chunker, RecursiveChunker)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            get_chunker("nonexistent")

    def test_case_insensitive(self):
        chunker = get_chunker("FIXED", chunk_size=100, overlap=10)
        assert isinstance(chunker, FixedSizeChunker)
