"""
Text chunking strategies for splitting documents into retrieval-sized units.
"""

from __future__ import annotations

import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ChunkResult:
    """A single chunk produced by a chunker."""

    chunk_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""

    @classmethod
    def create(cls, content: str, doc_id: str = "", metadata: Optional[Dict] = None) -> "ChunkResult":
        return cls(
            chunk_id=str(uuid.uuid4()),
            content=content,
            doc_id=doc_id,
            metadata=metadata or {},
        )


class BaseChunker(ABC):
    """Abstract base class for all chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, doc_id: str = "") -> List[ChunkResult]:
        """Split *text* into a list of ChunkResult objects."""
        ...


class FixedSizeChunker(BaseChunker):
    """
    Splits text into fixed-size character windows with optional overlap.

    Parameters
    ----------
    chunk_size:
        Maximum number of characters per chunk.
    overlap:
        Number of characters to repeat from the end of the previous chunk.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        if overlap >= chunk_size:
            raise ValueError("overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str, doc_id: str = "") -> List[ChunkResult]:
        chunks: List[ChunkResult] = []
        start = 0
        step = self.chunk_size - self.overlap
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            chunks.append(
                ChunkResult.create(
                    content=chunk_text,
                    doc_id=doc_id,
                    metadata={"start_char": start, "end_char": end, "strategy": "fixed_size"},
                )
            )
            start += step
        return chunks


class SentenceChunker(BaseChunker):
    """
    Splits text on sentence boundaries, grouping sentences until a target
    character limit is reached.

    Parameters
    ----------
    max_chunk_size:
        Soft upper-bound on characters per chunk.
    """

    _SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, max_chunk_size: int = 512) -> None:
        self.max_chunk_size = max_chunk_size

    def chunk(self, text: str, doc_id: str = "") -> List[ChunkResult]:
        sentences = self._SENTENCE_RE.split(text.strip())
        chunks: List[ChunkResult] = []
        current_parts: List[str] = []
        current_len = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            if current_len + len(sentence) > self.max_chunk_size and current_parts:
                chunks.append(
                    ChunkResult.create(
                        content=" ".join(current_parts),
                        doc_id=doc_id,
                        metadata={"strategy": "sentence"},
                    )
                )
                current_parts = []
                current_len = 0
            current_parts.append(sentence)
            current_len += len(sentence) + 1

        if current_parts:
            chunks.append(
                ChunkResult.create(
                    content=" ".join(current_parts),
                    doc_id=doc_id,
                    metadata={"strategy": "sentence"},
                )
            )
        return chunks


class RecursiveChunker(BaseChunker):
    """
    Recursively splits text by a hierarchy of separators, falling back to
    character-level splitting if no separator is found within the chunk size.

    Parameters
    ----------
    chunk_size:
        Target maximum chunk size in characters.
    overlap:
        Overlap between consecutive chunks.
    separators:
        Ordered list of separator strings to try.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        separators: Optional[List[str]] = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = separators if separators is not None else self.DEFAULT_SEPARATORS

    def chunk(self, text: str, doc_id: str = "") -> List[ChunkResult]:
        raw_chunks = self._split(text, self.separators)
        results: List[ChunkResult] = []
        for raw in raw_chunks:
            results.append(
                ChunkResult.create(
                    content=raw,
                    doc_id=doc_id,
                    metadata={"strategy": "recursive"},
                )
            )
        return results

    def _split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        separator = ""
        remaining_seps: List[str] = []
        for i, sep in enumerate(separators):
            if sep == "" or sep in text:
                separator = sep
                remaining_seps = separators[i + 1 :]
                break

        if separator == "":
            # Character-level split
            chunks: List[str] = []
            step = self.chunk_size - self.overlap
            for start in range(0, len(text), step):
                chunks.append(text[start : start + self.chunk_size])
            return chunks

        parts = text.split(separator)
        merged: List[str] = []
        current = ""
        for part in parts:
            candidate = (current + separator + part).strip() if current else part.strip()
            if len(candidate) > self.chunk_size and current:
                merged.extend(self._split(current.strip(), remaining_seps or [""]))
                current = part
            else:
                current = candidate
        if current:
            merged.extend(self._split(current.strip(), remaining_seps or [""]))
        return [m for m in merged if m]


def get_chunker(strategy: str, **kwargs: Any) -> BaseChunker:
    """
    Factory function returning a chunker for the given *strategy* name.

    Supported strategies: ``"fixed"``, ``"sentence"``, ``"recursive"``.
    """
    strategy = strategy.lower()
    if strategy == "fixed":
        return FixedSizeChunker(**kwargs)
    if strategy == "sentence":
        return SentenceChunker(**kwargs)
    if strategy == "recursive":
        return RecursiveChunker(**kwargs)
    raise ValueError(f"Unknown chunking strategy: '{strategy}'. Choose from: fixed, sentence, recursive")
