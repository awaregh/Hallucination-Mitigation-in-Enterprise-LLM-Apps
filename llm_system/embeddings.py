"""
Embedding abstractions supporting mock, OpenAI, and SentenceTransformer backends.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class BaseEmbedder(ABC):
    """Abstract base class for all embedding providers."""

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string and return the embedding vector."""
        ...

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings and return a list of embedding vectors."""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimensionality of the embedding vectors."""
        ...


class MockEmbedder(BaseEmbedder):
    """
    Deterministic mock embedder for testing — returns reproducible pseudo-random
    unit-normalised vectors derived from the text content via hashing.

    Parameters
    ----------
    dimension:
        Size of the embedding vectors (default 384).
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dimension = dimension

    def embed_text(self, text: str) -> List[float]:
        seed = int(abs(hash(text)) % (2**31))
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self._dimension).astype(np.float32)
        vec = vec / (np.linalg.norm(vec) + 1e-10)
        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dimension


class OpenAIEmbedder(BaseEmbedder):
    """
    OpenAI embedding client with automatic retry on transient failures.

    Parameters
    ----------
    model:
        OpenAI embedding model name.
    api_key:
        OpenAI API key (falls back to OPENAI_API_KEY env var if omitted).
    dimension:
        Expected output dimensionality (used for validation).
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        dimension: int = 1536,
    ) -> None:
        try:
            import openai  # noqa: PLC0415
            from tenacity import retry, stop_after_attempt, wait_exponential  # noqa: PLC0415

            self._client = openai.OpenAI(api_key=api_key)
            self._model = model
            self._dimension = dimension
            self._retry = retry(
                stop=stop_after_attempt(3),
                wait=wait_exponential(multiplier=1, min=1, max=10),
                reraise=True,
            )
        except ImportError as exc:
            raise ImportError("openai and tenacity packages are required for OpenAIEmbedder") from exc

    def embed_text(self, text: str) -> List[float]:
        response = self._client.embeddings.create(input=[text], model=self._model)
        return response.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(input=texts, model=self._model)
        return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]

    @property
    def dimension(self) -> int:
        return self._dimension


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    SentenceTransformers embedding client.

    Parameters
    ----------
    model_name:
        HuggingFace model name (e.g. 'all-MiniLM-L6-v2').
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            self._model = SentenceTransformer(model_name)
            self._model_name = model_name
        except ImportError as exc:
            raise ImportError("sentence-transformers package is required for SentenceTransformerEmbedder") from exc

    def embed_text(self, text: str) -> List[float]:
        vec = self._model.encode([text], convert_to_numpy=True)[0]
        return vec.tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        vecs = self._model.encode(texts, convert_to_numpy=True)
        return [v.tolist() for v in vecs]

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()


def get_embedder(provider: str, config: Optional[Dict[str, Any]] = None) -> BaseEmbedder:
    """
    Factory function returning an embedder for the given *provider*.

    Supported providers: ``"mock"``, ``"openai"``, ``"sentence_transformer"``.
    """
    config = config or {}
    provider = provider.lower()
    if provider == "mock":
        return MockEmbedder(**config)
    if provider == "openai":
        return OpenAIEmbedder(**config)
    if provider in {"sentence_transformer", "st"}:
        return SentenceTransformerEmbedder(**config)
    raise ValueError(f"Unknown embedder provider: '{provider}'. Choose from: mock, openai, sentence_transformer")
