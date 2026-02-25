"""
Retrieval layer implementing multiple RAG retrieval strategies via the strategy pattern.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .embeddings import BaseEmbedder, MockEmbedder
from .vector_store import BaseVectorStore, InMemoryVectorStore, SearchResult

logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Abstract base retriever. All strategies implement this interface."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Retrieve the *top_k* most relevant chunks for *query*."""
        ...

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        """Add raw text chunks to the backing store (convenience method)."""
        raise NotImplementedError


class NaiveRAGRetriever(BaseRetriever):
    """
    Simple dense-retrieval RAG: embed the query and perform a cosine similarity
    search against the vector store.
    """

    def __init__(self, embedder: BaseEmbedder, vector_store: BaseVectorStore) -> None:
        self._embedder = embedder
        self._store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        query_emb = self._embedder.embed_text(query)
        return self._store.search(query_emb, top_k=top_k)

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        import uuid  # noqa: PLC0415

        ids = [str(uuid.uuid4()) for _ in texts]
        embeddings = self._embedder.embed_batch(texts)
        self._store.add_documents(ids, embeddings, texts, metadatas)


class HybridSearchRetriever(BaseRetriever):
    """
    Combines BM25 keyword search and dense vector retrieval.

    Results are merged using Reciprocal Rank Fusion (RRF).

    Parameters
    ----------
    embedder:
        Dense embedding provider.
    vector_store:
        Dense vector store.
    alpha:
        Weight for dense retrieval in RRF (0 = pure BM25, 1 = pure dense).
    rrf_k:
        RRF constant k (default 60, as in the original paper).
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        alpha: float = 0.5,
        rrf_k: int = 60,
    ) -> None:
        self._embedder = embedder
        self._store = vector_store
        self._alpha = alpha
        self._rrf_k = rrf_k
        self._corpus: List[str] = []
        self._corpus_ids: List[str] = []
        self._bm25: Any = None

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        import uuid  # noqa: PLC0415

        ids = [str(uuid.uuid4()) for _ in texts]
        embeddings = self._embedder.embed_batch(texts)
        self._store.add_documents(ids, embeddings, texts, metadatas)
        self._corpus.extend(texts)
        self._corpus_ids.extend(ids)
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        try:
            from rank_bm25 import BM25Okapi  # noqa: PLC0415

            tokenized = [doc.lower().split() for doc in self._corpus]
            self._bm25 = BM25Okapi(tokenized)
        except ImportError:
            logger.warning("rank-bm25 not installed; BM25 component will be skipped")
            self._bm25 = None

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        fetch_k = top_k * 3

        # Dense retrieval
        query_emb = self._embedder.embed_text(query)
        dense_results = self._store.search(query_emb, top_k=fetch_k)

        # BM25 retrieval
        bm25_results: List[SearchResult] = []
        if self._bm25 and self._corpus:
            tokenized_query = query.lower().split()
            scores = self._bm25.get_scores(tokenized_query)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:fetch_k]
            for rank, idx in enumerate(top_indices):
                bm25_results.append(
                    SearchResult(
                        chunk_id=self._corpus_ids[idx],
                        content=self._corpus[idx],
                        metadata={},
                        score=float(scores[idx]),
                    )
                )

        return self._rrf_merge(dense_results, bm25_results, top_k)

    def _rrf_merge(
        self,
        dense: List[SearchResult],
        bm25: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        rrf_scores: Dict[str, float] = defaultdict(float)
        result_map: Dict[str, SearchResult] = {}

        for rank, r in enumerate(dense):
            rrf_scores[r.chunk_id] += self._alpha / (self._rrf_k + rank + 1)
            result_map[r.chunk_id] = r

        for rank, r in enumerate(bm25):
            rrf_scores[r.chunk_id] += (1 - self._alpha) / (self._rrf_k + rank + 1)
            result_map.setdefault(r.chunk_id, r)

        sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:top_k]
        merged = []
        for cid in sorted_ids:
            r = result_map[cid]
            merged.append(
                SearchResult(chunk_id=r.chunk_id, content=r.content, metadata=r.metadata, score=rrf_scores[cid])
            )
        return merged


class RerankingRetriever(BaseRetriever):
    """
    Wraps another retriever and applies a cross-encoder reranking step.

    The cross-encoder is mocked in this implementation (scores by query-token
    overlap ratio). Replace ``_cross_encode`` with a real model in production.

    Parameters
    ----------
    base_retriever:
        Underlying retriever that provides the candidate pool.
    candidate_multiplier:
        Fetch ``top_k * candidate_multiplier`` candidates before reranking.
    """

    def __init__(self, base_retriever: BaseRetriever, candidate_multiplier: int = 3) -> None:
        self._base = base_retriever
        self._multiplier = candidate_multiplier

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        candidates = self._base.retrieve(query, top_k=top_k * self._multiplier)
        reranked = sorted(candidates, key=lambda r: self._cross_encode(query, r.content), reverse=True)
        for i, r in enumerate(reranked[:top_k]):
            r.score = self._cross_encode(query, r.content)
        return reranked[:top_k]

    def _cross_encode(self, query: str, passage: str) -> float:
        """Mock cross-encoder: token overlap ratio (jaccard-like)."""
        q_tokens = set(query.lower().split())
        p_tokens = set(passage.lower().split())
        if not q_tokens or not p_tokens:
            return 0.0
        intersection = q_tokens & p_tokens
        return len(intersection) / len(q_tokens | p_tokens)

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        self._base.add_texts(texts, metadatas)


class MultiQueryRetriever(BaseRetriever):
    """
    Generates N query variants using an LLM, retrieves for each, and deduplicates
    results using max-score pooling.

    Parameters
    ----------
    base_retriever:
        Retriever to use for each query variant.
    n_variants:
        Number of query variants to generate.
    llm_client:
        Optional LLM client for generating variants. Falls back to simple rule-based
        expansion if not provided.
    """

    def __init__(
        self,
        base_retriever: BaseRetriever,
        n_variants: int = 3,
        llm_client: Any = None,
    ) -> None:
        self._base = base_retriever
        self._n_variants = n_variants
        self._llm = llm_client

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        variants = self._generate_variants(query)
        seen: Dict[str, SearchResult] = {}
        for variant in variants:
            for r in self._base.retrieve(variant, top_k=top_k):
                if r.chunk_id not in seen or r.score > seen[r.chunk_id].score:
                    seen[r.chunk_id] = r
        merged = sorted(seen.values(), key=lambda r: r.score, reverse=True)
        return merged[:top_k]

    def _generate_variants(self, query: str) -> List[str]:
        if self._llm:
            try:
                prompt = (
                    f"Generate {self._n_variants} alternative phrasings of this query for information retrieval.\n"
                    f"Query: {query}\n"
                    "Return only the rephrased queries, one per line."
                )
                response = self._llm.complete(prompt, system="You are a helpful search query optimizer.")
                lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
                variants = lines[: self._n_variants]
                return [query] + variants
            except Exception as exc:  # noqa: BLE001
                logger.warning("LLM variant generation failed: %s. Falling back to rule-based.", exc)

        # Rule-based fallback: simple expansions
        variants = [query]
        words = query.split()
        if len(words) > 3:
            variants.append(" ".join(words[1:]))  # drop first word
            variants.append(" ".join(words[:-1]))  # drop last word
        if not query.endswith("?"):
            variants.append(query + "?")
        return variants[: self._n_variants + 1]

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        self._base.add_texts(texts, metadatas)


class ContextCompressionRetriever(BaseRetriever):
    """
    Wraps another retriever and filters out chunks with low relevance to the query.

    Uses simple token-overlap scoring as a relevance filter. Replace with a
    trained relevance classifier for production use.

    Parameters
    ----------
    base_retriever:
        Underlying retriever providing the candidate pool.
    relevance_threshold:
        Minimum score (0–1) for a chunk to be retained.
    """

    def __init__(self, base_retriever: BaseRetriever, relevance_threshold: float = 0.05) -> None:
        self._base = base_retriever
        self._threshold = relevance_threshold

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        candidates = self._base.retrieve(query, top_k=top_k * 2)
        filtered = [r for r in candidates if self._relevance_score(query, r.content) >= self._threshold]
        return filtered[:top_k] if filtered else candidates[:top_k]

    def _relevance_score(self, query: str, passage: str) -> float:
        q_tokens = set(query.lower().split())
        p_tokens = set(passage.lower().split())
        if not q_tokens:
            return 0.0
        return len(q_tokens & p_tokens) / len(q_tokens)

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict]] = None) -> None:
        self._base.add_texts(texts, metadatas)


class RetrieverFactory:
    """Factory for creating retriever instances by strategy name."""

    @staticmethod
    def create(
        strategy: str,
        embedder: Optional[BaseEmbedder] = None,
        vector_store: Optional[BaseVectorStore] = None,
        **kwargs: Any,
    ) -> BaseRetriever:
        """
        Create and return a retriever.

        Supported strategies: ``"naive"``, ``"hybrid"``, ``"reranking"``,
        ``"multi_query"``, ``"compression"``.
        """
        embedder = embedder or MockEmbedder()
        vector_store = vector_store or InMemoryVectorStore()
        strategy = strategy.lower()

        if strategy == "naive":
            return NaiveRAGRetriever(embedder, vector_store)
        if strategy == "hybrid":
            return HybridSearchRetriever(embedder, vector_store, **kwargs)
        if strategy == "reranking":
            base = NaiveRAGRetriever(embedder, vector_store)
            return RerankingRetriever(base, **kwargs)
        if strategy == "multi_query":
            base = NaiveRAGRetriever(embedder, vector_store)
            return MultiQueryRetriever(base, **kwargs)
        if strategy == "compression":
            base = NaiveRAGRetriever(embedder, vector_store)
            return ContextCompressionRetriever(base, **kwargs)
        raise ValueError(
            f"Unknown retrieval strategy: '{strategy}'. "
            "Choose from: naive, hybrid, reranking, multi_query, compression"
        )
