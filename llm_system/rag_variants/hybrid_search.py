"""
Hybrid search RAG: BM25 keyword search + dense retrieval fused via Reciprocal Rank Fusion.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from ..embeddings import BaseEmbedder, MockEmbedder
from ..orchestrator import GenerationResponse, LLMClient, LLMOrchestrator, MockLLMClient
from ..retrieval import HybridSearchRetriever
from ..vector_store import BaseVectorStore, InMemoryVectorStore

logger = logging.getLogger(__name__)


class HybridRAGSystem:
    """
    Hybrid RAG pipeline combining BM25 and dense retrieval.

    Retrieval uses Reciprocal Rank Fusion (RRF) to merge ranked lists from
    keyword-based BM25 and embedding-based dense retrieval.

    Parameters
    ----------
    llm_client:
        LLM client for generation.
    embedder:
        Dense embedding provider.
    vector_store:
        Vector store for dense retrieval.
    alpha:
        Weight for dense retrieval in RRF (0 = pure BM25, 1 = pure dense).
    top_k:
        Number of final chunks to provide to the LLM.
    """

    STRATEGY = "hybrid_rag"

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        embedder: Optional[BaseEmbedder] = None,
        vector_store: Optional[BaseVectorStore] = None,
        alpha: float = 0.5,
        top_k: int = 5,
    ) -> None:
        self._llm = llm_client or MockLLMClient()
        self._embedder = embedder or MockEmbedder()
        self._store = vector_store or InMemoryVectorStore()
        self._alpha = alpha
        self._top_k = top_k
        self._retriever = HybridSearchRetriever(self._embedder, self._store, alpha=alpha)
        self._orchestrator = LLMOrchestrator(
            llm_client=self._llm,
            retriever=self._retriever,
            strategy=self.STRATEGY,
        )

    def add_documents(self, texts: List[str], metadatas: Optional[List[Any]] = None) -> None:
        """Index text chunks for both BM25 and dense retrieval."""
        self._retriever.add_texts(texts, metadatas)
        logger.info("HybridRAGSystem: indexed %d chunks (alpha=%.2f)", len(texts), self._alpha)

    def query(self, query: str) -> GenerationResponse:
        """Run the hybrid RAG pipeline for *query*."""
        logger.debug("HybridRAGSystem.query: %s", query[:100])
        response = self._orchestrator.generate(query, top_k=self._top_k)
        response.strategy = self.STRATEGY
        return response
