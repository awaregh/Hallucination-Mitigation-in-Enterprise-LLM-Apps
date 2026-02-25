"""
Reranking RAG: retrieves an over-sampled candidate set, reranks, then generates.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from ..embeddings import BaseEmbedder, MockEmbedder
from ..orchestrator import GenerationResponse, LLMClient, LLMOrchestrator, MockLLMClient
from ..retrieval import NaiveRAGRetriever, RerankingRetriever
from ..vector_store import BaseVectorStore, InMemoryVectorStore

logger = logging.getLogger(__name__)


class RerankingRAGSystem:
    """
    Reranking RAG pipeline.

    Retrieves ``top_k * candidate_multiplier`` candidates with dense retrieval,
    then applies a cross-encoder (mocked as token overlap) to select the best
    ``top_k`` chunks for generation.

    Parameters
    ----------
    llm_client:
        LLM client for generation.
    embedder:
        Dense embedding provider.
    vector_store:
        Vector store backend.
    top_k:
        Number of chunks provided to the LLM after reranking.
    candidate_multiplier:
        Over-sampling factor for the initial retrieval pass.
    """

    STRATEGY = "reranking_rag"

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        embedder: Optional[BaseEmbedder] = None,
        vector_store: Optional[BaseVectorStore] = None,
        top_k: int = 5,
        candidate_multiplier: int = 3,
    ) -> None:
        self._llm = llm_client or MockLLMClient()
        self._embedder = embedder or MockEmbedder()
        self._store = vector_store or InMemoryVectorStore()
        self._top_k = top_k

        base_retriever = NaiveRAGRetriever(self._embedder, self._store)
        self._retriever = RerankingRetriever(base_retriever, candidate_multiplier=candidate_multiplier)
        self._orchestrator = LLMOrchestrator(
            llm_client=self._llm,
            retriever=self._retriever,
            strategy=self.STRATEGY,
        )

    def add_documents(self, texts: List[str], metadatas: Optional[List[Any]] = None) -> None:
        """Index text chunks for retrieval."""
        self._retriever.add_texts(texts, metadatas)
        logger.info("RerankingRAGSystem: indexed %d chunks", len(texts))

    def query(self, query: str) -> GenerationResponse:
        """Run the reranking RAG pipeline for *query*."""
        logger.debug("RerankingRAGSystem.query: %s", query[:100])
        response = self._orchestrator.generate(query, top_k=self._top_k)
        response.strategy = self.STRATEGY
        return response
