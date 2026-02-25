"""
Multi-query RAG: generates query variants, retrieves for each, deduplicates via MMR.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from ..embeddings import BaseEmbedder, MockEmbedder
from ..orchestrator import GenerationResponse, LLMClient, LLMOrchestrator, MockLLMClient
from ..retrieval import MultiQueryRetriever, NaiveRAGRetriever
from ..vector_store import BaseVectorStore, InMemoryVectorStore

logger = logging.getLogger(__name__)


class MultiQueryRAGSystem:
    """
    Multi-query RAG pipeline.

    Generates N rephrased variants of the user query, retrieves chunks for each
    variant independently, then deduplicates results using max-score pooling
    before passing to the LLM for generation.

    Parameters
    ----------
    llm_client:
        LLM client for generation (and optionally for query variant generation).
    embedder:
        Dense embedding provider.
    vector_store:
        Vector store backend.
    n_variants:
        Number of query variants to generate.
    top_k:
        Number of final unique chunks to provide to the LLM.
    use_llm_for_variants:
        If ``True``, use the LLM client to generate query variants.
        If ``False``, uses rule-based expansion (no extra LLM calls).
    """

    STRATEGY = "multi_query_rag"

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        embedder: Optional[BaseEmbedder] = None,
        vector_store: Optional[BaseVectorStore] = None,
        n_variants: int = 3,
        top_k: int = 5,
        use_llm_for_variants: bool = False,
    ) -> None:
        self._llm = llm_client or MockLLMClient()
        self._embedder = embedder or MockEmbedder()
        self._store = vector_store or InMemoryVectorStore()
        self._top_k = top_k

        base_retriever = NaiveRAGRetriever(self._embedder, self._store)
        variant_llm = self._llm if use_llm_for_variants else None
        self._retriever = MultiQueryRetriever(
            base_retriever,
            n_variants=n_variants,
            llm_client=variant_llm,
        )
        self._orchestrator = LLMOrchestrator(
            llm_client=self._llm,
            retriever=self._retriever,
            strategy=self.STRATEGY,
        )

    def add_documents(self, texts: List[str], metadatas: Optional[List[Any]] = None) -> None:
        """Index text chunks for retrieval."""
        self._retriever.add_texts(texts, metadatas)
        logger.info("MultiQueryRAGSystem: indexed %d chunks", len(texts))

    def query(self, query: str) -> GenerationResponse:
        """Run the multi-query RAG pipeline for *query*."""
        logger.debug("MultiQueryRAGSystem.query: %s", query[:100])
        response = self._orchestrator.generate(query, top_k=self._top_k)
        response.strategy = self.STRATEGY
        return response
