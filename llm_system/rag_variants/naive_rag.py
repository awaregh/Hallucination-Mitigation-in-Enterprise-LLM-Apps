"""
Naive RAG implementation: embed query -> cosine similarity retrieval -> grounded generation.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from ..embeddings import BaseEmbedder, MockEmbedder
from ..orchestrator import GenerationResponse, LLMClient, LLMOrchestrator, MockLLMClient
from ..retrieval import NaiveRAGRetriever, RetrieverFactory
from ..vector_store import BaseVectorStore, InMemoryVectorStore

logger = logging.getLogger(__name__)


class NaiveRAGSystem:
    """
    Naive RAG pipeline.

    Implements the simplest possible RAG approach:
    1. Embed the query using a dense embedder.
    2. Retrieve the top-K most similar chunks from the vector store.
    3. Build a grounded prompt and generate an answer.

    Parameters
    ----------
    llm_client:
        LLM client for generation.  Defaults to :class:`MockLLMClient`.
    embedder:
        Embedding provider.  Defaults to :class:`MockEmbedder`.
    vector_store:
        Vector store backend.  Defaults to :class:`InMemoryVectorStore`.
    top_k:
        Number of chunks to retrieve.
    """

    STRATEGY = "naive_rag"

    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        embedder: Optional[BaseEmbedder] = None,
        vector_store: Optional[BaseVectorStore] = None,
        top_k: int = 5,
    ) -> None:
        self._llm = llm_client or MockLLMClient()
        self._embedder = embedder or MockEmbedder()
        self._store = vector_store or InMemoryVectorStore()
        self._top_k = top_k
        self._retriever = NaiveRAGRetriever(self._embedder, self._store)
        self._orchestrator = LLMOrchestrator(
            llm_client=self._llm,
            retriever=self._retriever,
            strategy=self.STRATEGY,
        )

    def add_documents(self, texts: List[str], metadatas: Optional[List[Any]] = None) -> None:
        """Index a list of text chunks into the vector store."""
        self._retriever.add_texts(texts, metadatas)
        logger.info("NaiveRAGSystem: indexed %d chunks", len(texts))

    def query(self, query: str) -> GenerationResponse:
        """Run the full naive RAG pipeline for *query*."""
        logger.debug("NaiveRAGSystem.query: %s", query[:100])
        response = self._orchestrator.generate(query, top_k=self._top_k)
        response.strategy = self.STRATEGY
        return response
