"""
LLM orchestration layer combining retrieval, guardrails, and generation.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    def complete(self, prompt: str, system: str = "") -> str:
        """Send a completion request and return the response text."""
        ...


class MockLLMClient(LLMClient):
    """
    Mock LLM client for testing.

    Returns canned responses that include citation markers so the validation
    layer can exercise citation extraction logic.
    """

    CANNED_RESPONSE = (
        "Based on the provided context, the answer to your question is as follows. "
        "According to the documentation [Source: doc_001], the system works by "
        "processing queries through an embedding pipeline. "
        "The vector store [Source: doc_002] stores dense representations for similarity search. "
        "This approach significantly reduces hallucinations compared to a baseline LLM."
    )

    def __init__(self, response_override: Optional[str] = None) -> None:
        self._response = response_override or self.CANNED_RESPONSE
        self._call_count = 0

    def complete(self, prompt: str, system: str = "") -> str:
        self._call_count += 1
        logger.debug("MockLLMClient.complete called (call #%d)", self._call_count)
        return self._response

    @property
    def call_count(self) -> int:
        return self._call_count


class OpenAILLMClient(LLMClient):
    """
    OpenAI LLM client with automatic retry on transient errors.

    Parameters
    ----------
    model:
        OpenAI model identifier.
    temperature:
        Sampling temperature.
    max_tokens:
        Maximum tokens in the completion.
    api_key:
        OpenAI API key.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 1024,
        api_key: Optional[str] = None,
    ) -> None:
        try:
            import openai  # noqa: PLC0415
            from tenacity import retry, stop_after_attempt, wait_exponential  # noqa: PLC0415

            self._client = openai.OpenAI(api_key=api_key)
            self._model = model
            self._temperature = temperature
            self._max_tokens = max_tokens
        except ImportError as exc:
            raise ImportError("openai and tenacity packages are required for OpenAILLMClient") from exc

    def complete(self, prompt: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        return response.choices[0].message.content or ""


@dataclass
class GenerationRequest:
    """All inputs needed to produce a grounded generation."""

    query: str
    context_chunks: List[Any] = field(default_factory=list)  # List[SearchResult]
    system_prompt: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResponse:
    """The output of a grounded generation call."""

    answer: str
    citations: List[str] = field(default_factory=list)
    confidence: float = 0.0
    latency_ms: float = 0.0
    tokens_used: int = 0
    strategy: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


class LLMOrchestrator:
    """
    Combines retriever + guardrails + validator + LLM client into a single
    grounded generation pipeline.

    Parameters
    ----------
    llm_client:
        The LLM client to use for text generation.
    retriever:
        Optional retriever for RAG-based grounding. If None, no context is fetched.
    system_prompt:
        Default system prompt. Can be overridden per request.
    strategy:
        Name of the strategy (used for tagging GenerationResponse).
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful assistant. Answer questions based ONLY on the provided context. "
        "Always cite your sources using [Source: <id>] notation. "
        "If the context does not contain enough information to answer, say so explicitly."
    )

    def __init__(
        self,
        llm_client: LLMClient,
        retriever: Any = None,  # BaseRetriever
        system_prompt: Optional[str] = None,
        strategy: str = "naive_rag",
    ) -> None:
        self._llm = llm_client
        self._retriever = retriever
        self._system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._strategy = strategy

    def generate(self, query: str, top_k: int = 5) -> GenerationResponse:
        """Run the full grounded generation pipeline for *query*."""
        t_start = time.perf_counter()

        # Retrieval
        context_chunks = []
        if self._retriever:
            try:
                context_chunks = self._retriever.retrieve(query, top_k=top_k)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Retrieval failed: %s", exc)

        request = GenerationRequest(
            query=query,
            context_chunks=context_chunks,
            system_prompt=self._system_prompt,
        )

        prompt = self._build_prompt(request)

        try:
            answer = self._llm.complete(prompt, system=self._system_prompt)
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM completion failed: %s", exc)
            answer = "I was unable to generate a response due to an internal error."

        latency_ms = (time.perf_counter() - t_start) * 1000
        citations = self._extract_citations(answer, context_chunks)

        # Rough confidence estimate: citation coverage
        confidence = min(1.0, len(citations) / max(1, len(context_chunks))) if context_chunks else 0.3

        return GenerationResponse(
            answer=answer,
            citations=citations,
            confidence=confidence,
            latency_ms=round(latency_ms, 2),
            tokens_used=self._estimate_tokens(prompt + answer),
            strategy=self._strategy,
        )

    def _build_prompt(self, request: GenerationRequest) -> str:
        """Assemble the full prompt from query and retrieved context chunks."""
        lines = ["## Retrieved Context\n"]
        for i, chunk in enumerate(request.context_chunks, start=1):
            chunk_id = getattr(chunk, "chunk_id", f"chunk_{i}")
            content = getattr(chunk, "content", str(chunk))
            lines.append(f"[Source: {chunk_id}]\n{content}\n")

        if not request.context_chunks:
            lines.append("(No context retrieved.)\n")

        lines.append(f"\n## Question\n{request.query}\n\n## Answer")
        return "\n".join(lines)

    def _extract_citations(self, answer: str, chunks: List[Any]) -> List[str]:
        """Extract citation IDs referenced in *answer* that match retrieved chunks."""
        import re  # noqa: PLC0415

        pattern = re.compile(r"\[Source:\s*([^\]]+)\]")
        referenced = set(pattern.findall(answer))
        chunk_ids = {getattr(c, "chunk_id", "") for c in chunks}
        return sorted(referenced & chunk_ids) if chunk_ids else sorted(referenced)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token count estimate (1 token ≈ 4 characters)."""
        return max(1, len(text) // 4)
