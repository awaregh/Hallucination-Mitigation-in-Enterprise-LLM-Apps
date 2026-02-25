"""
Configuration dataclasses for the hallucination mitigation system.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LLMConfig:
    """Configuration for the LLM client."""

    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 1024
    api_key: Optional[str] = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY"))


@dataclass
class RAGConfig:
    """Configuration for the RAG retrieval pipeline."""

    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    retrieval_strategy: str = "naive"  # naive | hybrid | reranking | multi_query


@dataclass
class GuardrailConfig:
    """Configuration for guardrail enforcement."""

    enable_prompt_guardrails: bool = True
    enable_output_validation: bool = True
    allow_list: List[str] = field(default_factory=list)
    deny_list: List[str] = field(default_factory=list)


@dataclass
class ValidationConfig:
    """Configuration for the validation pipeline."""

    enable_schema_validation: bool = True
    enable_citation_check: bool = True
    enable_self_critique: bool = False  # Disabled by default — adds latency
    max_critique_rounds: int = 2


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation harness."""

    dataset_path: str = "evaluation/datasets/factual_queries.jsonl"
    output_path: str = "results/evaluation_results.json"
    strategies_to_test: List[str] = field(
        default_factory=lambda: ["naive", "hybrid", "reranking", "multi_query"]
    )


@dataclass
class SystemConfig:
    """Top-level system configuration composing all sub-configs."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    guardrail: GuardrailConfig = field(default_factory=GuardrailConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    @classmethod
    def from_env(cls) -> "SystemConfig":
        """Construct a SystemConfig populated from environment variables where relevant."""
        return cls(
            llm=LLMConfig(
                model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
                temperature=float(os.environ.get("LLM_TEMPERATURE", "0.0")),
                max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "1024")),
            )
        )
