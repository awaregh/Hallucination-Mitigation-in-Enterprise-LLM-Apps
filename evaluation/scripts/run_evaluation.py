#!/usr/bin/env python3
"""
CLI script for running the hallucination mitigation evaluation harness.

Usage:
    python evaluation/scripts/run_evaluation.py \
        --dataset evaluation/datasets/factual_queries.jsonl \
        --strategy all \
        --output results/factual_results.json \
        --mock
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from evaluation.harness import EvaluationHarness, MetricsSummary
from llm_system.orchestrator import MockLLMClient
from llm_system.rag_variants import (
    HybridRAGSystem,
    MultiQueryRAGSystem,
    NaiveRAGSystem,
    RerankingRAGSystem,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Sample documents indexed into all systems for demo purposes
SAMPLE_DOCUMENTS = [
    "Retrieval-Augmented Generation (RAG) combines a retrieval component with a generative language model. "
    "The retriever fetches relevant documents from a knowledge base using embedding similarity. "
    "The generator then conditions on these documents to produce grounded answers. [Source: doc_001]",

    "Vector databases store dense embedding representations of text chunks and support efficient "
    "approximate nearest neighbor (ANN) search. Popular vector databases include Chroma, Pinecone, "
    "Weaviate, and FAISS. [Source: doc_002]",

    "Cosine similarity measures the angle between two embedding vectors. It ranges from -1 to 1, "
    "where 1 means identical direction. It is preferred for semantic search because it is magnitude-invariant. "
    "[Source: doc_003]",

    "BM25 is a keyword-based ranking function that scores documents by term frequency and inverse document "
    "frequency. Hybrid search combines BM25 with dense retrieval to get the benefits of both lexical "
    "and semantic matching. [Source: doc_004]",

    "Hallucinations in LLMs are outputs that are factually incorrect or not grounded in the provided context. "
    "They are a major risk in enterprise deployments, especially in legal, medical, and financial domains. "
    "[Source: doc_005]",

    "Chunking strategies determine how documents are split for indexing. Fixed-size chunking splits by "
    "character count with overlap. Sentence chunking splits on sentence boundaries. Recursive chunking "
    "uses a hierarchy of separators. [Source: doc_006]",

    "Reciprocal Rank Fusion (RRF) merges multiple ranked lists by computing 1/(k + rank) for each item "
    "in each list and summing scores. It is robust to different score scales and is widely used in "
    "hybrid search systems. [Source: doc_007]",

    "Self-critique prompts the LLM to review its own output against the provided context, identifying "
    "potential hallucinations, unsupported claims, or missing citations. It adds latency but improves "
    "output quality. [Source: doc_008]",

    "LangChain is a framework for building LLM applications, providing abstractions for chains, agents, "
    "retrievers, and memory. It integrates with OpenAI, Anthropic, and open-source models. [Source: doc_009]",

    "Temperature in LLM inference controls output randomness. Lower temperature (near 0) produces more "
    "deterministic, conservative outputs. Higher temperature increases diversity but also hallucination risk. "
    "Enterprise applications typically use temperature 0.0 to 0.2. [Source: doc_010]",
]


def build_systems(mock: bool = True) -> dict:
    """Build all RAG system instances, optionally using mock LLM."""
    llm = MockLLMClient() if mock else None  # Would be OpenAILLMClient() in production

    systems = {
        "naive_rag": NaiveRAGSystem(llm_client=llm),
        "hybrid_rag": HybridRAGSystem(llm_client=llm),
        "reranking_rag": RerankingRAGSystem(llm_client=llm),
        "multi_query_rag": MultiQueryRAGSystem(llm_client=llm),
    }

    # Index sample documents into all systems
    for name, system in systems.items():
        system.add_documents(SAMPLE_DOCUMENTS)
        logger.info("Indexed %d documents into %s", len(SAMPLE_DOCUMENTS), name)

    return systems


def print_results_table(summaries: list[MetricsSummary]) -> None:
    """Print a formatted results table to stdout."""
    try:
        from rich.console import Console  # noqa: PLC0415
        from rich.table import Table  # noqa: PLC0415

        console = Console()
        table = Table(title="Hallucination Mitigation Evaluation Results", show_header=True)
        table.add_column("Strategy", style="cyan")
        table.add_column("Queries", justify="right")
        table.add_column("Hallucination Rate", justify="right", style="red")
        table.add_column("Citation Accuracy", justify="right", style="green")
        table.add_column("Avg Latency (ms)", justify="right")
        table.add_column("Avg Tokens", justify="right")
        table.add_column("Refusal Rate", justify="right")
        table.add_column("Validation Pass", justify="right", style="green")

        for s in summaries:
            table.add_row(
                s.strategy,
                str(s.n_queries),
                f"{s.hallucination_rate:.1%}",
                f"{s.citation_accuracy:.1%}",
                f"{s.avg_latency_ms:.0f}",
                f"{s.avg_tokens:.0f}",
                f"{s.refusal_rate:.1%}",
                f"{s.validation_pass_rate:.1%}",
            )

        console.print(table)
    except ImportError:
        # Fallback to plain text
        print(f"\n{'Strategy':<20} {'Queries':>8} {'Hallu%':>10} {'Cite%':>10} {'Latency':>10}")
        print("-" * 70)
        for s in summaries:
            print(
                f"{s.strategy:<20} {s.n_queries:>8} "
                f"{s.hallucination_rate:>10.1%} {s.citation_accuracy:>10.1%} "
                f"{s.avg_latency_ms:>10.0f}ms"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run hallucination mitigation evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="evaluation/datasets/factual_queries.jsonl",
        help="Path to JSONL dataset file",
    )
    parser.add_argument(
        "--strategy",
        choices=["naive", "hybrid", "reranking", "multi_query", "all"],
        default="all",
        help="Which strategy to test",
    )
    parser.add_argument(
        "--output",
        default="results/evaluation_results.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Use mock LLM client (no API key required)",
    )
    args = parser.parse_args()

    all_systems = build_systems(mock=args.mock)

    if args.strategy == "all":
        systems = all_systems
    else:
        strategy_map = {
            "naive": "naive_rag",
            "hybrid": "hybrid_rag",
            "reranking": "reranking_rag",
            "multi_query": "multi_query_rag",
        }
        key = strategy_map[args.strategy]
        systems = {key: all_systems[key]}

    harness = EvaluationHarness()
    logger.info("Starting evaluation: dataset=%s, strategies=%s", args.dataset, list(systems.keys()))

    results = harness.run_evaluation(args.dataset, systems)
    harness.save_results(results, args.output)

    summaries = harness.compute_metrics(results)
    print_results_table(summaries)

    # Also save metrics summary
    metrics_path = args.output.replace(".json", "_metrics.json")
    from dataclasses import asdict  # noqa: PLC0415
    with open(metrics_path, "w") as fh:
        json.dump([asdict(s) for s in summaries], fh, indent=2)
    logger.info("Metrics saved to %s", metrics_path)


if __name__ == "__main__":
    main()
