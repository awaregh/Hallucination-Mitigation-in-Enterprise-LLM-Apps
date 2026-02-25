"""
Evaluation harness for benchmarking hallucination mitigation strategies.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result for a single query evaluated under a single strategy."""

    query_id: str
    strategy: str
    answer: str
    latency_ms: float
    tokens_used: int
    passed_validation: bool
    citation_rate: float
    hallucination_detected: bool
    query: str = ""
    expected_answer: str = ""
    category: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricsSummary:
    """Aggregate metrics for a strategy over a dataset."""

    strategy: str
    n_queries: int
    hallucination_rate: float      # fraction of queries where hallucination_detected == True
    citation_accuracy: float       # mean citation_rate across queries
    avg_latency_ms: float
    avg_tokens: float
    refusal_rate: float            # fraction of answers containing refusal language
    validation_pass_rate: float    # fraction of queries that passed validation


class EvaluationHarness:
    """
    Orchestrates loading datasets, running queries against one or more strategies,
    computing aggregate metrics, and saving results.
    """

    _REFUSAL_MARKERS = [
        "i don't know",
        "i cannot",
        "insufficient context",
        "not in the context",
        "i could not find",
        "no information",
        "unable to answer",
    ]

    def load_dataset(self, path: str) -> List[Dict[str, Any]]:
        """Load a JSONL dataset from *path* and return a list of query dicts."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        records = []
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        logger.info("Loaded %d queries from %s", len(records), path)
        return records

    def run_query(
        self,
        query_dict: Dict[str, Any],
        strategy_name: str,
        system: Any,  # Any RAG system with a .query(str) -> GenerationResponse method
    ) -> EvaluationResult:
        """
        Evaluate a single *query_dict* against *system*.

        Parameters
        ----------
        query_dict:
            A dict from the JSONL dataset (must contain ``"id"`` and ``"query"``).
        strategy_name:
            Human-readable strategy identifier.
        system:
            A RAG system exposing ``.query(query: str) -> GenerationResponse``.
        """
        query = query_dict.get("query", "")
        query_id = query_dict.get("id", "unknown")

        t_start = time.perf_counter()
        try:
            response = system.query(query)
            latency_ms = (time.perf_counter() - t_start) * 1000
        except Exception as exc:  # noqa: BLE001
            logger.error("Query %s failed: %s", query_id, exc)
            latency_ms = (time.perf_counter() - t_start) * 1000
            return EvaluationResult(
                query_id=query_id,
                strategy=strategy_name,
                answer="",
                latency_ms=round(latency_ms, 2),
                tokens_used=0,
                passed_validation=False,
                citation_rate=0.0,
                hallucination_detected=True,
                query=query,
                expected_answer=query_dict.get("expected_answer", ""),
                category=query_dict.get("category", ""),
                metadata={"error": str(exc)},
            )

        answer = response.answer
        citation_rate = response.confidence  # proxy; replaced by verifier in full pipeline
        hallucination_detected = self._detect_hallucination(answer, query_dict)
        passed_validation = not hallucination_detected and len(answer) > 20

        return EvaluationResult(
            query_id=query_id,
            strategy=strategy_name,
            answer=answer,
            latency_ms=round(response.latency_ms, 2),
            tokens_used=response.tokens_used,
            passed_validation=passed_validation,
            citation_rate=citation_rate,
            hallucination_detected=hallucination_detected,
            query=query,
            expected_answer=query_dict.get("expected_answer", ""),
            category=query_dict.get("category", ""),
        )

    def run_evaluation(
        self,
        dataset_path: str,
        strategies: Dict[str, Any],
    ) -> List[EvaluationResult]:
        """
        Run all queries from *dataset_path* against each strategy in *strategies*.

        Parameters
        ----------
        dataset_path:
            Path to a JSONL dataset file.
        strategies:
            Mapping of strategy name -> RAG system instance.
        """
        dataset = self.load_dataset(dataset_path)
        results: List[EvaluationResult] = []

        for strategy_name, system in strategies.items():
            logger.info("Evaluating strategy: %s (%d queries)", strategy_name, len(dataset))
            for query_dict in dataset:
                result = self.run_query(query_dict, strategy_name, system)
                results.append(result)

        logger.info("Evaluation complete: %d total results", len(results))
        return results

    def compute_metrics(self, results: List[EvaluationResult]) -> List[MetricsSummary]:
        """Compute per-strategy aggregate metrics from a list of results."""
        from collections import defaultdict  # noqa: PLC0415

        by_strategy: Dict[str, List[EvaluationResult]] = defaultdict(list)
        for r in results:
            by_strategy[r.strategy].append(r)

        summaries: List[MetricsSummary] = []
        for strategy, strategy_results in by_strategy.items():
            n = len(strategy_results)
            if n == 0:
                continue

            hallucination_rate = sum(1 for r in strategy_results if r.hallucination_detected) / n
            citation_accuracy = sum(r.citation_rate for r in strategy_results) / n
            avg_latency = sum(r.latency_ms for r in strategy_results) / n
            avg_tokens = sum(r.tokens_used for r in strategy_results) / n
            refusal_rate = sum(1 for r in strategy_results if self._is_refusal(r.answer)) / n
            validation_pass_rate = sum(1 for r in strategy_results if r.passed_validation) / n

            summaries.append(
                MetricsSummary(
                    strategy=strategy,
                    n_queries=n,
                    hallucination_rate=round(hallucination_rate, 4),
                    citation_accuracy=round(citation_accuracy, 4),
                    avg_latency_ms=round(avg_latency, 2),
                    avg_tokens=round(avg_tokens, 2),
                    refusal_rate=round(refusal_rate, 4),
                    validation_pass_rate=round(validation_pass_rate, 4),
                )
            )

        return sorted(summaries, key=lambda s: s.hallucination_rate)

    def save_results(self, results: List[EvaluationResult], output_path: str) -> None:
        """Save results to a JSON file at *output_path*."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump([asdict(r) for r in results], fh, indent=2)
        logger.info("Saved %d results to %s", len(results), output_path)

    def _detect_hallucination(self, answer: str, query_dict: Dict[str, Any]) -> bool:
        """
        Heuristic hallucination detection.

        In a production system this would call the self-critique validator or
        an NLI model.  Here we use simple heuristics.
        """
        if not answer or len(answer) < 10:
            return True

        # If query is marked high hallucination risk and answer is confident
        risk = query_dict.get("should_hallucinate_risk", "")
        if risk == "high" and not self._is_refusal(answer):
            import re  # noqa: PLC0415
            overconfident = re.search(
                r"\b(definitely|certainly|the answer is|always|never)\b", answer, re.IGNORECASE
            )
            if overconfident:
                return True

        return False

    def _is_refusal(self, answer: str) -> bool:
        answer_lower = answer.lower()
        return any(marker in answer_lower for marker in self._REFUSAL_MARKERS)
