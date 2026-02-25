#!/usr/bin/env python3
"""
Failure mode analysis script — categorizes failures and identifies worst-performing strategies.

Usage:
    python evaluation/scripts/analyze_failure_modes.py \
        --results results/evaluation_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


FAILURE_CATEGORIES = {
    "hallucination": "Answer contains hallucinated content (hallucination_detected=True)",
    "citation_missing": "Answer lacks any citation markers",
    "refusal_error": "System refused when it should have answered (or vice versa)",
    "validation_failure": "Answer failed validation pipeline checks",
    "empty_answer": "Answer is empty or extremely short",
}

_REFUSAL_MARKERS = [
    "i don't know", "i cannot", "insufficient context",
    "not in the context", "i could not find", "unable to answer",
]

_CITATION_PATTERN = r"\[Source\s*:\s*[^\]]+\]"


def load_results(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def categorize_failure(result: Dict[str, Any]) -> List[str]:
    """Return a list of failure categories for a single result."""
    import re  # noqa: PLC0415

    failures = []
    answer = result.get("answer", "")
    answer_lower = answer.lower()

    if result.get("hallucination_detected", False):
        failures.append("hallucination")

    if not re.search(_CITATION_PATTERN, answer, re.IGNORECASE):
        failures.append("citation_missing")

    if not result.get("passed_validation", True):
        failures.append("validation_failure")

    if not answer or len(answer.strip()) < 20:
        failures.append("empty_answer")

    # Refusal error: refused a factual question (non-adversarial, non-missing-knowledge)
    category = result.get("category", "")
    is_refusal = any(m in answer_lower for m in _REFUSAL_MARKERS)
    if category == "factual" and is_refusal:
        failures.append("refusal_error")

    return failures


def analyze(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute failure mode analysis across all results."""
    by_strategy: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        by_strategy[r["strategy"]].append(r)

    # Per-failure-category counts per strategy
    strategy_failure_counts: Dict[str, Dict[str, int]] = {}
    for strategy, items in by_strategy.items():
        counts: Dict[str, int] = {k: 0 for k in FAILURE_CATEGORIES}
        for item in items:
            for failure in categorize_failure(item):
                if failure in counts:
                    counts[failure] += 1
        strategy_failure_counts[strategy] = counts

    # Worst strategy per failure type
    worst_by_failure: Dict[str, Tuple[str, int]] = {}
    for failure_type in FAILURE_CATEGORIES:
        worst_strategy = max(
            strategy_failure_counts.items(),
            key=lambda x: x[1].get(failure_type, 0),
            default=("none", {failure_type: 0}),
        )
        worst_by_failure[failure_type] = (worst_strategy[0], worst_strategy[1].get(failure_type, 0))

    return {
        "strategy_failure_counts": strategy_failure_counts,
        "worst_by_failure": worst_by_failure,
        "total_results": len(results),
    }


def print_analysis(analysis: Dict[str, Any]) -> None:
    strategy_counts = analysis["strategy_failure_counts"]
    worst = analysis["worst_by_failure"]

    print("## Failure Mode Analysis\n")
    print(f"Total results analyzed: {analysis['total_results']}\n")

    # Failure mode taxonomy table
    print("### Failure Mode Taxonomy\n")
    for ftype, description in FAILURE_CATEGORIES.items():
        print(f"- **{ftype}**: {description}")
    print()

    # Per-strategy failure table
    print("### Failure Counts by Strategy\n")
    failure_types = list(FAILURE_CATEGORIES.keys())
    header = "| Strategy | " + " | ".join(failure_types) + " | Total Failures |"
    sep = "| --- | " + " | ".join(["---"] * len(failure_types)) + " | --- |"
    print(header)
    print(sep)

    for strategy, counts in sorted(strategy_counts.items()):
        total_failures = sum(counts.values())
        row_vals = [str(counts.get(ft, 0)) for ft in failure_types]
        print(f"| {strategy} | " + " | ".join(row_vals) + f" | {total_failures} |")

    print()

    # Worst performers
    print("### Worst-Performing Strategy by Failure Type\n")
    print("| Failure Type | Worst Strategy | Count |")
    print("| --- | --- | --- |")
    for ftype, (strategy, count) in worst.items():
        print(f"| {ftype} | {strategy} | {count} |")

    print()

    # Recommendations
    print("### Recommendations Based on Failure Analysis\n")
    for ftype, (strategy, count) in worst.items():
        if count > 0:
            recs = {
                "hallucination": "Implement self-critique validation or stricter grounding prompts.",
                "citation_missing": "Enforce citation requirement via CitationRequiredConstraint.",
                "refusal_error": "Review grounding prompt — system may be over-refusing on factual queries.",
                "validation_failure": "Investigate validation pipeline configuration for this strategy.",
                "empty_answer": "Check LLM client connectivity and context building logic.",
            }
            print(f"- **{ftype}** (worst: {strategy}): {recs.get(ftype, 'Review strategy configuration.')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze failure modes from evaluation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results", required=True, help="Path to results JSON")
    parser.add_argument("--output", default=None, help="Optional output path for analysis markdown")
    args = parser.parse_args()

    results = load_results(args.results)
    analysis = analyze(results)
    print_analysis(analysis)

    if args.output:
        import io  # noqa: PLC0415
        from contextlib import redirect_stdout  # noqa: PLC0415

        buf = io.StringIO()
        with redirect_stdout(buf):
            print_analysis(analysis)
        Path(args.output).write_text(buf.getvalue(), encoding="utf-8")
        print(f"\nAnalysis written to {args.output}")


if __name__ == "__main__":
    main()
