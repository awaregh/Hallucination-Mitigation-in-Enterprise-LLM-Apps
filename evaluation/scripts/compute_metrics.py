#!/usr/bin/env python3
"""
Compute and compare metrics across strategies from evaluation results JSON.

Usage:
    python evaluation/scripts/compute_metrics.py \
        --results results/evaluation_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def load_results(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def compute_metrics(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_strategy: Dict[str, List[Dict]] = defaultdict(list)
    for r in results:
        by_strategy[r["strategy"]].append(r)

    summaries = []
    for strategy, items in by_strategy.items():
        n = len(items)
        if n == 0:
            continue

        hallucination_rate = sum(1 for r in items if r.get("hallucination_detected", False)) / n
        citation_accuracy = sum(r.get("citation_rate", 0.0) for r in items) / n
        avg_latency = sum(r.get("latency_ms", 0.0) for r in items) / n
        avg_tokens = sum(r.get("tokens_used", 0) for r in items) / n
        validation_pass_rate = sum(1 for r in items if r.get("passed_validation", False)) / n

        refusal_markers = [
            "i don't know", "i cannot", "insufficient context",
            "not in the context", "i could not find",
        ]
        refusal_rate = sum(
            1 for r in items
            if any(m in r.get("answer", "").lower() for m in refusal_markers)
        ) / n

        summaries.append(
            {
                "strategy": strategy,
                "n_queries": n,
                "hallucination_rate": round(hallucination_rate, 4),
                "citation_accuracy": round(citation_accuracy, 4),
                "avg_latency_ms": round(avg_latency, 2),
                "avg_tokens": round(avg_tokens, 2),
                "refusal_rate": round(refusal_rate, 4),
                "validation_pass_rate": round(validation_pass_rate, 4),
            }
        )

    return sorted(summaries, key=lambda s: s["hallucination_rate"])


def print_markdown_table(summaries: List[Dict[str, Any]]) -> None:
    headers = [
        "Strategy", "Queries", "Hallucination Rate", "Citation Accuracy",
        "Avg Latency (ms)", "Avg Tokens", "Refusal Rate", "Validation Pass Rate",
    ]
    print("## Evaluation Metrics Summary\n")
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")

    for s in summaries:
        row = [
            s["strategy"],
            str(s["n_queries"]),
            f"{s['hallucination_rate']:.1%}",
            f"{s['citation_accuracy']:.1%}",
            f"{s['avg_latency_ms']:.0f}",
            f"{s['avg_tokens']:.0f}",
            f"{s['refusal_rate']:.1%}",
            f"{s['validation_pass_rate']:.1%}",
        ]
        print("| " + " | ".join(row) + " |")

    print()
    best = summaries[0] if summaries else None
    if best:
        print(f"\n**Best strategy by hallucination rate:** `{best['strategy']}` "
              f"({best['hallucination_rate']:.1%} hallucination rate)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute metrics from evaluation results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to results JSON from run_evaluation.py",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path to save metrics markdown",
    )
    args = parser.parse_args()

    results = load_results(args.results)
    summaries = compute_metrics(results)
    print_markdown_table(summaries)

    if args.output:
        import io  # noqa: PLC0415
        from contextlib import redirect_stdout  # noqa: PLC0415

        buf = io.StringIO()
        with redirect_stdout(buf):
            print_markdown_table(summaries)
        Path(args.output).write_text(buf.getvalue(), encoding="utf-8")
        print(f"\nMetrics written to {args.output}")


if __name__ == "__main__":
    main()
