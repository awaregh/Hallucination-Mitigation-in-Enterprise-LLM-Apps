"""
Multi-step validation pipeline for composing multiple validators.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .schema_validator import BaseValidator, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationStep:
    """
    A single step in the validation pipeline.

    Attributes
    ----------
    name:
        Human-readable step name for logging and reporting.
    validator:
        A :class:`BaseValidator` instance to run.
    weight:
        Contribution weight to the aggregate score (default 1.0).
    required:
        If ``True``, a failure in this step short-circuits the pipeline.
    """

    name: str
    validator: BaseValidator
    weight: float = 1.0
    required: bool = False


@dataclass
class PipelineResult:
    """Aggregate result of running all validation steps."""

    passed: bool
    score: float  # Weighted average score 0.0–1.0
    step_results: List[Dict[str, Any]] = field(default_factory=list)
    total_latency_ms: float = 0.0

    @property
    def failed_steps(self) -> List[str]:
        """Names of steps that did not pass."""
        return [r["step"] for r in self.step_results if not r["passed"]]


class ValidationPipeline:
    """
    Runs a sequence of :class:`ValidationStep` objects, aggregating results.

    Steps run in insertion order.  If a step is marked ``required=True`` and
    fails, the pipeline short-circuits and marks remaining steps as skipped.
    """

    def __init__(self) -> None:
        self._steps: List[ValidationStep] = []

    def add_step(self, step: ValidationStep) -> None:
        """Append *step* to the pipeline."""
        self._steps.append(step)

    def run(
        self,
        response: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> PipelineResult:
        """
        Execute all pipeline steps against *response*.

        Parameters
        ----------
        response:
            The LLM-generated response to validate.
        context:
            Optional context dict passed to each validator (e.g. retrieved chunks).
        """
        t_start = time.perf_counter()
        context = context or {}
        step_results: List[Dict[str, Any]] = []
        total_weight = sum(s.weight for s in self._steps)
        weighted_score = 0.0
        overall_passed = True

        for step in self._steps:
            step_start = time.perf_counter()
            try:
                result: ValidationResult = step.validator.validate(response, context)
            except Exception as exc:  # noqa: BLE001
                logger.error("Validation step '%s' raised an exception: %s", step.name, exc)
                result = ValidationResult(is_valid=False, errors=[str(exc)], score=0.0)

            step_latency = (time.perf_counter() - step_start) * 1000
            weighted_score += result.score * step.weight

            step_results.append(
                {
                    "step": step.name,
                    "passed": result.is_valid,
                    "score": result.score,
                    "errors": result.errors,
                    "warnings": result.warnings,
                    "latency_ms": round(step_latency, 2),
                }
            )

            if step.required and not result.is_valid:
                logger.warning("Required validation step '%s' failed — short-circuiting pipeline.", step.name)
                overall_passed = False
                # Mark remaining steps as skipped
                for remaining in self._steps[self._steps.index(step) + 1 :]:
                    step_results.append(
                        {
                            "step": remaining.name,
                            "passed": False,
                            "score": 0.0,
                            "errors": ["Skipped due to earlier required step failure"],
                            "warnings": [],
                            "latency_ms": 0.0,
                        }
                    )
                break

        # If any required step failed we already set overall_passed = False
        if overall_passed:
            overall_passed = all(r["passed"] for r in step_results if not r["errors"] == ["Skipped due to earlier required step failure"])

        aggregate_score = weighted_score / total_weight if total_weight > 0 else 0.0
        total_latency = (time.perf_counter() - t_start) * 1000

        return PipelineResult(
            passed=overall_passed,
            score=round(aggregate_score, 4),
            step_results=step_results,
            total_latency_ms=round(total_latency, 2),
        )
