"""
LLM self-critique validation for hallucination detection.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

CRITIQUE_PROMPT = """You are an expert fact-checker reviewing an AI-generated answer for hallucinations.

## Original Context (ground truth)
{context}

## AI-Generated Answer
{answer}

## Your Task
Analyze the answer and identify any statements that:
1. Are not supported by or contradict the context
2. Add information not present in the context
3. Misrepresent or distort information from the context
4. Express inappropriate confidence about uncertain facts

Respond in the following JSON format:
{{
  "has_hallucinations": true/false,
  "confidence": 0.0-1.0,
  "issues": ["issue 1", "issue 2", ...],
  "revised_answer": "corrected answer here, or null if no revision needed"
}}"""


@dataclass
class CritiqueResult:
    """Result of an LLM self-critique pass."""

    has_hallucinations: bool
    confidence: float
    issues: List[str] = field(default_factory=list)
    revised_answer: Optional[str] = None
    raw_critique: str = ""


class SelfCritiqueValidator:
    """
    Uses an LLM to critique its own or another model's output against the
    retrieved context, identifying potential hallucinations.
    """

    def critique(
        self,
        answer: str,
        context: str,
        llm_client: Any,  # LLMClient
    ) -> CritiqueResult:
        """
        Ask *llm_client* to critique *answer* against *context*.

        Parameters
        ----------
        answer:
            The LLM-generated answer to critique.
        context:
            The source context the answer should be grounded in.
        llm_client:
            Any object implementing ``.complete(prompt, system)`` -> str.

        Returns
        -------
        CritiqueResult
        """
        prompt = CRITIQUE_PROMPT.format(
            context=context[:4000],  # truncate very long contexts
            answer=answer,
        )
        system = (
            "You are a meticulous AI quality reviewer. "
            "Your job is to detect hallucinations and factual errors in AI responses. "
            "Always respond with valid JSON matching the specified format."
        )

        try:
            raw = llm_client.complete(prompt, system=system)
        except Exception as exc:  # noqa: BLE001
            logger.error("LLM critique call failed: %s", exc)
            return CritiqueResult(
                has_hallucinations=False,
                confidence=0.0,
                issues=["Critique failed due to LLM error"],
                raw_critique="",
            )

        return self._parse_critique(raw)

    def _parse_critique(self, raw: str) -> CritiqueResult:
        """Parse the JSON critique response, with fallback for malformed output."""
        # Attempt to extract JSON block if wrapped in markdown code fences
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        json_str = json_match.group(1) if json_match else raw.strip()

        try:
            data = json.loads(json_str)
            return CritiqueResult(
                has_hallucinations=bool(data.get("has_hallucinations", False)),
                confidence=float(data.get("confidence", 0.5)),
                issues=data.get("issues", []),
                revised_answer=data.get("revised_answer"),
                raw_critique=raw,
            )
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("Failed to parse critique JSON (%s). Performing heuristic analysis.", exc)
            return self._heuristic_critique(raw)

    @staticmethod
    def _heuristic_critique(raw: str) -> CritiqueResult:
        """Fallback heuristic when JSON parsing fails."""
        raw_lower = raw.lower()
        has_hallucinations = any(
            kw in raw_lower
            for kw in ["hallucination", "incorrect", "not in context", "fabricated", "unsupported"]
        )
        return CritiqueResult(
            has_hallucinations=has_hallucinations,
            confidence=0.4,
            issues=["Unable to parse structured critique — heuristic analysis applied"],
            revised_answer=None,
            raw_critique=raw,
        )
