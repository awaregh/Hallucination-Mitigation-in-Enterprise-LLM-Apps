"""
Instruction constraint enforcement for LLM prompt and response validation.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ConstraintViolation:
    """Describes a single constraint violation."""

    constraint_name: str
    severity: str  # "critical" | "warning" | "info"
    description: str


class BaseConstraint(ABC):
    """Abstract base class for all constraints."""

    @abstractmethod
    def check(self, prompt: str, response: str) -> Optional[ConstraintViolation]:
        """
        Inspect *prompt* and *response*.

        Returns a :class:`ConstraintViolation` if the constraint is violated,
        or ``None`` if it passes.
        """
        ...


class LengthConstraint(BaseConstraint):
    """
    Ensures the response is within an acceptable length range.

    Parameters
    ----------
    min_chars:
        Minimum required characters in the response.
    max_chars:
        Maximum allowed characters in the response.
    """

    def __init__(self, min_chars: int = 10, max_chars: int = 8000) -> None:
        self.min_chars = min_chars
        self.max_chars = max_chars

    def check(self, prompt: str, response: str) -> Optional[ConstraintViolation]:
        length = len(response)
        if length < self.min_chars:
            return ConstraintViolation(
                constraint_name="LengthConstraint",
                severity="warning",
                description=f"Response too short ({length} chars < minimum {self.min_chars}).",
            )
        if length > self.max_chars:
            return ConstraintViolation(
                constraint_name="LengthConstraint",
                severity="warning",
                description=f"Response too long ({length} chars > maximum {self.max_chars}).",
            )
        return None


class CitationRequiredConstraint(BaseConstraint):
    """
    Checks that the response contains at least one citation marker of the form
    ``[Source: ...]``.
    """

    _CITATION_PATTERN = re.compile(r"\[Source\s*:\s*[^\]]+\]", re.IGNORECASE)

    def check(self, prompt: str, response: str) -> Optional[ConstraintViolation]:
        if not self._CITATION_PATTERN.search(response):
            return ConstraintViolation(
                constraint_name="CitationRequiredConstraint",
                severity="critical",
                description="Response contains no [Source: ...] citations.",
            )
        return None


class NoHallucinationMarkerConstraint(BaseConstraint):
    """
    Checks that the response contains appropriate uncertainty markers when it
    cannot be grounded.  Specifically, warns if the response makes strong
    assertions without any hedging language when the prompt asks for grounding.
    """

    _REFUSAL_PATTERNS = [
        re.compile(r"i don'?t know", re.IGNORECASE),
        re.compile(r"i cannot (find|determine|answer)", re.IGNORECASE),
        re.compile(r"insufficient.{0,20}context", re.IGNORECASE),
        re.compile(r"not (in|found in).{0,20}(context|document|source)", re.IGNORECASE),
    ]
    _CONFIDENT_CLAIM = re.compile(r"\b(definitely|certainly|always|never|the fact is|it is known)\b", re.IGNORECASE)

    def check(self, prompt: str, response: str) -> Optional[ConstraintViolation]:
        has_refusal = any(p.search(response) for p in self._REFUSAL_PATTERNS)
        has_overconfidence = bool(self._CONFIDENT_CLAIM.search(response))
        if has_overconfidence and not has_refusal:
            return ConstraintViolation(
                constraint_name="NoHallucinationMarkerConstraint",
                severity="warning",
                description="Response contains overconfident language without uncertainty markers.",
            )
        return None


class DenyListConstraint(BaseConstraint):
    """
    Blocks responses containing terms from a deny list.

    Parameters
    ----------
    deny_list:
        List of banned terms or phrases (case-insensitive).
    """

    def __init__(self, deny_list: Optional[List[str]] = None) -> None:
        self.deny_list = [term.lower() for term in (deny_list or [])]

    def check(self, prompt: str, response: str) -> Optional[ConstraintViolation]:
        response_lower = response.lower()
        for term in self.deny_list:
            if term in response_lower:
                return ConstraintViolation(
                    constraint_name="DenyListConstraint",
                    severity="critical",
                    description=f"Response contains denied term: '{term}'.",
                )
        return None


class AllowListConstraint(BaseConstraint):
    """
    Warns if the response discusses topics not on the allow list.

    This is a heuristic check: if the allow list is non-empty and none of the
    approved topic keywords appear in the response, a violation is raised.

    Parameters
    ----------
    allow_list:
        List of approved topic keywords. If empty, this constraint is a no-op.
    """

    def __init__(self, allow_list: Optional[List[str]] = None) -> None:
        self.allow_list = [term.lower() for term in (allow_list or [])]

    def check(self, prompt: str, response: str) -> Optional[ConstraintViolation]:
        if not self.allow_list:
            return None
        response_lower = response.lower()
        if not any(term in response_lower for term in self.allow_list):
            return ConstraintViolation(
                constraint_name="AllowListConstraint",
                severity="warning",
                description="Response does not appear to discuss any approved topics.",
            )
        return None


class ConstraintChecker:
    """
    Runs a collection of constraints against a prompt/response pair and
    aggregates violations.
    """

    def __init__(self, constraints: Optional[List[BaseConstraint]] = None) -> None:
        self._constraints: List[BaseConstraint] = constraints or []

    def add_constraint(self, constraint: BaseConstraint) -> None:
        """Append *constraint* to the checker's list."""
        self._constraints.append(constraint)

    def check_all(self, prompt: str, response: str) -> List[ConstraintViolation]:
        """Run all constraints and return the list of violations (may be empty)."""
        violations: List[ConstraintViolation] = []
        for constraint in self._constraints:
            result = constraint.check(prompt, response)
            if result is not None:
                violations.append(result)
        return violations

    def has_critical_violations(self, prompt: str, response: str) -> bool:
        """Return ``True`` if any critical violations are found."""
        for v in self.check_all(prompt, response):
            if v.severity == "critical":
                return True
        return False
