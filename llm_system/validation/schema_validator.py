"""
Schema-based output validation for structured and free-text LLM responses.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    score: float = 1.0  # 0.0 (invalid) – 1.0 (perfect)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge two results, combining errors/warnings and averaging scores."""
        return ValidationResult(
            is_valid=self.is_valid and other.is_valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            score=(self.score + other.score) / 2,
        )


class BaseValidator(ABC):
    """Abstract base class for all validators."""

    @abstractmethod
    def validate(self, response: str, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate *response* and return a :class:`ValidationResult`."""
        ...


class ResponseSchema(BaseModel):
    """Pydantic model for a fully-structured LLM response."""

    answer: str
    citations: List[str] = []
    confidence: float = 0.0
    reasoning: Optional[str] = None

    @field_validator("confidence")
    @classmethod
    def confidence_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return v

    @field_validator("answer")
    @classmethod
    def answer_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("answer must not be empty")
        return v


class JSONSchemaValidator(BaseValidator):
    """
    Validates that a response is valid JSON conforming to a given JSON Schema.

    Parameters
    ----------
    schema:
        A JSON Schema dict.  If omitted, only JSON-parsability is checked.
    """

    def __init__(self, schema: Optional[Dict] = None) -> None:
        self._schema = schema

    def validate(self, response: str, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        try:
            data = json.loads(response)
        except json.JSONDecodeError as exc:
            return ValidationResult(is_valid=False, errors=[f"Invalid JSON: {exc}"], score=0.0)

        if self._schema:
            try:
                import jsonschema  # noqa: PLC0415

                jsonschema.validate(instance=data, schema=self._schema)
            except jsonschema.ValidationError as exc:
                return ValidationResult(
                    is_valid=False,
                    errors=[f"Schema validation error: {exc.message}"],
                    score=0.3,
                )
            except ImportError:
                logger.warning("jsonschema not installed; skipping schema validation")

        return ValidationResult(is_valid=True, score=1.0)


class StructuredResponseValidator(BaseValidator):
    """
    Validates that a free-text response contains required sections or markers.

    Parameters
    ----------
    required_sections:
        List of section headers or keywords that must appear in the response.
    """

    def __init__(self, required_sections: Optional[List[str]] = None) -> None:
        self._required = required_sections or []

    def validate(self, response: str, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []
        response_lower = response.lower()

        for section in self._required:
            if section.lower() not in response_lower:
                errors.append(f"Required section/keyword missing: '{section}'")

        if len(response.split()) < 10:
            warnings.append("Response appears very short (< 10 words).")

        score = max(0.0, 1.0 - (len(errors) * 0.2))
        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings, score=score)
