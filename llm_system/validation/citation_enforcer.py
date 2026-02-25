"""
Citation extraction and verification for grounded LLM responses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Citation:
    """A single extracted citation from an LLM response."""

    text: str        # The bracketed citation text as it appears, e.g. "[Source: doc_001]"
    source_id: str   # Parsed source identifier, e.g. "doc_001"
    chunk_id: str = ""
    verified: bool = False


@dataclass
class CitationVerificationResult:
    """Aggregate result of verifying all citations in a response."""

    total_citations: int
    verified_citations: int
    unsupported_claims: List[str] = field(default_factory=list)
    citation_rate: float = 0.0  # verified / total (0 if total == 0)
    details: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.total_citations > 0:
            self.citation_rate = self.verified_citations / self.total_citations


class CitationExtractor:
    """
    Parses citation markers from LLM-generated text.

    Supports patterns:
    - ``[Source: <id>]``
    - ``[Ref: <id>]``
    - ``(Source: <id>)``
    - ``[<id>]`` (numeric or alphanumeric short ids)
    """

    _PATTERNS = [
        re.compile(r"\[Source\s*:\s*([^\]]+)\]", re.IGNORECASE),
        re.compile(r"\[Ref\s*:\s*([^\]]+)\]", re.IGNORECASE),
        re.compile(r"\(Source\s*:\s*([^)]+)\)", re.IGNORECASE),
        re.compile(r"\[(\w[\w_\-\.]{1,30})\]"),  # short alphanumeric ids
    ]

    def extract(self, text: str) -> List[Citation]:
        """Extract all citation markers from *text* and return a deduplicated list."""
        seen: set = set()
        citations: List[Citation] = []
        for pattern in self._PATTERNS:
            for match in pattern.finditer(text):
                source_id = match.group(1).strip()
                if source_id.lower() in seen:
                    continue
                seen.add(source_id.lower())
                citations.append(Citation(text=match.group(0), source_id=source_id))
        return citations


class CitationVerifier:
    """
    Verifies that citations in an LLM response are supported by the retrieved chunks.
    """

    def __init__(self) -> None:
        self._extractor = CitationExtractor()

    def verify(
        self,
        answer: str,
        chunks: List[Any],  # List[SearchResult]
    ) -> CitationVerificationResult:
        """
        Cross-reference citations in *answer* against the content of *chunks*.

        A citation is considered verified if its source_id matches the chunk_id
        of any retrieved chunk (case-insensitive prefix match).

        Parameters
        ----------
        answer:
            The generated LLM answer text.
        chunks:
            The list of SearchResult objects retrieved for this query.
        """
        citations = self._extractor.extract(answer)
        chunk_ids = {getattr(c, "chunk_id", "").lower() for c in chunks}

        verified_count = 0
        details = []
        unsupported: List[str] = []

        for citation in citations:
            sid = citation.source_id.lower()
            # Support partial / prefix matching for flexibility
            is_verified = any(sid in cid or cid in sid for cid in chunk_ids)
            citation.verified = is_verified
            if is_verified:
                verified_count += 1
            else:
                unsupported.append(citation.source_id)
            details.append(
                {
                    "source_id": citation.source_id,
                    "verified": is_verified,
                    "citation_text": citation.text,
                }
            )

        return CitationVerificationResult(
            total_citations=len(citations),
            verified_citations=verified_count,
            unsupported_claims=unsupported,
            details=details,
        )
