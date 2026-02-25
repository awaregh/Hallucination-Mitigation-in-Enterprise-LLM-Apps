"""
Unit tests for llm_system.validation modules.
"""

import pytest

from llm_system.validation.citation_enforcer import (
    Citation,
    CitationExtractor,
    CitationVerificationResult,
    CitationVerifier,
)
from llm_system.validation.pipeline import (
    PipelineResult,
    ValidationPipeline,
    ValidationStep,
)
from llm_system.validation.schema_validator import (
    StructuredResponseValidator,
    ValidationResult,
)
from llm_system.vector_store import SearchResult


class TestCitationExtractor:
    def setup_method(self):
        self.extractor = CitationExtractor()

    def test_extracts_source_citations(self):
        text = "This is grounded [Source: doc_001] and also here [Source: doc_002]."
        citations = self.extractor.extract(text)
        source_ids = [c.source_id for c in citations]
        assert "doc_001" in source_ids
        assert "doc_002" in source_ids

    def test_extracts_ref_citations(self):
        text = "According to [Ref: paper_42], the results show improvement."
        citations = self.extractor.extract(text)
        assert any(c.source_id == "paper_42" for c in citations)

    def test_case_insensitive_source(self):
        text = "As stated in [source: abc_123], the answer is clear."
        citations = self.extractor.extract(text)
        assert len(citations) >= 1

    def test_deduplicates_citations(self):
        text = "First mention [Source: doc_001] and second mention [Source: doc_001]."
        citations = self.extractor.extract(text)
        source_ids = [c.source_id for c in citations]
        assert source_ids.count("doc_001") == 1

    def test_no_citations_returns_empty(self):
        text = "This text has no citation markers at all."
        citations = self.extractor.extract(text)
        assert citations == []

    def test_citation_text_preserved(self):
        text = "See [Source: chunk_abc]."
        citations = self.extractor.extract(text)
        assert len(citations) == 1
        assert "[Source: chunk_abc]" in citations[0].text


class TestCitationVerifier:
    def setup_method(self):
        self.verifier = CitationVerifier()

    def _make_chunk(self, chunk_id: str, content: str = "") -> SearchResult:
        return SearchResult(chunk_id=chunk_id, content=content or f"Content for {chunk_id}")

    def test_verifies_matching_citation(self):
        answer = "According to [Source: chunk_001], RAG reduces hallucinations."
        chunks = [self._make_chunk("chunk_001")]
        result = self.verifier.verify(answer, chunks)
        assert result.total_citations == 1
        assert result.verified_citations == 1
        assert result.citation_rate == 1.0

    def test_unverified_citation(self):
        answer = "According to [Source: nonexistent_chunk], something happened."
        chunks = [self._make_chunk("chunk_001")]
        result = self.verifier.verify(answer, chunks)
        assert result.total_citations == 1
        assert result.verified_citations == 0
        assert "nonexistent_chunk" in result.unsupported_claims

    def test_mixed_verified_and_unverified(self):
        answer = "From [Source: chunk_001] and [Source: fake_ref], we know this."
        chunks = [self._make_chunk("chunk_001")]
        result = self.verifier.verify(answer, chunks)
        assert result.total_citations == 2
        assert result.verified_citations == 1

    def test_no_citations(self):
        answer = "This answer has no citation markers."
        chunks = [self._make_chunk("chunk_001")]
        result = self.verifier.verify(answer, chunks)
        assert result.total_citations == 0
        assert result.citation_rate == 0.0

    def test_empty_chunks(self):
        answer = "From [Source: doc_001], we learn this."
        result = self.verifier.verify(answer, [])
        assert result.total_citations == 1
        assert result.verified_citations == 0


class TestValidationPipeline:
    def test_pipeline_runs_single_step(self):
        validator = StructuredResponseValidator(required_sections=["answer"])
        step = ValidationStep(name="structure_check", validator=validator, weight=1.0, required=False)
        pipeline = ValidationPipeline()
        pipeline.add_step(step)

        result = pipeline.run("This is an answer to the question.")
        assert isinstance(result, PipelineResult)
        assert isinstance(result.score, float)
        assert 0.0 <= result.score <= 1.0

    def test_pipeline_passes_when_all_steps_pass(self):
        validator = StructuredResponseValidator(required_sections=[])
        step = ValidationStep(name="permissive_check", validator=validator, weight=1.0)
        pipeline = ValidationPipeline()
        pipeline.add_step(step)

        result = pipeline.run("Any response text here.")
        assert result.passed is True
        assert result.score > 0.0

    def test_pipeline_required_step_short_circuits(self):
        # A validator that always fails
        class AlwaysFailValidator(StructuredResponseValidator):
            def validate(self, response, context=None):
                from llm_system.validation.schema_validator import ValidationResult  # noqa: PLC0415
                return ValidationResult(is_valid=False, errors=["Always fails"], score=0.0)

        class AlwaysPassValidator(StructuredResponseValidator):
            def validate(self, response, context=None):
                from llm_system.validation.schema_validator import ValidationResult  # noqa: PLC0415
                return ValidationResult(is_valid=True, score=1.0)

        pipeline = ValidationPipeline()
        pipeline.add_step(ValidationStep("fail_step", AlwaysFailValidator(), required=True))
        pipeline.add_step(ValidationStep("pass_step", AlwaysPassValidator(), required=False))

        result = pipeline.run("test response")
        # After required failure, second step should be skipped
        skipped = [r for r in result.step_results if "Skipped" in r.get("errors", [""])[0]]
        assert len(skipped) >= 1

    def test_pipeline_latency_recorded(self):
        validator = StructuredResponseValidator()
        pipeline = ValidationPipeline()
        pipeline.add_step(ValidationStep("latency_test", validator))
        result = pipeline.run("test")
        assert result.total_latency_ms >= 0.0

    def test_pipeline_step_results_count(self):
        validator = StructuredResponseValidator()
        pipeline = ValidationPipeline()
        for i in range(3):
            pipeline.add_step(ValidationStep(f"step_{i}", validator))
        result = pipeline.run("test response")
        assert len(result.step_results) >= 3
