"""
Unit tests for llm_system.guardrails modules.
"""

import pytest

from llm_system.guardrails.instruction_constraints import (
    AllowListConstraint,
    CitationRequiredConstraint,
    ConstraintChecker,
    DenyListConstraint,
    LengthConstraint,
)
from llm_system.guardrails.prompt_templates import (
    BASELINE_TEMPLATE,
    GROUNDED_TEMPLATE,
    get_template,
)
from llm_system.guardrails.system_roles import ROLES, RoleEnforcer


class TestPromptTemplates:
    def test_get_baseline_template(self):
        template = get_template("baseline")
        assert template.name == "baseline"
        assert len(template.system_prompt) > 0

    def test_get_grounded_template(self):
        template = get_template("grounded")
        assert template.name == "grounded"
        assert "context" in template.user_template.lower()

    def test_get_strict_template(self):
        template = get_template("strict")
        assert template.name == "strict"
        assert len(template.constraints) > 0

    def test_get_conservative_template(self):
        template = get_template("conservative")
        assert template.name == "conservative"

    def test_unknown_template_raises(self):
        with pytest.raises(ValueError, match="Unknown template"):
            get_template("nonexistent_template")

    def test_case_insensitive(self):
        template = get_template("BASELINE")
        assert template.name == "baseline"

    def test_format_user_fills_query(self):
        template = get_template("grounded")
        formatted = template.format_user(query="What is RAG?", context="RAG stands for retrieval-augmented generation.")
        assert "What is RAG?" in formatted
        assert "RAG stands for retrieval-augmented generation." in formatted

    def test_baseline_format_user(self):
        template = get_template("baseline")
        formatted = template.format_user(query="Hello?")
        assert "Hello?" in formatted


class TestDenyListConstraint:
    def test_no_violation_on_clean_response(self):
        constraint = DenyListConstraint(deny_list=["forbidden", "banned_word"])
        result = constraint.check("query", "This is a perfectly acceptable response.")
        assert result is None

    def test_violation_on_denied_term(self):
        constraint = DenyListConstraint(deny_list=["forbidden"])
        result = constraint.check("query", "This response contains a forbidden phrase.")
        assert result is not None
        assert result.severity == "critical"
        assert "forbidden" in result.description

    def test_case_insensitive_denial(self):
        constraint = DenyListConstraint(deny_list=["forbidden"])
        result = constraint.check("query", "This is FORBIDDEN.")
        assert result is not None

    def test_empty_deny_list_never_violates(self):
        constraint = DenyListConstraint(deny_list=[])
        result = constraint.check("query", "Any response at all.")
        assert result is None

    def test_multiple_denied_terms_first_match_returned(self):
        constraint = DenyListConstraint(deny_list=["bad", "worse"])
        result = constraint.check("query", "This is bad and worse.")
        assert result is not None


class TestAllowListConstraint:
    def test_no_violation_when_approved_topic_present(self):
        constraint = AllowListConstraint(allow_list=["rag", "retrieval"])
        result = constraint.check("query", "This discusses RAG and retrieval techniques.")
        assert result is None

    def test_violation_when_no_approved_topic(self):
        constraint = AllowListConstraint(allow_list=["rag", "embeddings"])
        result = constraint.check("query", "This discusses completely unrelated topics.")
        assert result is not None
        assert result.severity == "warning"

    def test_empty_allow_list_never_violates(self):
        constraint = AllowListConstraint(allow_list=[])
        result = constraint.check("query", "Any response about anything.")
        assert result is None

    def test_case_insensitive_allow(self):
        constraint = AllowListConstraint(allow_list=["vector"])
        result = constraint.check("query", "We use VECTOR databases here.")
        assert result is None  # Should not violate — "vector" found case-insensitively


class TestConstraintChecker:
    def test_no_violations_on_good_response(self):
        checker = ConstraintChecker([
            LengthConstraint(min_chars=10, max_chars=1000),
            DenyListConstraint(deny_list=["forbidden"]),
        ])
        violations = checker.check_all("query", "This is a good response without any issues.")
        assert len(violations) == 0

    def test_collects_all_violations(self):
        checker = ConstraintChecker([
            LengthConstraint(min_chars=1000),  # Will fail — response too short
            DenyListConstraint(deny_list=["forbidden"]),
        ])
        violations = checker.check_all("query", "Short forbidden response.")
        assert len(violations) == 2

    def test_has_critical_violations_true(self):
        checker = ConstraintChecker([
            DenyListConstraint(deny_list=["banned"]),
        ])
        assert checker.has_critical_violations("query", "This is banned content.") is True

    def test_has_critical_violations_false(self):
        checker = ConstraintChecker([
            LengthConstraint(min_chars=5),
        ])
        assert checker.has_critical_violations("query", "This is a perfectly fine response.") is False

    def test_add_constraint(self):
        checker = ConstraintChecker()
        assert len(checker._constraints) == 0
        checker.add_constraint(DenyListConstraint(deny_list=["bad"]))
        assert len(checker._constraints) == 1


class TestSystemRoles:
    def test_roles_dict_populated(self):
        assert "analyst" in ROLES
        assert "support_agent" in ROLES
        assert "policy_advisor" in ROLES
        assert "knowledge_assistant" in ROLES

    def test_role_has_system_prompt(self):
        for name, role in ROLES.items():
            assert len(role.system_prompt) > 0, f"Role '{name}' has empty system prompt"

    def test_role_enforcer_apply_role(self):
        enforcer = RoleEnforcer()
        prompt = enforcer.apply_role("analyst")
        assert len(prompt) > 0
        assert "analyst" in prompt.lower() or "data" in prompt.lower()

    def test_role_enforcer_apply_role_with_base(self):
        enforcer = RoleEnforcer()
        prompt = enforcer.apply_role("support_agent", base_prompt="Additional instruction here.")
        assert "Additional instruction here." in prompt

    def test_unknown_role_raises(self):
        enforcer = RoleEnforcer()
        with pytest.raises(ValueError, match="Unknown role"):
            enforcer.apply_role("nonexistent_role")

    def test_validate_role_compliance(self):
        enforcer = RoleEnforcer()
        role = ROLES["analyst"]
        # A clean response should pass compliance
        assert enforcer.validate_role_compliance("RAG reduces hallucinations by grounding responses.", role) is True

    def test_list_roles(self):
        enforcer = RoleEnforcer()
        roles = enforcer.list_roles()
        assert isinstance(roles, list)
        assert len(roles) >= 4
