"""
Guardrails package — prompt templates, instruction constraints, and system roles.
"""

from .instruction_constraints import (
    AllowListConstraint,
    BaseConstraint,
    CitationRequiredConstraint,
    ConstraintChecker,
    ConstraintViolation,
    DenyListConstraint,
    LengthConstraint,
    NoHallucinationMarkerConstraint,
)
from .prompt_templates import (
    BASELINE_TEMPLATE,
    CONSERVATIVE_TEMPLATE,
    GROUNDED_TEMPLATE,
    STRICT_TEMPLATE,
    PromptTemplate,
    get_template,
)
from .system_roles import ROLES, RoleEnforcer, SystemRole

__all__ = [
    "PromptTemplate",
    "BASELINE_TEMPLATE",
    "GROUNDED_TEMPLATE",
    "STRICT_TEMPLATE",
    "CONSERVATIVE_TEMPLATE",
    "get_template",
    "ConstraintViolation",
    "BaseConstraint",
    "LengthConstraint",
    "CitationRequiredConstraint",
    "NoHallucinationMarkerConstraint",
    "DenyListConstraint",
    "AllowListConstraint",
    "ConstraintChecker",
    "SystemRole",
    "ROLES",
    "RoleEnforcer",
]
