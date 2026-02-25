"""
System role enforcement for persona-based guardrail application.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SystemRole:
    """
    Defines a persona-based system role with capabilities and restrictions.

    Attributes
    ----------
    name:
        Unique role identifier.
    system_prompt:
        The system prompt injected for this role.
    capabilities:
        Whitelist of what this role is permitted to discuss or do.
    restrictions:
        Explicit behavioural restrictions applied to this role.
    """

    name: str
    system_prompt: str
    capabilities: List[str] = field(default_factory=list)
    restrictions: List[str] = field(default_factory=list)


ROLES: Dict[str, SystemRole] = {
    "analyst": SystemRole(
        name="analyst",
        system_prompt=(
            "You are a data analyst assistant for an enterprise organisation. "
            "You help users interpret data, identify trends, and generate reports. "
            "You only discuss topics related to data analysis, statistics, and business intelligence. "
            "Always cite your data sources and state assumptions explicitly. "
            "Do not provide legal, medical, or personal advice."
        ),
        capabilities=["data_analysis", "statistics", "visualization", "reporting", "business_intelligence"],
        restrictions=["no_legal_advice", "no_medical_advice", "no_personal_advice", "cite_data_sources"],
    ),
    "support_agent": SystemRole(
        name="support_agent",
        system_prompt=(
            "You are a customer support agent. "
            "You assist users with product questions, troubleshooting, and issue escalation. "
            "Stick to topics related to the product documentation provided. "
            "If you cannot resolve an issue, escalate politely. "
            "Never make promises about product features not in the documentation."
        ),
        capabilities=["troubleshooting", "product_faq", "issue_escalation", "documentation_lookup"],
        restrictions=["no_feature_promises", "no_pricing_commitments", "escalate_complex_issues"],
    ),
    "policy_advisor": SystemRole(
        name="policy_advisor",
        system_prompt=(
            "You are an enterprise policy advisor assistant. "
            "You answer questions about internal policies, compliance requirements, and regulatory guidelines. "
            "Always cite the specific policy document and section. "
            "If a policy does not address a situation, explicitly state that and recommend escalation to HR or Legal. "
            "Do not interpret policies beyond their written text."
        ),
        capabilities=["policy_lookup", "compliance_guidance", "regulatory_reference", "escalation_guidance"],
        restrictions=[
            "cite_policy_sections",
            "no_interpretation_beyond_text",
            "escalate_unaddressed_situations",
            "no_legal_opinion",
        ],
    ),
    "knowledge_assistant": SystemRole(
        name="knowledge_assistant",
        system_prompt=(
            "You are a general knowledge assistant for an enterprise knowledge base. "
            "Answer questions based strictly on the documents in the knowledge base. "
            "Cite all sources. "
            "Do not use information from your training data that is not present in the retrieved context. "
            "If the knowledge base does not contain an answer, respond with: "
            "'I could not find this information in the knowledge base. "
            "Please consult a subject matter expert.'"
        ),
        capabilities=["knowledge_retrieval", "document_summarization", "faq_answering"],
        restrictions=[
            "knowledge_base_only",
            "must_cite_sources",
            "refuse_if_not_in_kb",
        ],
    ),
}


class RoleEnforcer:
    """
    Applies a named system role to a base prompt and validates response compliance.
    """

    def apply_role(self, role_name: str, base_prompt: str = "") -> str:
        """
        Return the system prompt for *role_name*, optionally appending *base_prompt*.

        Parameters
        ----------
        role_name:
            Key into :data:`ROLES`.
        base_prompt:
            Additional prompt text to append after the role system prompt.
        """
        role = self._get_role(role_name)
        combined = role.system_prompt
        if base_prompt:
            combined = combined.rstrip() + "\n\n" + base_prompt.strip()
        return combined

    def validate_role_compliance(self, response: str, role: SystemRole) -> bool:
        """
        Heuristically validate that *response* complies with *role* restrictions.

        Returns ``True`` if no restriction violations are detected.
        This is a lightweight check — production systems should use a dedicated
        classifier or secondary LLM call.
        """
        response_lower = response.lower()

        restriction_signals = {
            "no_legal_advice": [r"\blegal advice\b", r"\byou should consult a lawyer\b"],
            "no_medical_advice": [r"\bmedical advice\b", r"\bconsult a doctor\b"],
            "cite_data_sources": [],  # Checked separately by CitationRequiredConstraint
            "no_feature_promises": [r"\bwe will add\b", r"\bcoming soon\b", r"\bwe promise\b"],
            "no_pricing_commitments": [r"\bprice will be\b", r"\bcost is fixed at\b"],
            "knowledge_base_only": [],
        }

        for restriction in role.restrictions:
            patterns = restriction_signals.get(restriction, [])
            for pattern in patterns:
                if re.search(pattern, response_lower):
                    logger.warning(
                        "Role compliance violation: restriction '%s' triggered by pattern '%s'",
                        restriction,
                        pattern,
                    )
                    return False
        return True

    def list_roles(self) -> List[str]:
        """Return available role names."""
        return list(ROLES.keys())

    def _get_role(self, role_name: str) -> SystemRole:
        if role_name not in ROLES:
            raise ValueError(f"Unknown role '{role_name}'. Available: {list(ROLES.keys())}")
        return ROLES[role_name]
