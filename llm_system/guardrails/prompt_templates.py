"""
Prompt template system for grounding LLM responses and reducing hallucinations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class PromptTemplate:
    """
    A reusable prompt template with a system prompt, a user message template,
    and a set of behavioural constraints communicated in the prompt.
    """

    name: str
    system_prompt: str
    user_template: str
    constraints: List[str] = field(default_factory=list)

    def format_user(self, query: str, context: str = "") -> str:
        """Fill the user template with *query* and *context*."""
        return self.user_template.format(query=query, context=context)


# ---------------------------------------------------------------------------
# Built-in templates
# ---------------------------------------------------------------------------

BASELINE_TEMPLATE = PromptTemplate(
    name="baseline",
    system_prompt=(
        "You are a helpful assistant. Answer the user's question to the best of your ability."
    ),
    user_template="{query}",
    constraints=[],
)

GROUNDED_TEMPLATE = PromptTemplate(
    name="grounded",
    system_prompt=(
        "You are a helpful assistant that answers questions based ONLY on the provided context.\n"
        "Rules:\n"
        "1. Only use information present in the context below.\n"
        "2. Cite every factual claim using [Source: <id>] notation.\n"
        "3. If the context does not contain sufficient information, say: "
        "\"I don't have enough context to answer this accurately.\"\n"
        "4. Do not add information from your training data."
    ),
    user_template=(
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer (cite all sources):"
    ),
    constraints=[
        "must_cite_sources",
        "answer_from_context_only",
        "refuse_if_insufficient_context",
    ],
)

STRICT_TEMPLATE = PromptTemplate(
    name="strict",
    system_prompt=(
        "You are a precise enterprise assistant with strict citation requirements.\n"
        "MANDATORY RULES — violating any rule will be considered a critical failure:\n"
        "1. Every sentence that contains a factual claim MUST include a [Source: <id>] citation.\n"
        "2. You MUST NOT generate any information not explicitly present in the provided context.\n"
        "3. If you cannot find an answer in the context, respond ONLY with: "
        "\"INSUFFICIENT_CONTEXT: I cannot find this information in the provided sources.\"\n"
        "4. Do not infer, extrapolate, or speculate beyond the context.\n"
        "5. Use verbatim or near-verbatim quotes where possible."
    ),
    user_template=(
        "=== CONTEXT ===\n{context}\n=== END CONTEXT ===\n\n"
        "Query: {query}\n\n"
        "Response (strict citation required):"
    ),
    constraints=[
        "every_claim_must_be_cited",
        "verbatim_quotes_preferred",
        "refuse_with_code_if_unknown",
        "no_extrapolation",
    ],
)

CONSERVATIVE_TEMPLATE = PromptTemplate(
    name="conservative",
    system_prompt=(
        "You are an extremely conservative assistant deployed in a high-stakes enterprise environment.\n"
        "Your primary obligation is accuracy over completeness.\n"
        "RULES:\n"
        "1. Answer only what you can verify from the context.\n"
        "2. If there is ANY ambiguity, ask a clarifying question instead of guessing.\n"
        "3. Prefix uncertain statements with \"UNCERTAIN:\".\n"
        "4. Prefix unsupported claims with \"UNSUPPORTED:\".\n"
        "5. Cite every claim: [Source: <id>].\n"
        "6. When in doubt, decline to answer rather than risk a hallucination.\n"
        "7. Never provide legal, medical, or financial advice without explicit disclaimer."
    ),
    user_template=(
        "Verified Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Conservative Answer:"
    ),
    constraints=[
        "prefer_refusal_over_hallucination",
        "mark_uncertain_claims",
        "always_cite",
        "request_clarification_on_ambiguity",
    ],
)

_TEMPLATE_REGISTRY: Dict[str, PromptTemplate] = {
    t.name: t
    for t in [BASELINE_TEMPLATE, GROUNDED_TEMPLATE, STRICT_TEMPLATE, CONSERVATIVE_TEMPLATE]
}


def get_template(name: str) -> PromptTemplate:
    """
    Retrieve a built-in template by name.

    Available names: ``"baseline"``, ``"grounded"``, ``"strict"``, ``"conservative"``.
    """
    name = name.lower()
    if name not in _TEMPLATE_REGISTRY:
        raise ValueError(
            f"Unknown template '{name}'. Available: {list(_TEMPLATE_REGISTRY.keys())}"
        )
    return _TEMPLATE_REGISTRY[name]
