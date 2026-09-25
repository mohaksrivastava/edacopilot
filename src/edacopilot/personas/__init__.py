"""The persona engine (ARCHITECTURE.md, Section 8), deterministic parts.

    verdict = propose_with_rationales(candidate_set)
    verdict.card            # "consensus" or "divergence"
    verdict.proposals       # one per distinct method, personas grouped
    verdict.picks           # one per persona, each with its rationale

Section 8.4's LLM-written rationales land in M9. What is here is Section
10.5's deterministic fallback, which is also how the whole test suite runs.
"""

from __future__ import annotations

from edacopilot.eligibility.engine import CandidateSet

from .engine import (
    PREFERENCE_KEYS,
    PersonaPick,
    PersonaVerdict,
    Proposal,
    detect_divergence,
    pick,
    propose,
)
from .policy import (
    PERSONA_ORDER,
    NormalityPolicy,
    PersonaPolicy,
    VariancePolicy,
    get_persona,
    load_personas,
    load_policy,
)
from .rationale import render_card, render_rationale, render_verdict


def propose_with_rationales(candidates: CandidateSet) -> PersonaVerdict:
    """Every persona's pick, grouped into proposals, each with its
    deterministic rationale. The normal entry point."""
    return render_verdict(propose(candidates), candidates.checks)


__all__ = [
    "PERSONA_ORDER",
    "PREFERENCE_KEYS",
    "CandidateSet",
    "NormalityPolicy",
    "PersonaPick",
    "PersonaPolicy",
    "PersonaVerdict",
    "Proposal",
    "VariancePolicy",
    "detect_divergence",
    "get_persona",
    "load_personas",
    "load_policy",
    "pick",
    "propose",
    "propose_with_rationales",
    "render_card",
    "render_rationale",
    "render_verdict",
]
