"""The deterministic eligibility engine (ARCHITECTURE.md, Section 7).

Public surface:

    spec = QuestionSpec(goal=Goal.COMPARE_GROUPS, variables={...}, design=...)
    spec = validate_spec(spec, df)        # fills spec.ambiguities
    if spec.ambiguities:                  # ask the user; do not proceed
        ...
    candidates = select_candidates(spec, df)

Nothing here calls an LLM, and nothing here consults a persona: rule 2 says
no persona may propose an ineligible method, which only holds if
eligibility is decided before any persona sees the question.
"""

from __future__ import annotations

from .checks import CheckCache, ResolutionContext, resolve
from .column_matching import ColumnMatch, ColumnMatchConfig, match_column
from .engine import CandidateSet, select_candidates
from .rules import RULES, UnsupportedQuestionError
from .spec import (
    AmbiguousSpecError,
    Design,
    Goal,
    InvalidSpecError,
    QuestionSpec,
    describe_spec,
    require_resolved,
    subject_column,
    validate_spec,
)

__all__ = [
    "RULES",
    "AmbiguousSpecError",
    "CandidateSet",
    "CheckCache",
    "ColumnMatch",
    "ColumnMatchConfig",
    "Design",
    "Goal",
    "InvalidSpecError",
    "QuestionSpec",
    "ResolutionContext",
    "UnsupportedQuestionError",
    "describe_spec",
    "match_column",
    "require_resolved",
    "resolve",
    "subject_column",
    "select_candidates",
    "validate_spec",
]
