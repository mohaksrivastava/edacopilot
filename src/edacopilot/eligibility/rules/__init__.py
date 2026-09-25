"""`RULES[spec.goal]` (ARCHITECTURE.md, Section 7.2), one entry per Goal.

Goals with no hypothesis-testing methods behind them are absent on purpose,
and `select_candidates` reports that as a missing feature naming the
milestone rather than as a statistical verdict: DESCRIBE is profiling
(Section 6.1), MISSINGNESS is M10, OUTLIERS and TRANSFORM are M11.
"""

from __future__ import annotations

from ..spec import Goal
from ._base import (
    Method,
    MethodFamily,
    RuleSet,
    UnsupportedQuestionError,
    levels,
    two_levels,
)
from .association import ALL_FAMILIES as ASSOCIATION_FAMILIES
from .association import RULES_ASSOCIATION
from .compare_groups import ALL_FAMILIES as COMPARE_GROUPS_FAMILIES
from .compare_groups import RULES_COMPARE_GROUPS, RULES_EQUIVALENCE
from .distribution_fit import ALL_FAMILIES as DISTRIBUTION_FIT_FAMILIES
from .distribution_fit import RULES_DISTRIBUTION_FIT, RULES_TREND

RULES: dict[Goal, RuleSet] = {
    Goal.COMPARE_GROUPS: RULES_COMPARE_GROUPS,
    Goal.ASSOCIATION: RULES_ASSOCIATION,
    Goal.DISTRIBUTION_FIT: RULES_DISTRIBUTION_FIT,
    Goal.EQUIVALENCE: RULES_EQUIVALENCE,
    Goal.TREND: RULES_TREND,
}

# Which milestone will serve each goal that has no rules yet.
PENDING_GOALS: dict[Goal, str] = {
    Goal.DESCRIBE: "profiling (Section 6.1), which describes rather than tests",
    Goal.MISSINGNESS: "M10 (Section 6.3)",
    Goal.OUTLIERS: "M11 (Section 6.4)",
    Goal.TRANSFORM: "M11 (Section 6.5)",
}

ALL_FAMILIES: tuple[MethodFamily, ...] = (
    *COMPARE_GROUPS_FAMILIES,
    *ASSOCIATION_FAMILIES,
    *DISTRIBUTION_FIT_FAMILIES,
)

__all__ = [
    "ALL_FAMILIES",
    "PENDING_GOALS",
    "RULES",
    "Method",
    "MethodFamily",
    "RuleSet",
    "UnsupportedQuestionError",
    "levels",
    "two_levels",
]
