"""QuestionSpec and its deterministic validation (ARCHITECTURE.md, Section 7.1).

An LLM turns the user's words into a `QuestionSpec`; nothing in this module
calls an LLM. `validate_spec` then checks that spec against the actual data
and returns a copy carrying any `ambiguities` it found. Section 7.1's rule
is that a non-empty `ambiguities` list stops the turn: the orchestrator asks
the user, and `select_candidates` refuses to run (`AmbiguousSpecError`).

The design cross-check is the point of this module. Choosing between an
independent and a paired design is the single most consequential decision in
a group comparison -- it changes which tests are valid, and getting it wrong
is the most common junior-analyst error (Section 15.3's `paired_as_independent`
trap). So `design` is *never* inferred here. When the data contradicts the
design the LLM proposed, or when the data alone cannot settle it, this module
raises an ambiguity in the user's own terms and stops. It never silently
switches `design` to whatever the ids suggest.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from edacore.assumptions import check_design_crossing
from edacore.contracts import CheckStatus, SemanticType
from edacore.profiling import detect_structure, infer_semantic_types


class Goal(StrEnum):
    DESCRIBE = "describe"
    COMPARE_GROUPS = "compare_groups"
    ASSOCIATION = "association"
    DISTRIBUTION_FIT = "distribution_fit"
    EQUIVALENCE = "equivalence"
    TREND = "trend"
    MISSINGNESS = "missingness"
    OUTLIERS = "outliers"
    TRANSFORM = "transform"


class Design(StrEnum):
    INDEPENDENT = "independent"
    PAIRED = "paired"
    REPEATED = "repeated"
    UNKNOWN = "unknown"


class QuestionSpec(BaseModel):
    """A user question, resolved to columns and a design (Section 7.1)."""

    model_config = ConfigDict(frozen=True)

    goal: Goal
    variables: dict[str, str] = Field(default_factory=dict)
    design: Design = Design.UNKNOWN
    alternative: str = "two-sided"
    alpha: float | None = None
    equivalence_bounds: tuple[float, float] | None = None
    # Section 7.1's original schema had no way to carry the reference a
    # DISTRIBUTION_FIT question is asked against, but every method in that
    # goal's families needs one: one_sample_t's mu0, binomial_test's p0,
    # chi2_goodness_of_fit's expected proportions. Added in M4, alongside
    # equivalence_bounds, which plays exactly the same role for EQUIVALENCE.
    reference_value: float | None = None
    reference_proportions: dict[str, float] | None = None
    user_text: str = ""
    confirmed_by_user: set[str] = Field(default_factory=set)
    ambiguities: list[str] = Field(default_factory=list)

    def with_ambiguities(self, ambiguities: list[str]) -> QuestionSpec:
        return self.model_copy(update={"ambiguities": ambiguities})


class AmbiguousSpecError(RuntimeError):
    """Raised when a spec with unresolved ambiguities reaches the engine.

    Section 7.1: "If `ambiguities` is non-empty, the orchestrator asks the
    user and does not proceed." This is the deterministic backstop for that
    rule -- an ambiguity cannot be skipped by calling the engine directly.
    """

    def __init__(self, ambiguities: list[str]) -> None:
        self.ambiguities = ambiguities
        joined = "\n  - ".join(ambiguities)
        super().__init__(
            "the question has unresolved ambiguities and must be confirmed with the user "
            f"before any test runs:\n  - {joined}"
        )


class InvalidSpecError(ValueError):
    """Raised for a spec that cannot be repaired by asking the user: a named
    column that does not exist, or a role the goal does not accept. These
    are spec-building bugs, not questions for the user."""


# Which `variables` roles each goal requires. A spec may carry extra roles
# (e.g. "subject" alongside "outcome"/"group"); only the listed ones are
# mandatory.
_REQUIRED_ROLES: dict[Goal, tuple[str, ...]] = {
    Goal.DESCRIBE: ("variable",),
    Goal.COMPARE_GROUPS: ("outcome", "group"),
    Goal.ASSOCIATION: ("x", "y"),
    Goal.DISTRIBUTION_FIT: ("variable",),
    Goal.EQUIVALENCE: ("outcome", "group"),
    Goal.TREND: ("variable",),
    Goal.MISSINGNESS: (),
    Goal.OUTLIERS: ("variable",),
    Goal.TRANSFORM: ("variable",),
}

# Roles that name a subject/unit identifier rather than a measured variable.
ID_ROLES = ("subject", "id", "entity")

_GROUPING_SCALES = frozenset(
    {SemanticType.NOMINAL, SemanticType.ORDINAL, SemanticType.BINARY, SemanticType.DISCRETE}
)


def _id_column(spec: QuestionSpec, df: pd.DataFrame) -> tuple[str | None, bool]:
    """The subject-identifier column for this spec, and whether it was named
    by the spec (True) or inferred from the data's structure (False)."""
    for role in ID_ROLES:
        named = spec.variables.get(role)
        if named is not None:
            return named, True
    entity_id = detect_structure(df).entity_id
    if entity_id is not None and entity_id in df.columns:
        return entity_id, False
    return None, False


def _design_ambiguities(spec: QuestionSpec, df: pd.DataFrame) -> list[str]:
    """Section 7.1 step 2: cross-check the proposed design against the ids.

    Every branch that cannot be settled from the data alone produces a
    question for the user; none of them changes `spec.design`.
    """
    group = spec.variables.get("group")
    if group is None or group not in df.columns:
        return []

    id_col, id_named = _id_column(spec, df)
    if id_col is None:
        if spec.design is Design.UNKNOWN:
            return [
                f"Is each row of '{group}' a different subject, or are the same subjects "
                "measured under more than one condition? There is no id column to tell "
                "from the data, and the answer decides which tests are valid."
            ]
        return []

    crossing = check_design_crossing(df, group=group, id_col=id_col)
    n_crossing = int(crossing.statistic or 0)
    found_via = (
        f"'{id_col}'"
        if id_named
        else f"'{id_col}' (which looks like a subject id, though the question did not name it)"
    )

    if n_crossing > 0:
        if spec.design is Design.INDEPENDENT:
            return [
                f"{n_crossing} value(s) of {found_via} appear in more than one '{group}' "
                f"group, so the same subject seems to be measured more than once. The "
                f"question was read as comparing independent groups. Are these paired "
                f"measurements on the same subjects, or does '{id_col}' mean something "
                f"else here?"
            ]
        if spec.design is Design.UNKNOWN:
            return [
                f"{n_crossing} value(s) of {found_via} appear in more than one '{group}' "
                f"group. Are these repeated measurements on the same subjects (paired), "
                f"or independent groups that happen to share id values?"
            ]
        return []

    if spec.design in (Design.PAIRED, Design.REPEATED):
        return [
            f"The question was read as comparing related (paired) measurements, but no "
            f"value of {found_via} appears in more than one '{group}' group, so nothing "
            f"links a row in one group to a row in another. Are these independent groups, "
            f"or is the subject identifier a different column?"
        ]

    if crossing.status is CheckStatus.FAIL:
        # Ids repeat, but only within a single group: clustering, not pairing.
        return [
            f"Some values of {found_via} appear more than once within the same '{group}' "
            f"group, so rows are clustered rather than independent. Clustered data needs "
            f"a method that accounts for it; should these rows be aggregated per subject "
            f"first?"
        ]

    if spec.design is Design.UNKNOWN:
        return [
            f"Each value of {found_via} appears exactly once, which is consistent with "
            f"independent groups. Can you confirm the '{group}' groups are independent?"
        ]
    return []


def _type_ambiguities(spec: QuestionSpec, df: pd.DataFrame) -> list[str]:
    """Section 7.1 step 1's second half: roles whose column has a semantic
    type the goal cannot use, where the type itself is the open question.

    A numeric-looking Likert column is the case this exists for
    (Section 15.3's `ordinal_as_numeric` trap): treating it as continuous is
    a decision, so the user makes it.
    """
    ambiguities: list[str] = []
    inferred = infer_semantic_types(df)

    for role in ("outcome", "x", "y", "variable"):
        col = spec.variables.get(role)
        if col is None or col not in df.columns:
            continue
        semantic_type, confidence = inferred[col]
        profile_flags = _numeric_coded_ordinal(df[col], semantic_type)
        if profile_flags and role not in spec.confirmed_by_user:
            ambiguities.append(
                f"'{col}' is stored as numbers but only takes {df[col].nunique()} distinct "
                f"values, so it may be an ordinal rating rather than a measured quantity. "
                f"Averages and Pearson correlations assume the gaps between values are "
                f"equal; is that true here?"
            )
        elif confidence < 0.8 and role not in spec.confirmed_by_user:
            ambiguities.append(
                f"'{col}' was read as {semantic_type.value}, but only with "
                f"{confidence:.0%} confidence. Is that the right reading?"
            )
    return ambiguities


def _numeric_coded_ordinal(series: pd.Series, semantic_type: SemanticType) -> bool:
    return semantic_type is SemanticType.ORDINAL and pd.api.types.is_numeric_dtype(series)


def validate_spec(spec: QuestionSpec, df: pd.DataFrame) -> QuestionSpec:
    """Validate `spec` against `df` and return a copy carrying `ambiguities`.

    Raises `InvalidSpecError` for problems the user cannot resolve by
    answering a question (a missing column, a missing required role, a
    grouping column with fewer than two levels). Everything else becomes an
    ambiguity, and `select_candidates` will refuse to run until they are
    resolved.
    """
    missing_roles = [role for role in _REQUIRED_ROLES[spec.goal] if not spec.variables.get(role)]
    if missing_roles:
        raise InvalidSpecError(
            f"goal '{spec.goal.value}' needs {_REQUIRED_ROLES[spec.goal]}; missing: {missing_roles}"
        )

    unknown = {role: col for role, col in spec.variables.items() if col not in df.columns}
    if unknown:
        raise InvalidSpecError(f"columns named in the question are not in the data: {unknown}")

    ambiguities: list[str] = []
    group = spec.variables.get("group")
    if group is not None:
        n_levels = int(df[group].nunique(dropna=True))
        if n_levels < 2:
            raise InvalidSpecError(
                f"'{group}' has {n_levels} distinct value(s); a group comparison needs at least 2"
            )
        semantic_type, _ = infer_semantic_types(df)[group]
        if semantic_type not in _GROUPING_SCALES:
            ambiguities.append(
                f"'{group}' was read as {semantic_type.value}, which is an odd thing to "
                f"group by ({n_levels} distinct values). Is it really a grouping variable?"
            )

    ambiguities.extend(_design_ambiguities(spec, df))
    ambiguities.extend(_type_ambiguities(spec, df))
    return spec.with_ambiguities(ambiguities)


def require_resolved(spec: QuestionSpec) -> None:
    """Section 7.1 step 3, as a guard the engine calls before anything else."""
    if spec.ambiguities:
        raise AmbiguousSpecError(list(spec.ambiguities))


def describe_spec(spec: QuestionSpec) -> dict[str, Any]:
    """A flat, JSON-ready summary for provenance and card rendering."""
    return {
        "goal": spec.goal.value,
        "design": spec.design.value,
        "variables": dict(spec.variables),
        "alternative": spec.alternative,
        "alpha": spec.alpha,
        "confirmed_by_user": sorted(spec.confirmed_by_user),
        "ambiguities": list(spec.ambiguities),
    }
