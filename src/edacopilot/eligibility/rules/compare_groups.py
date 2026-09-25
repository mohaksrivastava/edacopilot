"""COMPARE_GROUPS and EQUIVALENCE families (ARCHITECTURE.md, Section 7.3).

Which family a question lands in is decided by three things only: the
outcome's measurement level, the number of group levels, and the design the
user confirmed. None of them is guessed here -- `validate_spec` has already
raised an ambiguity for anything the data could not settle, so by the time
this module runs the design is a fact.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from edacore.contracts import SemanticType

from ..checks import VariableMap, infer_types, numeric_like
from ..spec import Design, Goal, QuestionSpec
from ._base import Method, MethodFamily, RuleSet, UnsupportedQuestionError, levels, two_levels


def _vars(spec: QuestionSpec, df: pd.DataFrame) -> VariableMap:
    vmap: VariableMap = {
        "outcome": spec.variables["outcome"],
        "group": spec.variables["group"],
    }
    for role in ("subject", "id", "entity"):
        if role in spec.variables:
            vmap["subject"] = spec.variables[role]
            break
    for role in ("factor_b", "factor2"):
        if role in spec.variables:
            vmap[role] = spec.variables[role]
    return vmap


def _grouped(spec: QuestionSpec, _: pd.DataFrame) -> dict[str, Any]:
    return {"group_col": spec.variables["group"], "value_col": spec.variables["outcome"]}


def _outcome_group(spec: QuestionSpec, _: pd.DataFrame) -> dict[str, Any]:
    return {"outcome": spec.variables["outcome"], "group": spec.variables["group"]}


def _subject_within(spec: QuestionSpec, _: pd.DataFrame) -> dict[str, Any]:
    return {
        "dv": spec.variables["outcome"],
        "subject": _subject_col(spec),
        "within": spec.variables["group"],
    }


def _subject_col(spec: QuestionSpec) -> str:
    for role in ("subject", "id", "entity"):
        if role in spec.variables:
            return spec.variables[role]
    raise UnsupportedQuestionError(
        "a related-samples comparison needs a subject identifier column; name it as the "
        "'subject' role in the question"
    )


def _paired_ab(spec: QuestionSpec, df: pd.DataFrame) -> dict[str, Any]:
    """Two-level paired data reaches edacore's paired tests as two columns.

    The question arrives in long form (outcome / group / subject), so the
    two level names double as the wide column names the orchestrator will
    pivot to. Recording them here keeps the candidate's params honest about
    what will actually be run.
    """
    first, second = two_levels(df, spec.variables["group"])
    return {"a": str(first), "b": str(second)}


def _proportion_params(spec: QuestionSpec, df: pd.DataFrame) -> dict[str, Any]:
    group_levels = two_levels(df, spec.variables["group"])
    outcome_levels = levels(df, spec.variables["outcome"])
    return {
        "outcome": spec.variables["outcome"],
        "group": spec.variables["group"],
        "groups": (group_levels[0], group_levels[1]),
        "event": sorted(outcome_levels, key=str)[-1],
    }


def _table_params(spec: QuestionSpec, _: pd.DataFrame) -> dict[str, Any]:
    return {"a": spec.variables["outcome"], "b": spec.variables["group"]}


def _factorial_params(spec: QuestionSpec, _: pd.DataFrame) -> dict[str, Any]:
    return {
        "outcome": spec.variables["outcome"],
        "factors": [spec.variables["group"], _second_factor(spec)],
    }


def _tost_params(spec: QuestionSpec, _: pd.DataFrame) -> dict[str, Any]:
    if spec.equivalence_bounds is None:
        raise UnsupportedQuestionError(
            "an equivalence question needs equivalence_bounds: the smallest difference "
            "that would matter, in the outcome's own units"
        )
    low, high = spec.equivalence_bounds
    return {
        "outcome": spec.variables["outcome"],
        "group": spec.variables["group"],
        "low": low,
        "high": high,
    }


TWO_INDEPENDENT_NUMERIC = MethodFamily(
    name="two_independent_numeric",
    variables=_vars,
    methods=(
        Method("student_t", _grouped, interpretability=1),
        Method("welch_t", _grouped, interpretability=1),
        Method("yuen_trimmed_t", _grouped, interpretability=2),
        Method("mann_whitney", _grouped, interpretability=3),
        Method("brunner_munzel", _grouped, interpretability=4),
        Method("permutation_test_2s", _grouped, interpretability=2),
        Method("bootstrap_diff", _grouped, interpretability=2),
        Method("ks_two_sample", _grouped, interpretability=6),
    ),
)

TWO_PAIRED_NUMERIC = MethodFamily(
    name="two_paired_numeric",
    variables=_vars,
    methods=(
        Method("paired_t", _paired_ab, interpretability=1),
        Method("wilcoxon_signed_rank", _paired_ab, interpretability=3),
        Method("sign_test_paired", _paired_ab, interpretability=4),
        Method("permutation_test_paired", _paired_ab, interpretability=2),
    ),
)

K_INDEPENDENT_NUMERIC = MethodFamily(
    name="k_independent_numeric",
    variables=_vars,
    methods=(
        Method("one_way_anova", _outcome_group, interpretability=1),
        Method("welch_anova", _outcome_group, interpretability=1),
        Method("alexander_govern", _outcome_group, interpretability=3),
        Method("kruskal_wallis", _outcome_group, interpretability=2),
        Method("permutation_anova", _outcome_group, interpretability=2),
    ),
)

K_REPEATED_NUMERIC = MethodFamily(
    name="k_repeated_numeric",
    variables=_vars,
    methods=(
        Method("repeated_measures_anova", _subject_within, interpretability=1),
        Method("friedman", _subject_within, interpretability=2),
    ),
)

K_REPEATED_BINARY = MethodFamily(
    name="k_repeated_binary",
    variables=_vars,
    methods=(Method("cochran_q", _subject_within, interpretability=1),),
)

# The column name the factorial family derives for its cells. Prefixed so it
# cannot collide with a user column, and visible in any fact_id that mentions
# it, so a check reported against it is self-explaining.
FACTORIAL_CELL_COLUMN = "_cell"


def _second_factor(spec: QuestionSpec) -> str:
    second = spec.variables.get("factor_b") or spec.variables.get("factor2")
    if second is None:
        raise UnsupportedQuestionError("a factorial question needs a second factor")
    return second


def _add_cell_column(spec: QuestionSpec, df: pd.DataFrame) -> pd.DataFrame:
    """A factorial design's variance and normality assumptions are about its
    cells, not about either factor on its own: two levels of factor A can
    have identical spread overall while one A-B cell is far more variable
    than the rest."""
    first, second = spec.variables["group"], _second_factor(spec)
    cells = df[first].astype(str) + " x " + df[second].astype(str)
    return df.assign(**{FACTORIAL_CELL_COLUMN: cells})


def _factorial_vars(spec: QuestionSpec, df: pd.DataFrame) -> VariableMap:
    return {
        "outcome": spec.variables["outcome"],
        "group": FACTORIAL_CELL_COLUMN,
        "factor_a": spec.variables["group"],
        "factor_b": _second_factor(spec),
    }


FACTORIAL_NUMERIC = MethodFamily(
    name="factorial_numeric",
    variables=_factorial_vars,
    prepare=_add_cell_column,
    methods=(
        Method("two_way_anova", _factorial_params, interpretability=1),
        Method("aligned_rank_transform_anova", _factorial_params, interpretability=3),
    ),
)

BINARY_OUTCOME_GROUPS = MethodFamily(
    name="binary_outcome_groups",
    variables=_vars,
    methods=(
        Method("two_proportion_z", _proportion_params, interpretability=1),
        Method("chi2_independence", _table_params, interpretability=2),
        Method("fisher_exact", _table_params, interpretability=2),
    ),
)

EQUIVALENCE_TWO_GROUPS = MethodFamily(
    name="equivalence_two_groups",
    variables=_vars,
    methods=(Method("tost_equivalence", _tost_params, interpretability=1),),
)

ALL_FAMILIES: tuple[MethodFamily, ...] = (
    TWO_INDEPENDENT_NUMERIC,
    TWO_PAIRED_NUMERIC,
    K_INDEPENDENT_NUMERIC,
    K_REPEATED_NUMERIC,
    K_REPEATED_BINARY,
    FACTORIAL_NUMERIC,
    BINARY_OUTCOME_GROUPS,
    EQUIVALENCE_TWO_GROUPS,
)


def _has_second_factor(spec: QuestionSpec) -> bool:
    return bool(spec.variables.get("factor_b") or spec.variables.get("factor2"))


def pick_family(spec: QuestionSpec, df: pd.DataFrame) -> MethodFamily:
    """Section 7.3's trigger column, as code.

    Order matters: a factorial question is still a group comparison, and a
    binary outcome is still "numeric" to pandas, so the more specific
    triggers are tested first.
    """
    outcome, group = spec.variables["outcome"], spec.variables["group"]
    outcome_type, _ = infer_types(df)[outcome]
    k = len(levels(df, group))
    design = spec.design

    if _has_second_factor(spec):
        if not numeric_like(outcome_type) and outcome_type is not SemanticType.ORDINAL:
            raise UnsupportedQuestionError(
                f"a factorial comparison needs a numeric or ordinal outcome; '{outcome}' "
                f"was read as {outcome_type.value}"
            )
        return FACTORIAL_NUMERIC

    if outcome_type is SemanticType.BINARY:
        if design in (Design.PAIRED, Design.REPEATED):
            return K_REPEATED_BINARY
        return BINARY_OUTCOME_GROUPS

    if not (numeric_like(outcome_type) or outcome_type is SemanticType.ORDINAL):
        raise UnsupportedQuestionError(
            f"'{outcome}' was read as {outcome_type.value}; comparing it across groups "
            f"is a categorical-association question (goal=association), not a group "
            f"comparison of a measured quantity"
        )

    related = design in (Design.PAIRED, Design.REPEATED)
    if k == 2:
        return TWO_PAIRED_NUMERIC if related else TWO_INDEPENDENT_NUMERIC
    return K_REPEATED_NUMERIC if related else K_INDEPENDENT_NUMERIC


def pick_equivalence_family(spec: QuestionSpec, df: pd.DataFrame) -> MethodFamily:
    k = len(levels(df, spec.variables["group"]))
    if k != 2:
        raise UnsupportedQuestionError(
            f"equivalence testing in v1 covers two groups; '{spec.variables['group']}' has {k}"
        )
    return EQUIVALENCE_TWO_GROUPS


RULES_COMPARE_GROUPS = RuleSet(
    goal_name=Goal.COMPARE_GROUPS.value,
    family=pick_family,
    families=tuple(f.name for f in ALL_FAMILIES if f is not EQUIVALENCE_TWO_GROUPS),
)

RULES_EQUIVALENCE = RuleSet(
    goal_name=Goal.EQUIVALENCE.value,
    family=pick_equivalence_family,
    families=(EQUIVALENCE_TWO_GROUPS.name,),
)
