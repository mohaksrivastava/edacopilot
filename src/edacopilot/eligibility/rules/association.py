"""ASSOCIATION families (ARCHITECTURE.md, Section 7.3).

Section 7.3 lists five association families, and which one a question lands
in depends entirely on the two variables' measurement levels. That is also
where `ordinal_as_numeric` (Section 15.3) bites: a Likert column read as
continuous routes the question to `numeric_numeric` and offers Pearson,
which assumes the gap from 1 to 2 means the same as the gap from 4 to 5.
`validate_spec` raises that as an ambiguity before this module ever runs, so
the routing below can trust the types it is given.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from edacore.contracts import SemanticType

from ..checks import VariableMap, infer_types, numeric_like
from ..spec import Design, Goal, QuestionSpec
from ._base import Method, MethodFamily, RuleSet, UnsupportedQuestionError, levels


def _xy(spec: QuestionSpec, _: pd.DataFrame) -> dict[str, Any]:
    return {"x": spec.variables["x"], "y": spec.variables["y"]}


def _ab(spec: QuestionSpec, _: pd.DataFrame) -> dict[str, Any]:
    return {"a": spec.variables["x"], "b": spec.variables["y"]}


def _xy_vars(spec: QuestionSpec, _: pd.DataFrame) -> VariableMap:
    vmap: VariableMap = {"x": spec.variables["x"], "y": spec.variables["y"]}
    vmap["a"], vmap["b"] = vmap["x"], vmap["y"]
    for role in ("subject", "id", "entity"):
        if role in spec.variables:
            vmap["subject"] = spec.variables[role]
            break
    return vmap


def _binary_numeric_vars(spec: QuestionSpec, df: pd.DataFrame) -> VariableMap:
    binary, numeric = _split_binary_numeric(spec, df)
    # Both methods here read the pair the same way -- the binary column
    # groups, the other is measured -- so the map says so. Crucially it
    # does NOT keep the binary column under `y`: "numeric" resolves over
    # x/y/outcome, and leaving it there would make point_biserial fail its
    # own hard assumption on the very column it is designed to take.
    return {
        "x": numeric,
        "binary": binary,
        "outcome": numeric,
        "group": binary,
    }


def _split_binary_numeric(spec: QuestionSpec, df: pd.DataFrame) -> tuple[str, str]:
    x, y = spec.variables["x"], spec.variables["y"]
    types = infer_types(df)
    if types[x][0] is SemanticType.BINARY:
        return x, y
    return y, x


def _point_biserial_params(spec: QuestionSpec, df: pd.DataFrame) -> dict[str, Any]:
    binary, numeric = _split_binary_numeric(spec, df)
    return {"binary": binary, "x": numeric}


def _mann_whitney_params(spec: QuestionSpec, df: pd.DataFrame) -> dict[str, Any]:
    binary, numeric = _split_binary_numeric(spec, df)
    return {"group_col": binary, "value_col": numeric}


def _ordinal_binary_vars(spec: QuestionSpec, df: pd.DataFrame) -> VariableMap:
    binary, ordinal = _split_binary_ordinal(spec, df)
    return {
        "x": spec.variables["x"],
        "y": spec.variables["y"],
        "binary": binary,
        "ordinal": ordinal,
    }


def _split_binary_ordinal(spec: QuestionSpec, df: pd.DataFrame) -> tuple[str, str]:
    x, y = spec.variables["x"], spec.variables["y"]
    types = infer_types(df)
    if types[x][0] is SemanticType.BINARY:
        return x, y
    return y, x


def _trend_params(spec: QuestionSpec, df: pd.DataFrame) -> dict[str, Any]:
    binary, ordinal = _split_binary_ordinal(spec, df)
    event = sorted(levels(df, binary), key=str)[-1]
    return {"binary": binary, "ordinal": ordinal, "event": event}


def _partial_params(spec: QuestionSpec, _: pd.DataFrame) -> dict[str, Any]:
    covars = [col for role, col in spec.variables.items() if role.startswith("covar")]
    if not covars:
        raise UnsupportedQuestionError(
            "a partial correlation needs at least one control variable; name it as a "
            "'covariate' role in the question"
        )
    return {"x": spec.variables["x"], "y": spec.variables["y"], "covars": covars}


NUMERIC_NUMERIC = MethodFamily(
    name="numeric_numeric",
    variables=_xy_vars,
    methods=(
        Method("pearson", _xy, interpretability=1),
        Method("spearman", _xy, interpretability=2),
        Method("kendall_tau", _xy, interpretability=3),
        Method("distance_correlation", _xy, interpretability=5),
        Method("mutual_information", _xy, interpretability=6),
    ),
)

ORDINAL_ANY = MethodFamily(
    name="ordinal_any",
    variables=_ordinal_binary_vars,
    methods=(
        Method("spearman", _xy, interpretability=1),
        Method("kendall_tau", _xy, interpretability=2),
        Method("cochran_armitage_trend", _trend_params, interpretability=3),
    ),
    note="cochran_armitage_trend only applies to a binary outcome against an ordinal exposure",
)

BINARY_NUMERIC = MethodFamily(
    name="binary_numeric",
    variables=_binary_numeric_vars,
    methods=(
        Method("point_biserial", _point_biserial_params, interpretability=1),
        Method("mann_whitney", _mann_whitney_params, interpretability=2),
    ),
)

TWO_CATEGORICAL_INDEPENDENT = MethodFamily(
    name="two_categorical_independent",
    variables=_xy_vars,
    methods=(
        Method("chi2_independence", _ab, interpretability=1),
        Method("fisher_exact", _ab, interpretability=2),
        Method("g_test", _ab, interpretability=3),
    ),
)

TWO_BINARY_PAIRED = MethodFamily(
    name="two_binary_paired",
    variables=_xy_vars,
    methods=(Method("mcnemar", _ab, interpretability=1),),
)

PARTIAL_CORRELATION = MethodFamily(
    name="numeric_numeric_controlled",
    variables=_xy_vars,
    methods=(Method("partial_correlation", _partial_params, interpretability=2),),
    note=(
        "Section 7.3 lists partial_correlation under the correlation methods without giving "
        "it a family of its own; it gets one here because it is the only association method "
        "that needs a third variable, and offering it without one would always fail"
    ),
)

ALL_FAMILIES: tuple[MethodFamily, ...] = (
    NUMERIC_NUMERIC,
    ORDINAL_ANY,
    BINARY_NUMERIC,
    TWO_CATEGORICAL_INDEPENDENT,
    TWO_BINARY_PAIRED,
    PARTIAL_CORRELATION,
)

_CATEGORICAL = frozenset({SemanticType.NOMINAL, SemanticType.ORDINAL, SemanticType.BINARY})


def pick_family(spec: QuestionSpec, df: pd.DataFrame) -> MethodFamily:
    """Section 7.3's association triggers, most specific first."""
    x, y = spec.variables["x"], spec.variables["y"]
    types = infer_types(df)
    tx, ty = types[x][0], types[y][0]

    if any(role.startswith("covar") for role in spec.variables):
        if not (numeric_like(tx) and numeric_like(ty)):
            raise UnsupportedQuestionError(
                "controlling for a third variable is only defined here for two numeric "
                f"variables; '{x}' is {tx.value} and '{y}' is {ty.value}"
            )
        return PARTIAL_CORRELATION

    both_binary = tx is SemanticType.BINARY and ty is SemanticType.BINARY
    if both_binary and spec.design in (Design.PAIRED, Design.REPEATED):
        return TWO_BINARY_PAIRED

    binary_and_ordinal = (tx is SemanticType.BINARY and ty is SemanticType.ORDINAL) or (
        ty is SemanticType.BINARY and tx is SemanticType.ORDINAL
    )
    if binary_and_ordinal:
        return ORDINAL_ANY

    binary_and_numeric = (tx is SemanticType.BINARY and numeric_like(ty)) or (
        ty is SemanticType.BINARY and numeric_like(tx)
    )
    if binary_and_numeric:
        return BINARY_NUMERIC

    if tx in _CATEGORICAL and ty in _CATEGORICAL:
        if SemanticType.ORDINAL in (tx, ty) and not both_binary:
            return ORDINAL_ANY
        return TWO_CATEGORICAL_INDEPENDENT

    if SemanticType.ORDINAL in (tx, ty):
        return ORDINAL_ANY

    if numeric_like(tx) and numeric_like(ty):
        return NUMERIC_NUMERIC

    raise UnsupportedQuestionError(
        f"no v1 association family covers '{x}' ({tx.value}) against '{y}' ({ty.value})"
    )


RULES_ASSOCIATION = RuleSet(
    goal_name=Goal.ASSOCIATION.value,
    family=pick_family,
    families=tuple(f.name for f in ALL_FAMILIES),
)
