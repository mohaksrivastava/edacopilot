"""DISTRIBUTION_FIT and TREND families (ARCHITECTURE.md, Section 7.3).

A distribution-fit question compares one variable against a reference the
question itself supplies -- a value (`one_sample_t`'s mu0, `binomial_test`'s
p0) or a set of proportions (`chi2_goodness_of_fit`). Section 7.1's original
`QuestionSpec` had nowhere to put those; M4 added `reference_value` and
`reference_proportions` for exactly this, mirroring `equivalence_bounds`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from edacore.contracts import SemanticType

from ..checks import VariableMap, infer_types, numeric_like
from ..spec import Goal, QuestionSpec
from ._base import Method, MethodFamily, RuleSet, UnsupportedQuestionError, levels


def _vars(spec: QuestionSpec, _: pd.DataFrame) -> VariableMap:
    return {"variable": spec.variables["variable"], "outcome": spec.variables["variable"]}


def _reference_value(spec: QuestionSpec, what: str) -> float:
    if spec.reference_value is None:
        raise UnsupportedQuestionError(
            f"this question needs a reference {what} to compare against "
            "(QuestionSpec.reference_value)"
        )
    return spec.reference_value


def _mu0_params(spec: QuestionSpec, _: pd.DataFrame) -> dict[str, Any]:
    return {"col": spec.variables["variable"], "mu0": _reference_value(spec, "value")}


def _col_only(spec: QuestionSpec, _: pd.DataFrame) -> dict[str, Any]:
    return {"col": spec.variables["variable"]}


def _p0_params(spec: QuestionSpec, _: pd.DataFrame) -> dict[str, Any]:
    return {"col": spec.variables["variable"], "p0": _reference_value(spec, "proportion")}


def _expected_params(spec: QuestionSpec, df: pd.DataFrame) -> dict[str, Any]:
    col = spec.variables["variable"]
    if spec.reference_proportions is not None:
        return {"col": col, "expected": dict(spec.reference_proportions)}
    found = levels(df, col)
    raise UnsupportedQuestionError(
        f"a goodness-of-fit test needs the proportions to compare against "
        f"(QuestionSpec.reference_proportions), one per level of '{col}': {found}"
    )


ONE_SAMPLE_NUMERIC = MethodFamily(
    name="one_sample_numeric",
    variables=_vars,
    methods=(
        Method("one_sample_t", _mu0_params, interpretability=1),
        Method("wilcoxon_one_sample", _mu0_params, interpretability=3),
        Method("sign_test", _mu0_params, interpretability=4),
        Method("bootstrap_one_sample", _col_only, interpretability=2),
    ),
)

ONE_SAMPLE_CATEGORICAL = MethodFamily(
    name="one_sample_categorical",
    variables=_vars,
    methods=(
        Method("binomial_test", _p0_params, interpretability=1),
        Method("chi2_goodness_of_fit", _expected_params, interpretability=2),
    ),
)

ALL_FAMILIES: tuple[MethodFamily, ...] = (ONE_SAMPLE_NUMERIC, ONE_SAMPLE_CATEGORICAL)

# Section 7.3's TREND family. Its methods live in Section 6.9
# (`timeseries.py`), which is M12 -- so the family is declared here, with the
# names it will offer, and asking for it raises rather than reporting
# perfectly good methods as INELIGIBLE (they are not ineligible; they do not
# exist yet).
TS_STATIONARITY_METHODS = ("stationarity_verdict", "stl_decompose", "detect_change_points")


def pick_family(spec: QuestionSpec, df: pd.DataFrame) -> MethodFamily:
    col = spec.variables["variable"]
    semantic_type, _ = infer_types(df)[col]
    if numeric_like(semantic_type):
        return ONE_SAMPLE_NUMERIC
    if semantic_type in (SemanticType.BINARY, SemanticType.NOMINAL, SemanticType.ORDINAL):
        return ONE_SAMPLE_CATEGORICAL
    raise UnsupportedQuestionError(
        f"'{col}' was read as {semantic_type.value}; there is no v1 one-sample family for it"
    )


def pick_trend_family(spec: QuestionSpec, df: pd.DataFrame) -> MethodFamily:
    raise UnsupportedQuestionError(
        "trend questions use the ts_stationarity family "
        f"({', '.join(TS_STATIONARITY_METHODS)}), which lands with Section 6.9 in M12"
    )


RULES_DISTRIBUTION_FIT = RuleSet(
    goal_name=Goal.DISTRIBUTION_FIT.value,
    family=pick_family,
    families=tuple(f.name for f in ALL_FAMILIES),
)

RULES_TREND = RuleSet(
    goal_name=Goal.TREND.value,
    family=pick_trend_family,
    families=("ts_stationarity",),
)
