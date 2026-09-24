"""edacore.assumptions vs R (tests/fixtures/r_reference/). Point estimates
(test statistics) are checked to 1e-6 for every check that has one.

Three p-values are NOT bit-matched against R — see assumptions.py's module
docstring and ARCHITECTURE.md Section 18's M2 changelog entry:
check_normality_anderson, check_normality_lilliefors, check_sphericity_mauchly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import edacore
from edacore.contracts import CheckStatus, SemanticType
from edacore.registry import registry
from tests.helpers.codegen_check import assert_codegen_matches

FIXTURES = Path(__file__).parents[2] / "fixtures" / "r_reference"
_NAMESPACE = {"edacore": edacore}


def _data(name: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "data" / f"{name}.csv")


def _ref(name: str) -> dict[str, Any]:
    with (FIXTURES / f"{name}.json").open() as f:
        result: dict[str, Any] = json.load(f)
        return result


def _close(got: float, want: float, *, abs_tol: float = 1e-6) -> None:
    assert got == pytest.approx(want, abs=abs_tol, rel=1e-6)


# --------------------------------------------------------------------------
# Normality
# --------------------------------------------------------------------------


@pytest.mark.parametrize("sample", ["normal_sample", "skewed_sample"])
def test_check_normality_shapiro_matches_r(sample: str) -> None:
    df = _data(sample)
    ref = _ref(f"check_normality_shapiro__{sample}")
    checks = edacore.assumptions.check_normality_shapiro(df, "x")
    _close(checks[0].statistic, ref["statistic"])
    _close(checks[0].p_value, ref["p_value"])
    assert_codegen_matches(
        registry, "check_normality_shapiro", {"col": "x", "by": None}, df, namespace=_NAMESPACE
    )


@pytest.mark.parametrize("sample", ["normal_sample", "skewed_sample"])
def test_check_normality_dagostino_matches_r(sample: str) -> None:
    df = _data(sample)
    ref = _ref(f"check_normality_dagostino__{sample}")
    checks = edacore.assumptions.check_normality_dagostino(df, "x")
    _close(checks[0].statistic, ref["statistic"])
    _close(checks[0].p_value, ref["p_value"])
    assert_codegen_matches(
        registry, "check_normality_dagostino", {"col": "x", "by": None}, df, namespace=_NAMESPACE
    )


@pytest.mark.parametrize("sample", ["normal_sample", "skewed_sample"])
def test_check_normality_anderson_statistic_matches_r(sample: str) -> None:
    df = _data(sample)
    ref = _ref(f"check_normality_anderson__{sample}")
    checks = edacore.assumptions.check_normality_anderson(df, "x")
    _close(checks[0].statistic, ref["statistic"], abs_tol=1e-8)
    assert_codegen_matches(
        registry, "check_normality_anderson", {"col": "x", "by": None}, df, namespace=_NAMESPACE
    )


@pytest.mark.parametrize("sample", ["normal_sample", "skewed_sample"])
def test_check_normality_lilliefors_statistic_matches_r(sample: str) -> None:
    df = _data(sample)
    ref = _ref(f"check_normality_lilliefors__{sample}")
    checks = edacore.assumptions.check_normality_lilliefors(df, "x")
    _close(checks[0].statistic, ref["statistic"], abs_tol=1e-8)
    assert_codegen_matches(
        registry, "check_normality_lilliefors", {"col": "x", "by": None}, df, namespace=_NAMESPACE
    )


def test_check_normality_descriptive_matches_r() -> None:
    df = _data("normal_sample")
    ref = _ref("check_normality_descriptive__normal_sample")
    checks = edacore.assumptions.check_normality_descriptive(df, "x")
    _close(checks[0].statistic, ref["skewness"])
    assert checks[0].status is CheckStatus.PASS
    assert_codegen_matches(
        registry, "check_normality_descriptive", {"col": "x", "by": None}, df, namespace=_NAMESPACE
    )


def test_check_normality_descriptive_flags_skewed_data() -> None:
    df = _data("skewed_sample")
    checks = edacore.assumptions.check_normality_descriptive(df, "x")
    assert checks[0].status in (CheckStatus.BORDERLINE, CheckStatus.FAIL)


def test_qq_correlation_matches_r() -> None:
    df = _data("normal_sample")
    ref = _ref("qq_correlation__normal_sample")
    checks = edacore.assumptions.qq_correlation(df, "x")
    _close(checks[0].statistic, ref["r"])
    assert_codegen_matches(
        registry, "qq_correlation", {"col": "x", "by": None}, df, namespace=_NAMESPACE
    )


# --------------------------------------------------------------------------
# Homoscedasticity
# --------------------------------------------------------------------------


@pytest.mark.parametrize("variant", ["equal_var", "unequal_var"])
def test_check_equal_variance_levene_matches_r(variant: str) -> None:
    df = _data(f"two_groups_{variant}")
    ref = _ref(f"check_equal_variance_levene__{variant}")
    check = edacore.assumptions.check_equal_variance_levene(df, "value", "group")
    _close(check.statistic, ref["statistic"])
    _close(check.p_value, ref["p_value"])
    assert_codegen_matches(
        registry,
        "check_equal_variance_levene",
        {"col": "value", "group": "group"},
        df,
        namespace=_NAMESPACE,
    )


@pytest.mark.parametrize("variant", ["equal_var", "unequal_var"])
def test_check_equal_variance_bartlett_matches_r(variant: str) -> None:
    df = _data(f"two_groups_{variant}")
    ref = _ref(f"check_equal_variance_bartlett__{variant}")
    check = edacore.assumptions.check_equal_variance_bartlett(df, "value", "group")
    _close(check.statistic, ref["statistic"])
    _close(check.p_value, ref["p_value"])
    assert_codegen_matches(
        registry,
        "check_equal_variance_bartlett",
        {"col": "value", "group": "group"},
        df,
        namespace=_NAMESPACE,
    )


@pytest.mark.parametrize("variant", ["equal_var", "unequal_var"])
def test_check_equal_variance_fligner_matches_r(variant: str) -> None:
    df = _data(f"two_groups_{variant}")
    ref = _ref(f"check_equal_variance_fligner__{variant}")
    check = edacore.assumptions.check_equal_variance_fligner(df, "value", "group")
    _close(check.statistic, ref["statistic"])
    _close(check.p_value, ref["p_value"])
    assert_codegen_matches(
        registry,
        "check_equal_variance_fligner",
        {"col": "value", "group": "group"},
        df,
        namespace=_NAMESPACE,
    )


@pytest.mark.parametrize("variant", ["equal_var", "unequal_var"])
def test_check_variance_ratio_matches_r(variant: str) -> None:
    df = _data(f"two_groups_{variant}")
    ref = _ref(f"check_variance_ratio__{variant}")
    check = edacore.assumptions.check_variance_ratio(df, "value", "group")
    _close(check.statistic, ref["ratio"])
    assert_codegen_matches(
        registry,
        "check_variance_ratio",
        {"col": "value", "group": "group"},
        df,
        namespace=_NAMESPACE,
    )


# --------------------------------------------------------------------------
# Sample size / design (deterministic)
# --------------------------------------------------------------------------


def test_check_sample_size() -> None:
    df = _data("two_groups_equal_var")
    check = edacore.assumptions.check_sample_size(df, "value", "group", min_n=10)
    assert check.status is CheckStatus.PASS
    check_fail = edacore.assumptions.check_sample_size(df, "value", "group", min_n=1000)
    assert check_fail.status is CheckStatus.FAIL
    assert_codegen_matches(
        registry,
        "check_sample_size",
        {"col": "value", "group": "group", "min_n": 10},
        df,
        namespace=_NAMESPACE,
    )


def test_check_expected_counts_matches_r() -> None:
    df = _data("contingency_2x2")
    long = df.loc[df.index.repeat(df.Freq)].reset_index(drop=True)
    ref = _ref("check_expected_counts__contingency_2x2")
    check = edacore.assumptions.check_expected_counts(long, "exposure", "outcome")
    _close(check.statistic, ref["min_expected"])
    assert_codegen_matches(
        registry,
        "check_expected_counts",
        {"a": "exposure", "b": "outcome"},
        long,
        namespace=_NAMESPACE,
    )


def test_check_independence_design_untestable_without_repeats() -> None:
    df = pd.DataFrame({"id": range(20), "value": range(20)})
    check = edacore.assumptions.check_independence_design(df, id_col="id")
    assert check.status is CheckStatus.UNTESTABLE
    assert_codegen_matches(
        registry, "check_independence_design", {"id_col": "id"}, df, namespace=_NAMESPACE
    )


def test_check_independence_design_fails_with_repeated_ids() -> None:
    df = pd.DataFrame({"id": [1, 1, 2, 2, 3, 3], "value": range(6)})
    check = edacore.assumptions.check_independence_design(df, id_col="id")
    assert check.status is CheckStatus.FAIL


def test_check_paired_structure() -> None:
    df = _data("paired_before_after")
    check = edacore.assumptions.check_paired_structure(df, "before", "after")
    assert check.status is CheckStatus.PASS
    assert_codegen_matches(
        registry,
        "check_paired_structure",
        {"a": "before", "b": "after", "id_col": None},
        df,
        namespace=_NAMESPACE,
    )


def test_check_paired_structure_fails_on_duplicate_ids() -> None:
    df = pd.DataFrame({"before": [1, 2, 3], "after": [2, 3, 4], "subject": [1, 1, 2]})
    check = edacore.assumptions.check_paired_structure(df, "before", "after", id_col="subject")
    assert check.status is CheckStatus.FAIL


def test_check_measurement_level() -> None:
    df = pd.DataFrame({"income": [10.0, 20.0, 30.0, 40.0, 50.5]})
    check = edacore.assumptions.check_measurement_level(df, "income", SemanticType.CONTINUOUS)
    assert check.status is CheckStatus.PASS
    assert_codegen_matches(
        registry,
        "check_measurement_level",
        {"col": "income", "required": SemanticType.CONTINUOUS},
        df,
        namespace=_NAMESPACE,
    )


def test_check_measurement_level_fails_for_wrong_scale() -> None:
    df = pd.DataFrame({"region": ["A", "B", "A", "C"] * 5})
    check = edacore.assumptions.check_measurement_level(df, "region", SemanticType.CONTINUOUS)
    assert check.status is CheckStatus.FAIL


# --------------------------------------------------------------------------
# Sphericity
# --------------------------------------------------------------------------


def test_check_sphericity_mauchly_statistic_matches_r() -> None:
    # repeated_measures_wide.csv columns are subject,t1,t2,t3,t4 - reshape to long first
    wide = _data("repeated_measures_wide")
    long = wide.melt(id_vars="subject", var_name="within", value_name="dv")
    ref = _ref("check_sphericity_mauchly__repeated_measures")
    check = edacore.assumptions.check_sphericity_mauchly(long, "subject", "within", "dv")
    _close(check.statistic, ref["statistic"], abs_tol=1e-8)
    assert_codegen_matches(
        registry,
        "check_sphericity_mauchly",
        {"subject": "subject", "within": "within", "dv": "dv"},
        long,
        namespace=_NAMESPACE,
    )


# --------------------------------------------------------------------------
# Regression-based
# --------------------------------------------------------------------------


def test_check_linearity_matches_r_reset() -> None:
    """check_linearity implements RESET (Ramsey), so it's compared against
    the fixture's reset_statistic/reset_p_value, not harvey_collier_*."""
    df = _data("regression_xy")
    ref = _ref("check_linearity__regression_xy")
    check = edacore.assumptions.check_linearity(df, "x", "y")
    _close(check.statistic, ref["reset_statistic"], abs_tol=1e-8)
    _close(check.p_value, ref["reset_p_value"], abs_tol=1e-8)
    assert_codegen_matches(
        registry, "check_linearity", {"x": "x", "y": "y"}, df, namespace=_NAMESPACE
    )


def test_check_monotonicity_matches_r() -> None:
    df = _data("monotonic_xy")
    ref = _ref("check_monotonicity__monotonic_xy")
    check = edacore.assumptions.check_monotonicity(df, "x", "y")
    _close(check.statistic, ref["rho"])
    _close(check.p_value, ref["p_value"])
    assert_codegen_matches(
        registry, "check_monotonicity", {"x": "x", "y": "y"}, df, namespace=_NAMESPACE
    )


def test_check_homoscedasticity_bp_matches_r() -> None:
    df = _data("regression_xy")
    ref = _ref("check_homoscedasticity_bp__regression_xy")
    check = edacore.assumptions.check_homoscedasticity_bp(df, "x", "y")
    _close(check.statistic, ref["statistic"])
    _close(check.p_value, ref["p_value"])
    assert_codegen_matches(
        registry, "check_homoscedasticity_bp", {"x": "x", "y": "y"}, df, namespace=_NAMESPACE
    )


def test_check_autocorrelation_dw_matches_r() -> None:
    """R's durbinWatsonTest ran on lm(y ~ x)'s residuals, not on y itself."""
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant

    df = _data("regression_xy")
    ref = _ref("check_autocorrelation_dw__regression_xy")
    fit = OLS(df["y"].to_numpy(dtype=float), add_constant(df["x"].to_numpy(dtype=float))).fit()
    resid_df = df.assign(resid=fit.resid)
    check = edacore.assumptions.check_autocorrelation_dw(resid_df, "resid")
    _close(check.statistic, ref["statistic"])
    assert_codegen_matches(
        registry, "check_autocorrelation_dw", {"resid": "resid"}, resid_df, namespace=_NAMESPACE
    )


def test_check_ljung_box_matches_r() -> None:
    df = _data("autocorrelated_series")
    ref = _ref("check_ljung_box__autocorrelated_series")
    check = edacore.assumptions.check_ljung_box(df, "x", lags=10)
    _close(check.statistic, ref["statistic"])
    _close(check.p_value, ref["p_value"])
    assert_codegen_matches(
        registry, "check_ljung_box", {"col": "x", "lags": 10}, df, namespace=_NAMESPACE
    )


def test_check_multicollinearity_vif_matches_r() -> None:
    df = _data("vif_predictors")
    ref = _ref("check_multicollinearity_vif__vif_predictors")
    check = edacore.assumptions.check_multicollinearity_vif(df, ["x1", "x2", "x3"])
    worst_col = max(ref["vif"], key=lambda c: ref["vif"][c])
    _close(check.statistic, ref["vif"][worst_col])
    assert check.status is CheckStatus.FAIL
    assert_codegen_matches(
        registry,
        "check_multicollinearity_vif",
        {"cols": ["x1", "x2", "x3"]},
        df,
        namespace=_NAMESPACE,
    )


# --------------------------------------------------------------------------
# Distribution shape
# --------------------------------------------------------------------------


def test_check_same_shape_matches_r() -> None:
    df = _data("two_groups_equal_var")
    ref = _ref("check_same_shape__two_groups_equal_var")
    check = edacore.assumptions.check_same_shape(df, "value", "group")
    _close(check.statistic, ref["statistic"])
    _close(check.p_value, ref["p_value"])
    assert_codegen_matches(
        registry, "check_same_shape", {"col": "value", "group": "group"}, df, namespace=_NAMESPACE
    )


def test_check_same_shape_requires_exactly_two_groups() -> None:
    df = _data("three_groups")
    with pytest.raises(ValueError, match="exactly 2 groups"):
        edacore.assumptions.check_same_shape(df, "value", "group")


# --------------------------------------------------------------------------
# False positives (rule 6): normal / equal-variance data must PASS
# --------------------------------------------------------------------------


def test_normal_data_passes_normality_checks_at_expected_rate() -> None:
    """At alpha=0.05, a true normality test should reject (FAIL) a genuinely
    normal sample about 5% of the time — check across many independent
    samples that it's not wildly more than that."""
    rng = np.random.default_rng(123)
    fails = 0
    n_trials = 200
    for _ in range(n_trials):
        x = rng.normal(size=100)
        df = pd.DataFrame({"x": x})
        check = edacore.assumptions.check_normality_shapiro(df, "x")[0]
        if check.status is CheckStatus.FAIL:
            fails += 1
    fail_rate = fails / n_trials
    # Nominal is 5%; allow generous slack for sampling variability (binomial
    # with n=200, p=0.05 has sd ~1.5%, so an 8pp margin is a loose 5+ sigma).
    assert fail_rate <= 0.13, (
        f"shapiro FAILed {fail_rate:.1%} of truly-normal samples, expected ~5%"
    )


def test_equal_variance_data_passes_variance_checks() -> None:
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "value": np.concatenate([rng.normal(50, 10, 100), rng.normal(55, 10, 100)]),
            "group": ["A"] * 100 + ["B"] * 100,
        }
    )
    assert (
        edacore.assumptions.check_equal_variance_levene(df, "value", "group").status
        is CheckStatus.PASS
    )
    assert (
        edacore.assumptions.check_equal_variance_bartlett(df, "value", "group").status
        is CheckStatus.PASS
    )
    assert (
        edacore.assumptions.check_equal_variance_fligner(df, "value", "group").status
        is CheckStatus.PASS
    )
    assert edacore.assumptions.check_variance_ratio(df, "value", "group").status is CheckStatus.PASS


def test_unequal_variance_data_fails_variance_checks() -> None:
    rng = np.random.default_rng(9)
    df = pd.DataFrame(
        {
            "value": np.concatenate([rng.normal(50, 5, 150), rng.normal(50, 40, 150)]),
            "group": ["A"] * 150 + ["B"] * 150,
        }
    )
    assert (
        edacore.assumptions.check_equal_variance_levene(df, "value", "group").status
        is CheckStatus.FAIL
    )
    assert edacore.assumptions.check_variance_ratio(df, "value", "group").status is CheckStatus.FAIL


def test_all_hypothesis_stage_check_functions_are_registered() -> None:
    names = {
        spec.name
        for spec in registry.list(stage="hypothesis")
        if spec.name.startswith("check_") or spec.name == "qq_correlation"
    }
    assert names == {
        "check_normality_shapiro",
        "check_normality_dagostino",
        "check_normality_anderson",
        "check_normality_lilliefors",
        "check_normality_descriptive",
        "qq_correlation",
        "check_equal_variance_levene",
        "check_equal_variance_bartlett",
        "check_equal_variance_fligner",
        "check_variance_ratio",
        "check_sample_size",
        "check_expected_counts",
        "check_independence_design",
        "check_paired_structure",
        "check_sphericity_mauchly",
        "check_linearity",
        "check_monotonicity",
        "check_homoscedasticity_bp",
        "check_autocorrelation_dw",
        "check_ljung_box",
        "check_multicollinearity_vif",
        "check_measurement_level",
        "check_same_shape",
    }
