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
from scipy import stats

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


def test_check_sphericity_mauchly_matches_r_exactly() -> None:
    """This is an exact port of R's mauchly.test.SSD (statistic and
    p-value, including its two-term Box correction), not an approximation."""
    # repeated_measures_wide.csv columns are subject,t1,t2,t3,t4 - reshape to long first
    wide = _data("repeated_measures_wide")
    long = wide.melt(id_vars="subject", var_name="within", value_name="dv")
    ref = _ref("check_sphericity_mauchly__repeated_measures")
    check = edacore.assumptions.check_sphericity_mauchly(long, "subject", "within", "dv")
    _close(check.statistic, ref["statistic"], abs_tol=1e-8)
    _close(check.p_value, ref["p_value"], abs_tol=1e-10)
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


def test_check_monotonicity_passes_on_monotonic_data() -> None:
    """The regression this exists for: until M4 this check took Spearman's
    own p-value as its verdict, so a *strong* monotone association (tiny p)
    was reported as FAILING monotonicity. The M2 test compared only rho and
    p against R and never asserted a status, which is how the inversion
    survived -- so this asserts the status first.

    `monotonic_xy` is y = log(x) + noise: monotone, not linear, which is
    exactly the shape Spearman is for. The R fixture's Spearman rho is
    still checked, against the function that actually reports it.
    """
    df = _data("monotonic_xy")
    check = edacore.assumptions.check_monotonicity(df, "x", "y")
    assert check.status is CheckStatus.PASS
    assert check.statistic == pytest.approx(0.0, abs=0.05)

    ref = _ref("check_monotonicity__monotonic_xy")
    spearman = edacore.stattests.correlation.spearman(df, "x", "y")
    _close(spearman.estimate, ref["rho"])

    assert_codegen_matches(
        registry, "check_monotonicity", {"x": "x", "y": "y"}, df, namespace=_NAMESPACE
    )


def test_check_monotonicity_fails_on_a_u_shape() -> None:
    """A U shape is a strong, perfectly ordered relationship that Spearman's
    rho reports as roughly zero. That is the case the check has to catch."""
    x = np.linspace(-10, 10, 120)
    df = pd.DataFrame({"x": x, "y": x**2})
    check = edacore.assumptions.check_monotonicity(df, "x", "y")
    assert check.status is CheckStatus.FAIL
    assert abs(stats.spearmanr(df["x"], df["y"]).statistic) < 0.1


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


def test_check_same_shape_handles_more_than_two_groups() -> None:
    """Kruskal-Wallis declares `same_shape` too, so the check cannot be
    two-group-only. With k > 2 it compares every pair and Holm-adjusts, so
    the reported p-value is the worst pair's ADJUSTED one -- never the raw
    minimum, which over C(k,2) pairs would reject on chance alone and hand
    the user a caveat that means nothing."""
    df = _data("three_groups")
    check = edacore.assumptions.check_same_shape(df, "value", "group")
    assert check.status is CheckStatus.PASS
    assert "3 pairs" in check.threshold and "Holm" in check.threshold


def test_check_same_shape_needs_at_least_two_groups() -> None:
    df = _data("three_groups")
    with pytest.raises(ValueError, match="at least 2 groups"):
        edacore.assumptions.check_same_shape(df[df["group"] == "A"], "value", "group")


def test_check_same_shape_detects_a_differently_shaped_group() -> None:
    rng = np.random.default_rng(77)
    df = pd.DataFrame(
        {
            "value": np.concatenate(
                [rng.normal(0, 1, 200), rng.normal(0, 1, 200), rng.exponential(1, 200)]
            ),
            "group": ["a"] * 200 + ["b"] * 200 + ["c"] * 200,
        }
    )
    assert edacore.assumptions.check_same_shape(df, "value", "group").status is CheckStatus.FAIL


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
        "check_balanced_design",
        "check_design_crossing",
        "check_expected_counts_gof",
        "check_influential_outliers",
        "check_proportion_counts",
        "check_symmetry",
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


# --------------------------------------------------------------------------
# M4 additions (Section 6.6 gap-closing for the eligibility engine)
# --------------------------------------------------------------------------


def test_check_symmetry_matches_r_skewness() -> None:
    df = _data("paired_before_after").assign(d=lambda f: f["after"] - f["before"])
    ref = _ref("check_symmetry__paired_before_after")
    check = edacore.assumptions.check_symmetry(df, "d")[0]
    _close(check.statistic, ref["skewness"])
    assert check.status is CheckStatus.PASS

    skewed = _data("skewed_sample")
    ref_skewed = _ref("check_symmetry__skewed_sample")
    skewed_check = edacore.assumptions.check_symmetry(skewed, "x")[0]
    _close(skewed_check.statistic, ref_skewed["skewness"])
    assert skewed_check.status is CheckStatus.FAIL

    assert_codegen_matches(
        registry, "check_symmetry", {"col": "d", "by": None}, df, namespace=_NAMESPACE
    )


def test_check_influential_outliers_matches_r_cooks_distance() -> None:
    """Both branches: a normal regression, and the same data with one
    planted high-leverage point."""
    df = _data("regression_xy")
    ref = _ref("check_influential_outliers__regression_xy")
    check = edacore.assumptions.check_influential_outliers(df, "x", "y")
    _close(check.statistic, ref["max_cooks_d"])
    assert check.status is CheckStatus.PASS

    influential = _data("regression_xy_influential")
    ref_influential = _ref("check_influential_outliers__regression_xy_influential")
    influential_check = edacore.assumptions.check_influential_outliers(influential, "x", "y")
    assert influential_check.statistic == pytest.approx(ref_influential["max_cooks_d"], rel=1e-9)
    assert influential_check.status is CheckStatus.FAIL

    assert_codegen_matches(
        registry, "check_influential_outliers", {"x": "x", "y": "y"}, df, namespace=_NAMESPACE
    )


def test_check_influential_outliers_does_not_use_the_4_over_n_screen_as_a_verdict() -> None:
    """4/n flags individual points to LOOK at; at any real n some point
    exceeds it by chance. Using it as the dataset's verdict would make
    almost every correlation a caveat and train the user to ignore it."""
    df = _data("regression_xy")
    check = edacore.assumptions.check_influential_outliers(df, "x", "y")
    ref = _ref("check_influential_outliers__regression_xy")
    assert check.statistic > ref["threshold_4_over_n"]
    assert check.status is CheckStatus.PASS


def test_check_balanced_design_matches_the_r_cell_counts() -> None:
    df = _data("repeated_measures_long")
    ref = _ref("check_balanced_design__repeated_measures")
    check = edacore.assumptions.check_balanced_design(df, "subject", "condition")
    assert check.statistic == ref["min_cell"] == ref["max_cell"] == 1
    assert check.status is CheckStatus.PASS

    assert (
        edacore.assumptions.check_balanced_design(
            df.drop(df.index[-1]), "subject", "condition"
        ).status
        is CheckStatus.FAIL
    )

    assert_codegen_matches(
        registry,
        "check_balanced_design",
        {"subject": "subject", "within": "condition"},
        df,
        namespace=_NAMESPACE,
    )


def test_check_proportion_counts_matches_r() -> None:
    ref = _ref("check_proportion_counts__proportions_two_sample")
    df = pd.DataFrame(
        {
            "g": ["a"] * ref["n1"] + ["b"] * ref["n2"],
            "o": [1] * ref["x1"]
            + [0] * (ref["n1"] - ref["x1"])
            + [1] * ref["x2"]
            + [0] * (ref["n2"] - ref["x2"]),
        }
    )
    check = edacore.assumptions.check_proportion_counts(df, "o", "g", 1)
    assert check.statistic == ref["min_successes_or_failures"]
    assert check.status is CheckStatus.PASS

    rare = pd.DataFrame(
        {"g": ["a"] * 50 + ["b"] * 50, "o": [1] * 3 + [0] * 47 + [1] * 5 + [0] * 45}
    )
    assert edacore.assumptions.check_proportion_counts(rare, "o", "g", 1).status is CheckStatus.FAIL

    assert_codegen_matches(
        registry,
        "check_proportion_counts",
        {"outcome": "o", "group": "g", "event": 1},
        df,
        namespace=_NAMESPACE,
    )


def test_check_expected_counts_gof_matches_r() -> None:
    counts = _data("category_counts")
    ref = _ref("check_expected_counts_gof__category_counts")
    df = pd.DataFrame({"category": counts["category"].repeat(counts["count"]).to_numpy()})
    expected = dict(zip("ABCD", ref["expected_probs"], strict=True))
    check = edacore.assumptions.check_expected_counts_gof(df, "category", expected)
    _close(check.statistic, ref["min_expected"])
    assert check.status is CheckStatus.PASS

    sparse = pd.DataFrame({"category": list("ABCD") + ["A"] * 4})
    assert (
        edacore.assumptions.check_expected_counts_gof(sparse, "category", expected).status
        is CheckStatus.FAIL
    )

    assert_codegen_matches(
        registry,
        "check_expected_counts_gof",
        {"col": "category", "expected": expected},
        df,
        namespace=_NAMESPACE,
    )


def test_check_design_crossing_distinguishes_pairing_from_clustering() -> None:
    """The distinction Section 7.1 rests on: an id under several groups
    means the design is related; an id repeated inside one group means the
    rows are clustered, which is a different problem with a different fix."""
    independent = pd.DataFrame({"id": range(6), "g": ["a"] * 3 + ["b"] * 3})
    crossing = pd.DataFrame({"id": list(range(3)) * 2, "g": ["a"] * 3 + ["b"] * 3})
    clustered = pd.DataFrame({"id": [1, 1, 2, 3, 4, 5], "g": ["a"] * 3 + ["b"] * 3})

    assert (
        edacore.assumptions.check_design_crossing(independent, "g", "id").status is CheckStatus.PASS
    )

    crossed = edacore.assumptions.check_design_crossing(crossing, "g", "id")
    assert crossed.status is CheckStatus.FAIL
    assert crossed.statistic == 3
    assert "more than one" in crossed.threshold

    clustered_check = edacore.assumptions.check_design_crossing(clustered, "g", "id")
    assert clustered_check.status is CheckStatus.FAIL
    assert clustered_check.statistic == 0
    assert "within a single group" in clustered_check.threshold

    assert_codegen_matches(
        registry,
        "check_design_crossing",
        {"group": "g", "id_col": "id"},
        crossing,
        namespace=_NAMESPACE,
    )
