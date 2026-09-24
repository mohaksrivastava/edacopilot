"""edacore.stattests.correlation vs R; mutual_information and
distance_correlation's p-value are checked by simulation (no R reference)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import edacore
from edacore.registry import registry
from tests.helpers.codegen_check import assert_codegen_matches
from tests.unit.edacore._r_helpers import data, ref

_NS = {"edacore": edacore}
cor = edacore.stattests.correlation


def test_pearson_matches_r_with_fisher_z_ci() -> None:
    df, r = data("correlation_small_no_ties"), ref("pearson__correlation_small_no_ties")
    res = cor.pearson(df, "x", "y")
    assert res.estimate == pytest.approx(r["estimate"], rel=1e-9)
    assert res.statistic == pytest.approx(r["statistic"], rel=1e-9)  # t, not r
    assert res.p_value == pytest.approx(r["p_value"], rel=1e-9)
    assert res.ci is not None
    assert res.ci[0] == pytest.approx(r["ci_low"], rel=1e-9)
    assert res.ci[1] == pytest.approx(r["ci_high"], rel=1e-9)
    assert_codegen_matches(
        registry,
        "pearson",
        {"x": "x", "y": "y", "ci": 0.95, "nan_policy": "omit"},
        df,
        namespace=_NS,
    )


def test_spearman_exact_matches_r_small_n_no_ties() -> None:
    df, r = data("correlation_small_no_ties"), ref("spearman__correlation_small_no_ties")
    res = cor.spearman(df, "x", "y")
    assert res.estimate == pytest.approx(r["estimate"], rel=1e-9)
    assert res.p_value == pytest.approx(r["p_value"], rel=1e-9)
    assert any("exact" in w for w in res.warnings)
    assert_codegen_matches(
        registry, "spearman", {"x": "x", "y": "y", "nan_policy": "omit"}, df, namespace=_NS
    )


@pytest.mark.parametrize("n", [9, 20, 49, 100])
def test_spearman_matches_r_prho_across_regimes(n: int) -> None:
    # n=9: C_pRho's exact enumeration; n>9: AS 89 Edgeworth series.
    df, r = data(f"rank_corr_n{n}"), ref(f"spearman__rank_corr_n{n}")
    res = cor.spearman(df, "x", "y")
    assert res.estimate == pytest.approx(r["estimate"], rel=1e-9)
    assert res.statistic == pytest.approx(r["statistic_S"], rel=1e-9)
    assert res.p_value == pytest.approx(r["p_value"], rel=1e-9)
    assert any(("exact" if n <= 9 else "AS 89") in w for w in res.warnings)


@pytest.mark.parametrize("n", [9, 20, 49, 100])
def test_kendall_matches_r_exact_below_50_normal_above(n: int) -> None:
    df, r = data(f"rank_corr_n{n}"), ref(f"kendall_tau__rank_corr_n{n}")
    res = cor.kendall_tau(df, "x", "y")
    assert res.statistic_name == r["statistic_name"]  # T (exact) or z
    assert res.estimate == pytest.approx(r["estimate"], rel=1e-9)
    assert res.statistic == pytest.approx(r["statistic"], rel=1e-9)
    assert res.p_value == pytest.approx(r["p_value"], rel=1e-9)


def test_spearman_ties_use_t_approximation_and_say_so() -> None:
    df = data("correlation_with_ties")
    res = cor.spearman(df, "x", "y")
    assert res.p_value == pytest.approx(stats.spearmanr(df["x"], df["y"]).pvalue, rel=1e-12)
    assert any("ties present" in w for w in res.warnings)


def test_prho_exact_tails_are_complementary() -> None:
    # P[S >= s] + P[S < s] = 1 over the enumerated null, every attainable s.
    for n in (3, 5, 9):
        for s in range(0, n * (n * n - 1) // 3 + 3, 2):
            upper = cor._prho(n, s, lower_tail=False)
            lower = cor._prho(n, s, lower_tail=True)
            assert upper + lower == pytest.approx(1.0)


def test_kendall_tau_b_with_ties_matches_r() -> None:
    df, r = data("correlation_with_ties"), ref("kendall_tau__correlation_with_ties")
    res = cor.kendall_tau(df, "x", "y")
    assert res.estimate == pytest.approx(r["estimate"], rel=1e-9)
    assert res.statistic == pytest.approx(r["statistic"], rel=1e-9)  # z
    assert res.p_value == pytest.approx(r["p_value"], rel=1e-9)
    assert_codegen_matches(
        registry, "kendall_tau", {"x": "x", "y": "y", "nan_policy": "omit"}, df, namespace=_NS
    )


def test_kendall_exact_small_n_no_ties_is_exact() -> None:
    df = pd.DataFrame({"x": [1.0, 2, 3, 4, 5, 6], "y": [2.0, 1, 4, 3, 6, 5]})
    res = cor.kendall_tau(df, "x", "y")
    assert any("exact" in w for w in res.warnings)
    # 3 discordant pairs of 15: permutations of 6 with <=3 inversions =
    # 1+5+14+29 = 49, two-sided p = 2*49/720 (Mahonian numbers).
    assert res.p_value == pytest.approx(98 / 720)


def test_point_biserial_equals_pearson_on_zero_one_coding() -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"b": rng.integers(0, 2, 40), "x": rng.normal(size=40)})
    pb = cor.point_biserial(df, "b", "x")
    pe = cor.pearson(df.assign(b=df["b"].astype(float)), "b", "x")
    assert pb.estimate == pytest.approx(pe.estimate)
    assert pb.p_value == pytest.approx(pe.p_value)
    with pytest.raises(ValueError, match="exactly 2 distinct"):
        cor.point_biserial(df.assign(b=[0, 1, 2, 3] * 10), "b", "x")
    assert_codegen_matches(
        registry,
        "point_biserial",
        {"binary": "b", "x": "x", "ci": 0.95, "nan_policy": "omit"},
        df,
        namespace=_NS,
    )


def test_partial_correlation_matches_ppcor() -> None:
    df = data("partial_correlation_data")
    r = ref("partial_correlation__partial_correlation_data")
    res = cor.partial_correlation(df, "x", "y", ["z"])
    assert res.estimate == pytest.approx(r["estimate"], rel=1e-9)
    assert res.statistic == pytest.approx(r["statistic"], rel=1e-9)
    assert res.p_value == pytest.approx(r["p_value"], rel=1e-9)
    assert res.df == r["df_used"]
    assert_codegen_matches(
        registry,
        "partial_correlation",
        {"x": "x", "y": "y", "covars": ["z"], "nan_policy": "omit"},
        df,
        namespace=_NS,
    )


def test_distance_correlation_statistic_matches_energy() -> None:
    df = data("partial_correlation_data")
    r = ref("distance_correlation__partial_correlation_data")
    res = cor.distance_correlation(df, "x", "y", n_resamples=99)
    assert res.estimate == pytest.approx(r["estimate"], rel=1e-9)
    assert_codegen_matches(
        registry,
        "distance_correlation",
        {"x": "x", "y": "y", "n_resamples": 99, "random_state": 0, "nan_policy": "omit"},
        df,
        namespace=_NS,
    )


def test_distance_correlation_pvalue_behaviour_by_simulation() -> None:
    # Permutation p can't match R's (RNG); check its behaviour instead:
    # tiny under a strong NONLINEAR dependence, rarely significant under
    # independence.
    rng = np.random.default_rng(3)
    x = rng.normal(size=80)
    dep = cor.distance_correlation(pd.DataFrame({"x": x, "y": x**2}), "x", "y", n_resamples=299)
    assert dep.p_value is not None and dep.p_value < 0.01
    ps = []
    for i in range(40):
        indep = pd.DataFrame({"x": rng.normal(size=30), "y": rng.normal(size=30)})
        res = cor.distance_correlation(indep, "x", "y", n_resamples=99, random_state=i)
        assert res.p_value is not None
        ps.append(res.p_value)
    assert np.mean([p < 0.05 for p in ps]) < 0.2  # nominal 5%; loose bound for 40 trials


@pytest.mark.parametrize("rho", [0.3, 0.7])
def test_mutual_information_matches_analytic_bivariate_normal(rho: float) -> None:
    # KSG at n=1500, k=5: measured |bias| <= ~0.015 nats across rho in
    # [0, 0.9] at n=3000; tolerance 0.05 nats leaves headroom for n=1500
    # sampling variance while still rejecting a wrong estimator (true MI
    # at these rho is 0.047-0.34 nats above independence).
    rng = np.random.default_rng(0)
    xy = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], 1500)
    res = cor.mutual_information(pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1]}), "x", "y")
    assert res.estimate == pytest.approx(-0.5 * np.log(1 - rho**2), abs=0.05)


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.7, 0.95])
@pytest.mark.parametrize("x_scale", [1.0, 1000.0])
@pytest.mark.parametrize("k", [3, 5])
def test_mutual_information_matches_sklearn(rho: float, x_scale: float, k: int) -> None:
    # Same algorithm (KSG 1, max-norm) and preprocessing (unit-variance
    # scaling); only the 1e-10 jitter draws differ, which cannot reorder
    # neighbours in continuous data. x_scale=1000 guards scale invariance.
    from sklearn.feature_selection import mutual_info_regression

    rng = np.random.default_rng(6)
    xy = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], 800)
    df = pd.DataFrame({"x": xy[:, 0] * x_scale, "y": xy[:, 1]})
    ours = cor.mutual_information(df, "x", "y", k=k).estimate
    theirs = mutual_info_regression(
        df[["x"]].to_numpy(), df["y"].to_numpy(), n_neighbors=k, random_state=0
    )[0]
    assert ours == pytest.approx(theirs, rel=1e-9, abs=1e-12)


def test_mutual_information_ties_are_finite_and_disclosed() -> None:
    rng = np.random.default_rng(7)
    x = rng.integers(1, 6, 300).astype(float)  # Likert-like: heavy ties
    df = pd.DataFrame({"x": x, "y": x + rng.integers(0, 3, 300)})
    res = cor.mutual_information(df, "x", "y")
    assert res.estimate is not None and np.isfinite(res.estimate) and res.estimate > 0.3
    assert any("tied values" in w for w in res.warnings)
    with pytest.raises(ValueError, match="constant"):
        cor.mutual_information(df.assign(x=1.0), "x", "y")


def test_mutual_information_independent_is_near_zero_and_codegen() -> None:
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"x": rng.normal(size=500), "y": rng.normal(size=500)})
    assert cor.mutual_information(df, "x", "y").estimate == pytest.approx(0, abs=0.05)
    assert_codegen_matches(
        registry,
        "mutual_information",
        {"x": "x", "y": "y", "k": 5, "random_state": 0, "nan_policy": "omit"},
        df,
        namespace=_NS,
    )


def test_correlation_matrix_matches_pairwise_and_adjusts_within_family() -> None:
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"a": rng.normal(size=60)})
    df["b"] = df["a"] + rng.normal(size=60)
    df["c"] = rng.normal(size=60)
    res = cor.correlation_matrix(df, ["a", "b", "c"], method="pearson")
    assert len(res) == 3
    single = cor.pearson(df, "a", "b")
    assert res[0].estimate == single.estimate and res[0].p_value == single.p_value
    assert all(r.p_adjusted is not None and r.p_adjusted >= (r.p_value or 0) for r in res)
    with pytest.raises(ValueError, match="method must be"):
        cor.correlation_matrix(df, ["a", "b"], method="bogus")
    assert_codegen_matches(
        registry,
        "correlation_matrix",
        {"cols": ["a", "b", "c"], "method": "pearson", "adjust": "holm", "nan_policy": "omit"},
        df,
        namespace=_NS,
    )
