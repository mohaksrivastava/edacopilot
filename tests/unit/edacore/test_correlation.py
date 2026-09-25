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
        registry,
        "spearman",
        {"x": "x", "y": "y", "ci": 0.95, "nan_policy": "omit"},
        df,
        namespace=_NS,
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
        registry,
        "kendall_tau",
        {"x": "x", "y": "y", "ci": 0.95, "nan_policy": "omit"},
        df,
        namespace=_NS,
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
        {"x": "x", "y": "y", "covars": ["z"], "ci": 0.95, "nan_policy": "omit"},
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


# --------------------------------------------------------------------------
# M3.4: correlation effect sizes and their CIs (Section 16's M3 criterion)
# --------------------------------------------------------------------------


def test_pearson_and_point_biserial_report_r_as_the_effect_size() -> None:
    """r IS the effect size for a correlation, so effect_size mirrors
    estimate and shares its Fisher-z interval -- not a second, different
    interval."""
    df = data("correlation_small_no_ties")
    result = cor.pearson(df, "x", "y")
    assert result.effect_size_name == "pearson_r"
    assert result.effect_size == result.estimate
    assert result.effect_size_ci == result.ci

    binary_df = pd.DataFrame({"g": [0, 0, 1, 1, 0, 1, 1, 0], "v": list(df["y"])})
    pb = cor.point_biserial(binary_df, "g", "v")
    assert pb.effect_size_name == "point_biserial_r"
    assert pb.effect_size == pb.estimate
    assert pb.effect_size_ci == pb.ci


@pytest.mark.parametrize(
    ("fixture", "dataset"),
    [
        ("spearman_ci__correlation_small_no_ties", "correlation_small_no_ties"),
        ("spearman_ci__rank_corr_n100", "rank_corr_n100"),
    ],
)
def test_spearman_ci_matches_desctools(fixture: str, dataset: str) -> None:
    """R's cor.test gives no CI for rho; DescTools::SpearmanRho is the
    reference (Fisher z with SE = 1/sqrt(n - 3)). Two sample sizes, because
    the SE is the only place n enters."""
    r = ref(fixture)
    result = cor.spearman(data(dataset), "x", "y")
    assert result.effect_size_name == "spearman_rho"
    assert result.effect_size == pytest.approx(r["estimate"], abs=1e-9)
    assert result.effect_size_ci is not None
    assert result.effect_size_ci[0] == pytest.approx(r["ci_low"], abs=1e-9)
    assert result.effect_size_ci[1] == pytest.approx(r["ci_high"], abs=1e-9)
    assert result.effect_size_ci == result.ci


@pytest.mark.parametrize(
    ("fixture", "dataset"),
    [
        ("kendall_tau_ci__correlation_with_ties", "correlation_with_ties"),
        ("kendall_tau_ci__rank_corr_n20", "rank_corr_n20"),
    ],
)
def test_kendall_tau_ci_matches_desctools(fixture: str, dataset: str) -> None:
    """DescTools::KendallTauB's delta-method interval, which (unlike a
    Fisher-z one) accounts for ties. Both a tied table and continuous
    tie-free data, since the latter drives the contingency table to n x n
    and exercises the cumulative-sum path in `_con_dis_pairs`."""
    r = ref(fixture)
    result = cor.kendall_tau(data(dataset), "x", "y")
    assert result.effect_size_name == "kendall_tau_b"
    assert result.effect_size == pytest.approx(r["estimate"], abs=1e-9)
    assert result.effect_size_ci is not None
    assert result.effect_size_ci[0] == pytest.approx(r["ci_low"], abs=1e-9)
    assert result.effect_size_ci[1] == pytest.approx(r["ci_high"], abs=1e-9)


def test_kendall_tau_ci_is_not_a_fisher_z_interval() -> None:
    """Guards the specific shortcut this could have been implemented as:
    with ties, the delta-method SE and a Fisher-z SE disagree materially."""
    r = ref("kendall_tau_ci__correlation_with_ties")
    df = data("correlation_with_ties")
    n = len(df)
    tau = r["estimate"]
    fisher_low = np.tanh(np.arctanh(tau) - 1.959963984540054 / np.sqrt(n - 3))
    assert fisher_low != pytest.approx(r["ci_low"], abs=1e-3)


def test_partial_correlation_ci_matches_fisher_z_with_n_minus_k_minus_3() -> None:
    """No installed R package reports a CI for a partial correlation, so
    the fixture evaluates the same published Fisher-z formula in R (see the
    M3.4 section of scripts/generate_r_fixtures.R and ARCHITECTURE.md
    Section 18). What this test does establish independently: the df used
    is n - k - 3, not n - 3."""
    r = ref("partial_correlation_ci__partial_correlation_data")
    df = data("partial_correlation_data")
    result = cor.partial_correlation(df, "x", "y", ["z"])
    assert result.effect_size_name == "partial_r"
    assert result.effect_size == pytest.approx(r["estimate"], abs=1e-9)
    assert result.effect_size_ci is not None
    assert result.effect_size_ci[0] == pytest.approx(r["ci_low"], abs=1e-9)
    assert result.effect_size_ci[1] == pytest.approx(r["ci_high"], abs=1e-9)

    rng = np.random.default_rng(0)
    two_covars = cor.partial_correlation(
        df.assign(z2=rng.normal(size=len(df))), "x", "y", ["z", "z2"]
    )
    assert two_covars.effect_size_ci is not None
    assert two_covars.effect_size_ci[0] != pytest.approx(result.effect_size_ci[0], abs=1e-9)


def test_distance_correlation_and_mutual_information_stay_exempt() -> None:
    """Documented exemptions (docs/m3_effect_size_audit.md): dCor has no
    analytic CI and a bootstrap of it is badly biased near 0; the KSG
    mutual-information estimator has no standard CI at all."""
    df = data("partial_correlation_data")
    for result in (
        cor.distance_correlation(df, "x", "y", n_resamples=50),
        cor.mutual_information(df, "x", "y"),
    ):
        assert result.estimate is not None
        assert result.effect_size_ci is None


def test_kendall_tau_b_from_the_table_agrees_with_scipys_rank_formula() -> None:
    """Two independent routes to tau-b: scipy's rank-based formula (which
    drives the p-value) and the concordant/discordant counts read off the
    contingency table (which drives the CI). If they disagree, the table
    construction in `_kendall_concordance_excess` is wrong."""
    for dataset in ("correlation_with_ties", "rank_corr_n20", "rank_corr_n49"):
        df = data(dataset)
        x, y = df["x"].to_numpy(), df["y"].to_numpy()
        from_table = cor._kendall_tau_b_from_table(x, y)
        assert from_table == pytest.approx(stats.kendalltau(x, y, variant="b").statistic, abs=1e-12)


def test_kendall_tau_ci_falls_back_to_fisher_z_on_a_table_too_large_to_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The delta-method table is (distinct x) x (distinct y), so tie-free
    continuous data makes it n x n. Past the cap the CI degrades to a
    Fisher-z interval and says so, rather than trying to allocate it."""
    df = data("rank_corr_n20")
    full = cor.kendall_tau(df, "x", "y")
    monkeypatch.setattr(cor, "MAX_KENDALL_TABLE_CELLS", 10)
    degraded = cor.kendall_tau(df, "x", "y")

    assert degraded.estimate == full.estimate
    assert degraded.p_value == full.p_value
    assert degraded.effect_size_ci != full.effect_size_ci
    assert any("Fisher-z approximation" in w for w in degraded.warnings)
    assert not any("Fisher-z approximation" in w for w in full.warnings)

    n = len(df)
    expected = np.tanh(
        np.arctanh(full.estimate) + np.array([-1, 1]) * stats.norm.ppf(0.975) / np.sqrt(n - 3)
    )
    assert degraded.effect_size_ci is not None
    assert degraded.effect_size_ci[0] == pytest.approx(expected[0])
    assert degraded.effect_size_ci[1] == pytest.approx(expected[1])
