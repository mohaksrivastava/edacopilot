"""edacore.stattests.categorical vs R. Documented departure: the fisher
2x2 conditional-MLE odds ratio compared at 1e-4 (two root-finders, same
estimand)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import edacore
from edacore.registry import registry
from tests.helpers.codegen_check import assert_codegen_matches
from tests.unit.edacore._r_helpers import data, expanded, ref

_NS = {"edacore": edacore}
cat = edacore.stattests.categorical


@pytest.mark.parametrize(
    "name,a,b",
    [("contingency_2x2", "exposure", "outcome"), ("contingency_3x3", "row", "col")],
)
def test_chi2_independence_matches_r(name: str, a: str, b: str) -> None:
    df, r = expanded(name), ref(f"chi2_independence__{name}")
    res = cat.chi2_independence(df, a, b)
    assert res.statistic == pytest.approx(r["statistic"], rel=1e-9)
    assert res.p_value == pytest.approx(r["p_value"], rel=1e-9)
    assert res.df == r["df"]
    assert res.effect_size_name == "cramers_v"
    assert_codegen_matches(
        registry, "chi2_independence", {"a": a, "b": b, "nan_policy": "omit"}, df, namespace=_NS
    )


def test_fisher_exact_2x2_matches_r_conditional_mle() -> None:
    df, r = expanded("contingency_2x2"), ref("fisher_exact__contingency_2x2")
    res = cat.fisher_exact(df, "exposure", "outcome")
    assert res.p_value == pytest.approx(r["p_value"], rel=1e-9)
    assert res.effect_size_name == "odds_ratio_conditional_mle"
    assert res.effect_size == pytest.approx(r["odds_ratio_conditional_mle"], rel=1e-4)
    assert res.ci is not None
    assert res.ci[0] == pytest.approx(r["ci_low"], rel=1e-4)
    assert res.ci[1] == pytest.approx(r["ci_high"], rel=1e-4)
    # The sample OR (ad/bc = 30*35/(20*15) = 3.5) is NOT what R reports.
    assert res.effect_size != pytest.approx(3.5, rel=1e-3)
    assert_codegen_matches(
        registry,
        "fisher_exact",
        {
            "a": "exposure",
            "b": "outcome",
            "ci_level": 0.95,
            "random_state": 0,
            "nan_policy": "omit",
        },
        df,
        namespace=_NS,
    )


def test_fisher_exact_rxc_matches_r_exact_and_is_deterministic() -> None:
    df, r = expanded("contingency_3x3"), ref("fisher_exact__contingency_3x3")
    res = cat.fisher_exact(df, "row", "col")
    assert res.p_value == pytest.approx(r["p_value"], rel=1e-9)
    assert cat.fisher_exact(df, "row", "col").p_value == res.p_value
    assert any("exact p-value by full enumeration" in w for w in res.warnings)
    assert res.effect_size is None
    assert_codegen_matches(
        registry,
        "fisher_exact",
        {"a": "row", "b": "col", "ci_level": 0.95, "random_state": 0, "nan_policy": "omit"},
        df,
        namespace=_NS,
    )


def test_fisher_exact_rxc_enumeration_agrees_with_scipy_full_permutation() -> None:
    # Independent check: scipy's PermutationMethod(n_resamples=inf) enumerates
    # all 7! pairings of the raw observations (its own exact path).
    table = np.array([[2, 1, 0], [0, 2, 0], [1, 0, 1]])
    ours = cat._fisher_rxc_exact_p(table)
    theirs = stats.fisher_exact(table, method=stats.PermutationMethod(n_resamples=np.inf))
    assert ours == pytest.approx(float(theirs.pvalue), rel=1e-9)


def test_fisher_exact_rxc_monte_carlo_fallback_is_seeded_and_disclosed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cat, "MAX_FISHER_TABLES", 10)
    df = expanded("contingency_3x3")
    first = cat.fisher_exact(df, "row", "col", random_state=1)
    assert cat.fisher_exact(df, "row", "col", random_state=1).p_value == first.p_value
    assert any("Monte Carlo" in w and "random_state=1" in w for w in first.warnings)


def test_g_test_matches_r_default_no_correction() -> None:
    df, r = expanded("contingency_2x2"), ref("g_test__contingency_2x2")
    res = cat.g_test(df, "exposure", "outcome")
    assert res.statistic == pytest.approx(r["statistic"], rel=1e-9)
    assert res.p_value == pytest.approx(r["p_value"], rel=1e-9)
    williams = cat.g_test(df, "exposure", "outcome", correct="williams")
    assert williams.statistic < res.statistic
    with pytest.raises(ValueError, match="correct must be"):
        cat.g_test(df, "exposure", "outcome", correct="yates")
    assert_codegen_matches(
        registry,
        "g_test",
        {"a": "exposure", "b": "outcome", "correct": "none", "nan_policy": "omit"},
        df,
        namespace=_NS,
    )


def test_cochran_armitage_matches_r() -> None:
    df, r = expanded("trend_table"), ref("cochran_armitage_trend__trend_table")
    res = cat.cochran_armitage_trend(df, "outcome", "dose", event="no")
    assert res.statistic == pytest.approx(r["statistic"], rel=1e-9)
    assert res.p_value == pytest.approx(r["p_value"], rel=1e-6)
    assert_codegen_matches(
        registry,
        "cochran_armitage_trend",
        {"binary": "outcome", "ordinal": "dose", "event": "no", "nan_policy": "omit"},
        df,
        namespace=_NS,
    )


def test_two_proportion_z_matches_prop_test() -> None:
    r = ref("two_proportion_z__proportions_two_sample")
    df = pd.DataFrame(
        {
            "outcome": ["yes"] * 34 + ["no"] * 46 + ["yes"] * 21 + ["no"] * 54,
            "group": ["A"] * 80 + ["B"] * 75,
        }
    )
    res = cat.two_proportion_z(df, "outcome", "group", ("A", "B"), event="yes")
    assert res.statistic == pytest.approx(r["statistic"], rel=1e-9)
    assert res.p_value == pytest.approx(r["p_value"], rel=1e-9)
    assert res.ci is not None
    assert res.ci[0] == pytest.approx(r["ci_low"], abs=1e-9)
    assert res.ci[1] == pytest.approx(r["ci_high"], abs=1e-9)
    assert_codegen_matches(
        registry,
        "two_proportion_z",
        {
            "outcome": "outcome",
            "group": "group",
            "groups": ("A", "B"),
            "event": "yes",
            "ci": 0.95,
            "nan_policy": "omit",
        },
        df,
        namespace=_NS,
    )


def test_mcnemar_matches_r_symmetric_table_edge_case() -> None:
    # b01 == b10: R skips the continuity correction -> statistic exactly 0.
    df, r = data("paired_binary"), ref("mcnemar__paired_binary")
    res = cat.mcnemar(df, "before", "after")
    assert res.statistic == r["statistic"] == 0
    assert res.p_value == r["p_value"] == 1
    assert_codegen_matches(
        registry,
        "mcnemar",
        {"a": "before", "b": "after", "ci": 0.95, "nan_policy": "omit"},
        df,
        namespace=_NS,
    )


def test_mcnemar_asymmetric_matches_r_reference_formula() -> None:
    # Asymmetric case, verified against R via the M3 2a pairwise fixture.
    r = ref("cochran_q_posthoc__cochran_q_binary")
    res = cat.mcnemar(data("cochran_q_binary"), "t1", "t2")
    assert res.statistic == pytest.approx(r["statistic"][0], rel=1e-9)
    assert res.p_value == pytest.approx(r["p_unadj"][0], rel=1e-9)


# --------------------------------------------------------------------------
# M3.4: mcnemar's odds-ratio CI and fisher_exact's wired effect_size_ci
# --------------------------------------------------------------------------


def test_mcnemar_odds_ratio_ci_matches_r_exact_conditional() -> None:
    """Asymmetric off-diagonal counts, so an inverted b01/b10 would fail.
    base R's mcnemar.test reports no effect size, so the reference is the
    p/(1-p) transform of binom.test's exact interval (see the M3.4 section
    of scripts/generate_r_fixtures.R)."""
    df = data("cochran_q_binary")
    r = ref("mcnemar_or_ci__cochran_q_binary_t1_t2")
    result = cat.mcnemar(df, "t1", "t2")
    assert result.statistic == pytest.approx(r["statistic"], abs=1e-6)
    assert result.p_value == pytest.approx(r["p_value"], abs=1e-6)
    assert result.effect_size_name == "odds_ratio"
    assert result.effect_size == pytest.approx(r["estimate"], abs=1e-6)
    assert result.effect_size_ci is not None
    assert result.effect_size_ci[0] == pytest.approx(r["ci_low"], abs=1e-6, rel=1e-6)
    assert result.effect_size_ci[1] == pytest.approx(r["ci_high"], abs=1e-6, rel=1e-6)
    # An odds ratio has no Cohen-style magnitude convention.
    assert result.effect_magnitude is None
    assert r["b01"] != r["b10"]


def test_mcnemar_odds_ratio_ci_on_a_symmetric_table() -> None:
    df = data("paired_binary")
    r = ref("mcnemar_or_ci__paired_binary")
    result = cat.mcnemar(df, "before", "after")
    assert result.effect_size == pytest.approx(r["estimate"])
    assert result.effect_size_ci is not None
    assert result.effect_size_ci[0] == pytest.approx(r["ci_low"], rel=1e-9)
    assert result.effect_size_ci[1] == pytest.approx(r["ci_high"], rel=1e-9)


def test_mcnemar_odds_ratio_undefined_without_discordant_pairs() -> None:
    df = pd.DataFrame({"a": [0, 1, 0, 1], "b": [0, 1, 0, 1]})
    result = cat.mcnemar(df, "a", "b")
    assert result.effect_size_ci is None
    assert any("no discordant pairs" in w for w in result.warnings)


def test_fisher_exact_2x2_effect_size_ci_is_the_odds_ratio_interval() -> None:
    df = expanded("contingency_2x2")
    r = ref("fisher_exact__contingency_2x2")
    result = cat.fisher_exact(df, "exposure", "outcome")
    assert result.effect_size_ci == result.ci
    assert result.effect_size_ci is not None
    assert result.effect_size_ci[0] == pytest.approx(r["ci_low"], rel=1e-4)
    assert result.effect_size_ci[1] == pytest.approx(r["ci_high"], rel=1e-4)


def test_fisher_exact_rxc_has_no_effect_size() -> None:
    """Documented exemption (docs/m3_effect_size_audit.md): there is no
    odds ratio beyond a 2x2 table."""
    result = cat.fisher_exact(expanded("contingency_3x3"), "row", "col")
    assert result.effect_size is None
    assert result.effect_size_name is None
    assert result.effect_size_ci is None
