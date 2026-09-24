"""edacore.stattests.categorical vs R. Documented departures: fisher_exact
r x c p-value (scipy's exact convention differs from R's), and the fisher
2x2 conditional-MLE odds ratio compared at 1e-4 (two root-finders, same
estimand)."""

from __future__ import annotations

import pandas as pd
import pytest
import scipy

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
        {"a": "exposure", "b": "outcome", "nan_policy": "omit"},
        df,
        namespace=_NS,
    )


_SCIPY = tuple(int(x) for x in scipy.__version__.split(".")[:2])


@pytest.mark.skipif(_SCIPY < (1, 15), reason="r x c fisher_exact needs scipy>=1.15")
def test_fisher_exact_rxc_is_exact_but_documented_to_differ_from_r() -> None:
    df, r = expanded("contingency_3x3"), ref("fisher_exact__contingency_3x3")
    res = cat.fisher_exact(df, "row", "col")
    assert res.p_value is not None and 0 < res.p_value <= 1
    assert res.p_value != pytest.approx(r["p_value"], rel=1e-3)  # documented divergence
    assert any("r x c" in w for w in res.warnings)
    assert res.effect_size is None


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
        registry, "mcnemar", {"a": "before", "b": "after", "nan_policy": "omit"}, df, namespace=_NS
    )


def test_mcnemar_asymmetric_matches_r_reference_formula() -> None:
    # Asymmetric case, verified against R via the M3 2a pairwise fixture.
    r = ref("cochran_q_posthoc__cochran_q_binary")
    res = cat.mcnemar(data("cochran_q_binary"), "t1", "t2")
    assert res.statistic == pytest.approx(r["statistic"][0], rel=1e-9)
    assert res.p_value == pytest.approx(r["p_unadj"][0], rel=1e-9)


@pytest.mark.skipif(_SCIPY >= (1, 15), reason="only applies below scipy 1.15")
def test_fisher_exact_rxc_raises_clear_error_on_old_scipy() -> None:
    with pytest.raises(ValueError, match="scipy>=1.15"):
        cat.fisher_exact(expanded("contingency_3x3"), "row", "col")
