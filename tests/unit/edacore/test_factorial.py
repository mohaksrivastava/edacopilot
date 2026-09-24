"""edacore.stattests.factorial vs R. Uses an UNBALANCED fixture: balanced
data hides the Type III sum-to-zero-contrast requirement."""

from __future__ import annotations

import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

import edacore
from edacore.registry import registry
from tests.helpers.codegen_check import assert_codegen_matches
from tests.unit.edacore._r_helpers import data, ref

_NS = {"edacore": edacore}
_FACTORS = ["factor_a", "factor_b"]
fac = edacore.stattests.factorial


def test_two_way_anova_type3_matches_r() -> None:
    df = data("factorial_unbalanced")
    r = ref("two_way_anova__factorial_unbalanced")
    results = fac.two_way_anova(df, "value", _FACTORS, typ=3)
    assert len(results) == 3
    for res, stat, p in zip(results, r["statistic"], r["p_value"], strict=True):
        assert res.statistic == pytest.approx(stat, rel=1e-6)
        assert res.p_value == pytest.approx(p, rel=1e-6)
        assert res.df == (1.0, float(r["df_residual"]))
    assert any("unbalanced" in n for n in results[0].validity_notes)
    assert_codegen_matches(
        registry,
        "two_way_anova",
        {"outcome": "value", "factors": _FACTORS, "typ": 3, "nan_policy": "omit"},
        df,
        namespace=_NS,
    )


def test_type3_differs_from_naive_treatment_contrasts() -> None:
    """The trap itself: default treatment contrasts give a different
    (wrong) Type III answer for an unbalanced design."""
    df = data("factorial_unbalanced")
    fit = smf.ols("value ~ C(factor_a) * C(factor_b)", data=df).fit()
    naive = sm.stats.anova_lm(fit, typ=3)
    right = fac.two_way_anova(df, "value", _FACTORS, typ=3)
    assert naive.loc["C(factor_b)", "PR(>F)"] != pytest.approx(right[1].p_value, rel=1e-3)


def test_two_way_anova_type2_runs_and_codegen() -> None:
    df = data("factorial_unbalanced")
    results = fac.two_way_anova(df, "value", _FACTORS)
    assert len(results) == 3 and "Type II SS" in results[0].estimand
    assert_codegen_matches(
        registry,
        "two_way_anova",
        {"outcome": "value", "factors": _FACTORS, "typ": 2, "nan_policy": "omit"},
        df,
        namespace=_NS,
    )


def test_two_way_anova_rejects_bad_args() -> None:
    df = data("factorial_unbalanced")
    with pytest.raises(ValueError, match="exactly 2 factors"):
        fac.two_way_anova(df, "value", ["factor_a"])
    with pytest.raises(ValueError, match="typ must be"):
        fac.two_way_anova(df, "value", _FACTORS, typ=1)


def test_art_anova_matches_artool() -> None:
    df = data("factorial_unbalanced")
    r = ref("art_anova__factorial_unbalanced")
    results = fac.aligned_rank_transform_anova(df, "value", _FACTORS)
    for res, stat, p in zip(results, r["statistic"], r["p_value"], strict=True):
        assert res.statistic == pytest.approx(stat, rel=1e-6)
        assert res.p_value == pytest.approx(p, rel=1e-6)
    assert_codegen_matches(
        registry,
        "aligned_rank_transform_anova",
        {"outcome": "value", "factors": _FACTORS, "nan_policy": "omit"},
        df,
        namespace=_NS,
    )
