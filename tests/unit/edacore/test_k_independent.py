"""edacore.stattests.k_independent vs R (tests/fixtures/r_reference/).

Two functions needed a formula ported from R source rather than a
scipy/statsmodels builtin -- see k_independent.py's module docstring for
the full writeup (ARCHITECTURE.md Section 18, M3 part 2a entry):
alexander_govern (no scipy/statsmodels equivalent at all) and
welch_anova's omega-squared (effectsize's F-based approximation, not the
classical SS-based formula edacore.effect_sizes.omega_squared uses).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import edacore
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


def _close(got: float | None, want: float, *, abs_tol: float = 1e-6) -> None:
    assert got is not None
    assert got == pytest.approx(want, abs=abs_tol, rel=1e-6)


# --------------------------------------------------------------------------
# one_way_anova
# --------------------------------------------------------------------------


def test_one_way_anova_matches_r() -> None:
    df = _data("three_groups")
    ref = _ref("one_way_anova__three_groups")
    result = edacore.stattests.k_independent.one_way_anova(df, "value", "group")
    _close(result.statistic, ref["statistic"])
    assert result.df is not None and result.df[0] == pytest.approx(ref["df1"])
    assert result.df[1] == pytest.approx(ref["df2"])
    _close(result.p_value, ref["p_value"])
    assert_codegen_matches(
        registry,
        "one_way_anova",
        {"outcome": "value", "group": "group", "ci": 0.95, "nan_policy": "omit"},
        df,
        namespace=_NAMESPACE,
    )


def test_one_way_anova_reports_both_effect_sizes() -> None:
    df = _data("three_groups")
    result = edacore.stattests.k_independent.one_way_anova(df, "value", "group")
    assert result.effect_size_name == "eta_squared"
    assert any("omega_squared" in note for note in result.validity_notes)


def test_one_way_anova_requires_at_least_two_groups() -> None:
    df = pd.DataFrame({"value": [1.0, 2.0, 3.0], "group": ["A", "A", "A"]})
    with pytest.raises(ValueError, match="at least 2 distinct values"):
        edacore.stattests.k_independent.one_way_anova(df, "value", "group")


# --------------------------------------------------------------------------
# welch_anova
# --------------------------------------------------------------------------


def test_welch_anova_matches_r() -> None:
    df = _data("k_groups_unequal_var")
    ref = _ref("welch_anova__k_groups_unequal_var")
    result = edacore.stattests.k_independent.welch_anova(df, "value", "group")
    _close(result.statistic, ref["statistic"])
    assert result.df is not None and result.df[0] == pytest.approx(ref["df1"])
    assert result.df[1] == pytest.approx(ref["df2"], rel=1e-6)
    _close(result.p_value, ref["p_value"])
    _close(result.effect_size, ref["omega_squared"])
    assert_codegen_matches(
        registry,
        "welch_anova",
        {"outcome": "value", "group": "group", "ci": 0.95, "nan_policy": "omit"},
        df,
        namespace=_NAMESPACE,
    )


# --------------------------------------------------------------------------
# alexander_govern
# --------------------------------------------------------------------------


def test_alexander_govern_matches_r() -> None:
    df = _data("k_groups_unequal_var")
    ref = _ref("alexander_govern__k_groups_unequal_var")
    result = edacore.stattests.k_independent.alexander_govern(df, "value", "group")
    _close(result.statistic, ref["statistic"])
    assert result.df == pytest.approx(ref["df"])
    _close(result.p_value, ref["p_value"])
    assert_codegen_matches(
        registry,
        "alexander_govern",
        {"outcome": "value", "group": "group", "nan_policy": "omit"},
        df,
        namespace=_NAMESPACE,
    )


def test_alexander_govern_has_no_effect_size() -> None:
    df = _data("k_groups_unequal_var")
    result = edacore.stattests.k_independent.alexander_govern(df, "value", "group")
    assert result.effect_size is None


# --------------------------------------------------------------------------
# kruskal_wallis
# --------------------------------------------------------------------------


def test_kruskal_wallis_matches_r() -> None:
    df = _data("three_groups")
    ref = _ref("kruskal_wallis__three_groups")
    result = edacore.stattests.k_independent.kruskal_wallis(df, "value", "group")
    _close(result.statistic, ref["statistic"])
    assert result.df == pytest.approx(ref["df"])
    _close(result.p_value, ref["p_value"])
    assert_codegen_matches(
        registry,
        "kruskal_wallis",
        {"outcome": "value", "group": "group", "ci": 0.95, "nan_policy": "omit"},
        df,
        namespace=_NAMESPACE,
    )


# --------------------------------------------------------------------------
# permutation_anova
# --------------------------------------------------------------------------


def test_permutation_anova_matches_r_exact_fixture() -> None:
    df = _data("permutation_anova_small")
    ref = _ref("permutation_anova__small_exact")
    result = edacore.stattests.k_independent.permutation_anova(df, "value", "group")
    _close(result.statistic, ref["observed_f"])
    _close(result.p_value, ref["p_value"])
    assert any("exact" in w for w in result.warnings)
    assert_codegen_matches(
        registry,
        "permutation_anova",
        {
            "outcome": "value",
            "group": "group",
            "n_resamples": 500,
            "random_state": 0,
            "nan_policy": "omit",
        },
        df,
        namespace=_NAMESPACE,
    )


def test_permutation_anova_uses_monte_carlo_for_large_groups() -> None:
    df = _data("three_groups")  # 3 groups x 30 -> combinatorially huge
    result = edacore.stattests.k_independent.permutation_anova(
        df, "value", "group", n_resamples=500
    )
    assert any("monte_carlo" in w for w in result.warnings)


# --------------------------------------------------------------------------
# nan_policy / degenerate input
# --------------------------------------------------------------------------


def test_one_way_anova_nan_policy_omit_drops_and_warns() -> None:
    df = pd.DataFrame(
        {"value": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0], "group": ["A", "A", "A", "B", "B", "B"]}
    )
    result = edacore.stattests.k_independent.one_way_anova(df, "value", "group")
    assert result.n == {"A": 2, "B": 3}
    assert any("missing" in w for w in result.warnings)


def test_one_way_anova_nan_policy_raise() -> None:
    df = pd.DataFrame({"value": [1.0, np.nan, 3.0], "group": ["A", "A", "B"]})
    with pytest.raises(ValueError, match="missing"):
        edacore.stattests.k_independent.one_way_anova(df, "value", "group", nan_policy="raise")


# --------------------------------------------------------------------------
# M3.4: welch_anova's omega-squared CI (Section 16's M3 criterion)
# --------------------------------------------------------------------------


def test_welch_anova_omega_squared_ci_matches_r() -> None:
    """On a dataset with a real effect: k_groups_unequal_var has Welch
    F < 1, making omega^2 = 0 and its interval [0, 1], which would pass
    against almost anything."""
    df = _data("k_groups_unequal_var_effect")
    ref = _ref("welch_anova__k_groups_unequal_var_effect")
    result = edacore.stattests.k_independent.welch_anova(df, "value", "group")
    _close(result.statistic, ref["statistic"])
    _close(result.p_value, ref["p_value"])
    assert result.effect_size_name == "omega_squared"
    _close(result.effect_size, ref["omega_squared"])
    assert result.effect_size_ci is not None
    _close(result.effect_size_ci[0], ref["ci_low"])
    _close(result.effect_size_ci[1], ref["ci_high"])
    assert result.effect_size_ci[0] > 0


def test_welch_anova_omega_squared_ci_is_zero_bounded_on_a_null_dataset() -> None:
    """The other side of the same inversion: F < 1 gives omega^2 = 0 and a
    lower bound pinned at 0, not a negative number."""
    df = _data("k_groups_unequal_var")
    ref = _ref("omega_squared_welch__k_groups_unequal_var")
    result = edacore.stattests.k_independent.welch_anova(df, "value", "group")
    _close(result.effect_size, ref["estimate"])
    assert result.effect_size_ci == (ref["ci_low"], ref["ci_high"]) == (0.0, 1.0)
