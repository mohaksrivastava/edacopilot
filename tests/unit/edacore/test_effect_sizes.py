"""edacore.effect_sizes vs R's effectsize package
(tests/fixtures/r_reference/). Every point estimate is checked to 1e-6.

As of M2.1, CIs are checked for every function EXCEPT kendalls_w and
epsilon_squared: those two genuinely bootstrap in R (plain percentile, not
BCa), so a specific interval can't be fixture-matched — their CIs are
validated by simulated coverage instead, in
test_effect_size_ci_coverage.py (marked slow). cohens_w's CI formula is
implemented and internally consistent (see module docstring) but not yet
checked against a fixture: the M2 fixture generation never captured its
CI. See ARCHITECTURE.md Section 18's M2.1 changelog entry.
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


def _assert_close(got: float, want: float, *, abs_tol: float = 1e-6) -> None:
    assert got == pytest.approx(want, abs=abs_tol, rel=1e-6)


# --------------------------------------------------------------------------
# Two independent groups
# --------------------------------------------------------------------------


def test_cohens_d_matches_r() -> None:
    df = _data("two_groups_equal_var")
    ref = _ref("cohens_d__two_groups_equal_var")
    result = edacore.effect_sizes.cohens_d(df, "value", "group", ("A", "B"))
    _assert_close(result["estimate"], ref["estimate"])
    _assert_close(result["ci_low"], ref["ci_low"])
    _assert_close(result["ci_high"], ref["ci_high"])
    assert_codegen_matches(
        registry,
        "cohens_d",
        {"outcome": "value", "group": "group", "groups": ("A", "B"), "ci": 0.95},
        df,
        namespace=_NAMESPACE,
    )


def test_hedges_g_matches_r() -> None:
    df = _data("two_groups_equal_var")
    ref = _ref("hedges_g__two_groups_equal_var")
    result = edacore.effect_sizes.hedges_g(df, "value", "group", ("A", "B"))
    _assert_close(result["estimate"], ref["estimate"])
    _assert_close(result["ci_low"], ref["ci_low"])
    _assert_close(result["ci_high"], ref["ci_high"])
    assert_codegen_matches(
        registry,
        "hedges_g",
        {"outcome": "value", "group": "group", "groups": ("A", "B"), "ci": 0.95},
        df,
        namespace=_NAMESPACE,
    )


def test_glass_delta_matches_r() -> None:
    df = _data("two_groups_unequal_var")
    ref = _ref("glass_delta__two_groups_unequal_var")
    result = edacore.effect_sizes.glass_delta(df, "value", "group", ("A", "B"))
    _assert_close(result["estimate"], ref["estimate"])
    _assert_close(result["ci_low"], ref["ci_low"])
    _assert_close(result["ci_high"], ref["ci_high"])
    assert_codegen_matches(
        registry,
        "glass_delta",
        {"outcome": "value", "group": "group", "groups": ("A", "B"), "ci": 0.95},
        df,
        namespace=_NAMESPACE,
    )


def test_rank_biserial_matches_r() -> None:
    df = _data("two_groups_equal_var")
    ref = _ref("rank_biserial__two_groups_equal_var")
    result = edacore.effect_sizes.rank_biserial(df, "value", "group", ("A", "B"))
    _assert_close(result["estimate"], ref["estimate"])
    _assert_close(result["ci_low"], ref["ci_low"])
    _assert_close(result["ci_high"], ref["ci_high"])
    assert_codegen_matches(
        registry,
        "rank_biserial",
        {"outcome": "value", "group": "group", "groups": ("A", "B"), "ci": 0.95},
        df,
        namespace=_NAMESPACE,
    )


def test_cliffs_delta_matches_r() -> None:
    df = _data("two_groups_equal_var")
    ref = _ref("cliffs_delta__two_groups_equal_var")
    result = edacore.effect_sizes.cliffs_delta(df, "value", "group", ("A", "B"))
    _assert_close(result["estimate"], ref["estimate"])
    _assert_close(result["ci_low"], ref["ci_low"])
    _assert_close(result["ci_high"], ref["ci_high"])
    assert_codegen_matches(
        registry,
        "cliffs_delta",
        {"outcome": "value", "group": "group", "groups": ("A", "B"), "ci": 0.95},
        df,
        namespace=_NAMESPACE,
    )


# --------------------------------------------------------------------------
# Paired
# --------------------------------------------------------------------------


def test_d_z_matches_r() -> None:
    df = _data("paired_before_after")
    ref = _ref("d_z__paired_before_after")
    result = edacore.effect_sizes.d_z(df, "before", "after")
    _assert_close(result["estimate"], ref["estimate"])
    _assert_close(result["ci_low"], ref["ci_low"])
    _assert_close(result["ci_high"], ref["ci_high"])
    assert_codegen_matches(
        registry, "d_z", {"a": "before", "b": "after", "ci": 0.95}, df, namespace=_NAMESPACE
    )


# --------------------------------------------------------------------------
# k independent groups
# --------------------------------------------------------------------------


def test_eta_squared_matches_r() -> None:
    df = _data("three_groups")
    ref = _ref("eta_squared__three_groups")
    result = edacore.effect_sizes.eta_squared(df, "value", "group")
    _assert_close(result["estimate"], ref["estimate"])
    _assert_close(result["ci_low"], ref["ci_low"], abs_tol=1e-5)
    _assert_close(result["ci_high"], ref["ci_high"])
    assert_codegen_matches(
        registry,
        "eta_squared",
        {"outcome": "value", "group": "group", "ci": 0.95},
        df,
        namespace=_NAMESPACE,
    )


def test_partial_eta_squared_equals_eta_squared_for_one_way() -> None:
    df = _data("three_groups")
    ref = _ref("partial_eta_squared__three_groups")
    result = edacore.effect_sizes.partial_eta_squared(df, "value", "group")
    _assert_close(result["estimate"], ref["estimate"])
    _assert_close(result["ci_low"], ref["ci_low"], abs_tol=1e-5)
    _assert_close(result["ci_high"], ref["ci_high"])
    assert_codegen_matches(
        registry,
        "partial_eta_squared",
        {"outcome": "value", "group": "group", "ci": 0.95},
        df,
        namespace=_NAMESPACE,
    )


def test_omega_squared_matches_r() -> None:
    df = _data("three_groups")
    ref = _ref("omega_squared__three_groups")
    result = edacore.effect_sizes.omega_squared(df, "value", "group")
    _assert_close(result["estimate"], ref["estimate"])
    _assert_close(result["ci_low"], ref["ci_low"], abs_tol=1e-5)
    _assert_close(result["ci_high"], ref["ci_high"])
    assert_codegen_matches(
        registry,
        "omega_squared",
        {"outcome": "value", "group": "group", "ci": 0.95},
        df,
        namespace=_NAMESPACE,
    )


def test_epsilon_squared_matches_r_point_estimate() -> None:
    df = _data("three_groups")
    ref = _ref("epsilon_squared__three_groups")
    result = edacore.effect_sizes.epsilon_squared(df, "value", "group")
    _assert_close(result["estimate"], ref["estimate"])
    assert_codegen_matches(
        registry,
        "epsilon_squared",
        {"outcome": "value", "group": "group", "ci": 0.95},
        df,
        namespace=_NAMESPACE,
    )


def test_kendalls_w_matches_r_point_estimate() -> None:
    df = _data("repeated_measures_long")
    ref = _ref("kendalls_w__repeated_measures")
    result = edacore.effect_sizes.kendalls_w(df, "value", "condition", "subject")
    _assert_close(result["estimate"], ref["estimate"], abs_tol=1e-4)
    assert_codegen_matches(
        registry,
        "kendalls_w",
        {"value": "value", "condition": "condition", "subject": "subject", "ci": 0.95},
        df,
        namespace=_NAMESPACE,
    )


def test_kendalls_w_warns_below_50_subjects() -> None:
    # repeated_measures_long.csv has 25 subjects.
    df = _data("repeated_measures_long")
    result = edacore.effect_sizes.kendalls_w(df, "value", "condition", "subject")
    assert result["warnings"]
    assert "25 subjects" in result["warnings"][0]


def test_kendalls_w_no_warning_at_50_or_more_subjects() -> None:
    rng = np.random.default_rng(0)
    n_subj, k = 50, 4
    subject_fx = rng.normal(0, 8, (n_subj, 1))
    values = subject_fx + np.array([0.0, 2.0, 5.0, 4.0]) + rng.normal(0, 3, (n_subj, k))
    df = pd.DataFrame(
        {
            "subject": np.repeat(np.arange(n_subj), k),
            "condition": np.tile(np.arange(k), n_subj),
            "value": values.ravel(),
        }
    )
    result = edacore.effect_sizes.kendalls_w(df, "value", "condition", "subject")
    assert result["warnings"] == []


# --------------------------------------------------------------------------
# Categorical
# --------------------------------------------------------------------------


def test_cramers_v_matches_r() -> None:
    df = _data("contingency_3x3")
    long = df.loc[df.index.repeat(df.Freq)].reset_index(drop=True)
    ref = _ref("cramers_v__contingency_3x3")
    result = edacore.effect_sizes.cramers_v(long, "row", "col")
    _assert_close(result["estimate"], ref["estimate"])
    _assert_close(result["ci_low"], ref["ci_low"], abs_tol=1e-5)
    _assert_close(result["ci_high"], ref["ci_high"])
    assert_codegen_matches(
        registry,
        "cramers_v",
        {"a": "row", "b": "col", "bias_correct": True, "ci": 0.95},
        long,
        namespace=_NAMESPACE,
    )


def test_phi_matches_r() -> None:
    df = _data("contingency_2x2")
    long = df.loc[df.index.repeat(df.Freq)].reset_index(drop=True)
    ref = _ref("phi__contingency_2x2")
    result = edacore.effect_sizes.phi(long, "exposure", "outcome")
    _assert_close(result["estimate"], ref["estimate"])
    _assert_close(result["ci_low"], ref["ci_low"], abs_tol=1e-5)
    _assert_close(result["ci_high"], ref["ci_high"])
    assert_codegen_matches(
        registry, "phi", {"a": "exposure", "b": "outcome", "ci": 0.95}, long, namespace=_NAMESPACE
    )


def test_cohens_w_matches_r_point_estimate() -> None:
    df = _data("contingency_3x3")
    long = df.loc[df.index.repeat(df.Freq)].reset_index(drop=True)
    ref = _ref("cohens_w__contingency_3x3")
    result = edacore.effect_sizes.cohens_w(long, "row", "col")
    _assert_close(result["estimate"], ref["estimate"])
    assert_codegen_matches(
        registry, "cohens_w", {"a": "row", "b": "col", "ci": 0.95}, long, namespace=_NAMESPACE
    )


def _long_2x2() -> pd.DataFrame:
    df = _data("contingency_2x2")
    return df.loc[df.index.repeat(df.Freq)].reset_index(drop=True)


def test_odds_ratio_matches_r() -> None:
    long = _long_2x2()
    ref = _ref("odds_ratio__contingency_2x2")
    result = edacore.effect_sizes.odds_ratio(
        long, "outcome", "exposure", ("exposed", "unexposed"), "event"
    )
    _assert_close(result["estimate"], ref["estimate"])
    _assert_close(result["ci_low"], ref["ci_low"])
    _assert_close(result["ci_high"], ref["ci_high"])
    assert_codegen_matches(
        registry,
        "odds_ratio",
        {
            "outcome": "outcome",
            "group": "exposure",
            "groups": ("exposed", "unexposed"),
            "event": "event",
            "ci": 0.95,
        },
        long,
        namespace=_NAMESPACE,
    )


def test_risk_ratio_matches_r() -> None:
    long = _long_2x2()
    ref = _ref("risk_ratio__contingency_2x2")
    result = edacore.effect_sizes.risk_ratio(
        long, "outcome", "exposure", ("exposed", "unexposed"), "event"
    )
    _assert_close(result["estimate"], ref["estimate"])
    _assert_close(result["ci_low"], ref["ci_low"])
    _assert_close(result["ci_high"], ref["ci_high"])
    assert_codegen_matches(
        registry,
        "risk_ratio",
        {
            "outcome": "outcome",
            "group": "exposure",
            "groups": ("exposed", "unexposed"),
            "event": "event",
            "ci": 0.95,
        },
        long,
        namespace=_NAMESPACE,
    )


def test_risk_difference_matches_r() -> None:
    long = _long_2x2()
    ref = _ref("risk_difference__contingency_2x2")
    result = edacore.effect_sizes.risk_difference(
        long, "outcome", "exposure", ("exposed", "unexposed"), "event"
    )
    _assert_close(result["estimate"], ref["estimate"])
    _assert_close(result["ci_low"], ref["ci_low"])
    _assert_close(result["ci_high"], ref["ci_high"])
    assert_codegen_matches(
        registry,
        "risk_difference",
        {
            "outcome": "outcome",
            "group": "exposure",
            "groups": ("exposed", "unexposed"),
            "event": "event",
            "ci": 0.95,
        },
        long,
        namespace=_NAMESPACE,
    )


def test_cohens_h_matches_r() -> None:
    long = _long_2x2()
    ref = _ref("cohens_h__contingency_2x2")
    result = edacore.effect_sizes.cohens_h(
        long, "outcome", "exposure", ("exposed", "unexposed"), "event"
    )
    _assert_close(result["estimate"], ref["estimate"])
    _assert_close(result["ci_low"], ref["ci_low"])
    _assert_close(result["ci_high"], ref["ci_high"])
    assert_codegen_matches(
        registry,
        "cohens_h",
        {
            "outcome": "outcome",
            "group": "exposure",
            "groups": ("exposed", "unexposed"),
            "event": "event",
            "ci": 0.95,
        },
        long,
        namespace=_NAMESPACE,
    )


# --------------------------------------------------------------------------
# magnitude_label / bootstrap_effect_ci (not registered, not R-fixture-tested)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "measure", "expected"),
    [
        (0.05, "cohens_d", "negligible"),
        (0.3, "cohens_d", "small"),
        (0.6, "cohens_d", "medium"),
        (0.9, "cohens_d", "large"),
        (-0.9, "cohens_d", "large"),
    ],
)
def test_magnitude_label(value: float, measure: str, expected: str) -> None:
    assert edacore.effect_sizes.magnitude_label(value, measure) == expected


def test_magnitude_label_unknown_measure_raises() -> None:
    with pytest.raises(KeyError):
        edacore.effect_sizes.magnitude_label(0.5, "not_a_real_measure")


def test_bootstrap_effect_ci_reproducible_with_same_seed() -> None:
    df = _data("two_groups_equal_var")

    def _mean_diff(sample_df: pd.DataFrame) -> float:
        return float(
            sample_df.loc[sample_df.group == "A", "value"].mean()
            - sample_df.loc[sample_df.group == "B", "value"].mean()
        )

    ci1 = edacore.effect_sizes.bootstrap_effect_ci(_mean_diff, df, n_boot=200, random_state=0)
    ci2 = edacore.effect_sizes.bootstrap_effect_ci(_mean_diff, df, n_boot=200, random_state=0)
    assert ci1 == ci2
