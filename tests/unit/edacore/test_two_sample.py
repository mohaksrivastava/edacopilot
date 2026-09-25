"""edacore.stattests.two_sample vs R (tests/fixtures/r_reference/).

Three documented, deliberate departures from scipy's/R's shared defaults
(diagnosed against the R fixtures, not guessed -- see two_sample.py's
module docstring and ARCHITECTURE.md Section 18's M3 changelog entry):

- mann_whitney and wilcoxon_signed_rank: p-value method selection
  (exact vs. asymptotic, continuity correction) matches R's wilcox.test
  defaults rather than scipy's own 'auto' selection.
- yuen_trimmed_t: the statistic is kept signed (matching `estimate` and
  the rest of this module's convention), unlike R's WRS2::yuen which
  always reports abs(t). Magnitude matches R exactly; sign is the
  intentional difference.

The two permutation tests (permutation_test_2s, permutation_test_paired)
are checked against R's exact-enumeration fixtures directly, and
additionally cross-checked against a hand-rolled itertools enumeration
independent of both R and scipy.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import edacore
from edacore.registry import registry
from edacore.stattests._shared import (
    MAX_EXACT_PERMUTATIONS,
    n_arrangements_paired,
    n_arrangements_two_sample,
)
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
# student_t / welch_t
# --------------------------------------------------------------------------


def test_student_t_matches_r() -> None:
    df = _data("two_groups_equal_var")
    ref = _ref("student_t__two_groups_equal_var")
    result = edacore.stattests.two_sample.student_t(df, "group", "value")
    _close(result.statistic, ref["statistic"])
    assert result.df == pytest.approx(ref["df"])
    _close(result.p_value, ref["p_value"])
    assert result.estimate is not None
    _close(result.estimate, ref["estimate_diff"])
    assert result.ci is not None
    _close(result.ci[0], ref["ci_low"])
    _close(result.ci[1], ref["ci_high"])
    assert_codegen_matches(
        registry,
        "student_t",
        {"group_col": "group", "value_col": "value", "ci": 0.95, "nan_policy": "omit"},
        df,
        namespace=_NAMESPACE,
    )


def test_welch_t_matches_r() -> None:
    df = _data("two_groups_unequal_var")
    ref = _ref("welch_t__two_groups_unequal_var")
    result = edacore.stattests.two_sample.welch_t(df, "group", "value")
    _close(result.statistic, ref["statistic"])
    assert result.df == pytest.approx(ref["df"])
    _close(result.p_value, ref["p_value"])
    assert result.estimate is not None
    _close(result.estimate, ref["estimate_diff"])
    assert result.ci is not None
    _close(result.ci[0], ref["ci_low"])
    _close(result.ci[1], ref["ci_high"])
    assert_codegen_matches(
        registry,
        "welch_t",
        {"group_col": "group", "value_col": "value", "ci": 0.95, "nan_policy": "omit"},
        df,
        namespace=_NAMESPACE,
    )


def test_student_t_requires_exactly_two_groups() -> None:
    df = pd.DataFrame({"g": ["A", "B", "C"], "v": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="exactly 2 distinct values"):
        edacore.stattests.two_sample.student_t(df, "g", "v")


def test_student_t_nan_policy_omit_drops_and_warns() -> None:
    df = pd.DataFrame({"g": ["A", "A", "A", "B", "B", "B"], "v": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0]})
    result = edacore.stattests.two_sample.student_t(df, "g", "v")
    assert result.n == {"A": 2, "B": 3}
    assert any("missing value" in w for w in result.warnings)


# --------------------------------------------------------------------------
# yuen_trimmed_t
# --------------------------------------------------------------------------


def test_yuen_trimmed_t_matches_r() -> None:
    df = _data("two_groups_unequal_var")
    ref = _ref("yuen_trimmed_t__two_groups_unequal_var")
    result = edacore.stattests.two_sample.yuen_trimmed_t(df, "group", "value", trim=0.2)
    # Magnitude matches R; sign is the documented, intentional difference
    # (R's WRS2::yuen hard-codes abs(dif/se), we keep it signed).
    _close(abs(result.statistic), ref["statistic"])
    assert result.df == pytest.approx(ref["df"])
    _close(result.p_value, ref["p_value"])
    assert result.estimate is not None
    _close(result.estimate, ref["estimate_diff"])
    assert result.ci is not None
    _close(result.ci[0], ref["ci_low"])
    _close(result.ci[1], ref["ci_high"])
    assert_codegen_matches(
        registry,
        "yuen_trimmed_t",
        {"group_col": "group", "value_col": "value", "trim": 0.2, "ci": 0.95, "nan_policy": "omit"},
        df,
        namespace=_NAMESPACE,
    )


def test_yuen_trimmed_t_zero_trim_close_to_student_t_statistic() -> None:
    # trim=0 degenerates to (a slightly different variance estimator than)
    # Welch's t; not identical to student_t's pooled-variance t, but same
    # sign and same order of magnitude.
    df = _data("two_groups_equal_var")
    yuen = edacore.stattests.two_sample.yuen_trimmed_t(df, "group", "value", trim=0.0)
    welch = edacore.stattests.two_sample.welch_t(df, "group", "value")
    assert yuen.statistic == pytest.approx(welch.statistic, rel=1e-6)


# --------------------------------------------------------------------------
# mann_whitney / brunner_munzel / ks_two_sample
# --------------------------------------------------------------------------


def test_mann_whitney_matches_r() -> None:
    df = _data("two_groups_equal_var")
    ref = _ref("mann_whitney__two_groups_equal_var")
    result = edacore.stattests.two_sample.mann_whitney(df, "group", "value")
    _close(result.statistic, ref["statistic"])
    _close(result.p_value, ref["p_value"])
    assert_codegen_matches(
        registry,
        "mann_whitney",
        {"group_col": "group", "value_col": "value", "ci": 0.95, "nan_policy": "omit"},
        df,
        namespace=_NAMESPACE,
    )


def test_brunner_munzel_matches_r() -> None:
    df = _data("two_groups_equal_var")
    ref = _ref("brunner_munzel__two_groups_equal_var")
    result = edacore.stattests.two_sample.brunner_munzel(df, "group", "value")
    _close(result.statistic, ref["statistic"])
    assert result.df == pytest.approx(ref["df"])
    _close(result.p_value, ref["p_value"])
    assert result.estimate is not None
    _close(result.estimate, ref["estimate"])
    assert_codegen_matches(
        registry,
        "brunner_munzel",
        {"group_col": "group", "value_col": "value", "ci": 0.95, "nan_policy": "omit"},
        df,
        namespace=_NAMESPACE,
    )


def test_ks_two_sample_matches_r() -> None:
    df = _data("two_groups_equal_var")
    ref = _ref("ks_two_sample__two_groups_equal_var")
    result = edacore.stattests.two_sample.ks_two_sample(df, "group", "value")
    _close(result.statistic, ref["statistic"])
    _close(result.p_value, ref["p_value"])
    assert_codegen_matches(
        registry,
        "ks_two_sample",
        {"group_col": "group", "value_col": "value", "nan_policy": "omit"},
        df,
        namespace=_NAMESPACE,
    )


def test_ks_two_sample_identical_distributions_gives_high_p() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 200)
    df = pd.DataFrame({"g": ["A"] * 200 + ["B"] * 200, "v": np.concatenate([x, x.copy()])})
    result = edacore.stattests.two_sample.ks_two_sample(df, "g", "v")
    assert result.statistic == 0.0
    assert result.p_value == 1.0


# --------------------------------------------------------------------------
# bootstrap_diff
# --------------------------------------------------------------------------


def test_bootstrap_diff_codegen() -> None:
    df = _data("two_groups_equal_var")
    assert_codegen_matches(
        registry,
        "bootstrap_diff",
        {
            "group_col": "group",
            "value_col": "value",
            "stat": "mean",
            "ci": 0.95,
            "n_boot": 200,
            "random_state": 0,
            "nan_policy": "omit",
        },
        df,
        namespace=_NAMESPACE,
    )


# --------------------------------------------------------------------------
# permutation_test_2s: matches R's exact-enumeration fixture, and
# cross-checked against a hand-rolled enumeration independent of both R
# and scipy
# --------------------------------------------------------------------------


def _hand_rolled_exact_p_two_sample(x: np.ndarray, y: np.ndarray) -> float:
    combined = np.concatenate([x, y])
    n1 = len(x)
    observed = abs(x.mean() - y.mean())
    n_extreme = 0
    n_total = 0
    for idx in itertools.combinations(range(len(combined)), n1):
        idx_set = set(idx)
        a = combined[list(idx_set)]
        b = combined[[i for i in range(len(combined)) if i not in idx_set]]
        n_total += 1
        if abs(a.mean() - b.mean()) >= observed - 1e-12:
            n_extreme += 1
    return n_extreme / n_total


def test_permutation_test_2s_matches_r_exact_fixture() -> None:
    df = _data("permutation_two_sample_small")
    ref = _ref("permutation_test_2s__small_exact")
    result = edacore.stattests.two_sample.permutation_test_2s(df, "group", "value")
    _close(result.statistic, ref["observed_diff"])
    _close(result.p_value, ref["p_value"])
    assert any("exact" in w for w in result.warnings)
    assert_codegen_matches(
        registry,
        "permutation_test_2s",
        {
            "group_col": "group",
            "value_col": "value",
            "n_resamples": 500,
            "ci": 0.95,
            "n_boot": 200,
            "random_state": 0,
            "nan_policy": "omit",
        },
        df,
        namespace=_NAMESPACE,
    )


def test_permutation_test_2s_exact_matches_hand_rolled_enumeration() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(5, 2, 6)
    y = rng.normal(7, 2, 6)
    df = pd.DataFrame({"g": ["A"] * 6 + ["B"] * 6, "v": np.concatenate([x, y])})

    result = edacore.stattests.two_sample.permutation_test_2s(df, "g", "v")
    expected_p = _hand_rolled_exact_p_two_sample(x, y)

    assert any("exact" in w for w in result.warnings)
    assert result.p_value == pytest.approx(expected_p, abs=1e-9)


def test_permutation_test_2s_uses_monte_carlo_above_threshold() -> None:
    # n_arrangements_two_sample(15, 15) = C(30,15) ~ 1.55e8, far above
    # MAX_EXACT_PERMUTATIONS.
    rng = np.random.default_rng(2)
    x = rng.normal(5, 2, 15)
    y = rng.normal(6, 2, 15)
    df = pd.DataFrame({"g": ["A"] * 15 + ["B"] * 15, "v": np.concatenate([x, y])})

    assert n_arrangements_two_sample(15, 15) > MAX_EXACT_PERMUTATIONS
    result = edacore.stattests.two_sample.permutation_test_2s(df, "g", "v", n_resamples=1000)
    assert any("monte_carlo" in w for w in result.warnings)


# --------------------------------------------------------------------------
# paired_t / wilcoxon_signed_rank / sign_test_paired
# --------------------------------------------------------------------------


def test_paired_t_matches_r() -> None:
    df = _data("paired_before_after")
    ref = _ref("paired_t__paired_before_after")
    # R computed t.test(after, before, paired=TRUE): diff = after - before.
    result = edacore.stattests.two_sample.paired_t(df, "after", "before")
    _close(result.statistic, ref["statistic"])
    assert result.df == pytest.approx(ref["df"])
    _close(result.p_value, ref["p_value"])
    assert result.estimate is not None
    _close(result.estimate, ref["estimate"])
    assert result.ci is not None
    _close(result.ci[0], ref["ci_low"])
    _close(result.ci[1], ref["ci_high"])
    assert_codegen_matches(
        registry,
        "paired_t",
        {"a": "after", "b": "before", "ci": 0.95, "nan_policy": "omit"},
        df,
        namespace=_NAMESPACE,
    )


def test_paired_t_drops_pairs_with_either_side_missing() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [1.5, np.nan, 3.5, 4.5]})
    result = edacore.stattests.two_sample.paired_t(df, "a", "b")
    assert result.n == {"a-b": 2}
    assert any("2 pair(s)" in w for w in result.warnings)


def test_wilcoxon_signed_rank_matches_r() -> None:
    df = _data("paired_before_after")
    ref = _ref("wilcoxon_signed_rank__paired_before_after")
    result = edacore.stattests.two_sample.wilcoxon_signed_rank(df, "after", "before")
    _close(result.statistic, ref["statistic"])
    _close(result.p_value, ref["p_value"])
    assert_codegen_matches(
        registry,
        "wilcoxon_signed_rank",
        {"a": "after", "b": "before", "ci": 0.95, "nan_policy": "omit"},
        df,
        namespace=_NAMESPACE,
    )


def test_sign_test_paired_matches_r() -> None:
    df = _data("paired_before_after")
    ref = _ref("sign_test_paired__paired_before_after")
    result = edacore.stattests.two_sample.sign_test_paired(df, "after", "before")
    _close(result.statistic, ref["statistic"])
    _close(result.p_value, ref["p_value"])
    assert result.estimate is not None
    _close(result.estimate, ref["estimate"])
    assert_codegen_matches(
        registry,
        "sign_test_paired",
        {"a": "after", "b": "before", "ci": 0.95, "nan_policy": "omit"},
        df,
        namespace=_NAMESPACE,
    )


def test_sign_test_paired_excludes_zero_differences() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [1.0, 1.0, 5.0, 2.0]})
    result = edacore.stattests.two_sample.sign_test_paired(df, "a", "b")
    assert result.n == {"a-b": 3}
    assert any("ties" in w for w in result.warnings)


# --------------------------------------------------------------------------
# permutation_test_paired: matches R's exact-enumeration fixture, and
# cross-checked against a hand-rolled sign-flip enumeration independent of
# both R and scipy
# --------------------------------------------------------------------------


def _hand_rolled_exact_p_paired(diffs: np.ndarray) -> float:
    n = len(diffs)
    observed = abs(diffs.mean())
    n_extreme = 0
    n_total = 0
    for signs in itertools.product([1, -1], repeat=n):
        flipped = diffs * np.array(signs)
        n_total += 1
        if abs(flipped.mean()) >= observed - 1e-12:
            n_extreme += 1
    return n_extreme / n_total


def test_permutation_test_paired_matches_r_exact_fixture() -> None:
    df = _data("permutation_paired_small")
    ref = _ref("permutation_test_paired__small_exact")
    # R's diff is after - before; permutation_paired_small has before/after
    # columns, so a=after, b=before to get the same sign.
    result = edacore.stattests.two_sample.permutation_test_paired(df, "after", "before")
    _close(result.statistic, ref["observed_diff"])
    _close(result.p_value, ref["p_value"])
    assert any("exact" in w for w in result.warnings)
    assert_codegen_matches(
        registry,
        "permutation_test_paired",
        {
            "a": "after",
            "b": "before",
            "n_resamples": 500,
            "ci": 0.95,
            "n_boot": 200,
            "random_state": 0,
            "nan_policy": "omit",
        },
        df,
        namespace=_NAMESPACE,
    )


def test_permutation_test_paired_exact_matches_hand_rolled_enumeration() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(5, 2, 6)
    b = a + rng.normal(1, 1, 6)
    df = pd.DataFrame({"a": a, "b": b})

    result = edacore.stattests.two_sample.permutation_test_paired(df, "a", "b")
    expected_p = _hand_rolled_exact_p_paired(a - b)

    assert any("exact" in w for w in result.warnings)
    assert result.p_value == pytest.approx(expected_p, abs=1e-9)


def test_permutation_test_paired_uses_monte_carlo_above_threshold() -> None:
    # n_arrangements_paired(20) = 2**20 ~ 1.05e6, above MAX_EXACT_PERMUTATIONS.
    rng = np.random.default_rng(4)
    a = rng.normal(5, 2, 20)
    b = a + rng.normal(0.5, 1, 20)
    df = pd.DataFrame({"a": a, "b": b})

    assert n_arrangements_paired(20) > MAX_EXACT_PERMUTATIONS
    result = edacore.stattests.two_sample.permutation_test_paired(df, "a", "b", n_resamples=1000)
    assert any("monte_carlo" in w for w in result.warnings)


# --------------------------------------------------------------------------
# M3.4: effect size + CI on every two-sample test (Section 16's M3 criterion)
# --------------------------------------------------------------------------


def _assert_effect_ci(result: Any, ref: dict[str, Any], name: str) -> None:
    assert result.effect_size_name == name
    _close(result.effect_size, ref["estimate"])
    assert result.effect_size_ci is not None
    _close(result.effect_size_ci[0], ref["ci_low"])
    _close(result.effect_size_ci[1], ref["ci_high"])


def test_student_t_reports_hedges_g_not_cohens_d() -> None:
    """Section 6.7 specifies Hedges' g. Up to M3.3 this returned Cohen's d
    under the name `cohens_d`; the two differ by the exact J correction."""
    df = _data("two_groups_equal_var")
    ref = _ref("hedges_g__two_groups_equal_var")
    result = edacore.stattests.two_sample.student_t(df, "group", "value")
    _assert_effect_ci(result, ref, "hedges_g")
    d = edacore.effect_sizes.cohens_d(df, "value", "group", ("A", "B"))
    assert result.effect_size != pytest.approx(d["estimate"], abs=1e-9)


def test_welch_t_reports_hedges_g_av_with_ci() -> None:
    """Up to M3.3 welch_t's effect size was labelled hedges_g but held
    Cohen's d(av): the right denominator, no J correction, no CI."""
    df = _data("two_groups_unequal_var")
    ref = _ref("hedges_g_av__two_groups_unequal_var")
    result = edacore.stattests.two_sample.welch_t(df, "group", "value")
    _assert_effect_ci(result, ref, "hedges_g_av")
    x = df.loc[df["group"] == "A", "value"].to_numpy()
    y = df.loc[df["group"] == "B", "value"].to_numpy()
    uncorrected_d_av = (x.mean() - y.mean()) / np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    assert result.effect_size != pytest.approx(uncorrected_d_av, abs=1e-9)


def test_yuen_trimmed_t_effect_size_is_the_trimmed_mean_difference() -> None:
    df = _data("two_groups_unequal_var")
    ref = _ref("yuen_trimmed_t__two_groups_unequal_var")
    result = edacore.stattests.two_sample.yuen_trimmed_t(df, "group", "value")
    assert result.effect_size_name == "trimmed_mean_difference"
    _close(result.effect_size, ref["estimate_diff"])
    assert result.effect_size_ci is not None
    _close(result.effect_size_ci[0], ref["ci_low"])
    _close(result.effect_size_ci[1], ref["ci_high"])
    # Unstandardized: no magnitude label (effect_sizes.UNLABELLED_MEASURES).
    assert result.effect_magnitude is None


def test_mann_whitney_effect_size_ci_matches_r() -> None:
    ref = _ref("rank_biserial__two_groups_equal_var")
    result = edacore.stattests.two_sample.mann_whitney(
        _data("two_groups_equal_var"), "group", "value"
    )
    _assert_effect_ci(result, ref, "rank_biserial")


def test_brunner_munzel_effect_size_ci_matches_r() -> None:
    ref = _ref("brunner_munzel_ci__two_groups_equal_var")
    result = edacore.stattests.two_sample.brunner_munzel(
        _data("two_groups_equal_var"), "group", "value"
    )
    _assert_effect_ci(result, ref, "relative_effect")
    assert result.effect_magnitude is None


def test_paired_t_effect_size_ci_matches_r() -> None:
    ref = _ref("d_z__paired_before_after")
    result = edacore.stattests.two_sample.paired_t(_data("paired_before_after"), "after", "before")
    _assert_effect_ci(result, ref, "d_z")


def test_wilcoxon_signed_rank_effect_size_ci_matches_r_paired_se() -> None:
    """The paired signed-rank SE, not Mann-Whitney's: checked by asserting
    the interval differs from the independent-samples one on the same
    numbers."""
    df = _data("paired_before_after")
    ref = _ref("rank_biserial_paired__paired_before_after")
    result = edacore.stattests.two_sample.wilcoxon_signed_rank(df, "after", "before")
    _assert_effect_ci(result, ref, "rank_biserial")
    long = pd.DataFrame(
        {
            "value": list(df["after"]) + list(df["before"]),
            "group": ["after"] * len(df) + ["before"] * len(df),
        }
    )
    independent = edacore.effect_sizes.rank_biserial(long, "value", "group", ("after", "before"))
    assert result.effect_size_ci[0] != pytest.approx(independent["ci_low"], abs=1e-9)


def test_sign_test_paired_effect_size_ci_matches_r() -> None:
    ref = _ref("sign_test_paired_ci__paired_before_after")
    result = edacore.stattests.two_sample.sign_test_paired(
        _data("paired_before_after"), "after", "before"
    )
    _assert_effect_ci(result, ref, "proportion")
    assert result.ci == result.effect_size_ci


@pytest.mark.parametrize("paired", [False, True])
def test_permutation_tests_report_a_seeded_bootstrap_ci(paired: bool) -> None:
    """A permutation test has no analytic interval, so Section 6.7's
    "bootstrap if no analytic CI" rule applies. Not R-matchable (resampling
    draws), so what is asserted is: an interval exists, it brackets the
    observed difference, it is disclosed as a bootstrap, and the same seed
    reproduces it exactly (rule 7)."""
    if paired:
        df = _data("permutation_paired_small")
        call = lambda seed: edacore.stattests.two_sample.permutation_test_paired(  # noqa: E731
            df, "after", "before", n_boot=300, random_state=seed
        )
    else:
        df = _data("permutation_two_sample_small")
        call = lambda seed: edacore.stattests.two_sample.permutation_test_2s(  # noqa: E731
            df, "group", "value", n_boot=300, random_state=seed
        )

    result = call(0)
    assert result.effect_size_name == "mean_difference"
    assert result.effect_size == result.estimate
    assert result.effect_size_ci is not None
    low, high = result.effect_size_ci
    assert low < result.estimate < high
    assert any("BCa bootstrap" in w for w in result.warnings)
    assert call(0).effect_size_ci == (low, high)
    assert call(7).effect_size_ci != (low, high)
