"""edacore.stattests.one_sample vs R (tests/fixtures/r_reference/).

One documented, deliberate departure from scipy's defaults: the Wilcoxon
statistic and p-value in wilcoxon_one_sample match R's wilcox.test
conventions rather than scipy's own defaults -- see
_wilcoxon_signed_rank_r_matched's docstring in one_sample.py and
ARCHITECTURE.md Section 18's M3 changelog entry.
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
# one_sample_t
# --------------------------------------------------------------------------


def test_one_sample_t_matches_r() -> None:
    df = _data("normal_sample")
    ref = _ref("one_sample_t__normal_sample")
    result = edacore.stattests.one_sample.one_sample_t(df, "x", ref["mu0"])
    _close(result.statistic, ref["statistic"])
    assert result.df == pytest.approx(ref["df"])
    _close(result.p_value, ref["p_value"])
    # R's `estimate`/CI are on the raw-mean scale; ours are mean - mu0.
    _close(result.estimate + ref["mu0"], ref["estimate"])
    assert result.ci is not None
    _close(result.ci[0] + ref["mu0"], ref["ci_low"])
    _close(result.ci[1] + ref["mu0"], ref["ci_high"])
    assert_codegen_matches(
        registry,
        "one_sample_t",
        {"col": "x", "mu0": ref["mu0"], "ci": 0.95, "nan_policy": "omit"},
        df,
        namespace=_NAMESPACE,
    )


def test_one_sample_t_ci_contains_estimate() -> None:
    df = _data("normal_sample")
    result = edacore.stattests.one_sample.one_sample_t(df, "x", 50.0)
    assert result.ci is not None and result.estimate is not None
    assert result.ci[0] <= result.estimate <= result.ci[1]


def test_one_sample_t_nan_policy_omit_warns_and_drops() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, np.nan, 4.0, 5.0]})
    result = edacore.stattests.one_sample.one_sample_t(df, "x", 0.0)
    assert result.n == {"x": 4}
    assert any("1 missing value" in w for w in result.warnings)


def test_one_sample_t_nan_policy_raise() -> None:
    df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    with pytest.raises(ValueError, match="missing value"):
        edacore.stattests.one_sample.one_sample_t(df, "x", 0.0, nan_policy="raise")


# --------------------------------------------------------------------------
# wilcoxon_one_sample
# --------------------------------------------------------------------------


def test_wilcoxon_one_sample_matches_r() -> None:
    df = _data("normal_sample")
    ref = _ref("wilcoxon_one_sample__normal_sample")
    result = edacore.stattests.one_sample.wilcoxon_one_sample(df, "x", ref["mu0"])
    _close(result.statistic, ref["statistic"])
    _close(result.p_value, ref["p_value"])
    assert_codegen_matches(
        registry,
        "wilcoxon_one_sample",
        {"col": "x", "mu0": ref["mu0"], "nan_policy": "omit"},
        df,
        namespace=_NAMESPACE,
    )


def test_wilcoxon_one_sample_rank_biserial_sign_matches_direction() -> None:
    df = pd.DataFrame({"x": [10.0, 11.0, 12.0, 13.0, 14.0]})
    result = edacore.stattests.one_sample.wilcoxon_one_sample(df, "x", 0.0)
    assert result.effect_size is not None
    assert result.effect_size > 0  # all values above mu0 -> positive rank-biserial


# --------------------------------------------------------------------------
# sign_test
# --------------------------------------------------------------------------


def test_sign_test_matches_r() -> None:
    df = _data("normal_sample")
    ref = _ref("sign_test__normal_sample")
    result = edacore.stattests.one_sample.sign_test(df, "x", ref["mu0"])
    _close(result.statistic, ref["statistic"])
    _close(result.p_value, ref["p_value"])
    assert result.estimate is not None
    _close(result.estimate, ref["estimate"])
    assert_codegen_matches(
        registry,
        "sign_test",
        {"col": "x", "mu0": ref["mu0"], "nan_policy": "omit"},
        df,
        namespace=_NAMESPACE,
    )


def test_sign_test_excludes_exact_ties() -> None:
    df = pd.DataFrame({"x": [5.0, 5.0, 6.0, 4.0, 7.0]})
    result = edacore.stattests.one_sample.sign_test(df, "x", 5.0)
    assert result.n == {"x": 3}
    assert any("ties" in w for w in result.warnings)


# --------------------------------------------------------------------------
# binomial_test
# --------------------------------------------------------------------------


def test_binomial_test_matches_r() -> None:
    df = _data("binary_sample")
    ref = _ref("binomial_test__binary_sample")
    result = edacore.stattests.one_sample.binomial_test(df, "x", ref["p0"])
    _close(result.statistic, ref["statistic"])
    _close(result.p_value, ref["p_value"])
    assert result.estimate is not None
    _close(result.estimate, ref["estimate"])
    assert result.ci is not None
    _close(result.ci[0], ref["ci_low"])
    _close(result.ci[1], ref["ci_high"])
    assert_codegen_matches(
        registry,
        "binomial_test",
        {"col": "x", "p0": ref["p0"], "ci": 0.95, "nan_policy": "omit"},
        df,
        namespace=_NAMESPACE,
    )


def test_binomial_test_estimate_is_sample_proportion() -> None:
    df = pd.DataFrame({"x": [1, 1, 1, 0, 0]})
    result = edacore.stattests.one_sample.binomial_test(df, "x", 0.5)
    assert result.estimate == pytest.approx(0.6)


# --------------------------------------------------------------------------
# chi2_goodness_of_fit
# --------------------------------------------------------------------------


def test_chi2_goodness_of_fit_matches_r() -> None:
    counts = _data("category_counts")
    df = counts.loc[counts.index.repeat(counts["count"])]
    ref = _ref("chi2_goodness_of_fit__category_counts")
    expected = dict(zip(counts["category"], ref["expected_probs"], strict=True))

    result = edacore.stattests.one_sample.chi2_goodness_of_fit(df, "category", expected)
    _close(result.statistic, ref["statistic"])
    assert result.df == pytest.approx(ref["df"])
    _close(result.p_value, ref["p_value"])
    assert_codegen_matches(
        registry,
        "chi2_goodness_of_fit",
        {"col": "category", "expected": expected},
        df,
        namespace=_NAMESPACE,
    )


def test_chi2_goodness_of_fit_df_is_categories_minus_one() -> None:
    df = pd.DataFrame({"x": ["a"] * 10 + ["b"] * 10 + ["c"] * 10})
    result = edacore.stattests.one_sample.chi2_goodness_of_fit(
        df, "x", {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
    )
    assert result.df == 2.0
    assert result.p_value is not None and result.p_value > 0.9  # exactly matches expected


# --------------------------------------------------------------------------
# bootstrap_one_sample
# --------------------------------------------------------------------------


def test_bootstrap_one_sample_codegen() -> None:
    df = _data("normal_sample")
    assert_codegen_matches(
        registry,
        "bootstrap_one_sample",
        {
            "col": "x",
            "stat": "mean",
            "ci": 0.95,
            "n_boot": 200,
            "random_state": 0,
            "nan_policy": "omit",
        },
        df,
        namespace=_NAMESPACE,
    )


def test_bootstrap_one_sample_reproducible_with_same_random_state() -> None:
    df = _data("normal_sample")
    r1 = edacore.stattests.one_sample.bootstrap_one_sample(df, "x", n_boot=200, random_state=42)
    r2 = edacore.stattests.one_sample.bootstrap_one_sample(df, "x", n_boot=200, random_state=42)
    assert r1.ci == r2.ci


def test_bootstrap_one_sample_rejects_unknown_stat() -> None:
    df = _data("normal_sample")
    with pytest.raises(ValueError, match="stat must be"):
        edacore.stattests.one_sample.bootstrap_one_sample(df, "x", stat="mode")


def test_chi2_goodness_of_fit_rejects_proportions_not_summing_to_one() -> None:
    df = pd.DataFrame({"c": ["a", "b", "b", "c"]})
    with pytest.raises(ValueError, match="must sum to 1"):
        edacore.stattests.one_sample.chi2_goodness_of_fit(df, "c", {"a": 1, "b": 1, "c": 1})
