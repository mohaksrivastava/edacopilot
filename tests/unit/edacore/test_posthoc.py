"""edacore.stattests.posthoc vs R (tests/fixtures/r_reference/).

Package/parameter choices, each confirmed to reproduce R's default output
exactly -- see posthoc.py's module docstring for the full writeup
(ARCHITECTURE.md Section 18, M3 part 2a entry). Comparisons are looked up
by an order-independent key since this project's pair ordering doesn't
always match R's row/column ordering, but the underlying numbers do.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import edacore
from edacore.contracts import PairwiseComparison, PostHocResult
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


def _by_pair(result: PostHocResult) -> dict[str, PairwiseComparison]:
    """Indexes each comparison under both group orders and both the
    " - " (FSA/PMCMRplus) and "-" (stats::TukeyHSD) label spacing R uses
    across the different fixtures."""
    out: dict[str, PairwiseComparison] = {}
    for c in result.comparisons:
        for a, b in [(c.group_a, c.group_b), (c.group_b, c.group_a)]:
            out[f"{a} - {b}"] = c
            out[f"{a}-{b}"] = c
    return out


def _binary_long(name: str) -> pd.DataFrame:
    wide = _data(name)
    return (
        wide.reset_index()
        .rename(columns={"index": "subject"})
        .melt(id_vars="subject", var_name="condition", value_name="value")
    )


# --------------------------------------------------------------------------
# tukey_hsd
# --------------------------------------------------------------------------


def test_tukey_hsd_matches_r() -> None:
    df = _data("three_groups")
    ref = _ref("tukey_hsd__three_groups")
    result = edacore.stattests.posthoc.tukey_hsd(df, "value", "group")
    pairs = _by_pair(result)
    for comparison, diff, lwr, upr, p_adj in zip(
        ref["comparisons"], ref["diff"], ref["lwr"], ref["upr"], ref["p_adj"], strict=True
    ):
        c = pairs[comparison]
        assert c.estimate == pytest.approx(diff, abs=1e-6)
        assert c.ci is not None
        assert c.ci[0] == pytest.approx(lwr, abs=1e-5)
        assert c.ci[1] == pytest.approx(upr, abs=1e-5)
        assert c.p_adjusted == pytest.approx(p_adj, abs=1e-5)
    assert_codegen_matches(
        registry,
        "tukey_hsd",
        {"outcome": "value", "group": "group", "nan_policy": "omit"},
        df,
        namespace=_NAMESPACE,
    )


def test_tukey_hsd_estimate_direction_matches_group_labels() -> None:
    df = _data("three_groups")
    result = edacore.stattests.posthoc.tukey_hsd(df, "value", "group")
    means = df.groupby("group")["value"].mean()
    for c in result.comparisons:
        assert c.estimate == pytest.approx(means[c.group_a] - means[c.group_b], abs=1e-6)


# --------------------------------------------------------------------------
# games_howell
# --------------------------------------------------------------------------


def test_games_howell_matches_r() -> None:
    df = _data("k_groups_unequal_var")
    ref = _ref("games_howell__k_groups_unequal_var")
    result = edacore.stattests.posthoc.games_howell(df, "value", "group")
    pairs = _by_pair(result)
    for comparison, stat, p in zip(
        ref["comparisons"], ref["statistic"], ref["p_value"], strict=True
    ):
        c = pairs[comparison]
        assert c.statistic == pytest.approx(stat, abs=1e-4)
        assert c.p_value == pytest.approx(p, abs=1e-4)
    assert_codegen_matches(
        registry,
        "games_howell",
        {"outcome": "value", "group": "group", "nan_policy": "omit"},
        df,
        namespace=_NAMESPACE,
    )


# --------------------------------------------------------------------------
# dunn_test
# --------------------------------------------------------------------------


def test_dunn_test_matches_r_default_holm() -> None:
    df = _data("three_groups")
    ref = _ref("dunn_test__three_groups")
    result = edacore.stattests.posthoc.dunn_test(df, "value", "group")
    assert result.p_adjust_method == "holm"
    pairs = _by_pair(result)
    for comparison, stat, p_unadj, p_adj in zip(
        ref["comparisons"], ref["statistic"], ref["p_unadj"], ref["p_adj"], strict=True
    ):
        c = pairs[comparison]
        assert c.statistic == pytest.approx(stat, abs=1e-6)
        assert c.p_value == pytest.approx(p_unadj, abs=1e-6)
        assert c.p_adjusted == pytest.approx(p_adj, abs=1e-6)
    assert_codegen_matches(
        registry,
        "dunn_test",
        {"outcome": "value", "group": "group", "p_adjust": "holm", "nan_policy": "omit"},
        df,
        namespace=_NAMESPACE,
    )


# --------------------------------------------------------------------------
# nemenyi_friedman / conover_friedman
# --------------------------------------------------------------------------


def test_nemenyi_friedman_matches_r() -> None:
    df = _data("repeated_measures_long")
    ref = _ref("nemenyi_friedman__repeated_measures")
    result = edacore.stattests.posthoc.nemenyi_friedman(df, "value", "subject", "condition")
    pairs = _by_pair(result)
    for comparison, p in zip(ref["comparisons"], ref["p_value"], strict=True):
        assert pairs[comparison].p_value == pytest.approx(p, abs=1e-4)
    assert_codegen_matches(
        registry,
        "nemenyi_friedman",
        {"dv": "value", "subject": "subject", "within": "condition", "nan_policy": "omit"},
        df,
        namespace=_NAMESPACE,
    )


def test_conover_friedman_matches_r_single_step() -> None:
    df = _data("repeated_measures_long")
    ref = _ref("conover_friedman__repeated_measures")
    result = edacore.stattests.posthoc.conover_friedman(df, "value", "subject", "condition")
    assert result.p_adjust_method == "single-step"
    pairs = _by_pair(result)
    for comparison, p in zip(ref["comparisons"], ref["p_value"], strict=True):
        assert pairs[comparison].p_value == pytest.approx(p, abs=1e-4)
    assert_codegen_matches(
        registry,
        "conover_friedman",
        {"dv": "value", "subject": "subject", "within": "condition", "nan_policy": "omit"},
        df,
        namespace=_NAMESPACE,
    )


# --------------------------------------------------------------------------
# paired_posthoc / mcnemar_posthoc
# --------------------------------------------------------------------------


def test_paired_posthoc_matches_r() -> None:
    df = _data("repeated_measures_long")
    ref = _ref("repeated_measures_posthoc__repeated_measures")
    result = edacore.stattests.posthoc.paired_posthoc(df, "value", "subject", "condition")
    assert result.p_adjust_method == "holm"
    pairs = _by_pair(result)
    for comparison, p_adj in zip(ref["comparisons"], ref["p_adj"], strict=True):
        assert pairs[comparison].p_adjusted == pytest.approx(p_adj, abs=1e-5)
    assert_codegen_matches(
        registry,
        "paired_posthoc",
        {
            "dv": "value",
            "subject": "subject",
            "within": "condition",
            "p_adjust": "holm",
            "nan_policy": "omit",
        },
        df,
        namespace=_NAMESPACE,
    )


def test_mcnemar_posthoc_matches_r() -> None:
    df = _binary_long("cochran_q_binary")
    ref = _ref("cochran_q_posthoc__cochran_q_binary")
    result = edacore.stattests.posthoc.mcnemar_posthoc(df, "value", "subject", "condition")
    pairs = _by_pair(result)
    for comparison, stat, p_unadj, p_adj in zip(
        ref["comparisons"], ref["statistic"], ref["p_unadj"], ref["p_adj"], strict=True
    ):
        c = pairs[comparison]
        assert c.statistic == pytest.approx(stat, abs=1e-6)
        assert c.p_value == pytest.approx(p_unadj, abs=1e-6)
        assert c.p_adjusted == pytest.approx(p_adj, abs=1e-6)
    assert_codegen_matches(
        registry,
        "mcnemar_posthoc",
        {
            "dv": "value",
            "subject": "subject",
            "within": "condition",
            "p_adjust": "holm",
            "nan_policy": "omit",
        },
        df,
        namespace=_NAMESPACE,
    )


# --------------------------------------------------------------------------
# permutation_posthoc (no R fixture: built on already-R-verified
# permutation_test_2s, not a separate R-ported formula)
# --------------------------------------------------------------------------


def test_permutation_posthoc_p_adjusted_are_holm_monotone() -> None:
    df = _data("three_groups")
    result = edacore.stattests.posthoc.permutation_posthoc(df, "value", "group", n_resamples=500)
    assert len(result.comparisons) == 3
    for c in result.comparisons:
        assert c.p_adjusted is not None and c.p_adjusted >= c.p_value


def test_permutation_posthoc_codegen() -> None:
    df = _data("three_groups")
    assert_codegen_matches(
        registry,
        "permutation_posthoc",
        {
            "outcome": "value",
            "group": "group",
            "n_resamples": 500,
            "random_state": 0,
            "p_adjust": "holm",
            "nan_policy": "omit",
        },
        df,
        namespace=_NAMESPACE,
    )
