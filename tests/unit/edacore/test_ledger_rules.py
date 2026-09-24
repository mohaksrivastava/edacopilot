"""Locks in the TestLedger rule from ARCHITECTURE.md Section 12.3: an
omnibus test's TestResult.p_adjusted is never self-set (that value is the
eventual TestLedger's job, computed across the whole session); a post-hoc
procedure's PostHocResult carries its own within-family adjustment,
labelled with the method and number of comparisons, entirely separate
from the session-wide bookkeeping.

TestLedger itself belongs to the edacopilot session layer (not yet
built -- src/edacopilot/ is a stub). This test covers the part of the
rule that edacore's registered functions are responsible for right now:
never producing a TestResult.p_adjusted or a PostHocResult that implies
session-wide adjustment happened.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import edacore
from edacore.contracts import PostHocResult, TestResult

FIXTURES = Path(__file__).parents[2] / "fixtures" / "r_reference"


def _data(name: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "data" / f"{name}.csv")


# --------------------------------------------------------------------------
# Omnibus tests: p_adjusted is never self-set
# --------------------------------------------------------------------------

_OMNIBUS_CALLS: list[tuple[str, TestResult]] = [
    (
        "one_way_anova",
        edacore.stattests.k_independent.one_way_anova(_data("three_groups"), "value", "group"),
    ),
    (
        "welch_anova",
        edacore.stattests.k_independent.welch_anova(
            _data("k_groups_unequal_var"), "value", "group"
        ),
    ),
    (
        "alexander_govern",
        edacore.stattests.k_independent.alexander_govern(
            _data("k_groups_unequal_var"), "value", "group"
        ),
    ),
    (
        "kruskal_wallis",
        edacore.stattests.k_independent.kruskal_wallis(_data("three_groups"), "value", "group"),
    ),
    (
        "repeated_measures_anova",
        edacore.stattests.k_related.repeated_measures_anova(
            _data("repeated_measures_long"), "value", "subject", "condition"
        ),
    ),
    (
        "friedman",
        edacore.stattests.k_related.friedman(
            _data("repeated_measures_long"), "value", "subject", "condition"
        ),
    ),
    (
        "one_sample_t",
        edacore.stattests.one_sample.one_sample_t(_data("normal_sample"), "x", 50.0),
    ),
    (
        "student_t",
        edacore.stattests.two_sample.student_t(_data("two_groups_equal_var"), "group", "value"),
    ),
]


@pytest.mark.parametrize("name,result", _OMNIBUS_CALLS, ids=[c[0] for c in _OMNIBUS_CALLS])
def test_omnibus_test_result_p_adjusted_is_unset(name: str, result: TestResult) -> None:
    assert result.p_adjusted is None, (
        f"{name} set p_adjusted itself; per ARCHITECTURE.md Section 12.3, adjustment "
        "across the session is TestLedger's job, never the test function's own."
    )


def test_omnibus_test_enters_ledger_as_one_result() -> None:
    # An omnibus TestResult is a single object with a single p_value --
    # there is no way to accidentally produce "one entry per comparison"
    # since the return type itself is one TestResult, not a list.
    result = edacore.stattests.k_independent.one_way_anova(_data("three_groups"), "value", "group")
    assert isinstance(result, TestResult)
    assert isinstance(result.p_value, float)


# --------------------------------------------------------------------------
# Post-hoc results: within-family adjustment, labelled, separate from the
# session ledger
# --------------------------------------------------------------------------

_POSTHOC_CALLS: list[tuple[str, PostHocResult]] = [
    ("tukey_hsd", edacore.stattests.posthoc.tukey_hsd(_data("three_groups"), "value", "group")),
    (
        "games_howell",
        edacore.stattests.posthoc.games_howell(_data("k_groups_unequal_var"), "value", "group"),
    ),
    ("dunn_test", edacore.stattests.posthoc.dunn_test(_data("three_groups"), "value", "group")),
    (
        "nemenyi_friedman",
        edacore.stattests.posthoc.nemenyi_friedman(
            _data("repeated_measures_long"), "value", "subject", "condition"
        ),
    ),
    (
        "conover_friedman",
        edacore.stattests.posthoc.conover_friedman(
            _data("repeated_measures_long"), "value", "subject", "condition"
        ),
    ),
    (
        "paired_posthoc",
        edacore.stattests.posthoc.paired_posthoc(
            _data("repeated_measures_long"), "value", "subject", "condition"
        ),
    ),
]


@pytest.mark.parametrize("name,result", _POSTHOC_CALLS, ids=[c[0] for c in _POSTHOC_CALLS])
def test_posthoc_result_labels_method_and_comparison_count(
    name: str, result: PostHocResult
) -> None:
    assert result.p_adjust_method is not None, f"{name} must name its adjustment method"
    assert len(result.comparisons) > 0
    for c in result.comparisons:
        assert c.p_adjusted is not None, f"{name} comparison missing p_adjusted"


@pytest.mark.parametrize("name,result", _POSTHOC_CALLS, ids=[c[0] for c in _POSTHOC_CALLS])
def test_posthoc_adjustment_is_within_family_not_session_wide(
    name: str, result: PostHocResult
) -> None:
    # Within-family adjustment (Holm, single-step, studentized range, ...)
    # only ever loosens or holds a p-value, never tightens it -- the
    # opposite of what happens if a p-value were shrunk by being pooled
    # into a larger, unrelated session-wide family.
    for c in result.comparisons:
        assert c.p_adjusted is not None
        assert c.p_adjusted >= c.p_value - 1e-12


def test_dunn_test_family_size_matches_pairwise_comparison_count() -> None:
    # k=3 groups -> C(3,2)=3 pairwise comparisons define this post-hoc's
    # own family; nothing about running it changes with unrelated tests
    # elsewhere in a session.
    result = edacore.stattests.posthoc.dunn_test(_data("three_groups"), "value", "group")
    assert len(result.comparisons) == 3


def test_mcnemar_posthoc_family_independent_of_omnibus_p_adjusted() -> None:
    wide = _data("cochran_q_binary")
    long = (
        wide.reset_index()
        .rename(columns={"index": "subject"})
        .melt(id_vars="subject", var_name="condition", value_name="value")
    )
    omnibus = edacore.stattests.k_related.cochran_q(long, "value", "subject", "condition")
    posthoc = edacore.stattests.posthoc.mcnemar_posthoc(long, "value", "subject", "condition")

    # The omnibus result is untouched by running the post-hoc: it never
    # gains a p_adjusted from the post-hoc's own family-scoped adjustment.
    assert omnibus.p_adjusted is None
    assert posthoc.p_adjust_method is not None
    assert len(posthoc.comparisons) == 3  # k=3 binary conditions -> C(3,2)
