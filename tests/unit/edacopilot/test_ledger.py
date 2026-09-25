"""The test ledger (ARCHITECTURE.md, Section 12.3).

The ledger exists because of one specific way analyses go wrong: run twenty
tests, report the one under 0.05. So the properties worth testing are that
the adjustment is the project's own `adjust_pvalues` on the same inputs
(M6's acceptance criterion), that it is recomputed rather than accumulated,
and that Section 12.3's omnibus/post-hoc rule holds exactly.
"""

from __future__ import annotations

import pytest

from edacopilot.session import TestLedger, is_posthoc
from edacore.contracts import PairwiseComparison, PostHocResult, TestResult
from edacore.multiplicity import adjust_pvalues


def _result(fact_id: str, p: float, function: str = "welch_t") -> TestResult:
    return TestResult(
        fact_id=fact_id,
        function=function,
        estimand="difference in means",
        statistic=1.0,
        statistic_name="t",
        p_value=p,
        ci_level=0.95,
        n={"A": 10, "B": 10},
    )


def _posthoc(n_comparisons: int = 3, method: str = "tukey") -> PostHocResult:
    return PostHocResult(
        fact_id="tukey_hsd.value",
        function="tukey_hsd",
        method=method,
        p_adjust_method="tukey",
        comparisons=[
            PairwiseComparison(
                group_a=f"g{i}",
                group_b=f"g{i + 1}",
                p_value=0.01 * (i + 1),
                p_adjusted=0.03 * (i + 1),
            )
            for i in range(n_comparisons)
        ],
        n={"total": 30},
    )


# --------------------------------------------------------------------------
# The acceptance criterion: adjustment matches adjust_pvalues
# --------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["holm", "bonferroni", "fdr_bh", "fdr_by"])
def test_adjusted_p_values_match_adjust_pvalues_on_the_same_inputs(method: str) -> None:
    """M6's acceptance criterion. The ledger delegates to edacore rather
    than reimplementing, and this proves it did not drift."""
    raw = [0.001, 0.02, 0.04, 0.3, 0.8]
    ledger = TestLedger(method=method)  # type: ignore[arg-type]
    for index, p in enumerate(raw):
        ledger.record(_result(f"f{index}", p))

    expected = [float(p) for p in adjust_pvalues(raw, method=method)]
    assert [e.p_adjusted for e in ledger.entries] == pytest.approx(expected)


def test_adjustment_is_recomputed_not_accumulated() -> None:
    """Holm is a step-down method: adding a test changes the adjusted
    p-value of every earlier one. A ledger that computed each entry once
    would report stale numbers for everything but the newest test."""
    ledger = TestLedger()
    ledger.record(_result("f0", 0.04))
    after_one = ledger.by_id("t0").p_adjusted

    for index in range(1, 5):
        ledger.record(_result(f"f{index}", 0.04))
    after_five = ledger.by_id("t0").p_adjusted

    assert after_one == pytest.approx(0.04)
    assert after_five is not None and after_five > after_one  # type: ignore[operator]
    assert [e.p_adjusted for e in ledger.entries] == pytest.approx(
        [float(p) for p in adjust_pvalues([0.04] * 5, method="holm")]
    )


def test_changing_the_method_recomputes_everything() -> None:
    """Section 12.3: "using the active persona method or the user's session
    setting" -- so the setting has to be changeable after the fact."""
    ledger = TestLedger()
    for index, p in enumerate([0.01, 0.02, 0.03]):
        ledger.record(_result(f"f{index}", p))
    holm = [e.p_adjusted for e in ledger.entries]

    ledger.set_method("bonferroni")
    bonferroni = [e.p_adjusted for e in ledger.entries]
    assert holm != bonferroni
    assert bonferroni == pytest.approx(
        [float(p) for p in adjust_pvalues([0.01, 0.02, 0.03], method="bonferroni")]
    )


def test_per_family_adjustment_is_separate_from_the_session_wide_one() -> None:
    """Section 12.3 asks for both. A family of two adjusted among five
    session tests is not the same number as a family of two on its own."""
    ledger = TestLedger()
    ledger.record(_result("a0", 0.01), family="income")
    ledger.record(_result("a1", 0.02), family="income")
    ledger.record(_result("b0", 0.03), family="age")
    ledger.record(_result("b1", 0.04), family="age")

    income = [e for e in ledger.entries if e.family == "income"]
    assert [e.p_adjusted_in_family for e in income] == pytest.approx(
        [float(p) for p in adjust_pvalues([0.01, 0.02], method="holm")]
    )
    assert income[0].p_adjusted != pytest.approx(income[0].p_adjusted_in_family)
    assert ledger.families() == ["age", "income"]


# --------------------------------------------------------------------------
# Section 12.3's omnibus / post-hoc rule
# --------------------------------------------------------------------------


def test_an_omnibus_test_is_exactly_one_entry() -> None:
    """A one-way ANOVA over five groups is one conclusion, not ten."""
    ledger = TestLedger()
    ledger.record(_result("one_way_anova.value", 0.003, function="one_way_anova"))
    assert len(ledger) == 1
    assert ledger.n_tests == 1
    assert ledger.entries[0].p_value == pytest.approx(0.003)


def test_a_posthoc_does_not_count_toward_the_session_total() -> None:
    """Its comparisons are already adjusted within their own family by its
    own procedure; pooling them into a session-wide correction would adjust
    them twice and make the family's numbers uninterpretable."""
    ledger = TestLedger()
    ledger.record(_result("one_way_anova.value", 0.003, function="one_way_anova"))
    entry = ledger.record_posthoc(_posthoc())

    assert len(ledger) == 2, "the post-hoc is still recorded, for provenance"
    assert ledger.n_tests == 1, "but it does not count"
    assert entry.counts_toward_session is False
    assert entry.p_adjusted is None
    assert entry.internal_adjust_method == "tukey"
    assert entry.n_comparisons == 3


def test_a_posthoc_does_not_change_any_other_entrys_adjusted_p() -> None:
    """The sharpest form of the rule: adding a post-hoc must leave every
    real test's adjusted p-value exactly where it was."""
    ledger = TestLedger()
    for index, p in enumerate([0.01, 0.02, 0.04]):
        ledger.record(_result(f"f{index}", p))
    before = [e.p_adjusted for e in ledger.entries]

    ledger.record_posthoc(_posthoc(n_comparisons=6))
    after = [e.p_adjusted for e in ledger.entries if e.counts_toward_session]
    assert after == pytest.approx(before)


def test_the_omnibus_that_triggered_a_posthoc_still_counts() -> None:
    """Section 12.3's reasoning: running a follow-up does not change how
    many session-level conclusions the omnibus result itself supports."""
    ledger = TestLedger()
    ledger.record(_result("kruskal_wallis.value", 0.01, function="kruskal_wallis"))
    ledger.record_posthoc(_posthoc())
    ledger.record(_result("welch_t.other", 0.04))
    assert ledger.n_tests == 2
    assert [e.p_adjusted for e in ledger.entries if e.counts_toward_session] == pytest.approx(
        [float(p) for p in adjust_pvalues([0.01, 0.04], method="holm")]
    )


def test_posthoc_functions_are_recognised_from_the_registry() -> None:
    """A future post-hoc registered with kind="posthoc" is handled without
    editing the module's own list."""
    assert is_posthoc("tukey_hsd")
    assert is_posthoc("dunn_test")
    assert not is_posthoc("welch_t")
    assert not is_posthoc("something_unregistered")


def test_posthoc_summaries_report_the_internal_family_without_readjusting() -> None:
    ledger = TestLedger()
    ledger.record_posthoc(_posthoc(n_comparisons=4))
    summary = ledger.posthoc_summaries()[0]
    assert summary.n_comparisons == 4
    assert summary.p_adjust_method == "tukey"


# --------------------------------------------------------------------------
# Shape
# --------------------------------------------------------------------------


def test_rows_show_raw_and_adjusted_p_and_the_count() -> None:
    """Section 12.3: "The result card always shows raw p, adjusted p, and
    the test count"."""
    ledger = TestLedger()
    ledger.record(_result("f0", 0.01), family="income")
    row = ledger.rows()[0]
    assert row["p_value"] == pytest.approx(0.01)
    assert row["p_adjusted"] is not None
    assert row["p_adjusted_in_family"] is not None
    assert row["family"] == "income"
    assert ledger.n_tests == 1


def test_an_empty_ledger_is_well_defined() -> None:
    ledger = TestLedger()
    assert ledger.n_tests == 0
    assert ledger.rows() == []
    assert ledger.families() == []


def test_a_test_with_no_p_value_does_not_enter_the_adjustment() -> None:
    """`bootstrap_one_sample` and `bootstrap_diff` report an interval and no
    p-value (Section 6.7's "CI only"). There is nothing to adjust."""
    ledger = TestLedger()
    ledger.record(_result("boot", 0.02))
    no_p = TestResult(
        fact_id="bootstrap_diff.value",
        function="bootstrap_diff",
        estimand="difference in means",
        statistic=1.0,
        statistic_name="mean_diff",
        p_value=None,
        ci_level=0.95,
        n={"A": 10},
    )
    ledger.record(no_p)
    assert ledger.n_tests == 1
    assert ledger.entries[1].p_adjusted is None
