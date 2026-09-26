"""Section 6.12's nine interpretation guards.

One test per guard for the case it fires on, and one for the case it must
stay silent on — a guard that fires on everything teaches nothing, and a
guard that never fires is decoration. The two ends of the table get an
end-to-end test against real data (`mann_whitney_shape` off a real
`check_same_shape`, `tiny_effect_large_n` off a real t-test on a large
sample), because those two depend on numbers rather than on flags the
caller sets.

The three guards that need a stage this build does not have yet
(`imputed_data` and `outliers_removed` need M10/M11, `post_selection` needs
the orchestrator to mark the comparison) are tested through the context
they will be handed. They are implemented now because M7's result cards
show validity notes, and a card missing a guard is indistinguishable from
a result that has no problems.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edacore.assumptions import check_same_shape
from edacore.contracts import AssumptionCheck, CheckStatus, TestResult
from edacore.stattests import correlation, k_independent, two_sample
from edacore.validity_notes import (
    GUARD_NAMES,
    GUARDS,
    SMALL_N,
    ValidityContext,
    attach,
    evaluate,
    notes_for,
)


def _result(**overrides: object) -> TestResult:
    base: dict[str, object] = {
        "fact_id": "t.demo",
        "function": "welch_t",
        "estimand": "difference in means",
        "statistic": 2.0,
        "statistic_name": "t",
        "p_value": 0.01,
        "n": {"a": 60, "b": 60},
    }
    base.update(overrides)
    return TestResult(**base)  # type: ignore[arg-type]


def _shape_check(status: CheckStatus) -> AssumptionCheck:
    return AssumptionCheck(
        fact_id="same_shape.v.g",
        assumption="same_distribution_shape",
        method="ks_centered_scaled",
        statistic=0.3,
        p_value=0.001,
        threshold="p < 0.05 -> fail",
        status=status,
        consequence="Different shapes change what a rank test is comparing.",
    )


# --------------------------------------------------------------------------
# the table itself
# --------------------------------------------------------------------------


def test_all_nine_guards_from_section_6_12_exist() -> None:
    """Section 6.12 names nine; a missing one would silently never fire."""
    assert GUARD_NAMES == (
        "non_significance",
        "tiny_effect_large_n",
        "large_effect_small_n",
        "correlation_not_causation",
        "multiple_testing",
        "mann_whitney_shape",
        "post_selection",
        "imputed_data",
        "outliers_removed",
    )
    assert len(GUARDS) == 9


def test_a_clean_result_gets_no_notes() -> None:
    """The baseline that makes every other test meaningful."""
    clean = _result(p_value=0.01, effect_magnitude="small", n={"a": 60, "b": 60})
    assert notes_for(clean) == []


# --------------------------------------------------------------------------
# 1. non_significance
# --------------------------------------------------------------------------


def test_non_significance_quotes_the_interval() -> None:
    fired = evaluate(_result(p_value=0.42, ci=(-1.5, 3.25)))
    note = fired["non_significance"]
    assert "Not significant does not mean no effect" in note
    assert "-1.5" in note and "3.25" in note
    assert "equivalence test" in note


def test_non_significance_falls_back_to_the_effect_size_interval() -> None:
    """A rank test may have no CI on its estimand but one on its effect
    size, which answers the same question in standardised units."""
    note = evaluate(_result(p_value=0.42, ci=None, effect_size_ci=(-0.2, 0.44)))["non_significance"]
    assert "-0.2" in note and "0.44" in note


def test_non_significance_says_so_when_there_is_no_interval_at_all() -> None:
    note = evaluate(_result(p_value=0.42, ci=None, effect_size_ci=None))["non_significance"]
    assert "No interval was computed" in note


def test_non_significance_is_silent_when_significant() -> None:
    assert "non_significance" not in evaluate(_result(p_value=0.001, ci=(1.0, 2.0)))


def test_non_significance_respects_a_non_default_alpha() -> None:
    result = _result(p_value=0.03, ci=(0.1, 2.0))
    assert "non_significance" not in evaluate(result, ValidityContext(alpha=0.05))
    assert "non_significance" in evaluate(result, ValidityContext(alpha=0.01))


# --------------------------------------------------------------------------
# 2. tiny_effect_large_n
# --------------------------------------------------------------------------


def test_tiny_effect_large_n_fires_on_a_significant_negligible_effect() -> None:
    note = evaluate(
        _result(p_value=0.001, effect_magnitude="negligible", n={"a": 50_000, "b": 50_000})
    )["tiny_effect_large_n"]
    assert "negligible in size" in note
    assert "n=100000" in note


def test_tiny_effect_large_n_end_to_end_on_a_real_huge_n_tiny_effect() -> None:
    """Section 15.3's `huge_n_tiny_effect` scenario: "`tiny_effect_large_n`
    note fires". Asserted against a real t-test rather than a hand-built
    result, because the whole claim is about what real data does."""
    rng = np.random.default_rng(0)
    n = 200_000
    df = pd.DataFrame(
        {
            "v": np.concatenate([rng.normal(0.0, 1, n), rng.normal(0.01, 1, n)]),
            "g": ["a"] * n + ["b"] * n,
        }
    )
    result = two_sample.welch_t(df, "g", "v")
    assert result.p_value is not None and result.p_value < 0.05
    assert result.effect_magnitude == "negligible"
    assert "tiny_effect_large_n" in evaluate(result)


def test_tiny_effect_large_n_is_silent_when_not_significant() -> None:
    assert "tiny_effect_large_n" not in evaluate(
        _result(p_value=0.4, effect_magnitude="negligible", ci=(-1.0, 1.0))
    )


# --------------------------------------------------------------------------
# 3. large_effect_small_n
# --------------------------------------------------------------------------


def test_large_effect_small_n_fires_on_the_smallest_group() -> None:
    """ "n small" is the smallest group, not the total (Section 18, M6.1).

    This is the case the decision turns on: 200 against 12 is a preliminary
    result whatever the total says.
    """
    note = evaluate(_result(effect_magnitude="large", n={"a": 200, "b": 12}))[
        "large_effect_small_n"
    ]
    assert "treat as preliminary" in note
    assert "n=12" in note


def test_large_effect_small_n_is_silent_when_every_group_is_big_enough() -> None:
    assert "large_effect_small_n" not in evaluate(
        _result(effect_magnitude="large", n={"a": SMALL_N, "b": SMALL_N})
    )


@pytest.mark.parametrize("magnitude", ["negligible", "small"])
def test_large_effect_small_n_needs_at_least_a_medium_effect(magnitude: str) -> None:
    assert "large_effect_small_n" not in evaluate(
        _result(effect_magnitude=magnitude, n={"a": 8, "b": 9})
    )


# --------------------------------------------------------------------------
# 4. correlation_not_causation
# --------------------------------------------------------------------------


def test_correlation_not_causation_fires_on_a_real_correlation() -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(size=40)})
    df["y"] = df["x"] * 0.8 + rng.normal(size=40) * 0.5
    result = correlation.pearson(df, "x", "y")
    assert (
        "does not show that one variable causes the other"
        in (evaluate(result)["correlation_not_causation"])
    )


def test_correlation_not_causation_fires_on_a_contingency_table_test() -> None:
    """chi-square independence has no `correlation` tag but makes the same
    claim, so it is named explicitly."""
    assert "correlation_not_causation" in evaluate(_result(function="chi2_independence"))


def test_correlation_not_causation_fires_when_the_goal_is_association() -> None:
    fired = evaluate(_result(function="mann_whitney"), ValidityContext(goal="association"))
    assert "correlation_not_causation" in fired


def test_correlation_not_causation_is_silent_on_a_group_comparison() -> None:
    assert "correlation_not_causation" not in evaluate(_result(function="welch_t"))


# --------------------------------------------------------------------------
# 5. multiple_testing
# --------------------------------------------------------------------------


def test_multiple_testing_names_the_count_and_the_adjusted_p() -> None:
    note = evaluate(
        _result(),
        ValidityContext(test_number=7, adjust_method="holm", p_adjusted=0.031),
    )["multiple_testing"]
    assert "test #7" in note
    assert "holm" in note
    assert "0.031" in note


def test_multiple_testing_is_silent_on_the_first_test() -> None:
    assert "multiple_testing" not in evaluate(_result(), ValidityContext(test_number=1))


# --------------------------------------------------------------------------
# 6. mann_whitney_shape
# --------------------------------------------------------------------------


def test_mann_whitney_shape_fires_end_to_end_when_shapes_really_differ() -> None:
    rng = np.random.default_rng(7)
    n = 200
    df = pd.DataFrame(
        {
            "v": np.concatenate([rng.normal(0, 1, n), rng.exponential(1, n)]),
            "g": ["a"] * n + ["b"] * n,
        }
    )
    check = check_same_shape(df, "v", "g")
    assert check.status is CheckStatus.FAIL, "the fixture no longer plants differing shapes"

    result = two_sample.mann_whitney(df, "g", "v")
    note = evaluate(result, ValidityContext(checks=[check]))["mann_whitney_shape"]
    assert "not a difference in medians" in note


def test_mann_whitney_shape_covers_kruskal_wallis() -> None:
    """The decision recorded in Section 18 (M6.1): Kruskal-Wallis declares
    the same assumption and invites the same misreading."""
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "v": np.concatenate(
                [rng.normal(0, 1, 40), rng.normal(0.5, 1, 40), rng.normal(1, 1, 40)]
            ),
            "g": ["a"] * 40 + ["b"] * 40 + ["c"] * 40,
        }
    )
    result = k_independent.kruskal_wallis(df, "v", "g")
    fired = evaluate(result, ValidityContext(checks=[_shape_check(CheckStatus.FAIL)]))
    assert "mann_whitney_shape" in fired


def test_mann_whitney_shape_is_silent_when_the_shapes_match() -> None:
    result = _result(function="mann_whitney")
    assert "mann_whitney_shape" not in evaluate(
        result, ValidityContext(checks=[_shape_check(CheckStatus.PASS)])
    )


def test_mann_whitney_shape_is_silent_for_a_parametric_test() -> None:
    assert "mann_whitney_shape" not in evaluate(
        _result(function="welch_t"), ValidityContext(checks=[_shape_check(CheckStatus.FAIL)])
    )


# --------------------------------------------------------------------------
# 7. post_selection
# --------------------------------------------------------------------------


def test_post_selection_fires_only_when_the_orchestrator_marks_it() -> None:
    assert "post_selection" not in evaluate(_result())
    note = evaluate(_result(), ValidityContext(chosen_after_exploring=True))["post_selection"]
    assert "its p-value is optimistic" in note


# --------------------------------------------------------------------------
# 8. imputed_data
# --------------------------------------------------------------------------


def test_imputed_data_reports_the_share_of_filled_values() -> None:
    note = evaluate(
        _result(),
        ValidityContext(
            variables={"outcome": "income", "group": "region"},
            imputed_fractions={"income": 0.18},
        ),
    )["imputed_data"]
    assert "18%" in note
    assert "understates uncertainty" in note


def test_imputed_data_ignores_a_column_this_test_does_not_use() -> None:
    """Imputing an unrelated column does not weaken this result."""
    assert "imputed_data" not in evaluate(
        _result(),
        ValidityContext(
            variables={"outcome": "income", "group": "region"},
            imputed_fractions={"unrelated_column": 0.9},
        ),
    )


# --------------------------------------------------------------------------
# 9. outliers_removed
# --------------------------------------------------------------------------


def test_outliers_removed_reports_the_count() -> None:
    note = evaluate(
        _result(),
        ValidityContext(variables={"outcome": "income"}, outliers_removed={"income": 14}),
    )["outliers_removed"]
    assert "14 outlier(s) were removed" in note


def test_outliers_removed_ignores_an_unrelated_column() -> None:
    assert "outliers_removed" not in evaluate(
        _result(),
        ValidityContext(variables={"outcome": "income"}, outliers_removed={"age": 14}),
    )


# --------------------------------------------------------------------------
# attach()
# --------------------------------------------------------------------------


def test_attach_appends_in_section_6_12_order() -> None:
    result = _result(
        function="pearson",
        p_value=0.42,
        ci=(-0.1, 0.5),
        n={"total": 20},
        effect_magnitude="medium",
    )
    context = ValidityContext(test_number=4, adjust_method="holm", p_adjusted=0.9)
    attached = attach(result, context)
    assert [note.split(" ")[0] for note in attached.validity_notes][:1] == ["Not"]
    assert len(attached.validity_notes) == 4
    # The order is the table's, not the order the guards happened to fire.
    assert "preliminary" in attached.validity_notes[1]
    assert "Association only" in attached.validity_notes[2]
    assert "test #4" in attached.validity_notes[3]


def test_attach_keeps_notes_a_test_function_already_wrote() -> None:
    """Several edacore tests use `validity_notes` to carry their own facts
    (Mauchly's test, the runs count). Those must survive."""
    result = _result(p_value=0.42, ci=(-1.0, 1.0), validity_notes=["Mauchly W = 0.71, p = 0.03"])
    attached = attach(result)
    assert attached.validity_notes[0] == "Mauchly W = 0.71, p = 0.03"
    assert len(attached.validity_notes) == 2


def test_attach_is_idempotent() -> None:
    result = _result(p_value=0.42, ci=(-1.0, 1.0))
    once = attach(result)
    assert attach(once).validity_notes == once.validity_notes


def test_attach_returns_the_same_object_when_nothing_fires() -> None:
    clean = _result(effect_magnitude="small")
    assert attach(clean) is clean


def test_attach_does_not_mutate_its_input() -> None:
    """Rule 6: nothing in edacore mutates what it is given."""
    result = _result(p_value=0.42, ci=(-1.0, 1.0))
    attach(result)
    assert result.validity_notes == []
