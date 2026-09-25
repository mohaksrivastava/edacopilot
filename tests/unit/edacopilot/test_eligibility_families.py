"""Table-driven eligibility tests, one table per method family
(ARCHITECTURE.md, Section 15.2).

Each case is: synthetic data with a known property, a confirmed question,
and the expected status of *every* candidate the family offers -- not just
the interesting one. Asserting the whole table is the point: a rule that
accidentally downgrades a method nobody was looking at is exactly the kind
of regression a spot-check misses.

Datasets are built to make the intended verdict unambiguous (clean normal
data really is normal; heteroscedastic data really has a 4x SD ratio), so a
status here is a statement about the engine, not about a marginal p-value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edacopilot.eligibility import (
    Design,
    Goal,
    QuestionSpec,
    UnsupportedQuestionError,
    select_candidates,
    validate_spec,
)
from edacore.contracts import Eligibility

ELIGIBLE = Eligibility.ELIGIBLE
CAVEAT = Eligibility.CAVEAT
INELIGIBLE = Eligibility.INELIGIBLE


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _statuses(spec: QuestionSpec, df: pd.DataFrame) -> tuple[str, dict[str, Eligibility]]:
    validated = validate_spec(spec, df)
    assert validated.ambiguities == [], f"unexpected ambiguities: {validated.ambiguities}"
    candidates = select_candidates(validated, df)
    return candidates.family, candidates.statuses()


def _assert_table(
    spec: QuestionSpec,
    df: pd.DataFrame,
    expected_family: str,
    expected: dict[str, Eligibility],
) -> None:
    family, statuses = _statuses(spec, df)
    assert family == expected_family
    assert statuses == expected


# --------------------------------------------------------------------------
# two_independent_numeric
# --------------------------------------------------------------------------


def _two_group_spec() -> QuestionSpec:
    return QuestionSpec(
        goal=Goal.COMPARE_GROUPS,
        variables={"outcome": "value", "group": "group"},
        design=Design.INDEPENDENT,
        confirmed_by_user={"design"},
    )


def _two_groups(
    n_a: int, n_b: int, sd_a: float, sd_b: float, seed: int, mean_b: float = 12.0
) -> pd.DataFrame:
    rng = _rng(seed)
    return pd.DataFrame(
        {
            "value": np.concatenate([rng.normal(10, sd_a, n_a), rng.normal(mean_b, sd_b, n_b)]),
            "group": ["A"] * n_a + ["B"] * n_b,
        }
    )


def test_two_independent_numeric_all_assumptions_met() -> None:
    """Clean, equal-variance, normal data: everything is eligible, including
    Student's t. If this table ever shows a caveat, a check has drifted."""
    _assert_table(
        _two_group_spec(),
        _two_groups(40, 40, 2.0, 2.0, seed=1),
        "two_independent_numeric",
        {
            "student_t": ELIGIBLE,
            "welch_t": ELIGIBLE,
            "yuen_trimmed_t": ELIGIBLE,
            "mann_whitney": ELIGIBLE,
            "brunner_munzel": ELIGIBLE,
            "permutation_test_2s": ELIGIBLE,
            "bootstrap_diff": ELIGIBLE,
            "ks_two_sample": ELIGIBLE,
        },
    )


def test_two_independent_numeric_unequal_variance_caveats_student_t_only() -> None:
    """Section 15.3's `heteroscedastic_groups` property, isolated: unequal
    variance is Student's problem alone. Welch exists precisely for it, and
    the rank-based and resampling methods do not assume it."""
    _assert_table(
        _two_group_spec(),
        _two_groups(60, 25, 3.0, 12.0, seed=21),
        "two_independent_numeric",
        {
            "student_t": CAVEAT,
            "welch_t": ELIGIBLE,
            "yuen_trimmed_t": ELIGIBLE,
            "mann_whitney": ELIGIBLE,
            "brunner_munzel": ELIGIBLE,
            "permutation_test_2s": ELIGIBLE,
            "bootstrap_diff": ELIGIBLE,
            "ks_two_sample": ELIGIBLE,
        },
    )


def test_two_independent_numeric_ordinal_outcome_blocks_the_mean_based_methods() -> None:
    """An ordinal outcome is a HARD failure for the t-tests (a mean of
    Likert codes is not a quantity), but rank-based methods are fine."""
    rng = _rng(22)
    df = pd.DataFrame({"value": rng.integers(1, 6, 80), "group": ["A"] * 40 + ["B"] * 40})
    spec = _two_group_spec().model_copy(update={"confirmed_by_user": {"design", "outcome"}})
    _assert_table(
        spec,
        df,
        "two_independent_numeric",
        {
            "student_t": INELIGIBLE,
            "welch_t": INELIGIBLE,
            "yuen_trimmed_t": INELIGIBLE,
            "mann_whitney": ELIGIBLE,
            "brunner_munzel": ELIGIBLE,
            "permutation_test_2s": INELIGIBLE,
            "bootstrap_diff": INELIGIBLE,
            "ks_two_sample": INELIGIBLE,
        },
    )


def test_two_independent_numeric_one_observation_in_a_group_blocks_student_t() -> None:
    """`min_n_per_group>=2` is Student's only count-based hard assumption."""
    df = _two_groups(30, 30, 2.0, 2.0, seed=23)
    df = pd.concat([df[df["group"] == "A"], df[df["group"] == "B"].head(1)], ignore_index=True)
    _, statuses = _statuses(_two_group_spec(), df)
    assert statuses["student_t"] is INELIGIBLE
    assert statuses["welch_t"] is not INELIGIBLE


# --------------------------------------------------------------------------
# two_paired_numeric
# --------------------------------------------------------------------------


def _paired_df(n: int, seed: int, skew: bool = False) -> pd.DataFrame:
    rng = _rng(seed)
    base = rng.normal(100, 15, n)
    delta = rng.exponential(6, n) if skew else rng.normal(5, 4, n)
    return pd.DataFrame(
        {
            "subject_id": np.tile(np.arange(n), 2),
            "group": ["pre"] * n + ["post"] * n,
            "value": np.concatenate([base, base + delta]),
        }
    )


def _paired_spec() -> QuestionSpec:
    return QuestionSpec(
        goal=Goal.COMPARE_GROUPS,
        variables={"outcome": "value", "group": "group", "subject": "subject_id"},
        design=Design.PAIRED,
        confirmed_by_user={"design"},
    )


def test_two_paired_numeric_all_assumptions_met() -> None:
    _assert_table(
        _paired_spec(),
        _paired_df(40, seed=24),
        "two_paired_numeric",
        {
            "paired_t": ELIGIBLE,
            "wilcoxon_signed_rank": ELIGIBLE,
            "sign_test_paired": ELIGIBLE,
            "permutation_test_paired": ELIGIBLE,
        },
    )


def test_two_paired_numeric_skewed_differences_caveat_the_right_two() -> None:
    """Skewed differences are a caveat for the paired t (normality of
    differences) and for Wilcoxon (symmetry of differences) -- and for
    nothing else. The sign test only uses the sign, so it is untouched."""
    _, statuses = _statuses(_paired_spec(), _paired_df(40, seed=25, skew=True))
    assert statuses["paired_t"] is CAVEAT
    assert statuses["wilcoxon_signed_rank"] is CAVEAT
    assert statuses["sign_test_paired"] is ELIGIBLE
    assert statuses["permutation_test_paired"] is ELIGIBLE


# --------------------------------------------------------------------------
# k_independent_numeric
# --------------------------------------------------------------------------


def _k_groups(sds: tuple[float, ...], n: int, seed: int) -> pd.DataFrame:
    rng = _rng(seed)
    means = (50.0, 55.0, 60.0)
    return pd.DataFrame(
        {
            "value": np.concatenate(
                [rng.normal(m, sd, n) for m, sd in zip(means, sds, strict=True)]
            ),
            "group": sum(([label] * n for label in "ABC"), []),
        }
    )


def _k_spec() -> QuestionSpec:
    return QuestionSpec(
        goal=Goal.COMPARE_GROUPS,
        variables={"outcome": "value", "group": "group"},
        design=Design.INDEPENDENT,
        confirmed_by_user={"design"},
    )


def test_k_independent_numeric_all_assumptions_met() -> None:
    _assert_table(
        _k_spec(),
        _k_groups((8.0, 8.0, 8.0), 30, seed=0),
        "k_independent_numeric",
        {
            "one_way_anova": ELIGIBLE,
            "welch_anova": ELIGIBLE,
            "alexander_govern": ELIGIBLE,
            "kruskal_wallis": ELIGIBLE,
            "permutation_anova": ELIGIBLE,
        },
    )


def test_k_independent_numeric_unequal_variance_caveats_only_the_pooled_anova() -> None:
    _, statuses = _statuses(_k_spec(), _k_groups((3.0, 9.0, 20.0), 30, seed=27))
    assert statuses["one_way_anova"] is CAVEAT
    assert statuses["welch_anova"] is ELIGIBLE
    assert statuses["kruskal_wallis"] is ELIGIBLE
    assert statuses["permutation_anova"] is ELIGIBLE


# --------------------------------------------------------------------------
# k_repeated_numeric / k_repeated_binary
# --------------------------------------------------------------------------


def _repeated(n_subjects: int, k: int, seed: int, binary: bool = False) -> pd.DataFrame:
    rng = _rng(seed)
    if binary:
        values = np.concatenate([rng.binomial(1, p, n_subjects) for p in np.linspace(0.35, 0.7, k)])
    else:
        base = rng.normal(50, 8, n_subjects)
        values = np.concatenate(
            [base + shift + rng.normal(0, 3, n_subjects) for shift in np.linspace(0, 9, k)]
        )
    return pd.DataFrame(
        {
            "subject_id": np.tile(np.arange(n_subjects), k),
            "condition": np.repeat([f"t{i}" for i in range(1, k + 1)], n_subjects),
            "value": values,
        }
    )


def _repeated_spec() -> QuestionSpec:
    return QuestionSpec(
        goal=Goal.COMPARE_GROUPS,
        variables={"outcome": "value", "group": "condition", "subject": "subject_id"},
        design=Design.REPEATED,
        confirmed_by_user={"design"},
    )


def test_k_repeated_numeric_all_assumptions_met() -> None:
    _assert_table(
        _repeated_spec(),
        _repeated(25, 3, seed=28),
        "k_repeated_numeric",
        {"repeated_measures_anova": ELIGIBLE, "friedman": ELIGIBLE},
    )


def test_k_repeated_numeric_unbalanced_blocks_the_anova_but_not_friedman() -> None:
    """`balanced` is a hard assumption of the repeated-measures ANOVA
    alone: without every subject in every condition there is no
    within-subject error term to build."""
    df = _repeated(25, 3, seed=29)
    df = df.drop(df.index[-1])
    _, statuses = _statuses(_repeated_spec(), df)
    assert statuses["repeated_measures_anova"] is INELIGIBLE
    assert statuses["friedman"] is ELIGIBLE


def test_k_repeated_binary_routes_to_cochran_q() -> None:
    _assert_table(
        _repeated_spec(),
        _repeated(30, 3, seed=30, binary=True),
        "k_repeated_binary",
        {"cochran_q": ELIGIBLE},
    )


# --------------------------------------------------------------------------
# factorial_numeric
# --------------------------------------------------------------------------


def test_factorial_numeric_routes_on_a_second_factor() -> None:
    rng = _rng(31)
    n = 12
    df = pd.DataFrame(
        {
            "value": np.concatenate([rng.normal(m, 2, n) for m in (10.0, 12.0, 9.0, 15.0)]),
            "factor_a": np.repeat(["A1", "A1", "A2", "A2"], n),
            "factor_b": np.repeat(["B1", "B2", "B1", "B2"], n),
        }
    )
    spec = QuestionSpec(
        goal=Goal.COMPARE_GROUPS,
        variables={"outcome": "value", "group": "factor_a", "factor_b": "factor_b"},
        design=Design.INDEPENDENT,
        confirmed_by_user={"design"},
    )
    _assert_table(
        spec,
        df,
        "factorial_numeric",
        {"two_way_anova": ELIGIBLE, "aligned_rank_transform_anova": ELIGIBLE},
    )


# --------------------------------------------------------------------------
# binary_outcome_groups
# --------------------------------------------------------------------------


def _binary_outcome(n_per_group: int, p_a: float, p_b: float, seed: int) -> pd.DataFrame:
    rng = _rng(seed)
    return pd.DataFrame(
        {
            "converted": np.concatenate(
                [rng.binomial(1, p_a, n_per_group), rng.binomial(1, p_b, n_per_group)]
            ),
            "arm": ["control"] * n_per_group + ["treatment"] * n_per_group,
        }
    )


def _binary_spec() -> QuestionSpec:
    return QuestionSpec(
        goal=Goal.COMPARE_GROUPS,
        variables={"outcome": "converted", "group": "arm"},
        design=Design.INDEPENDENT,
        confirmed_by_user={"design"},
    )


def test_binary_outcome_groups_all_assumptions_met() -> None:
    _assert_table(
        _binary_spec(),
        _binary_outcome(120, 0.35, 0.5, seed=32),
        "binary_outcome_groups",
        {"two_proportion_z": ELIGIBLE, "chi2_independence": ELIGIBLE, "fisher_exact": ELIGIBLE},
    )


def test_binary_outcome_groups_rare_events_block_the_approximations_not_fisher() -> None:
    """The whole point of Fisher's exact test: when the counts are too small
    for the normal and chi-square approximations, it still applies."""
    _, statuses = _statuses(_binary_spec(), _binary_outcome(25, 0.04, 0.10, seed=33))
    assert statuses["two_proportion_z"] is INELIGIBLE
    assert statuses["chi2_independence"] is INELIGIBLE
    assert statuses["fisher_exact"] is ELIGIBLE


# --------------------------------------------------------------------------
# numeric_numeric / ordinal_any / binary_numeric / categorical
# --------------------------------------------------------------------------


def _assoc_spec(x: str, y: str, **extra: object) -> QuestionSpec:
    return QuestionSpec(
        goal=Goal.ASSOCIATION,
        variables={"x": x, "y": y},
        design=Design.INDEPENDENT,
        confirmed_by_user={"design", "x", "y"},
        **extra,  # type: ignore[arg-type]
    )


def test_numeric_numeric_linear_normal_data_is_all_eligible() -> None:
    rng = _rng(34)
    x = rng.normal(0, 1, 120)
    df = pd.DataFrame({"x": x, "y": 0.6 * x + rng.normal(0, 0.8, 120)})
    _assert_table(
        _assoc_spec("x", "y"),
        df,
        "numeric_numeric",
        {
            "pearson": ELIGIBLE,
            "spearman": ELIGIBLE,
            "kendall_tau": ELIGIBLE,
            "distance_correlation": ELIGIBLE,
            "mutual_information": ELIGIBLE,
        },
    )


def test_numeric_numeric_curved_relationship_caveats_pearson_alone() -> None:
    """A monotone but curved relationship: Pearson's linearity assumption
    fails, Spearman's monotonicity assumption holds. This is the case the
    two exist to separate."""
    rng = _rng(35)
    x = np.linspace(1, 50, 120)
    df = pd.DataFrame({"x": x, "y": np.log(x) + rng.normal(0, 0.05, 120)})
    _, statuses = _statuses(_assoc_spec("x", "y"), df)
    assert statuses["pearson"] is CAVEAT
    assert statuses["spearman"] is ELIGIBLE
    assert statuses["kendall_tau"] is ELIGIBLE


def test_binary_numeric_routes_to_point_biserial_and_mann_whitney() -> None:
    rng = _rng(36)
    group = np.repeat([0, 1], 60)
    df = pd.DataFrame({"flag": group, "score": rng.normal(10, 2, 120) + 2 * group})
    _assert_table(
        _assoc_spec("flag", "score"),
        df,
        "binary_numeric",
        {"point_biserial": ELIGIBLE, "mann_whitney": ELIGIBLE},
    )


def test_two_categorical_independent_all_assumptions_met() -> None:
    rng = _rng(37)
    df = pd.DataFrame(
        {
            "region": rng.choice(["north", "south", "east"], 300),
            "channel": rng.choice(["web", "store"], 300),
        }
    )
    _assert_table(
        _assoc_spec("region", "channel"),
        df,
        "two_categorical_independent",
        {"chi2_independence": ELIGIBLE, "fisher_exact": ELIGIBLE, "g_test": ELIGIBLE},
    )


def test_two_binary_paired_routes_to_mcnemar_when_the_design_is_paired() -> None:
    rng = _rng(38)
    before = rng.binomial(1, 0.5, 80)
    df = pd.DataFrame(
        {"before": before, "after": np.where(rng.random(80) < 0.3, 1 - before, before)}
    )
    spec = QuestionSpec(
        goal=Goal.ASSOCIATION,
        variables={"x": "before", "y": "after"},
        design=Design.PAIRED,
        confirmed_by_user={"design", "x", "y"},
    )
    _assert_table(spec, df, "two_binary_paired", {"mcnemar": ELIGIBLE})


# --------------------------------------------------------------------------
# one_sample families
# --------------------------------------------------------------------------


def test_one_sample_numeric_all_assumptions_met() -> None:
    rng = _rng(39)
    df = pd.DataFrame({"weight": rng.normal(70, 8, 80)})
    spec = QuestionSpec(
        goal=Goal.DISTRIBUTION_FIT,
        variables={"variable": "weight"},
        reference_value=68.0,
        confirmed_by_user={"variable"},
    )
    _assert_table(
        spec,
        df,
        "one_sample_numeric",
        {
            "one_sample_t": ELIGIBLE,
            "wilcoxon_one_sample": ELIGIBLE,
            "sign_test": ELIGIBLE,
            "bootstrap_one_sample": ELIGIBLE,
        },
    )


def test_one_sample_numeric_skewed_caveats_the_mean_and_rank_methods() -> None:
    rng = _rng(40)
    df = pd.DataFrame({"wait": rng.exponential(10, 80)})
    spec = QuestionSpec(
        goal=Goal.DISTRIBUTION_FIT,
        variables={"variable": "wait"},
        reference_value=9.0,
        confirmed_by_user={"variable"},
    )
    _, statuses = _statuses(spec, df)
    assert statuses["one_sample_t"] is CAVEAT
    assert statuses["wilcoxon_one_sample"] is CAVEAT
    assert statuses["sign_test"] is ELIGIBLE
    assert statuses["bootstrap_one_sample"] is ELIGIBLE


def test_one_sample_numeric_without_a_reference_value_is_unavailable_not_ineligible() -> None:
    """A missing reference is a question to ask, not a verdict about the
    data -- so the method is reported as unavailable, with the reason."""
    rng = _rng(41)
    df = pd.DataFrame({"weight": rng.normal(70, 8, 50)})
    spec = QuestionSpec(
        goal=Goal.DISTRIBUTION_FIT,
        variables={"variable": "weight"},
        confirmed_by_user={"variable"},
    )
    candidates = select_candidates(validate_spec(spec, df), df)
    assert "one_sample_t" in candidates.unavailable
    assert "reference_value" in candidates.unavailable["one_sample_t"]
    assert candidates.statuses() == {"bootstrap_one_sample": ELIGIBLE}


def test_one_sample_categorical_routes_and_needs_its_reference() -> None:
    rng = _rng(42)
    df = pd.DataFrame({"colour": rng.choice(["red", "green", "blue"], 150)})
    spec = QuestionSpec(
        goal=Goal.DISTRIBUTION_FIT,
        variables={"variable": "colour"},
        reference_proportions={"red": 1 / 3, "green": 1 / 3, "blue": 1 / 3},
        confirmed_by_user={"variable"},
    )
    family, statuses = _statuses(spec, df)
    assert family == "one_sample_categorical"
    assert statuses["chi2_goodness_of_fit"] is ELIGIBLE


# --------------------------------------------------------------------------
# equivalence + unsupported goals
# --------------------------------------------------------------------------


def test_equivalence_two_groups_needs_bounds() -> None:
    df = _two_groups(50, 50, 2.0, 2.0, seed=43, mean_b=10.1)
    with_bounds = QuestionSpec(
        goal=Goal.EQUIVALENCE,
        variables={"outcome": "value", "group": "group"},
        design=Design.INDEPENDENT,
        equivalence_bounds=(-1.0, 1.0),
        confirmed_by_user={"design"},
    )
    _assert_table(with_bounds, df, "equivalence_two_groups", {"tost_equivalence": ELIGIBLE})

    without = with_bounds.model_copy(update={"equivalence_bounds": None})
    candidates = select_candidates(validate_spec(without, df), df)
    assert "tost_equivalence" in candidates.unavailable
    assert candidates.candidates == []


def test_trend_says_which_milestone_serves_it() -> None:
    """A TREND question is well-formed; its methods just do not exist yet.
    Reporting them as INELIGIBLE would misstate a missing feature as a
    statistical verdict."""
    df = pd.DataFrame({"sales": np.arange(50.0)})
    spec = QuestionSpec(goal=Goal.TREND, variables={"variable": "sales"})
    with pytest.raises(UnsupportedQuestionError, match="M12"):
        select_candidates(validate_spec(spec, df), df)


def test_goals_with_no_testing_methods_name_their_milestone() -> None:
    df = pd.DataFrame({"x": np.arange(20.0)})
    for goal, milestone in ((Goal.OUTLIERS, "M11"), (Goal.TRANSFORM, "M11")):
        spec = QuestionSpec(goal=goal, variables={"variable": "x"})
        with pytest.raises(UnsupportedQuestionError, match=milestone):
            select_candidates(validate_spec(spec, df), df)
