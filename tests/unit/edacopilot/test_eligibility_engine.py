"""Engine mechanics and invariants (ARCHITECTURE.md, Section 7.2).

The family tables assert *what* the engine decides. This file asserts the
properties that have to hold whatever it decides: the status rule itself,
the ranking, the cache, and the invariants that keep the assumption
vocabulary and the engine from drifting apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from edacopilot.eligibility import (
    CheckCache,
    Design,
    Goal,
    QuestionSpec,
    select_candidates,
    validate_spec,
)
from edacopilot.eligibility.checks import ResolutionContext, resolve
from edacopilot.eligibility.rules import ALL_FAMILIES, PENDING_GOALS, RULES
from edacore.contracts import CheckStatus, Eligibility
from edacore.registry import registry


def _clean_two_groups(seed: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "value": np.concatenate([rng.normal(10, 2, 40), rng.normal(12, 2, 40)]),
            "group": ["A"] * 40 + ["B"] * 40,
        }
    )


def _spec() -> QuestionSpec:
    return QuestionSpec(
        goal=Goal.COMPARE_GROUPS,
        variables={"outcome": "value", "group": "group"},
        design=Design.INDEPENDENT,
        confirmed_by_user={"design"},
    )


def _validated(df: pd.DataFrame) -> QuestionSpec:
    return validate_spec(_spec(), df)


# --------------------------------------------------------------------------
# Invariants across the whole registry
# --------------------------------------------------------------------------


# The assumption-vocabulary invariant (every declared assumption resolves to
# a check or an explicit ask-user handler) lives in
# test_assumption_registry_invariant.py, which owns it for the whole
# registry rather than just the hypothesis stage.


def test_no_hard_assumption_ever_resolves_to_borderline() -> None:
    """Section 7.2's rule makes only a hard FAIL blocking, so a hard
    BORDERLINE would be silently ignored. Hard assumptions are type, design
    and count checks, which have no borderline band -- this keeps it that
    way, so if one ever gains one it fails here instead of vanishing."""
    df = _clean_two_groups()
    spec = _validated(df)
    ctx = ResolutionContext(
        df=df, spec=spec, variables={"outcome": "value", "group": "group"}, cache=CheckCache(df)
    )
    hard_names: set[str] = set()
    for function_spec in registry.list(stage="hypothesis", kind="test"):
        hard_names |= set(function_spec.assumptions.hard)

    for name in sorted(hard_names):
        graded = resolve(name, ctx).graded
        assert all(c.status is not CheckStatus.BORDERLINE for c in graded), name


def test_every_family_offers_only_registered_functions() -> None:
    for family in ALL_FAMILIES:
        for method in family.methods:
            spec = registry.get(method.function)
            assert spec.kind == "test", f"{family.name} offers non-test {method.function}"


def test_rules_cover_every_goal_or_name_the_milestone() -> None:
    assert set(RULES) | set(PENDING_GOALS) == set(Goal)


# --------------------------------------------------------------------------
# Section 7.2's status rule
# --------------------------------------------------------------------------


def test_untestable_assumptions_never_block_but_are_always_surfaced() -> None:
    """Exchangeability cannot be checked, so it must not make a permutation
    test ineligible -- but it must appear on the candidate, because the user
    is the only one who can vouch for it."""
    df = _clean_two_groups()
    candidates = select_candidates(_validated(df), df)
    permutation = candidates.get("permutation_test_2s")
    assert permutation.eligibility is Eligibility.ELIGIBLE
    assert any("exchangeability" in reason for reason in permutation.reasons)


def test_independence_without_an_id_column_is_disclosed_on_every_candidate() -> None:
    """With no id column, independence rests on the user's word. Every
    candidate that assumes it says so."""
    df = _clean_two_groups()
    candidates = select_candidates(_validated(df), df)
    for candidate in candidates.candidates:
        assumptions = registry.get(candidate.function).assumptions
        if "independent" in assumptions.hard:
            assert any("independence.design" in r for r in candidate.reasons), candidate.function


def test_a_candidate_reason_always_names_a_check_that_is_in_the_evidence() -> None:
    """Section 5.3: `reasons` holds fact_ids "supporting the status". A
    reason pointing at a check the set does not carry would be unauditable."""
    df = _clean_two_groups()
    candidates = select_candidates(_validated(df), df)
    known = {check.fact_id for check in candidates.checks}
    for candidate in candidates.candidates:
        assert set(candidate.reasons) <= known, candidate.function


def test_ineligible_reasons_are_exactly_the_failed_hard_checks() -> None:
    rng = np.random.default_rng(51)
    df = pd.DataFrame({"value": rng.integers(1, 6, 60), "group": ["A"] * 30 + ["B"] * 30})
    spec = _spec().model_copy(update={"confirmed_by_user": {"design", "outcome"}})
    candidates = select_candidates(validate_spec(spec, df), df)
    student = candidates.get("student_t")
    assert student.eligibility is Eligibility.INELIGIBLE
    by_id = {c.fact_id: c for c in candidates.checks}
    blocking = [r for r in student.reasons if by_id[r].status is CheckStatus.FAIL]
    assert blocking, "an INELIGIBLE candidate must name the check that blocked it"


# --------------------------------------------------------------------------
# Ranking (Section 7.2)
# --------------------------------------------------------------------------


def test_ranking_puts_eligible_before_caveat_before_ineligible() -> None:
    rng = np.random.default_rng(52)
    df = pd.DataFrame(
        {
            "value": np.concatenate([rng.normal(10, 2, 50), rng.normal(12, 9, 20)]),
            "group": ["A"] * 50 + ["B"] * 20,
        }
    )
    candidates = select_candidates(_validated(df), df)
    order = [_rank_of(c.eligibility) for c in candidates.candidates]
    assert order == sorted(order)


def _rank_of(eligibility: Eligibility) -> int:
    return {Eligibility.ELIGIBLE: 0, Eligibility.CAVEAT: 1, Eligibility.INELIGIBLE: 2}[eligibility]


def test_fewer_caveats_ranks_higher_within_a_status() -> None:
    """Section 15.3's `heteroscedastic_groups` expectation, as a ranking
    property: Welch carries one caveat, Student two, so Welch comes first."""
    rng = np.random.default_rng(53)
    df = pd.DataFrame(
        {
            "value": np.concatenate([rng.exponential(10, 50), rng.exponential(25, 20)]),
            "group": ["A"] * 50 + ["B"] * 20,
        }
    )
    candidates = select_candidates(_validated(df), df)
    functions = [c.function for c in candidates.candidates]
    assert candidates.get("welch_t").eligibility is Eligibility.CAVEAT
    assert candidates.get("student_t").eligibility is Eligibility.CAVEAT
    assert functions.index("welch_t") < functions.index("student_t")


def test_ranking_is_stable_across_runs() -> None:
    """Rule 7: the same data must produce the same proposal every time."""
    df = _clean_two_groups()
    first = [c.function for c in select_candidates(_validated(df), df).candidates]
    second = [c.function for c in select_candidates(_validated(df), df).candidates]
    assert first == second


def test_robust_methods_outrank_equally_uncaveated_non_robust_ones() -> None:
    df = _clean_two_groups()
    candidates = select_candidates(_validated(df), df)
    eligible = [c.function for c in candidates.eligible]
    assert eligible.index("yuen_trimmed_t") < eligible.index("mann_whitney")


# --------------------------------------------------------------------------
# The cache (Section 7.2: "cached per data version")
# --------------------------------------------------------------------------


def test_a_check_is_computed_once_per_dataset_version_however_many_methods_ask() -> None:
    df = _clean_two_groups()
    cache = CheckCache(df)
    select_candidates(_validated(df), df, cache=cache)
    runs_after_first = cache.n_runs

    select_candidates(_validated(df), df, cache=cache)
    assert cache.n_runs == runs_after_first

    # Eight methods share one normality check and one variance check.
    distinct_checks = {key[1] for key in cache._cache}
    assert runs_after_first < len(distinct_checks) * 8


def test_changing_the_data_changes_the_version_so_nothing_stale_is_reused() -> None:
    df = _clean_two_groups()
    other = df.copy()
    other.loc[0, "value"] = other.loc[0, "value"] + 1000
    assert CheckCache(df).dataset_version != CheckCache(other).dataset_version


def test_an_explicit_dataset_version_is_carried_onto_the_result() -> None:
    """M6's DatasetStore will supply its own version id rather than a
    content hash; the engine records whichever it was given."""
    df = _clean_two_groups()
    candidates = select_candidates(_validated(df), df, dataset_version="v3-imputed")
    assert candidates.dataset_version == "v3-imputed"


# --------------------------------------------------------------------------
# CandidateSet shape
# --------------------------------------------------------------------------


def test_candidates_carry_their_params_tags_and_estimand() -> None:
    df = _clean_two_groups()
    welch = select_candidates(_validated(df), df).get("welch_t")
    assert welch.params == {"group_col": "group", "value_col": "value"}
    assert "parametric" in welch.tags
    assert "difference in means" in welch.estimand


def test_checks_are_deduplicated_and_sorted() -> None:
    df = _clean_two_groups()
    checks = select_candidates(_validated(df), df).checks
    fact_ids = [c.fact_id for c in checks]
    assert fact_ids == sorted(fact_ids)
    assert len(fact_ids) == len(set(fact_ids))
