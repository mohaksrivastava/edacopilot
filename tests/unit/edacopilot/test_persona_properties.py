"""Rule 2 as a property, and Section 7.4 exactly.

Section 16's M5 acceptance criteria are "worked example 7.4 reproduces
exactly" and "divergence detection correct". The first is pinned here
against the spec's own table. The second is covered by the property test:
across randomly generated data for every method family, no persona ever
proposes an INELIGIBLE candidate.

The property test is the one that matters. `pick` has several paths that
can reach a candidate -- the eligible pool, the least-caveated fallback,
two relaxation rules -- and a targeted test only proves the path it walks.
Hypothesis drives real data through every family instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from edacopilot.eligibility import (
    Design,
    Goal,
    QuestionSpec,
    UnsupportedQuestionError,
    select_candidates,
    validate_spec,
)
from edacopilot.personas import load_personas, pick, propose_with_rationales, render_card
from edacore.contracts import Eligibility
from tests.scenarios.generators import worked_example_7_4

# --------------------------------------------------------------------------
# Section 7.4, exactly
# --------------------------------------------------------------------------


def _worked_example_verdict():
    df = worked_example_7_4()
    spec = QuestionSpec(
        goal=Goal.COMPARE_GROUPS,
        variables={"outcome": "income", "group": "gender"},
        design=Design.INDEPENDENT,
        confirmed_by_user={"design"},
    )
    return propose_with_rationales(select_candidates(validate_spec(spec, df), df))


def test_worked_example_7_4_persona_picks() -> None:
    """Section 7.4: "Professor -> mann_whitney ... Consultant -> mann_whitney
    too ... Maverick -> yuen_trimmed_t"."""
    assert _worked_example_verdict().functions() == {
        "professor": "mann_whitney",
        "consultant": "mann_whitney",
        "maverick": "yuen_trimmed_t",
    }


def test_worked_example_7_4_is_a_divergence_card_with_two_proposals() -> None:
    """Section 7.4: "divergence card with two distinct proposals (Professor
    + Consultant merged, Maverick separate)"."""
    verdict = _worked_example_verdict()
    assert verdict.card == "divergence"
    assert not verdict.is_consensus
    assert len(verdict.proposals) == 2

    merged, separate = verdict.proposals
    assert merged.function == "mann_whitney"
    assert merged.personas == ["professor", "consultant"]
    assert separate.function == "yuen_trimmed_t"
    assert separate.personas == ["maverick"]


def test_worked_example_7_4_card_text_names_both_proposals() -> None:
    card = render_card(_worked_example_verdict())
    assert "Professor + Consultant" in card
    assert "Mann-Whitney" in card
    assert "Maverick" in card
    assert "Yuen" in card


def test_worked_example_7_4_consultants_clt_shortcut_does_not_fire() -> None:
    """Section 7.4's stated reason for the consultant's pick: its CLT rule
    does not apply here. Under Cochran's rule the skew of 1.89 needs
    n > 89, and there are 38 -- so the shortcut is unavailable and the
    consultant lands on the same rank-based method as the professor."""
    verdict = _worked_example_verdict()
    assert verdict.pick_for("consultant").relaxations == []


def test_worked_example_7_4_every_pick_has_a_rationale() -> None:
    for persona_pick in _worked_example_verdict().picks:
        assert persona_pick.rationale.endswith(".")
        assert persona_pick.candidate is not None


# --------------------------------------------------------------------------
# Rule 2 as a property
# --------------------------------------------------------------------------


@st.composite
def _group_data(draw: st.DrawFn) -> tuple[pd.DataFrame, QuestionSpec]:
    """Random two-or-more-group numeric data, with shape and size drawn.

    Drawing the *distribution* as well as the size is what makes this a
    real test: skew, heavy tails, tiny groups and unequal variances are
    exactly the conditions that push candidates into CAVEAT and
    INELIGIBLE, which is where a persona could go wrong.
    """
    seed = draw(st.integers(min_value=0, max_value=10_000))
    n_groups = draw(st.integers(min_value=2, max_value=4))
    n_per_group = draw(st.integers(min_value=3, max_value=60))
    shape = draw(st.sampled_from(["normal", "lognormal", "heavy_tailed", "uniform", "ordinal"]))
    spread = draw(st.sampled_from([(1.0, 1.0), (1.0, 8.0)]))

    rng = np.random.default_rng(seed)
    chunks = []
    for index in range(n_groups):
        scale = spread[index % len(spread)]
        if shape == "normal":
            chunks.append(rng.normal(10 + index, 2 * scale, n_per_group))
        elif shape == "lognormal":
            chunks.append(rng.lognormal(1 + 0.1 * index, 0.8 * scale, n_per_group))
        elif shape == "heavy_tailed":
            chunks.append(rng.standard_t(2, n_per_group) * scale + index)
        elif shape == "uniform":
            chunks.append(rng.uniform(0, 10 * scale, n_per_group))
        else:
            chunks.append(rng.integers(1, 6, n_per_group).astype(float))

    df = pd.DataFrame(
        {
            "value": np.concatenate(chunks),
            "group": sum(([f"g{i}"] * n_per_group for i in range(n_groups)), []),
            "subject": range(n_groups * n_per_group),
        }
    )
    spec = QuestionSpec(
        goal=Goal.COMPARE_GROUPS,
        variables={"outcome": "value", "group": "group", "subject": "subject"},
        design=Design.INDEPENDENT,
        confirmed_by_user={"design", "outcome"},
    )
    return df, spec


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_group_data())
def test_no_persona_ever_picks_an_ineligible_candidate(
    case: tuple[pd.DataFrame, QuestionSpec],
) -> None:
    """Rule 2. Whatever the data, whichever path `pick` takes."""
    df, spec = case
    validated = validate_spec(spec, df)
    if validated.ambiguities:
        return
    try:
        candidates = select_candidates(validated, df)
    except (UnsupportedQuestionError, ValueError):
        return

    ineligible = {
        c.function for c in candidates.candidates if c.eligibility is Eligibility.INELIGIBLE
    }
    for policy in load_personas():
        chosen = pick(policy, candidates)
        if chosen.candidate is None:
            continue
        assert chosen.candidate.function not in ineligible, (
            f"{policy.id} picked {chosen.candidate.function}, which is INELIGIBLE"
        )
        assert chosen.candidate.tags & policy.method_pool_tags
        assert chosen.eligibility is not Eligibility.INELIGIBLE


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_group_data())
def test_a_persona_never_relaxes_its_way_past_a_failing_hard_assumption(
    case: tuple[pd.DataFrame, QuestionSpec],
) -> None:
    """The relaxation rules touch soft assumptions only. If a persona
    reports ELIGIBLE, the engine must not have found a hard failure."""
    df, spec = case
    validated = validate_spec(spec, df)
    if validated.ambiguities:
        return
    try:
        candidates = select_candidates(validated, df)
    except (UnsupportedQuestionError, ValueError):
        return

    for policy in load_personas():
        chosen = pick(policy, candidates)
        if chosen.candidate is None or not chosen.relaxations:
            continue
        assert chosen.candidate.eligibility is Eligibility.CAVEAT, (
            f"{policy.id} relaxed a candidate that was "
            f"{chosen.candidate.eligibility.value}, not a caveat"
        )


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(_group_data())
def test_every_verdict_is_a_consensus_or_a_divergence_of_at_most_three(
    case: tuple[pd.DataFrame, QuestionSpec],
) -> None:
    df, spec = case
    validated = validate_spec(spec, df)
    if validated.ambiguities:
        return
    try:
        candidates = select_candidates(validated, df)
    except (UnsupportedQuestionError, ValueError):
        return

    verdict = propose_with_rationales(candidates)
    assert 1 <= len(verdict.proposals) <= 3
    assert verdict.card == ("consensus" if len(verdict.proposals) == 1 else "divergence")
    assert sum(len(p.personas) for p in verdict.proposals) == 3
    assert render_card(verdict)


# --------------------------------------------------------------------------
# Every family, not just two independent groups
# --------------------------------------------------------------------------


def _family_cases() -> list[tuple[str, pd.DataFrame, QuestionSpec]]:
    """One well-formed question per family the personas can serve."""
    rng = np.random.default_rng(7)
    cases: list[tuple[str, pd.DataFrame, QuestionSpec]] = []

    paired_base = rng.normal(100, 15, 30)
    paired = pd.DataFrame(
        {
            "subject": np.tile(np.arange(30), 2),
            "group": ["pre"] * 30 + ["post"] * 30,
            "value": np.concatenate([paired_base, paired_base + rng.normal(5, 4, 30)]),
        }
    )
    cases.append(
        (
            "two_paired_numeric",
            paired,
            QuestionSpec(
                goal=Goal.COMPARE_GROUPS,
                variables={"outcome": "value", "group": "group", "subject": "subject"},
                design=Design.PAIRED,
                confirmed_by_user={"design"},
            ),
        )
    )

    k_groups = pd.DataFrame(
        {
            "value": np.concatenate([rng.normal(m, 8, 25) for m in (50, 55, 62)]),
            "group": sum(([label] * 25 for label in "ABC"), []),
        }
    )
    cases.append(
        (
            "k_independent_numeric",
            k_groups,
            QuestionSpec(
                goal=Goal.COMPARE_GROUPS,
                variables={"outcome": "value", "group": "group"},
                design=Design.INDEPENDENT,
                confirmed_by_user={"design"},
            ),
        )
    )

    base = rng.normal(50, 8, 20)
    repeated = pd.DataFrame(
        {
            "subject": np.tile(np.arange(20), 3),
            "condition": np.repeat(["t1", "t2", "t3"], 20),
            "value": np.concatenate([base + s + rng.normal(0, 3, 20) for s in (0, 5, 9)]),
        }
    )
    cases.append(
        (
            "k_repeated_numeric",
            repeated,
            QuestionSpec(
                goal=Goal.COMPARE_GROUPS,
                variables={"outcome": "value", "group": "condition", "subject": "subject"},
                design=Design.REPEATED,
                confirmed_by_user={"design"},
            ),
        )
    )

    x = rng.normal(0, 1, 90)
    numeric_pair = pd.DataFrame({"x": x, "y": 0.6 * x + rng.normal(0, 0.9, 90)})
    cases.append(
        (
            "numeric_numeric",
            numeric_pair,
            QuestionSpec(
                goal=Goal.ASSOCIATION,
                variables={"x": "x", "y": "y"},
                design=Design.INDEPENDENT,
                confirmed_by_user={"design", "x", "y"},
            ),
        )
    )

    binary = pd.DataFrame(
        {
            "converted": np.concatenate([rng.binomial(1, 0.35, 100), rng.binomial(1, 0.5, 100)]),
            "arm": ["control"] * 100 + ["treatment"] * 100,
        }
    )
    cases.append(
        (
            "binary_outcome_groups",
            binary,
            QuestionSpec(
                goal=Goal.COMPARE_GROUPS,
                variables={"outcome": "converted", "group": "arm"},
                design=Design.INDEPENDENT,
                confirmed_by_user={"design"},
            ),
        )
    )

    one_sample = pd.DataFrame({"weight": rng.normal(70, 8, 60)})
    cases.append(
        (
            "one_sample_numeric",
            one_sample,
            QuestionSpec(
                goal=Goal.DISTRIBUTION_FIT,
                variables={"variable": "weight"},
                reference_value=68.0,
                confirmed_by_user={"variable"},
            ),
        )
    )
    return cases


@pytest.mark.parametrize("case", _family_cases(), ids=lambda c: c[0])
def test_rule_two_holds_in_every_family(
    case: tuple[str, pd.DataFrame, QuestionSpec],
) -> None:
    _, df, spec = case
    candidates = select_candidates(validate_spec(spec, df), df)
    ineligible = {
        c.function for c in candidates.candidates if c.eligibility is Eligibility.INELIGIBLE
    }
    verdict = propose_with_rationales(candidates)
    for persona_pick in verdict.picks:
        assert persona_pick.function not in ineligible
        assert persona_pick.rationale


@pytest.mark.parametrize("case", _family_cases(), ids=lambda c: c[0])
def test_every_family_produces_a_renderable_card(
    case: tuple[str, pd.DataFrame, QuestionSpec],
) -> None:
    _, df, spec = case
    verdict = propose_with_rationales(select_candidates(validate_spec(spec, df), df))
    card = render_card(verdict)
    assert card and card.endswith(".")
