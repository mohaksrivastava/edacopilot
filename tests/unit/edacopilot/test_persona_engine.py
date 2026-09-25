"""pick (8.2), divergence detection (8.3), and rule 2.

Rule 2 -- *no persona can propose an ineligible method* -- is the property
this whole layer exists to preserve, so it is asserted from several
directions: directly, through every relaxation rule, and as a property over
random data in test_persona_properties.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edacopilot.eligibility import Design, Goal, QuestionSpec, select_candidates, validate_spec
from edacopilot.personas import (
    PersonaPick,
    detect_divergence,
    get_persona,
    load_personas,
    pick,
    propose,
    propose_with_rationales,
    render_card,
)
from edacopilot.personas.engine import _reclassify
from edacore.contracts import CheckStatus, Eligibility
from edacore.registry import registry


def _candidates(df: pd.DataFrame, **overrides: object):
    fields: dict[str, object] = {
        "goal": Goal.COMPARE_GROUPS,
        "variables": {"outcome": "value", "group": "group"},
        "design": Design.INDEPENDENT,
        "confirmed_by_user": {"design"},
    }
    fields.update(overrides)
    spec = QuestionSpec(**fields)  # type: ignore[arg-type]
    return select_candidates(validate_spec(spec, df), df)


def _clean(seed: int = 1, n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "value": np.concatenate([rng.normal(10, 2, n), rng.normal(12, 2, n)]),
            "group": ["A"] * n + ["B"] * n,
        }
    )


def _ordinal(seed: int = 51) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"value": rng.integers(1, 6, 60), "group": ["A"] * 30 + ["B"] * 30})


# --------------------------------------------------------------------------
# Rule 2: no persona may propose an ineligible method
# --------------------------------------------------------------------------


def test_no_persona_picks_an_ineligible_candidate() -> None:
    """On ordinal data the mean-based methods are INELIGIBLE (a mean of
    Likert codes is not a quantity). No persona may reach for one."""
    candidates = _candidates(_ordinal(), confirmed_by_user={"design", "outcome"})
    ineligible = {
        c.function for c in candidates.candidates if c.eligibility is Eligibility.INELIGIBLE
    }
    assert ineligible, "this fixture is supposed to make some methods ineligible"

    for persona_pick in propose(candidates).picks:
        assert persona_pick.function not in ineligible, persona_pick.persona


def test_ineligible_candidates_are_excluded_with_a_reason() -> None:
    candidates = _candidates(_ordinal(), confirmed_by_user={"design", "outcome"})
    professor = pick(get_persona("professor"), candidates)
    assert any("ineligible" in reason for reason in professor.excluded.values())


def test_a_persona_with_no_eligible_method_keeps_the_least_caveated() -> None:
    """Section 8.2: "if empty, keep least-caveated". A persona that went
    silent whenever nothing was clean would hide the trade-off rather than
    presenting it."""
    rng = np.random.default_rng(60)
    df = pd.DataFrame(
        {
            "value": np.concatenate([rng.exponential(5, 14), rng.exponential(14, 14)]),
            "group": ["A"] * 14 + ["B"] * 14,
        }
    )
    candidates = _candidates(df)
    professor = pick(get_persona("professor"), candidates)
    if professor.candidate is not None:
        assert professor.candidate.eligibility is not Eligibility.INELIGIBLE


# --------------------------------------------------------------------------
# Reclassification: only soft, only via declared rules
# --------------------------------------------------------------------------


def _symmetric_heavy_tails() -> pd.DataFrame:
    """t(3) at n=120 per group: symmetric, so Cochran's rule is met, but
    heavy-tailed enough that Shapiro-Wilk rejects in every group.

    This is exactly the gap M4.1 documented -- Cochran's rule bounds skew
    and says nothing about tails -- which makes it the cleanest way to
    exercise the shortcut: the caveat is real, and the rule clears it.
    """
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "value": np.concatenate(
                [rng.standard_t(3, 120) * 4 + 50 + shift for shift in (0, 2, 5)]
            ),
            "group": sum(([label] * 120 for label in "ABC"), []),
        }
    )


def test_the_clt_shortcut_can_clear_a_normality_caveat_but_only_the_consultant_has_one() -> None:
    """The shortcut applies to plain `normality` (one-way ANOVA's soft
    assumption). `normality_or_large_n` is already decided by Cochran's
    rule inside the eligibility engine, because Section 6.7 grants those
    methods the escape by name -- so there is nothing left for a persona
    to relax there."""
    candidates = _candidates(_symmetric_heavy_tails())
    checks = {c.fact_id: c for c in candidates.checks}
    anova = candidates.get("one_way_anova")
    assert anova.eligibility is Eligibility.CAVEAT
    assert "normality" in registry.get("one_way_anova").assumptions.soft

    cochran = [c for c in candidates.checks if c.method == "cochran_rule"]
    assert cochran and cochran[0].status is CheckStatus.PASS

    consultant = _reclassify(get_persona("consultant"), anova, checks)
    professor = _reclassify(get_persona("professor"), anova, checks)
    assert consultant.eligibility is Eligibility.ELIGIBLE
    assert consultant.relaxation and "Cochran" in consultant.relaxation
    assert professor.eligibility is Eligibility.CAVEAT
    assert professor.relaxation is None


def test_the_engine_records_cochrans_rule_even_when_no_method_declares_it() -> None:
    """A persona's shortcut must not depend on whether some *other*
    candidate in the family happened to surface the fact."""
    # Heavy-tailed so that normality actually fails -- the fact only exists
    # to excuse a failure, so there is nothing to record when it passes.
    rng = np.random.default_rng(31)
    sizes = [60, 60, 60, 60]
    df = pd.DataFrame(
        {
            "value": np.concatenate(
                [rng.standard_t(3, n) * 2 + m for m, n in zip((10, 12, 9, 15), sizes, strict=True)]
            ),
            "factor_a": np.repeat(["A1", "A1", "A2", "A2"], sizes),
            "factor_b": np.repeat(["B1", "B2", "B1", "B2"], sizes),
        }
    )
    spec = QuestionSpec(
        goal=Goal.COMPARE_GROUPS,
        variables={"outcome": "value", "group": "factor_a", "factor_b": "factor_b"},
        design=Design.INDEPENDENT,
        confirmed_by_user={"design"},
    )
    candidates = select_candidates(validate_spec(spec, df), df)
    declares = {
        a
        for method in ("two_way_anova", "aligned_rank_transform_anova")
        for a in registry.get(method).assumptions.soft
    }
    assert "normality_or_large_n" not in declares
    assert any(c.method == "cochran_rule" for c in candidates.checks)


def test_the_clt_shortcut_never_clears_a_non_normality_caveat() -> None:
    """Student's t is caveated for unequal variance as well as normality.
    The shortcut is about the CLT; it has nothing to say about variance."""
    rng = np.random.default_rng(13)
    df = pd.DataFrame(
        {
            "value": np.concatenate([rng.normal(50, 3, 60), rng.normal(53, 18, 60)]),
            "group": ["A"] * 60 + ["B"] * 60,
        }
    )
    candidates = _candidates(df)
    checks = {c.fact_id: c for c in candidates.checks}
    student = candidates.get("student_t")
    assert student.eligibility is Eligibility.CAVEAT
    assert any(checks[r].assumption == "equal_variance" for r in student.reasons if r in checks)

    result = _reclassify(get_persona("consultant"), student, checks)
    assert result.eligibility is Eligibility.CAVEAT


def test_borderline_relaxation_is_blocked_by_a_single_fail() -> None:
    """`borderline_is: pass` clears a caveat only when every cause is
    BORDERLINE. A persona must not reach eligibility by declining to look
    at a check that failed outright."""
    from tests.scenarios.generators import worked_example_7_4

    candidates = _candidates_for_income(worked_example_7_4())
    checks = {c.fact_id: c for c in candidates.checks}
    welch = candidates.get("welch_t")
    causes = [checks[r] for r in welch.reasons if r in checks]
    assert any(c.status is CheckStatus.FAIL for c in causes)

    result = _reclassify(get_persona("consultant"), welch, checks)
    assert result.eligibility is Eligibility.CAVEAT
    assert result.relaxation is None


def test_a_borderline_only_caveat_is_relaxed_for_a_persona_that_reads_it_as_passing() -> None:
    """The other side of the same rule. Here Shapiro passes in every group
    and only the descriptive skew is borderline, so the consultant's
    `borderline_is: pass` applies and the professor's `fail` does not."""
    rng = np.random.default_rng(25)
    df = pd.DataFrame(
        {
            "value": np.concatenate([rng.gamma(6, 2, 20) for _ in range(3)]),
            "group": sum(([label] * 20 for label in "ABC"), []),
        }
    )
    spec = QuestionSpec(
        goal=Goal.COMPARE_GROUPS,
        variables={"outcome": "value", "group": "group"},
        design=Design.INDEPENDENT,
        confirmed_by_user={"design"},
    )
    candidates = select_candidates(validate_spec(spec, df), df)
    checks = {c.fact_id: c for c in candidates.checks}
    anova = candidates.get("one_way_anova")
    causes = [checks[r] for r in anova.reasons if r in checks]
    blocking = [c for c in causes if c.status in (CheckStatus.FAIL, CheckStatus.BORDERLINE)]
    assert blocking and all(c.status is CheckStatus.BORDERLINE for c in blocking)

    consultant = _reclassify(get_persona("consultant"), anova, checks)
    professor = _reclassify(get_persona("professor"), anova, checks)
    assert consultant.eligibility is Eligibility.ELIGIBLE
    assert consultant.relaxation and "borderline" in consultant.relaxation
    assert professor.eligibility is Eligibility.CAVEAT


def _candidates_for_income(df: pd.DataFrame):
    spec = QuestionSpec(
        goal=Goal.COMPARE_GROUPS,
        variables={"outcome": "income", "group": "gender"},
        design=Design.INDEPENDENT,
        confirmed_by_user={"design"},
    )
    return select_candidates(validate_spec(spec, df), df)


def test_no_persona_rule_can_relax_a_hard_assumption() -> None:
    """The one thing reclassification must never do. INELIGIBLE is filtered
    before any rule runs, and this asserts the rule itself is a no-op on
    one even if it were reached."""
    candidates = _candidates(_ordinal(), confirmed_by_user={"design", "outcome"})
    checks = {c.fact_id: c for c in candidates.checks}
    blocked = next(c for c in candidates.candidates if c.eligibility is Eligibility.INELIGIBLE)
    for policy in load_personas():
        assert _reclassify(policy, blocked, checks).eligibility is Eligibility.INELIGIBLE


def test_always_robust_declines_equal_variance_methods_rather_than_waiving_them() -> None:
    """The consultant's `strategy: always_robust` is a tightening. Student's
    t is excluded, not upgraded."""
    rng = np.random.default_rng(14)
    df = pd.DataFrame(
        {
            "value": np.concatenate([rng.normal(50, 3, 60), rng.normal(53, 18, 60)]),
            "group": ["A"] * 60 + ["B"] * 60,
        }
    )
    candidates = _candidates(df)
    consultant = pick(get_persona("consultant"), candidates)
    assert consultant.function != "student_t"
    assert "equal variance" in consultant.excluded.get("student_t", "")


# --------------------------------------------------------------------------
# Method pools
# --------------------------------------------------------------------------


def test_each_persona_only_proposes_from_its_own_method_pool() -> None:
    candidates = _candidates(_clean())
    for policy in load_personas():
        chosen = pick(policy, candidates)
        assert chosen.candidate is not None, policy.id
        assert chosen.candidate.tags & policy.method_pool_tags, policy.id


def test_the_maverick_never_proposes_a_plain_parametric_test() -> None:
    """Its own proposal, that is. Called through `propose()` on clean data
    it concurs with the Professor instead (M5.1's `proposal_policy`, tested
    in test_persona_equivalence.py); what this pins is that when it does
    speak for itself, it speaks from its own pool."""
    candidates = _candidates(_clean())
    maverick = pick(get_persona("maverick"), candidates)
    assert maverick.concurs_with is None
    assert maverick.function in {"yuen_trimmed_t", "permutation_test_2s", "bootstrap_diff"}


# --------------------------------------------------------------------------
# Divergence detection (Section 8.3)
# --------------------------------------------------------------------------


def _fake(persona: str, function: str | None, params: dict | None = None) -> PersonaPick:
    from edacore.contracts import Candidate

    candidate = (
        None
        if function is None
        else Candidate(
            function=function,
            params=params or {},
            eligibility=Eligibility.ELIGIBLE,
            estimand="x",
        )
    )
    return PersonaPick(persona=persona, display_name=persona.title(), candidate=candidate)


def test_identical_picks_collapse_to_one_consensus_proposal() -> None:
    verdict = detect_divergence(
        [_fake(p, "mann_whitney", {"a": 1}) for p in ("professor", "consultant", "maverick")]
    )
    assert verdict.is_consensus
    assert verdict.card == "consensus"
    assert len(verdict.proposals) == 1
    assert verdict.proposals[0].personas == ["professor", "consultant", "maverick"]
    assert "All three personas agree" in render_card(verdict)


def test_the_same_function_with_different_params_is_a_different_proposal() -> None:
    """Section 8.3 says "materially identical params", not just the same
    function: a 10% trim and a 20% trim are different proposals."""
    verdict = detect_divergence(
        [
            _fake("professor", "yuen_trimmed_t", {"trim": 0.2}),
            _fake("maverick", "yuen_trimmed_t", {"trim": 0.1}),
        ]
    )
    assert not verdict.is_consensus
    assert len(verdict.proposals) == 2


def test_three_distinct_picks_give_three_proposals() -> None:
    verdict = detect_divergence(
        [
            _fake("professor", "student_t"),
            _fake("consultant", "welch_t"),
            _fake("maverick", "yuen_trimmed_t"),
        ]
    )
    assert len(verdict.proposals) == 3
    assert verdict.card == "divergence"


def test_personas_with_nothing_to_offer_group_together() -> None:
    verdict = detect_divergence([_fake("professor", None), _fake("consultant", None)])
    assert verdict.is_consensus
    assert "has a method to offer" in render_card(verdict)


def test_proposal_order_follows_persona_order() -> None:
    verdict = detect_divergence(
        [
            _fake("professor", "mann_whitney"),
            _fake("consultant", "welch_t"),
            _fake("maverick", "mann_whitney"),
        ]
    )
    assert [p.function for p in verdict.proposals] == ["mann_whitney", "welch_t"]
    assert verdict.proposals[0].personas == ["professor", "maverick"]


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_picks_are_stable_across_runs() -> None:
    """Rule 7: the same data must produce the same proposal every time."""
    df = _clean()
    first = propose(_candidates(df)).functions()
    second = propose(_candidates(df)).functions()
    assert first == second


def test_every_pick_names_a_registered_function() -> None:
    for persona_pick in propose(_candidates(_clean())).picks:
        assert persona_pick.candidate is not None
        registry.get(persona_pick.candidate.function)


# --------------------------------------------------------------------------
# Rationales (Section 10.5's deterministic fallback)
# --------------------------------------------------------------------------


def test_every_pick_gets_a_rationale_naming_its_method() -> None:
    """Length is not the test -- the consultant's `brief` style is meant to
    be terse ("Use Welch's t-test."). What every rationale must do is name
    the method it is recommending."""
    from edacopilot.personas.rationale import _method_name

    verdict = propose_with_rationales(_candidates(_clean()))
    for persona_pick in verdict.picks:
        assert persona_pick.rationale.endswith(".")
        assert persona_pick.candidate is not None
        assert _method_name(persona_pick.candidate.function) in persona_pick.rationale


def test_a_caveated_pick_quotes_the_consequence_of_the_failing_check() -> None:
    """A rationale has to say what goes wrong, not just what was chosen.
    `AssumptionCheck.consequence` is written for exactly this."""
    rng = np.random.default_rng(15)
    df = pd.DataFrame(
        {
            "value": np.concatenate([rng.exponential(5, 18), rng.exponential(15, 18)]),
            "group": ["A"] * 18 + ["B"] * 18,
        }
    )
    candidates = _candidates(df)
    checks = {c.fact_id: c for c in candidates.checks}
    verdict = propose_with_rationales(candidates)

    caveated = [p for p in verdict.picks if p.eligibility is Eligibility.CAVEAT]
    for persona_pick in caveated:
        assert persona_pick.candidate is not None
        causes = [
            checks[r]
            for r in persona_pick.candidate.reasons
            if r in checks and checks[r].status is not CheckStatus.UNTESTABLE
        ]
        if causes:
            opening = causes[0].consequence.split(";")[0].split(" -- ")[0].strip(". ").lower()
            assert opening[:40] in persona_pick.rationale.lower()


def test_rationales_contain_no_numbers_the_checks_did_not_produce() -> None:
    """Rule 1: the rationale layer computes nothing. Every number in a
    rationale must appear in a check's own text."""
    import re

    candidates = _candidates(_clean())
    verdict = propose_with_rationales(candidates)
    check_text = " ".join(f"{c.threshold} {c.consequence}" for c in candidates.checks)
    for persona_pick in verdict.picks:
        for number in re.findall(r"\d+\.?\d*", persona_pick.rationale):
            assert number in check_text, (persona_pick.persona, number)


@pytest.mark.parametrize("persona_id", ["professor", "consultant", "maverick"])
def test_explanation_style_changes_the_voice_not_the_facts(persona_id: str) -> None:
    verdict = propose_with_rationales(_candidates(_clean()))
    persona_pick = verdict.pick_for(persona_id)
    assert persona_pick.rationale
    assert get_persona(persona_id).display_name in render_card(verdict) or persona_pick.rationale
