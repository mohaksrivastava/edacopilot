"""Practical equivalence (8.3) and the Maverick's proposal policy (8.1).

Both features exist to stop a divergence card appearing where the personas
do not actually disagree. A card the user learns to dismiss is worse than no
card, because the one time it matters it will be dismissed too.

Both are also collapse-only, and that is what most of this file asserts: a
rule may turn a divergence into a consensus and never the reverse, and it
may not fire when the evidence for it is missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

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
from edacopilot.personas.policy import (
    PERSONA_DIR,
    EquivalenceRule,
    PersonaPolicy,
    load_equivalences,
)
from edacore.contracts import Candidate, CheckStatus, Eligibility


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


def _unequal_variance(seed: int = 14) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "value": np.concatenate([rng.normal(50, 3, 60), rng.normal(53, 18, 60)]),
            "group": ["A"] * 60 + ["B"] * 60,
        }
    )


# --------------------------------------------------------------------------
# The rules themselves
# --------------------------------------------------------------------------


def test_the_shipped_rules_load_and_are_conditional() -> None:
    """Every rule must carry a condition. An unconditional one could merge
    methods that genuinely disagree."""
    rules = load_equivalences()
    assert rules, "equivalences.yaml should ship at least the student/welch rule"
    for rule in rules:
        assert rule.when_passes
        assert len(rule.functions) >= 2
        assert rule.note


def test_the_student_welch_rule_is_present_as_specified() -> None:
    rule = next(r for r in load_equivalences() if r.id == "student_vs_welch")
    assert rule.functions == frozenset({"student_t", "welch_t"})
    assert rule.when_passes == "equal_variance"


def test_a_rule_without_a_condition_is_rejected() -> None:
    with pytest.raises(ValidationError):
        EquivalenceRule.model_validate(
            {"id": "x", "functions": ["student_t", "welch_t"], "note": "n"}
        )


def test_a_rule_needs_at_least_two_functions() -> None:
    with pytest.raises(ValidationError):
        EquivalenceRule.model_validate(
            {"id": "x", "functions": ["student_t"], "when_passes": "equal_variance", "note": "n"}
        )


def test_equivalences_yaml_is_not_mistaken_for_a_persona() -> None:
    assert [p.id for p in load_personas()] == ["professor", "consultant", "maverick"]


# --------------------------------------------------------------------------
# Collapsing, and refusing to collapse
# --------------------------------------------------------------------------


def _fake(persona: str, function: str) -> PersonaPick:
    return PersonaPick(
        persona=persona,
        display_name=get_persona(persona).display_name,
        candidate=Candidate(function=function, eligibility=Eligibility.ELIGIBLE, estimand="x"),
        eligibility=Eligibility.ELIGIBLE,
    )


def _variance_check(status: CheckStatus):
    from edacore.contracts import AssumptionCheck

    return AssumptionCheck(
        fact_id="variance.levene.value.group",
        assumption="equal_variance",
        method="levene",
        threshold="p < 0.05 -> fail",
        status=status,
        consequence="unequal variances distort a pooled test",
    )


def test_student_and_welch_collapse_when_equal_variance_passes() -> None:
    verdict = detect_divergence(
        [_fake("professor", "student_t"), _fake("consultant", "welch_t")],
        [_variance_check(CheckStatus.PASS)],
    )
    assert verdict.is_consensus
    assert len(verdict.proposals) == 1
    proposal = verdict.proposals[0]
    assert proposal.is_equivalence
    assert proposal.equivalent_methods == {"professor": "student_t", "consultant": "welch_t"}
    assert "same answer" in proposal.equivalence_note


@pytest.mark.parametrize("status", [CheckStatus.FAIL, CheckStatus.BORDERLINE])
def test_they_do_not_collapse_when_equal_variance_does_not_pass(status: CheckStatus) -> None:
    """The case the condition exists for: when the variances differ, the
    two tests genuinely disagree and the user should see both."""
    verdict = detect_divergence(
        [_fake("professor", "student_t"), _fake("consultant", "welch_t")],
        [_variance_check(status)],
    )
    assert not verdict.is_consensus
    assert len(verdict.proposals) == 2


def test_an_absent_check_does_not_satisfy_a_rule() -> None:
    """Not evaluated is not the same as passed. If nothing checked the
    variances, there is no evidence the two methods agree."""
    verdict = detect_divergence(
        [_fake("professor", "student_t"), _fake("consultant", "welch_t")], []
    )
    assert not verdict.is_consensus


def test_a_rule_never_splits_an_existing_agreement() -> None:
    """Collapse-only. Two personas naming the same method stay one
    proposal whatever the rules say."""
    verdict = detect_divergence(
        [_fake("professor", "welch_t"), _fake("consultant", "welch_t")],
        [_variance_check(CheckStatus.PASS)],
    )
    assert verdict.is_consensus
    assert not verdict.proposals[0].is_equivalence


def test_an_unrelated_method_is_not_swept_into_an_equivalence() -> None:
    verdict = detect_divergence(
        [
            _fake("professor", "student_t"),
            _fake("consultant", "welch_t"),
            _fake("maverick", "yuen_trimmed_t"),
        ],
        [_variance_check(CheckStatus.PASS)],
    )
    assert len(verdict.proposals) == 2
    collapsed = verdict.proposals[0]
    assert collapsed.is_equivalence
    assert verdict.proposals[1].function == "yuen_trimmed_t"


def test_the_consensus_card_names_each_method_and_why_they_agree() -> None:
    """Section 8.3's requirement for an equivalence: "a one-line note
    naming each persona's method and why the conclusion is the same"."""
    verdict = detect_divergence(
        [_fake("professor", "student_t"), _fake("consultant", "welch_t")],
        [_variance_check(CheckStatus.PASS)],
    )
    card = render_card(verdict)
    assert "Professor" in card and "Student's t-test" in card
    assert "Consultant" in card and "Welch's t-test" in card
    assert "same answer" in card


def test_equivalence_end_to_end_on_clean_data() -> None:
    """The real path: clean data, so the professor picks Student's t and
    the consultant picks Welch, and the card is one consensus."""
    verdict = propose_with_rationales(_candidates(_clean()))
    assert verdict.is_consensus
    assert verdict.pick_for("professor").function == "student_t"
    assert verdict.pick_for("consultant").function == "welch_t"
    assert verdict.proposals[0].is_equivalence


def test_no_equivalence_when_variances_actually_differ() -> None:
    verdict = propose_with_rationales(_candidates(_unequal_variance()))
    assert not verdict.is_consensus


# --------------------------------------------------------------------------
# The Maverick's proposal policy
# --------------------------------------------------------------------------


def test_the_maverick_concurs_when_nothing_is_compromised() -> None:
    verdict = propose_with_rationales(_candidates(_clean()))
    maverick = verdict.pick_for("maverick")
    assert maverick.concurs_with == "professor"
    assert maverick.function == verdict.pick_for("professor").function
    assert "without adding information" in maverick.concur_note
    assert "concurs with Professor" in maverick.rationale


def test_the_maverick_proposes_when_a_soft_assumption_is_not_met() -> None:
    from tests.scenarios.generators import heavy_tails_small_n

    verdict = propose_with_rationales(_candidates(heavy_tails_small_n()))
    maverick = verdict.pick_for("maverick")
    assert maverick.concurs_with is None
    assert maverick.function == "yuen_trimmed_t"


def test_the_maverick_concurs_rather_than_offering_nothing() -> None:
    """Its old behaviour on an ordinal association question was "no method
    to offer", which reads as absence. Concurring says it looked."""
    from tests.scenarios.generators import ordinal_as_numeric

    df = ordinal_as_numeric()
    spec = QuestionSpec(
        goal=Goal.ASSOCIATION,
        variables={"x": "satisfaction", "y": "tenure_months"},
        design=Design.INDEPENDENT,
        confirmed_by_user={"design", "x", "y"},
    )
    verdict = propose_with_rationales(select_candidates(validate_spec(spec, df), df))
    maverick = verdict.pick_for("maverick")
    assert maverick.candidate is not None
    assert maverick.concurs_with == "professor"
    assert "method pool" in maverick.concur_note


def test_only_the_maverick_has_a_proposal_policy() -> None:
    with_policy = [p.id for p in load_personas() if p.proposal_policy is not None]
    assert with_policy == ["maverick"]


def test_a_persona_deferring_to_itself_is_rejected(tmp_path) -> None:
    raw = yaml.safe_load((PERSONA_DIR / "maverick.yaml").read_text(encoding="utf-8"))
    raw["proposal_policy"]["otherwise_concur_with"] = "nobody"
    with pytest.raises(ValidationError):
        PersonaPolicy.model_validate(raw | {"proposal_policy": {"propose_when": []}})


def test_pick_without_prior_picks_still_proposes() -> None:
    """`pick` must stay usable standalone: with nobody to defer to, the
    persona proposes its own method rather than returning nothing."""
    candidates = _candidates(_clean())
    standalone = pick(get_persona("maverick"), candidates)
    assert standalone.candidate is not None
    assert standalone.concurs_with is None
    assert standalone.function == "yuen_trimmed_t"


def test_concurring_never_yields_an_ineligible_candidate() -> None:
    """Rule 2 again, through the new path: a persona that defers inherits
    the other's candidate, which must still be one the engine allowed."""
    rng = np.random.default_rng(51)
    df = pd.DataFrame({"value": rng.integers(1, 6, 60), "group": ["A"] * 30 + ["B"] * 30})
    candidates = _candidates(df, confirmed_by_user={"design", "outcome"})
    ineligible = {
        c.function for c in candidates.candidates if c.eligibility is Eligibility.INELIGIBLE
    }
    for persona_pick in propose(candidates).picks:
        assert persona_pick.function not in ineligible


def test_seven_four_is_still_a_two_proposal_divergence() -> None:
    """The acceptance criterion M5.1 must not break."""
    from tests.scenarios.generators import worked_example_7_4

    df = worked_example_7_4()
    spec = QuestionSpec(
        goal=Goal.COMPARE_GROUPS,
        variables={"outcome": "income", "group": "gender"},
        design=Design.INDEPENDENT,
        confirmed_by_user={"design"},
    )
    verdict = propose_with_rationales(select_candidates(validate_spec(spec, df), df))
    assert verdict.card == "divergence"
    assert len(verdict.proposals) == 2
    assert verdict.functions() == {
        "professor": "mann_whitney",
        "consultant": "mann_whitney",
        "maverick": "yuen_trimmed_t",
    }
    assert verdict.pick_for("maverick").concurs_with is None
