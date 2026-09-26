"""The three properties M7 must not be able to lose.

These are not tests of behaviour so much as tests of the shape of the
system. Each corresponds to a rule that, if it broke, would break quietly:
the transcripts would still render, the cards would still say sensible
things, and the system would be doing something it promises not to do.

1. **Nothing executes without an explicit accept or override** (rule 3).
   Property-tested over random action sequences, because the risk is not
   that `ask()` obviously runs a test — it is that some path added later
   (a suggestion followed automatically, a stage that "helpfully" finishes
   a question) quietly does.
2. **The design question is asked before any persona card**, whenever
   repeated ids are detected (Section 7.1). This is the
   `paired_as_independent` trap and the single most consequential
   confirmation in the product.
3. **An INELIGIBLE override never completes without both a typed reason
   and a second confirmation** (Section 7.5).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from edacopilot.orchestrator import IntentType, Stage
from edacopilot.orchestrator.loop import OverrideRequired
from edacopilot.session import Session
from edacore.contracts import Eligibility
from tests.scenarios.generators import ordinal_as_numeric, paired_as_independent


def _frame(seed: int = 5, n: int = 45) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "value": np.concatenate([rng.normal(10, 2, n), rng.normal(11, 3, n)]),
            "group": ["A"] * n + ["B"] * n,
        }
    )


@pytest.fixture
def session(tmp_path: Path) -> Session:
    return Session.start(_frame(), session_id="inv", root=tmp_path)


# --------------------------------------------------------------------------
# 1. Nothing executes without an explicit accept/override (rule 3)
# --------------------------------------------------------------------------

# Every action a user can take that is *not* an acceptance. Each is a
# (name, callable) pair so a failure names the action that broke the rule.
READ_ONLY_ACTIONS: dict[str, object] = {
    "ask_structured": lambda s: s.ask(
        goal="compare_groups",
        outcome="value",
        group="group",
        design="independent",
        confirmed_by_user=["design"],
    ),
    "ask_free_text": lambda s: s.ask("is value different between the groups?"),
    "ask_nothing": lambda s: s.ask(),
    "ask_nonsense": lambda s: s.ask("wibble the frobnicator"),
    "explain_term": lambda s: s.explain("sphericity"),
    "explain_method": lambda s: s.explain("welch_t"),
    "explain_unknown": lambda s: s.explain("quux"),
    "goto_profile": lambda s: s.goto_stage("profile"),
    "goto_hypothesis": lambda s: s.goto_stage("hypothesis"),
    "goto_missingness": lambda s: s.goto_stage("missingness"),
    "goto_nonexistent": lambda s: s.goto_stage("nowhere"),
    "skip": lambda s: s.skip(),
    "modify_alpha": lambda s: s.modify(alpha=0.01),
    "ledger": lambda s: s.orchestrator.ledger_card(),
    "show": lambda s: s.ask("plot value"),
    "answer": lambda s: s.answer(design="independent"),
    "post_hoc": lambda s: s.ask("run post-hoc comparisons"),
}


@settings(
    max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(actions=st.lists(st.sampled_from(sorted(READ_ONLY_ACTIONS)), min_size=1, max_size=8))
def test_no_sequence_of_non_accepting_actions_creates_a_step(
    tmp_path_factory: pytest.TempPathFactory, actions: list[str]
) -> None:
    """Rule 3, as a property over random action sequences.

    A fresh session per example, because the property is about the
    sequence, not about one session accumulating state.
    """
    root = tmp_path_factory.mktemp("rule3")
    session = Session.start(_frame(), session_id="rule3", root=root)

    for name in actions:
        card = READ_ONLY_ACTIONS[name](session)  # type: ignore[operator]
        # Every action must still produce something renderable; an action
        # that raised would hide the property rather than satisfy it.
        assert card.to_text()

    assert len(session.provenance) == 0, (
        f"a step was recorded by {actions}, none of which is an accept or an override"
    )
    assert session.test_ledger.n_tests == 0
    assert len(session.test_ledger.entries) == 0
    assert list(session.store.versions) == ["v0"], "a data version was created"


def test_asking_the_same_question_repeatedly_never_executes(session: Session) -> None:
    """The specific shape of the bug rule 3 exists to prevent: a system
    that runs the test once it is confident enough about the question."""
    for _ in range(5):
        session.ask(
            goal="compare_groups",
            outcome="value",
            group="group",
            design="independent",
            confirmed_by_user=["design"],
        )
    assert len(session.provenance) == 0


def test_only_accept_and_override_reach_record_step(session: Session) -> None:
    """The same property stated structurally rather than empirically.

    A new conversational method that forgot the rule would pass the
    property test above only until someone added it to the action list;
    this fails immediately, because it checks the call graph instead.
    """
    import inspect

    from edacopilot.orchestrator import loop

    methods = dict(inspect.getmembers(loop.Orchestrator, inspect.isfunction))

    records = {
        name for name, function in methods.items() if "record_step" in inspect.getsource(function)
    }
    assert records == {"_run", "_run_post_hoc"}, (
        f"record_step is reached from {sorted(records)}; only the two execution helpers "
        f"may reach it (rule 3)"
    )

    executes = {
        name
        for name, function in methods.items()
        if name not in records
        and any(
            call in inspect.getsource(function) for call in ("self._run(", "self._run_post_hoc(")
        )
    }
    assert executes == {"accept", "override"}, (
        f"an execution helper is called from {sorted(executes)}; only accept and override "
        f"may run anything (rule 3)"
    )


def test_a_suggestion_is_never_followed_automatically(session: Session) -> None:
    """Section 9.5: suggestions are shown, never executed."""
    card = session.ask(
        goal="compare_groups",
        outcome="value",
        group="group",
        design="independent",
        confirmed_by_user=["design"],
    )
    # This data produces a divergence, so `accept()` with no persona asks
    # which one rather than choosing -- itself part of the property.
    asked = session.accept()
    assert asked.kind == "warning"
    assert asked.actions, "a card that says 'choose one' and offers no way to choose"
    assert len(session.provenance) == 0

    result = session.accept("professor")
    before = len(session.provenance)
    assert result.actions, "a result card with no next steps"
    # Rendering the card, and reading every action it offers, changes nothing.
    result.to_text()
    for action in result.actions:
        assert action.intent.type is not IntentType.ACCEPT or action.call == "session.accept()"
    assert len(session.provenance) == before
    assert card.kind in ("consensus", "divergence")


# --------------------------------------------------------------------------
# 2. The design question precedes any persona card
# --------------------------------------------------------------------------


def test_repeated_ids_produce_a_question_card_not_a_persona_card(tmp_path: Path) -> None:
    session = Session.start(paired_as_independent(), session_id="paired", root=tmp_path)
    card = session.ask(
        goal="compare_groups", outcome="score", group="condition", design="independent"
    )

    assert card.kind == "question", f"got a {card.kind} card before the design was confirmed"
    assert card.proposals == [], "a persona proposed something before the design was settled"
    assert session.orchestrator.state.pending is None, "a candidate set was left on screen"
    assert "paired" in card.body_md.lower()


@pytest.mark.parametrize("design", ["independent", "unknown", "paired"])
def test_the_design_question_precedes_the_card_whichever_design_was_proposed(
    tmp_path: Path, design: str
) -> None:
    """The mirror-image errors too (Section 7.1).

    `paired` on data with repeated ids is the one case that resolves
    without a question, because the data agrees with the claim; the other
    two must ask.
    """
    session = Session.start(paired_as_independent(), session_id=f"d-{design}", root=tmp_path)
    card = session.ask(
        goal="compare_groups",
        outcome="score",
        group="condition",
        design=design,
        **(
            {"subject": "subject_id", "confirmed_by_user": ["design"]} if design == "paired" else {}
        ),
    )
    if design == "paired":
        assert card.kind in ("consensus", "divergence")
    else:
        assert card.kind == "question"
        assert card.proposals == []


def test_the_engine_never_edits_the_design_itself(tmp_path: Path) -> None:
    """Section 7.1: "The engine never edits `spec.design` itself, in either
    direction." A system that quietly corrected the design would be making
    the most consequential call in the analysis on the user's behalf."""
    session = Session.start(paired_as_independent(), session_id="noedit", root=tmp_path)
    session.ask(goal="compare_groups", outcome="score", group="condition", design="independent")
    question = session.orchestrator.state.open_question
    assert question is not None and question.spec is not None
    assert question.spec.design.value == "independent"


def test_answering_the_design_question_resumes_the_turn(tmp_path: Path) -> None:
    session = Session.start(paired_as_independent(), session_id="resume", root=tmp_path)
    session.ask(goal="compare_groups", outcome="score", group="condition", design="independent")
    card = session.answer(design="paired")
    assert card.kind in ("consensus", "divergence")
    assert card.proposals, "no persona proposed anything after the design was confirmed"
    pending = session.orchestrator.state.pending
    assert pending is not None
    assert pending.candidates.family == "two_paired_numeric"
    assert len(session.provenance) == 0, "answering a question executed something"


# --------------------------------------------------------------------------
# 3. An INELIGIBLE override needs both gates (Section 7.5)
# --------------------------------------------------------------------------


@pytest.fixture
def ineligible_session(tmp_path: Path) -> Session:
    """A session whose proposal contains an INELIGIBLE candidate.

    `cochran_armitage_trend` needs a binary outcome; `tenure_months` is not
    one, so a hard assumption fails.
    """
    session = Session.start(ordinal_as_numeric(), session_id="inelig", root=tmp_path)
    session.ask(
        goal="association",
        x="satisfaction",
        y="tenure_months",
        design="independent",
        confirmed_by_user=["design", "x", "y"],
    )
    return session


INELIGIBLE = "cochran_armitage_trend"


def test_the_fixture_really_offers_an_ineligible_candidate(ineligible_session: Session) -> None:
    """Without this, the three tests below would pass vacuously."""
    pending = ineligible_session.orchestrator.state.pending
    assert pending is not None
    assert pending.candidates.get(INELIGIBLE).eligibility is Eligibility.INELIGIBLE


@pytest.mark.parametrize(
    ("reason", "confirm"),
    [
        (None, False),
        (None, True),
        ("", True),
        ("   ", True),
        ("a reason", False),
    ],
    ids=["neither", "confirm-only", "empty-reason", "blank-reason", "reason-only"],
)
def test_an_ineligible_override_without_both_gates_does_not_run(
    ineligible_session: Session, reason: str | None, confirm: bool
) -> None:
    card = ineligible_session.override(INELIGIBLE, reason, confirm=confirm)
    assert card.kind == "override_confirm"
    assert card.result is None
    assert len(ineligible_session.provenance) == 0
    assert ineligible_session.test_ledger.n_tests == 0


def test_the_ineligible_card_names_the_failure_and_its_consequence(
    ineligible_session: Session,
) -> None:
    """Section 7.5: the card "shows the failed hard assumption and its
    consequence". A confirmation dialog that only said "are you sure" would
    teach nothing, which is the point of the friction."""
    card = ineligible_session.override(INELIGIBLE, "because")
    assert card.diagnostics, "no failed assumption was shown"
    assert all(check.status.value == "fail" for check in card.diagnostics)
    assert all(check.consequence for check in card.diagnostics)
    for check in card.diagnostics:
        assert check.consequence in card.body_md


def test_both_gates_together_run_it_and_record_the_reason(
    ineligible_session: Session,
) -> None:
    card = ineligible_session.override(INELIGIBLE, "the reviewer asked for it", confirm=True)
    assert card.kind == "result"
    assert card.result is not None
    step = ineligible_session.provenance.steps[-1]
    assert step.chosen_via == "override"
    assert step.override_reason == "the reviewer asked for it"
    assert step.code.startswith("# Override: the reviewer asked for it\n")


def test_the_second_confirmation_is_separate_from_the_reason(
    ineligible_session: Session,
) -> None:
    """The two gates are two decisions, so confirming once must not carry
    over to a later override of a different method."""
    ineligible_session.override(INELIGIBLE, "reason one", confirm=True)
    ineligible_session.ask(
        goal="association",
        x="satisfaction",
        y="tenure_months",
        design="independent",
        confirmed_by_user=["design", "x", "y"],
    )
    card = ineligible_session.override(INELIGIBLE, "reason two")
    assert card.kind == "override_confirm", "a previous confirmation was reused"


def test_the_internal_path_raises_rather_than_running_an_ineligible_method(
    ineligible_session: Session,
) -> None:
    """The Python API returns a card, because a UI needs one. The path
    underneath must not be reachable without the reason at all: a bug there
    should fail loudly instead of silently running the method."""
    orchestrator = ineligible_session.orchestrator
    pending = orchestrator.state.pending
    assert pending is not None
    with pytest.raises(OverrideRequired):
        orchestrator._run(
            pending, pending.candidates.get(INELIGIBLE), chosen_via="override", reason=None
        )
    assert len(ineligible_session.provenance) == 0


def test_a_caveat_override_needs_a_reason_but_not_a_confirmation(tmp_path: Path) -> None:
    """Section 7.5's lighter gate, for contrast with the one above."""
    from tests.scenarios.generators import heteroscedastic_groups

    session = Session.start(heteroscedastic_groups(), session_id="cav", root=tmp_path)
    session.ask(
        goal="compare_groups",
        outcome="value",
        group="group",
        design="independent",
        confirmed_by_user=["design"],
    )
    pending = session.orchestrator.state.pending
    assert pending is not None
    caveated = next(
        c.function
        for c in pending.candidates.candidates
        if c.eligibility is Eligibility.CAVEAT
        and c.function not in {p.function for p in pending.verdict.picks}
    )

    assert session.override(caveated).kind == "override_confirm"
    assert len(session.provenance) == 0

    ran = session.override(caveated, "the team reports this test by convention")
    assert ran.kind == "result"
    assert session.provenance.steps[-1].override_reason


def test_choosing_an_eligible_method_needs_no_friction(session: Session) -> None:
    """And the case with no gate at all: the engine already ruled it valid."""
    session.ask(
        goal="compare_groups",
        outcome="value",
        group="group",
        design="independent",
        confirmed_by_user=["design"],
    )
    pending = session.orchestrator.state.pending
    assert pending is not None
    proposed = {p.function for p in pending.verdict.picks}
    unproposed = next(
        c.function
        for c in pending.candidates.candidates
        if c.eligibility is Eligibility.ELIGIBLE and c.function not in proposed
    )
    card = session.override(unproposed)
    assert card.kind == "result"
    assert session.provenance.steps[-1].override_reason is None


# --------------------------------------------------------------------------
# Stage scoping (Section 9.3)
# --------------------------------------------------------------------------


def test_a_stage_cannot_execute_a_function_outside_its_scope(session: Session) -> None:
    from edacopilot.stages import FunctionNotAllowedError, get_stage
    from edacore.contracts import Candidate

    explore = get_stage(Stage.EXPLORE)
    with pytest.raises(FunctionNotAllowedError):
        explore.check_allowed("welch_t")

    hypothesis = get_stage(Stage.HYPOTHESIS)
    hypothesis.check_allowed("welch_t")  # in scope
    with pytest.raises(FunctionNotAllowedError):
        hypothesis.execute(
            Candidate(
                function="detect_duplicates",
                eligibility=Eligibility.ELIGIBLE,
                estimand="",
            ),
            session,
        )
