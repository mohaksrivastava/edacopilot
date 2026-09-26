"""Golden conversation transcripts (ARCHITECTURE.md, Section 15.5).

Section 15.5: "Golden transcripts in `tests/conversations/` replay a
sequence of user actions against a mocked LLM (recorded structured outputs)
and assert the cards, steps, versions and exported notebook."

There is no mock here, because in M7 there is nothing to mock: the system
runs in deterministic mode (rule 9), which is Section 10.5's fallback all
the way down. When M9 adds the LLM, the recorded outputs slot in at
`parse_intent` and `build_question_spec` and these same transcripts keep
their meaning — which is the point of driving them through the Python API
rather than through a UI.

Each transcript is a list of turns. A turn is one user action and the card
it produced, and the rendered transcript shows both, so a diff in a golden
file is a diff in what a user would see. That is a deliberately strict
test: it fails on wording changes as well as behaviour changes. Wording is
most of what this product is — a card nobody reads is a card that did not
work — so it should not be able to change unnoticed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from edacopilot.orchestrator.cards import Card
from edacopilot.session import Session
from tests.scenarios.generators import (
    ordinal_as_numeric,
    paired_as_independent,
    worked_example_7_4,
)

Turn = Callable[[Session], Card]


@dataclass(frozen=True)
class Step:
    """One user action: what they did, and how to do it."""

    say: str
    run: Turn
    # An aside printed before the card, where the transcript needs to
    # explain something the cards themselves cannot.
    note: str = ""


@dataclass(frozen=True)
class Transcript:
    name: str
    title: str
    note: str
    data: Callable[[], pd.DataFrame]
    steps: list[Step] = field(default_factory=list)

    @property
    def slug(self) -> str:
        return self.name

    def start(self, root: Path) -> Session:
        return Session.start(self.data(), session_id=self.name, root=root)


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------


def three_clinics() -> pd.DataFrame:
    """Three groups with a real difference and heavy right skew, so the
    omnibus is significant and the parametric options are caveated."""
    rng = np.random.default_rng(4)
    return pd.DataFrame(
        {
            "wait_minutes": np.concatenate(
                [rng.exponential(2, 40), rng.exponential(3, 40), rng.exponential(6, 40)]
            ),
            "clinic": ["A"] * 40 + ["B"] * 40 + ["C"] * 40,
        }
    )


def income_with_missing() -> pd.DataFrame:
    """15% of `income` missing, so a jump to HYPOTHESIS has something
    concrete to warn about (Section 9.2's own example)."""
    rng = np.random.default_rng(2)
    income = rng.lognormal(10, 0.6, 120)
    income[rng.choice(120, 18, replace=False)] = np.nan
    return pd.DataFrame({"income": income, "region": ["north"] * 60 + ["south"] * 60})


# --------------------------------------------------------------------------
# the six transcripts
# --------------------------------------------------------------------------


def _worked_example() -> Transcript:
    return Transcript(
        name="worked_example_7_4",
        title="The Section 7.4 worked example, end to end",
        note=(
            "`income` by `gender`, n = 38 and 41, strong right skew. Section 7.4 predicts "
            "a two-proposal divergence card: Professor and Consultant on Mann-Whitney, the "
            "Maverick on Yuen's trimmed-mean t-test, which is what the card below shows."
        ),
        data=worked_example_7_4,
        steps=[
            Step('session.goto_stage("profile")', lambda s: s.goto_stage("profile")),
            Step(
                'session.ask(goal="compare_groups", outcome="income", group="gender",\n'
                '            design="independent", confirmed_by_user=["design"])',
                lambda s: s.ask(
                    goal="compare_groups",
                    outcome="income",
                    group="gender",
                    design="independent",
                    confirmed_by_user=["design"],
                ),
                note=(
                    "The design is stated and confirmed up front here. The next transcript "
                    "shows what happens when it is not."
                ),
            ),
            Step('session.explain("student_t")', lambda s: s.explain("student_t")),
            Step('session.accept("professor")', lambda s: s.accept("professor")),
            Step("session.ledger()", lambda s: s.orchestrator.ledger_card()),
        ],
    )


def _paired_trap() -> Transcript:
    return Transcript(
        name="paired_as_independent",
        title="The paired_as_independent trap: the design question comes first",
        note=(
            "Section 15.3's trap. The same 30 subjects are measured before and after, but "
            "the question is asked as though the groups were independent. The engine "
            "refuses to propose anything until the design is settled — no persona is "
            "consulted, because `select_candidates` raises `AmbiguousSpecError` on a spec "
            "with open ambiguities and a persona can only choose from a `CandidateSet`."
        ),
        data=paired_as_independent,
        steps=[
            Step(
                'session.ask(goal="compare_groups", outcome="score", group="condition",\n'
                '            design="independent")',
                lambda s: s.ask(
                    goal="compare_groups",
                    outcome="score",
                    group="condition",
                    design="independent",
                ),
            ),
            Step('session.answer(design="paired")', lambda s: s.answer(design="paired")),
            Step("session.accept()", lambda s: s.accept()),
        ],
    )


def _ineligible_override() -> Transcript:
    return Transcript(
        name="ineligible_override",
        title="Overriding an ineligible method: a reason and a second confirmation",
        note=(
            "Section 7.5's harder gate. `cochran_armitage_trend` needs a binary outcome and "
            "`tenure_months` is not one, so a hard assumption fails and the method is "
            "INELIGIBLE. The first override call is refused with the failed assumption and "
            "its consequence; only a second call carrying both the reason and "
            "`confirm=True` runs it."
        ),
        data=ordinal_as_numeric,
        steps=[
            Step(
                'session.ask(goal="association", x="satisfaction", y="tenure_months",\n'
                '            design="independent", confirmed_by_user=["design", "x", "y"])',
                lambda s: s.ask(
                    goal="association",
                    x="satisfaction",
                    y="tenure_months",
                    design="independent",
                    confirmed_by_user=["design", "x", "y"],
                ),
            ),
            Step(
                'session.override("cochran_armitage_trend",\n'
                '                 reason="the reviewer asked for a trend test")',
                lambda s: s.override(
                    "cochran_armitage_trend", reason="the reviewer asked for a trend test"
                ),
                note="A reason alone is not enough for an ineligible method.",
            ),
            Step(
                'session.override("cochran_armitage_trend",\n'
                '                 reason="the reviewer asked for a trend test", confirm=True)',
                lambda s: s.override(
                    "cochran_armitage_trend",
                    reason="the reviewer asked for a trend test",
                    confirm=True,
                ),
                note="The reason is written into the recorded code as a comment.",
            ),
        ],
    )


def _undo_and_branch() -> Transcript:
    return Transcript(
        name="undo_then_branch",
        title="Undo, then branch from the version the undo moved off",
        note=(
            "An undo moves the branch head; it does not delete the version and it does not "
            "erase the step from the provenance log. So a version you undo is still there "
            "to branch from, which is what makes an undo safe to try.\n\n"
            "The data version here is created with `session.record_step(new_data=...)`, "
            "standing in for the cleaning step the QUALITY stage will make in M11. Every "
            "other action in this transcript is the ordinary conversational API."
        ),
        data=worked_example_7_4,
        steps=[
            Step(
                'session.record_step(stage="quality",\n'
                '                    new_data=session.data.query("income < 120000"),\n'
                '                    label="trimmed the top tail")',
                lambda s: _record_and_report(s),
            ),
            Step("session.undo()", lambda s: s.undo()),
            Step(
                'session.branch("kept-the-tail", version="v1")',
                lambda s: s.branch("kept-the-tail", version="v1"),
                note=(
                    "Branching from `v1` — the version the undo moved off — recovers it "
                    "without losing the main branch."
                ),
            ),
            Step('session.switch_branch("main")', lambda s: s.switch_branch("main")),
        ],
    )


def _kruskal_then_posthoc() -> Transcript:
    return Transcript(
        name="kruskal_then_posthoc",
        title="A significant Kruskal-Wallis, then an accepted post-hoc",
        note=(
            "Waiting times at three clinics, heavily right-skewed. The omnibus says the "
            "clinics differ but not which; the suggested next step is the post-hoc whose "
            "own correction matches the design. Section 12.3's rule is visible in the "
            "ledger at the end: the post-hoc is recorded but does not count toward the "
            "session adjustment, because its comparisons are already corrected within "
            "their own family."
        ),
        data=three_clinics,
        steps=[
            Step(
                'session.ask(goal="compare_groups", outcome="wait_minutes",\n'
                '            group="clinic", design="independent",\n'
                '            confirmed_by_user=["design"])',
                lambda s: s.ask(
                    goal="compare_groups",
                    outcome="wait_minutes",
                    group="clinic",
                    design="independent",
                    confirmed_by_user=["design"],
                ),
            ),
            Step('session.accept("professor")', lambda s: s.accept("professor")),
            Step(
                'session.ask("run post-hoc comparisons")',
                lambda s: s.ask("run post-hoc comparisons"),
                note=(
                    "Free text, routed by Section 10.5's keyword rules. The suggestion "
                    "button on the previous card produces the same intent."
                ),
            ),
            Step("session.accept()", lambda s: s.accept()),
            Step("session.ledger()", lambda s: s.orchestrator.ledger_card()),
        ],
    )


def _jump_with_missing_data() -> Transcript:
    return Transcript(
        name="jump_with_missing_data",
        title="Jumping to HYPOTHESIS with untreated missing data",
        note=(
            "Section 9.2: the stage order is a suggestion, not a rule. Jumping past an "
            "unvisited stage is allowed and produces one line saying what it costs — with "
            "a number from this dataset, not a general caution. The test then reports the "
            "n it actually used, which is smaller than the row count."
        ),
        data=income_with_missing,
        steps=[
            Step('session.goto_stage("profile")', lambda s: s.goto_stage("profile")),
            Step(
                'session.goto_stage("hypothesis")',
                lambda s: s.goto_stage("hypothesis"),
                note="Quality, missingness, outliers and transform were never visited.",
            ),
            Step(
                'session.ask(goal="compare_groups", outcome="income", group="region",\n'
                '            design="independent", confirmed_by_user=["design"])',
                lambda s: s.ask(
                    goal="compare_groups",
                    outcome="income",
                    group="region",
                    design="independent",
                    confirmed_by_user=["design"],
                ),
            ),
            Step("session.accept()", lambda s: s.accept()),
        ],
    )


def _record_and_report(session: Session) -> Card:
    """The stand-in for M11's cleaning step, rendered as a card.

    Written out rather than hidden so the transcript does not imply the
    TRANSFORM stage exists yet.
    """
    trimmed = session.data.query("income < 120000")
    step = session.record_step(stage="quality", new_data=trimmed, label="trimmed the top tail")
    return Card(
        kind="info",
        title="Data version created",
        stage="quality",
        body_md=(
            f"Step `{step.step_id}` created version `{step.output_version}` from "
            f"`{step.input_version}`: {len(session.data)} rows, down from "
            f"{len(session.original())}.\n\n"
            f"Nothing is edited in place — `v0` stays read-only, so this is reversible."
        ),
    )


def transcripts() -> list[Transcript]:
    """Section 15.5's golden transcripts, in a fixed order."""
    return [
        _worked_example(),
        _paired_trap(),
        _ineligible_override(),
        _undo_and_branch(),
        _kruskal_then_posthoc(),
        _jump_with_missing_data(),
    ]


def by_name(name: str) -> Transcript:
    for transcript in transcripts():
        if transcript.name == name:
            return transcript
    raise KeyError(name)
