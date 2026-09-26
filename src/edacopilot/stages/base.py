"""The stage-module interface (ARCHITECTURE.md, Section 9.3).

Section 3.3's argument, restated as a type: a stage is a *scoped skill*,
not an autonomous agent. It has its own allowed function subset and its own
questions, it operates on the one shared `Session`, and it never addresses
the user — the orchestrator does that. So a stage returns cards and
outcomes; it does not print, prompt or decide when to run.

`allowed_functions` is the scope, and it is enforced rather than
documented: `StageModule.check_allowed` raises if a stage tries to execute
a function outside its own list. A stage that could call anything would
make Section 9.3's table a comment.

Two names Section 9.3 uses without defining, filled in here the same way
`PostHocResult` was: `StepOutcome` (what `execute` returns) and
`Suggestion` (what `next_suggestions` returns, defined with the cards it
renders into).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from edacopilot.eligibility import CandidateSet, QuestionSpec
from edacopilot.orchestrator.cards import Card, Suggestion
from edacopilot.orchestrator.intents import Intent
from edacore.contracts import AssumptionCheck, Candidate, PostHocResult, TestResult, TransformRecord

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from edacopilot.session.session import Session


class StageNotAvailableError(NotImplementedError):
    """Raised when a stage that is a stub in this build is asked to work.

    Carries the milestone, so the message can say *when* rather than only
    that the answer is no.
    """

    def __init__(self, stage: str, milestone: str, what: str) -> None:
        self.stage = stage
        self.milestone = milestone
        super().__init__(
            f"the {stage} stage ({what}) is available in a later version ({milestone})"
        )


class FunctionNotAllowedError(RuntimeError):
    """A stage tried to execute a function outside its own scope."""


class StepOutcome(BaseModel):
    """What executing a candidate produced (Section 9.3's `execute`).

    Every field is optional because stages differ in what they produce: a
    hypothesis test produces a result and no new data, a transform produces
    new data and no result.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    result: TestResult | PostHocResult | TransformRecord | None = None
    new_data: pd.DataFrame | None = None
    family: str = ""
    plots: list[str] = Field(default_factory=list)
    # Extra body text for the result card, beyond what the result itself
    # renders -- e.g. that a post-hoc's p-values are already adjusted.
    note: str = ""
    # Source that has to run before the call for the recorded step to be
    # reproducible: a paired test's long-to-wide pivot, for instance. Empty
    # for the common case where the method runs against the frame as it is.
    code_preamble: str = ""
    df_var: str = "df"


@runtime_checkable
class StageModule(Protocol):
    """Section 9.3's protocol."""

    name: str
    allowed_functions: list[str]

    def entry_summary(self, session: Session) -> Card: ...

    def build_spec(self, intent: Intent, session: Session) -> QuestionSpec: ...

    def diagnose(self, spec: QuestionSpec, session: Session) -> list[AssumptionCheck]: ...

    def candidates(
        self, spec: QuestionSpec, checks: list[AssumptionCheck], session: Session
    ) -> CandidateSet: ...

    def execute(self, candidate: Candidate, session: Session) -> StepOutcome: ...

    def next_suggestions(self, session: Session) -> list[Suggestion]: ...


class BaseStage:
    """Shared machinery: scope enforcement and the default refusals.

    A stage that does not pose questions inherits `build_spec` raising,
    rather than each stub reimplementing the same refusal — and rather than
    returning something empty, which would let a read-only stage silently
    reach the eligibility engine.
    """

    name: str = ""
    allowed_functions: list[str] = []
    # What this stage does, in the words Section 9.3's table uses. Shown on
    # a stub's info card.
    summary: str = ""
    milestone: str = ""

    def entry_summary(self, session: Session) -> Card:
        """What this stage will look at (Section 9.2).

        Declared on the base so every stage has one -- the state machine
        shows an entry card on every transition, and a stage without one
        would leave a jump silently doing nothing.
        """
        raise NotImplementedError(f"{type(self).__name__} has no entry_summary")

    def check_allowed(self, function: str) -> None:
        if function not in self.allowed_functions:
            raise FunctionNotAllowedError(
                f"the {self.name} stage may not call '{function}'; its scope is "
                f"{sorted(self.allowed_functions)} (Section 9.3)"
            )

    def build_spec(self, intent: Intent, session: Session) -> QuestionSpec:
        raise StageNotAvailableError(self.name, self.milestone or "a later milestone", self.summary)

    def diagnose(self, spec: QuestionSpec, session: Session) -> list[AssumptionCheck]:
        return list(session.candidates(spec).checks)

    def candidates(
        self, spec: QuestionSpec, checks: list[AssumptionCheck], session: Session
    ) -> CandidateSet:
        return session.candidates(spec)

    def execute(self, candidate: Candidate, session: Session) -> StepOutcome:
        raise StageNotAvailableError(self.name, self.milestone or "a later milestone", self.summary)

    def next_suggestions(self, session: Session) -> list[Suggestion]:
        return []
