"""The provenance log (ARCHITECTURE.md, Section 12.2).

Every accepted step is recorded with enough detail to answer "why is this
number here?" — the question, the checks that were run, what each persona
proposed, what was chosen and by whom, and the exact code that produced it.
That last field is what makes rule 7 (every accepted step reproducible as
plain Python) hold: the export in Section 14 is a replay of `Step.code`,
not a reconstruction.

**Rule 4 is the hard boundary of this module.** The log records the
*analysis*: datasets, methods, results. It stores nothing about the user —
no timing of how long they took to decide, no count of how often they
overrode a recommendation, no score of any kind. A `Step` has no field
that could hold one, and `assert_no_user_evaluation` exists so that stays
true as the model grows.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from edacopilot.eligibility.spec import QuestionSpec, describe_spec
from edacore.contracts import Candidate

ChosenVia = Literal["professor", "consultant", "maverick", "consensus", "override"]


class ProposalView(BaseModel):
    """What one persona proposed, flattened for the log."""

    model_config = ConfigDict(frozen=True)

    persona: str
    function: str | None
    eligibility: str | None = None
    rationale: str = ""
    concurs_with: str | None = None


class Step(BaseModel):
    """One accepted step (Section 12.2)."""

    model_config = ConfigDict(frozen=True)

    step_id: str
    stage: str
    branch: str
    question: QuestionSpec | None = None
    checks: list[str] = Field(default_factory=list)
    proposals: list[ProposalView] = Field(default_factory=list)
    chosen: Candidate | None = None
    chosen_via: ChosenVia = "consensus"
    override_reason: str | None = None
    input_version: str = ""
    output_version: str | None = None
    result_fact_id: str | None = None
    code: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def is_override(self) -> bool:
        return self.chosen_via == "override"

    def summary(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "stage": self.stage,
            "branch": self.branch,
            "chosen": self.chosen.function if self.chosen else None,
            "chosen_via": self.chosen_via,
            "override_reason": self.override_reason,
            "input_version": self.input_version,
            "output_version": self.output_version,
            "question": describe_spec(self.question) if self.question else None,
        }


# Field names that would amount to a judgement about the user rather than
# about the analysis. Section 3's rule 4: "Do not store, score, or report
# anything about the user's skill, habits, or behaviour."
_USER_EVALUATION_TERMS = frozenset(
    {
        "skill",
        "score",
        "rating",
        "competence",
        "experience_level",
        "mistakes",
        "errors_made",
        "warnings_ignored",
        "time_taken",
        "duration_seconds",
        "hesitation",
        "attempts",
        "retries",
        "user_level",
        "proficiency",
        "accuracy",
    }
)


def assert_no_user_evaluation(model: type[BaseModel] = Step) -> None:
    """Rule 4, as a check a test can run.

    A reviewer cannot reasonably re-read the whole model on every change;
    this makes the boundary explicit, so a field like `warnings_ignored`
    fails loudly instead of quietly shipping.
    """
    offending = sorted(set(model.model_fields) & _USER_EVALUATION_TERMS)
    if offending:
        raise AssertionError(
            f"{model.__name__} has field(s) {offending}, which record something about the "
            f"user rather than the analysis (rule 4)"
        )


class ProvenanceLog(BaseModel):
    """Append-only history of accepted steps.

    Append-only on purpose: an undo moves the dataset head, it does not
    erase the fact that the step happened. "What did I do and then undo?"
    is a question the log should be able to answer.
    """

    model_config = ConfigDict(frozen=False)

    steps: list[Step] = Field(default_factory=list)

    def append(self, step: Step) -> Step:
        if any(existing.step_id == step.step_id for existing in self.steps):
            raise ValueError(f"step '{step.step_id}' is already in the log")
        self.steps.append(step)
        return step

    def next_step_id(self) -> str:
        return f"s{len(self.steps)}"

    def for_branch(self, branch: str) -> list[Step]:
        return [step for step in self.steps if step.branch == branch]

    def by_id(self, step_id: str) -> Step:
        for step in self.steps:
            if step.step_id == step_id:
                return step
        raise KeyError(f"no step '{step_id}'")

    def overrides(self) -> list[Step]:
        """Every step where the user chose against the personas. Section 7.5
        requires a typed reason for these; this is how they are audited."""
        return [step for step in self.steps if step.is_override]

    def code(self, branch: str | None = None) -> str:
        """The session as a runnable script (rule 7), in order."""
        steps = self.for_branch(branch) if branch else self.steps
        return "\n".join(step.code for step in steps if step.code)

    def __len__(self) -> int:
        return len(self.steps)
