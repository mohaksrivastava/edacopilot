"""Structured output schemas for the seven LLM calls (ARCHITECTURE.md,
Section 3.4). Every one is validated with pydantic; a validation failure
is what sends a call through the retry-then-fallback seam in `client.py`.

`Intent` and `QuestionSpec` already exist (`orchestrator.intents`,
`eligibility.spec`) and are reused directly rather than redefined here --
`build_question_spec`'s draft schema is narrower than the full
`QuestionSpec`, because `ambiguities` is `validate_spec`'s output, not an
input the model should be filling in, and `confirmed_by_user` is something
only the user, not the model, can set.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from edacopilot.eligibility.spec import Design, Goal


class QuestionSpecDraft(BaseModel):
    """`build_question_spec`'s output: the fields an LLM may fill.

    Turned into a full `QuestionSpec` by the caller, with `ambiguities=[]`
    and `confirmed_by_user=set()` -- both are settled later, by
    `validate_spec` and by the user respectively, never by this draft.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    goal: Goal
    variables: dict[str, str] = Field(default_factory=dict)
    design: Design = Design.UNKNOWN
    alternative: str = "two-sided"
    alpha: float | None = None
    reference_value: float | None = None
    reference_proportions: dict[str, float] | None = None


class PersonaRationale(BaseModel):
    """One persona's explanation (Section 8.4)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    persona: str
    summary: str  # <= 2 sentences
    pros: list[str] = Field(default_factory=list)  # <= 3
    cons: list[str] = Field(default_factory=list)  # <= 3
    consequence_if_ignored: str
    cited_facts: list[str] = Field(default_factory=list)


class RationaleBundle(BaseModel):
    """Section 8.4's full output for one proposal card."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rationales: list[PersonaRationale]
    comparison: str = ""  # <= 4 sentences: when you'd pick each


class ElicitationQuestion(BaseModel):
    """One `elicit_mnar` domain question (Section 3.4)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    question: str
    rationale: str = ""


class ElicitationQuestions(BaseModel):
    """`elicit_mnar`'s top-level output: 2-4 questions, one call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    questions: list[ElicitationQuestion] = Field(default_factory=list)


class NextStepSuggestion(BaseModel):
    """`suggest_next_step`'s output. Never executes anything (rule 3)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    suggestion: str
    target_stage: str | None = None
    rationale: str = ""


class GroundedAnswer(BaseModel):
    """`answer_free_question`'s output, grounded in this session's facts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    answer: str
    cited_facts: list[str] = Field(default_factory=list)


class FunctionCall(BaseModel):
    """`select_adhoc_function`'s output: a read-only registry call.

    Validated against `registry.json_schema(function)` by the caller, and
    restricted to `read_only=True` functions (Section 10.1) -- both are
    enforced in `calls.py`, not here, since this schema only fixes shape.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    function: str
    params: dict[str, str | float | int | bool] = Field(default_factory=dict)
