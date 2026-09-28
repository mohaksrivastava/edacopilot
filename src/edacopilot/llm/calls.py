"""The seven LLM calls (ARCHITECTURE.md, Section 3.4).

Each function here does one thing: build a prompt through `ContextBuilder`
and the call's template, ask `LLMClient` for a validated answer, and
either return it or let `LLMFallbackRequired` propagate. **None of these
functions catches that exception themselves** -- the caller (an
orchestrator method, or a test) decides what deterministic fallback to
run, because the right fallback differs per call: `parse_intent` and
`answer_free_question` have one in `llm/fallback.py`; `build_question_spec`
falls back to the form card; `suggest_next_step` and `select_adhoc_function`
fall back to logic that already lives in the orchestrator (Section 10.5's
own five-call list does not cover these two, because there was nothing to
duplicate -- the deterministic behaviour predates this module).

`write_rationale` is the one call with a second kind of failure: valid
JSON, valid schema, but not grounded in this session's facts (Section 8.4).
It retries once with the fact-check failure appended, then raises
`LLMFallbackRequired` like everything else.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel

from edacopilot.orchestrator.intents import Intent

from .client import LLMFallbackRequired
from .context import ContextBuilder, PromptContext
from .contracts import (
    ElicitationQuestion,
    ElicitationQuestions,
    FunctionCall,
    GroundedAnswer,
    NextStepSuggestion,
    QuestionSpecDraft,
    RationaleBundle,
)
from .factcheck import Fact, check_rationale
from .prompts._shared import build_messages

if TYPE_CHECKING:
    from edacopilot.eligibility.spec import QuestionSpec
    from edacopilot.session import Session

__all__ = [
    "answer_free_question",
    "build_question_spec",
    "elicit_mnar",
    "parse_intent",
    "select_adhoc_function",
    "suggest_next_step",
    "write_rationale",
]


ModelT = TypeVar("ModelT", bound=BaseModel)


def _context(session: Session, *, pii_confirmed: bool = False) -> PromptContext:
    level = session.config.get("privacy", {}).get("level", "standard")
    return ContextBuilder(session).build(level=level, pii_confirmed=pii_confirmed)


def _ask(
    session: Session,
    call: str,
    context: PromptContext,
    schema: type[ModelT],
    extra: dict[str, Any] | None = None,
) -> ModelT:
    messages = build_messages(call, context, schema, extra=extra)
    result = session.llm.complete_structured(call, messages, schema)
    session.log_prompt(call, messages)
    return result


def parse_intent(text: str, session: Session, *, awaiting_answer: bool = False) -> Intent:
    """Section 3.4's `parse_intent`. Raises `LLMFallbackRequired` on failure."""
    context = _context(session)
    return _ask(
        session,
        "parse_intent",
        context,
        Intent,
        extra={"text": text, "awaiting_answer": awaiting_answer},
    )


def build_question_spec(text: str, session: Session) -> QuestionSpec:
    """Section 3.4's `build_question_spec`, returned as a full `QuestionSpec`.

    `ambiguities` and `confirmed_by_user` are never the model's to set --
    the first is `validate_spec`'s output, the second only the user can
    grant -- so both are fixed here, not asked for in the schema.
    """
    from edacopilot.eligibility.spec import QuestionSpec

    context = _context(session)
    draft = _ask(session, "build_question_spec", context, QuestionSpecDraft, extra={"text": text})
    return QuestionSpec(
        goal=draft.goal,
        variables=draft.variables,
        design=draft.design,
        alternative=draft.alternative,
        alpha=draft.alpha,
        reference_value=draft.reference_value,
        reference_proportions=draft.reference_proportions,
        user_text=text,
    )


def write_rationale(
    session: Session,
    *,
    facts: dict[str, Fact],
    picked_functions: set[str],
    extra: dict[str, Any],
) -> RationaleBundle:
    """Section 8.4: one retry on a fact-check failure, then fall back.

    `extra` carries the candidates/picks/checks the template needs --
    built by the caller (the persona engine already has all of it), not
    reconstructed here.
    """
    context = _context(session)
    messages = build_messages("write_rationale", context, RationaleBundle, extra=extra)
    bundle = session.llm.complete_structured("write_rationale", messages, RationaleBundle)
    session.log_prompt("write_rationale", messages)

    failure = check_rationale(bundle, facts=facts, picked_functions=picked_functions)
    if failure is None:
        return bundle

    retry_messages = [
        *messages,
        {"role": "assistant", "content": bundle.model_dump_json()},
        {
            "role": "user",
            "content": f"That rationale failed a fact-check: {failure.reason}. Fix it and resend.",
        },
    ]
    bundle = session.llm.complete_structured("write_rationale", retry_messages, RationaleBundle)
    session.log_prompt("write_rationale", retry_messages)
    failure = check_rationale(bundle, facts=facts, picked_functions=picked_functions)
    if failure is not None:
        raise LLMFallbackRequired("write_rationale", failure.reason)
    return bundle


def elicit_mnar(column: str, session: Session, *, missing_pct: float) -> list[ElicitationQuestion]:
    """Section 3.4's `elicit_mnar`. No deterministic fallback exists yet
    (Section 10.5: "arrives with the missingness stage in M10"); a caller
    without a working LLM has nothing to fall back to until then."""
    context = _context(session)
    result = _ask(
        session,
        "elicit_mnar",
        context,
        ElicitationQuestions,
        extra={"column": column, "missing_pct": missing_pct},
    )
    return list(result.questions)


def suggest_next_step(session: Session) -> NextStepSuggestion:
    """Section 3.4's `suggest_next_step`. Falls back to the orchestrator's
    own `stage.next_suggestions`/`result_suggestions` rules, which predate
    this module and already implement the deterministic version."""
    context = _context(session)
    return _ask(session, "suggest_next_step", context, NextStepSuggestion)


def answer_free_question(topic: str, session: Session) -> GroundedAnswer:
    """Section 3.4's `answer_free_question`. Falls back to
    `llm.fallback.answer_free_question`'s glossary lookup."""
    context = _context(session)
    return _ask(session, "answer_free_question", context, GroundedAnswer, extra={"topic": topic})


def select_adhoc_function(
    text: str, session: Session, *, candidates: list[dict[str, Any]]
) -> FunctionCall:
    """Section 3.4's `select_adhoc_function`.

    `candidates` must already be filtered to `read_only=True` registry
    entries by the caller -- this function does not re-filter that list,
    but it does enforce the result stays inside it: Section 10.1 restricts
    this call to registered, read-only functions, and a name outside
    `candidates` is exactly as wrong as a schema violation, so it takes
    the same path -- `LLMFallbackRequired`, not a value the caller has to
    separately remember to check.
    """
    context = _context(session)
    call = _ask(
        session,
        "select_adhoc_function",
        context,
        FunctionCall,
        extra={"text": text, "candidates": candidates},
    )
    allowed = {candidate["name"] for candidate in candidates}
    if call.function not in allowed:
        raise LLMFallbackRequired(
            "select_adhoc_function", f"{call.function!r} is not among the offered candidates"
        )
    return call
