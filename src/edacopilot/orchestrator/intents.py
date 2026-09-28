"""Intents (ARCHITECTURE.md, Section 9.1).

An `Intent` is the one thing the orchestrator's turn loop consumes. Two
things produce one: a button click, which constructs it directly with
`confidence=1.0` and no LLM involved, and free text, which goes through
`parse_intent`. In deterministic mode (Section 10.5, and the whole test
suite) that parser is the keyword ruleset in `edacopilot.llm.fallback`.

Keeping both paths behind one type is what makes the conversation tests
worth anything: a golden transcript that drives the session through the
Python API exercises the same loop a button click will in M8, so a UI bug
is the only thing M8 can introduce.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# Below this, the orchestrator asks a clarifying question instead of acting
# (Section 9.1). It applies to parsed free text only; a button is certain.
CONFIDENCE_FLOOR = 0.6


class IntentType(StrEnum):
    ASK_ANALYSIS = "ask_analysis"
    ACCEPT = "accept"
    MODIFY = "modify"
    OVERRIDE = "override"
    EXPLAIN = "explain"
    GOTO_STAGE = "goto_stage"
    UNDO = "undo"
    BRANCH = "branch"
    SWITCH_BRANCH = "switch_branch"
    SHOW = "show"
    SHOW_DIAGNOSTIC_PLOT = "show_diagnostic_plot"
    SKIP = "skip"
    EXPORT = "export"
    SETTINGS = "settings"
    ANSWER_CLARIFICATION = "answer_clarification"
    OTHER = "other"


class Intent(BaseModel):
    """What the user wants, structured (Section 9.1)."""

    model_config = ConfigDict(frozen=True)

    type: IntentType
    persona: str | None = None
    target_stage: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0

    @property
    def is_confident(self) -> bool:
        return self.confidence >= CONFIDENCE_FLOOR


def button(intent_type: IntentType, **payload: Any) -> Intent:
    """An intent from a button click: certain by construction.

    Section 9.1: "Button clicks produce `Intent`s directly (no LLM)." The
    fields the `Intent` model names explicitly are lifted out of the
    payload so a button and a parsed message produce the same shape.
    """
    persona = payload.pop("persona", None)
    target_stage = payload.pop("target_stage", None)
    return Intent(
        type=intent_type,
        persona=persona,
        target_stage=target_stage,
        payload=payload,
        confidence=1.0,
    )
