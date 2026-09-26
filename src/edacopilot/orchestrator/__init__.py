"""The orchestrator (ARCHITECTURE.md, Section 9).

The turn loop, the stage state machine and the cards. This is the only
component that talks to the user (Section 3.3): stage modules are scoped
skills operating on one shared `Session`, and they never address the user
themselves.
"""

from __future__ import annotations

from .cards import ActionButton, Card, CardKind, ProposalView, Suggestion
from .intents import CONFIDENCE_FLOOR, Intent, IntentType, button
from .loop import Orchestrator, OverrideRequired, TurnState
from .state_machine import (
    CONDITIONAL_STAGES,
    STAGE_ORDER,
    Stage,
    applicable_stages,
    jump_warning,
    next_stage,
    parse_stage,
    skipped_stages,
)

__all__ = [
    "CONDITIONAL_STAGES",
    "CONFIDENCE_FLOOR",
    "STAGE_ORDER",
    "ActionButton",
    "Card",
    "CardKind",
    "Intent",
    "IntentType",
    "Orchestrator",
    "OverrideRequired",
    "ProposalView",
    "Stage",
    "Suggestion",
    "TurnState",
    "applicable_stages",
    "button",
    "jump_warning",
    "next_stage",
    "parse_stage",
    "skipped_stages",
]
