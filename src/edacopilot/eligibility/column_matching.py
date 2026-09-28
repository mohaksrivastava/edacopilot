"""Fuzzy column-name resolution for `QuestionSpec` validation (Section 7.1).

A column name in a `QuestionSpec` came from an LLM reading free text (or
the deterministic form fallback) -- a typo ("icnome") is common and,
unlike a genuinely wrong request, has one obvious fix. This module is the
deterministic layer that makes that fix, never the LLM: `difflib` (stdlib,
no new dependency -- evaluated against `rapidfuzz` and found to already
clear the bar on every case this was written for) scores every real column
against the name as typed, and three thresholds decide what happens next:

- one candidate at or above `threshold`, with every other candidate more
  than `margin` behind it: resolved silently... except never silently --
  `resolved.correction` is a sentence for the card, so a wrong guess is
  something the user can see and reject, not something that just happened.
- more than one candidate within `margin` of the best score, or the best
  score is between `floor` and `threshold`: an ambiguity, with the
  candidates as buttons (`ColumnMatch.candidates`).
- nothing clears `floor`: not returned as a match at all -- the caller
  raises `InvalidSpecError`, because a name with no plausible column
  behind it is a spec-building bug (Section 7.1), not a question the user
  can resolve by picking from a list.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field

DEFAULT_THRESHOLD = 0.8
DEFAULT_MARGIN = 0.15
DEFAULT_FLOOR = 0.5
MAX_CANDIDATES = 3


@dataclass(frozen=True)
class ColumnMatchConfig:
    threshold: float = DEFAULT_THRESHOLD
    margin: float = DEFAULT_MARGIN
    floor: float = DEFAULT_FLOOR
    max_candidates: int = MAX_CANDIDATES


@dataclass(frozen=True)
class ColumnMatch:
    """The result of resolving one typed name against the real columns."""

    original: str
    resolved: str | None  # set only when a single confident match was found
    correction: str | None  # a sentence for the card, set iff resolved is
    candidates: list[str] = field(default_factory=list)  # set iff ambiguous


def match_column(name: str, columns: list[str], config: ColumnMatchConfig) -> ColumnMatch:
    """Resolve one column name against the real ones.

    `name in columns` is handled by the caller (this function is only
    reached for names that are *not* already a real column), so every
    branch here is about what to do with a wrong one.
    """
    scored = sorted(
        ((column, difflib.SequenceMatcher(None, name, column).ratio()) for column in columns),
        key=lambda pair: -pair[1],
    )
    if not scored or scored[0][1] < config.floor:
        return ColumnMatch(original=name, resolved=None, correction=None, candidates=[])

    best_column, best_score = scored[0]
    close = [column for column, score in scored if best_score - score <= config.margin]

    if best_score >= config.threshold and len(close) == 1:
        return ColumnMatch(
            original=name,
            resolved=best_column,
            correction=f"Using `{best_column}` (you wrote '{name}')",
            candidates=[],
        )

    candidates = (
        close if len(close) > 1 else [column for column, _ in scored[: config.max_candidates]]
    )
    return ColumnMatch(
        original=name,
        resolved=None,
        correction=None,
        candidates=candidates[: config.max_candidates],
    )
