"""Fuzzy column-name resolution (ARCHITECTURE.md, Section 7.1's
column-resolution step, `eligibility/column_matching.py`)."""

from __future__ import annotations

from edacopilot.eligibility.column_matching import ColumnMatchConfig, match_column

COLUMNS = ["income", "income_log", "age", "gender", "wait_minutes"]


def test_a_clear_typo_resolves_with_a_correction() -> None:
    match = match_column("icnome", COLUMNS, ColumnMatchConfig())
    assert match.resolved == "income"
    assert match.correction == "Using `income` (you wrote 'icnome')"
    assert match.candidates == []


def test_a_close_second_candidate_is_not_silently_corrected() -> None:
    """The income vs. income_log case: both are plausible, so this must
    not guess -- it must ask."""
    match = match_column("income_l", COLUMNS, ColumnMatchConfig())
    assert match.resolved is None
    assert match.correction is None
    assert set(match.candidates) == {"income", "income_log"}


def test_a_confident_match_still_wins_even_with_a_similar_column_present() -> None:
    """The other half of the income/income_log pairing: a name close
    enough to one and far enough from the other resolves cleanly."""
    match = match_column("incom_log", COLUMNS, ColumnMatchConfig())
    assert match.resolved == "income_log"

    match = match_column("incme", COLUMNS, ColumnMatchConfig())
    assert match.resolved == "income"


def test_nothing_plausible_returns_no_match_at_all() -> None:
    match = match_column("zzzzzzz", COLUMNS, ColumnMatchConfig())
    assert match.resolved is None
    assert match.candidates == []


def test_an_exact_match_is_never_reached_by_the_caller() -> None:
    """Documented as the caller's job (validate_spec checks `in df.columns`
    first), but confirm match_column itself doesn't need to special-case
    it: an exact name scores 1.0 and resolves trivially."""
    match = match_column("income", COLUMNS, ColumnMatchConfig())
    assert match.resolved == "income"
    assert match.correction == "Using `income` (you wrote 'income')"


def test_candidates_are_capped_at_max_candidates() -> None:
    config = ColumnMatchConfig(max_candidates=2)
    match = match_column("xncxme", COLUMNS, config)
    assert len(match.candidates) <= 2


def test_thresholds_are_configurable() -> None:
    """A stricter threshold turns a would-be correction into an ambiguity."""
    lenient = match_column("gendre", COLUMNS, ColumnMatchConfig(threshold=0.5))
    strict = match_column("gendre", COLUMNS, ColumnMatchConfig(threshold=0.95))
    assert lenient.resolved == "gender"
    assert strict.resolved is None
