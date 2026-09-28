"""Section 7.1's column resolution, through the orchestrator (M9.2): a
correction note on the resulting card, and an ambiguity card with one
button per candidate that a click actually resolves."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from edacopilot.session import Session


def _income_session(tmp_path: Path) -> Session:
    rng = np.random.default_rng(5)
    income = rng.lognormal(10, 0.6, 80)
    df = pd.DataFrame(
        {"income": income, "income_log": np.log(income), "gender": ["F"] * 40 + ["M"] * 40}
    )
    return Session.start(df, session_id="colmatch", root=tmp_path)


def test_a_corrected_typo_shows_a_visible_note_on_the_proposal_card(tmp_path: Path) -> None:
    session = _income_session(tmp_path)
    card = session.ask(
        goal="compare_groups",
        outcome="icnome",
        group="gender",
        design="independent",
        confirmed_by_user=["design"],
    )
    assert card.kind in ("consensus", "divergence")
    assert "Using `income` (you wrote 'icnome')" in card.body_md


def test_no_correction_note_when_the_column_was_already_right(tmp_path: Path) -> None:
    session = _income_session(tmp_path)
    card = session.ask(
        goal="compare_groups",
        outcome="income",
        group="gender",
        design="independent",
        confirmed_by_user=["design"],
    )
    assert "you wrote" not in card.body_md


def test_an_ambiguous_column_produces_one_button_per_candidate(tmp_path: Path) -> None:
    session = _income_session(tmp_path)
    card = session.ask(
        goal="compare_groups",
        outcome="income_l",
        group="gender",
        design="independent",
        confirmed_by_user=["design"],
    )
    assert card.kind == "question"
    assert card.title == "Which column did you mean?"
    labels = {action.label for action in card.actions}
    assert labels == {"outcome: use `income`", "outcome: use `income_log`"}


def test_clicking_a_candidate_button_resolves_and_proceeds(tmp_path: Path) -> None:
    session = _income_session(tmp_path)
    ambiguous = session.ask(
        goal="compare_groups",
        outcome="income_l",
        group="gender",
        design="independent",
        confirmed_by_user=["design"],
    )
    button = next(a for a in ambiguous.actions if "income_log" in a.label)
    resolved = session.orchestrator.handle(button.intent)
    assert resolved.kind in ("consensus", "divergence")

    # Matches the equivalent direct API call exactly (Section 13's own
    # click-equals-API-call invariant, applied here too).
    session_direct = Session.start(
        session.data, session_id="colmatch-direct", root=session.store.directory.parent
    )
    direct = session_direct.ask(
        goal="compare_groups",
        outcome="income_log",
        group="gender",
        design="independent",
        confirmed_by_user=["design"],
    )
    assert resolved.to_text() == direct.to_text()


def test_column_ambiguity_is_answered_the_same_way_the_api_allows(tmp_path: Path) -> None:
    """`session.answer(outcome=...)` (the same method a design answer
    uses) resolves a column ambiguity too -- no new API surface."""
    session = _income_session(tmp_path)
    session.ask(
        goal="compare_groups",
        outcome="income_l",
        group="gender",
        design="independent",
        confirmed_by_user=["design"],
    )
    resolved = session.answer(outcome="income")
    assert resolved.kind in ("consensus", "divergence")
