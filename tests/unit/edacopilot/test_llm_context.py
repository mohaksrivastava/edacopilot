"""Section 11's privacy table, and the canary test Section 15's evals ask
for: plant a unique value nowhere-else-occurring in raw rows/text, and
prove it never reaches a prompt in either privacy level."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from edacopilot.llm.context import ContextBuilder
from edacopilot.llm.prompts._shared import render_context
from edacopilot.session import Session

CANARY = "CANARY-9f3c1b7e-do-not-leak"


def _session_with_pii(tmp_path) -> Session:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "income": rng.lognormal(10, 0.5, 40),
            "gender": ["F"] * 20 + ["M"] * 20,
            "email": [f"person{i}@example.com" for i in range(40)],
            "notes": [f"case {i}: {CANARY}" for i in range(40)],
        }
    )
    return Session.start(df, session_id="canary", root=tmp_path)


def test_no_canary_reaches_the_rendered_context_in_standard_mode(tmp_path) -> None:  # type: ignore[no-untyped-def]
    session = _session_with_pii(tmp_path)
    context = ContextBuilder(session).build(level="standard")
    rendered = render_context(context)
    assert CANARY not in rendered


def test_no_canary_reaches_the_rendered_context_in_strict_mode(tmp_path) -> None:  # type: ignore[no-untyped-def]
    session = _session_with_pii(tmp_path)
    context = ContextBuilder(session).build(level="strict", pii_confirmed=True)
    rendered = render_context(context)
    assert CANARY not in rendered


def test_pii_forces_strict_even_when_standard_was_asked_for(tmp_path) -> None:  # type: ignore[no-untyped-def]
    session = _session_with_pii(tmp_path)
    context = ContextBuilder(session).build(level="standard")
    assert context.privacy_level == "strict"


def test_no_real_column_name_appears_in_strict_mode(tmp_path) -> None:  # type: ignore[no-untyped-def]
    session = _session_with_pii(tmp_path)
    context = ContextBuilder(session).build(level="strict", pii_confirmed=True)
    rendered = render_context(context)
    for real_name in ("income", "gender", "email", "notes"):
        assert real_name not in rendered
    payload = json.loads(rendered)
    aliases = {col["name"] for col in payload["columns"]}
    assert aliases == {"col_01", "col_02", "col_03", "col_04"}


def test_confirmed_standard_mode_shows_real_names_but_not_pii_values(tmp_path) -> None:  # type: ignore[no-untyped-def]
    session = _session_with_pii(tmp_path)
    context = ContextBuilder(session).build(level="standard", pii_confirmed=True)
    assert context.privacy_level == "standard"
    rendered = render_context(context)
    assert "income" in rendered
    assert "gender" in rendered
    assert CANARY not in rendered
    assert "@example.com" not in rendered


def test_category_levels_are_capped_at_ten(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(5)
    df = pd.DataFrame({"level": [f"L{i}" for i in rng.integers(0, 15, 300)]})
    session = Session.start(df, session_id="levels", root=tmp_path)
    context = ContextBuilder(session).build(level="standard")
    col = next(c for c in context.columns if c.name == "level")
    assert col.top_levels is not None
    assert len(col.top_levels) <= 10


def test_no_raw_row_values_appear_for_a_continuous_column(tmp_path) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(9)
    values = rng.uniform(1000, 2000, 30)
    df = pd.DataFrame({"amount": values})
    session = Session.start(df, session_id="rows", root=tmp_path)
    context = ContextBuilder(session).build(level="standard")
    rendered = render_context(context)
    # A raw value, rendered to full precision, must not appear verbatim --
    # only rounded summary statistics may.
    raw_repr = repr(float(values[0]))
    assert raw_repr not in rendered
