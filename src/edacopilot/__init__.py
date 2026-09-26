"""edacopilot: the agent layer on top of edacore (ARCHITECTURE.md, Section 3.1).

    import edacopilot as eda
    session = eda.start(df, target="default_flag", name="loan_eda")

`start` is Section 13.1's entry point: it creates a `Session` and renders
the panel. Everything the panel can do, the session can do directly —
`session.ask(...)`, `session.accept("professor")`, `session.override(...)`
— because the panel is a view over that API and holds no logic of its own.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from edacopilot.session import Session, list_sessions, resume

__version__ = "0.1.0"


def start(
    df: pd.DataFrame,
    *,
    target: str | None = None,
    name: str | None = None,
    display: bool = True,
    **kwargs: Any,
) -> Session:
    """Start a session and render the panel (Section 13.1).

    Returns the `Session`, not the panel: the session is the object worth
    holding on to, and every action is available on it whether or not the
    widgets rendered. `display=False` builds the panel without showing it,
    which is what the magics and the tests want.

    The panel is attached as `session._panel` so `%eda` can append cards to
    the same view rather than opening a second one.
    """
    from edacopilot.ui.panel import Panel

    config = dict(kwargs.pop("config", None) or {})
    if target is not None:
        config["target"] = target

    session = Session.start(df, config=config, **kwargs)
    panel = Panel(session, name=name or session.session_id)
    session._panel = panel  # type: ignore[attr-defined]
    panel.show(session.goto_stage("profile"))
    if display:  # pragma: no cover - needs a live kernel
        panel.display()
    return session


def load_ipython_extension(ipython: Any) -> None:
    """`%load_ext edacopilot` (Section 13.1)."""
    from edacopilot.ui.magics import load_ipython_extension as register

    register(ipython)


__all__ = [
    "Session",
    "__version__",
    "list_sessions",
    "load_ipython_extension",
    "resume",
    "start",
]
