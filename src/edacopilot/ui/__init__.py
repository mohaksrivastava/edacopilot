"""The Jupyter UI (ARCHITECTURE.md, Section 13).

`Panel` draws a `Session`; `render_card` draws one `Card`. Neither decides
anything — every button emits the `Intent` the card already carries, into
the same `Orchestrator.handle` the Python API calls.
"""

from __future__ import annotations

from .cards import (
    KIND_COLOURS,
    KIND_LABELS,
    STATUS_MARKS,
    action_row,
    diagnostics_widget,
    render_card,
    result_widget,
)
from .magics import EdaMagicError, load_ipython_extension, run_magic
from .markup import to_html
from .panel import MAX_HISTORY, Panel

__all__ = [
    "KIND_COLOURS",
    "KIND_LABELS",
    "MAX_HISTORY",
    "STATUS_MARKS",
    "EdaMagicError",
    "Panel",
    "action_row",
    "diagnostics_widget",
    "load_ipython_extension",
    "render_card",
    "result_widget",
    "run_magic",
    "to_html",
]
