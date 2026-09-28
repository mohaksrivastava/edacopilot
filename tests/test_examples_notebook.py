"""`examples/m8_demo.ipynb` executes top to bottom (m8.1, Task A1).

No separate CI step is needed for this: it is an ordinary test under
`tests/`, so `pytest -m "not llm and not slow"` (the existing CI step)
already runs it. The root `conftest.py`'s Agg backend applies to this
process, not to the notebook's own kernel subprocess, but nothing in
`edacore.viz` uses `pyplot` (Section 6.11's M8 changelog entry — each
function builds a bare `Figure`), so there is no interactive-backend
dependency for the kernel to pick up differently.

**`conftest.py`'s block on a live `litellm.completion` call also does not
reach the notebook's kernel subprocess** (M9): a monkeypatch in this
process has no effect there. The notebook sets
`config={"llm": {"deterministic_mode": True}}` on every `eda.start(...)`
call itself, for the same reason `tests/conversations/transcripts.py`'s
`Transcript.start` now does -- this notebook's own text already says
"no LLM call anywhere", and one of its cells calls `session.ask(...)`
with free text, which is exactly the call M9 wired to try a live model
first if nothing tells it not to.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError, CellTimeoutError

NOTEBOOK = Path(__file__).resolve().parents[1] / "examples" / "m8_demo.ipynb"

# Generous rather than tight: `CellTimeoutError` is a sibling of
# `CellExecutionError`, not a subclass (nbclient.exceptions), so a timeout
# would otherwise escape as an uncaught exception rather than a clean
# `pytest.fail`. 300s covers a cold kernel start plus the full edacore
# import graph on a slow filesystem, which a 120s budget did not.
CELL_TIMEOUT = 300


def test_m8_demo_notebook_executes_top_to_bottom() -> None:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(nb, timeout=CELL_TIMEOUT, kernel_name="python3")
    try:
        client.execute()
    except (CellExecutionError, CellTimeoutError) as exc:  # pragma: no cover - failure path
        pytest.fail(f"examples/m8_demo.ipynb raised during execution:\n{exc}")
