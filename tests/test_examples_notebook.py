"""`examples/m8_demo.ipynb` executes top to bottom (m8.1, Task A1).

No separate CI step is needed for this: it is an ordinary test under
`tests/`, so `pytest -m "not llm and not slow"` (the existing CI step)
already runs it. The root `conftest.py`'s Agg backend applies to this
process, not to the notebook's own kernel subprocess, but nothing in
`edacore.viz` uses `pyplot` (Section 6.11's M8 changelog entry — each
function builds a bare `Figure`), so there is no interactive-backend
dependency for the kernel to pick up differently.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

NOTEBOOK = Path(__file__).resolve().parents[1] / "examples" / "m8_demo.ipynb"


def test_m8_demo_notebook_executes_top_to_bottom() -> None:
    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = NotebookClient(nb, timeout=120, kernel_name="python3")
    try:
        client.execute()
    except CellExecutionError as exc:  # pragma: no cover - failure path
        pytest.fail(f"examples/m8_demo.ipynb raised during execution:\n{exc}")
