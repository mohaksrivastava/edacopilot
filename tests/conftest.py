"""Test-suite-wide setup.

Two things, both about matplotlib, both about the suite not falling over
for reasons unrelated to the code under test.

**The Agg backend, chosen before anything imports pyplot.** A test run has
no display, and matplotlib's default backend selection would otherwise
depend on what happens to be installed on the machine — an interactive
backend on a developer's laptop and Agg on CI, which is exactly the shape
of difference that makes a suite pass locally and fail in CI. Setting it
here, in the root conftest, is the only place early enough to be certain.

**Closing figures.** `pyplot` keeps a global registry of every figure ever
created and never drops one on its own; matplotlib warns after 20 and the
memory is held until the process exits. `edacore.viz` has 25 functions and
the tests call them repeatedly, so a suite that did not close them would
leak steadily and emit warnings that mean nothing. The autouse fixture
closes whatever a test left open, so no individual test has to remember.
"""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib
import pytest

# Before any `import matplotlib.pyplot` anywhere in the suite.
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _close_matplotlib_figures() -> Iterator[None]:
    """Close every figure a test leaves open (see the module docstring)."""
    yield
    import matplotlib.pyplot as plt

    plt.close("all")
