"""Test-suite-wide setup.

Three things: two about matplotlib, one about the LLM layer (M9). All
three are about the suite not falling over, or reaching the network, for
reasons unrelated to the code under test.

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
from typing import Any

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


def _no_network_completion(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError(
        "litellm.completion was called without being mocked. M7-M8's whole suite "
        "predates the LLM layer and calls session.ask('free text') throughout on "
        "the assumption that nothing reaches a network -- rule 9 ('the system "
        "must run with the LLM switched off') is what makes that assumption "
        "still true after M9. Patch litellm.completion explicitly in any test "
        "that means to exercise the LLM path."
    )


@pytest.fixture(autouse=True)
def _no_live_llm_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test may reach a real model unless it says so (Section 15.6:
    "no network in CI"). A test exercising the LLM path re-patches
    `litellm.completion` itself, which simply shadows this one for its
    own scope; every other test gets `LLMFallbackRequired` for free,
    which is exactly Section 10.5's deterministic path -- the same one
    these tests already assumed before M9 existed."""
    monkeypatch.setattr("litellm.completion", _no_network_completion)
