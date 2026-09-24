"""Hypothesis tests (ARCHITECTURE.md, Section 6.7): one-sample, two-sample
independent, and two-sample paired. Split across submodules; import both for
registration side-effects.
"""

from __future__ import annotations

from edacore.stattests import one_sample as one_sample
from edacore.stattests import two_sample as two_sample
