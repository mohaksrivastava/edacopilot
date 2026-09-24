"""Hypothesis tests (ARCHITECTURE.md, Section 6.7): one-sample, two-sample
independent, and two-sample paired. Split across submodules; import both for
registration side-effects.
"""

from __future__ import annotations

from edacore.stattests import categorical as categorical
from edacore.stattests import correlation as correlation
from edacore.stattests import distribution as distribution
from edacore.stattests import factorial as factorial
from edacore.stattests import k_independent as k_independent
from edacore.stattests import k_related as k_related
from edacore.stattests import one_sample as one_sample
from edacore.stattests import posthoc as posthoc
from edacore.stattests import two_sample as two_sample
