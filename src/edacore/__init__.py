"""edacore: deterministic EDA statistics library, no LLM dependency.

Usable standalone (``import edacore``) or as the deterministic core beneath
the ``edacopilot`` agent layer (ARCHITECTURE.md, Section 3.1).
"""

from __future__ import annotations

from edacore import assumptions as assumptions
from edacore import effect_sizes as effect_sizes
from edacore import multiplicity as multiplicity
from edacore import power as power
from edacore import profiling as profiling
from edacore import stattests as stattests

__version__ = "0.1.0"
