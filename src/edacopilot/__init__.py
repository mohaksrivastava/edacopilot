"""edacopilot: the agent layer on top of edacore (ARCHITECTURE.md, Section 3.1).

``Session`` and ``resume()`` are Section 12's session layer (M6).
``start()`` and ``load_ipython_extension()`` land with the orchestrator and
Jupyter UI milestones (M7-M8).
"""

from __future__ import annotations

from edacopilot.session import Session, list_sessions, resume

__version__ = "0.1.0"

__all__ = ["Session", "__version__", "list_sessions", "resume"]
