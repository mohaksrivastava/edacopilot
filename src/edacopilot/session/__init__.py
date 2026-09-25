"""Session state (ARCHITECTURE.md, Section 12).

    session = Session.start(df)
    candidates = session.candidates(spec)     # cached per dataset version
    verdict = session.propose(spec)
    session.accept(stage="hypothesis", chosen=..., result=...)
    session.undo(); session.branch_from("consultant-cleaned")
    session = resume(session.session_id)      # after a kernel restart

`.edacopilot/` holds the user's data as parquet. See `GITIGNORE_WARNING`.
"""

from __future__ import annotations

from .dataset_store import (
    MAIN_BRANCH,
    PARQUET_COMPRESSION,
    ROOT_VERSION,
    DatasetStore,
    ReadOnlyVersionError,
    UnknownBranchError,
    UnknownVersionError,
    VersionRecord,
)
from .ledger import (
    DEFAULT_FAMILY,
    POSTHOC_FUNCTIONS,
    LedgerEntry,
    TestLedger,
    is_posthoc,
)
from .provenance import (
    ProposalView,
    ProvenanceLog,
    Step,
    assert_no_user_evaluation,
)
from .session import (
    GITIGNORE_WARNING,
    SESSION_DIR,
    STATE_FILE,
    Session,
    SessionNotFoundError,
    list_sessions,
    resume,
)

__all__ = [
    "DEFAULT_FAMILY",
    "GITIGNORE_WARNING",
    "MAIN_BRANCH",
    "PARQUET_COMPRESSION",
    "POSTHOC_FUNCTIONS",
    "ROOT_VERSION",
    "SESSION_DIR",
    "STATE_FILE",
    "DatasetStore",
    "LedgerEntry",
    "ProposalView",
    "ProvenanceLog",
    "ReadOnlyVersionError",
    "Session",
    "SessionNotFoundError",
    "Step",
    "TestLedger",
    "UnknownBranchError",
    "UnknownVersionError",
    "VersionRecord",
    "assert_no_user_evaluation",
    "is_posthoc",
    "list_sessions",
    "resume",
]
