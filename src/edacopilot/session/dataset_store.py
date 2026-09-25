"""The dataset version DAG (ARCHITECTURE.md, Section 12.1).

Every accepted transform creates a *new* version rather than changing the
old one. That is what makes undo, branching and "show me what this looked
like three steps ago" possible at all, and it is the storage-level
expression of rule 6: no function mutates its input.

Two properties are load-bearing and enforced here rather than by
convention:

- **v0 is read-only.** The original data is the one thing a session can
  always fall back to, and a bug that silently rewrote it would be
  unrecoverable. It can never be the output of a step, and nothing can
  overwrite its parquet file.
- **Only v0 and the active head stay in memory.** Section 12.1 asks for
  this, and the reason is that a long session branches: a dozen versions of
  a million-row frame is gigabytes of nothing anyone is looking at. Other
  versions load from parquet on demand.

**Parquet compression is zstd, not the pandas default of snappy.** Measured
on this project's own WSL setup, writing a 1M-row frame to `/mnt/c` takes
2.1 s with snappy and 1.1 s with zstd, for 19.3 MB against 12.7 MB. The
Windows drive is reached over a 9p filesystem that is roughly an order of
magnitude slower than WSL's native one (0.14 s for the same write), so the
bottleneck is bytes crossing that boundary, and the compression that
produces fewer of them wins on *both* time and space.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from edacore.contracts import DatasetProfile
from edacore.profiling import profile_dataset

ROOT_VERSION = "v0"
MAIN_BRANCH = "main"

# See the module docstring: chosen from measurements on /mnt/c under WSL,
# where it is both faster and smaller than the default.
PARQUET_COMPRESSION: Literal["zstd"] = "zstd"


class VersionRecord(BaseModel):
    """One node of the DAG, as persisted (Section 12.1's `DatasetVersion`).

    `profile` is deliberately absent: Section 12.1 calls for it to be
    "recomputed lazily", and a `DatasetProfile` of a wide frame is large
    enough that storing one per version would dominate `session.json`.
    `DatasetStore.profile()` computes and caches it on first access.
    """

    model_config = ConfigDict(frozen=True)

    version_id: str
    parent: str | None
    branch: str
    created_by_step: str | None = None
    label: str = ""
    n_rows: int = 0
    n_cols: int = 0

    @property
    def is_root(self) -> bool:
        return self.parent is None


class ReadOnlyVersionError(RuntimeError):
    """Raised on any attempt to overwrite v0 (Section 12.1: "v0 is the
    original data, read-only")."""


class UnknownVersionError(KeyError):
    pass


class UnknownBranchError(KeyError):
    pass


@dataclass
class DatasetStore:
    """The version DAG for one session.

    Construct through `DatasetStore.create(df, directory)` for a new
    session, or `DatasetStore.restore(...)` when resuming.
    """

    directory: Path
    versions: dict[str, VersionRecord] = field(default_factory=dict)
    branch_heads: dict[str, str] = field(default_factory=dict)
    active_branch: str = MAIN_BRANCH
    _memory: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    _profiles: dict[str, DatasetProfile] = field(default_factory=dict, repr=False)

    # ---- construction ---------------------------------------------------

    @classmethod
    def create(cls, df: pd.DataFrame, directory: Path) -> DatasetStore:
        directory.mkdir(parents=True, exist_ok=True)
        store = cls(directory=directory)
        record = VersionRecord(
            version_id=ROOT_VERSION,
            parent=None,
            branch=MAIN_BRANCH,
            label="original data",
            n_rows=len(df),
            n_cols=df.shape[1],
        )
        store._write(record.version_id, df)
        store.versions[record.version_id] = record
        store.branch_heads[MAIN_BRANCH] = record.version_id
        store._memory[record.version_id] = df
        return store

    @classmethod
    def restore(
        cls,
        directory: Path,
        versions: dict[str, VersionRecord],
        branch_heads: dict[str, str],
        active_branch: str,
    ) -> DatasetStore:
        store = cls(
            directory=directory,
            versions=dict(versions),
            branch_heads=dict(branch_heads),
            active_branch=active_branch,
        )
        return store

    # ---- storage --------------------------------------------------------

    def path_for(self, version_id: str) -> Path:
        return self.directory / f"{version_id}.parquet"

    def _write(self, version_id: str, df: pd.DataFrame) -> None:
        path = self.path_for(version_id)
        if version_id == ROOT_VERSION and path.exists():
            raise ReadOnlyVersionError(
                "v0 holds the original data and is read-only; it is the one thing a "
                "session can always fall back to"
            )
        df.to_parquet(path, compression=PARQUET_COMPRESSION, index=False)

    def _evict(self) -> None:
        """Section 12.1: keep only v0 and the active head in memory."""
        keep = {ROOT_VERSION, self.head}
        for version_id in list(self._memory):
            if version_id not in keep:
                del self._memory[version_id]

    # ---- the DAG --------------------------------------------------------

    @property
    def head(self) -> str:
        return self.branch_heads[self.active_branch]

    @property
    def branches(self) -> list[str]:
        return sorted(self.branch_heads)

    def record(self, version_id: str) -> VersionRecord:
        try:
            return self.versions[version_id]
        except KeyError as exc:
            raise UnknownVersionError(
                f"no version '{version_id}'; have {sorted(self.versions)}"
            ) from exc

    def get(self, version_id: str | None = None) -> pd.DataFrame:
        """The data for a version, from memory or from parquet."""
        version_id = version_id or self.head
        self.record(version_id)
        cached = self._memory.get(version_id)
        if cached is not None:
            return cached
        df = pd.read_parquet(self.path_for(version_id))
        if version_id in (ROOT_VERSION, self.head):
            self._memory[version_id] = df
        return df

    def profile(self, version_id: str | None = None) -> DatasetProfile:
        """Section 12.1's lazily recomputed profile, cached per version."""
        version_id = version_id or self.head
        self.record(version_id)
        if version_id not in self._profiles:
            self._profiles[version_id] = profile_dataset(self.get(version_id))
        return self._profiles[version_id]

    def _next_id(self) -> str:
        return f"v{len(self.versions)}"

    def add_version(
        self,
        df: pd.DataFrame,
        *,
        created_by_step: str | None = None,
        label: str = "",
        parent: str | None = None,
    ) -> VersionRecord:
        """Record a transformed dataset as a child of `parent` (the current
        head by default) and move the active branch head to it."""
        parent_id = parent or self.head
        self.record(parent_id)
        record = VersionRecord(
            version_id=self._next_id(),
            parent=parent_id,
            branch=self.active_branch,
            created_by_step=created_by_step,
            label=label,
            n_rows=len(df),
            n_cols=df.shape[1],
        )
        self._write(record.version_id, df)
        self.versions[record.version_id] = record
        self.branch_heads[self.active_branch] = record.version_id
        self._memory[record.version_id] = df
        self._evict()
        return record

    def undo(self) -> VersionRecord:
        """Move the active branch head to its parent (Section 12.1).

        The version itself is *not* deleted: a branch may point at it, and
        an undo the user regrets should be recoverable by switching back or
        branching from it.
        """
        current = self.record(self.head)
        if current.parent is None:
            raise ReadOnlyVersionError(
                "already at the original data (v0); there is nothing to undo"
            )
        self.branch_heads[self.active_branch] = current.parent
        self._evict()
        return self.record(current.parent)

    def branch(self, name: str, *, from_version: str | None = None) -> VersionRecord:
        """Start a new branch at any version (Section 12.1's example: compare
        consultant-cleaned against professor-cleaned data)."""
        if name in self.branch_heads:
            raise ValueError(f"branch '{name}' already exists")
        if not name.strip():
            raise ValueError("a branch needs a name")
        start = self.record(from_version or self.head)
        self.branch_heads[name] = start.version_id
        self.active_branch = name
        self._evict()
        return start

    def switch_branch(self, name: str) -> VersionRecord:
        if name not in self.branch_heads:
            raise UnknownBranchError(f"no branch '{name}'; have {self.branches}")
        self.active_branch = name
        self._evict()
        return self.record(self.head)

    def ancestry(self, version_id: str | None = None) -> list[str]:
        """This version and every ancestor, newest first."""
        current: str | None = version_id or self.head
        chain: list[str] = []
        while current is not None:
            chain.append(current)
            current = self.record(current).parent
        return chain

    def children(self, version_id: str) -> list[str]:
        return sorted(v.version_id for v in self.versions.values() if v.parent == version_id)

    def in_memory(self) -> set[str]:
        """Which versions are currently held in memory. Exposed so the
        eviction promise in Section 12.1 can be asserted, not assumed."""
        return set(self._memory)

    def to_state(self) -> dict[str, Any]:
        return {
            "versions": [v.model_dump() for v in self.versions.values()],
            "branch_heads": dict(self.branch_heads),
            "active_branch": self.active_branch,
        }
