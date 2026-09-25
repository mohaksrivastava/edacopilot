"""The dataset version DAG (ARCHITECTURE.md, Section 12.1).

Three properties carry real weight here and are asserted rather than
assumed: v0 is genuinely unwritable, only v0 and the active head stay in
memory, and undo/branch/switch round-trip without losing a version.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edacopilot.session import (
    MAIN_BRANCH,
    PARQUET_COMPRESSION,
    ROOT_VERSION,
    DatasetStore,
    ReadOnlyVersionError,
    UnknownBranchError,
    UnknownVersionError,
)


def _frame(n: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {"value": rng.normal(0, 1, n), "group": ["A"] * (n // 2) + ["B"] * (n - n // 2)}
    )


@pytest.fixture
def store(tmp_path: Path) -> DatasetStore:
    return DatasetStore.create(_frame(), tmp_path / "s1")


def test_a_new_store_has_only_v0_on_main(store: DatasetStore) -> None:
    assert store.head == ROOT_VERSION
    assert store.active_branch == MAIN_BRANCH
    assert store.branches == [MAIN_BRANCH]
    assert store.record(ROOT_VERSION).is_root


def test_v0_is_written_as_parquet_with_zstd(store: DatasetStore) -> None:
    """zstd rather than the pandas default: on /mnt/c under WSL it is both
    faster and smaller, because the bottleneck is bytes crossing a slow
    filesystem boundary (see dataset_store's module docstring)."""
    assert PARQUET_COMPRESSION == "zstd"
    assert store.path_for(ROOT_VERSION).exists()
    pd.testing.assert_frame_equal(store.get(ROOT_VERSION), _frame())


def test_v0_cannot_be_overwritten(store: DatasetStore) -> None:
    """The original data is the one thing a session can always fall back
    to; a bug that rewrote it would be unrecoverable."""
    with pytest.raises(ReadOnlyVersionError, match="read-only"):
        store._write(ROOT_VERSION, _frame(10))


def test_a_transform_creates_a_child_and_moves_the_head(store: DatasetStore) -> None:
    child = store.add_version(_frame(30), created_by_step="s1", label="filtered")
    assert child.version_id == "v1"
    assert child.parent == ROOT_VERSION
    assert store.head == "v1"
    assert store.ancestry() == ["v1", ROOT_VERSION]
    assert store.children(ROOT_VERSION) == ["v1"]
    assert len(store.get()) == 30
    # v0 is untouched.
    assert len(store.get(ROOT_VERSION)) == 50


def test_only_v0_and_the_active_head_stay_in_memory(store: DatasetStore) -> None:
    """Section 12.1's memory rule. A long session branches; a dozen
    versions of a large frame is gigabytes nobody is looking at."""
    store.add_version(_frame(30))
    store.add_version(_frame(20))
    store.add_version(_frame(10))
    assert store.in_memory() == {ROOT_VERSION, store.head}
    # An older version still loads, and loading it does not pin it.
    assert len(store.get("v1")) == 30
    assert store.in_memory() == {ROOT_VERSION, store.head}


def test_undo_moves_to_the_parent_without_deleting_the_version(store: DatasetStore) -> None:
    """An undo the user regrets must be recoverable -- so the version stays
    and only the head moves."""
    store.add_version(_frame(30))
    store.add_version(_frame(20))
    assert store.head == "v2"

    store.undo()
    assert store.head == "v1"
    assert "v2" in store.versions
    assert store.path_for("v2").exists()

    store.undo()
    assert store.head == ROOT_VERSION


def test_undo_at_the_root_is_refused(store: DatasetStore) -> None:
    with pytest.raises(ReadOnlyVersionError, match="nothing to undo"):
        store.undo()


def test_branching_from_any_version_and_switching_back(store: DatasetStore) -> None:
    """Section 12.1's own example: compare consultant-cleaned against
    professor-cleaned data."""
    store.add_version(_frame(40), label="shared cleaning")
    shared = store.head

    store.branch("consultant", from_version=shared)
    store.add_version(_frame(35), label="listwise drop")
    consultant_head = store.head

    store.switch_branch(MAIN_BRANCH)
    assert store.head == shared

    store.branch("professor", from_version=shared)
    store.add_version(_frame(39), label="imputed")
    professor_head = store.head

    assert store.branches == ["consultant", MAIN_BRANCH, "professor"]
    assert consultant_head != professor_head
    assert store.record(consultant_head).parent == shared
    assert store.record(professor_head).parent == shared

    store.switch_branch("consultant")
    assert store.head == consultant_head
    assert len(store.get()) == 35


def test_a_full_undo_branch_switch_round_trip_returns_the_same_data(
    store: DatasetStore,
) -> None:
    original = store.get(ROOT_VERSION).copy()
    store.add_version(_frame(30))
    store.branch("side")
    store.add_version(_frame(25))
    store.switch_branch(MAIN_BRANCH)
    store.undo()
    assert store.head == ROOT_VERSION
    pd.testing.assert_frame_equal(store.get(), original)


def test_a_duplicate_branch_name_is_refused(store: DatasetStore) -> None:
    store.branch("side")
    store.switch_branch(MAIN_BRANCH)
    with pytest.raises(ValueError, match="already exists"):
        store.branch("side")


def test_an_unnamed_branch_is_refused(store: DatasetStore) -> None:
    with pytest.raises(ValueError, match="needs a name"):
        store.branch("   ")


def test_unknown_versions_and_branches_raise(store: DatasetStore) -> None:
    with pytest.raises(UnknownVersionError):
        store.get("v99")
    with pytest.raises(UnknownBranchError):
        store.switch_branch("nope")


def test_profiles_are_computed_lazily_and_cached(store: DatasetStore) -> None:
    """Section 12.1: "recomputed lazily". A DatasetProfile per version would
    otherwise dominate session.json."""
    assert store._profiles == {}
    first = store.profile()
    assert store.profile() is first
    assert first.n_rows == 50


def test_branch_recorded_on_each_version(store: DatasetStore) -> None:
    store.branch("side")
    child = store.add_version(_frame(20))
    assert child.branch == "side"
    assert store.record(ROOT_VERSION).branch == MAIN_BRANCH
