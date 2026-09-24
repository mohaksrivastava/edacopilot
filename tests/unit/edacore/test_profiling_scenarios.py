"""M1 acceptance criteria (ARCHITECTURE.md Section 16, milestone M1 row):

    Trap scenarios ordinal_as_numeric, sentinel_missing, target_leakage,
    duplicate_rows, pii_present detected

Each test below plants one problem via tests/scenarios/generators.py and
checks the specific detector *and* that profile_dataset surfaces it, since
profile_dataset is what the orchestrator will actually call.
"""

from __future__ import annotations

import edacore
from tests.scenarios.generators import (
    duplicate_rows,
    ordinal_as_numeric,
    pii_present,
    sentinel_missing,
    target_leakage,
)


def test_ordinal_as_numeric_flagged_as_ordinal_candidate() -> None:
    df = ordinal_as_numeric()

    assert "satisfaction" in edacore.profiling.detect_ordinal_candidates(df)

    profile = edacore.profiling.profile_dataset(df)
    column = next(c for c in profile.columns if c.name == "satisfaction")
    assert "ordinal_candidate" in column.flags


def test_sentinel_missing_detected_before_summaries() -> None:
    df = sentinel_missing()

    hits = edacore.profiling.detect_sentinel_values(df)
    assert hits["income"] == [-999]

    profile = edacore.profiling.profile_dataset(df)
    column = next(c for c in profile.columns if c.name == "income")
    assert -999 in column.sentinel_candidates


def test_target_leakage_flagged_in_profile() -> None:
    df, target = target_leakage()

    flags = edacore.profiling.detect_leakage_candidates(df, target=target)
    assert any(flag.column == "outcome_encoded" for flag in flags)

    profile = edacore.profiling.profile_dataset(df, target=target)
    assert "outcome_encoded" in profile.leakage_candidates


def test_duplicate_rows_detected_in_profile() -> None:
    df = duplicate_rows()

    report = edacore.profiling.detect_duplicates(df)
    assert report.n_exact > 0

    profile = edacore.profiling.profile_dataset(df)
    assert profile.duplicate_rows == report.n_exact
    assert profile.duplicate_rows > 0


def test_pii_present_detected() -> None:
    df = pii_present()

    hits = edacore.profiling.detect_pii(df)
    assert "email" in hits
    assert "phone" in hits

    profile = edacore.profiling.profile_dataset(df)
    assert "email" in profile.pii_columns
    assert "phone" in profile.pii_columns
