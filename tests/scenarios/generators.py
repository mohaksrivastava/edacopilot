"""Synthetic "trap" datasets (ARCHITECTURE.md, Section 15.3).

Each generator plants one problem a junior analyst commonly misses. Only the
five scenarios M1's acceptance criteria names are implemented here
(ordinal_as_numeric, sentinel_missing, target_leakage, duplicate_rows,
pii_present); the rest belong to the milestones that exercise them
(paired_as_independent -> M4, mar_by_group -> M10, non_stationary_ts -> M12,
etc.) and should be added alongside that work, not before.

Every generator is seeded for reproducibility — there is no run-to-run
variance to chase down if a scenario test fails.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ordinal_as_numeric(n: int = 200) -> pd.DataFrame:
    """A 1-5 Likert scale stored as plain int. Must be flagged as an ordinal
    candidate, not treated as a generic discrete/continuous numeric column."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "customer_id": range(1, n + 1),
            "satisfaction": rng.integers(1, 6, size=n),
            "tenure_months": rng.integers(1, 120, size=n),
        }
    )


def sentinel_missing(n: int = 200) -> pd.DataFrame:
    """-999 used in place of NaN for missing income. Must be detected as a
    sentinel before any numeric summary is computed from raw values."""
    rng = np.random.default_rng(1)
    income = rng.normal(60000, 15000, size=n).round(2)
    income[rng.random(n) < 0.08] = -999
    return pd.DataFrame({"customer_id": range(1, n + 1), "income": income})


def target_leakage(n: int = 200) -> tuple[pd.DataFrame, str]:
    """A column that is a direct copy of the target. Must be flagged in
    PROFILE. Returns (df, target_column_name)."""
    rng = np.random.default_rng(2)
    outcome = rng.integers(0, 2, size=n)
    df = pd.DataFrame(
        {
            "customer_id": range(1, n + 1),
            "outcome": outcome,
            "outcome_encoded": outcome,
            "tenure_months": rng.integers(1, 120, size=n),
        }
    )
    return df, "outcome"


def duplicate_rows(n: int = 200, duplicate_frac: float = 0.05) -> pd.DataFrame:
    """~5% exact duplicate rows appended to the dataset. Must be detected in
    PROFILE."""
    rng = np.random.default_rng(3)
    base = pd.DataFrame(
        {
            "customer_id": range(1, n + 1),
            "age": rng.integers(18, 80, size=n),
            "region": rng.choice(["A", "B", "C"], size=n),
        }
    )
    n_dupes = max(1, round(n * duplicate_frac))
    dupes = base.sample(n=n_dupes, random_state=3)
    return pd.concat([base, dupes], ignore_index=True)


def pii_present(n: int = 200) -> pd.DataFrame:
    """Email and phone columns. Strict mode should be suggested and nothing
    from these columns should ever reach the LLM."""
    rng = np.random.default_rng(4)
    return pd.DataFrame(
        {
            "customer_id": range(1, n + 1),
            "email": [f"user{i}@example.com" for i in range(n)],
            "phone": [f"555-{100 + (i % 800):03d}-{1000 + i:04d}" for i in range(n)],
            "age": rng.integers(18, 80, size=n),
        }
    )
