"""False-positive tests for edacore.profiling.

Detection heuristics that never fire are useless; ones that fire too eagerly
are worse, since a junior analyst learns to ignore the tool. This file
checks the other side of test_profiling_scenarios.py's trap detections:

- a clean dataset with none of the planted problems must produce zero flags
- near-miss cases that resemble a trap but aren't one must NOT be flagged
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import edacore
from edacore.contracts import SemanticType


def test_clean_dataset_produces_zero_flags() -> None:
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame(
        {
            "income": rng.normal(60000, 12000, n).round(2),
            "age": rng.integers(18, 65, n),
            "region": rng.choice(["North", "South", "East", "West"], n),
            "signup_date": pd.date_range("2023-01-01", periods=n, freq="h"),
        }
    )

    profile = edacore.profiling.profile_dataset(df)

    for column in profile.columns:
        assert column.flags == [], f"{column.name} unexpectedly flagged: {column.flags}"
    assert profile.duplicate_rows == 0
    assert profile.leakage_candidates == []
    assert profile.pii_columns == []


def test_wide_range_integer_age_is_not_flagged_ordinal() -> None:
    """18-65 is a real age range, not a Likert scale, even though it's an
    all-integer column with modest cardinality relative to n."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"age": rng.integers(18, 66, size=300)})

    assert "age" not in edacore.profiling.detect_ordinal_candidates(df)
    assert "age" not in edacore.profiling.detect_numeric_coded_categoricals(df)

    semantic_type, _ = edacore.profiling.infer_semantic_types(df)["age"]
    assert semantic_type is SemanticType.DISCRETE


def test_legitimate_zeros_are_not_flagged_as_sentinels() -> None:
    """A real 'number of purchases' column has genuine zeros; 0 isn't in
    DEFAULT_SENTINEL_CANDIDATES and must not be treated as disguised
    missingness."""
    df = pd.DataFrame({"purchases": [0, 0, 3, 5, 0, 2, 0, 8, 0, 1]})

    hits = edacore.profiling.detect_sentinel_values(df)
    assert "purchases" not in hits


def test_strong_but_legitimate_predictor_is_not_flagged_as_leakage() -> None:
    """A real, noisy, strongly-correlated predictor (r ~ 0.87) is not the
    same as a leaked column that duplicates the target (r >= 0.98)."""
    rng = np.random.default_rng(2)
    n = 300
    base = rng.normal(size=n)
    target = base + rng.normal(scale=0.6, size=n)
    df = pd.DataFrame(
        {
            "target": target,
            "strong_predictor": base,
            "noise": rng.normal(size=n),
        }
    )

    corr = float(df["target"].corr(df["strong_predictor"]))
    assert 0.5 < abs(corr) < 0.95, f"test fixture drifted: r={corr}"

    flags = edacore.profiling.detect_leakage_candidates(df, target="target")
    flagged = {flag.column for flag in flags}
    assert "strong_predictor" not in flagged
    assert "noise" not in flagged
