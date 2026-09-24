"""Tests for edacore.profiling (Section 6.1, stage `profile`).

Every one of the 17 registered functions here is also run through
assert_codegen_matches: registry.to_code(...) must reproduce what the
function itself returns (rule 7), as required for every M1 function.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import edacore
from edacore.contracts import SemanticType
from edacore.profiling import DEFAULT_SENTINEL_CANDIDATES
from edacore.registry import registry
from tests.helpers.codegen_check import assert_codegen_matches

_NAMESPACE = {"edacore": edacore}


def _check(name: str, params: dict[str, Any], df: pd.DataFrame) -> None:
    assert_codegen_matches(registry, name, params, df, namespace=_NAMESPACE)


@pytest.fixture
def mixed_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": range(1, 21),
            "age": [23, 45, 31, 60, 29, 40, 35, 28, 50, 33, 41, 27, 38, 52, 30, 44, 36, 25, 48, 39],
            "satisfaction": [1, 2, 3, 4, 5] * 4,
            "region": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"] * 2,
            "signup_date": pd.date_range("2024-01-01", periods=20, freq="D"),
        }
    )


def test_infer_semantic_types(mixed_df: pd.DataFrame) -> None:
    types = edacore.profiling.infer_semantic_types(mixed_df)
    assert types["customer_id"][0] is SemanticType.IDENTIFIER
    assert types["age"][0] is SemanticType.DISCRETE
    assert types["satisfaction"][0] is SemanticType.ORDINAL
    assert types["region"][0] is SemanticType.NOMINAL
    assert types["signup_date"][0] is SemanticType.DATETIME
    _check("infer_semantic_types", {}, mixed_df)


def test_detect_identifier_columns(mixed_df: pd.DataFrame) -> None:
    assert edacore.profiling.detect_identifier_columns(mixed_df) == ["customer_id"]
    _check("detect_identifier_columns", {}, mixed_df)


def test_detect_numeric_coded_categoricals(mixed_df: pd.DataFrame) -> None:
    result = edacore.profiling.detect_numeric_coded_categoricals(mixed_df, max_unique=15)
    assert "satisfaction" in result
    assert "age" not in result
    assert "customer_id" not in result
    _check("detect_numeric_coded_categoricals", {"max_unique": 15}, mixed_df)


def test_detect_ordinal_candidates(mixed_df: pd.DataFrame) -> None:
    assert edacore.profiling.detect_ordinal_candidates(mixed_df) == ["satisfaction"]
    _check("detect_ordinal_candidates", {}, mixed_df)


def test_detect_ordinal_candidates_text_vocabulary() -> None:
    df = pd.DataFrame({"rating": ["low", "medium", "high", "low", "high", "medium"]})
    assert edacore.profiling.detect_ordinal_candidates(df) == ["rating"]
    _check("detect_ordinal_candidates", {}, df)


def test_detect_mixed_types() -> None:
    df = pd.DataFrame({"col": pd.Series([1, "two", 3.0, "four", 5], dtype=object)})
    result = edacore.profiling.detect_mixed_types(df)
    assert "col" in result
    _check("detect_mixed_types", {}, df)


def test_detect_sentinel_values() -> None:
    df = pd.DataFrame({"income": [50000, -999, 60000, -999, 70000]})
    result = edacore.profiling.detect_sentinel_values(df)
    assert result["income"] == [-999]
    _check("detect_sentinel_values", {"candidates": DEFAULT_SENTINEL_CANDIDATES}, df)


def test_detect_constant_columns() -> None:
    df = pd.DataFrame({"flag": [1] * 19 + [2], "mix": list(range(20))})
    result = edacore.profiling.detect_constant_columns(df, near_zero_var_ratio=0.9)
    assert "flag" in result
    assert "mix" not in result
    _check("detect_constant_columns", {"near_zero_var_ratio": 0.9}, df)


def test_detect_duplicates() -> None:
    df = pd.DataFrame({"a": [1, 1, 2, 3], "b": [1, 1, 2, 4]})

    report = edacore.profiling.detect_duplicates(df)
    assert report.n_exact == 2
    _check("detect_duplicates", {"subset": None}, df)

    report_subset = edacore.profiling.detect_duplicates(df, subset=["a"])
    assert report_subset.n_key_duplicates == 2
    _check("detect_duplicates", {"subset": ["a"]}, df)


def test_detect_leakage_candidates_name_match() -> None:
    df = pd.DataFrame(
        {
            "outcome": [0, 1] * 5,
            "outcome_flag": [0, 1] * 5,
            "unrelated": [5, 3, 8, 1, 9, 2, 7, 4, 6, 0],
        }
    )
    flags = edacore.profiling.detect_leakage_candidates(df, target="outcome")
    flagged = {f.column for f in flags}
    assert "outcome_flag" in flagged
    assert "unrelated" not in flagged
    _check("detect_leakage_candidates", {"target": "outcome"}, df)


def test_detect_leakage_candidates_correlation() -> None:
    rng = np.random.default_rng(0)
    target = rng.normal(size=50)
    df = pd.DataFrame(
        {
            "target": target,
            "leaked": target * 2 + rng.normal(scale=1e-9, size=50),
            "noise": rng.normal(size=50),
        }
    )
    flags = edacore.profiling.detect_leakage_candidates(df, target="target")
    flagged = {f.column for f in flags}
    assert "leaked" in flagged
    assert "noise" not in flagged
    _check("detect_leakage_candidates", {"target": "target"}, df)


def test_detect_weight_columns() -> None:
    df = pd.DataFrame({"sample_weight": [1.0] * 5, "income": [1, 2, 3, 4, 5]})
    assert edacore.profiling.detect_weight_columns(df) == ["sample_weight"]
    _check("detect_weight_columns", {}, df)


def test_detect_pii() -> None:
    df = pd.DataFrame(
        {
            "email": [f"user{i}@example.com" for i in range(10)],
            "phone": [f"555-123-{1000 + i}" for i in range(10)],
            "notes": ["hello world"] * 10,
        }
    )
    result = edacore.profiling.detect_pii(df)
    assert result.get("email") == ["email"]
    assert result.get("phone") == ["phone"]
    assert "notes" not in result
    _check("detect_pii", {"columns": None}, df)


def test_summarize_numeric() -> None:
    df = pd.DataFrame({"income": [10.0, 20.0, 30.0, 40.0, np.nan]})
    result = edacore.profiling.summarize_numeric(df, "income")
    assert result["n"] == 5
    assert result["n_missing"] == 1
    assert result["mean"] == pytest.approx(25.0)
    _check("summarize_numeric", {"col": "income"}, df)


def test_summarize_categorical() -> None:
    df = pd.DataFrame({"region": ["A", "A", "B", "C"]})
    result = edacore.profiling.summarize_categorical(df, "region")
    assert result["n_levels"] == 3
    _check("summarize_categorical", {"col": "region"}, df)


def test_summarize_datetime() -> None:
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10, freq="D")})
    result = edacore.profiling.summarize_datetime(df, "date")
    assert result["n"] == 10
    _check("summarize_datetime", {"col": "date"}, df)


def test_summarize_text() -> None:
    df = pd.DataFrame({"comment": ["a short note", "another longer note about the product"]})
    result = edacore.profiling.summarize_text(df, "comment")
    assert result["n"] == 2
    _check("summarize_text", {"col": "comment"}, df)


def test_profile_dataset(mixed_df: pd.DataFrame) -> None:
    profile = edacore.profiling.profile_dataset(mixed_df, target=None)
    assert profile.n_rows == 20
    assert profile.n_cols == 5
    assert {c.name for c in profile.columns} == set(mixed_df.columns)
    _check("profile_dataset", {"target": None}, mixed_df)


def test_profile_dataset_with_target() -> None:
    df = pd.DataFrame({"outcome": [0, 1] * 10, "outcome_copy": [0, 1] * 10})
    profile = edacore.profiling.profile_dataset(df, target="outcome")
    assert "outcome_copy" in profile.leakage_candidates
    _check("profile_dataset", {"target": "outcome"}, df)


def test_all_profile_stage_functions_are_registered() -> None:
    names = {spec.name for spec in registry.list(stage="profile")}
    assert names == {
        "profile_dataset",
        "infer_semantic_types",
        "detect_identifier_columns",
        "detect_numeric_coded_categoricals",
        "detect_ordinal_candidates",
        "detect_mixed_types",
        "detect_sentinel_values",
        "detect_constant_columns",
        "detect_duplicates",
        "detect_structure",
        "detect_leakage_candidates",
        "detect_weight_columns",
        "detect_pii",
        "summarize_numeric",
        "summarize_categorical",
        "summarize_datetime",
        "summarize_text",
    }
