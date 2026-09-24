"""Scenario tests for edacore.profiling.detect_structure.

Three shapes an analyst needs distinguished, since getting this wrong means
treating non-independent rows as independent (Section 6.9's independence
guard, and the paired_as_independent trap in Section 15.3, both depend on
this being right):

- a single regular time series (one row per timestamp, no repeating entity)
- panel data (an entity repeats across multiple timestamps)
- repeated measures (an entity repeats, but there's no time index at all —
  "repeated IDs across groups")

Each is also checked with assert_codegen_matches per M1 policy.
"""

from __future__ import annotations

import pandas as pd

import edacore
from edacore.registry import registry
from tests.helpers.codegen_check import assert_codegen_matches

_NAMESPACE = {"edacore": edacore}


def test_detect_structure_single_time_series() -> None:
    df = pd.DataFrame(
        {"date": pd.date_range("2024-01-01", periods=30, freq="D"), "value": range(30)}
    )
    report = edacore.profiling.detect_structure(df)
    assert report.structure == "time_series"
    assert report.time_index == "date"
    assert report.entity_id is None
    assert_codegen_matches(registry, "detect_structure", {}, df, namespace=_NAMESPACE)


def test_detect_structure_panel_data() -> None:
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "date": list(dates) * 4,
            "entity_id": sum(([i] * 5 for i in range(4)), []),
            "value": range(20),
        }
    )
    report = edacore.profiling.detect_structure(df)
    assert report.structure == "panel"
    assert report.time_index == "date"
    assert report.entity_id == "entity_id"
    assert_codegen_matches(registry, "detect_structure", {}, df, namespace=_NAMESPACE)


def test_detect_structure_repeated_ids_across_groups() -> None:
    """Same subject measured several times with no time column at all —
    the design most often mistaken for independent rows."""
    df = pd.DataFrame(
        {
            "patient_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "measurement": [10, 12, 11, 20, 19, 21, 15, 14, 16],
        }
    )
    report = edacore.profiling.detect_structure(df)
    assert report.structure == "repeated_measures"
    assert report.time_index is None
    assert report.entity_id == "patient_id"
    assert_codegen_matches(registry, "detect_structure", {}, df, namespace=_NAMESPACE)


def test_detect_structure_cross_sectional_no_repeats() -> None:
    df = pd.DataFrame(
        {
            "customer_id": range(1, 21),
            "age": range(20, 40),
            "region": ["A", "B"] * 10,
        }
    )
    report = edacore.profiling.detect_structure(df)
    assert report.structure == "cross_sectional"
    assert report.time_index is None
    assert report.entity_id is None
    assert_codegen_matches(registry, "detect_structure", {}, df, namespace=_NAMESPACE)
