"""M0 acceptance test: the registry registers a function and renders its code."""

from __future__ import annotations

import pandas as pd
import pytest

from edacore.registry import Registry
from tests.helpers.codegen_check import assert_codegen_matches


@pytest.fixture
def fixture_df() -> pd.DataFrame:
    return pd.DataFrame({"income": [10.0, 20.0, 30.0, 40.0], "region": ["A", "A", "B", "B"]})


def test_register_and_render_dummy_function(fixture_df: pd.DataFrame) -> None:
    registry = Registry()

    @registry.register(
        name="dummy_mean",
        kind="profile",
        stage="profile",
        tags={"dummy"},
        assumptions={"hard": ["numeric_outcome"], "soft": []},
        estimand="arithmetic mean",
        code_template="{df}[{col!r}].mean()",
    )
    def dummy_mean(df: pd.DataFrame, col: str) -> float:
        return float(df[col].mean())

    spec = registry.get("dummy_mean")
    assert spec.kind == "profile"
    assert spec.stage == "profile"
    assert spec.tags == {"dummy"}
    assert spec.assumptions.hard == ["numeric_outcome"]
    assert spec.assumptions.soft == []
    assert spec.estimand == "arithmetic mean"
    assert spec.read_only is True

    assert registry.list(stage="profile") == [spec]
    assert registry.list(kind="check") == []
    assert registry.list(tags={"dummy"}) == [spec]
    assert registry.list(tags={"nonexistent"}) == []

    code = registry.to_code("dummy_mean", {"col": "income"}, df_var="sales")
    assert code == "sales['income'].mean()"

    schema = registry.json_schema("dummy_mean")
    assert schema["properties"]["col"]["type"] == "string"
    assert schema["required"] == ["col"]
    assert "df" not in schema["properties"]

    # The exported code must reproduce what the function itself returns.
    assert_codegen_matches(registry, "dummy_mean", {"col": "income"}, fixture_df)


def test_list_filters_combine(fixture_df: pd.DataFrame) -> None:
    registry = Registry()

    @registry.register(
        name="dummy_mean",
        kind="profile",
        stage="profile",
        tags={"dummy", "numeric"},
        code_template="{df}[{col!r}].mean()",
    )
    def dummy_mean(df: pd.DataFrame, col: str) -> float:
        return float(df[col].mean())

    @registry.register(
        name="dummy_group_count",
        kind="check",
        stage="profile",
        tags={"dummy", "categorical"},
        code_template="{df}[{col!r}].nunique()",
    )
    def dummy_group_count(df: pd.DataFrame, col: str) -> int:
        return int(df[col].nunique())

    mean_spec = registry.get("dummy_mean")
    count_spec = registry.get("dummy_group_count")

    # stage alone: both match.
    assert registry.list(stage="profile") == [mean_spec, count_spec]
    # stage + kind: narrows to one.
    assert registry.list(stage="profile", kind="check") == [count_spec]
    # stage + kind + tags: still one, and a non-matching tag excludes it.
    assert registry.list(stage="profile", kind="check", tags={"dummy"}) == [count_spec]
    assert registry.list(stage="profile", kind="check", tags={"numeric"}) == []

    assert_codegen_matches(registry, "dummy_group_count", {"col": "region"}, fixture_df)


def test_register_rejects_duplicate_name() -> None:
    registry = Registry()

    @registry.register(name="dup", kind="profile", stage="profile", code_template="{df}")
    def first(df: pd.DataFrame) -> None: ...

    try:

        @registry.register(name="dup", kind="profile", stage="profile", code_template="{df}")
        def second(df: pd.DataFrame) -> None: ...
    except ValueError as exc:
        assert "dup" in str(exc)
    else:
        raise AssertionError("expected ValueError for duplicate registration")


def test_get_unknown_function_raises_key_error() -> None:
    registry = Registry()
    try:
        registry.get("does_not_exist")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for unknown function")
