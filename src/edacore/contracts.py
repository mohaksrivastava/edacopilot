"""Shared statistical data contracts (ARCHITECTURE.md, Section 5).

These are the objects that flow between the eligibility engine, the persona
engine, and the UI. They are frozen: a fact about the data, once computed, is
never mutated in place — a new version is created instead (see
ProvenanceLog, Section 12).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator


def to_jsonable(value: Any) -> Any:
    """Convert numpy scalars and containers of them to plain Python.

    Any field typed `Any` can end up holding a `numpy.int64` -- a method
    param built from a column's own values, a summary statistic taken
    straight from pandas -- and pydantic cannot serialise one. The failure
    surfaces far from the cause: not when the value is stored, but later,
    when `session.json` is written, as
    `PydanticSerializationError: Unable to serialize unknown type`.

    Section 12.4 makes that a correctness problem rather than an
    inconvenience: a session that cannot be written cannot be resumed, and
    the step that could not be saved is the one the user just accepted. So
    the conversion happens at the boundary, on the way in.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [to_jsonable(item) for item in value]
        return type(value)(converted) if isinstance(value, tuple) else converted
    return value


class SemanticType(StrEnum):
    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    BINARY = "binary"
    NOMINAL = "nominal"
    ORDINAL = "ordinal"
    DATETIME = "datetime"
    TEXT = "text"
    IDENTIFIER = "identifier"
    CONSTANT = "constant"
    MIXED = "mixed"


class ColumnProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    dtype: str
    semantic_type: SemanticType
    semantic_confidence: float
    n: int
    n_missing: int
    n_unique: int
    sentinel_candidates: list[Any] = Field(default_factory=list)
    summary: dict[str, float | int | str] = Field(default_factory=dict)
    flags: list[str] = Field(default_factory=list)


class DatasetProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_rows: int
    n_cols: int
    columns: list[ColumnProfile]
    structure: Literal["cross_sectional", "time_series", "panel", "repeated_measures", "unknown"]
    time_index: str | None = None
    entity_id: str | None = None
    target: str | None = None
    duplicate_rows: int
    leakage_candidates: list[str] = Field(default_factory=list)
    pii_columns: list[str] = Field(default_factory=list)


class CheckStatus(StrEnum):
    PASS = "pass"
    BORDERLINE = "borderline"
    FAIL = "fail"
    NOT_APPLICABLE = "n/a"
    UNTESTABLE = "untestable"


class AssumptionCheck(BaseModel):
    model_config = ConfigDict(frozen=True)

    fact_id: str
    assumption: str
    method: str
    scope: dict[str, str] = Field(default_factory=dict)
    statistic: float | None = None
    p_value: float | None = None
    threshold: str
    status: CheckStatus
    consequence: str
    plot_ref: str | None = None


class Eligibility(StrEnum):
    ELIGIBLE = "eligible"
    CAVEAT = "caveat"
    INELIGIBLE = "ineligible"


class Candidate(BaseModel):
    model_config = ConfigDict(frozen=True)

    function: str
    params: dict[str, Any] = Field(default_factory=dict)
    eligibility: Eligibility
    reasons: list[str] = Field(default_factory=list)
    tags: set[str] = Field(default_factory=set)
    estimand: str

    @field_validator("params", mode="before")
    @classmethod
    def _plain_python_params(cls, params: Any) -> Any:
        """A param taken from the data can be a numpy scalar -- see
        `to_jsonable`. Converted here so a `Candidate` is always writable."""
        return to_jsonable(params)

    @field_serializer("tags")
    def _sorted_tags(self, tags: set[str]) -> list[str]:
        """Serialise tags in a fixed order.

        A `set` dumps in CPython's iteration order, which for two sets with
        the same members can differ if their insertion histories collided
        differently in the hash table -- and whether that happens depends on
        the per-process string hash seed. That made a serialised `Candidate`
        non-canonical: a session written to JSON and read back produced an
        equal model that dumped to different bytes, roughly one process in
        ten. Sorting makes serialisation a function of the value alone,
        which is what persistence (Section 12.4), state comparison and
        readable diffs all depend on.
        """
        return sorted(tags)


class TestResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    fact_id: str
    function: str
    estimand: str
    statistic: float
    statistic_name: str
    df: float | tuple[float, float] | None = None
    p_value: float | None = None
    p_adjusted: float | None = None
    estimate: float | None = None
    ci: tuple[float, float] | None = None
    ci_level: float = 0.95
    effect_size: float | None = None
    effect_size_name: str | None = None
    effect_size_ci: tuple[float, float] | None = None
    effect_magnitude: Literal["negligible", "small", "medium", "large"] | None = None
    n: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    validity_notes: list[str] = Field(default_factory=list)


class PairwiseComparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    group_a: str
    group_b: str
    statistic: float | None = None
    p_value: float
    p_adjusted: float | None = None
    estimate: float | None = None
    ci: tuple[float, float] | None = None


class PostHocResult(BaseModel):
    """A k-group post-hoc procedure's pairwise comparisons (Section 6.7's
    k-group tables) — not spelled out as a model in Section 5, added here
    the same way DuplicateReport/StructureReport/LeakageFlag were for M1:
    a rich return type in place of a bare list of tuples (rule 6)."""

    model_config = ConfigDict(frozen=True)

    fact_id: str
    function: str
    method: str
    p_adjust_method: str | None = None
    comparisons: list[PairwiseComparison]
    n: dict[str, int] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class TransformRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    function: str
    params: dict[str, Any] = Field(default_factory=dict)
    columns_affected: list[str]
    rows_before: int
    rows_after: int
    summary_before: dict[str, Any] = Field(default_factory=dict)
    summary_after: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)

    @field_validator("params", "summary_before", "summary_after", mode="before")
    @classmethod
    def _plain_python(cls, value: Any) -> Any:
        return to_jsonable(value)


# --- Profiling-stage result types (Section 6.1) -----------------------------
# Section 5 doesn't spell these out, but Section 6.1 names them as the return
# types of detect_duplicates, detect_structure, and detect_leakage_candidates,
# and rule 6 requires rich return objects rather than bare tuples.


class DuplicateReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    n_exact: int
    exact_duplicate_indices: list[int] = Field(default_factory=list)
    key_columns: list[str] | None = None
    n_key_duplicates: int = 0
    key_duplicate_indices: list[int] = Field(default_factory=list)


class StructureReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    structure: Literal["cross_sectional", "time_series", "panel", "repeated_measures", "unknown"]
    time_index: str | None = None
    entity_id: str | None = None
    confidence: float
    reasons: list[str] = Field(default_factory=list)


class LeakageFlag(BaseModel):
    model_config = ConfigDict(frozen=True)

    column: str
    reason: str
    method: str
    score: float | None = None
