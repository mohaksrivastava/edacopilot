"""Every BaseModel in ARCHITECTURE.md Section 5 round-trips through
model_dump() / model_validate(), and every enum member survives it too.

Section 5 defines: SemanticType, ColumnProfile, DatasetProfile (5.1);
CheckStatus, AssumptionCheck (5.2); Eligibility, Candidate, TestResult (5.3);
TransformRecord (5.3). All nine live in edacore.contracts.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from edacore.contracts import (
    AssumptionCheck,
    Candidate,
    CheckStatus,
    ColumnProfile,
    DatasetProfile,
    Eligibility,
    SemanticType,
    TestResult,
    TransformRecord,
)


def _round_trip(model: BaseModel) -> None:
    cls = type(model)
    restored = cls.model_validate(model.model_dump())
    assert restored == model


def test_column_profile_round_trip() -> None:
    profile = ColumnProfile(
        name="income",
        dtype="float64",
        semantic_type=SemanticType.CONTINUOUS,
        semantic_confidence=0.95,
        n=100,
        n_missing=3,
        n_unique=97,
        sentinel_candidates=[-999, "NA"],
        summary={"mean": 42.5, "n": 100},
        flags=["high_cardinality"],
    )
    _round_trip(profile)
    assert profile.model_dump()["semantic_type"] == "continuous"


def test_dataset_profile_round_trip() -> None:
    column = ColumnProfile(
        name="income",
        dtype="float64",
        semantic_type=SemanticType.CONTINUOUS,
        semantic_confidence=0.95,
        n=100,
        n_missing=3,
        n_unique=97,
    )
    profile = DatasetProfile(
        n_rows=100,
        n_cols=1,
        columns=[column],
        structure="cross_sectional",
        time_index=None,
        entity_id=None,
        target="income",
        duplicate_rows=0,
        leakage_candidates=[],
        pii_columns=[],
    )
    _round_trip(profile)


def test_assumption_check_round_trip_and_frozen() -> None:
    check = AssumptionCheck(
        fact_id="normality.shapiro.group=A",
        assumption="normality",
        method="shapiro_wilk",
        scope={"variable": "income", "group": "A"},
        statistic=0.86,
        p_value=0.001,
        threshold="p < 0.05 -> fail",
        status=CheckStatus.PASS,
        consequence="Nothing goes wrong; normality holds.",
    )
    _round_trip(check)
    assert check.model_dump()["status"] == "pass"

    with pytest.raises(ValidationError):
        check.status = CheckStatus.FAIL


def test_candidate_round_trip() -> None:
    candidate = Candidate(
        function="welch_t_test",
        params={"outcome": "income", "group": "region"},
        eligibility=Eligibility.CAVEAT,
        reasons=["normality.shapiro.group=A"],
        tags={"parametric", "robust_to_unequal_var"},
        estimand="difference in means",
    )
    _round_trip(candidate)
    assert candidate.model_dump()["eligibility"] == "caveat"


def test_test_result_round_trip() -> None:
    result = TestResult(
        fact_id="welch_t.income.region",
        function="welch_t_test",
        estimand="difference in means",
        statistic=2.31,
        statistic_name="t",
        df=(38.4, 41.0),
        p_value=0.02,
        p_adjusted=0.04,
        estimate=5.2,
        ci=(0.8, 9.6),
        effect_size=0.51,
        effect_size_name="hedges_g",
        effect_size_ci=(0.1, 0.9),
        effect_magnitude="medium",
        n={"A": 38, "B": 41},
        warnings=["2 rows dropped (nan_policy=omit)"],
        validity_notes=["Association only; not causation."],
    )
    _round_trip(result)


def test_transform_record_round_trip() -> None:
    record = TransformRecord(
        function="winsorize",
        params={"limits": (0.01, 0.01)},
        columns_affected=["income"],
        rows_before=100,
        rows_after=100,
        summary_before={"max": 999999},
        summary_after={"max": 120000},
        warnings=["2 values capped at the 99th percentile"],
    )
    _round_trip(record)


@pytest.mark.parametrize("member", list(SemanticType))
def test_semantic_type_members_round_trip(member: SemanticType) -> None:
    assert SemanticType(member.value) is member


@pytest.mark.parametrize("member", list(CheckStatus))
def test_check_status_members_round_trip(member: CheckStatus) -> None:
    assert CheckStatus(member.value) is member


@pytest.mark.parametrize("member", list(Eligibility))
def test_eligibility_members_round_trip(member: Eligibility) -> None:
    assert Eligibility(member.value) is member
