"""edacore.power vs R's pwr package (tests/fixtures/r_reference/
required_sample_size__*.json, minimum_detectable_effect__*.json).

pwr's required_sample_size solutions come from R's `uniroot`, whose default
tolerance (~1.2e-4 relative) is looser than the root-finder here — verified
by checking R's own reference n evaluates to power=0.80000003, not exactly
0.8, for the same formula. So required_sample_size tests use a tolerance
loose enough to absorb that (1e-4 to 1e-5, tightest that passes for each
test family); minimum_detectable_effect solves the same equation for the
effect size instead of n, converges tighter, and matches to 1e-6.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import edacore
from edacore.registry import registry
from tests.helpers.codegen_check import assert_codegen_matches_no_df

FIXTURES = Path(__file__).parents[2] / "fixtures" / "r_reference"
_NAMESPACE = {"edacore": edacore}


def _ref(name: str) -> dict[str, Any]:
    with (FIXTURES / f"{name}.json").open() as f:
        result: dict[str, Any] = json.load(f)
        return result


def test_required_sample_size_two_sample_t_matches_r() -> None:
    ref = _ref("required_sample_size__two_sample_t")
    result = edacore.power.required_sample_size(
        "two_sample_t", effect=ref["d"], alpha=ref["sig.level"], power=ref["power"]
    )
    assert result["n_per_group"] == pytest.approx(ref["n_per_group"], abs=1e-4)
    assert_codegen_matches_no_df(
        registry,
        "required_sample_size",
        {
            "test": "two_sample_t",
            "effect": ref["d"],
            "alpha": ref["sig.level"],
            "power": ref["power"],
            "k": None,
            "df": None,
        },
        namespace=_NAMESPACE,
    )


def test_required_sample_size_anova_matches_r() -> None:
    ref = _ref("required_sample_size__anova")
    result = edacore.power.required_sample_size(
        "anova", effect=ref["f"], alpha=ref["sig.level"], power=ref["power"], k=ref["k"]
    )
    assert result["n_per_group"] == pytest.approx(ref["n_per_group"], abs=1e-4)
    assert_codegen_matches_no_df(
        registry,
        "required_sample_size",
        {
            "test": "anova",
            "effect": ref["f"],
            "alpha": ref["sig.level"],
            "power": ref["power"],
            "k": ref["k"],
            "df": None,
        },
        namespace=_NAMESPACE,
    )


def test_required_sample_size_correlation_matches_r() -> None:
    ref = _ref("required_sample_size__correlation")
    result = edacore.power.required_sample_size(
        "correlation", effect=ref["r"], alpha=ref["sig.level"], power=ref["power"]
    )
    assert result["n"] == pytest.approx(ref["n"], abs=1e-5)
    assert_codegen_matches_no_df(
        registry,
        "required_sample_size",
        {
            "test": "correlation",
            "effect": ref["r"],
            "alpha": ref["sig.level"],
            "power": ref["power"],
            "k": None,
            "df": None,
        },
        namespace=_NAMESPACE,
    )


def test_required_sample_size_chi_square_matches_r() -> None:
    ref = _ref("required_sample_size__chisq")
    result = edacore.power.required_sample_size(
        "chi_square", effect=ref["w"], alpha=ref["sig.level"], power=ref["power"], df=ref["df"]
    )
    assert result["n"] == pytest.approx(ref["n"], abs=1e-5)
    assert_codegen_matches_no_df(
        registry,
        "required_sample_size",
        {
            "test": "chi_square",
            "effect": ref["w"],
            "alpha": ref["sig.level"],
            "power": ref["power"],
            "k": None,
            "df": ref["df"],
        },
        namespace=_NAMESPACE,
    )


def test_minimum_detectable_effect_two_sample_t_matches_r() -> None:
    ref = _ref("minimum_detectable_effect__two_sample_t")
    result = edacore.power.minimum_detectable_effect(
        "two_sample_t", n=ref["n"], alpha=ref["sig.level"], power=ref["power"]
    )
    assert result["d"] == pytest.approx(ref["d"], abs=1e-6)
    assert_codegen_matches_no_df(
        registry,
        "minimum_detectable_effect",
        {
            "test": "two_sample_t",
            "n": ref["n"],
            "alpha": ref["sig.level"],
            "power": ref["power"],
            "k": None,
            "df": None,
        },
        namespace=_NAMESPACE,
    )


def test_required_sample_size_anova_without_k_raises() -> None:
    with pytest.raises(ValueError, match="requires k"):
        edacore.power.required_sample_size("anova", effect=0.25)


def test_required_sample_size_chi_square_without_df_raises() -> None:
    with pytest.raises(ValueError, match="requires df"):
        edacore.power.required_sample_size("chi_square", effect=0.3)


def test_required_sample_size_unknown_test_raises() -> None:
    with pytest.raises(ValueError, match="unknown test family"):
        edacore.power.required_sample_size("not_a_real_test", effect=0.3)  # type: ignore[arg-type]


def test_larger_effect_needs_smaller_sample_size() -> None:
    small_effect_n = edacore.power.required_sample_size("two_sample_t", effect=0.2)["n_per_group"]
    large_effect_n = edacore.power.required_sample_size("two_sample_t", effect=0.8)["n_per_group"]
    assert large_effect_n < small_effect_n
