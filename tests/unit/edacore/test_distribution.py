"""edacore.stattests.distribution vs R."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import edacore
from edacore.registry import registry
from tests.helpers.codegen_check import assert_codegen_matches
from tests.unit.edacore._r_helpers import data, ref

_NS = {"edacore": edacore}
dist = edacore.stattests.distribution


def test_anderson_ksamp_statistic_matches_r_and_capping_is_disclosed() -> None:
    df, r = data("anderson_ksamp_data"), ref("anderson_ksamp__anderson_ksamp_data")
    res = dist.anderson_ksamp(df, "value", "group")
    # R's fixture stores its printed matrix, rounded to 3 decimals.
    assert res.statistic == pytest.approx(r["tav2_v2"], abs=1e-3)
    # scipy caps at 0.001; R's true p is ~1e-13. Must be disclosed.
    assert res.p_value == 0.001 and r["p_value_v2"] < 1e-10
    assert any("capped" in w for w in res.warnings)
    assert_codegen_matches(
        registry,
        "anderson_ksamp",
        {"outcome": "value", "group": "group", "nan_policy": "omit"},
        df,
        namespace=_NS,
    )


def test_anderson_ksamp_uncapped_pvalue_has_no_cap_warning() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"v": rng.normal(size=90), "g": np.repeat(["a", "b", "c"], 30)})
    res = dist.anderson_ksamp(df, "v", "g")
    p = res.p_value or 0.0
    if 0.001 < p < 0.25:
        assert not any("capped" in w for w in res.warnings)
    else:
        assert any("capped" in w for w in res.warnings)


def _two_group(name: str) -> pd.DataFrame:
    t = data(name)
    return pd.DataFrame({"v": list(t.x) + list(t.y), "g": ["a"] * len(t) + ["b"] * len(t)})


def test_tost_matches_toster_raw_bounds() -> None:
    df = _two_group("tost_equivalence_data")
    r = ref("tost_equivalence__tost_equivalence_data")
    res = dist.tost_equivalence(
        df, "v", "g", low=r["low_bound"], high=r["high_bound"], var_equal=True
    )
    assert res.df == r["df"]
    assert res.p_value == pytest.approx(max(r["p_lower"], r["p_upper"]), rel=1e-9)
    notes = res.validity_notes[0]
    assert f"t_lower={r['t_lower']:.6g}" in notes and f"t_upper={r['t_upper']:.6g}" in notes
    assert "raw" in notes
    assert_codegen_matches(
        registry,
        "tost_equivalence",
        {
            "outcome": "v",
            "group": "g",
            "low": -1.5,
            "high": 1.5,
            "var_equal": True,
            "nan_policy": "omit",
        },
        df,
        namespace=_NS,
    )


def test_tost_rejects_inverted_bounds_and_welch_path_runs() -> None:
    df = _two_group("tost_equivalence_data")
    with pytest.raises(ValueError, match="must be less than"):
        dist.tost_equivalence(df, "v", "g", low=1.0, high=-1.0)
    assert dist.tost_equivalence(df, "v", "g", low=-1.5, high=1.5).df != 48  # Welch df


def test_runs_test_matches_randtests_fixture() -> None:
    df, r = data("runs_test_data"), ref("runs_test__runs_test_data")
    res = dist.runs_test(df, "x")
    assert res.statistic == pytest.approx(r["statistic"], abs=1e-9)
    assert res.p_value == pytest.approx(r["p_value"], abs=1e-9)
    assert res.n["above"] == r["n1"] and res.n["below"] == r["n2"]
    assert f"runs={r['n_runs']}" in res.validity_notes[0]
    assert_codegen_matches(
        registry, "runs_test", {"col": "x", "nan_policy": "omit"}, df, namespace=_NS
    )


def test_runs_test_detects_nonrandom_sequences() -> None:
    # The R fixture happens to sit exactly at z=0; also check direction.
    clustered = pd.DataFrame({"x": [1.0] * 15 + [2.0] * 15})
    alternating = pd.DataFrame({"x": [1.0, 2.0] * 15})
    c = dist.runs_test(clustered, "x")
    a = dist.runs_test(alternating, "x")
    assert c.statistic < 0 and c.p_value is not None and c.p_value < 0.01
    assert a.statistic > 0 and a.p_value is not None and a.p_value < 0.01
    with pytest.raises(ValueError, match="both sides"):
        dist.runs_test(pd.DataFrame({"x": [1.0] * 10}), "x")
