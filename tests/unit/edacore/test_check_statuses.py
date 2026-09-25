"""Every assumption check must PASS on clean data and FAIL on data built to
violate it (ARCHITECTURE.md, Section 6.6).

The existing R-fixture tests check that each function computes the right
*number*. They do not check that it reaches the right *verdict* from that
number, and the two are not the same thing: `check_monotonicity` matched R's
Spearman rho to 1e-9 for two milestones while reporting FAIL on perfectly
monotone data, because it read a small p-value as evidence against
monotonicity rather than for it. That bug was invisible to a test that only
compared statistics. This file closes that gap for all of them.

**Why "clean data must PASS" is asserted across seeds rather than on one.**
Most of these checks are hypothesis tests with a 5% size, so by construction
they reject roughly 1 clean dataset in 20, and the BORDERLINE band (0.05 <= p
< 0.10) catches ~5% more. Asserting PASS on a single hand-picked seed would
therefore be seed-shopping: it would pass for a check that is right and for
one that is wrong 60% of the time. So clean data is required to (a) never
FAIL, across ten independent seeds, and (b) PASS in at least eight of them.
An inverted check fails that immediately -- the pre-M4 `check_monotonicity`
scored 0/10.

Violating data is built to be unambiguous, so every check is required to
FAIL on all ten seeds. Anything less means the violation was too mild to be
a test of the verdict.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest

from edacore.contracts import CheckStatus, SemanticType
from edacore.registry import registry

N = 120
SEEDS = range(10)
MIN_CLEAN_PASSES = 8

Rng = np.random.Generator
Factory = Callable[[Rng], pd.DataFrame]


@dataclass(frozen=True)
class StatusCase:
    """One check, with data that satisfies it and data that violates it."""

    check: str
    params: dict[str, Any]
    clean: Factory
    violating: Factory
    violates: str
    # `check_independence_design` cannot confirm independence from values
    # (Section 5.2) -- UNTESTABLE *is* its correct clean-data verdict.
    clean_status: CheckStatus = CheckStatus.PASS
    notes: str = field(default="")


# --------------------------------------------------------------------------
# Data builders
# --------------------------------------------------------------------------


def _col(values: np.ndarray, name: str = "x") -> pd.DataFrame:
    return pd.DataFrame({name: values})


def _two_groups(rng: Rng, sd_a: float, sd_b: float, n: int = 60) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "v": np.concatenate([rng.normal(50, sd_a, n), rng.normal(50, sd_b, n)]),
            "g": ["A"] * n + ["B"] * n,
        }
    )


def _xy(rng: Rng, f: Callable[[np.ndarray, Rng], np.ndarray]) -> pd.DataFrame:
    x = np.linspace(1, 50, N)
    return pd.DataFrame({"x": x, "y": f(x, rng)})


def _repeated(rng: Rng, *, balanced: bool = True, spherical: bool = True) -> pd.DataFrame:
    n_subjects, k = 25, 4
    base = rng.normal(50, 8, n_subjects)
    sds = [3.0] * k if spherical else [1.0, 4.0, 12.0, 30.0]
    wide = pd.DataFrame(
        {
            "subject": np.arange(n_subjects),
            **{f"t{i + 1}": base + i + rng.normal(0, sds[i], n_subjects) for i in range(k)},
        }
    )
    long = wide.melt(id_vars="subject", var_name="condition", value_name="value")
    return long.drop(long.index[-1]) if not balanced else long


def _table(counts: dict[tuple[str, str], int]) -> pd.DataFrame:
    return pd.DataFrame([{"a": a, "b": b} for (a, b), k in counts.items() for _ in range(k)])


def _normal(rng: Rng) -> pd.DataFrame:
    return _col(rng.normal(50, 10, N))


def _lognormal(rng: Rng) -> pd.DataFrame:
    return _col(rng.lognormal(3, 1.5, N))


def _equal_variance(rng: Rng) -> pd.DataFrame:
    return _two_groups(rng, 10, 10)


def _unequal_variance(rng: Rng) -> pd.DataFrame:
    return _two_groups(rng, 2, 25)


def _linear(rng: Rng) -> pd.DataFrame:
    return _xy(rng, lambda x, r: 2 * x + r.normal(0, 2, len(x)))


def _u_shaped(rng: Rng) -> pd.DataFrame:
    return _xy(rng, lambda x, r: (x - 25) ** 2 + r.normal(0, 5, len(x)))


# --------------------------------------------------------------------------
# The table: one row per registered check
# --------------------------------------------------------------------------

CASES: tuple[StatusCase, ...] = (
    # --- normality -------------------------------------------------------
    StatusCase(
        "check_normality_shapiro",
        {"col": "x", "by": None},
        _normal,
        _lognormal,
        "a strongly right-skewed lognormal sample",
    ),
    StatusCase(
        "check_normality_dagostino",
        {"col": "x", "by": None},
        _normal,
        _lognormal,
        "a strongly right-skewed lognormal sample",
    ),
    StatusCase(
        "check_normality_anderson",
        {"col": "x", "by": None},
        _normal,
        _lognormal,
        "a strongly right-skewed lognormal sample",
    ),
    StatusCase(
        "check_normality_lilliefors",
        {"col": "x", "by": None},
        _normal,
        _lognormal,
        "a strongly right-skewed lognormal sample",
    ),
    StatusCase(
        "check_normality_descriptive",
        {"col": "x", "by": None},
        _normal,
        _lognormal,
        "skew and excess kurtosis far outside the borderline bands",
    ),
    StatusCase(
        "qq_correlation",
        {"col": "x", "by": None},
        _normal,
        _lognormal,
        "a sample whose quantiles bend away from the normal line",
    ),
    # --- variance --------------------------------------------------------
    StatusCase(
        "check_equal_variance_levene",
        {"col": "v", "group": "g"},
        _equal_variance,
        _unequal_variance,
        "group SDs differing by more than 10x",
    ),
    StatusCase(
        "check_equal_variance_bartlett",
        {"col": "v", "group": "g"},
        _equal_variance,
        _unequal_variance,
        "group SDs differing by more than 10x",
    ),
    StatusCase(
        "check_equal_variance_fligner",
        {"col": "v", "group": "g"},
        _equal_variance,
        _unequal_variance,
        "group SDs differing by more than 10x",
    ),
    StatusCase(
        "check_variance_ratio",
        {"col": "v", "group": "g"},
        _equal_variance,
        _unequal_variance,
        "an SD ratio far above the 2x descriptive threshold",
    ),
    # --- counts and structure (deterministic, no p-value) ----------------
    StatusCase(
        "check_sample_size",
        {"col": "v", "group": "g", "min_n": 2},
        _equal_variance,
        lambda rng: pd.DataFrame({"v": [1.0, 2.0, 3.0], "g": ["A", "A", "B"]}),
        "a group with a single observation",
    ),
    StatusCase(
        "check_expected_counts",
        {"a": "a", "b": "b"},
        lambda rng: _table({("x", "p"): 60, ("x", "q"): 55, ("y", "p"): 58, ("y", "q"): 62}),
        lambda rng: _table({("x", "p"): 40, ("x", "q"): 1, ("y", "p"): 1, ("y", "q"): 2}),
        "expected cell counts below 1",
    ),
    StatusCase(
        "check_expected_counts_gof",
        {"col": "a", "expected": {"x": 0.5, "y": 0.5}},
        lambda rng: pd.DataFrame({"a": ["x"] * 60 + ["y"] * 60}),
        lambda rng: pd.DataFrame({"a": ["x"] * 5 + ["y"] * 3}),
        "a total too small for the chi-square approximation",
    ),
    StatusCase(
        "check_measurement_level",
        {"col": "x", "required": SemanticType.CONTINUOUS},
        _normal,
        lambda rng: _col(rng.integers(1, 6, N)),
        "a 1-5 Likert column where a continuous one is required",
    ),
    StatusCase(
        "check_paired_structure",
        {"a": "before", "b": "after", "id_col": "sid"},
        lambda rng: pd.DataFrame(
            {
                "sid": np.arange(40),
                "before": rng.normal(0, 1, 40),
                "after": rng.normal(0, 1, 40),
            }
        ),
        lambda rng: pd.DataFrame(
            {
                "sid": [0] * 40,
                "before": rng.normal(0, 1, 40),
                "after": np.append(rng.normal(0, 1, 39), np.nan),
            }
        ),
        "duplicate subject ids and unequal non-missing counts",
    ),
    StatusCase(
        "check_design_crossing",
        {"group": "g", "id_col": "sid"},
        lambda rng: pd.DataFrame({"sid": np.arange(40), "g": ["A"] * 20 + ["B"] * 20}),
        lambda rng: pd.DataFrame({"sid": np.tile(np.arange(20), 2), "g": ["A"] * 20 + ["B"] * 20}),
        "every subject appearing under both groups",
    ),
    StatusCase(
        "check_independence_design",
        {"id_col": "sid"},
        lambda rng: pd.DataFrame({"sid": np.arange(40)}),
        lambda rng: pd.DataFrame({"sid": np.tile(np.arange(20), 2)}),
        "repeated subject ids",
        clean_status=CheckStatus.UNTESTABLE,
        notes=(
            "Independence cannot be confirmed from values (Section 5.2), so UNTESTABLE "
            "is the correct clean-data verdict -- this is the explicit ask-user handler, "
            "not a check that happens to be silent."
        ),
    ),
    StatusCase(
        "check_balanced_design",
        {"subject": "subject", "within": "condition"},
        _repeated,
        lambda rng: _repeated(rng, balanced=False),
        "a subject missing one condition",
    ),
    StatusCase(
        "check_proportion_counts",
        {"outcome": "o", "group": "g", "event": 1},
        lambda rng: pd.DataFrame(
            {"g": ["A"] * 60 + ["B"] * 60, "o": list(rng.binomial(1, 0.5, 120))}
        ),
        lambda rng: pd.DataFrame(
            {"g": ["A"] * 60 + ["B"] * 60, "o": [1] * 2 + [0] * 58 + [1] * 3 + [0] * 57}
        ),
        "fewer than 10 events in each group",
    ),
    # --- shape and relationship -----------------------------------------
    StatusCase(
        "check_same_shape",
        {"col": "v", "group": "g"},
        _equal_variance,
        lambda rng: pd.DataFrame(
            {
                "v": np.concatenate(
                    [
                        rng.normal(0, 1, 200),
                        np.concatenate([rng.normal(-3, 0.4, 100), rng.normal(3, 0.4, 100)]),
                    ]
                ),
                "g": ["A"] * 200 + ["B"] * 200,
            }
        ),
        "a unimodal group against a strongly bimodal one",
    ),
    StatusCase(
        "check_symmetry",
        {"col": "x", "by": None},
        _normal,
        _lognormal,
        "a strongly right-skewed sample",
    ),
    StatusCase(
        "check_linearity",
        {"x": "x", "y": "y"},
        _linear,
        _u_shaped,
        "a quadratic relationship",
    ),
    StatusCase(
        "check_monotonicity",
        {"x": "x", "y": "y"},
        lambda rng: _xy(rng, lambda x, r: np.log(x) + r.normal(0, 0.05, len(x))),
        _u_shaped,
        "a U shape, which Spearman's rho reports as no association at all",
    ),
    StatusCase(
        "check_homoscedasticity_bp",
        {"x": "x", "y": "y"},
        _linear,
        lambda rng: _xy(rng, lambda x, r: 2 * x + r.normal(0, 1, len(x)) * x),
        "residual spread growing with x",
    ),
    StatusCase(
        "check_influential_outliers",
        {"x": "x", "y": "y"},
        _linear,
        lambda rng: pd.DataFrame(
            {
                "x": np.append(np.linspace(1, 50, N), 400.0),
                "y": np.append(2 * np.linspace(1, 50, N) + rng.normal(0, 2, N), 5.0),
            }
        ),
        "one point far out in x pulling the fit on its own",
    ),
    # --- time series and regression diagnostics --------------------------
    StatusCase(
        "check_autocorrelation_dw",
        {"resid": "r"},
        lambda rng: _col(rng.normal(0, 1, N), "r"),
        lambda rng: _col(np.cumsum(rng.normal(0, 1, N)), "r"),
        "a random walk, whose residuals are strongly autocorrelated",
    ),
    StatusCase(
        "check_ljung_box",
        {"col": "x", "lags": 10},
        lambda rng: _col(rng.normal(0, 1, N)),
        lambda rng: _col(np.cumsum(rng.normal(0, 1, N))),
        "a random walk",
    ),
    StatusCase(
        "check_multicollinearity_vif",
        {"cols": ["x1", "x2", "x3"]},
        lambda rng: pd.DataFrame(
            {"x1": rng.normal(0, 1, N), "x2": rng.normal(0, 1, N), "x3": rng.normal(0, 1, N)}
        ),
        lambda rng: _collinear(rng),
        "a predictor that is nearly a linear combination of the others (VIF ~ 100)",
    ),
    StatusCase(
        "check_sphericity_mauchly",
        {"subject": "subject", "within": "condition", "dv": "value"},
        _repeated,
        lambda rng: _repeated(rng, spherical=False),
        "condition variances spanning a 30x range",
    ),
)


def _collinear(rng: Rng) -> pd.DataFrame:
    """x3 is nearly, but not exactly, a combination of x1 and x2.

    Exactly collinear predictors give an infinite VIF and a "poorly
    conditioned" warning from statsmodels -- which tests the linear-algebra
    edge case rather than the check's threshold. The residual noise keeps
    VIF finite (~100) and safely above the > 10 FAIL cutoff.
    """
    x1, x2 = rng.normal(0, 1, N), rng.normal(0, 1, N)
    return pd.DataFrame(
        {"x1": x1, "x2": x2, "x3": 0.9 * x1 + 0.25 * x2 + 0.1 * rng.normal(0, 1, N)}
    )


_BY_NAME = {case.check: case for case in CASES}


def _statuses(case: StatusCase, df: pd.DataFrame) -> list[CheckStatus]:
    result = registry.get(case.check).func(df, **case.params)
    items = result if isinstance(result, list) else [result]
    return [item.status for item in items]


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_every_registered_check_has_a_status_case() -> None:
    """A check with no row here could return any status on any data and
    nothing would notice."""
    registered = {spec.name for spec in registry.list(stage="hypothesis", kind="check")}
    assert registered == set(_BY_NAME), {
        "registered_without_a_case": sorted(registered - set(_BY_NAME)),
        "case_without_a_check": sorted(set(_BY_NAME) - registered),
    }


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.check)
def test_check_passes_on_clean_data(case: StatusCase) -> None:
    """Never FAIL on clean data, and PASS on at least 8 of 10 seeds.

    Not 10 of 10: these are 5%-size tests, so they reject clean data about
    once in twenty runs by construction, and a BORDERLINE band catches
    another slice. Demanding a perfect record would only prove that one
    seed was chosen carefully.
    """
    observed = [
        status
        for seed in SEEDS
        for status in _statuses(case, case.clean(np.random.default_rng(seed)))
    ]
    failures = [s for s in observed if s is CheckStatus.FAIL]
    assert not failures, (
        f"{case.check} reported FAIL on clean data "
        f"({len(failures)}/{len(observed)} results across {len(list(SEEDS))} seeds)"
    )
    expected = sum(1 for s in observed if s is case.clean_status)
    assert expected >= MIN_CLEAN_PASSES, (
        f"{case.check} reached {case.clean_status.value} only {expected}/{len(observed)} "
        f"times on clean data; the rest were "
        f"{sorted({s.value for s in observed if s is not case.clean_status})}"
    )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c.check)
def test_check_fails_on_violating_data(case: StatusCase) -> None:
    """FAIL on every seed. The violation is built to be unambiguous, so
    anything less means the data was too mild to test the verdict rather
    than that the check is lenient."""
    observed = [
        status
        for seed in SEEDS
        for status in _statuses(case, case.violating(np.random.default_rng(seed)))
    ]
    not_failed = [s.value for s in observed if s is not CheckStatus.FAIL]
    assert not not_failed, (
        f"{case.check} did not FAIL on data with {case.violates}; "
        f"got {sorted(set(not_failed))} on {len(not_failed)}/{len(observed)} results"
    )


def test_monotonicity_regression_inverted_verdict() -> None:
    """The specific bug this file exists for, pinned separately so it cannot
    be lost in a table edit.

    Before M4, `check_monotonicity` used Spearman's own p-value as its
    verdict. A small p means a STRONG monotone association, so the most
    perfectly monotone data possible was reported as failing monotonicity --
    and a U shape, which is the real failure, passed.
    """
    rng = np.random.default_rng(0)
    x = np.linspace(1, 50, N)
    monotone = pd.DataFrame({"x": x, "y": np.log(x) + rng.normal(0, 0.05, N)})
    u_shape = pd.DataFrame({"x": x, "y": (x - 25) ** 2 + rng.normal(0, 5, N)})

    check = registry.get("check_monotonicity").func
    assert check(monotone, x="x", y="y").status is CheckStatus.PASS
    assert check(u_shape, x="x", y="y").status is CheckStatus.FAIL
