"""Distribution / equivalence / other tests (ARCHITECTURE.md, Section 6.7).
Verified against R (tests/fixtures/r_reference/).

- anderson_ksamp: scipy's statistic is the standardized T.AD of
  kSamples::ad.test (version 2, midrank/bias-corrected) and matches R.
  scipy's *p-value* is interpolated from a critical-value table and
  capped to [0.001, 0.25]; R's asymptotic p-value is not (e.g. 1e-13 on
  the module's own fixture). When the cap is hit, the p-value is replaced
  by a seeded permutation p-value (9999 resamples; resolution 1e-4, so it
  cannot reproduce R's 1e-13 either), disclosed in `warnings`.
- tost_equivalence: bounds are RAW (same units as the outcome), matching
  TOSTER::t_TOST's default `eqbound_type="raw"`, not standardized
  (Cohen's d) bounds. Overall p-value is max(p_lower, p_upper).
- runs_test: randtests::runs.test's defaults -- dichotomize at the
  median (values equal to the threshold are dropped) and use the normal
  approximation.
"""

from __future__ import annotations

import warnings as _warnings

import numpy as np
import pandas as pd
from scipy import stats

from edacore.contracts import TestResult
from edacore.registry import register
from edacore.stattests._shared import NanPolicy
from edacore.stattests.k_independent import _clean_k_groups
from edacore.stattests.two_sample import _clean_groups

# scipy's anderson_ksamp interpolates its p-value within this range only.
_AD_P_FLOOR, _AD_P_CEILING = 0.001, 0.25
_AD_N_RESAMPLES = 9999


@register(
    name="anderson_ksamp",
    kind="test",
    stage="hypothesis",
    tags={"nonparametric", "distribution", "k_independent"},
    assumptions={"hard": ["numeric_or_ordinal", "independent"], "soft": []},
    estimand="whether k samples come from a common distribution",
    code_template=(
        "edacore.stattests.distribution.anderson_ksamp("
        "{df}, outcome={outcome!r}, group={group!r}, random_state={random_state}, "
        "nan_policy={nan_policy!r})"
    ),
)
def anderson_ksamp(
    df: pd.DataFrame,
    outcome: str,
    group: str,
    random_state: int = 0,
    nan_policy: NanPolicy = "omit",
) -> TestResult:
    """`random_state` seeds the permutation p-value used only when the
    asymptotic one hits scipy's [0.001, 0.25] cap."""
    warnings: list[str] = []
    clean = _clean_k_groups(df, outcome, group, nan_policy, warnings)
    samples = [g[outcome].to_numpy(dtype=float) for _, g in clean.groupby(group, observed=True)]

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        result = stats.anderson_ksamp(samples)
    p_value = float(result.pvalue)
    if p_value <= _AD_P_FLOOR or p_value >= _AD_P_CEILING:
        method = stats.PermutationMethod(
            n_resamples=_AD_N_RESAMPLES, rng=np.random.default_rng(random_state)
        )
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            p_value = float(stats.anderson_ksamp(samples, method=method).pvalue)
        warnings.append(
            f"asymptotic p-value hit scipy's cap ({float(result.pvalue)}; its table only "
            f"covers [{_AD_P_FLOOR}, {_AD_P_CEILING}]), so the p-value is from "
            f"{_AD_N_RESAMPLES} permutations (random_state={random_state}); smallest "
            f"reportable value {1 / (_AD_N_RESAMPLES + 1):.0e}. R's kSamples::ad.test "
            "reports the uncapped asymptotic p-value instead"
        )

    return TestResult(
        fact_id=f"anderson_ksamp.{outcome}",
        function="anderson_ksamp",
        estimand=f"common distribution of '{outcome}' across groups of '{group}'",
        statistic=float(result.statistic),
        statistic_name="T_AD",
        df=float(len(samples) - 1),
        p_value=p_value,
        n={str(k): int(v) for k, v in clean.groupby(group, observed=True).size().items()},
        warnings=warnings,
    )


@register(
    name="tost_equivalence",
    kind="test",
    stage="hypothesis",
    tags={"equivalence", "two_sample", "independent"},
    assumptions={"hard": ["numeric_outcome", "independent"], "soft": ["normality_or_large_n"]},
    estimand="whether the mean difference lies within raw equivalence bounds",
    code_template=(
        "edacore.stattests.distribution.tost_equivalence("
        "{df}, outcome={outcome!r}, group={group!r}, low={low}, high={high}, "
        "var_equal={var_equal}, nan_policy={nan_policy!r})"
    ),
)
def tost_equivalence(
    df: pd.DataFrame,
    outcome: str,
    group: str,
    low: float,
    high: float,
    var_equal: bool = False,
    nan_policy: NanPolicy = "omit",
) -> TestResult:
    """`low`/`high` are RAW bounds on mean(group1) - mean(group2), in the
    outcome's own units (not standardized). Equivalence is concluded when
    p_value (the larger of the two one-sided p-values) is below alpha."""
    if low >= high:
        raise ValueError(f"low ({low}) must be less than high ({high})")
    warnings: list[str] = []
    x, y, g1, g2 = _clean_groups(df, group, outcome, nan_policy, warnings)
    n1, n2 = len(x), len(y)
    diff = float(x.mean() - y.mean())
    v1, v2 = x.var(ddof=1), y.var(ddof=1)

    dof: float
    if var_equal:
        dof = n1 + n2 - 2
        pooled = ((n1 - 1) * v1 + (n2 - 1) * v2) / dof
        se = float(np.sqrt(pooled * (1 / n1 + 1 / n2)))
    else:
        se = float(np.sqrt(v1 / n1 + v2 / n2))
        dof = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))

    t_lower = (diff - low) / se
    t_upper = (diff - high) / se
    p_lower = float(stats.t.sf(t_lower, dof))
    p_upper = float(stats.t.cdf(t_upper, dof))
    p_value = max(p_lower, p_upper)

    return TestResult(
        fact_id=f"tost_equivalence.{outcome}",
        function="tost_equivalence",
        estimand=(
            f"mean difference of '{outcome}' ('{g1}' - '{g2}') within raw bounds [{low}, {high}]"
        ),
        statistic=float(t_lower if p_lower >= p_upper else t_upper),
        statistic_name="t_tost",
        df=float(dof),
        p_value=p_value,
        estimate=diff,
        n={str(g1): n1, str(g2): n2},
        warnings=warnings,
        validity_notes=[
            f"raw (unstandardized) bounds [{low}, {high}]; "
            f"t_lower={t_lower:.6g} (p={p_lower:.6g}), t_upper={t_upper:.6g} (p={p_upper:.6g})"
        ],
    )


@register(
    name="runs_test",
    kind="test",
    stage="hypothesis",
    tags={"randomness"},
    assumptions={"hard": ["ordered_sequence"], "soft": []},
    estimand="randomness of an ordered sequence",
    code_template=(
        "edacore.stattests.distribution.runs_test({df}, col={col!r}, nan_policy={nan_policy!r})"
    ),
)
def runs_test(df: pd.DataFrame, col: str, nan_policy: NanPolicy = "omit") -> TestResult:
    """Wald-Wolfowitz runs test. Values are dichotomized at the median
    (randtests::runs.test's default threshold); values exactly equal to
    the median are dropped. Normal-approximation p-value."""
    warnings: list[str] = []
    series = df[col]
    n_missing = int(series.isna().sum())
    if n_missing:
        if nan_policy == "raise":
            raise ValueError(f"{n_missing} missing value(s) in '{col}', nan_policy=raise")
        warnings.append(f"{n_missing} missing value(s) dropped from '{col}' (nan_policy='omit')")
    x = series.dropna().to_numpy(dtype=float)
    threshold = float(np.median(x))
    kept = x[x != threshold]
    if len(kept) < len(x):
        warnings.append(f"{len(x) - len(kept)} value(s) equal to the median threshold dropped")
    signs = kept > threshold
    n1, n2 = int(signs.sum()), int((~signs).sum())
    n = n1 + n2
    if n1 == 0 or n2 == 0:
        raise ValueError("runs_test needs values on both sides of the median")
    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))

    mu = 2 * n1 * n2 / n + 1
    var = (mu - 1) * (mu - 2) / (n - 1)
    z = (runs - mu) / np.sqrt(var) if var > 0 else 0.0
    p_value = float(2 * stats.norm.sf(abs(z)))

    return TestResult(
        fact_id=f"runs_test.{col}",
        function="runs_test",
        estimand=f"randomness of '{col}' (dichotomized at the median, {threshold:g})",
        statistic=float(z),
        statistic_name="z",
        p_value=p_value,
        n={"total": n, "above": n1, "below": n2},
        warnings=warnings,
        validity_notes=[f"runs={runs}, expected={mu:.6g}"],
    )
