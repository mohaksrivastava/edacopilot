"""Correlation hypothesis tests (ARCHITECTURE.md, Section 6.7).

Verified against R (tests/fixtures/r_reference/) except mutual_information
(no R reference; tested against the analytic bivariate-normal formula).

- pearson: CI via Fisher z (scipy's own `pearsonr(...).confidence_interval()`
  already uses this, confirmed matching R's `cor.test` exactly).
- spearman: without ties, R's `cor.test` (exact=NULL) calls C_pRho for
  every n<=1290: full enumeration of the null distribution of
  S = sum(d^2) for n<=9, and the AS 89 Edgeworth series (Best & Roberts
  1975, Appl. Statist. 24:377) above that. scipy's `spearmanr` only has
  the t approximation (~2x off R at n=8), so `_prho` implements AS 89
  from the published algorithm, with R's calling convention
  (`round(q) + 2*lower_tail`, two-sided = min(2*tail, 1)). Verified
  against R at n=8, 9, 20, 49, 100. With ties, or n>1290, R and scipy
  both use the t approximation.
- kendall_tau: without ties and n<50, R uses the exact null distribution
  of the concordance count T; scipy's `kendalltau(method='exact')`
  computes the same distribution. Otherwise both use the normal
  approximation with R's tie-corrected variance.
- odds_ratio-style iterative solves aren't used here; the noncentral-F
  CI machinery is (see effect_sizes.py).
- partial_correlation: ppcor::pcor.test's general (any number of control
  variables) formula, ported directly: partial correlation from the
  inverse of the full correlation matrix (precision matrix), not
  iterative residualization. Verified to ~1e-9 against R.
- distance_correlation: energy::dcor's exact statistic formula (Szekely &
  Rizzo), verified to ~1e-9. Its own permutation p-value is NOT
  R-comparable (different RNG); tested by simulation behavior instead
  (test_correlation.py), not a fixture.
- mutual_information: no R reference at all. KSG (Kraskov-Stogbauer-
  Grassberger 2004) k-NN estimator on `scipy.spatial.cKDTree`, with
  sklearn's `mutual_info_regression` preprocessing (unit-variance scaling,
  seeded tie-breaking jitter). Kept in-house (written when sklearn could
  not be imported on the maintainer's Windows machine) and tested against
  sklearn, which it matches to ~1e-12 on continuous data.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations, permutations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from scipy.special import digamma
from statsmodels.stats.multitest import multipletests

from edacore.contracts import TestResult
from edacore.effect_sizes import _fisher_z_ci, magnitude_label
from edacore.registry import register
from edacore.stattests._shared import NanPolicy


def _clean_pair(
    df: pd.DataFrame, x: str, y: str, nan_policy: NanPolicy, warnings: list[str]
) -> pd.DataFrame:
    sub = df[[x, y]]
    n_before = len(sub)
    clean = sub.dropna()
    n_dropped = n_before - len(clean)
    if n_dropped > 0:
        if nan_policy == "raise":
            raise ValueError(f"{n_dropped} row(s) missing '{x}'/'{y}', nan_policy=raise")
        warnings.append(f"{n_dropped} row(s) dropped: missing '{x}' or '{y}' (nan_policy='omit')")
    return clean


def _spearman_rho_ci(rho: float, n: int, ci: float) -> tuple[float, float]:
    """CI for Spearman's rho. R's `cor.test` reports none, so the reference
    is `DescTools::SpearmanRho(conf.level=)`: Fisher z with
    SE = 1 / sqrt(n - 3), clamped to [-1, 1]. Verified to ~1e-16.
    """
    return _fisher_z_ci(rho, 1 / np.sqrt(n - 3), ci)


# Above this many contingency-table cells, tau-b's delta-method CI would
# allocate more memory than it is worth: the table is (distinct x) x
# (distinct y), so tie-free continuous data makes it n x n. At the cap the
# working set is roughly 100 MB, reached at about n = 2000 tie-free
# observations; ordinal/tied data (what DescTools::KendallTauB is actually
# meant for) stays far below it at any n.
MAX_KENDALL_TABLE_CELLS = 4_000_000


def _kendall_concordance_excess(table: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-cell (concordant - discordant) pair counts, and their total
    (C - D), as `DescTools::ConDisPairs` computes them: for each cell, the
    observations in the two quadrants that agree with it (up-left plus
    down-right) minus the two that disagree (up-right plus down-left).

    Only the difference is returned because only the difference is used --
    by tau-b itself and by every term of the delta-method variance -- which
    halves the temporaries this has to hold. DescTools re-sums a quadrant
    per cell, O((r*c)^2); this gets the same numbers from one 2-D
    cumulative sum.
    """
    n_rows, n_cols = table.shape
    cum = np.zeros((n_rows + 1, n_cols + 1))
    cum[1:, 1:] = table.cumsum(axis=0).cumsum(axis=1)
    total = cum[n_rows, n_cols]

    excess = cum[:n_rows, :n_cols].copy()  # up-left
    excess += total - cum[1:, n_cols][:, None] - cum[n_rows, 1:][None, :] + cum[1:, 1:]
    excess -= cum[:n_rows, n_cols][:, None] - cum[:n_rows, 1:]  # up-right
    excess -= cum[n_rows, :n_cols][None, :] - cum[1:, :n_cols]  # down-left
    return excess, float((excess * table).sum() / 2)


def _kendall_tau_b_ci(
    x: np.ndarray, y: np.ndarray, tau_b: float, ci: float, warnings: list[str]
) -> tuple[float, float]:
    """CI for tau-b, ported from `DescTools::KendallTauB(conf.level=)`.
    R's `cor.test` reports no interval for tau.

    The standard error is the delta-method asymptotic variance of tau-b
    computed from the joint contingency table (the SAS/SPSS "ASE"), which
    -- unlike a Fisher-z interval -- accounts for ties. Verified to ~1e-16
    against R on both a tied table and continuous (tie-free) data.

    Above `MAX_KENDALL_TABLE_CELLS` that table is too large to build, and
    the interval falls back to Fisher z with SE = 1/sqrt(n - 3),
    disclosed in `warnings`. That fallback ignores ties -- but a table
    that large is one with hardly any ties to account for, which is
    exactly when the two agree most closely.
    """
    n_obs = len(x)
    n_distinct_x, n_distinct_y = len(np.unique(x)), len(np.unique(y))
    if n_distinct_x * n_distinct_y > MAX_KENDALL_TABLE_CELLS:
        warnings.append(
            f"{n_distinct_x} x {n_distinct_y} distinct value pairs exceeds "
            f"{MAX_KENDALL_TABLE_CELLS:,} cells, so the CI is a Fisher-z approximation "
            "(SE = 1/sqrt(n - 3)) rather than DescTools::KendallTauB's tie-aware "
            "delta-method interval"
        )
        return _fisher_z_ci(tau_b, 1 / np.sqrt(n_obs - 3), ci)

    table = pd.crosstab(pd.Series(x), pd.Series(y)).to_numpy(dtype=float)
    excess, conc_minus_disc = _kendall_concordance_excess(table)
    n = float(table.sum())

    props = table / n
    p_diff = excess / n
    big_p_diff = 2 * conc_minus_disc / n**2
    row_p, col_p = props.sum(axis=1), props.sum(axis=0)
    delta1 = np.sqrt(1 - (row_p**2).sum())
    delta2 = np.sqrt(1 - (col_p**2).sum())
    tau_phi = (2 * p_diff + big_p_diff * col_p[None, :]) * delta2 * delta1 + (
        big_p_diff * row_p[:, None] * delta2
    ) / delta1
    variance = (
        ((props * tau_phi**2).sum() - (props * tau_phi).sum() ** 2) / (delta1 * delta2) ** 4
    ) / n
    if variance < np.finfo(float).eps * 10:
        variance = 0.0

    z = stats.norm.ppf(1 - (1 - ci) / 2)
    half_width = z * np.sqrt(variance)
    return float(max(tau_b - half_width, -1.0)), float(min(tau_b + half_width, 1.0))


def _kendall_tau_b_from_table(x: np.ndarray, y: np.ndarray) -> float:
    """tau-b recomputed from the contingency table, for cross-checking
    scipy's rank-based value (see `kendall_tau`). Small tables only."""
    table = pd.crosstab(pd.Series(x), pd.Series(y)).to_numpy(dtype=float)
    _, conc_minus_disc = _kendall_concordance_excess(table)
    n = float(table.sum())
    n0 = n * (n - 1) / 2
    row_totals, col_totals = table.sum(axis=1), table.sum(axis=0)
    ties_r = float((row_totals * (row_totals - 1) / 2).sum())
    ties_c = float((col_totals * (col_totals - 1) / 2).sum())
    return float(conc_minus_disc / np.sqrt((n0 - ties_r) * (n0 - ties_c)))


@register(
    name="pearson",
    kind="test",
    stage="hypothesis",
    tags={"correlation"},
    assumptions={
        "hard": ["numeric"],
        "soft": ["linearity", "bivariate_normality", "no_influential_outliers"],
    },
    estimand="linear association between two continuous variables",
    code_template=(
        "edacore.stattests.correlation.pearson("
        "{df}, x={x!r}, y={y!r}, ci={ci}, nan_policy={nan_policy!r})"
    ),
)
def pearson(
    df: pd.DataFrame, x: str, y: str, ci: float = 0.95, nan_policy: NanPolicy = "omit"
) -> TestResult:
    warnings: list[str] = []
    clean = _clean_pair(df, x, y, nan_policy, warnings)

    result = stats.pearsonr(clean[x].to_numpy(), clean[y].to_numpy())
    ci_bounds = result.confidence_interval(ci)
    n = len(clean)
    r = float(result.statistic)
    dof = n - 2
    # R's cor.test statistic is t = sqrt(df)*r/sqrt(1-r^2), not r itself
    # (scipy's own .statistic is r) -- matching this codebase's own
    # convention of `statistic` being the distributional test statistic
    # that drives p_value, with the effect size in `estimate`.
    t_stat = float(np.sqrt(dof) * r / np.sqrt(1 - r**2))

    return TestResult(
        fact_id=f"pearson.{x}.{y}",
        function="pearson",
        estimand=f"linear correlation between '{x}' and '{y}'",
        statistic=t_stat,
        statistic_name="t",
        df=float(dof),
        p_value=float(result.pvalue),
        estimate=r,
        ci=(float(ci_bounds.low), float(ci_bounds.high)),
        ci_level=ci,
        # r IS the effect size for a correlation, so it shares the
        # estimate's Fisher-z interval rather than getting a second one.
        effect_size=r,
        effect_size_name="pearson_r",
        effect_size_ci=(float(ci_bounds.low), float(ci_bounds.high)),
        effect_magnitude=magnitude_label(r, "pearson_r"),
        n={"total": n},
        warnings=warnings,
    )


# Largest n for which R's C_pRho enumerates the null distribution of S;
# above it, the AS 89 Edgeworth series. R stays in C_pRho up to n=1290.
_PRHO_N_EXACT = 9
_PRHO_N_MAX = 1290
# AS 89 Edgeworth-series coefficients (Best & Roberts 1975).
_AS89_C = (
    0.2274,
    0.2531,
    0.1745,
    0.0758,
    0.1033,
    0.3932,
    0.0879,
    0.0151,
    0.0072,
    0.0831,
    0.0131,
    4.6e-4,
)


@lru_cache(maxsize=_PRHO_N_EXACT)
def _spearman_s_null(n: int) -> np.ndarray:
    """Sorted S = sum((i - p_i)^2) over all n! permutations p (the exact
    null distribution of Spearman's S, no ties)."""
    perms = np.array(list(permutations(range(n))), dtype=np.int64)
    return np.sort(((perms - np.arange(n)) ** 2).sum(axis=1))


def _prho(n: int, s: float, lower_tail: bool) -> float:
    """AS 89: P[S >= s] (or P[S < s] if lower_tail) under H0, where
    S = (n^3 - n)(1 - rho)/6. Exact for n <= 9, Edgeworth series above."""
    tail_default = 0.0 if lower_tail else 1.0
    if s <= 0:
        return tail_default
    s_max = n * (n * n - 1) / 3
    if s > s_max:
        return 1 - tail_default
    if n <= _PRHO_N_EXACT:
        null = _spearman_s_null(n)
        n_ge = len(null) - int(np.searchsorted(null, s, side="left"))
        return (len(null) - n_ge if lower_tail else n_ge) / len(null)

    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12 = _AS89_C
    b = 1 / n
    x = (6 * (s - 1) * b / (n * n - 1) - 1) * np.sqrt(n - 1)
    y = x * x
    u = (
        x
        * b
        * (
            c1
            + b * (c2 + c3 * b)
            + y
            * (
                -c4
                + b * (c5 + c6 * b)
                - y * b * (c7 + c8 * b - y * (c9 - c10 * b + y * b * (c11 - c12 * y)))
            )
        )
    )
    correction = u / np.exp(y / 2)
    p = (-correction if lower_tail else correction) + float(
        stats.norm.cdf(x) if lower_tail else stats.norm.sf(x)
    )
    return float(min(max(p, 0.0), 1.0))


def _spearman_exact_p(rho: float, n: int) -> float:
    """Two-sided p-value exactly as R's cor.test(method='spearman') computes
    it without ties (n <= 1290)."""
    q = (n**3 - n) * (1 - rho) / 6
    if q > (n**3 - n) / 6:
        p = _prho(n, round(q), lower_tail=False)
    else:
        p = _prho(n, round(q) + 2, lower_tail=True)
    return min(2 * p, 1.0)


@register(
    name="spearman",
    kind="test",
    stage="hypothesis",
    tags={"correlation", "nonparametric"},
    assumptions={"hard": ["ordinal_or_higher"], "soft": ["monotonicity"]},
    estimand="monotonic association between two variables",
    code_template=(
        "edacore.stattests.correlation.spearman("
        "{df}, x={x!r}, y={y!r}, ci={ci}, nan_policy={nan_policy!r})"
    ),
)
def spearman(
    df: pd.DataFrame, x: str, y: str, ci: float = 0.95, nan_policy: NanPolicy = "omit"
) -> TestResult:
    warnings: list[str] = []
    clean = _clean_pair(df, x, y, nan_policy, warnings)
    xv, yv = clean[x].to_numpy(), clean[y].to_numpy()
    n = len(xv)
    has_ties = len(np.unique(xv)) < n or len(np.unique(yv)) < n

    result = stats.spearmanr(xv, yv)
    rho = float(result.statistic)
    p_value = float(result.pvalue)
    if has_ties:
        warnings.append("ties present: t-approximation p-value (as R's cor.test)")
    elif n <= _PRHO_N_EXACT:
        p_value = _spearman_exact_p(rho, n)
        warnings.append(f"exact p-value (full enumeration of the null distribution, n={n})")
    elif n <= _PRHO_N_MAX:
        p_value = _spearman_exact_p(rho, n)
        warnings.append(f"AS 89 Edgeworth-series p-value (n={n}, no ties; as R's cor.test)")
    else:
        warnings.append(f"n={n} > {_PRHO_N_MAX}: t-approximation p-value (as R's cor.test)")

    rho_ci = _spearman_rho_ci(rho, n, ci)

    return TestResult(
        fact_id=f"spearman.{x}.{y}",
        function="spearman",
        estimand=f"monotonic association between '{x}' and '{y}'",
        # R's S = (n^3 - n)(1 - rho)/6, the statistic its p-value is
        # computed from; rho itself is the estimate.
        statistic=float((n**3 - n) * (1 - rho) / 6),
        statistic_name="S",
        p_value=p_value,
        estimate=rho,
        ci=rho_ci,
        ci_level=ci,
        effect_size=rho,
        effect_size_name="spearman_rho",
        effect_size_ci=rho_ci,
        effect_magnitude=magnitude_label(rho, "spearman_rho"),
        n={"total": n},
        warnings=warnings,
    )


# R's cor.test(method='kendall') is exact for n below this (no ties).
_KENDALL_N_EXACT = 50


def _kendall_z_statistic(x: np.ndarray, y: np.ndarray, tau_b: float) -> float:
    """R's cor.test z-statistic for Kendall's tau-b, ported exactly from
    its source (the tie-corrected variance of the S statistic) -- scipy's
    kendalltau exposes only tau itself, not this statistic."""
    n = len(x)
    _, x_counts = np.unique(x, return_counts=True)
    _, y_counts = np.unique(y, return_counts=True)
    xties = x_counts[x_counts > 1]
    yties = y_counts[y_counts > 1]

    t0 = n * (n - 1) / 2
    t1 = float(np.sum(xties * (xties - 1))) / 2
    t2 = float(np.sum(yties * (yties - 1))) / 2
    s = tau_b * np.sqrt((t0 - t1) * (t0 - t2))

    v0 = n * (n - 1) * (2 * n + 5)
    vt = float(np.sum(xties * (xties - 1) * (2 * xties + 5)))
    vu = float(np.sum(yties * (yties - 1) * (2 * yties + 5)))
    v1 = float(np.sum(xties * (xties - 1))) * float(np.sum(yties * (yties - 1)))
    v2 = float(np.sum(xties * (xties - 1) * (xties - 2))) * float(
        np.sum(yties * (yties - 1) * (yties - 2))
    )
    var_s = (v0 - vt - vu) / 18 + v1 / (2 * n * (n - 1)) + v2 / (9 * n * (n - 1) * (n - 2))
    return float(s / np.sqrt(var_s))


@register(
    name="kendall_tau",
    kind="test",
    stage="hypothesis",
    tags={"correlation", "nonparametric"},
    assumptions={"hard": ["ordinal_or_higher"], "soft": []},
    estimand="ordinal association between two variables",
    code_template=(
        "edacore.stattests.correlation.kendall_tau("
        "{df}, x={x!r}, y={y!r}, ci={ci}, nan_policy={nan_policy!r})"
    ),
)
def kendall_tau(
    df: pd.DataFrame, x: str, y: str, ci: float = 0.95, nan_policy: NanPolicy = "omit"
) -> TestResult:
    """tau-b (scipy's default `variant='b'`, matching R's default). As in
    R's cor.test: exact null distribution of T (the concordant-pair count)
    without ties and n<50, reporting T; otherwise the normal approximation
    with R's tie-corrected variance, reporting z."""
    warnings: list[str] = []
    clean = _clean_pair(df, x, y, nan_policy, warnings)
    xv, yv = clean[x].to_numpy(), clean[y].to_numpy()
    n = len(xv)
    has_ties = len(np.unique(xv)) < n or len(np.unique(yv)) < n
    exact = not has_ties and n < _KENDALL_N_EXACT

    result = stats.kendalltau(xv, yv, variant="b", method="exact" if exact else "asymptotic")
    tau_b = float(result.statistic)
    p_value = float(result.pvalue)
    if exact:
        statistic, statistic_name = float(round((tau_b + 1) * n * (n - 1) / 4)), "T"
        warnings.append(f"exact p-value (null distribution of T, n={n}, no ties)")
    else:
        statistic, statistic_name = _kendall_z_statistic(xv, yv, tau_b), "z"
        p_value = float(2 * stats.norm.sf(abs(statistic)))
        reason = "ties present" if has_ties else f"n={n} >= {_KENDALL_N_EXACT}"
        warnings.append(f"{reason}: normal-approximation p-value (as R's cor.test)")

    tau_ci = _kendall_tau_b_ci(xv, yv, tau_b, ci, warnings)

    return TestResult(
        fact_id=f"kendall_tau.{x}.{y}",
        function="kendall_tau",
        estimand=f"ordinal association between '{x}' and '{y}'",
        statistic=statistic,
        statistic_name=statistic_name,
        p_value=p_value,
        estimate=tau_b,
        ci=tau_ci,
        ci_level=ci,
        effect_size=tau_b,
        effect_size_name="kendall_tau_b",
        effect_size_ci=tau_ci,
        effect_magnitude=magnitude_label(tau_b, "kendall_tau_b"),
        n={"total": n},
        warnings=warnings,
    )


@register(
    name="point_biserial",
    kind="test",
    stage="hypothesis",
    tags={"correlation"},
    assumptions={"hard": ["binary", "numeric"], "soft": ["normality_within_groups"]},
    estimand="association between a binary and a continuous variable",
    code_template=(
        "edacore.stattests.correlation.point_biserial("
        "{df}, binary={binary!r}, x={x!r}, ci={ci}, nan_policy={nan_policy!r})"
    ),
)
def point_biserial(
    df: pd.DataFrame, binary: str, x: str, ci: float = 0.95, nan_policy: NanPolicy = "omit"
) -> TestResult:
    """Algebraically identical to Pearson's r computed on the binary
    column coded as 0/1 -- same R function (cor.test), no separate
    fixture needed."""
    warnings: list[str] = []
    clean = _clean_pair(df, binary, x, nan_policy, warnings)
    levels = sorted(clean[binary].unique())
    if len(levels) != 2:
        raise ValueError(f"'{binary}' must have exactly 2 distinct values, found {len(levels)}")
    coded = clean[binary].map({levels[0]: 0.0, levels[1]: 1.0})

    result = stats.pearsonr(coded.to_numpy(), clean[x].to_numpy())
    ci_bounds = result.confidence_interval(ci)
    n = len(clean)
    r_pb = float(result.statistic)
    dof = n - 2
    t_stat = float(np.sqrt(dof) * r_pb / np.sqrt(1 - r_pb**2))

    return TestResult(
        fact_id=f"point_biserial.{binary}.{x}",
        function="point_biserial",
        estimand=f"association between '{binary}' (0={levels[0]!r}, 1={levels[1]!r}) and '{x}'",
        statistic=t_stat,
        statistic_name="t",
        df=float(dof),
        p_value=float(result.pvalue),
        estimate=r_pb,
        ci=(float(ci_bounds.low), float(ci_bounds.high)),
        ci_level=ci,
        effect_size=r_pb,
        effect_size_name="point_biserial_r",
        effect_size_ci=(float(ci_bounds.low), float(ci_bounds.high)),
        effect_magnitude=magnitude_label(r_pb, "point_biserial_r"),
        n={"total": n},
        warnings=warnings,
    )


@register(
    name="partial_correlation",
    kind="test",
    stage="hypothesis",
    tags={"correlation"},
    assumptions={"hard": ["numeric"], "soft": ["linearity"]},
    estimand="linear association between two variables, controlling for others",
    code_template=(
        "edacore.stattests.correlation.partial_correlation("
        "{df}, x={x!r}, y={y!r}, covars={covars!r}, ci={ci}, nan_policy={nan_policy!r})"
    ),
)
def partial_correlation(
    df: pd.DataFrame,
    x: str,
    y: str,
    covars: list[str],
    ci: float = 0.95,
    nan_policy: NanPolicy = "omit",
) -> TestResult:
    """Ported from ppcor::pcor.test's general formula: the partial
    correlation of x and y given any number of control variables is
    -P[x,y] / sqrt(P[x,x]*P[y,y]), where P is the inverse of the full
    correlation matrix (the precision matrix) -- not iterative
    residualization. Verified to ~1e-9 against R (method='pearson', its
    default)."""
    warnings: list[str] = []
    cols = [x, y, *covars]
    sub = df[cols]
    n_before = len(sub)
    clean = sub.dropna()
    n_dropped = n_before - len(clean)
    if n_dropped > 0:
        if nan_policy == "raise":
            raise ValueError(f"{n_dropped} row(s) missing data, nan_policy=raise")
        warnings.append(f"{n_dropped} row(s) dropped: missing data (nan_policy='omit')")

    corr = clean.to_numpy().T
    r_matrix = np.corrcoef(corr)
    precision = np.linalg.inv(r_matrix)
    estimate = float(-precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1]))

    n = len(clean)
    gp = len(covars)
    dof = n - gp - 2
    t_stat = estimate * np.sqrt(dof / (1 - estimate**2))
    p_value = float(2 * stats.t.sf(abs(t_stat), dof))
    # Fisher z with SE = 1/sqrt(n - k - 3), k = number of controlled
    # variables. No installed R package reports a CI for a partial
    # correlation, so unlike every other interval in this module the
    # fixture evaluates this same published formula in R rather than an
    # independent implementation of it (ARCHITECTURE.md Section 18, M3.4).
    r_ci = _fisher_z_ci(estimate, 1 / np.sqrt(n - gp - 3), ci)

    return TestResult(
        fact_id=f"partial_correlation.{x}.{y}",
        function="partial_correlation",
        estimand=f"correlation between '{x}' and '{y}', controlling for {covars}",
        statistic=float(t_stat),
        statistic_name="t",
        df=float(dof),
        p_value=p_value,
        estimate=estimate,
        ci=r_ci,
        ci_level=ci,
        effect_size=estimate,
        effect_size_name="partial_r",
        effect_size_ci=r_ci,
        effect_magnitude=magnitude_label(estimate, "partial_r"),
        n={"total": n},
        warnings=warnings,
    )


def _dcor_statistic(x: np.ndarray, y: np.ndarray) -> float:
    """Szekely & Rizzo's distance correlation, matching energy::dcor's
    formula exactly (double-centered distance matrices). Verified to
    ~1e-9 against R."""
    a = np.abs(x[:, None] - x[None, :])
    b = np.abs(y[:, None] - y[None, :])
    big_a = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
    big_b = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
    dcov2 = float(np.mean(big_a * big_b))
    dvar_x = float(np.mean(big_a * big_a))
    dvar_y = float(np.mean(big_b * big_b))
    denom = np.sqrt(dvar_x * dvar_y)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(max(dcov2, 0.0)) / np.sqrt(denom))


@register(
    name="distance_correlation",
    kind="test",
    stage="hypothesis",
    tags={"correlation", "resampling"},
    assumptions={"hard": ["numeric"], "soft": []},
    estimand="general (linear or nonlinear) association between two continuous variables",
    code_template=(
        "edacore.stattests.correlation.distance_correlation("
        "{df}, x={x!r}, y={y!r}, n_resamples={n_resamples}, random_state={random_state}, "
        "nan_policy={nan_policy!r})"
    ),
)
def distance_correlation(
    df: pd.DataFrame,
    x: str,
    y: str,
    n_resamples: int = 2000,
    random_state: int = 0,
    nan_policy: NanPolicy = "omit",
) -> TestResult:
    """Statistic matches energy::dcor exactly. Its p-value is from this
    module's own permutation test -- NOT comparable to
    energy::dcor.test's own permutation p-value bit-for-bit (different
    RNG); see test_correlation.py for the simulation-based check this
    warrants instead of an R fixture match."""
    warnings: list[str] = []
    clean = _clean_pair(df, x, y, nan_policy, warnings)
    xv, yv = clean[x].to_numpy(dtype=float), clean[y].to_numpy(dtype=float)

    observed = _dcor_statistic(xv, yv)
    rng = np.random.default_rng(random_state)
    n = len(xv)
    perm_stats = np.empty(n_resamples)
    for i in range(n_resamples):
        perm_stats[i] = _dcor_statistic(xv, rng.permutation(yv))
    p_value = float((np.sum(perm_stats >= observed - 1e-12) + 1) / (n_resamples + 1))

    return TestResult(
        fact_id=f"distance_correlation.{x}.{y}",
        function="distance_correlation",
        estimand=f"general association between '{x}' and '{y}'",
        statistic=observed,
        statistic_name="dcor",
        p_value=p_value,
        estimate=observed,
        n={"total": n},
        warnings=warnings,
    )


def _ksg_mutual_information(
    x: np.ndarray, y: np.ndarray, k: int = 5, random_state: int = 0
) -> float:
    """Kraskov-Stogbauer-Grassberger (2004) k-NN mutual information
    estimator (algorithm 1, max-norm), with sklearn's
    `mutual_info_regression` preprocessing: each variable scaled to unit
    variance (the max-norm neighbourhoods are not scale-invariant, MI is),
    plus 1e-10-relative jitter so tied values don't give zero radii."""
    n = len(x)
    rng = np.random.default_rng(random_state)
    scaled = []
    for v in (x, y):
        v = v / v.std()
        scaled.append(v + 1e-10 * max(1.0, float(np.mean(np.abs(v)))) * rng.standard_normal(n))
    xs, ys = scaled

    xy = np.column_stack([xs, ys])
    dists, _ = cKDTree(xy).query(xy, k=k + 1, p=np.inf)
    # Strictly inside the k-th neighbour's distance (the point itself included).
    radius = np.nextafter(dists[:, -1], 0)
    nx = cKDTree(xs[:, None]).query_ball_point(xs[:, None], radius, p=np.inf, return_length=True)
    ny = cKDTree(ys[:, None]).query_ball_point(ys[:, None], radius, p=np.inf, return_length=True)
    mi = digamma(n) + digamma(k) - float(np.mean(digamma(nx) + digamma(ny)))
    return max(float(mi), 0.0)


@register(
    name="mutual_information",
    kind="test",
    stage="hypothesis",
    tags={"correlation"},
    assumptions={"hard": ["numeric"], "soft": []},
    estimand="general statistical dependence between two continuous variables",
    code_template=(
        "edacore.stattests.correlation.mutual_information("
        "{df}, x={x!r}, y={y!r}, k={k}, random_state={random_state}, nan_policy={nan_policy!r})"
    ),
)
def mutual_information(
    df: pd.DataFrame,
    x: str,
    y: str,
    k: int = 5,
    random_state: int = 0,
    nan_policy: NanPolicy = "omit",
) -> TestResult:
    """No R reference exists for this estimator; verified against
    sklearn's `mutual_info_regression` (same algorithm and preprocessing)
    and against the analytic bivariate-normal formula -0.5*ln(1-rho^2)
    (test_correlation.py). `random_state` seeds the tie-breaking jitter.
    No p-value: mutual information has no standard null-hypothesis test
    the way a correlation coefficient does."""
    warnings: list[str] = []
    clean = _clean_pair(df, x, y, nan_policy, warnings)
    n = len(clean)
    if k >= n:
        raise ValueError(f"k ({k}) must be less than n ({n})")
    xv, yv = clean[x].to_numpy(dtype=float), clean[y].to_numpy(dtype=float)
    for name, v in ((x, xv), (y, yv)):
        if np.ptp(v) == 0:
            raise ValueError(f"'{name}' is constant; mutual information is undefined")
    if len(np.unique(xv)) < n or len(np.unique(yv)) < n:
        warnings.append(
            "tied values broken by 1e-10 jitter: the KSG estimator assumes continuous "
            "variables, so treat the estimate as approximate for discrete/ordinal data"
        )

    mi = _ksg_mutual_information(xv, yv, k, random_state)

    return TestResult(
        fact_id=f"mutual_information.{x}.{y}",
        function="mutual_information",
        estimand=f"statistical dependence between '{x}' and '{y}'",
        statistic=mi,
        statistic_name="mutual_information_nats",
        estimate=mi,
        n={"total": n},
        warnings=warnings,
    )


@register(
    name="correlation_matrix",
    kind="test",
    stage="hypothesis",
    tags={"correlation"},
    assumptions={"hard": ["numeric_or_ordinal"], "soft": []},
    estimand="pairwise association among several variables",
    code_template=(
        "edacore.stattests.correlation.correlation_matrix("
        "{df}, cols={cols!r}, method={method!r}, adjust={adjust!r}, nan_policy={nan_policy!r})"
    ),
)
def correlation_matrix(
    df: pd.DataFrame,
    cols: list[str],
    method: str = "pearson",
    adjust: str = "holm",
    nan_policy: NanPolicy = "omit",
) -> list[TestResult]:
    """Pairwise pearson/spearman/kendall_tau, each computed by this
    module's own already-R-verified function -- not a separately-derived
    formula -- with p-values Holm-adjusted (or `adjust`) across all pairs
    as one family."""
    if method not in ("pearson", "spearman", "kendall"):
        raise ValueError(f"method must be 'pearson', 'spearman', or 'kendall', got {method!r}")

    def _call(a: str, b: str) -> TestResult:
        if method == "pearson":
            return pearson(df, a, b, nan_policy=nan_policy)
        if method == "spearman":
            return spearman(df, a, b, nan_policy=nan_policy)
        return kendall_tau(df, a, b, nan_policy=nan_policy)

    results = [_call(a, b) for a, b in combinations(cols, 2)]

    p_values = [r.p_value for r in results]
    assert all(p is not None for p in p_values)
    _, p_adj, _, _ = multipletests(p_values, method=adjust)

    return [
        r.model_copy(update={"p_adjusted": float(pa)}) for r, pa in zip(results, p_adj, strict=True)
    ]
