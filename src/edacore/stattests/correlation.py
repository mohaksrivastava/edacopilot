"""Correlation hypothesis tests (ARCHITECTURE.md, Section 6.7).

Verified against R (tests/fixtures/r_reference/) except mutual_information
(no R reference; tested against the analytic bivariate-normal formula).

- pearson: CI via Fisher z (scipy's own `pearsonr(...).confidence_interval()`
  already uses this, confirmed matching R's `cor.test` exactly).
- spearman/kendall_tau: R's `cor.test` computes an EXACT p-value by
  default for small n without ties (Spearman: AS 89 algorithm, n<1290;
  Kendall: an exact recursive distribution, n<50); scipy's `spearmanr`
  has no exact-mode option at all, and `kendalltau`'s `method='auto'`
  only covers the Kendall case. Confirmed the gap is real (scipy's
  asymptotic Spearman p-value differs from R's exact one by ~2x on this
  module's own small-n fixture). Ported exact mode as full permutation
  enumeration (same MAX_EXACT_PERMUTATIONS threshold used throughout
  stattests/), verified to match R's exact p-value bit-for-bit on n=8 --
  but note this covers a narrower n range than R's specialized exact
  algorithms (n! grows far faster than R's polynomial-time methods), so
  agreement with R is not guaranteed for every case R itself treats as
  exact (e.g. n=15, no ties: R is still exact, this module already isn't).
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
- mutual_information: no R reference at all. Estimated via a KSG
  (Kraskov-Stogbauer-Grassberger 2004) k-NN estimator built directly on
  `scipy.spatial.cKDTree` -- deliberately NOT
  `sklearn.feature_selection.mutual_info_regression`, which hit a blocked
  compiled dependency in this environment (an Application Control policy
  blocking `_expected_mutual_info_fast`, the same class of local-only
  issue documented for mypy's DLL in ARCHITECTURE.md Section 18) and
  could not be verified to work at all, let alone match anything.
"""

from __future__ import annotations

from itertools import combinations, permutations
from math import factorial
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree
from scipy.special import digamma
from statsmodels.stats.multitest import multipletests

from edacore.contracts import TestResult
from edacore.registry import register
from edacore.stattests._shared import MAX_EXACT_PERMUTATIONS, NanPolicy


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
        n={"total": n},
        warnings=warnings,
    )


def _exact_permutation_p(a: np.ndarray, b: np.ndarray, corr_fn: Any) -> tuple[float, bool]:
    """Two-sided exact p-value for a rank correlation via full permutation
    enumeration, bounded by MAX_EXACT_PERMUTATIONS. Returns (p_value,
    used_exact)."""
    n = len(a)
    if factorial(n) > MAX_EXACT_PERMUTATIONS:
        return float("nan"), False
    observed = abs(corr_fn(a, b))
    n_extreme = 0
    n_total = 0
    for perm in permutations(range(n)):
        n_total += 1
        if abs(corr_fn(a, np.asarray(b)[list(perm)])) >= observed - 1e-10:
            n_extreme += 1
    return n_extreme / n_total, True


@register(
    name="spearman",
    kind="test",
    stage="hypothesis",
    tags={"correlation", "nonparametric"},
    assumptions={"hard": ["ordinal_or_higher"], "soft": ["monotonicity"]},
    estimand="monotonic association between two variables",
    code_template=(
        "edacore.stattests.correlation.spearman({df}, x={x!r}, y={y!r}, nan_policy={nan_policy!r})"
    ),
)
def spearman(df: pd.DataFrame, x: str, y: str, nan_policy: NanPolicy = "omit") -> TestResult:
    warnings: list[str] = []
    clean = _clean_pair(df, x, y, nan_policy, warnings)
    xv, yv = clean[x].to_numpy(), clean[y].to_numpy()
    n = len(xv)
    has_ties = len(np.unique(xv)) < n or len(np.unique(yv)) < n

    result = stats.spearmanr(xv, yv)
    p_value = float(result.pvalue)
    if not has_ties:

        def _rho(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.corrcoef(stats.rankdata(a), stats.rankdata(b))[0, 1])

        exact_p, used_exact = _exact_permutation_p(xv, yv, _rho)
        if used_exact:
            p_value = exact_p
            warnings.append(f"exact permutation p-value (n={n}, no ties)")
        else:
            warnings.append(
                f"n={n} too large for this module's exact permutation p-value "
                f"(n! > {MAX_EXACT_PERMUTATIONS}); using the asymptotic approximation, "
                "which may not match R's own exact algorithm (R stays exact up to n<1290)"
            )

    return TestResult(
        fact_id=f"spearman.{x}.{y}",
        function="spearman",
        estimand=f"monotonic association between '{x}' and '{y}'",
        statistic=float(result.statistic),
        statistic_name="rho",
        p_value=p_value,
        estimate=float(result.statistic),
        n={"total": n},
        warnings=warnings,
    )


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
        "{df}, x={x!r}, y={y!r}, nan_policy={nan_policy!r})"
    ),
)
def kendall_tau(df: pd.DataFrame, x: str, y: str, nan_policy: NanPolicy = "omit") -> TestResult:
    """tau-b (scipy's default `variant='b'`, matching R's default);
    scipy's `method='auto'` already falls back to the normal
    approximation exactly when R would (ties present), confirmed against
    an R fixture with ties -- verified bit-for-bit, no porting needed
    there. Exact mode (no ties, small n) uses the same permutation
    enumeration as spearman, for the same reason (no scipy equivalent)."""
    warnings: list[str] = []
    clean = _clean_pair(df, x, y, nan_policy, warnings)
    xv, yv = clean[x].to_numpy(), clean[y].to_numpy()
    n = len(xv)
    has_ties = len(np.unique(xv)) < n or len(np.unique(yv)) < n

    result = stats.kendalltau(xv, yv, variant="b", method="auto")
    p_value = float(result.pvalue)
    if not has_ties:

        def _tau(a: np.ndarray, b: np.ndarray) -> float:
            return float(stats.kendalltau(a, b, variant="b", method="asymptotic").statistic)

        exact_p, used_exact = _exact_permutation_p(xv, yv, _tau)
        if used_exact:
            p_value = exact_p
            warnings.append(f"exact permutation p-value (n={n}, no ties)")
        else:
            warnings.append(
                f"n={n} too large for this module's exact permutation p-value "
                f"(n! > {MAX_EXACT_PERMUTATIONS}); using the asymptotic approximation, "
                "which may not match R's own exact algorithm (R stays exact up to n<50)"
            )

    tau_b = float(result.statistic)
    z_stat = _kendall_z_statistic(xv, yv, tau_b)

    return TestResult(
        fact_id=f"kendall_tau.{x}.{y}",
        function="kendall_tau",
        estimand=f"ordinal association between '{x}' and '{y}'",
        statistic=z_stat,
        statistic_name="z",
        p_value=p_value,
        estimate=tau_b,
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
        "{df}, x={x!r}, y={y!r}, covars={covars!r}, nan_policy={nan_policy!r})"
    ),
)
def partial_correlation(
    df: pd.DataFrame, x: str, y: str, covars: list[str], nan_policy: NanPolicy = "omit"
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

    return TestResult(
        fact_id=f"partial_correlation.{x}.{y}",
        function="partial_correlation",
        estimand=f"correlation between '{x}' and '{y}', controlling for {covars}",
        statistic=float(t_stat),
        statistic_name="t",
        df=float(dof),
        p_value=p_value,
        estimate=estimate,
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


def _ksg_mutual_information(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """Kraskov-Stogbauer-Grassberger (2004) k-NN mutual information
    estimator, built directly on scipy.spatial.cKDTree (see module
    docstring for why not sklearn's mutual_info_regression)."""
    n = len(x)
    xy = np.column_stack([x, y])
    tree_xy = cKDTree(xy)
    dists, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
    eps = dists[:, -1]

    tree_x = cKDTree(x.reshape(-1, 1))
    tree_y = cKDTree(y.reshape(-1, 1))
    nx = np.array(
        [len(tree_x.query_ball_point([x[i]], eps[i] - 1e-10, p=np.inf)) - 1 for i in range(n)]
    )
    ny = np.array(
        [len(tree_y.query_ball_point([y[i]], eps[i] - 1e-10, p=np.inf)) - 1 for i in range(n)]
    )
    mi = digamma(k) - float(np.mean(digamma(nx + 1) + digamma(ny + 1))) + digamma(n)
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
        "{df}, x={x!r}, y={y!r}, k={k}, nan_policy={nan_policy!r})"
    ),
)
def mutual_information(
    df: pd.DataFrame, x: str, y: str, k: int = 5, nan_policy: NanPolicy = "omit"
) -> TestResult:
    """No R reference exists for this estimator; verified instead against
    the analytic bivariate-normal formula -0.5*ln(1-rho^2) by simulation
    (test_correlation.py), with a tolerance justified by the KSG
    estimator's known finite-sample bias/variance at the tested n and k.
    No p-value: mutual information has no standard null-hypothesis test
    the way a correlation coefficient does."""
    warnings: list[str] = []
    clean = _clean_pair(df, x, y, nan_policy, warnings)
    n = len(clean)
    if k >= n:
        raise ValueError(f"k ({k}) must be less than n ({n})")

    mi = _ksg_mutual_information(clean[x].to_numpy(dtype=float), clean[y].to_numpy(dtype=float), k)

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
