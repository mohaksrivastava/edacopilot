"""Effect sizes (ARCHITECTURE.md, Section 6.8).

Every point estimate here matches R's `effectsize` package
(tests/fixtures/r_reference/) to at least 1e-6. As of M2.1, every CI is
ported from effectsize's actual algorithm (read from its R source, not
guessed) rather than approximated — see each function's docstring for its
method and ARCHITECTURE.md Section 18's M2.1 changelog entry for the full
account of what was verified and what changed from M2's initial (partly
mistaken) assumptions about which functions bootstrap:

- cohens_d, hedges_g, glass_delta, d_z: noncentral-t inversion.
- odds_ratio, risk_ratio, risk_difference: log-scale / linear Wald.
- cohens_h: arcsine.
- eta_squared, partial_eta_squared, omega_squared: noncentral-F inversion,
  one-sided (ci_high fixed at 1).
- cramers_v, phi, cohens_w: noncentral chi-square inversion, one-sided.
- rank_biserial, cliffs_delta: exact closed form (Fisher-z transform with
  an analytic SE) — effectsize does NOT bootstrap these, despite M2
  assuming it did.
- kendalls_w, epsilon_squared: BCa bootstrap. effectsize itself uses plain
  percentile bootstrap for both (not BCa) — this module uses BCa instead
  for a tighter interval, validated by simulated coverage rather than
  fixture matching (test_effect_size_ci_coverage.py), since a bootstrap's
  specific resampling draws can't be matched to R's bit-for-bit anyway.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from edacore.registry import register

# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _hedges_j(df: float) -> float:
    """Exact small-sample bias correction factor (Hedges & Olkin 1985)."""
    return float(
        np.exp(special.gammaln(df / 2) - np.log(np.sqrt(df / 2)) - special.gammaln((df - 1) / 2))
    )


def _invert_nct_cdf(t_obs: float, df: float, target_p: float) -> float:
    """Find nc such that scipy.stats.nct.cdf(t_obs, df, nc) == target_p.

    nct.cdf is monotonically decreasing in nc. scipy's implementation has
    occasional isolated NaN glitches at specific nc values, so this brackets
    via a fine grid (skipping NaNs) rather than a doubling search that could
    land exactly on one.
    """
    span = max(50.0, 10 * abs(t_obs) + 50.0)
    grid = np.linspace(-span, span, 4001)
    with np.errstate(all="ignore"):
        vals = stats.nct.cdf(t_obs, df, grid) - target_p
    valid = ~np.isnan(vals)
    grid, vals = grid[valid], vals[valid]
    sign_changes = np.where(np.diff(np.sign(vals)) != 0)[0]
    if len(sign_changes) == 0:
        raise ValueError(f"no sign change found for t_obs={t_obs}, df={df}, target_p={target_p}")
    i = sign_changes[0]
    return float(
        optimize.brentq(
            lambda nc: float(stats.nct.cdf(t_obs, df, nc) - target_p), grid[i], grid[i + 1]
        )
    )


def _ncp_ci(t_obs: float, df: float, hn: float, ci: float) -> tuple[float, float]:
    """Noncentral-t confidence interval for a standardized mean difference,
    matching effectsize's `.get_ncp_t` + `sqrt(hn)` scaling exactly.

    Returns (nan, nan) on a degenerate sample -- a zero within-group SD
    makes t_obs infinite and the standardized effect size itself
    undefined, so there is no interval to invert. The point estimate is
    still reported (as +/-inf), which is what R does too; raising here
    would turn "this sample has no variance" into a crash partway through
    an otherwise valid test result.
    """
    if not np.isfinite(t_obs) or df < 1:
        return float("nan"), float("nan")
    alpha = 1 - ci
    ncp_low = _invert_nct_cdf(t_obs, df, 1 - alpha / 2)
    ncp_high = _invert_nct_cdf(t_obs, df, alpha / 2)
    return ncp_low * np.sqrt(hn), ncp_high * np.sqrt(hn)


def _invert_nc_cdf_nonneg(cdf_fn: Any, target_p: float, hi_guess: float = 500.0) -> float:
    """Find nc >= 0 such that cdf_fn(nc) == target_p, for a noncentral F or
    chi-square CDF (monotonically decreasing in nc, domain [0, inf)).

    Mirrors effectsize's `.get_ncp_F` / `.get_ncp_chi` edge case: if even
    nc=0 (the largest possible CDF value) is at or below target_p, no
    nonnegative nc can reach it, so the bound is 0 (matches R setting that
    ncp to 0 when the observed statistic is below the target quantile).
    """
    if cdf_fn(0.0) <= target_p:
        return 0.0
    grid = np.linspace(0, hi_guess, 4001)
    with np.errstate(all="ignore"):
        vals = np.array([cdf_fn(x) for x in grid]) - target_p
    valid = ~np.isnan(vals)
    grid, vals = grid[valid], vals[valid]
    sign_changes = np.where(np.diff(np.sign(vals)) != 0)[0]
    if len(sign_changes) == 0:
        raise ValueError(f"no sign change found for target_p={target_p} in [0, {hi_guess}]")
    i = sign_changes[0]
    return float(optimize.brentq(lambda nc: cdf_fn(nc) - target_p, grid[i], grid[i + 1]))


def _pve_ci_low(estimate: float, df1: float, df2: float, ci: float) -> float:
    """One-sided lower confidence bound for a proportion-of-variance
    effect size (eta-squared, omega-squared), via noncentral-F inversion.

    Matches effectsize's `.get_ncp_F` + eta2 back-conversion exactly
    (verified to ~1e-9 against R): reconstruct a pseudo-F from the point
    estimate, find the ncp for which that pseudo-F sits at the `ci`
    quantile of noncentral F(df1, df2, ncp), then convert that ncp back
    to the eta2 scale via ncp / (ncp + df2).
    """
    if estimate <= 0:
        return 0.0
    pseudo_f = (estimate / df1) / ((1 - estimate) / df2)
    ncp_low = _invert_nc_cdf_nonneg(lambda nc: float(stats.ncf.cdf(pseudo_f, df1, df2, nc)), ci)
    return ncp_low / (ncp_low + df2)


def _ncx2_ci_low(chi2_stat: float, df: float, ci: float) -> float:
    """ncp lower bound via noncentral chi-square inversion, matching
    effectsize's `.get_ncp_chi` (verified to ~1e-8 against R)."""
    return _invert_nc_cdf_nonneg(lambda nc: float(stats.ncx2.cdf(chi2_stat, df, nc)), ci)


def _log_scale_ci(estimate: float, se_log: float, ci: float) -> tuple[float, float]:
    """Wald CI on the log scale, for ratio measures (odds ratio, risk ratio)."""
    z = stats.norm.ppf(1 - (1 - ci) / 2)
    log_est = np.log(estimate)
    return float(np.exp(log_est - z * se_log)), float(np.exp(log_est + z * se_log))


def _fisher_z_ci(estimate: float, se_z: float, ci: float) -> tuple[float, float]:
    """Fisher-z CI for a correlation-like statistic bounded on [-1, 1],
    clamped to that range (the convention DescTools and effectsize share)."""
    z = stats.norm.ppf(1 - (1 - ci) / 2)
    z_est = np.arctanh(estimate)
    return (
        float(max(np.tanh(z_est - z * se_z), -1.0)),
        float(min(np.tanh(z_est + z * se_z), 1.0)),
    )


def _clopper_pearson_ci(successes: int, n: int, ci: float) -> tuple[float, float]:
    """Exact (Clopper-Pearson) interval for a binomial proportion -- the
    interval `stats::binom.test` reports."""
    interval = stats.binomtest(successes, n).proportion_ci(confidence_level=ci, method="exact")
    return float(interval.low), float(interval.high)


def _signed_rank_biserial(diffs: np.ndarray, ci: float) -> dict[str, float]:
    """Rank-biserial correlation for one-sample / paired data: the signed
    rank sums of the nonzero differences, (W+ - W-) / (nd(nd+1)/2).

    Matches effectsize's `.r_rbs(paired=TRUE)` point estimate and its
    `rank_biserial` CI: Fisher-z with the analytic standard error
    sqrt((2*nd^3 + 3*nd^2 + nd)/6) / (nd(nd+1)/2), clamped to [-1, 1]
    (verified to ~1e-16 against R). Not a bootstrap.
    """
    nonzero = diffs[diffs != 0]
    nd = len(nonzero)
    if nd == 0:
        return {"estimate": 0.0, "ci_low": -1.0, "ci_high": 1.0}
    ranks = stats.rankdata(np.abs(nonzero))
    max_w = nd * (nd + 1) / 2
    estimate = float((ranks[nonzero > 0].sum() - ranks[nonzero < 0].sum()) / max_w)
    se_z = float(np.sqrt((2 * nd**3 + 3 * nd**2 + nd) / 6) / max_w)
    ci_low, ci_high = _fisher_z_ci(estimate, se_z, ci)
    return {"estimate": estimate, "ci_low": ci_low, "ci_high": ci_high}


def partial_eta_squared_from_f(
    f_stat: float, df1: float, df2: float, ci: float = 0.95
) -> dict[str, float]:
    """Partial eta-squared for one term of any F-test: F*df1 / (F*df1 + df2),
    with the same one-sided noncentral-F CI `eta_squared` uses.

    This is exactly what `effectsize::eta_squared` computes for an
    `Anova.mlm` / `afex_aov` / ARTool anova table (confirmed by running both
    it and `effectsize::F_to_eta2` on the same fits and getting identical
    numbers), so it is the route for every multi-term or
    sphericity-corrected design, where reconstructing sums of squares would
    be ambiguous. Not registered: it converts already-computed test output
    rather than reading a DataFrame, like `magnitude_label`.
    """
    estimate = float((f_stat * df1) / (f_stat * df1 + df2))
    return {"estimate": estimate, "ci_low": _pve_ci_low(estimate, df1, df2, ci), "ci_high": 1.0}


def omega_squared_from_f(
    f_stat: float, df1: float, df2: float, ci: float = 0.95
) -> dict[str, float]:
    """Partial omega-squared from an F statistic:
    max(0, (F-1)*df1 / (F*df1 + df2 + 1)), with the one-sided noncentral-F
    CI. effectsize's `F_to_omega2`, used where no sums of squares exist
    (Welch ANOVA). Not registered, for the same reason as
    `partial_eta_squared_from_f`.
    """
    estimate = max(0.0, float(((f_stat - 1) * df1) / (f_stat * df1 + df2 + 1)))
    return {"estimate": estimate, "ci_low": _pve_ci_low(estimate, df1, df2, ci), "ci_high": 1.0}


def _two_by_two(
    df: pd.DataFrame, outcome: str, group: str, groups: tuple[str, str], event: Any
) -> tuple[int, int, int, int]:
    """Cell counts (a, b, c, d) for group1/group2 x event/no-event."""
    g1, g2 = groups
    a = int(((df[group] == g1) & (df[outcome] == event)).sum())
    b = int(((df[group] == g1) & (df[outcome] != event)).sum())
    c = int(((df[group] == g2) & (df[outcome] == event)).sum())
    d = int(((df[group] == g2) & (df[outcome] != event)).sum())
    return a, b, c, d


def bootstrap_effect_ci(
    fn: Any,
    df: pd.DataFrame,
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state: int = 0,
    **kwargs: Any,
) -> tuple[float, float]:
    """Percentile bootstrap CI for any `fn(df, **kwargs) -> float`.

    Resamples rows of `df` with replacement `n_boot` times. Not itself
    registered: it's a generic internal building block other functions
    call, not something a persona would select as its own step.
    """
    rng = np.random.default_rng(random_state)
    n = len(df)
    estimates = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        estimates[i] = fn(df.iloc[idx].reset_index(drop=True), **kwargs)
    alpha = 1 - ci
    # A degenerate resample (e.g. missing a whole group) can make `fn`
    # return NaN; drop those rather than letting them propagate.
    lo, hi = np.nanpercentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


_MAGNITUDE_THRESHOLDS: dict[str, tuple[float, float, float]] = {
    # measure -> (small, medium, large), compared against abs(value)
    "cohens_d": (0.2, 0.5, 0.8),
    "hedges_g": (0.2, 0.5, 0.8),
    "hedges_g_av": (0.2, 0.5, 0.8),
    "glass_delta": (0.2, 0.5, 0.8),
    "d_z": (0.2, 0.5, 0.8),
    "cohens_h": (0.2, 0.5, 0.8),
    "rank_biserial": (0.1, 0.3, 0.5),
    "cliffs_delta": (0.1, 0.3, 0.5),
    "eta_squared": (0.01, 0.06, 0.14),
    "partial_eta_squared": (0.01, 0.06, 0.14),
    "omega_squared": (0.01, 0.06, 0.14),
    "epsilon_squared": (0.01, 0.06, 0.14),
    "kendalls_w": (0.1, 0.3, 0.5),
    "cramers_v": (0.1, 0.3, 0.5),
    "phi": (0.1, 0.3, 0.5),
    "cohens_w": (0.1, 0.3, 0.5),
    # Correlation coefficients, on Cohen's r conventions. kendall_tau_b is
    # on the same scale by convention but is systematically smaller than
    # Pearson's r for the same association (tau ~ (2/pi)*arcsin(r) under
    # bivariate normality, e.g. r=0.5 -> tau=0.33), so these thresholds
    # UNDERSTATE a tau-b effect. Section 6.8's "thresholds are
    # field-dependent" caveat applies with extra force here.
    "pearson_r": (0.1, 0.3, 0.5),
    "point_biserial_r": (0.1, 0.3, 0.5),
    "spearman_rho": (0.1, 0.3, 0.5),
    "partial_r": (0.1, 0.3, 0.5),
    "kendall_tau_b": (0.1, 0.3, 0.5),
}

# Effect sizes this module reports that deliberately have NO magnitude label:
# unstandardized differences (a "medium" mean difference is meaningless
# without units), ratio measures with no Cohen convention, and the
# Brunner-Munzel relative effect. Call sites leave `effect_magnitude` unset
# for these rather than inventing thresholds.
UNLABELLED_MEASURES = frozenset(
    {
        "proportion",
        "mean_difference",
        "trimmed_mean_difference",
        "odds_ratio",
        "odds_ratio_conditional_mle",
        "relative_effect",
        "mutual_information_nats",
        "dcor",
    }
)


def _bca_from_bootstrap(
    estimates: np.ndarray, observed: float, jackknife_estimates: np.ndarray, ci: float
) -> tuple[float, float]:
    """BCa (bias-corrected and accelerated) percentile adjustment, given a
    bootstrap distribution, the whole-sample estimate, and leave-one-unit-out
    jackknife estimates for the acceleration term (Efron & Tibshirani 1993).
    """
    alpha = 1 - ci
    valid = ~np.isnan(estimates)
    estimates = estimates[valid]
    prop_less = (np.sum(estimates < observed) + 0.5 * np.sum(estimates == observed)) / len(
        estimates
    )
    prop_less = min(max(prop_less, 1e-6), 1 - 1e-6)
    z0 = stats.norm.ppf(prop_less)

    theta_dot = jackknife_estimates.mean()
    diffs = theta_dot - jackknife_estimates
    denom = 6 * (np.sum(diffs**2)) ** 1.5
    a = float(np.sum(diffs**3) / denom) if denom != 0 else 0.0

    def _adjusted_percentile(z: float) -> float:
        return float(stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))

    z_lo, z_hi = stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)
    p_lo, p_hi = _adjusted_percentile(z_lo), _adjusted_percentile(z_hi)
    lo, hi = np.nanpercentile(estimates, [100 * p_lo, 100 * p_hi])
    return float(lo), float(hi)


def _bca_matrix_row_bootstrap_ci(
    fn: Any, mat: np.ndarray, *, n_boot: int = 2000, ci: float = 0.95, random_state: int = 0
) -> tuple[float, float]:
    """BCa CI resampling whole rows of a numpy matrix (e.g. subjects in a
    subject x condition matrix, for kendalls_w) — required for
    repeated-measures data, where resampling rows independently would
    break the within-subject structure.

    Pure numpy, not pandas: rebuilding a DataFrame (concat + relabel) on
    every one of thousands of bootstrap iterations is 2-3 orders of
    magnitude slower than indexing an array, and this is on the hot path
    for every call.
    """
    rng = np.random.default_rng(random_state)
    n = mat.shape[0]
    estimates = np.array([fn(mat[rng.integers(0, n, size=n)]) for _ in range(n_boot)])
    observed = fn(mat)
    jackknife = np.array([fn(np.delete(mat, i, axis=0)) for i in range(n)])
    return _bca_from_bootstrap(estimates, observed, jackknife, ci)


def _bca_grouped_bootstrap_ci(
    fn: Any,
    arrays: list[np.ndarray],
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state: int = 0,
) -> tuple[float, float]:
    """BCa CI resampling within each group's array separately (preserving
    group sizes) — matches how R's own bootstrap for this family of
    statistics resamples (for epsilon_squared). Pure numpy; see
    `_bca_matrix_row_bootstrap_ci` for why."""
    rng = np.random.default_rng(random_state)
    estimates = np.array(
        [fn([rng.choice(a, size=len(a), replace=True) for a in arrays]) for _ in range(n_boot)]
    )
    observed = fn(arrays)
    jackknife = []
    for gi, a in enumerate(arrays):
        for i in range(len(a)):
            reduced = list(arrays)
            reduced[gi] = np.delete(a, i)
            jackknife.append(fn(reduced))
    return _bca_from_bootstrap(estimates, observed, np.array(jackknife), ci)


def magnitude_label(
    value: float, measure: str
) -> Literal["negligible", "small", "medium", "large"]:
    """Cohen's conventional magnitude labels. Not registered: a pure lookup,
    not a step a persona selects.

    These thresholds are Cohen's (1988) conventions, widely cited but not
    universal — what counts as a "large" effect is field-dependent (Section
    6.8). Treat the label as a starting point for discussion, not a verdict.
    """
    if measure not in _MAGNITUDE_THRESHOLDS:
        raise KeyError(f"no magnitude thresholds registered for measure '{measure}'")
    small, medium, large = _MAGNITUDE_THRESHOLDS[measure]
    magnitude = abs(value)
    if magnitude >= large:
        return "large"
    if magnitude >= medium:
        return "medium"
    if magnitude >= small:
        return "small"
    return "negligible"


# --------------------------------------------------------------------------
# Two independent groups
# --------------------------------------------------------------------------


@register(
    name="cohens_d",
    kind="effect",
    stage="hypothesis",
    code_template=(
        "edacore.effect_sizes.cohens_d({df}, outcome={outcome!r}, "
        "group={group!r}, groups={groups!r}, ci={ci})"
    ),
)
def cohens_d(
    df: pd.DataFrame, outcome: str, group: str, groups: tuple[str, str], ci: float = 0.95
) -> dict[str, float]:
    """Cohen's d for two independent groups (pooled SD denominator).

    Matches effectsize::cohens_d(pooled_sd=TRUE) exactly, including its
    noncentral-t confidence interval.
    """
    g1, g2 = groups
    x = df.loc[df[group] == g1, outcome].dropna().to_numpy(dtype=float)
    y = df.loc[df[group] == g2, outcome].dropna().to_numpy(dtype=float)
    n1, n2 = len(x), len(y)
    s1, s2 = x.std(ddof=1), y.std(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    d = (x.mean() - y.mean()) / pooled_sd

    se = pooled_sd * np.sqrt(1 / n1 + 1 / n2)
    t_obs = (x.mean() - y.mean()) / se
    ci_low, ci_high = _ncp_ci(t_obs, n1 + n2 - 2, 1 / n1 + 1 / n2, ci)
    return {"estimate": float(d), "ci_low": ci_low, "ci_high": ci_high}


@register(
    name="hedges_g",
    kind="effect",
    stage="hypothesis",
    code_template=(
        "edacore.effect_sizes.hedges_g({df}, outcome={outcome!r}, "
        "group={group!r}, groups={groups!r}, ci={ci})"
    ),
)
def hedges_g(
    df: pd.DataFrame, outcome: str, group: str, groups: tuple[str, str], ci: float = 0.95
) -> dict[str, float]:
    """Small-sample bias-corrected Cohen's d (exact J correction)."""
    d_result = cohens_d(df, outcome, group, groups, ci=ci)
    g1_vals, g2_vals = groups
    n1 = int((df[group] == g1_vals).sum())
    n2 = int((df[group] == g2_vals).sum())
    j = _hedges_j(n1 + n2 - 2)
    return {
        "estimate": d_result["estimate"] * j,
        "ci_low": d_result["ci_low"] * j,
        "ci_high": d_result["ci_high"] * j,
    }


@register(
    name="hedges_g_av",
    kind="effect",
    stage="hypothesis",
    code_template=(
        "edacore.effect_sizes.hedges_g_av({df}, outcome={outcome!r}, "
        "group={group!r}, groups={groups!r}, ci={ci})"
    ),
)
def hedges_g_av(
    df: pd.DataFrame, outcome: str, group: str, groups: tuple[str, str], ci: float = 0.95
) -> dict[str, float]:
    """Hedges' g with the *average*-variance denominator sqrt((s1^2+s2^2)/2)
    instead of the pooled one -- effectsize's `hedges_g(pooled_sd = FALSE)`,
    labelled "g(av)" there, and the effect size TOSTER pairs with a Welch
    t-test.

    Unlike `hedges_g`, this does not assume equal variances, which is why
    `welch_t` reports it: a Welch test that deliberately drops the
    equal-variance assumption should not have its effect size smuggle the
    assumption back in through a pooled SD. Both the J small-sample
    correction and the noncentral-t interval use the Welch-Satterthwaite
    df, not n1 + n2 - 2. Verified to ~1e-8 against R.
    """
    g1, g2 = groups
    x = df.loc[df[group] == g1, outcome].dropna().to_numpy(dtype=float)
    y = df.loc[df[group] == g2, outcome].dropna().to_numpy(dtype=float)
    n1, n2 = len(x), len(y)
    v1, v2 = x.var(ddof=1), y.var(ddof=1)
    denom_sd = np.sqrt((v1 + v2) / 2)
    d = (x.mean() - y.mean()) / denom_sd

    hn = 2 * (n2 * v1 + n1 * v2) / (n1 * n2 * (v1 + v2))
    se_sq1, se_sq2 = v1 / n1, v2 / n2
    se = np.sqrt(se_sq1 + se_sq2)
    df_welch = se**4 / (se_sq1**2 / (n1 - 1) + se_sq2**2 / (n2 - 1))
    t_obs = (x.mean() - y.mean()) / se
    ci_low, ci_high = _ncp_ci(t_obs, df_welch, hn, ci)
    j = _hedges_j(df_welch)
    return {"estimate": float(d * j), "ci_low": ci_low * j, "ci_high": ci_high * j}


@register(
    name="cohens_d_one_sample",
    kind="effect",
    stage="hypothesis",
    code_template=(
        "edacore.effect_sizes.cohens_d_one_sample({df}, col={col!r}, mu0={mu0}, ci={ci})"
    ),
)
def cohens_d_one_sample(
    df: pd.DataFrame, col: str, mu0: float, ci: float = 0.95
) -> dict[str, float]:
    """Cohen's d against a reference value: (mean(x) - mu0) / sd(x).

    Matches effectsize::cohens_d(x, mu=) including its noncentral-t
    interval (df = n - 1, hn = 1/n). Verified to ~1e-8 against R.
    """
    x = df[col].dropna().to_numpy(dtype=float)
    n = len(x)
    sd = x.std(ddof=1)
    d = (x.mean() - mu0) / sd
    t_obs = (x.mean() - mu0) / (sd / np.sqrt(n))
    ci_low, ci_high = _ncp_ci(t_obs, n - 1, 1 / n, ci)
    return {"estimate": float(d), "ci_low": ci_low, "ci_high": ci_high}


@register(
    name="glass_delta",
    kind="effect",
    stage="hypothesis",
    code_template=(
        "edacore.effect_sizes.glass_delta({df}, outcome={outcome!r}, "
        "group={group!r}, groups={groups!r}, ci={ci})"
    ),
)
def glass_delta(
    df: pd.DataFrame, outcome: str, group: str, groups: tuple[str, str], ci: float = 0.95
) -> dict[str, float]:
    """Glass's delta: (mean1 - mean2) / sd(group2), bias-corrected.

    `groups[1]` is the reference/control group whose SD is the denominator
    (matches effectsize::glass_delta's default y = second sample). Uses the
    same small-sample correction as hedges_g (effectsize's `adjust=TRUE`
    default), with df = n2 - 1.
    """
    g1, g2 = groups
    x = df.loc[df[group] == g1, outcome].dropna().to_numpy(dtype=float)
    y = df.loc[df[group] == g2, outcome].dropna().to_numpy(dtype=float)
    n1, n2 = len(x), len(y)
    s1, s2 = x.std(ddof=1), y.std(ddof=1)
    raw = (x.mean() - y.mean()) / s2

    hn = 1 / n2 + s1**2 / (n1 * s2**2)
    se = s2 * np.sqrt(hn)
    t_obs = (x.mean() - y.mean()) / se
    df1 = n2 - 1
    ci_low, ci_high = _ncp_ci(t_obs, df1, hn, ci)
    j = _hedges_j(df1)
    return {"estimate": raw * j, "ci_low": ci_low * j, "ci_high": ci_high * j}


@register(
    name="rank_biserial",
    kind="effect",
    stage="hypothesis",
    code_template=(
        "edacore.effect_sizes.rank_biserial({df}, outcome={outcome!r}, "
        "group={group!r}, groups={groups!r}, ci={ci})"
    ),
)
def rank_biserial(
    df: pd.DataFrame, outcome: str, group: str, groups: tuple[str, str], ci: float = 0.95
) -> dict[str, float]:
    """Rank-biserial correlation: 2U/(n1*n2) - 1, from the Mann-Whitney U
    statistic for group[0] vs group[1].

    The CI is an exact closed form, not a bootstrap: effectsize's own
    `rank_biserial` Fisher-z-transforms the correlation and uses an
    analytic standard error (verified to ~1e-13 against R) — ported
    directly rather than approximated.
    """
    g1, g2 = groups
    x = df.loc[df[group] == g1, outcome].dropna().to_numpy(dtype=float)
    y = df.loc[df[group] == g2, outcome].dropna().to_numpy(dtype=float)
    n1, n2 = len(x), len(y)
    u_stat, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    estimate = 2 * u_stat / (n1 * n2) - 1

    rank_z = np.arctanh(estimate)
    se = np.sqrt((n1 + n2 + 1) / (3 * n1 * n2))
    z = stats.norm.ppf(1 - (1 - ci) / 2)
    ci_low = float(np.tanh(rank_z - z * se))
    ci_high = float(np.tanh(rank_z + z * se))
    return {"estimate": float(estimate), "ci_low": ci_low, "ci_high": ci_high}


@register(
    name="cliffs_delta",
    kind="effect",
    stage="hypothesis",
    code_template=(
        "edacore.effect_sizes.cliffs_delta({df}, outcome={outcome!r}, "
        "group={group!r}, groups={groups!r}, ci={ci})"
    ),
)
def cliffs_delta(
    df: pd.DataFrame, outcome: str, group: str, groups: tuple[str, str], ci: float = 0.95
) -> dict[str, float]:
    """Cliff's delta. For two independent samples this is numerically
    identical to rank_biserial (effectsize labels both the same way); kept
    as a separate function because the two are conceptually distinct
    measures that happen to coincide here."""
    return rank_biserial(df, outcome, group, groups, ci=ci)


# --------------------------------------------------------------------------
# Paired
# --------------------------------------------------------------------------


@register(
    name="rank_biserial_one_sample",
    kind="effect",
    stage="hypothesis",
    code_template=(
        "edacore.effect_sizes.rank_biserial_one_sample({df}, col={col!r}, mu0={mu0}, ci={ci})"
    ),
)
def rank_biserial_one_sample(
    df: pd.DataFrame, col: str, mu0: float, ci: float = 0.95
) -> dict[str, float]:
    """Rank-biserial correlation against a reference value: the signed-rank
    version, computed on (x - mu0). Matches effectsize::rank_biserial(mu=)
    including its analytic (non-bootstrap) CI, to ~1e-16.
    """
    x = df[col].dropna().to_numpy(dtype=float)
    return _signed_rank_biserial(x - mu0, ci)


@register(
    name="rank_biserial_paired",
    kind="effect",
    stage="hypothesis",
    code_template=("edacore.effect_sizes.rank_biserial_paired({df}, a={a!r}, b={b!r}, ci={ci})"),
)
def rank_biserial_paired(df: pd.DataFrame, a: str, b: str, ci: float = 0.95) -> dict[str, float]:
    """Rank-biserial correlation for paired measurements, on (a - b).

    Matches effectsize::rank_biserial(a, b, paired=TRUE) including its
    analytic CI, to ~1e-16. Note the standard error differs from the
    independent-samples one in `rank_biserial` -- the signed-rank statistic
    has a different null variance from Mann-Whitney's U.
    """
    paired = df[[a, b]].dropna()
    diffs = (paired[a] - paired[b]).to_numpy(dtype=float)
    return _signed_rank_biserial(diffs, ci)


@register(
    name="d_z",
    kind="effect",
    stage="hypothesis",
    code_template="edacore.effect_sizes.d_z({df}, a={a!r}, b={b!r}, ci={ci})",
)
def d_z(df: pd.DataFrame, a: str, b: str, ci: float = 0.95) -> dict[str, float]:
    """Standardized mean difference for paired samples: mean(b - a) / sd(b - a)."""
    paired = df[[a, b]].dropna()
    diff = (paired[b] - paired[a]).to_numpy(dtype=float)
    n = len(diff)
    sd = diff.std(ddof=1)
    estimate = diff.mean() / sd

    se = sd / np.sqrt(n)
    t_obs = diff.mean() / se
    ci_low, ci_high = _ncp_ci(t_obs, n - 1, 1 / n, ci)
    return {"estimate": float(estimate), "ci_low": ci_low, "ci_high": ci_high}


# --------------------------------------------------------------------------
# k independent groups
# --------------------------------------------------------------------------


def _anova_sums_of_squares(df: pd.DataFrame, outcome: str, group: str) -> dict[str, float]:
    grand_mean = df[outcome].mean()
    groups = [g[outcome].to_numpy(dtype=float) for _, g in df.groupby(group, observed=True)]
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = ((df[outcome] - grand_mean) ** 2).sum()
    ss_within = ss_total - ss_between
    k = len(groups)
    n = len(df)
    return {"ss_between": ss_between, "ss_within": ss_within, "ss_total": ss_total, "k": k, "n": n}


@register(
    name="eta_squared",
    kind="effect",
    stage="hypothesis",
    code_template=(
        "edacore.effect_sizes.eta_squared({df}, outcome={outcome!r}, group={group!r}, ci={ci})"
    ),
)
def eta_squared(df: pd.DataFrame, outcome: str, group: str, ci: float = 0.95) -> dict[str, float]:
    """SS_between / SS_total for a one-way design.

    CI is one-sided (matching effectsize's `alternative="greater"`
    default): ci_low via noncentral-F inversion, ci_high fixed at 1.
    Verified to ~1e-9 against R.
    """
    ss = _anova_sums_of_squares(df, outcome, group)
    estimate = float(ss["ss_between"] / ss["ss_total"])
    df1, df2 = ss["k"] - 1, ss["n"] - ss["k"]
    ci_low = _pve_ci_low(estimate, df1, df2, ci)
    return {"estimate": estimate, "ci_low": ci_low, "ci_high": 1.0}


@register(
    name="partial_eta_squared",
    kind="effect",
    stage="hypothesis",
    code_template=(
        "edacore.effect_sizes.partial_eta_squared({df}, "
        "outcome={outcome!r}, group={group!r}, ci={ci})"
    ),
)
def partial_eta_squared(
    df: pd.DataFrame, outcome: str, group: str, ci: float = 0.95
) -> dict[str, float]:
    """Partial eta-squared for a one-way design.

    Mathematically identical to eta_squared here — partial and non-partial
    eta-squared only diverge with more than one factor (effectsize confirms
    this: "For one-way between subjects designs, partial eta squared is
    equivalent to eta squared"). Kept separate for when multi-factor
    designs (Section 6.7 two_way_anova) need the real partial version.
    """
    return eta_squared(df, outcome, group, ci=ci)


@register(
    name="omega_squared",
    kind="effect",
    stage="hypothesis",
    code_template=(
        "edacore.effect_sizes.omega_squared({df}, outcome={outcome!r}, group={group!r}, ci={ci})"
    ),
)
def omega_squared(df: pd.DataFrame, outcome: str, group: str, ci: float = 0.95) -> dict[str, float]:
    """Less-biased alternative to eta_squared:
    (SS_between - (k-1)*MS_within) / (SS_total + MS_within).

    CI is one-sided, via the same noncentral-F inversion as eta_squared
    but reconstructing a pseudo-F from the omega2 estimate first (matching
    effectsize's actual algorithm — verified to ~1e-9 against R, not an
    approximation). ci_high is fixed at 1.
    """
    ss = _anova_sums_of_squares(df, outcome, group)
    ms_within = ss["ss_within"] / (ss["n"] - ss["k"])
    estimate = float((ss["ss_between"] - (ss["k"] - 1) * ms_within) / (ss["ss_total"] + ms_within))
    df1, df2 = ss["k"] - 1, ss["n"] - ss["k"]
    ci_low = _pve_ci_low(estimate, df1, df2, ci)
    return {"estimate": estimate, "ci_low": ci_low, "ci_high": 1.0}


@register(
    name="epsilon_squared",
    kind="effect",
    stage="hypothesis",
    code_template=(
        "edacore.effect_sizes.epsilon_squared({df}, outcome={outcome!r}, group={group!r}, ci={ci})"
    ),
)
def epsilon_squared(
    df: pd.DataFrame, outcome: str, group: str, ci: float = 0.95
) -> dict[str, float]:
    """Rank epsilon-squared, the nonparametric (Kruskal-Wallis-based)
    analogue of eta_squared: H / (N - 1).

    Despite Section 6.8 grouping this with eta2/omega2, effectsize's own
    `rank_epsilon_squared` does NOT use a noncentral-F CI for it — its
    source (`.boot_two_group_es`) shows a plain percentile bootstrap that
    resamples WITHIN each group (preserving group sizes), R=200 by
    default. This uses a BCa bootstrap with that same within-group
    (stratified) resampling scheme instead of plain percentile, for a
    tighter interval; see test_effect_size_ci_coverage.py for the
    simulation-based validation this warrants in place of exact fixture
    matching.
    """

    def _eps2(arrays: list[np.ndarray]) -> float:
        if len(arrays) < 2 or any(len(a) == 0 for a in arrays):
            return float("nan")
        h, _ = stats.kruskal(*arrays)
        n_total = sum(len(a) for a in arrays)
        return float(h / (n_total - 1))

    groups = [g[outcome].to_numpy(dtype=float) for _, g in df.groupby(group, observed=True)]
    estimate = _eps2(groups)
    ci_low, ci_high = _bca_grouped_bootstrap_ci(_eps2, groups, ci=ci)
    return {"estimate": estimate, "ci_low": ci_low, "ci_high": ci_high}


@register(
    name="kendalls_w",
    kind="effect",
    stage="hypothesis",
    code_template=(
        "edacore.effect_sizes.kendalls_w({df}, value={value!r}, "
        "condition={condition!r}, subject={subject!r}, ci={ci})"
    ),
)
def kendalls_w(
    df: pd.DataFrame, value: str, condition: str, subject: str, ci: float = 0.95
) -> dict[str, Any]:
    """Kendall's W for repeated measures: Friedman chi-square / (N*(k-1)).

    effectsize's own `kendalls_w` CI is a plain percentile bootstrap
    (R's `boot::boot.ci(type="perc")`, R=200) resampling whole subjects
    (rows of its internal subject x condition matrix) — the same
    resampling unit this module already used. This uses BCa instead of
    plain percentile for a tighter interval; see
    test_effect_size_ci_coverage.py for the simulation-based validation
    that warrants in place of exact fixture matching.

    Small-n caveat: measured CI coverage for this BCa interval is below
    the nominal level for repeated-measures designs with few subjects,
    improving as the number of subjects grows (91.1% actual coverage of a
    95% CI at 25 subjects, 93.3% at 50, 94.5% at 100 — measured directly
    at 2000 simulated datasets per point, same data-generating process
    throughout). Treat a `kendalls_w` CI from a study with well under 100
    subjects as narrower than its stated confidence level; the point
    estimate itself has no such bias.
    """

    def _w(mat: np.ndarray) -> float:
        n_rows, n_cols = mat.shape
        if n_rows < 3:
            return float("nan")
        f_stat, _ = stats.friedmanchisquare(*[mat[:, j] for j in range(n_cols)])
        return float(f_stat / (n_rows * (n_cols - 1)))

    wide = df.pivot(index=subject, columns=condition, values=value)
    mat = wide.to_numpy(dtype=float)
    estimate = _w(mat)
    # Resampling individual rows would break the within-subject structure
    # (a resampled "subject" could end up missing conditions); resample
    # whole subjects (rows of the subject x condition matrix) instead.
    ci_low, ci_high = _bca_matrix_row_bootstrap_ci(_w, mat, ci=ci)

    n_subjects = mat.shape[0]
    warnings: list[str] = []
    if n_subjects < 50:
        warnings.append(
            f"CI computed from {n_subjects} subjects (< 50): measured BCa coverage is below "
            "the nominal level at this sample size (91.1% actual coverage of a 95% CI at 25 "
            "subjects, 93.3% at 50, 94.5% at 100 in simulation); treat this interval as "
            "narrower than stated. The point estimate is not affected."
        )
    return {"estimate": estimate, "ci_low": ci_low, "ci_high": ci_high, "warnings": warnings}


# --------------------------------------------------------------------------
# Categorical
# --------------------------------------------------------------------------


@register(
    name="cramers_v",
    kind="effect",
    stage="hypothesis",
    code_template=(
        "edacore.effect_sizes.cramers_v({df}, a={a!r}, "
        "b={b!r}, bias_correct={bias_correct}, ci={ci})"
    ),
)
def cramers_v(
    df: pd.DataFrame, a: str, b: str, bias_correct: bool = True, ci: float = 0.95
) -> dict[str, float]:
    """Cramer's V for an r x c contingency table (Bergsma's bias-corrected
    version by default, matching effectsize::cramers_v(adjust=TRUE)).

    CI is one-sided via noncentral chi-square inversion on the raw phi
    coefficient, then the same bias correction and k/l adjustment applied
    to the point estimate (verified to ~1e-8 against R); ci_high fixed at 1.
    """
    table = pd.crosstab(df[a], df[b]).to_numpy()
    chi2, _, _, _ = stats.chi2_contingency(table, correction=False)
    n = table.sum()
    r, c = table.shape
    df_chi2 = (r - 1) * (c - 1)

    phi_raw = np.sqrt(chi2 / n)
    ncp_low = _ncx2_ci_low(chi2, df_chi2, ci)
    phi_ci_low_raw = np.sqrt(ncp_low / n)

    if bias_correct:
        e = df_chi2 / (n - 1)
        phi_adj = np.sqrt(max(0.0, phi_raw**2 - e))
        phi_ci_low_adj = np.sqrt(max(0.0, phi_ci_low_raw**2 - e))
        k_dim = r - (r - 1) ** 2 / (n - 1)
        l_dim = c - (c - 1) ** 2 / (n - 1)
    else:
        phi_adj, phi_ci_low_adj = phi_raw, phi_ci_low_raw
        k_dim, l_dim = r, c

    denom = np.sqrt(min(k_dim - 1, l_dim - 1))
    estimate = float(phi_adj / denom)
    ci_low = float(phi_ci_low_adj / denom)
    return {"estimate": estimate, "ci_low": ci_low, "ci_high": 1.0}


@register(
    name="phi",
    kind="effect",
    stage="hypothesis",
    code_template="edacore.effect_sizes.phi({df}, a={a!r}, b={b!r}, ci={ci})",
)
def phi(df: pd.DataFrame, a: str, b: str, ci: float = 0.95) -> dict[str, float]:
    """Phi coefficient for a 2x2 table: sqrt(chi2/n), no continuity correction.

    CI is one-sided via noncentral chi-square inversion (verified to ~1e-9
    against R); ci_high fixed at 1.
    """
    table = pd.crosstab(df[a], df[b]).to_numpy()
    chi2, _, _, _ = stats.chi2_contingency(table, correction=False)
    n = table.sum()
    r, c = table.shape
    df_chi2 = (r - 1) * (c - 1)
    estimate = float(np.sqrt(chi2 / n))
    ncp_low = _ncx2_ci_low(chi2, df_chi2, ci)
    ci_low = float(np.sqrt(ncp_low / n))
    return {"estimate": estimate, "ci_low": ci_low, "ci_high": 1.0}


@register(
    name="cohens_w",
    kind="effect",
    stage="hypothesis",
    code_template="edacore.effect_sizes.cohens_w({df}, a={a!r}, b={b!r}, ci={ci})",
)
def cohens_w(df: pd.DataFrame, a: str, b: str, ci: float = 0.95) -> dict[str, float]:
    """Cohen's w for an r x c table: sqrt(chi2/n) (the general-table analogue of phi).

    CI is one-sided via noncentral chi-square inversion (verified to ~1e-9
    against R); ci_high is capped at sqrt(min(nrow, ncol) - 1) (the largest
    w an r x c table of this shape can produce), not 1 like phi/cramers_v.
    """
    table = pd.crosstab(df[a], df[b]).to_numpy()
    chi2, _, _, _ = stats.chi2_contingency(table, correction=False)
    n = table.sum()
    r, c = table.shape
    df_chi2 = (r - 1) * (c - 1)
    estimate = float(np.sqrt(chi2 / n))
    ncp_low = _ncx2_ci_low(chi2, df_chi2, ci)
    ci_low = float(np.sqrt(ncp_low / n))
    ci_high = float(np.sqrt(min(r, c) - 1))
    return {"estimate": estimate, "ci_low": ci_low, "ci_high": ci_high}


@register(
    name="odds_ratio",
    kind="effect",
    stage="hypothesis",
    code_template=(
        "edacore.effect_sizes.odds_ratio({df}, outcome={outcome!r}, "
        "group={group!r}, groups={groups!r}, event={event!r}, ci={ci})"
    ),
)
def odds_ratio(
    df: pd.DataFrame,
    outcome: str,
    group: str,
    groups: tuple[str, str],
    event: Any,
    ci: float = 0.95,
) -> dict[str, float]:
    """Odds ratio: odds of `event` in groups[0] vs groups[1]."""
    a, b, c, d = _two_by_two(df, outcome, group, groups, event)
    estimate = (a * d) / (b * c)
    se_log = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    ci_low, ci_high = _log_scale_ci(estimate, se_log, ci)
    return {"estimate": float(estimate), "ci_low": ci_low, "ci_high": ci_high}


@register(
    name="risk_ratio",
    kind="effect",
    stage="hypothesis",
    code_template=(
        "edacore.effect_sizes.risk_ratio({df}, outcome={outcome!r}, "
        "group={group!r}, groups={groups!r}, event={event!r}, ci={ci})"
    ),
)
def risk_ratio(
    df: pd.DataFrame,
    outcome: str,
    group: str,
    groups: tuple[str, str],
    event: Any,
    ci: float = 0.95,
) -> dict[str, float]:
    """Risk ratio: P(event | groups[0]) / P(event | groups[1])."""
    a, b, c, d = _two_by_two(df, outcome, group, groups, event)
    p1, p2 = a / (a + b), c / (c + d)
    estimate = p1 / p2
    se_log = np.sqrt((1 - p1) / a + (1 - p2) / c)
    ci_low, ci_high = _log_scale_ci(estimate, se_log, ci)
    return {"estimate": float(estimate), "ci_low": ci_low, "ci_high": ci_high}


@register(
    name="risk_difference",
    kind="effect",
    stage="hypothesis",
    code_template=(
        "edacore.effect_sizes.risk_difference({df}, outcome={outcome!r}, "
        "group={group!r}, groups={groups!r}, event={event!r}, ci={ci})"
    ),
)
def risk_difference(
    df: pd.DataFrame,
    outcome: str,
    group: str,
    groups: tuple[str, str],
    event: Any,
    ci: float = 0.95,
) -> dict[str, float]:
    """Risk difference: P(event | groups[0]) - P(event | groups[1]), with
    the Wald CI stats::prop.test(correct=FALSE) computes."""
    a, b, c, d = _two_by_two(df, outcome, group, groups, event)
    n1, n2 = a + b, c + d
    p1, p2 = a / n1, c / n2
    estimate = p1 - p2
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    z = stats.norm.ppf(1 - (1 - ci) / 2)
    return {
        "estimate": float(estimate),
        "ci_low": float(estimate - z * se),
        "ci_high": float(estimate + z * se),
    }


@register(
    name="cohens_h",
    kind="effect",
    stage="hypothesis",
    code_template=(
        "edacore.effect_sizes.cohens_h({df}, outcome={outcome!r}, "
        "group={group!r}, groups={groups!r}, event={event!r}, ci={ci})"
    ),
)
def cohens_h(
    df: pd.DataFrame,
    outcome: str,
    group: str,
    groups: tuple[str, str],
    event: Any,
    ci: float = 0.95,
) -> dict[str, float]:
    """Cohen's h: arcsine-transformed difference in proportions."""
    a, b, c, d = _two_by_two(df, outcome, group, groups, event)
    n1, n2 = a + b, c + d
    p1, p2 = a / n1, c / n2
    h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
    se_arcsin = 2 * np.sqrt(0.25 * (1 / n1 + 1 / n2))
    z = stats.norm.ppf(1 - (1 - ci) / 2)
    return {
        "estimate": float(h),
        "ci_low": float(h - z * se_arcsin),
        "ci_high": float(h + z * se_arcsin),
    }


@register(
    name="cohens_h_one_sample",
    kind="effect",
    stage="hypothesis",
    code_template=("edacore.effect_sizes.cohens_h_one_sample({df}, col={col!r}, p0={p0}, ci={ci})"),
)
def cohens_h_one_sample(
    df: pd.DataFrame, col: str, p0: float, ci: float = 0.95
) -> dict[str, float]:
    """Cohen's h against a reference proportion:
    2*asin(sqrt(p_hat)) - 2*asin(sqrt(p0)).

    h(p) is strictly increasing in p, so the exact (Clopper-Pearson)
    interval for p maps bound-for-bound onto an exact interval for h --
    no normal approximation, unlike the two-sample `cohens_h`. Verified
    against the same transform applied to `stats::binom.test`'s interval
    in R.
    """
    x = df[col].dropna().to_numpy(dtype=float)
    n = len(x)
    successes = int(x.sum())
    p_low, p_high = _clopper_pearson_ci(successes, n, ci)

    def _h(p: float) -> float:
        return float(2 * np.arcsin(np.sqrt(p)) - 2 * np.arcsin(np.sqrt(p0)))

    return {"estimate": _h(successes / n), "ci_low": _h(p_low), "ci_high": _h(p_high)}


@register(
    name="cohens_w_gof",
    kind="effect",
    stage="hypothesis",
    code_template=(
        "edacore.effect_sizes.cohens_w_gof({df}, col={col!r}, expected={expected!r}, ci={ci})"
    ),
)
def cohens_w_gof(
    df: pd.DataFrame, col: str, expected: dict[str, float], ci: float = 0.95
) -> dict[str, float]:
    """Cohen's w for a goodness-of-fit test: sqrt(chi2/n) against reference
    proportions, rather than against independence in a two-way table.

    Matches effectsize::cohens_w(x, p=) exactly, including its one-sided
    interval: ci_low by noncentral chi-square inversion, ci_high at the
    largest w these reference proportions admit, sqrt(1/min(p) - 1) -- not
    1, and not `cohens_w`'s sqrt(min(nrow, ncol) - 1).
    """
    observed_counts = df[col].value_counts()
    categories = list(expected.keys())
    observed = np.array([float(observed_counts.get(c, 0)) for c in categories])
    probs = np.array([expected[c] for c in categories], dtype=float)
    n = float(observed.sum())
    expected_counts = probs * n

    chi2 = float(np.sum((observed - expected_counts) ** 2 / expected_counts))
    dof = float(len(categories) - 1)
    estimate = float(np.sqrt(chi2 / n))
    ci_low = float(np.sqrt(_ncx2_ci_low(chi2, dof, ci) / n))
    ci_high = float(np.sqrt(1 / probs.min() - 1))
    return {"estimate": estimate, "ci_low": ci_low, "ci_high": ci_high}
