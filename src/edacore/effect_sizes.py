"""Effect sizes (ARCHITECTURE.md, Section 6.8).

Every point estimate here is verified to match R's `effectsize` package
(tests/fixtures/r_reference/) to at least 1e-6. Confidence intervals for
cohens_d, hedges_g, glass_delta, d_z, cohens_h, odds_ratio, risk_ratio, and
risk_difference use the exact method `effectsize`/`stats::prop.test` use
(noncentral-t inversion for the mean-difference family, log-scale Wald for
the ratio family, arcsine for cohens_h) and are verified the same way.

rank_biserial, cliffs_delta, eta_squared, partial_eta_squared,
omega_squared, epsilon_squared, kendalls_w, cramers_v, phi, and cohens_w
use `bootstrap_effect_ci` (percentile bootstrap) instead: `effectsize`
computes their CIs with noncentral-F / specialized asymptotic methods this
module doesn't reproduce, so treat the point estimate as R-verified and the
CI as approximate, not bit-for-bit matched. See ARCHITECTURE.md Section 18
Changelog, M2 entry.
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
    matching effectsize's `.get_ncp_t` + `sqrt(hn)` scaling exactly."""
    alpha = 1 - ci
    ncp_low = _invert_nct_cdf(t_obs, df, 1 - alpha / 2)
    ncp_high = _invert_nct_cdf(t_obs, df, alpha / 2)
    return ncp_low * np.sqrt(hn), ncp_high * np.sqrt(hn)


def _log_scale_ci(estimate: float, se_log: float, ci: float) -> tuple[float, float]:
    """Wald CI on the log scale, for ratio measures (odds ratio, risk ratio)."""
    z = stats.norm.ppf(1 - (1 - ci) / 2)
    log_est = np.log(estimate)
    return float(np.exp(log_est - z * se_log)), float(np.exp(log_est + z * se_log))


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


def _cluster_bootstrap_ci(
    fn: Any,
    df: pd.DataFrame,
    cluster: str,
    *,
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state: int = 0,
    **kwargs: Any,
) -> tuple[float, float]:
    """Like `bootstrap_effect_ci`, but resamples whole `cluster` groups
    (e.g. subjects) rather than individual rows — required for
    repeated-measures data, where resampling rows independently breaks the
    within-subject structure a function like kendalls_w depends on."""
    rng = np.random.default_rng(random_state)
    cluster_ids = df[cluster].unique()
    n = len(cluster_ids)
    estimates = np.empty(n_boot)
    for i in range(n_boot):
        chosen = rng.choice(cluster_ids, size=n, replace=True)
        # Relabel resampled clusters uniquely: the same subject drawn twice
        # must not collapse into one row group under a pivot/groupby.
        parts = [
            df.loc[df[cluster] == cid].assign(**{cluster: f"{cid}__{draw}"})
            for draw, cid in enumerate(chosen)
        ]
        resampled = pd.concat(parts, ignore_index=True)
        estimates[i] = fn(resampled, **kwargs)
    alpha = 1 - ci
    lo, hi = np.nanpercentile(estimates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


_MAGNITUDE_THRESHOLDS: dict[str, tuple[float, float, float]] = {
    # measure -> (small, medium, large), compared against abs(value)
    "cohens_d": (0.2, 0.5, 0.8),
    "hedges_g": (0.2, 0.5, 0.8),
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
}


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
    statistic for group[0] vs group[1]. CI is bootstrap (see module note)."""
    g1, g2 = groups
    x = df.loc[df[group] == g1, outcome].dropna().to_numpy(dtype=float)
    y = df.loc[df[group] == g2, outcome].dropna().to_numpy(dtype=float)
    n1, n2 = len(x), len(y)
    u_stat, _ = stats.mannwhitneyu(x, y, alternative="two-sided")
    estimate = 2 * u_stat / (n1 * n2) - 1

    def _stat(sample_df: pd.DataFrame) -> float:
        sx = sample_df.loc[sample_df[group] == g1, outcome].dropna().to_numpy(dtype=float)
        sy = sample_df.loc[sample_df[group] == g2, outcome].dropna().to_numpy(dtype=float)
        if len(sx) == 0 or len(sy) == 0:
            return float("nan")
        u, _ = stats.mannwhitneyu(sx, sy, alternative="two-sided")
        return float(2 * u / (len(sx) * len(sy)) - 1)

    ci_low, ci_high = bootstrap_effect_ci(_stat, df, ci=ci)
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

    CI is bootstrap (see module note); effectsize uses a noncentral-F CI.
    """
    ss = _anova_sums_of_squares(df, outcome, group)
    estimate = ss["ss_between"] / ss["ss_total"]

    def _stat(sample_df: pd.DataFrame) -> float:
        s = _anova_sums_of_squares(sample_df, outcome, group)
        if s["ss_total"] == 0:
            return float("nan")
        return float(s["ss_between"] / s["ss_total"])

    ci_low, ci_high = bootstrap_effect_ci(_stat, df, ci=ci)
    return {"estimate": float(estimate), "ci_low": ci_low, "ci_high": ci_high}


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
    (SS_between - (k-1)*MS_within) / (SS_total + MS_within)."""
    ss = _anova_sums_of_squares(df, outcome, group)
    ms_within = ss["ss_within"] / (ss["n"] - ss["k"])
    estimate = (ss["ss_between"] - (ss["k"] - 1) * ms_within) / (ss["ss_total"] + ms_within)

    def _stat(sample_df: pd.DataFrame) -> float:
        s = _anova_sums_of_squares(sample_df, outcome, group)
        if s["n"] <= s["k"]:
            return float("nan")
        msw = s["ss_within"] / (s["n"] - s["k"])
        denom = s["ss_total"] + msw
        if denom == 0:
            return float("nan")
        return float((s["ss_between"] - (s["k"] - 1) * msw) / denom)

    ci_low, ci_high = bootstrap_effect_ci(_stat, df, ci=ci)
    return {"estimate": float(estimate), "ci_low": ci_low, "ci_high": ci_high}


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
    analogue of eta_squared: H / (N - 1)."""
    groups = [g[outcome].to_numpy(dtype=float) for _, g in df.groupby(group, observed=True)]
    h_stat, _ = stats.kruskal(*groups)
    n = len(df)
    estimate = h_stat / (n - 1)

    def _stat(sample_df: pd.DataFrame) -> float:
        gs = [g[outcome].to_numpy(dtype=float) for _, g in sample_df.groupby(group, observed=True)]
        if len(gs) < 2 or any(len(g) == 0 for g in gs):
            return float("nan")
        h, _ = stats.kruskal(*gs)
        return float(h / (len(sample_df) - 1))

    ci_low, ci_high = bootstrap_effect_ci(_stat, df, ci=ci)
    return {"estimate": float(estimate), "ci_low": ci_low, "ci_high": ci_high}


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
) -> dict[str, float]:
    """Kendall's W for repeated measures: Friedman chi-square / (N*(k-1))."""
    wide = df.pivot(index=subject, columns=condition, values=value)
    n, k = wide.shape
    fr_stat, _ = stats.friedmanchisquare(*[wide[c].to_numpy(dtype=float) for c in wide.columns])
    estimate = fr_stat / (n * (k - 1))

    def _stat(sample_df: pd.DataFrame) -> float:
        w = sample_df.pivot_table(index=subject, columns=condition, values=value, aggfunc="mean")
        w = w.dropna()
        if w.shape[0] < 3:
            return float("nan")
        f_stat, _ = stats.friedmanchisquare(*[w[c].to_numpy(dtype=float) for c in w.columns])
        return float(f_stat / (w.shape[0] * (w.shape[1] - 1)))

    # Resampling individual rows would break the within-subject structure
    # (a resampled "subject" could end up missing conditions); resample
    # whole subjects instead.
    ci_low, ci_high = _cluster_bootstrap_ci(_stat, df, subject, ci=ci)
    return {"estimate": float(estimate), "ci_low": ci_low, "ci_high": ci_high}


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
    version by default, matching effectsize::cramers_v(adjust=TRUE))."""
    table = pd.crosstab(df[a], df[b]).to_numpy()
    chi2, _, _, _ = stats.chi2_contingency(table, correction=False)
    n = table.sum()
    r, c = table.shape
    if bias_correct:
        phi2_tilde = max(0.0, chi2 / n - (r - 1) * (c - 1) / (n - 1))
        r_tilde = r - (r - 1) ** 2 / (n - 1)
        k_tilde = c - (c - 1) ** 2 / (n - 1)
        estimate = float(np.sqrt(phi2_tilde / min(r_tilde - 1, k_tilde - 1)))
    else:
        estimate = float(np.sqrt((chi2 / n) / (min(r, c) - 1)))

    def _stat(sample_df: pd.DataFrame) -> float:
        t = pd.crosstab(sample_df[a], sample_df[b]).to_numpy()
        if t.shape[0] < 2 or t.shape[1] < 2:
            return float("nan")
        c2, _, _, _ = stats.chi2_contingency(t, correction=False)
        nn = t.sum()
        rr, cc = t.shape
        if bias_correct:
            p2 = max(0.0, c2 / nn - (rr - 1) * (cc - 1) / (nn - 1))
            rt = rr - (rr - 1) ** 2 / (nn - 1)
            kt = cc - (cc - 1) ** 2 / (nn - 1)
            return float(np.sqrt(p2 / min(rt - 1, kt - 1)))
        return float(np.sqrt((c2 / nn) / (min(rr, cc) - 1)))

    ci_low, ci_high = bootstrap_effect_ci(_stat, df, ci=ci)
    return {"estimate": estimate, "ci_low": ci_low, "ci_high": ci_high}


@register(
    name="phi",
    kind="effect",
    stage="hypothesis",
    code_template="edacore.effect_sizes.phi({df}, a={a!r}, b={b!r}, ci={ci})",
)
def phi(df: pd.DataFrame, a: str, b: str, ci: float = 0.95) -> dict[str, float]:
    """Phi coefficient for a 2x2 table: sqrt(chi2/n), no continuity correction."""
    table = pd.crosstab(df[a], df[b]).to_numpy()
    chi2, _, _, _ = stats.chi2_contingency(table, correction=False)
    n = table.sum()
    estimate = float(np.sqrt(chi2 / n))

    def _stat(sample_df: pd.DataFrame) -> float:
        t = pd.crosstab(sample_df[a], sample_df[b]).to_numpy()
        if t.shape != (2, 2):
            return float("nan")
        c2, _, _, _ = stats.chi2_contingency(t, correction=False)
        return float(np.sqrt(c2 / t.sum()))

    ci_low, ci_high = bootstrap_effect_ci(_stat, df, ci=ci)
    return {"estimate": estimate, "ci_low": ci_low, "ci_high": ci_high}


@register(
    name="cohens_w",
    kind="effect",
    stage="hypothesis",
    code_template="edacore.effect_sizes.cohens_w({df}, a={a!r}, b={b!r}, ci={ci})",
)
def cohens_w(df: pd.DataFrame, a: str, b: str, ci: float = 0.95) -> dict[str, float]:
    """Cohen's w for an r x c table: sqrt(chi2/n) (the general-table analogue of phi)."""
    table = pd.crosstab(df[a], df[b]).to_numpy()
    chi2, _, _, _ = stats.chi2_contingency(table, correction=False)
    n = table.sum()
    estimate = float(np.sqrt(chi2 / n))

    def _stat(sample_df: pd.DataFrame) -> float:
        t = pd.crosstab(sample_df[a], sample_df[b]).to_numpy()
        if t.shape[0] < 2 or t.shape[1] < 2:
            return float("nan")
        c2, _, _, _ = stats.chi2_contingency(t, correction=False)
        return float(np.sqrt(c2 / t.sum()))

    ci_low, ci_high = bootstrap_effect_ci(_stat, df, ci=ci)
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
