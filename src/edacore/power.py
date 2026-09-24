"""Power analysis (ARCHITECTURE.md, Section 6.8).

Formulas are direct translations of R's `pwr` package source (`pwr.t.test`,
`pwr.anova.test`, `pwr.r.test`, `pwr.chisq.test`), not statsmodels'
`stats.power` classes — those use a different parameterization for ANOVA
(total N vs per-group n) and don't cover `pwr.r.test`'s bias-corrected
Fisher-z formula at all. Verified against tests/fixtures/r_reference/
required_sample_size__*.json and minimum_detectable_effect__*.json.

No post-hoc "observed power": Section 6.8 is explicit that this is a
function of the p-value and misleading, so it isn't offered here.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy import optimize, stats

from edacore.registry import register

TestFamily = Literal["two_sample_t", "anova", "correlation", "chi_square"]


def _solve_monotonic(f: Any, target: float, lo: float, hi: float) -> float:
    """Find x in (lo, hi) such that f(x) == target, where f is monotonic.

    scipy's noncentral t/F/chi2 distributions produce isolated NaNs at
    large parameter values (same issue effect_sizes._invert_nct_cdf works
    around), so this scans a grid first and brackets between the last two
    non-NaN points that straddle `target`, rather than handing scipy's
    brentq a wide range it might probe a NaN point within.
    """
    grid = np.geomspace(lo, hi, 200)
    vals = np.array([f(x) - target for x in grid])
    valid = ~np.isnan(vals)
    grid, vals = grid[valid], vals[valid]
    sign_changes = np.where(np.diff(np.sign(vals)) != 0)[0]
    if len(sign_changes) == 0:
        raise ValueError(f"no root found for target={target} in [{lo}, {hi}]")
    i = sign_changes[0]
    return float(optimize.brentq(lambda x: f(x) - target, grid[i], grid[i + 1]))


def _two_sample_t_power(n_per_group: float, d: float, alpha: float) -> float:
    df = 2 * (n_per_group - 1)
    ncp = d * np.sqrt(n_per_group / 2)
    crit = stats.t.ppf(1 - alpha / 2, df)
    return float(stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp))


def _anova_power(n_per_group: float, k: int, f: float, alpha: float) -> float:
    lam = k * n_per_group * f**2
    df1, df2 = k - 1, (n_per_group - 1) * k
    crit = stats.f.ppf(1 - alpha, df1, df2)
    return float(stats.ncf.sf(crit, df1, df2, lam))


def _correlation_power(n: float, r: float, alpha: float) -> float:
    """pwr.r.test's two-sided, bias-corrected arctanh method exactly."""
    ttt = stats.t.ppf(1 - alpha / 2, df=n - 2)
    rc = np.sqrt(ttt**2 / (ttt**2 + n - 2))
    zr = np.arctanh(r) + r / (2 * (n - 1))
    zrc = np.arctanh(rc)
    lower = stats.norm.cdf((zr - zrc) * np.sqrt(n - 3))
    upper = stats.norm.cdf((-zr - zrc) * np.sqrt(n - 3))
    return float(lower + upper)


def _chisq_power(n: float, w: float, df: int, alpha: float) -> float:
    crit = stats.chi2.ppf(1 - alpha, df)
    return float(stats.ncx2.sf(crit, df, n * w**2))


@register(
    name="required_sample_size",
    kind="effect",
    stage="hypothesis",
    code_template=(
        "edacore.power.required_sample_size({test!r}, effect={effect}, alpha={alpha}, "
        "power={power}, k={k!r}, df={df!r})"
    ),
    takes_df=False,
)
def required_sample_size(
    test: TestFamily,
    effect: float,
    alpha: float = 0.05,
    power: float = 0.8,
    k: int | None = None,
    df: int | None = None,
) -> dict[str, float]:
    """Sample size needed to detect `effect` at the given alpha/power.

    `k` (number of groups) is required for test="anova"; `df` is required
    for test="chi_square". Returns {"n_per_group": ...} for two_sample_t
    and anova (matching pwr's per-group convention), {"n": ...} for
    correlation and chi_square.
    """
    if test == "two_sample_t":
        n = _solve_monotonic(lambda n: _two_sample_t_power(n, effect, alpha), power, 2 + 1e-8, 1e6)
        return {"n_per_group": n}
    if test == "anova":
        if k is None:
            raise ValueError("test='anova' requires k (number of groups)")
        n = _solve_monotonic(lambda n: _anova_power(n, k, effect, alpha), power, 2 + 1e-8, 1e6)
        return {"n_per_group": n}
    if test == "correlation":
        n = _solve_monotonic(lambda n: _correlation_power(n, effect, alpha), power, 4 + 1e-8, 1e6)
        return {"n": n}
    if test == "chi_square":
        if df is None:
            raise ValueError("test='chi_square' requires df")
        n = _solve_monotonic(lambda n: _chisq_power(n, effect, df, alpha), power, 1 + 1e-8, 1e6)
        return {"n": n}
    raise ValueError(f"unknown test family '{test}'")


@register(
    name="minimum_detectable_effect",
    kind="effect",
    stage="hypothesis",
    code_template=(
        "edacore.power.minimum_detectable_effect({test!r}, n={n}, alpha={alpha}, "
        "power={power}, k={k!r}, df={df!r})"
    ),
    takes_df=False,
)
def minimum_detectable_effect(
    test: TestFamily,
    n: float,
    alpha: float = 0.05,
    power: float = 0.8,
    k: int | None = None,
    df: int | None = None,
) -> dict[str, float]:
    """The smallest effect detectable at `n`, alpha, and power.

    `n` is per-group for two_sample_t/anova, total for correlation/chi_square
    (matching required_sample_size's output convention). Returns
    {"d": ...} / {"f": ...} / {"r": ...} / {"w": ...} depending on `test`.
    """
    if test == "two_sample_t":
        d = _solve_monotonic(lambda d: _two_sample_t_power(n, d, alpha), power, 1e-8, 20)
        return {"d": d}
    if test == "anova":
        if k is None:
            raise ValueError("test='anova' requires k (number of groups)")
        f = _solve_monotonic(lambda f: _anova_power(n, k, f, alpha), power, 1e-8, 20)
        return {"f": f}
    if test == "correlation":
        r = _solve_monotonic(lambda r: _correlation_power(n, r, alpha), power, 1e-8, 1 - 1e-8)
        return {"r": r}
    if test == "chi_square":
        if df is None:
            raise ValueError("test='chi_square' requires df")
        w = _solve_monotonic(lambda w: _chisq_power(n, w, df, alpha), power, 1e-8, 20)
        return {"w": float(w)}
    raise ValueError(f"unknown test family '{test}'")
