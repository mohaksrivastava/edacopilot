"""k-independent-group hypothesis tests (ARCHITECTURE.md, Section 6.7).

Verified against R (tests/fixtures/r_reference/). Two functions needed a
formula ported from R source rather than a scipy/statsmodels builtin:

- alexander_govern: no scipy/statsmodels equivalent exists at all; ported
  exactly from onewaytests::ag.test's source (Alexander & Govern 1994).
- welch_anova's omega-squared: effectsize::omega_squared applies its
  F_to_omega2 *approximation* formula (max(0, ((F-1)*df1)/(F*df1+df2+1)))
  when given a Welch htest object, not the classical SS-based formula
  edacore.effect_sizes.omega_squared uses for one_way_anova -- the two are
  genuinely different numbers for the same data (confirmed against the R
  fixture), not a rounding difference.
"""

from __future__ import annotations

from math import factorial
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from edacore.contracts import TestResult
from edacore.effect_sizes import (
    _anova_sums_of_squares,
    epsilon_squared,
    eta_squared,
    magnitude_label,
    omega_squared,
    omega_squared_from_f,
)
from edacore.registry import register
from edacore.stattests._shared import MAX_EXACT_PERMUTATIONS, NanPolicy


def _clean_k_groups(
    df: pd.DataFrame, outcome: str, group: str, nan_policy: NanPolicy, warnings: list[str]
) -> pd.DataFrame:
    sub = df[[outcome, group]]
    n_before = len(sub)
    clean = sub.dropna()
    n_dropped = n_before - len(clean)
    if n_dropped > 0:
        if nan_policy == "raise":
            raise ValueError(f"{n_dropped} row(s) missing '{outcome}'/'{group}', nan_policy=raise")
        warnings.append(
            f"{n_dropped} row(s) dropped: missing '{outcome}' or '{group}' (nan_policy='omit')"
        )
    n_groups = clean[group].nunique()
    if n_groups < 2:
        raise ValueError(f"'{group}' must have at least 2 distinct values, found {n_groups}")
    return clean


def _group_n(clean: pd.DataFrame, group: str) -> dict[str, int]:
    return {str(k): int(v) for k, v in clean.groupby(group, observed=True).size().items()}


@register(
    name="one_way_anova",
    kind="test",
    stage="hypothesis",
    tags={"parametric", "k_independent"},
    assumptions={
        "hard": ["numeric_outcome", "independent"],
        "soft": ["normality", "equal_variance"],
    },
    estimand="mean of the outcome across k independent groups",
    code_template=(
        "edacore.stattests.k_independent.one_way_anova("
        "{df}, outcome={outcome!r}, group={group!r}, ci={ci}, nan_policy={nan_policy!r})"
    ),
)
def one_way_anova(
    df: pd.DataFrame, outcome: str, group: str, ci: float = 0.95, nan_policy: NanPolicy = "omit"
) -> TestResult:
    """Effect size is eta-squared (primary); omega-squared (the
    less-biased alternative) is also computed and reported in
    validity_notes -- TestResult has one effect_size slot, and unlike
    mann_whitney's rank-biserial/Cliff's-delta pair these two are
    genuinely different numbers, not equivalent representations."""
    warnings: list[str] = []
    clean = _clean_k_groups(df, outcome, group, nan_policy, warnings)
    ss = _anova_sums_of_squares(clean, outcome, group)
    df1, df2 = ss["k"] - 1, ss["n"] - ss["k"]
    f_stat = (ss["ss_between"] / df1) / (ss["ss_within"] / df2)
    p_value = float(stats.f.sf(f_stat, df1, df2))

    eta = eta_squared(clean, outcome, group, ci)
    omega = omega_squared(clean, outcome, group, ci)

    return TestResult(
        fact_id=f"one_way_anova.{outcome}",
        function="one_way_anova",
        estimand=f"mean of '{outcome}' across groups of '{group}'",
        statistic=float(f_stat),
        statistic_name="F",
        df=(float(df1), float(df2)),
        p_value=p_value,
        effect_size=eta["estimate"],
        effect_size_name="eta_squared",
        effect_size_ci=(eta["ci_low"], eta["ci_high"]),
        effect_magnitude=magnitude_label(eta["estimate"], "eta_squared"),
        n=_group_n(clean, group),
        warnings=warnings,
        validity_notes=[
            f"omega_squared = {omega['estimate']:.6g} "
            f"(95% CI [{omega['ci_low']:.6g}, {omega['ci_high']:.6g}]) -- "
            "the less-biased alternative to eta_squared, not reported as the "
            "primary effect_size but included here for comparison."
        ],
    )


@register(
    name="welch_anova",
    kind="test",
    stage="hypothesis",
    tags={"parametric", "k_independent"},
    assumptions={"hard": ["numeric_outcome", "independent"], "soft": ["normality_or_large_n"]},
    estimand="mean of the outcome across k independent groups, unequal variance",
    code_template=(
        "edacore.stattests.k_independent.welch_anova("
        "{df}, outcome={outcome!r}, group={group!r}, ci={ci}, nan_policy={nan_policy!r})"
    ),
)
def welch_anova(
    df: pd.DataFrame, outcome: str, group: str, ci: float = 0.95, nan_policy: NanPolicy = "omit"
) -> TestResult:
    warnings: list[str] = []
    clean = _clean_k_groups(df, outcome, group, nan_policy, warnings)
    groups = [g[outcome].to_numpy(dtype=float) for _, g in clean.groupby(group, observed=True)]

    # scipy has no Welch-ANOVA entry point; replicate stats::oneway.test's
    # own formula directly (Welch 1951).
    k = len(groups)
    ni = np.array([len(g) for g in groups], dtype=float)
    mi = np.array([g.mean() for g in groups])
    vi = np.array([g.var(ddof=1) for g in groups])
    wi = ni / vi
    grand_mean = float(np.sum(wi * mi) / np.sum(wi))
    df1 = k - 1
    numer = float(np.sum(wi * (mi - grand_mean) ** 2) / df1)
    tmp = float(np.sum((1 - wi / np.sum(wi)) ** 2 / (ni - 1)) / (k**2 - 1))
    denom = 1 + 2 * (k - 2) * tmp
    f_stat = numer / denom
    df2 = 1 / (3 * tmp)
    p_value = float(stats.f.sf(f_stat, df1, df2))

    # effectsize::omega_squared's F_to_omega2 approximation for a Welch
    # htest (not the classical SS-based formula -- see module docstring),
    # with its one-sided noncentral-F interval on the Welch df.
    omega = omega_squared_from_f(f_stat, df1, df2, ci=ci)
    omega2 = omega["estimate"]

    return TestResult(
        fact_id=f"welch_anova.{outcome}",
        function="welch_anova",
        estimand=f"mean of '{outcome}' across groups of '{group}' (unequal variance)",
        statistic=float(f_stat),
        statistic_name="F",
        df=(float(df1), float(df2)),
        p_value=p_value,
        effect_size=float(omega2),
        effect_size_name="omega_squared",
        effect_size_ci=(omega["ci_low"], omega["ci_high"]),
        effect_magnitude=magnitude_label(float(omega2), "omega_squared")
        if omega2 > 0
        else "negligible",
        n=_group_n(clean, group),
        warnings=warnings,
    )


@register(
    name="alexander_govern",
    kind="test",
    stage="hypothesis",
    tags={"parametric", "k_independent", "robust"},
    assumptions={"hard": ["numeric_outcome", "independent"], "soft": ["normality"]},
    estimand="mean of the outcome across k independent groups, robust to unequal variance",
    code_template=(
        "edacore.stattests.k_independent.alexander_govern("
        "{df}, outcome={outcome!r}, group={group!r}, nan_policy={nan_policy!r})"
    ),
)
def alexander_govern(
    df: pd.DataFrame, outcome: str, group: str, nan_policy: NanPolicy = "omit"
) -> TestResult:
    """Ported exactly from onewaytests::ag.test's source (Alexander &
    Govern 1994) -- no scipy/statsmodels equivalent exists."""
    warnings: list[str] = []
    clean = _clean_k_groups(df, outcome, group, nan_policy, warnings)
    groups = [g[outcome].to_numpy(dtype=float) for _, g in clean.groupby(group, observed=True)]

    k = len(groups)
    ni = np.array([len(g) for g in groups], dtype=float)
    mi = np.array([g.mean() for g in groups])
    vi = np.array([g.var(ddof=1) for g in groups])
    se_i = np.sqrt(vi / ni)
    wi = (1 / se_i**2) / np.sum(1 / se_i**2)
    weighted_mean = float(np.sum(wi * mi))
    t_i = (mi - weighted_mean) / se_i
    a_i = ni - 1.5
    b_i = 48 * a_i**2
    c_i = np.sqrt(a_i * np.log(1 + t_i**2 / (ni - 1)))
    z_i = (
        c_i
        + (c_i**3 + 3 * c_i) / b_i
        - (4 * c_i**7 + 33 * c_i**5 + 240 * c_i**3 + 855 * c_i)
        / (10 * b_i**2 + 8 * b_i * c_i**4 + 1000 * b_i)
    )
    statistic = float(np.sum(z_i**2))
    dof = k - 1
    p_value = float(stats.chi2.sf(statistic, dof))

    return TestResult(
        fact_id=f"alexander_govern.{outcome}",
        function="alexander_govern",
        estimand=f"mean of '{outcome}' across groups of '{group}' (robust to unequal variance)",
        statistic=statistic,
        statistic_name="A",
        df=float(dof),
        p_value=p_value,
        n=_group_n(clean, group),
        warnings=warnings,
    )


@register(
    name="kruskal_wallis",
    kind="test",
    stage="hypothesis",
    tags={"nonparametric", "k_independent"},
    assumptions={"hard": ["ordinal_or_higher", "independent"], "soft": ["same_shape"]},
    estimand="stochastic dominance across k independent groups",
    code_template=(
        "edacore.stattests.k_independent.kruskal_wallis("
        "{df}, outcome={outcome!r}, group={group!r}, ci={ci}, nan_policy={nan_policy!r})"
    ),
)
def kruskal_wallis(
    df: pd.DataFrame, outcome: str, group: str, ci: float = 0.95, nan_policy: NanPolicy = "omit"
) -> TestResult:
    warnings: list[str] = []
    clean = _clean_k_groups(df, outcome, group, nan_policy, warnings)
    groups = [g[outcome].to_numpy(dtype=float) for _, g in clean.groupby(group, observed=True)]

    result = stats.kruskal(*groups)
    eps = epsilon_squared(clean, outcome, group, ci)

    return TestResult(
        fact_id=f"kruskal_wallis.{outcome}",
        function="kruskal_wallis",
        estimand=f"stochastic dominance of '{outcome}' across groups of '{group}'",
        statistic=float(result.statistic),
        statistic_name="H",
        df=float(len(groups) - 1),
        p_value=float(result.pvalue),
        effect_size=eps["estimate"],
        effect_size_name="epsilon_squared",
        effect_size_ci=(eps["ci_low"], eps["ci_high"]),
        effect_magnitude=magnitude_label(eps["estimate"], "epsilon_squared"),
        n=_group_n(clean, group),
        warnings=warnings,
    )


@register(
    name="permutation_anova",
    kind="test",
    stage="hypothesis",
    tags={"resampling", "k_independent"},
    assumptions={"hard": ["numeric_outcome", "independent"], "soft": []},
    estimand="mean of the outcome across k independent groups, permutation-based p-value",
    code_template=(
        "edacore.stattests.k_independent.permutation_anova("
        "{df}, outcome={outcome!r}, group={group!r}, n_resamples={n_resamples}, "
        "random_state={random_state}, nan_policy={nan_policy!r})"
    ),
)
def permutation_anova(
    df: pd.DataFrame,
    outcome: str,
    group: str,
    n_resamples: int = 10_000,
    random_state: int = 0,
    nan_policy: NanPolicy = "omit",
) -> TestResult:
    warnings: list[str] = []
    clean = _clean_k_groups(df, outcome, group, nan_policy, warnings)
    groups = [g[outcome].to_numpy(dtype=float) for _, g in clean.groupby(group, observed=True)]

    def _f_stat(*samples: np.ndarray, axis: int = -1) -> Any:
        return stats.f_oneway(*samples, axis=axis).statistic

    # Multinomial coefficient N! / (n1! n2! ... nk!): the number of
    # distinct ways to partition N observations into groups of these
    # sizes -- the k-group generalization of n_arrangements_two_sample.
    n_i = [len(g) for g in groups]
    n_arr = factorial(sum(n_i))
    for n in n_i:
        n_arr //= factorial(n)
    mode = "exact" if n_arr <= MAX_EXACT_PERMUTATIONS else "monte_carlo"
    resamples: float | int = np.inf if mode == "exact" else n_resamples

    result = stats.permutation_test(
        groups,
        _f_stat,
        permutation_type="independent",
        n_resamples=resamples,
        random_state=random_state,
        alternative="greater",
    )
    warnings.append(f"permutation mode: {mode} ({n_arr} possible arrangements)")

    eta = eta_squared(clean, outcome, group)

    return TestResult(
        fact_id=f"permutation_anova.{outcome}",
        function="permutation_anova",
        estimand=f"mean of '{outcome}' across groups of '{group}' (permutation-based p-value)",
        statistic=float(result.statistic),
        statistic_name="F",
        p_value=float(result.pvalue),
        effect_size=eta["estimate"],
        effect_size_name="eta_squared",
        effect_size_ci=(eta["ci_low"], eta["ci_high"]),
        effect_magnitude=magnitude_label(eta["estimate"], "eta_squared"),
        n=_group_n(clean, group),
        warnings=warnings,
    )
