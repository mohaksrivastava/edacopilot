"""Two-sample hypothesis tests: independent and paired (ARCHITECTURE.md,
Section 6.7).

Verified against R (tests/fixtures/r_reference/), with documented,
deliberate departures from scipy's defaults for mann_whitney and
wilcoxon_signed_rank (matching R's wilcox.test conventions) and for
yuen_trimmed_t's statistic sign (kept signed, unlike R's WRS2::yuen which
reports abs(t)) -- see ARCHITECTURE.md Section 18's M3 changelog entry.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats

from edacore.contracts import TestResult
from edacore.effect_sizes import magnitude_label
from edacore.registry import register
from edacore.stattests._shared import (
    MAX_EXACT_PERMUTATIONS,
    NanPolicy,
    clean_paired,
    n_arrangements_paired,
    n_arrangements_two_sample,
)
from edacore.stattests.one_sample import _wilcoxon_signed_rank_r_matched

PermMode = Literal["exact", "monte_carlo"]


def _clean_groups(
    df: pd.DataFrame, group_col: str, value_col: str, nan_policy: NanPolicy, warnings: list[str]
) -> tuple[np.ndarray, np.ndarray, Any, Any]:
    sub = df[[group_col, value_col]].dropna(subset=[group_col])
    labels = sub[group_col].unique()
    if len(labels) != 2:
        raise ValueError(f"'{group_col}' must have exactly 2 distinct values, found {len(labels)}")
    g1, g2 = labels[0], labels[1]

    n_before = len(sub)
    sub = sub.dropna(subset=[value_col])
    n_dropped = n_before - len(sub)
    if n_dropped > 0:
        if nan_policy == "raise":
            raise ValueError(f"{n_dropped} missing value(s) in '{value_col}', nan_policy='raise'")
        warnings.append(
            f"{n_dropped} missing value(s) dropped from '{value_col}' (nan_policy='omit')"
        )

    x = sub.loc[sub[group_col] == g1, value_col].to_numpy(dtype=float)
    y = sub.loc[sub[group_col] == g2, value_col].to_numpy(dtype=float)
    return x, y, g1, g2


def _winsorized_variance(x: np.ndarray, g: int) -> float:
    sorted_x = np.sort(x)
    n = len(sorted_x)
    winsorized = sorted_x.copy()
    if g > 0:
        winsorized[:g] = sorted_x[g]
        winsorized[n - g :] = sorted_x[n - g - 1]
    return float(np.var(winsorized, ddof=1))


def _trimmed_mean(x: np.ndarray, g: int) -> float:
    sorted_x = np.sort(x)
    n = len(sorted_x)
    if g == 0:
        return float(sorted_x.mean())
    return float(sorted_x[g : n - g].mean())


@register(
    name="student_t",
    kind="test",
    stage="hypothesis",
    tags={"parametric", "two_sample", "independent"},
    assumptions={"hard": ["numeric_outcome"], "soft": ["normality_or_large_n", "equal_variance"]},
    estimand="difference in means (independent groups, equal variance)",
    code_template=(
        "edacore.stattests.two_sample.student_t("
        "{df}, group_col={group_col!r}, value_col={value_col!r}, "
        "ci={ci}, nan_policy={nan_policy!r})"
    ),
)
def student_t(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    ci: float = 0.95,
    nan_policy: NanPolicy = "omit",
) -> TestResult:
    warnings: list[str] = []
    x, y, g1, g2 = _clean_groups(df, group_col, value_col, nan_policy, warnings)

    result = stats.ttest_ind(x, y, equal_var=True)
    ci_bounds = result.confidence_interval(ci)
    n1, n2 = len(x), len(y)
    pooled_sd = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2))
    d = float((x.mean() - y.mean()) / pooled_sd)

    return TestResult(
        fact_id=f"student_t.{value_col}",
        function="student_t",
        estimand=f"difference in means of '{value_col}' between '{g1}' and '{g2}'",
        statistic=float(result.statistic),
        statistic_name="t",
        df=float(result.df),
        p_value=float(result.pvalue),
        estimate=float(x.mean() - y.mean()),
        ci=(float(ci_bounds.low), float(ci_bounds.high)),
        ci_level=ci,
        effect_size=d,
        effect_size_name="cohens_d",
        effect_magnitude=magnitude_label(d, "cohens_d"),
        n={str(g1): n1, str(g2): n2},
        warnings=warnings,
    )


@register(
    name="welch_t",
    kind="test",
    stage="hypothesis",
    tags={"parametric", "two_sample", "independent"},
    assumptions={"hard": ["numeric_outcome"], "soft": ["normality_or_large_n"]},
    estimand="difference in means (independent groups, unequal variance)",
    code_template=(
        "edacore.stattests.two_sample.welch_t("
        "{df}, group_col={group_col!r}, value_col={value_col!r}, "
        "ci={ci}, nan_policy={nan_policy!r})"
    ),
)
def welch_t(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    ci: float = 0.95,
    nan_policy: NanPolicy = "omit",
) -> TestResult:
    warnings: list[str] = []
    x, y, g1, g2 = _clean_groups(df, group_col, value_col, nan_policy, warnings)

    result = stats.ttest_ind(x, y, equal_var=False)
    ci_bounds = result.confidence_interval(ci)
    n1, n2 = len(x), len(y)
    avg_var = (x.var(ddof=1) + y.var(ddof=1)) / 2
    d = float((x.mean() - y.mean()) / np.sqrt(avg_var))

    return TestResult(
        fact_id=f"welch_t.{value_col}",
        function="welch_t",
        estimand=f"difference in means of '{value_col}' between '{g1}' and '{g2}'",
        statistic=float(result.statistic),
        statistic_name="t",
        df=float(result.df),
        p_value=float(result.pvalue),
        estimate=float(x.mean() - y.mean()),
        ci=(float(ci_bounds.low), float(ci_bounds.high)),
        ci_level=ci,
        effect_size=d,
        effect_size_name="hedges_g",
        effect_magnitude=magnitude_label(d, "hedges_g"),
        n={str(g1): n1, str(g2): n2},
        warnings=warnings,
    )


@register(
    name="yuen_trimmed_t",
    kind="test",
    stage="hypothesis",
    tags={"robust", "two_sample", "independent"},
    assumptions={"hard": ["numeric_outcome"], "soft": []},
    estimand="difference in trimmed means (robust to outliers)",
    code_template=(
        "edacore.stattests.two_sample.yuen_trimmed_t("
        "{df}, group_col={group_col!r}, value_col={value_col!r}, "
        "trim={trim}, ci={ci}, nan_policy={nan_policy!r})"
    ),
)
def yuen_trimmed_t(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    trim: float = 0.2,
    ci: float = 0.95,
    nan_policy: NanPolicy = "omit",
) -> TestResult:
    """Yuen-Welch test on trimmed means (Yuen 1974), matching WRS2::yuen's
    formula for the statistic, df, p-value, estimate, and CI.

    One deliberate departure from WRS2::yuen: R's `test` field is always
    `abs(dif / se)` (unsigned), hard-coded in WRS2's source regardless of
    the sign of the trimmed-mean difference. This keeps the statistic
    signed (matching `estimate`'s sign and how student_t/welch_t/paired_t
    report signed statistics elsewhere in this module) -- confirmed
    against the R fixture, magnitude matches exactly, sign is the
    documented difference (ARCHITECTURE.md Section 18, M3 entry).
    """
    warnings: list[str] = []
    x, y, g1, g2 = _clean_groups(df, group_col, value_col, nan_policy, warnings)
    n1, n2 = len(x), len(y)
    g1_trim = int(np.floor(trim * n1))
    g2_trim = int(np.floor(trim * n2))
    h1, h2 = n1 - 2 * g1_trim, n2 - 2 * g2_trim

    m1, m2 = _trimmed_mean(x, g1_trim), _trimmed_mean(y, g2_trim)
    d1 = (n1 - 1) * _winsorized_variance(x, g1_trim) / (h1 * (h1 - 1))
    d2 = (n2 - 1) * _winsorized_variance(y, g2_trim) / (h2 * (h2 - 1))

    se = np.sqrt(d1 + d2)
    diff = m1 - m2
    t_stat = diff / se
    df_yuen = (d1 + d2) ** 2 / (d1**2 / (h1 - 1) + d2**2 / (h2 - 1))
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df_yuen))
    crit = stats.t.ppf(1 - (1 - ci) / 2, df_yuen)

    return TestResult(
        fact_id=f"yuen_trimmed_t.{value_col}",
        function="yuen_trimmed_t",
        estimand=f"difference in {trim:.0%}-trimmed means of '{value_col}' ('{g1}' vs '{g2}')",
        statistic=float(t_stat),
        statistic_name="t_yuen",
        df=float(df_yuen),
        p_value=float(p_value),
        estimate=float(diff),
        ci=(float(diff - crit * se), float(diff + crit * se)),
        ci_level=ci,
        n={str(g1): n1, str(g2): n2},
        warnings=warnings,
    )


@register(
    name="mann_whitney",
    kind="test",
    stage="hypothesis",
    tags={"nonparametric", "two_sample", "independent"},
    assumptions={"hard": ["ordinal_or_higher"], "soft": []},
    estimand="stochastic dominance between two independent groups",
    code_template=(
        "edacore.stattests.two_sample.mann_whitney("
        "{df}, group_col={group_col!r}, value_col={value_col!r}, nan_policy={nan_policy!r})"
    ),
)
def mann_whitney(
    df: pd.DataFrame, group_col: str, value_col: str, nan_policy: NanPolicy = "omit"
) -> TestResult:
    warnings: list[str] = []
    x, y, g1, g2 = _clean_groups(df, group_col, value_col, nan_policy, warnings)
    n1, n2 = len(x), len(y)

    # R's wilcox.test uses its exact permutation distribution when both
    # groups have fewer than 50 observations and there are no ties in the
    # combined data; otherwise a continuity-corrected normal approximation
    # (scipy's default use_continuity=True already matches that half).
    # scipy's own 'auto' method selection disagrees with R's threshold for
    # some n, giving a different p-value for the same data (confirmed
    # against the R fixture; ARCHITECTURE.md Section 18's M3 entry).
    combined = np.concatenate([x, y])
    has_ties = len(np.unique(combined)) != len(combined)
    method = "exact" if (n1 < 50 and n2 < 50 and not has_ties) else "asymptotic"
    result = stats.mannwhitneyu(x, y, alternative="two-sided", method=method)
    r = float(2 * result.statistic / (n1 * n2) - 1)

    return TestResult(
        fact_id=f"mann_whitney.{value_col}",
        function="mann_whitney",
        estimand=f"stochastic dominance of '{value_col}' between '{g1}' and '{g2}'",
        statistic=float(result.statistic),
        statistic_name="U",
        p_value=float(result.pvalue),
        effect_size=r,
        effect_size_name="rank_biserial",
        effect_magnitude=magnitude_label(r, "rank_biserial"),
        n={str(g1): n1, str(g2): n2},
        warnings=warnings,
    )


@register(
    name="brunner_munzel",
    kind="test",
    stage="hypothesis",
    tags={"nonparametric", "two_sample", "independent"},
    assumptions={"hard": ["ordinal_or_higher"], "soft": []},
    estimand="relative effect P(X < Y) + 0.5*P(X == Y) between two independent groups",
    code_template=(
        "edacore.stattests.two_sample.brunner_munzel("
        "{df}, group_col={group_col!r}, value_col={value_col!r}, nan_policy={nan_policy!r})"
    ),
)
def brunner_munzel(
    df: pd.DataFrame, group_col: str, value_col: str, nan_policy: NanPolicy = "omit"
) -> TestResult:
    warnings: list[str] = []
    x, y, g1, g2 = _clean_groups(df, group_col, value_col, nan_policy, warnings)

    result = stats.brunnermunzel(x, y)
    n1, n2 = len(x), len(y)
    combined_ranks = stats.rankdata(np.concatenate([x, y]))
    rank_x, rank_y = stats.rankdata(x), stats.rankdata(y)
    combined_rank_x, combined_rank_y = combined_ranks[:n1], combined_ranks[n1:]
    # Relative effect P(X<Y) + 0.5*P(X=Y): (mean rank of y - mean rank of
    # x, both in the combined sample) / N + 0.5 (Brunner & Munzel 2000).
    p_relative = float((combined_rank_y.mean() - combined_rank_x.mean()) / (n1 + n2) + 0.5)

    # scipy's t-distribution p-value uses this Satterthwaite-type df
    # internally (Brunner & Munzel 2000) but doesn't expose it on the
    # result; replicated here from scipy's own source so TestResult.df is
    # populated (confirmed against the R fixture, which reports the same
    # df to ~1e-10 relative -- brunnermunzel::brunnermunzel.test computes
    # it via compiled Fortran, not inspectable, but the value matches).
    temp_x = combined_rank_x - rank_x - combined_rank_x.mean() + rank_x.mean()
    temp_y = combined_rank_y - rank_y - combined_rank_y.mean() + rank_y.mean()
    s_x = float(np.sum(temp_x**2)) / (n1 - 1)
    s_y = float(np.sum(temp_y**2)) / (n2 - 1)
    df_numer = (n1 * s_x + n2 * s_y) ** 2
    df_denom = (n1 * s_x) ** 2 / (n1 - 1) + (n2 * s_y) ** 2 / (n2 - 1)
    df_bm = df_numer / df_denom

    return TestResult(
        fact_id=f"brunner_munzel.{value_col}",
        function="brunner_munzel",
        estimand=f"relative effect of '{value_col}' between '{g1}' and '{g2}'",
        statistic=float(result.statistic),
        statistic_name="W",
        df=float(df_bm),
        p_value=float(result.pvalue),
        estimate=p_relative,
        effect_size_name="relative_effect",
        n={str(g1): n1, str(g2): n2},
        warnings=warnings,
    )


@register(
    name="permutation_test_2s",
    kind="test",
    stage="hypothesis",
    tags={"resampling", "two_sample", "independent"},
    assumptions={"hard": ["numeric_outcome"], "soft": []},
    estimand="difference in means (independent groups, permutation-based p-value)",
    code_template=(
        "edacore.stattests.two_sample.permutation_test_2s("
        "{df}, group_col={group_col!r}, value_col={value_col!r}, n_resamples={n_resamples}, "
        "random_state={random_state}, nan_policy={nan_policy!r})"
    ),
)
def permutation_test_2s(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    n_resamples: int = 10_000,
    random_state: int = 0,
    nan_policy: NanPolicy = "omit",
) -> TestResult:
    warnings: list[str] = []
    x, y, g1, g2 = _clean_groups(df, group_col, value_col, nan_policy, warnings)
    n1, n2 = len(x), len(y)

    def _mean_diff(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
        result: np.ndarray = np.mean(a, axis=axis) - np.mean(b, axis=axis)
        return result

    n_arr = n_arrangements_two_sample(n1, n2)
    mode: PermMode = "exact" if n_arr <= MAX_EXACT_PERMUTATIONS else "monte_carlo"
    resamples: float | int = np.inf if mode == "exact" else n_resamples

    result = stats.permutation_test(
        (x, y),
        _mean_diff,
        permutation_type="independent",
        n_resamples=resamples,
        random_state=random_state,
        alternative="two-sided",
    )
    warnings.append(f"permutation mode: {mode} ({n_arr} possible arrangements)")

    return TestResult(
        fact_id=f"permutation_test_2s.{value_col}",
        function="permutation_test_2s",
        estimand=f"difference in means of '{value_col}' between '{g1}' and '{g2}'",
        statistic=float(result.statistic),
        statistic_name="mean_diff",
        p_value=float(result.pvalue),
        estimate=float(x.mean() - y.mean()),
        n={str(g1): n1, str(g2): n2},
        warnings=warnings,
    )


@register(
    name="bootstrap_diff",
    kind="test",
    stage="hypothesis",
    tags={"resampling", "two_sample", "independent"},
    assumptions={"hard": ["numeric_outcome"], "soft": []},
    estimand="difference in a summary statistic, with a bootstrap CI (no p-value)",
    code_template=(
        "edacore.stattests.two_sample.bootstrap_diff("
        "{df}, group_col={group_col!r}, value_col={value_col!r}, stat={stat!r}, ci={ci}, "
        "n_boot={n_boot}, random_state={random_state}, nan_policy={nan_policy!r})"
    ),
)
def bootstrap_diff(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    stat: str = "mean",
    ci: float = 0.95,
    n_boot: int = 2000,
    random_state: int = 0,
    nan_policy: NanPolicy = "omit",
) -> TestResult:
    if stat not in ("mean", "median"):
        raise ValueError(f"stat must be 'mean' or 'median', got {stat!r}")
    warnings: list[str] = []
    x, y, g1, g2 = _clean_groups(df, group_col, value_col, nan_policy, warnings)
    stat_fn: Any = np.mean if stat == "mean" else np.median

    def _diff(a: np.ndarray, b: np.ndarray, axis: int = -1) -> np.ndarray:
        result: np.ndarray = stat_fn(a, axis=axis) - stat_fn(b, axis=axis)
        return result

    result = stats.bootstrap(
        (x, y),
        _diff,
        method="BCa",
        confidence_level=ci,
        n_resamples=n_boot,
        random_state=random_state,
    )

    return TestResult(
        fact_id=f"bootstrap_diff.{value_col}",
        function="bootstrap_diff",
        estimand=f"difference in {stat} of '{value_col}' ('{g1}' vs '{g2}', bootstrap CI only)",
        statistic=float(_diff(x, y)),
        statistic_name=f"{stat}_diff",
        estimate=float(_diff(x, y)),
        ci=(float(result.confidence_interval.low), float(result.confidence_interval.high)),
        ci_level=ci,
        n={str(g1): len(x), str(g2): len(y)},
        warnings=warnings,
    )


@register(
    name="ks_two_sample",
    kind="test",
    stage="hypothesis",
    tags={"nonparametric", "two_sample", "independent", "distribution"},
    assumptions={"hard": ["numeric_or_ordinal"], "soft": []},
    estimand="whether two independent samples come from the same distribution",
    code_template=(
        "edacore.stattests.two_sample.ks_two_sample("
        "{df}, group_col={group_col!r}, value_col={value_col!r}, nan_policy={nan_policy!r})"
    ),
)
def ks_two_sample(
    df: pd.DataFrame, group_col: str, value_col: str, nan_policy: NanPolicy = "omit"
) -> TestResult:
    warnings: list[str] = []
    x, y, g1, g2 = _clean_groups(df, group_col, value_col, nan_policy, warnings)

    result = stats.ks_2samp(x, y)

    return TestResult(
        fact_id=f"ks_two_sample.{value_col}",
        function="ks_two_sample",
        estimand=f"distributional equality of '{value_col}' between '{g1}' and '{g2}'",
        statistic=float(result.statistic),
        statistic_name="D",
        p_value=float(result.pvalue),
        n={str(g1): len(x), str(g2): len(y)},
        warnings=warnings,
    )


@register(
    name="paired_t",
    kind="test",
    stage="hypothesis",
    tags={"parametric", "two_sample", "paired"},
    assumptions={"hard": ["numeric_outcome"], "soft": ["normality_of_differences"]},
    estimand="mean difference between paired measurements",
    code_template=(
        "edacore.stattests.two_sample.paired_t("
        "{df}, a={a!r}, b={b!r}, ci={ci}, nan_policy={nan_policy!r})"
    ),
)
def paired_t(
    df: pd.DataFrame, a: str, b: str, ci: float = 0.95, nan_policy: NanPolicy = "omit"
) -> TestResult:
    warnings: list[str] = []
    x, y = clean_paired(df, a, b, nan_policy, warnings)
    diffs = x - y

    result = stats.ttest_1samp(diffs, 0)
    ci_bounds = result.confidence_interval(ci)
    d_z = float(diffs.mean() / diffs.std(ddof=1))

    return TestResult(
        fact_id=f"paired_t.{a}_{b}",
        function="paired_t",
        estimand=f"mean difference between '{a}' and '{b}'",
        statistic=float(result.statistic),
        statistic_name="t",
        df=float(result.df),
        p_value=float(result.pvalue),
        estimate=float(diffs.mean()),
        ci=(float(ci_bounds.low), float(ci_bounds.high)),
        ci_level=ci,
        effect_size=d_z,
        effect_size_name="d_z",
        effect_magnitude=magnitude_label(d_z, "d_z"),
        n={f"{a}-{b}": len(diffs)},
        warnings=warnings,
    )


@register(
    name="wilcoxon_signed_rank",
    kind="test",
    stage="hypothesis",
    tags={"nonparametric", "two_sample", "paired"},
    assumptions={"hard": ["numeric_or_ordinal"], "soft": ["symmetry_of_differences"]},
    estimand="median difference between paired measurements",
    code_template=(
        "edacore.stattests.two_sample.wilcoxon_signed_rank("
        "{df}, a={a!r}, b={b!r}, nan_policy={nan_policy!r})"
    ),
)
def wilcoxon_signed_rank(
    df: pd.DataFrame, a: str, b: str, nan_policy: NanPolicy = "omit"
) -> TestResult:
    warnings: list[str] = []
    x, y = clean_paired(df, a, b, nan_policy, warnings)
    diffs = x - y

    v, p_value, r = _wilcoxon_signed_rank_r_matched(diffs)

    return TestResult(
        fact_id=f"wilcoxon_signed_rank.{a}_{b}",
        function="wilcoxon_signed_rank",
        estimand=f"median difference between '{a}' and '{b}'",
        statistic=v,
        statistic_name="V",
        p_value=p_value,
        estimate=float(np.median(diffs)),
        effect_size=r,
        effect_size_name="rank_biserial",
        effect_magnitude=magnitude_label(r, "rank_biserial"),
        n={f"{a}-{b}": len(diffs)},
        warnings=warnings,
    )


@register(
    name="sign_test_paired",
    kind="test",
    stage="hypothesis",
    tags={"nonparametric", "two_sample", "paired"},
    assumptions={"hard": ["ordinal_or_higher"], "soft": []},
    estimand="median difference between paired measurements (sign only)",
    code_template=(
        "edacore.stattests.two_sample.sign_test_paired("
        "{df}, a={a!r}, b={b!r}, nan_policy={nan_policy!r})"
    ),
)
def sign_test_paired(
    df: pd.DataFrame, a: str, b: str, nan_policy: NanPolicy = "omit"
) -> TestResult:
    warnings: list[str] = []
    x, y = clean_paired(df, a, b, nan_policy, warnings)
    diffs = x - y
    n_pos = int((diffs > 0).sum())
    n_nonzero = int((diffs != 0).sum())
    if n_nonzero < len(diffs):
        warnings.append(f"{len(diffs) - n_nonzero} pair(s) with zero difference excluded (ties)")

    result = stats.binomtest(n_pos, n_nonzero, p=0.5)
    proportion = n_pos / n_nonzero if n_nonzero else float("nan")

    return TestResult(
        fact_id=f"sign_test_paired.{a}_{b}",
        function="sign_test_paired",
        estimand=f"median difference between '{a}' and '{b}' (sign only)",
        statistic=float(n_pos),
        statistic_name="n_positive",
        p_value=float(result.pvalue),
        estimate=proportion,
        effect_size=proportion,
        effect_size_name="proportion",
        n={f"{a}-{b}": n_nonzero},
        warnings=warnings,
    )


@register(
    name="permutation_test_paired",
    kind="test",
    stage="hypothesis",
    tags={"resampling", "two_sample", "paired"},
    assumptions={"hard": ["numeric_outcome"], "soft": []},
    estimand="mean difference between paired measurements (permutation-based p-value)",
    code_template=(
        "edacore.stattests.two_sample.permutation_test_paired("
        "{df}, a={a!r}, b={b!r}, n_resamples={n_resamples}, random_state={random_state}, "
        "nan_policy={nan_policy!r})"
    ),
)
def permutation_test_paired(
    df: pd.DataFrame,
    a: str,
    b: str,
    n_resamples: int = 10_000,
    random_state: int = 0,
    nan_policy: NanPolicy = "omit",
) -> TestResult:
    warnings: list[str] = []
    x, y = clean_paired(df, a, b, nan_policy, warnings)
    n = len(x)

    def _mean_diff(a_arr: np.ndarray, b_arr: np.ndarray, axis: int = -1) -> np.ndarray:
        result: np.ndarray = np.mean(a_arr - b_arr, axis=axis)
        return result

    n_arr = n_arrangements_paired(n)
    mode: PermMode = "exact" if n_arr <= MAX_EXACT_PERMUTATIONS else "monte_carlo"
    resamples: float | int = np.inf if mode == "exact" else n_resamples

    result = stats.permutation_test(
        (x, y),
        _mean_diff,
        permutation_type="samples",
        n_resamples=resamples,
        random_state=random_state,
        alternative="two-sided",
    )
    warnings.append(f"permutation mode: {mode} ({n_arr} possible arrangements)")

    return TestResult(
        fact_id=f"permutation_test_paired.{a}_{b}",
        function="permutation_test_paired",
        estimand=f"mean difference between '{a}' and '{b}'",
        statistic=float(result.statistic),
        statistic_name="mean_diff",
        p_value=float(result.pvalue),
        estimate=float(np.mean(x - y)),
        n={f"{a}-{b}": n},
        warnings=warnings,
    )
