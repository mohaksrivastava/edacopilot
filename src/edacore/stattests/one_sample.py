"""One-sample hypothesis tests (ARCHITECTURE.md, Section 6.7).

Verified against R (tests/fixtures/r_reference/), with two documented,
deliberate departures from scipy's defaults to match R's wilcox.test
conventions -- see `_wilcoxon_signed_rank_r_matched`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from edacore.contracts import TestResult
from edacore.effect_sizes import magnitude_label
from edacore.registry import register
from edacore.stattests._shared import NanPolicy, clean_series


def _wilcoxon_signed_rank_r_matched(diffs: np.ndarray) -> tuple[float, float, float]:
    """Signed-rank statistic, p-value, and rank-biserial effect size for a
    Wilcoxon signed-rank test, matching R's `wilcox.test` conventions
    rather than scipy's `wilcoxon` defaults (confirmed against R fixtures;
    ARCHITECTURE.md Section 18's M3 changelog entry has the discrepancy
    writeup):

    - statistic: R's `V` is always the sum of ranks of the *positive*
      differences (W+), not scipy's `min(W+, W-)`.
    - p-value: R's default `exact = NULL` uses the exact permutation
      distribution when there are fewer than 50 nonzero differences and no
      ties among |differences|; otherwise a normal approximation with
      R's default continuity correction (`correct = TRUE`). scipy's own
      'auto' mode selection and default `correction=False` disagree with
      this for some n, giving a different p-value for the same data.

    Zero differences are excluded (both R and scipy do this).
    """
    nonzero = diffs[diffs != 0]
    n = len(nonzero)
    if n == 0:
        return 0.0, 1.0, 0.0

    abs_nonzero = np.abs(nonzero)
    ranks = stats.rankdata(abs_nonzero)
    w_pos = float(ranks[nonzero > 0].sum())
    w_neg = float(ranks[nonzero < 0].sum())
    total = w_pos + w_neg
    r = (w_pos - w_neg) / total if total > 0 else 0.0

    has_ties = len(np.unique(abs_nonzero)) != len(abs_nonzero)
    use_exact = n < 50 and not has_ties
    if use_exact:
        result = stats.wilcoxon(nonzero, mode="exact")
    else:
        result = stats.wilcoxon(nonzero, mode="approx", correction=True)

    return w_pos, float(result.pvalue), r


@register(
    name="one_sample_t",
    kind="test",
    stage="hypothesis",
    tags={"parametric", "one_sample"},
    assumptions={"hard": ["numeric_outcome"], "soft": ["normality_or_large_n"]},
    estimand="mean vs. a reference value",
    code_template=(
        "edacore.stattests.one_sample.one_sample_t("
        "{df}, col={col!r}, mu0={mu0}, ci={ci}, nan_policy={nan_policy!r})"
    ),
)
def one_sample_t(
    df: pd.DataFrame, col: str, mu0: float, ci: float = 0.95, nan_policy: NanPolicy = "omit"
) -> TestResult:
    warnings: list[str] = []
    x = clean_series(df[col], nan_policy, col, warnings)
    n = len(x)

    result = stats.ttest_1samp(x, mu0)
    # scipy's confidence_interval() is the CI for the raw sample mean
    # (it doesn't depend on mu0); shift by -mu0 so it's on the same scale
    # as `estimate` (the mean difference from mu0).
    ci_bounds = result.confidence_interval(ci)
    d = float((x.mean() - mu0) / x.std(ddof=1))

    return TestResult(
        fact_id=f"one_sample_t.{col}",
        function="one_sample_t",
        estimand="mean vs. a reference value",
        statistic=float(result.statistic),
        statistic_name="t",
        df=float(result.df),
        p_value=float(result.pvalue),
        estimate=float(x.mean() - mu0),
        ci=(float(ci_bounds.low - mu0), float(ci_bounds.high - mu0)),
        ci_level=ci,
        effect_size=d,
        effect_size_name="cohens_d",
        effect_magnitude=magnitude_label(d, "cohens_d"),
        n={col: n},
        warnings=warnings,
    )


@register(
    name="wilcoxon_one_sample",
    kind="test",
    stage="hypothesis",
    tags={"nonparametric", "one_sample"},
    assumptions={"hard": ["numeric_or_ordinal"], "soft": ["symmetry"]},
    estimand="median vs. a reference value",
    code_template=(
        "edacore.stattests.one_sample.wilcoxon_one_sample("
        "{df}, col={col!r}, mu0={mu0}, nan_policy={nan_policy!r})"
    ),
)
def wilcoxon_one_sample(
    df: pd.DataFrame, col: str, mu0: float, nan_policy: NanPolicy = "omit"
) -> TestResult:
    warnings: list[str] = []
    x = clean_series(df[col], nan_policy, col, warnings)
    diffs = x - mu0

    v, p_value, r = _wilcoxon_signed_rank_r_matched(diffs)

    return TestResult(
        fact_id=f"wilcoxon_one_sample.{col}",
        function="wilcoxon_one_sample",
        estimand="median vs. a reference value",
        statistic=v,
        statistic_name="V",
        p_value=p_value,
        estimate=float(np.median(diffs)),
        effect_size=r,
        effect_size_name="rank_biserial",
        effect_magnitude=magnitude_label(r, "rank_biserial"),
        n={col: len(x)},
        warnings=warnings,
    )


@register(
    name="sign_test",
    kind="test",
    stage="hypothesis",
    tags={"nonparametric", "one_sample"},
    assumptions={"hard": ["ordinal_or_higher"], "soft": []},
    estimand="median vs. a reference value (sign only)",
    code_template=(
        "edacore.stattests.one_sample.sign_test("
        "{df}, col={col!r}, mu0={mu0}, nan_policy={nan_policy!r})"
    ),
)
def sign_test(df: pd.DataFrame, col: str, mu0: float, nan_policy: NanPolicy = "omit") -> TestResult:
    warnings: list[str] = []
    x = clean_series(df[col], nan_policy, col, warnings)
    diffs = x - mu0
    n_pos = int((diffs > 0).sum())
    n_nonzero = int((diffs != 0).sum())
    if n_nonzero < len(diffs):
        warnings.append(f"{len(diffs) - n_nonzero} value(s) exactly equal to mu0 excluded (ties)")

    result = stats.binomtest(n_pos, n_nonzero, p=0.5)
    proportion = n_pos / n_nonzero if n_nonzero else float("nan")

    return TestResult(
        fact_id=f"sign_test.{col}",
        function="sign_test",
        estimand="median vs. a reference value (sign only)",
        statistic=float(n_pos),
        statistic_name="n_positive",
        p_value=float(result.pvalue),
        estimate=proportion,
        effect_size=proportion,
        effect_size_name="proportion",
        n={col: n_nonzero},
        warnings=warnings,
    )


@register(
    name="binomial_test",
    kind="test",
    stage="hypothesis",
    tags={"exact", "one_sample"},
    assumptions={"hard": ["binary_outcome"], "soft": []},
    estimand="proportion vs. a reference proportion",
    code_template=(
        "edacore.stattests.one_sample.binomial_test("
        "{df}, col={col!r}, p0={p0}, ci={ci}, nan_policy={nan_policy!r})"
    ),
)
def binomial_test(
    df: pd.DataFrame, col: str, p0: float, ci: float = 0.95, nan_policy: NanPolicy = "omit"
) -> TestResult:
    warnings: list[str] = []
    x = clean_series(df[col], nan_policy, col, warnings)
    n = len(x)
    successes = int(x.sum())

    result = stats.binomtest(successes, n, p=p0)
    ci_bounds = result.proportion_ci(confidence_level=ci)
    p_hat = successes / n
    h = 2 * np.arcsin(np.sqrt(p_hat)) - 2 * np.arcsin(np.sqrt(p0))

    return TestResult(
        fact_id=f"binomial_test.{col}",
        function="binomial_test",
        estimand="proportion vs. a reference proportion",
        statistic=float(successes),
        statistic_name="successes",
        p_value=float(result.pvalue),
        estimate=float(p_hat),
        ci=(float(ci_bounds.low), float(ci_bounds.high)),
        ci_level=ci,
        effect_size=float(h),
        effect_size_name="cohens_h",
        effect_magnitude=magnitude_label(float(h), "cohens_h"),
        n={col: n},
        warnings=warnings,
    )


@register(
    name="chi2_goodness_of_fit",
    kind="test",
    stage="hypothesis",
    tags={"categorical", "one_sample"},
    assumptions={"hard": ["categorical_outcome", "expected_counts"], "soft": []},
    estimand="category proportions vs. reference proportions",
    code_template=(
        "edacore.stattests.one_sample.chi2_goodness_of_fit("
        "{df}, col={col!r}, expected={expected!r})"
    ),
)
def chi2_goodness_of_fit(df: pd.DataFrame, col: str, expected: dict[str, float]) -> TestResult:
    """`expected` maps category -> expected proportion (must sum to 1)."""
    total = sum(expected.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"expected proportions must sum to 1, got {total}")
    observed_counts = df[col].value_counts()
    categories = list(expected.keys())
    observed = np.array([float(observed_counts.get(c, 0)) for c in categories])
    n = int(observed.sum())
    expected_counts = np.array([expected[c] * n for c in categories])

    result = stats.chisquare(observed, f_exp=expected_counts)
    w = float(np.sqrt(result.statistic / n))

    return TestResult(
        fact_id=f"chi2_goodness_of_fit.{col}",
        function="chi2_goodness_of_fit",
        estimand="category proportions vs. reference proportions",
        statistic=float(result.statistic),
        statistic_name="chi2",
        df=float(len(categories) - 1),
        p_value=float(result.pvalue),
        effect_size=w,
        effect_size_name="cohens_w",
        effect_magnitude=magnitude_label(w, "cohens_w"),
        n={col: n},
    )


@register(
    name="bootstrap_one_sample",
    kind="test",
    stage="hypothesis",
    tags={"resampling", "one_sample"},
    assumptions={"hard": ["numeric_outcome"], "soft": []},
    estimand="a summary statistic, with a bootstrap CI (no p-value)",
    code_template=(
        "edacore.stattests.one_sample.bootstrap_one_sample("
        "{df}, col={col!r}, stat={stat!r}, ci={ci}, n_boot={n_boot}, "
        "random_state={random_state}, nan_policy={nan_policy!r})"
    ),
)
def bootstrap_one_sample(
    df: pd.DataFrame,
    col: str,
    stat: str = "mean",
    ci: float = 0.95,
    n_boot: int = 2000,
    random_state: int = 0,
    nan_policy: NanPolicy = "omit",
) -> TestResult:
    warnings: list[str] = []
    x = clean_series(df[col], nan_policy, col, warnings)
    stat_fn: Any = np.mean if stat == "mean" else np.median
    if stat not in ("mean", "median"):
        raise ValueError(f"stat must be 'mean' or 'median', got {stat!r}")

    result = stats.bootstrap(
        (x,),
        stat_fn,
        method="BCa",
        confidence_level=ci,
        n_resamples=n_boot,
        random_state=random_state,
    )

    return TestResult(
        fact_id=f"bootstrap_one_sample.{col}",
        function="bootstrap_one_sample",
        estimand=f"{stat} (bootstrap CI only, no hypothesis test)",
        statistic=float(stat_fn(x)),
        statistic_name=stat,
        estimate=float(stat_fn(x)),
        ci=(float(result.confidence_interval.low), float(result.confidence_interval.high)),
        ci_level=ci,
        n={col: len(x)},
        warnings=warnings,
    )
