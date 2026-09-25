"""Categorical association hypothesis tests (ARCHITECTURE.md, Section
6.7). Verified against R (tests/fixtures/r_reference/).

- chi2_independence: scipy's `chi2_contingency` already defaults to
  `correction=True` (Yates, applied only at df=1), matching R's
  `chisq.test` default exactly -- no porting needed.
- fisher_exact: for 2x2 tables, uses
  `scipy.stats.contingency.odds_ratio(kind="conditional")` for the odds
  ratio -- this is the conditional MLE R's `fisher.test` reports, a
  genuinely different number from the naive sample odds ratio (ad/bc) and
  from `fisher_exact`'s own returned statistic (the unconditional MLE).
  For r x c tables (3+ rows/cols), scipy's `fisher_exact` is NOT exact:
  its default is an unseeded `MonteCarloMethod` (9999 draws), so its
  p-value changed from call to call. `_fisher_rxc_exact_p` instead
  enumerates every table with the observed margins and sums the
  probabilities of those with log P <= log P(observed) + 3.45254e-7 --
  R's FEXACT tolerance (`tol` in fexact.c, added to the observed path
  length in log space), i.e. R's `fisher.test` convention. Matches R to
  ~1e-12. Above MAX_FISHER_TABLES tables it falls back to a seeded Monte
  Carlo p-value, disclosed in `warnings`.
- g_test: ported DescTools::GTest's exact formula (read from its source),
  matched at its default `correct="none"` (not Williams-corrected).
- two_proportion_z: R's `prop.test` for two samples is, despite its name,
  exactly a Yates-corrected 2x2 chi-square test on [[x1,n1-x1],[x2,n2-x2]]
  (confirmed by reading prop.test's source) -- so its statistic/p-value
  come from the same `chi2_contingency(correction=True)` chi2_independence
  uses; only the CI needs its own formula (also ported from prop.test's
  source, since scipy has no equivalent).
- cochran_armitage_trend: ported DescTools::CochranArmitageTest's exact
  formula from its source; matched at integer scores 1..k for the ordered
  exposure levels (its own default via `scores(x, 1, "table")`).
- mcnemar: continuity-corrected by default per stats::mcnemar.test's
  default (`correct=TRUE`), same formula already verified against R in
  M3 part 2a's mcnemar_posthoc. Odds ratio is the standard b/c
  discordant-pairs ratio -- base R's mcnemar.test doesn't report one at
  all, so there's no R fixture to verify this specific number against;
  it's the standard, unambiguous textbook formula.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.special import gammaln
from scipy.stats.contingency import odds_ratio as _scipy_odds_ratio

from edacore.contracts import TestResult
from edacore.effect_sizes import (
    _clopper_pearson_ci,
    cohens_h,
    cramers_v,
    magnitude_label,
    risk_difference,
)
from edacore.registry import register
from edacore.stattests._shared import NanPolicy


def _clean_pair(
    df: pd.DataFrame, a: str, b: str, nan_policy: NanPolicy, warnings: list[str]
) -> pd.DataFrame:
    sub = df[[a, b]]
    n_before = len(sub)
    clean = sub.dropna()
    n_dropped = n_before - len(clean)
    if n_dropped > 0:
        if nan_policy == "raise":
            raise ValueError(f"{n_dropped} row(s) missing '{a}'/'{b}', nan_policy=raise")
        warnings.append(f"{n_dropped} row(s) dropped: missing '{a}' or '{b}' (nan_policy='omit')")
    return clean


@register(
    name="chi2_independence",
    kind="test",
    stage="hypothesis",
    tags={"categorical"},
    assumptions={"hard": ["categorical", "expected_counts"], "soft": []},
    estimand="association between two categorical variables",
    code_template=(
        "edacore.stattests.categorical.chi2_independence("
        "{df}, a={a!r}, b={b!r}, nan_policy={nan_policy!r})"
    ),
)
def chi2_independence(
    df: pd.DataFrame, a: str, b: str, nan_policy: NanPolicy = "omit"
) -> TestResult:
    warnings: list[str] = []
    clean = _clean_pair(df, a, b, nan_policy, warnings)
    table = pd.crosstab(clean[a], clean[b])

    result = stats.chi2_contingency(table.to_numpy())
    v = cramers_v(clean, a, b)

    return TestResult(
        fact_id=f"chi2_independence.{a}.{b}",
        function="chi2_independence",
        estimand=f"association between '{a}' and '{b}'",
        statistic=float(result.statistic),
        statistic_name="chi2",
        df=float(result.dof),
        p_value=float(result.pvalue),
        effect_size=v["estimate"],
        effect_size_name="cramers_v",
        effect_size_ci=(v["ci_low"], v["ci_high"]),
        effect_magnitude=magnitude_label(v["estimate"], "cramers_v"),
        n={"total": len(clean)},
        warnings=warnings,
    )


# Above this many tables with the observed margins, r x c Fisher falls back
# from full enumeration to a seeded Monte Carlo p-value.
MAX_FISHER_TABLES = 2_000_000
# R's FEXACT tolerance on the observed table's log-probability (fexact.c).
_FEXACT_TOL = 3.45254e-7
# Bounds the (states x column-vectors x rows) boolean array per chunk.
_FISHER_CHUNK_CELLS = 20_000_000


def _compositions(total: int, parts: int) -> np.ndarray:
    """All non-negative integer vectors of length `parts` summing to `total`."""
    bars = np.array(list(combinations(range(total + parts - 1), parts - 1)), dtype=np.int64)
    bars = bars.reshape(-1, parts - 1)
    edges = np.concatenate(
        [np.full((len(bars), 1), -1), bars, np.full((len(bars), 1), total + parts - 1)], axis=1
    )
    return np.diff(edges, axis=1) - 1


def _fisher_rxc_exact_p(table: np.ndarray) -> float | None:
    """Exact r x c Fisher p-value with R's fisher.test convention, or None
    if more than MAX_FISHER_TABLES tables share the observed margins.

    Builds every table column by column (the last column is then fixed by
    the row margins), tracking each partial table's log-probability under
    the multivariate hypergeometric null."""
    t = table if table.shape[0] <= table.shape[1] else table.T
    rows, cols = t.sum(axis=1), t.sum(axis=0)
    log_const = float(gammaln(rows + 1).sum() + gammaln(cols + 1).sum() - gammaln(t.sum() + 1))
    log_p_obs = log_const - float(gammaln(t + 1).sum())

    remaining = rows[None, :].astype(np.int64)
    log_p = np.array([log_const])
    for col_total in cols[:-1]:
        cand = _compositions(int(col_total), len(rows))
        cand_log = gammaln(cand + 1).sum(axis=1)
        chunk = max(1, _FISHER_CHUNK_CELLS // (len(cand) * len(rows)))
        new_remaining, new_log_p, n_states = [], [], 0
        for start in range(0, len(remaining), chunk):
            rem = remaining[start : start + chunk]
            si, ki = np.nonzero((cand[None, :, :] <= rem[:, None, :]).all(axis=2))
            n_states += len(si)
            if n_states > MAX_FISHER_TABLES:
                return None
            new_remaining.append(rem[si] - cand[ki])
            new_log_p.append(log_p[start + si] - cand_log[ki])
        remaining, log_p = np.concatenate(new_remaining), np.concatenate(new_log_p)

    log_p = log_p - gammaln(remaining + 1).sum(axis=1)
    p = float(np.exp(log_p[log_p <= log_p_obs + _FEXACT_TOL]).sum())
    return min(max(p, 0.0), 1.0)


@register(
    name="fisher_exact",
    kind="test",
    stage="hypothesis",
    tags={"categorical", "exact"},
    assumptions={"hard": ["categorical"], "soft": []},
    estimand="association between two categorical variables",
    code_template=(
        "edacore.stattests.categorical.fisher_exact("
        "{df}, a={a!r}, b={b!r}, ci_level={ci_level}, random_state={random_state}, "
        "nan_policy={nan_policy!r})"
    ),
)
def fisher_exact(
    df: pd.DataFrame,
    a: str,
    b: str,
    ci_level: float = 0.95,
    random_state: int = 0,
    nan_policy: NanPolicy = "omit",
) -> TestResult:
    """For a 2x2 table: reports the conditional MLE odds ratio (matching
    R's fisher.test) with its exact CI. For r x c: exact p-value by full
    enumeration, R's fisher.test convention (module docstring);
    `random_state` only matters above MAX_FISHER_TABLES, where a seeded
    Monte Carlo p-value is used instead."""
    warnings: list[str] = []
    clean = _clean_pair(df, a, b, nan_policy, warnings)
    table = pd.crosstab(clean[a], clean[b])
    arr = table.to_numpy()

    estimate: float | None = None
    ci: tuple[float, float] | None = None
    effect_size_name: str | None = None
    if arr.shape == (2, 2):
        p_value = float(stats.fisher_exact(arr).pvalue)
        or_result = _scipy_odds_ratio(arr, kind="conditional")
        estimate = float(or_result.statistic)
        effect_size_name = "odds_ratio_conditional_mle"
        low, high = or_result.confidence_interval(ci_level)
        ci = (float(low), float(high))
    elif min(arr.shape) < 2:
        p_value = 1.0  # only one table has these margins
    else:
        exact_p = _fisher_rxc_exact_p(arr)
        if exact_p is not None:
            p_value = exact_p
            warnings.append(
                f"{arr.shape[0]}x{arr.shape[1]} table: exact p-value by full enumeration "
                "(R's fisher.test convention); no odds ratio for tables larger than 2x2"
            )
        else:
            mc = stats.MonteCarloMethod(n_resamples=9999, rng=np.random.default_rng(random_state))
            p_value = float(stats.fisher_exact(arr, method=mc).pvalue)
            warnings.append(
                f"{arr.shape[0]}x{arr.shape[1]} table: more than {MAX_FISHER_TABLES:,} tables "
                "share these margins, so the p-value is Monte Carlo (9999 draws, "
                f"random_state={random_state}), not exact"
            )

    return TestResult(
        fact_id=f"fisher_exact.{a}.{b}",
        function="fisher_exact",
        estimand=f"association between '{a}' and '{b}'",
        statistic=p_value,
        statistic_name="p_value",
        p_value=p_value,
        estimate=estimate,
        ci=ci,
        ci_level=ci_level,
        effect_size=estimate,
        effect_size_name=effect_size_name,
        # The conditional-MLE odds ratio IS the effect size, so it shares
        # the estimate's exact interval (2x2 only; an r x c Fisher test has
        # no odds ratio -- see docs/m3_effect_size_audit.md).
        effect_size_ci=ci,
        n={"total": len(clean)},
        warnings=warnings,
    )


@register(
    name="g_test",
    kind="test",
    stage="hypothesis",
    tags={"categorical"},
    assumptions={"hard": ["categorical", "expected_counts"], "soft": []},
    estimand="association between two categorical variables (likelihood-ratio)",
    code_template=(
        "edacore.stattests.categorical.g_test("
        "{df}, a={a!r}, b={b!r}, correct={correct!r}, nan_policy={nan_policy!r})"
    ),
)
def g_test(
    df: pd.DataFrame,
    a: str,
    b: str,
    correct: str = "none",
    nan_policy: NanPolicy = "omit",
) -> TestResult:
    """Ported from DescTools::GTest's exact source formula. `correct`
    matches R's own options: "none" (default) or "williams"."""
    if correct not in ("none", "williams"):
        raise ValueError(f"correct must be 'none' or 'williams', got {correct!r}")
    warnings: list[str] = []
    clean = _clean_pair(df, a, b, nan_policy, warnings)
    table = pd.crosstab(clean[a], clean[b]).to_numpy(dtype=float)
    n = table.sum()
    row_sums = table.sum(axis=1, keepdims=True)
    col_sums = table.sum(axis=0, keepdims=True)
    expected = row_sums * col_sums / n

    nonzero = table > 0
    g = float(2 * np.sum(table[nonzero] * np.log(table[nonzero] / expected[nonzero])))

    q = 1.0
    if correct == "williams":
        row_tot = float(np.sum(1 / row_sums))
        col_tot = float(np.sum(1 / col_sums))
        nrows, ncols = table.shape
        q = 1 + ((n * row_tot - 1) * (n * col_tot - 1)) / (6 * n * (ncols - 1) * (nrows - 1))
    g_stat = g / q

    dof = (table.shape[0] - 1) * (table.shape[1] - 1)
    p_value = float(stats.chi2.sf(g_stat, dof))
    v = cramers_v(clean, a, b)

    return TestResult(
        fact_id=f"g_test.{a}.{b}",
        function="g_test",
        estimand=f"association between '{a}' and '{b}' (likelihood-ratio)",
        statistic=g_stat,
        statistic_name="G",
        df=float(dof),
        p_value=p_value,
        effect_size=v["estimate"],
        effect_size_name="cramers_v",
        effect_size_ci=(v["ci_low"], v["ci_high"]),
        effect_magnitude=magnitude_label(v["estimate"], "cramers_v"),
        n={"total": int(n)},
        warnings=warnings,
    )


@register(
    name="mcnemar",
    kind="test",
    stage="hypothesis",
    tags={"categorical", "paired"},
    assumptions={"hard": ["paired_binary"], "soft": []},
    estimand="change in a paired binary outcome",
    code_template=(
        "edacore.stattests.categorical.mcnemar("
        "{df}, a={a!r}, b={b!r}, ci={ci}, nan_policy={nan_policy!r})"
    ),
)
def mcnemar(
    df: pd.DataFrame, a: str, b: str, ci: float = 0.95, nan_policy: NanPolicy = "omit"
) -> TestResult:
    """The odds ratio b01/b10 is a strictly increasing transform
    p/(1 - p) of the binomial proportion b01/(b01 + b10), so the exact
    (Clopper-Pearson) interval for that proportion maps bound-for-bound
    onto an exact conditional interval for the odds ratio -- no normal
    approximation. base R's mcnemar.test reports no effect size at all,
    so this construction (not an R function's own output) is what the
    fixture records."""
    warnings: list[str] = []
    clean = _clean_pair(df, a, b, nan_policy, warnings)
    table = pd.crosstab(clean[a], clean[b]).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    table_arr = table.to_numpy()
    b01, b10 = int(table_arr[0, 1]), int(table_arr[1, 0])
    n_disc = b01 + b10

    if b01 == b10:
        # stats::mcnemar.test only applies the continuity correction when
        # the off-diagonal counts differ (its source: `any(x - t(x) != 0)`)
        # -- applying it unconditionally would make an exactly-symmetric
        # table (b01 == b10, statistic should be exactly 0) come out
        # nonzero instead, since (|0| - 1)^2 = 1 != 0.
        statistic, p_value = 0.0, 1.0
    else:
        statistic = float((abs(b01 - b10) - 1) ** 2 / n_disc)
        p_value = float(stats.chi2.sf(statistic, 1))
    odds_ratio = float(b01 / b10) if b10 > 0 else float("inf")
    if n_disc > 0:
        p_low, p_high = _clopper_pearson_ci(b01, n_disc, ci)
        or_ci: tuple[float, float] | None = (
            p_low / (1 - p_low) if p_low < 1 else float("inf"),
            p_high / (1 - p_high) if p_high < 1 else float("inf"),
        )
    else:
        or_ci = None
        warnings.append("no discordant pairs: the odds ratio is undefined")

    return TestResult(
        fact_id=f"mcnemar.{a}.{b}",
        function="mcnemar",
        estimand=f"change in '{a}' vs '{b}' (paired binary)",
        statistic=statistic,
        statistic_name="chi2",
        df=1.0,
        p_value=p_value,
        # An odds ratio has no Cohen-style magnitude convention (see
        # effect_sizes.UNLABELLED_MEASURES), so no effect_magnitude.
        effect_size=odds_ratio,
        effect_size_name="odds_ratio",
        effect_size_ci=or_ci,
        ci_level=ci,
        n={"total": len(clean)},
        warnings=warnings,
    )


@register(
    name="two_proportion_z",
    kind="test",
    stage="hypothesis",
    tags={"categorical"},
    assumptions={"hard": ["binary_outcome"], "soft": ["np_at_least_10"]},
    estimand="difference in proportions between two independent groups",
    code_template=(
        "edacore.stattests.categorical.two_proportion_z("
        "{df}, outcome={outcome!r}, group={group!r}, groups={groups!r}, event={event!r}, "
        "ci={ci}, nan_policy={nan_policy!r})"
    ),
)
def two_proportion_z(
    df: pd.DataFrame,
    outcome: str,
    group: str,
    groups: tuple[str, str],
    event: object,
    ci: float = 0.95,
    nan_policy: NanPolicy = "omit",
) -> TestResult:
    """Matches stats::prop.test's default (correct=TRUE): its statistic
    and p-value are, despite the name, exactly a Yates-corrected 2x2
    chi-square test (confirmed from its source) -- the CI is prop.test's
    own continuity-corrected Wald interval, ported directly (no scipy
    equivalent)."""
    warnings: list[str] = []
    clean = _clean_pair(df, outcome, group, nan_policy, warnings)
    g1, g2 = groups
    x1 = int(((clean[group] == g1) & (clean[outcome] == event)).sum())
    n1 = int((clean[group] == g1).sum())
    x2 = int(((clean[group] == g2) & (clean[outcome] == event)).sum())
    n2 = int((clean[group] == g2).sum())

    table = np.array([[x1, n1 - x1], [x2, n2 - x2]], dtype=float)
    result = stats.chi2_contingency(table, correction=True)

    p1, p2 = x1 / n1, x2 / n2
    delta = p1 - p2
    yates = min(0.5, abs(delta) / (1 / n1 + 1 / n2))
    z_crit = stats.norm.ppf((1 + ci) / 2)
    width = z_crit * np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) + yates * (1 / n1 + 1 / n2)
    ci_low, ci_high = max(delta - width, -1.0), min(delta + width, 1.0)

    rd = risk_difference(clean, outcome, group, groups, event, ci)
    h = cohens_h(clean, outcome, group, groups, event, ci)

    return TestResult(
        fact_id=f"two_proportion_z.{outcome}",
        function="two_proportion_z",
        estimand=f"difference in proportion of '{outcome}'=={event!r} between '{g1}' and '{g2}'",
        statistic=float(result.statistic),
        statistic_name="chi2",
        df=float(result.dof),
        p_value=float(result.pvalue),
        estimate=float(delta),
        ci=(float(ci_low), float(ci_high)),
        ci_level=ci,
        effect_size=h["estimate"],
        effect_size_name="cohens_h",
        effect_size_ci=(h["ci_low"], h["ci_high"]),
        effect_magnitude=magnitude_label(h["estimate"], "cohens_h"),
        n={g1: n1, g2: n2},
        warnings=warnings,
        validity_notes=[f"risk_difference = {rd['estimate']:.6g}"],
    )


@register(
    name="cochran_armitage_trend",
    kind="test",
    stage="hypothesis",
    tags={"categorical", "trend"},
    assumptions={"hard": ["binary_outcome", "ordinal_exposure"], "soft": []},
    estimand="linear trend in proportion across ordered exposure levels",
    code_template=(
        "edacore.stattests.categorical.cochran_armitage_trend("
        "{df}, binary={binary!r}, ordinal={ordinal!r}, event={event!r}, nan_policy={nan_policy!r})"
    ),
)
def cochran_armitage_trend(
    df: pd.DataFrame,
    binary: str,
    ordinal: str,
    event: object,
    nan_policy: NanPolicy = "omit",
) -> TestResult:
    """Ported from DescTools::CochranArmitageTest's exact source formula,
    matched at its default integer scores 1..k for the ordered exposure
    levels (in the order they first appear / their categorical order)."""
    warnings: list[str] = []
    clean = _clean_pair(df, binary, ordinal, nan_policy, warnings)
    levels = (
        clean[ordinal].cat.categories.tolist()
        if hasattr(clean[ordinal], "cat") and clean[ordinal].dtype.name == "category"
        else sorted(clean[ordinal].unique())
    )
    k = len(levels)
    scores = np.arange(1, k + 1, dtype=float)

    n_i = np.array([int((clean[ordinal] == lvl).sum()) for lvl in levels], dtype=float)
    x_i = np.array(
        [int(((clean[ordinal] == lvl) & (clean[binary] == event)).sum()) for lvl in levels],
        dtype=float,
    )
    n_total = n_i.sum()
    r_bar = float((n_i * scores).sum() / n_total)
    s2 = float((n_i * (scores - r_bar) ** 2).sum())
    p_dot = float(x_i.sum() / n_total)

    z = float((x_i * (scores - r_bar)).sum() / np.sqrt(p_dot * (1 - p_dot) * s2))
    p_value = float(2 * stats.norm.sf(abs(z)))

    return TestResult(
        fact_id=f"cochran_armitage_trend.{binary}.{ordinal}",
        function="cochran_armitage_trend",
        estimand=(
            f"linear trend in proportion of '{binary}'=={event!r} across levels of '{ordinal}'"
        ),
        statistic=z,
        statistic_name="Z",
        p_value=p_value,
        n={"total": int(n_total)},
        warnings=warnings,
    )
