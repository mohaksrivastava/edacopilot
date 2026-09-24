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
  For r x c tables (3+ rows/cols), scipy's own exact p-value does NOT
  match R's `fisher.test` (confirmed: both are genuinely exact, just
  different conventions for "at least as extreme" in a table with no
  natural ordering) -- used as-is per the maintainer's explicit choice;
  documented here rather than silently reconciled. r x c needs
  scipy>=1.15; at the declared floor (1.13) scipy's fisher_exact is 2x2
  only, so r x c raises a clear error there rather than failing obscurely.
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

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats.contingency import odds_ratio as _scipy_odds_ratio

from edacore.contracts import TestResult
from edacore.effect_sizes import cohens_h, cramers_v, magnitude_label, risk_difference
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


@register(
    name="fisher_exact",
    kind="test",
    stage="hypothesis",
    tags={"categorical", "exact"},
    assumptions={"hard": ["categorical"], "soft": []},
    estimand="association between two categorical variables",
    code_template=(
        "edacore.stattests.categorical.fisher_exact("
        "{df}, a={a!r}, b={b!r}, nan_policy={nan_policy!r})"
    ),
)
def fisher_exact(df: pd.DataFrame, a: str, b: str, nan_policy: NanPolicy = "omit") -> TestResult:
    """For a 2x2 table: reports the conditional MLE odds ratio (matching
    R's fisher.test) with its exact CI. For r x c: p-value only, using
    scipy's own exact convention -- documented not to bit-match R's
    fisher.test for tables larger than 2x2 (module docstring)."""
    warnings: list[str] = []
    clean = _clean_pair(df, a, b, nan_policy, warnings)
    table = pd.crosstab(clean[a], clean[b])
    arr = table.to_numpy()

    try:
        result = stats.fisher_exact(arr)
    except ValueError as exc:
        if arr.shape != (2, 2):
            raise ValueError(
                f"fisher_exact on a {arr.shape[0]}x{arr.shape[1]} table needs scipy>=1.15 "
                "(older scipy, including this project's declared floor of 1.13, supports 2x2 only)"
            ) from exc
        raise
    estimate: float | None = None
    ci: tuple[float, float] | None = None
    effect_size_name: str | None = None
    if arr.shape == (2, 2):
        or_result = _scipy_odds_ratio(arr, kind="conditional")
        estimate = float(or_result.statistic)
        effect_size_name = "odds_ratio_conditional_mle"
        low, high = or_result.confidence_interval(0.95)
        ci = (float(low), float(high))
    else:
        warnings.append(
            "r x c table: p-value uses scipy's own exact convention, which does not "
            "bit-match R's fisher.test for tables larger than 2x2 (see module docstring)"
        )

    return TestResult(
        fact_id=f"fisher_exact.{a}.{b}",
        function="fisher_exact",
        estimand=f"association between '{a}' and '{b}'",
        statistic=float(result.pvalue),
        statistic_name="p_value",
        p_value=float(result.pvalue),
        estimate=estimate,
        ci=ci,
        effect_size=estimate,
        effect_size_name=effect_size_name,
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
        "edacore.stattests.categorical.mcnemar({df}, a={a!r}, b={b!r}, nan_policy={nan_policy!r})"
    ),
)
def mcnemar(df: pd.DataFrame, a: str, b: str, nan_policy: NanPolicy = "omit") -> TestResult:
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

    return TestResult(
        fact_id=f"mcnemar.{a}.{b}",
        function="mcnemar",
        estimand=f"change in '{a}' vs '{b}' (paired binary)",
        statistic=statistic,
        statistic_name="chi2",
        df=1.0,
        p_value=p_value,
        effect_size=odds_ratio,
        effect_size_name="odds_ratio",
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
