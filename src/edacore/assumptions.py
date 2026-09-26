"""Assumption checks (ARCHITECTURE.md, Section 6.6) — shared by all stages.

Every function returns an `AssumptionCheck` with a `status` and a
one-sentence `consequence` written for a junior analyst (rule 1).

Test statistics are verified against R (tests/fixtures/r_reference/) to
1e-6 (check_sphericity_mauchly's exact port matches to ~1e-14). Two
p-values are deliberately NOT bit-matched — kept on scipy/statsmodels'
own methods rather than porting R's tables (see M2.1's ARCHITECTURE.md
Section 18 changelog entry for the reasoning):

- check_normality_anderson: scipy's `method="interpolate"` p-value uses a
  different reference table than R's `nortest::ad.test` (statistic matches
  to 1e-10; p-value differs by ~6e-4).
- check_normality_lilliefors: statsmodels' `lilliefors()` p-value uses a
  different approximation than R's `nortest::lillie.test` (statistic
  matches to 1e-13; p-value differs by ~1.1e-2).

check_sample_size, check_independence_design, check_paired_structure, and
check_measurement_level are deterministic (no test statistic), so they
have no R fixture — Section 15.1 only requires R fixtures for computed
statistics.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan, lilliefors
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools import add_constant

from edacore.contracts import AssumptionCheck, CheckStatus, SemanticType
from edacore.multiplicity import adjust_pvalues
from edacore.profiling import infer_semantic_types
from edacore.registry import register

DEFAULT_ALPHA = 0.05


def _anderson_darling_normal(x: np.ndarray) -> tuple[float, float]:
    """(statistic, p_value) for the Anderson-Darling normality test.

    scipy's `method="interpolate"` kwarg (which gives a p-value directly)
    was added after our declared scipy floor (>=1.15 — verified against
    the min-version CI leg, which pins exactly 1.15.0 and doesn't have
    it). Falls back to interpolating scipy's own significance-level /
    critical-value table for older scipy, so this works across the whole
    declared range rather than raising the floor for one convenience
    kwarg.
    """
    try:
        result = stats.anderson(x, dist="norm", method="interpolate")
        return float(result.statistic), float(result.pvalue)
    except TypeError:
        result = stats.anderson(x, dist="norm")
        # critical_values is ascending (e.g. [0.55, 0.62, 0.74, 0.86,
        # 1.02]); significance_level is descending (%) at the SAME
        # indices ([15, 10, 5, 2.5, 1]) -- np.interp needs ascending xp,
        # which critical_values already is, with fp aligned index-for-index.
        p = float(
            np.interp(result.statistic, result.critical_values, result.significance_level / 100)
        )
        return float(result.statistic), min(max(p, 0.0), 1.0)


def _p_status(p_value: float, alpha: float = DEFAULT_ALPHA) -> CheckStatus:
    """p < alpha -> FAIL; alpha <= p < 0.10 -> BORDERLINE; else PASS (Section 6.6)."""
    if p_value < alpha:
        return CheckStatus.FAIL
    if p_value < 0.10:
        return CheckStatus.BORDERLINE
    return CheckStatus.PASS


def _series_by(df: pd.DataFrame, col: str, by: str | None) -> dict[str, np.ndarray]:
    if by is None:
        return {"": df[col].dropna().to_numpy(dtype=float)}
    return {
        str(level): group[col].dropna().to_numpy(dtype=float)
        for level, group in df.groupby(by, observed=True)
    }


# --------------------------------------------------------------------------
# Normality
# --------------------------------------------------------------------------


@register(
    name="check_normality_shapiro",
    kind="check",
    stage="hypothesis",
    code_template="edacore.assumptions.check_normality_shapiro({df}, col={col!r}, by={by!r})",
)
def check_normality_shapiro(
    df: pd.DataFrame, col: str, by: str | None = None
) -> list[AssumptionCheck]:
    """Shapiro-Wilk normality test, one per level of `by` (or one overall)."""
    checks = []
    for level, x in _series_by(df, col, by).items():
        if len(x) > 5000:
            rng = np.random.default_rng(0)
            x = rng.choice(x, size=5000, replace=False)
        stat, p = stats.shapiro(x)
        scope = {"variable": col} | ({"group": level} if by else {})
        checks.append(
            AssumptionCheck(
                fact_id=f"normality.shapiro.{col}" + (f".{by}={level}" if by else ""),
                assumption="normality",
                method="shapiro_wilk",
                scope=scope,
                statistic=float(stat),
                p_value=float(p),
                threshold="p < 0.05 -> fail; 0.05 <= p < 0.10 -> borderline",
                status=_p_status(p),
                consequence=(
                    "If normality fails, parametric tests that assume it may "
                    "mislead; prefer a robust or nonparametric alternative."
                ),
            )
        )
    return checks


@register(
    name="check_normality_dagostino",
    kind="check",
    stage="hypothesis",
    code_template="edacore.assumptions.check_normality_dagostino({df}, col={col!r}, by={by!r})",
)
def check_normality_dagostino(
    df: pd.DataFrame, col: str, by: str | None = None
) -> list[AssumptionCheck]:
    """D'Agostino's skewness test (moments::agostino.test): tests whether
    sample skewness is consistent with a normal distribution. Needs n>=20."""
    checks = []
    for level, x in _series_by(df, col, by).items():
        n = len(x)
        if n < 20:
            scope = {"variable": col} | ({"group": level} if by else {})
            checks.append(
                AssumptionCheck(
                    fact_id=f"normality.dagostino.{col}" + (f".{by}={level}" if by else ""),
                    assumption="normality",
                    method="dagostino_skewness",
                    scope=scope,
                    statistic=None,
                    p_value=None,
                    threshold="requires n >= 20",
                    status=CheckStatus.NOT_APPLICABLE,
                    consequence=(
                        "Too few observations for this test; rely on the other normality checks."
                    ),
                )
            )
            continue
        m = x.mean()
        g1 = (np.sum((x - m) ** 3) / n) / (np.sum((x - m) ** 2) / n) ** 1.5
        y = g1 * np.sqrt((n + 1) * (n + 3) / (6 * (n - 2)))
        beta2 = (
            3 * (n**2 + 27 * n - 70) * (n + 1) * (n + 3) / ((n - 2) * (n + 5) * (n + 7) * (n + 9))
        )
        w2 = -1 + np.sqrt(2 * (beta2 - 1))
        delta = 1 / np.sqrt(np.log(np.sqrt(w2)))
        alpha_ = np.sqrt(2 / (w2 - 1))
        z = delta * np.log(y / alpha_ + np.sqrt((y / alpha_) ** 2 + 1))
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        scope = {"variable": col} | ({"group": level} if by else {})
        checks.append(
            AssumptionCheck(
                fact_id=f"normality.dagostino.{col}" + (f".{by}={level}" if by else ""),
                assumption="normality",
                method="dagostino_skewness",
                scope=scope,
                statistic=float(g1),
                p_value=float(p),
                threshold="p < 0.05 -> fail; 0.05 <= p < 0.10 -> borderline",
                status=_p_status(p),
                consequence=(
                    "If normality fails, parametric tests that assume it may "
                    "mislead; prefer a robust or nonparametric alternative."
                ),
            )
        )
    return checks


@register(
    name="check_normality_anderson",
    kind="check",
    stage="hypothesis",
    code_template="edacore.assumptions.check_normality_anderson({df}, col={col!r}, by={by!r})",
)
def check_normality_anderson(
    df: pd.DataFrame, col: str, by: str | None = None
) -> list[AssumptionCheck]:
    """Anderson-Darling normality test. p-value is scipy's table
    interpolation, not R's nortest table (see module docstring)."""
    checks = []
    for level, x in _series_by(df, col, by).items():
        statistic, p_value = _anderson_darling_normal(x)
        scope = {"variable": col} | ({"group": level} if by else {})
        checks.append(
            AssumptionCheck(
                fact_id=f"normality.anderson.{col}" + (f".{by}={level}" if by else ""),
                assumption="normality",
                method="anderson_darling",
                scope=scope,
                statistic=statistic,
                p_value=p_value,
                threshold="critical value at 5% significance",
                status=_p_status(p_value),
                consequence=(
                    "If normality fails, parametric tests that assume it may "
                    "mislead; prefer a robust or nonparametric alternative."
                ),
            )
        )
    return checks


@register(
    name="check_normality_lilliefors",
    kind="check",
    stage="hypothesis",
    code_template="edacore.assumptions.check_normality_lilliefors({df}, col={col!r}, by={by!r})",
)
def check_normality_lilliefors(
    df: pd.DataFrame, col: str, by: str | None = None
) -> list[AssumptionCheck]:
    """Lilliefors-corrected Kolmogorov-Smirnov normality test
    (statsmodels). p-value approximation differs slightly from R's
    nortest::lillie.test (see module docstring)."""
    checks = []
    for level, x in _series_by(df, col, by).items():
        stat, p = lilliefors(x, dist="norm")
        scope = {"variable": col} | ({"group": level} if by else {})
        checks.append(
            AssumptionCheck(
                fact_id=f"normality.lilliefors.{col}" + (f".{by}={level}" if by else ""),
                assumption="normality",
                method="lilliefors",
                scope=scope,
                statistic=float(stat),
                p_value=float(p),
                threshold="p < 0.05 -> fail; 0.05 <= p < 0.10 -> borderline",
                status=_p_status(float(p)),
                consequence=(
                    "If normality fails, parametric tests that assume it may "
                    "mislead; prefer a robust or nonparametric alternative."
                ),
            )
        )
    return checks


@register(
    name="check_normality_descriptive",
    kind="check",
    stage="hypothesis",
    code_template="edacore.assumptions.check_normality_descriptive({df}, col={col!r}, by={by!r})",
)
def check_normality_descriptive(
    df: pd.DataFrame, col: str, by: str | None = None
) -> list[AssumptionCheck]:
    """Skewness/excess-kurtosis rule of thumb: |skew|<1 and |excess
    kurtosis|<2 -> PASS; <2 / <7 -> BORDERLINE; else FAIL. Always run
    alongside a formal test (Section 6.6)."""
    checks = []
    for level, x in _series_by(df, col, by).items():
        skew = float(stats.skew(x))
        excess_kurt = float(stats.kurtosis(x, fisher=True))
        if abs(skew) < 1 and abs(excess_kurt) < 2:
            status = CheckStatus.PASS
        elif abs(skew) < 2 and abs(excess_kurt) < 7:
            status = CheckStatus.BORDERLINE
        else:
            status = CheckStatus.FAIL
        scope = {"variable": col} | ({"group": level} if by else {})
        checks.append(
            AssumptionCheck(
                fact_id=f"normality.descriptive.{col}" + (f".{by}={level}" if by else ""),
                assumption="normality",
                method="descriptive_skew_kurtosis",
                scope=scope,
                statistic=skew,
                p_value=None,
                threshold="|skew|<1 and |excess kurtosis|<2 -> pass; <2/<7 -> borderline",
                status=status,
                consequence=(
                    "Skew or heavy tails distort the mean and standard-deviation-based "
                    "methods; check alongside a formal normality test."
                ),
            )
        )
    return checks


@register(
    name="qq_correlation",
    kind="check",
    stage="hypothesis",
    code_template="edacore.assumptions.qq_correlation({df}, col={col!r}, by={by!r})",
)
def qq_correlation(df: pd.DataFrame, col: str, by: str | None = None) -> list[AssumptionCheck]:
    """Probability-plot correlation coefficient (Filliben-type r): the
    correlation between sorted data and normal quantiles. Feeds
    viz.qq_plot; also reported as an AssumptionCheck per rule 1."""
    checks = []
    for level, x in _series_by(df, col, by).items():
        n = len(x)
        sorted_x = np.sort(x)
        # R's ppoints(n): a = 3/8 for n<=10 (Filliben), else a = 1/2.
        a = 3 / 8 if n <= 10 else 1 / 2
        theoretical = stats.norm.ppf((np.arange(1, n + 1) - a) / (n + 1 - 2 * a))
        r = float(np.corrcoef(sorted_x, theoretical)[0, 1])
        status = (
            CheckStatus.PASS
            if r >= 0.99
            else CheckStatus.BORDERLINE
            if r >= 0.95
            else CheckStatus.FAIL
        )
        scope = {"variable": col} | ({"group": level} if by else {})
        checks.append(
            AssumptionCheck(
                fact_id=f"normality.qq_correlation.{col}" + (f".{by}={level}" if by else ""),
                assumption="normality",
                method="qq_correlation",
                scope=scope,
                statistic=r,
                p_value=None,
                threshold=(
                    "r>=0.99 -> pass; 0.95<=r<0.99 -> borderline;"
                    " r<0.95 -> fail (informal convention)"
                ),
                status=status,
                consequence=(
                    "A low QQ correlation means the data visibly deviates "
                    "from a straight line on a normal probability plot."
                ),
            )
        )
    return checks


# --------------------------------------------------------------------------
# Homoscedasticity
# --------------------------------------------------------------------------


def _groups(df: pd.DataFrame, col: str, group: str) -> list[np.ndarray]:
    return [g[col].dropna().to_numpy(dtype=float) for _, g in df.groupby(group, observed=True)]


@register(
    name="check_equal_variance_levene",
    kind="check",
    stage="hypothesis",
    code_template=(
        "edacore.assumptions.check_equal_variance_levene({df}, col={col!r}, group={group!r})"
    ),
)
def check_equal_variance_levene(df: pd.DataFrame, col: str, group: str) -> AssumptionCheck:
    """Levene's test, median-centred (Brown-Forsythe) — the default variance check."""
    stat, p = stats.levene(*_groups(df, col, group), center="median")
    return AssumptionCheck(
        fact_id=f"variance.levene.{col}.{group}",
        assumption="equal_variance",
        method="levene_median",
        scope={"variable": col, "group": group},
        statistic=float(stat),
        p_value=float(p),
        threshold="p < 0.05 -> fail; 0.05 <= p < 0.10 -> borderline",
        status=_p_status(float(p)),
        consequence=(
            "Unequal variances bias standard-error estimates in tests "
            "that assume equal spread; prefer a Welch-type correction."
        ),
    )


@register(
    name="check_equal_variance_bartlett",
    kind="check",
    stage="hypothesis",
    code_template=(
        "edacore.assumptions.check_equal_variance_bartlett({df}, col={col!r}, group={group!r})"
    ),
)
def check_equal_variance_bartlett(df: pd.DataFrame, col: str, group: str) -> AssumptionCheck:
    """Bartlett's test. Only valid if normality passes (Section 6.6) — it's
    itself sensitive to non-normality."""
    stat, p = stats.bartlett(*_groups(df, col, group))
    return AssumptionCheck(
        fact_id=f"variance.bartlett.{col}.{group}",
        assumption="equal_variance",
        method="bartlett",
        scope={"variable": col, "group": group},
        statistic=float(stat),
        p_value=float(p),
        threshold="p < 0.05 -> fail; 0.05 <= p < 0.10 -> borderline",
        status=_p_status(float(p)),
        consequence=(
            "Unequal variances bias standard-error estimates; Bartlett's "
            "result is only trustworthy when normality also holds."
        ),
    )


@register(
    name="check_equal_variance_fligner",
    kind="check",
    stage="hypothesis",
    code_template=(
        "edacore.assumptions.check_equal_variance_fligner({df}, col={col!r}, group={group!r})"
    ),
)
def check_equal_variance_fligner(df: pd.DataFrame, col: str, group: str) -> AssumptionCheck:
    """Fligner-Killeen test: nonparametric, robust to non-normality."""
    stat, p = stats.fligner(*_groups(df, col, group))
    return AssumptionCheck(
        fact_id=f"variance.fligner.{col}.{group}",
        assumption="equal_variance",
        method="fligner_killeen",
        scope={"variable": col, "group": group},
        statistic=float(stat),
        p_value=float(p),
        threshold="p < 0.05 -> fail; 0.05 <= p < 0.10 -> borderline",
        status=_p_status(float(p)),
        consequence=(
            "Unequal variances bias standard-error estimates in tests "
            "that assume equal spread; prefer a Welch-type correction."
        ),
    )


@register(
    name="check_variance_ratio",
    kind="check",
    stage="hypothesis",
    code_template="edacore.assumptions.check_variance_ratio({df}, col={col!r}, group={group!r})",
)
def check_variance_ratio(df: pd.DataFrame, col: str, group: str) -> AssumptionCheck:
    """Descriptive complement to the formal tests: max/min group SD ratio."""
    sds = [float(g.std(ddof=1)) for g in _groups(df, col, group)]
    ratio = max(sds) / min(sds)
    return AssumptionCheck(
        fact_id=f"variance.ratio.{col}.{group}",
        assumption="equal_variance",
        method="variance_ratio",
        scope={"variable": col, "group": group},
        statistic=ratio,
        p_value=None,
        threshold="max/min SD ratio < 2 -> pass",
        status=CheckStatus.PASS if ratio < 2 else CheckStatus.FAIL,
        consequence=(
            "A large spread ratio between groups means pooled-variance "
            "methods understate uncertainty for the noisier group."
        ),
    )


# --------------------------------------------------------------------------
# Sample size / design (deterministic, no R fixture)
# --------------------------------------------------------------------------


@register(
    name="check_sample_size",
    kind="check",
    stage="hypothesis",
    code_template=(
        "edacore.assumptions.check_sample_size({df}, col={col!r}, group={group!r}, min_n={min_n})"
    ),
)
def check_sample_size(df: pd.DataFrame, col: str, group: str, min_n: int) -> AssumptionCheck:
    """Every group must have at least `min_n` non-missing observations."""
    counts = {
        str(level): int(g[col].notna().sum()) for level, g in df.groupby(group, observed=True)
    }
    worst = min(counts.values())
    return AssumptionCheck(
        fact_id=f"sample_size.{col}.{group}",
        assumption="adequate_n",
        method="group_count",
        scope={"variable": col, "group": group},
        statistic=float(worst),
        p_value=None,
        threshold=f"every group n >= {min_n}",
        status=CheckStatus.PASS if worst >= min_n else CheckStatus.FAIL,
        consequence=(
            "Too few observations in a group makes the test's "
            "asymptotic assumptions unreliable and its result unstable."
        ),
    )


@register(
    name="check_expected_counts",
    kind="check",
    stage="hypothesis",
    code_template="edacore.assumptions.check_expected_counts({df}, a={a!r}, b={b!r})",
)
def check_expected_counts(df: pd.DataFrame, a: str, b: str) -> AssumptionCheck:
    """Expected cell counts for a chi-square test on columns `a` x `b`:
    all expected >= 1 and at most 20% of cells < 5 -> pass (hard for chi2)."""
    table = pd.crosstab(df[a], df[b]).to_numpy()
    _, _, _, expected = stats.chi2_contingency(table, correction=False)
    min_expected = float(expected.min())
    pct_below_5 = float((expected < 5).mean() * 100)
    passes = min_expected >= 1 and pct_below_5 <= 20
    return AssumptionCheck(
        fact_id=f"expected_counts.{a}.{b}",
        assumption="expected_counts",
        method="chi2_expected",
        scope={"a": a, "b": b},
        statistic=min_expected,
        p_value=None,
        threshold="all expected >= 1 and <= 20% of cells < 5 -> pass",
        status=CheckStatus.PASS if passes else CheckStatus.FAIL,
        consequence=(
            "Low expected cell counts make the chi-square approximation "
            "unreliable; consider Fisher's exact test instead."
        ),
    )


@register(
    name="check_independence_design",
    kind="check",
    stage="hypothesis",
    code_template="edacore.assumptions.check_independence_design({df}, id_col={id_col!r})",
)
def check_independence_design(df: pd.DataFrame, id_col: str | None = None) -> AssumptionCheck:
    """Independence can't be confirmed from data alone (UNTESTABLE) unless
    a given `id_col` already shows repeats, which is direct evidence
    against independent rows."""
    if id_col is not None and df[id_col].duplicated().any():
        n_repeated = int(df[id_col].duplicated().sum())
        return AssumptionCheck(
            fact_id=f"independence.design.{id_col}",
            assumption="independence",
            method="repeated_id_check",
            scope={"id": id_col},
            statistic=float(n_repeated),
            p_value=None,
            threshold="repeated id values -> rows are not independent",
            status=CheckStatus.FAIL,
            consequence=(
                "Repeated IDs mean some rows come from the same subject;"
                " treating them as independent understates uncertainty."
            ),
        )
    return AssumptionCheck(
        fact_id="independence.design" + (f".{id_col}" if id_col else ""),
        assumption="independence",
        method="repeated_id_check",
        scope={"id": id_col} if id_col else {},
        statistic=None,
        p_value=None,
        threshold="no id_col given, or no repeats found",
        status=CheckStatus.UNTESTABLE,
        consequence=(
            "Independence depends on how the data was collected, not just "
            "its values; confirm the design with whoever collected it."
        ),
    )


@register(
    name="check_paired_structure",
    kind="check",
    stage="hypothesis",
    code_template=(
        "edacore.assumptions.check_paired_structure({df}, a={a!r}, b={b!r}, id_col={id_col!r})"
    ),
)
def check_paired_structure(
    df: pd.DataFrame, a: str, b: str, id_col: str | None = None
) -> AssumptionCheck:
    """Checks that columns `a` and `b` really look like matched pairs:
    equal non-missing counts, and (if `id_col` given) no duplicate subject
    IDs, which would mean rows aren't one-pair-per-subject."""
    n_a, n_b = int(df[a].notna().sum()), int(df[b].notna().sum())
    equal_n = n_a == n_b
    id_ok = id_col is None or not df[id_col].duplicated().any()
    passes = equal_n and id_ok
    reasons = []
    if not equal_n:
        reasons.append(f"n({a})={n_a} != n({b})={n_b}")
    if not id_ok:
        reasons.append(f"'{id_col}' has duplicate values")
    return AssumptionCheck(
        fact_id=f"paired_structure.{a}.{b}",
        assumption="pairing",
        method="paired_structure_check",
        scope={"a": a, "b": b} | ({"id": id_col} if id_col else {}),
        statistic=float(n_a),
        p_value=None,
        threshold="equal non-missing n in both columns, and no duplicate subject ids",
        status=CheckStatus.PASS if passes else CheckStatus.FAIL,
        consequence=(
            "Wrong pairing structure is the most common junior-analyst error;"
            " a mismatched or duplicated design invalidates paired tests."
        ),
    )


@register(
    name="check_measurement_level",
    kind="check",
    stage="hypothesis",
    code_template=(
        "edacore.assumptions.check_measurement_level("
        "{df}, col={col!r}, required={required.value!r})"
    ),
)
def check_measurement_level(df: pd.DataFrame, col: str, required: SemanticType) -> AssumptionCheck:
    """A column's inferred semantic type must be compatible with a method's
    required scale (e.g. a method requiring CONTINUOUS rejects ORDINAL).

    `required` accepts a plain string (e.g. "continuous") as well as a
    SemanticType member — the exported code_template renders it as a
    string literal, so this normalizes either form the same way.
    """
    required = SemanticType(required)
    inferred, confidence = infer_semantic_types(df)[col]
    compatible = inferred == required or (
        required == SemanticType.DISCRETE
        and inferred in (SemanticType.CONTINUOUS, SemanticType.DISCRETE)
    )
    return AssumptionCheck(
        # `required` is part of the id: the same column is legitimately
        # checked against different required scales by different candidate
        # methods, and those are different facts, not one fact recomputed.
        fact_id=f"measurement_level.{col}.{required.value}",
        assumption="measurement_level",
        method="semantic_type_match",
        scope={"variable": col, "required": str(required)},
        statistic=confidence,
        p_value=None,
        threshold=f"inferred semantic type must be compatible with '{required}'",
        status=CheckStatus.PASS if compatible else CheckStatus.FAIL,
        # Phrased for the status it actually has. The single unconditional
        # wording read "'income' looks like continuous, not continuous" on
        # every PASS, which is the kind of line that teaches a reader to
        # stop reading diagnostics.
        consequence=(
            f"'{col}' is {inferred.value}, which a method built for {required.value} data can use."
            if compatible
            else (
                f"'{col}' looks like {inferred.value}, not {required.value}; a method "
                f"built for {required.value} data may not mean what it claims here."
            )
        ),
    )


# --------------------------------------------------------------------------
# Repeated measures
# --------------------------------------------------------------------------


def _helmert_contrasts(k: int) -> np.ndarray:
    """An orthonormal (k-1) x k contrast matrix. Mauchly's W is invariant
    to which orthonormal contrast basis is used."""
    m = np.zeros((k - 1, k))
    for i in range(k - 1):
        m[i, : i + 1] = -1.0 / (i + 1)
        m[i, i + 1] = 1.0
        m[i, :] *= np.sqrt((i + 1) / (i + 2))
    normalized: np.ndarray = m / np.linalg.norm(m, axis=1, keepdims=True)
    return normalized


@register(
    name="check_sphericity_mauchly",
    kind="check",
    stage="hypothesis",
    code_template=(
        "edacore.assumptions.check_sphericity_mauchly({df}, "
        "subject={subject!r}, within={within!r}, dv={dv!r})"
    ),
)
def check_sphericity_mauchly(
    df: pd.DataFrame, subject: str, within: str, dv: str
) -> AssumptionCheck:
    """Mauchly's test of sphericity for repeated-measures ANOVA.

    Ports R's stats:::mauchly.test.SSD exactly, including its two-term
    Box-type p-value correction (not just the leading chi-square term):
    p = Pr(chi2_f > z) + w2 * (Pr(chi2_{f+4} > z) - Pr(chi2_f > z)).
    Verified against the R fixture to ~1e-14.
    """
    wide = df.pivot(index=subject, columns=within, values=dv)
    n_subjects, k = wide.shape
    contrasts = _helmert_contrasts(k)
    y = wide.to_numpy() @ contrasts.T
    s = np.cov(y, rowvar=False, ddof=1)
    p, pp = k, k - 1
    w = np.linalg.det(s) / (np.trace(s) / pp) ** pp
    n = n_subjects - 1  # residual df, as in lm(wide ~ 1)
    rho = 1 - (2 * pp**2 + pp + 2) / (6 * pp * n)
    w2_num = (pp + 2) * (pp - 1) * (pp - 2) * (2 * pp**3 + 6 * pp**2 + 3 * p + 2)
    w2 = w2_num / (288 * (n * pp * rho) ** 2)
    z = -n * rho * np.log(w)
    f = pp * (pp + 1) / 2 - 1
    pr1 = float(stats.chi2.sf(z, f))
    pr2 = float(stats.chi2.sf(z, f + 4))
    p_value = pr1 + w2 * (pr2 - pr1)
    return AssumptionCheck(
        fact_id=f"sphericity.mauchly.{dv}.{within}",
        assumption="sphericity",
        method="mauchly",
        scope={"dv": dv, "within": within, "subject": subject},
        statistic=float(w),
        p_value=p_value,
        threshold="p < 0.05 -> fail (apply Greenhouse-Geisser/Huynh-Feldt correction)",
        status=_p_status(p_value),
        consequence=(
            "Violated sphericity inflates the repeated-measures ANOVA's "
            "false-positive rate; apply a Greenhouse-Geisser correction."
        ),
    )


# --------------------------------------------------------------------------
# Regression-based
# --------------------------------------------------------------------------


@register(
    name="check_linearity",
    kind="check",
    stage="hypothesis",
    code_template="edacore.assumptions.check_linearity({df}, x={x!r}, y={y!r})",
)
def check_linearity(df: pd.DataFrame, x: str, y: str) -> AssumptionCheck:
    """RESET test (Ramsey) for functional-form misspecification: adds
    squared/cubed fitted values to the regression and tests their joint
    significance. (R's harvtest/resettest cover the same ground; RESET is
    the one with a clean, well-known Python implementation via statsmodels.)"""
    xv = df[x].to_numpy(dtype=float)
    yv = df[y].to_numpy(dtype=float)
    exog = add_constant(xv)
    fit = OLS(yv, exog).fit()
    fitted = fit.fittedvalues
    exog_aug = np.column_stack([exog, fitted**2, fitted**3])
    fit_aug = OLS(yv, exog_aug).fit()
    f_stat, p_value, _ = fit_aug.compare_f_test(fit)
    return AssumptionCheck(
        fact_id=f"linearity.{x}.{y}",
        assumption="linearity",
        method="reset",
        scope={"x": x, "y": y},
        statistic=float(f_stat),
        p_value=float(p_value),
        threshold="p < 0.05 -> fail",
        status=_p_status(float(p_value)),
        consequence=(
            "A nonlinear relationship read as linear misstates both "
            "the size and the direction of the effect at the extremes."
        ),
    )


@register(
    name="check_monotonicity",
    kind="check",
    stage="hypothesis",
    code_template="edacore.assumptions.check_monotonicity({df}, x={x!r}, y={y!r})",
)
def check_monotonicity(df: pd.DataFrame, x: str, y: str) -> AssumptionCheck:
    """Is the x-y relationship monotone? Section 6.6's "Spearman vs lowess
    shape", measured on the lowess smooth.

    The question this answers is whether Spearman's rho can be *read* as a
    trend. A U-shaped relationship has a rho near zero while being a strong,
    perfectly ordered association; reporting "no monotonic association"
    there is the failure mode that matters.

    The statistic is the reversal fraction of the lowess smooth: of the
    total vertical movement of the fitted curve, how much goes against its
    dominant direction. A monotone curve gives ~0; a symmetric U approaches
    0.5. PASS < 0.05, BORDERLINE < 0.15, FAIL above.

    Until M4 this check took Spearman's own p-value as its verdict, which
    inverted the meaning entirely: a small p means a STRONG monotone
    association, so the most perfectly monotone data was reported as
    failing monotonicity. The M2 test only compared rho and p against R and
    never asserted a status, so the inversion survived (ARCHITECTURE.md
    Section 18, M4).
    """
    clean = df[[x, y]].dropna().sort_values(x)
    xv = clean[x].to_numpy(dtype=float)
    yv = clean[y].to_numpy(dtype=float)
    smoothed = lowess(yv, xv, frac=2 / 3, return_sorted=True)[:, 1]

    steps = np.diff(smoothed)
    up = float(steps[steps > 0].sum())
    down = float(abs(steps[steps < 0].sum()))
    total = up + down
    reversal = 0.0 if total == 0 else min(up, down) / total

    if reversal < 0.05:
        status = CheckStatus.PASS
    elif reversal < 0.15:
        status = CheckStatus.BORDERLINE
    else:
        status = CheckStatus.FAIL

    return AssumptionCheck(
        fact_id=f"monotonicity.{x}.{y}",
        assumption="monotonicity",
        method="lowess_reversal",
        scope={"x": x, "y": y},
        statistic=float(reversal),
        p_value=None,
        threshold=(
            "fraction of the lowess smooth's movement that reverses direction: "
            "< 0.05 -> pass; < 0.15 -> borderline; else fail"
        ),
        status=status,
        consequence=(
            "A non-monotonic relationship (a U shape, say) can leave Spearman's rho near "
            "zero while the two variables are strongly related, so the correlation would "
            "report 'no association' for data that has an obvious one."
        ),
    )


@register(
    name="check_homoscedasticity_bp",
    kind="check",
    stage="hypothesis",
    code_template="edacore.assumptions.check_homoscedasticity_bp({df}, x={x!r}, y={y!r})",
)
def check_homoscedasticity_bp(df: pd.DataFrame, x: str, y: str) -> AssumptionCheck:
    """Breusch-Pagan test for residual heteroscedasticity."""
    xv = df[x].to_numpy(dtype=float)
    yv = df[y].to_numpy(dtype=float)
    exog = add_constant(xv)
    fit = OLS(yv, exog).fit()
    lm_stat, lm_p, _, _ = het_breuschpagan(fit.resid, exog)
    return AssumptionCheck(
        fact_id=f"homoscedasticity.bp.{x}.{y}",
        assumption="residual_variance",
        method="breusch_pagan",
        scope={"x": x, "y": y},
        statistic=float(lm_stat),
        p_value=float(lm_p),
        threshold="p < 0.05 -> fail",
        status=_p_status(float(lm_p)),
        consequence=(
            "Heteroscedastic residuals make the regression's "
            "standard errors wrong, usually too small."
        ),
    )


@register(
    name="check_autocorrelation_dw",
    kind="check",
    stage="hypothesis",
    code_template="edacore.assumptions.check_autocorrelation_dw({df}, resid={resid!r})",
)
def check_autocorrelation_dw(df: pd.DataFrame, resid: str) -> AssumptionCheck:
    """Durbin-Watson statistic (~2 = no autocorrelation, <2 = positive,
    >2 = negative). No standard closed-form p-value; status uses the
    common informal band."""
    dw = float(durbin_watson(df[resid].to_numpy(dtype=float)))
    status = CheckStatus.PASS if 1.5 <= dw <= 2.5 else CheckStatus.FAIL
    return AssumptionCheck(
        fact_id=f"autocorrelation.dw.{resid}",
        assumption="independence_over_time",
        method="durbin_watson",
        scope={"resid": resid},
        statistic=dw,
        p_value=None,
        threshold="1.5 <= DW <= 2.5 -> pass (informal band around 2 = no autocorrelation)",
        status=status,
        consequence=(
            "Autocorrelated residuals mean nearby observations aren't "
            "independent, understating uncertainty in a time-ordered analysis."
        ),
    )


@register(
    name="check_ljung_box",
    kind="check",
    stage="hypothesis",
    code_template="edacore.assumptions.check_ljung_box({df}, col={col!r}, lags={lags})",
)
def check_ljung_box(df: pd.DataFrame, col: str, lags: int) -> AssumptionCheck:
    """Ljung-Box test for autocorrelation up to `lags`."""
    result = acorr_ljungbox(df[col].to_numpy(dtype=float), lags=[lags])
    stat = float(result["lb_stat"].iloc[0])
    p = float(result["lb_pvalue"].iloc[0])
    return AssumptionCheck(
        fact_id=f"autocorrelation.ljung_box.{col}",
        assumption="independence_over_time",
        method="ljung_box",
        scope={"variable": col, "lags": str(lags)},
        statistic=stat,
        p_value=p,
        threshold="p < 0.05 -> fail",
        status=_p_status(p),
        consequence=(
            "Significant autocorrelation means nearby observations aren't "
            "independent, understating uncertainty in a time-ordered analysis."
        ),
    )


@register(
    name="check_multicollinearity_vif",
    kind="check",
    stage="hypothesis",
    code_template="edacore.assumptions.check_multicollinearity_vif({df}, cols={cols!r})",
)
def check_multicollinearity_vif(df: pd.DataFrame, cols: list[str]) -> AssumptionCheck:
    """Variance inflation factor for each column in `cols`; status is the
    worst case (VIF > 5 -> borderline, > 10 -> fail)."""
    exog = add_constant(df[cols].to_numpy(dtype=float))
    vifs = {cols[i]: float(variance_inflation_factor(exog, i + 1)) for i in range(len(cols))}
    worst_col = max(vifs, key=lambda c: vifs[c])
    worst_vif = vifs[worst_col]
    if worst_vif > 10:
        status = CheckStatus.FAIL
    elif worst_vif > 5:
        status = CheckStatus.BORDERLINE
    else:
        status = CheckStatus.PASS
    return AssumptionCheck(
        fact_id=f"multicollinearity.vif.{'_'.join(cols)}",
        assumption="multicollinearity",
        method="vif",
        scope={"cols": ",".join(cols), "worst": worst_col},
        statistic=worst_vif,
        p_value=None,
        threshold="VIF > 5 -> borderline; VIF > 10 -> fail",
        status=status,
        consequence=(
            f"'{worst_col}' is highly collinear with the other predictors "
            f"(VIF={worst_vif:.1f}); its coefficient estimate is unstable."
        ),
    )


# --------------------------------------------------------------------------
# Distribution shape
# --------------------------------------------------------------------------


@register(
    name="check_same_shape",
    kind="check",
    stage="hypothesis",
    code_template="edacore.assumptions.check_same_shape({df}, col={col!r}, group={group!r})",
)
def check_same_shape(df: pd.DataFrame, col: str, group: str) -> AssumptionCheck:
    """KS test on centred/scaled groups: needed to read a Mann-Whitney or
    Kruskal-Wallis result as a difference in medians rather than just
    'stochastically larger'.

    With more than two groups (Kruskal-Wallis also declares this
    assumption) every pair is compared and the p-values are Holm-adjusted
    across the pairs, with the worst adjusted pair deciding. Holm, not the
    raw minimum: over C(k,2) pairs an unadjusted minimum would reject on
    chance alone and hand the user a caveat that means nothing. At k=2
    Holm is the identity, so two-group behaviour is unchanged.
    """
    groups = _groups(df, col, group)
    if len(groups) < 2:
        raise ValueError(f"check_same_shape needs at least 2 groups, got {len(groups)}")

    def _scaled(values: np.ndarray) -> np.ndarray:
        centered: np.ndarray = (values - values.mean()) / values.std(ddof=1)
        return centered

    pairs = [stats.ks_2samp(_scaled(a), _scaled(b)) for a, b in itertools.combinations(groups, 2)]
    raw_p = [float(result.pvalue) for result in pairs]
    adjusted = adjust_pvalues(raw_p, method="holm") if len(raw_p) > 1 else raw_p
    worst = int(np.argmax([-value for value in adjusted]))
    p = float(adjusted[worst])
    stat = float(pairs[worst].statistic)
    n_pairs = f" (worst of {len(pairs)} pairs, Holm-adjusted)" if len(pairs) > 1 else ""
    return AssumptionCheck(
        fact_id=f"same_shape.{col}.{group}",
        assumption="same_distribution_shape",
        method="ks_centered_scaled",
        scope={"variable": col, "group": group},
        statistic=stat,
        p_value=p,
        threshold=f"p < 0.05 -> fail; 0.05 <= p < 0.10 -> borderline{n_pairs}",
        status=_p_status(float(p)),
        consequence=(
            "Different distribution shapes mean a Mann-Whitney result reflects "
            "general stochastic dominance, not specifically a difference in medians."
        ),
    )


# --------------------------------------------------------------------------
# Design, symmetry, influence, and count adequacy (added in M4)
#
# Section 6.6's original table predates the eligibility engine (Section 7),
# which has to resolve EVERY hard/soft assumption name the Section 6.7
# registry uses. These five close the gap; each is named in some Section 6.7
# row's Hard or Soft column but had no check behind it.
# --------------------------------------------------------------------------


@register(
    name="check_expected_counts_gof",
    kind="check",
    stage="hypothesis",
    code_template=(
        "edacore.assumptions.check_expected_counts_gof({df}, col={col!r}, expected={expected!r})"
    ),
)
def check_expected_counts_gof(
    df: pd.DataFrame, col: str, expected: dict[str, float]
) -> AssumptionCheck:
    """`check_expected_counts` for a one-way goodness-of-fit test, where the
    expected counts come from the question's reference proportions rather
    than from a two-way table's margins.

    Same rule (all expected >= 1, at most 20% of cells below 5), because it
    is the same approximation being protected: chi-square's null
    distribution is asymptotic in the cell counts, however those counts
    were arrived at.
    """
    total = sum(expected.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"expected proportions must sum to 1, got {total}")
    observed_counts = df[col].value_counts()
    n = int(sum(int(observed_counts.get(level, 0)) for level in expected))
    expected_counts = np.array([expected[level] * n for level in expected], dtype=float)

    min_expected = float(expected_counts.min())
    pct_below_5 = float((expected_counts < 5).mean() * 100)
    passes = min_expected >= 1 and pct_below_5 <= 20
    return AssumptionCheck(
        fact_id=f"expected_counts.gof.{col}",
        assumption="expected_counts",
        method="chi2_expected_gof",
        scope={"variable": col},
        statistic=min_expected,
        p_value=None,
        threshold="all expected >= 1 and <= 20% of cells < 5 -> pass",
        status=CheckStatus.PASS if passes else CheckStatus.FAIL,
        consequence=(
            "Low expected cell counts make the chi-square approximation unreliable; with "
            "only a few categories an exact multinomial test is the honest alternative."
        ),
    )


@register(
    name="check_design_crossing",
    kind="check",
    stage="hypothesis",
    code_template=(
        "edacore.assumptions.check_design_crossing({df}, group={group!r}, id_col={id_col!r})"
    ),
)
def check_design_crossing(df: pd.DataFrame, group: str, id_col: str) -> AssumptionCheck:
    """Long-format design cross-check: does the same subject id appear under
    more than one level of `group`?

    This is the check behind Section 7.1's rule that a design must never be
    chosen silently. `check_paired_structure` answers the same question for
    WIDE data (one column per condition); this answers it for the long
    layout a `COMPARE_GROUPS` question actually arrives in, and it
    distinguishes the two ways ids can repeat, which mean different things:

    - an id under several group levels -> the groups are RELATED
      (paired/repeated), and any independent-samples test is wrong;
    - an id repeated within one level only -> clustering, which breaks
      independence without making the design paired.

    FAIL means "these rows are not independent", never "so use a paired
    test": which design is right is the user's to confirm (Section 7.1).
    """
    sub = df[[group, id_col]].dropna()
    levels_per_id = sub.groupby(id_col, observed=True)[group].nunique()
    rows_per_id = sub.groupby(id_col, observed=True).size()
    n_crossing = int((levels_per_id > 1).sum())
    n_repeated_within = int(((rows_per_id > 1) & (levels_per_id == 1)).sum())

    if n_crossing > 0:
        status = CheckStatus.FAIL
        threshold = f"{n_crossing} id(s) appear under more than one level of '{group}'"
        consequence = (
            f"The same '{id_col}' appears in more than one '{group}' group, so these are "
            "repeated measurements on the same subjects, not independent samples; an "
            "independent-samples test would treat one subject's two readings as two "
            "unrelated people and overstate how much information the data carries."
        )
    elif n_repeated_within > 0:
        status = CheckStatus.FAIL
        threshold = f"{n_repeated_within} id(s) appear more than once within a single group"
        consequence = (
            f"Some '{id_col}' values repeat within a group, so rows are clustered rather "
            "than independent; this does not make the design paired, but it does mean a "
            "test assuming independent rows understates uncertainty."
        )
    else:
        status = CheckStatus.PASS
        threshold = f"every '{id_col}' appears exactly once, under one level of '{group}'"
        consequence = (
            f"Each '{id_col}' contributes one row under one group, which is what an "
            "independent-samples test assumes."
        )

    return AssumptionCheck(
        fact_id=f"design_crossing.{group}.{id_col}",
        assumption="design_crossing",
        method="id_by_group_crosstab",
        scope={"group": group, "id": id_col},
        statistic=float(n_crossing),
        p_value=None,
        threshold=threshold,
        status=status,
        consequence=consequence,
    )


@register(
    name="check_symmetry",
    kind="check",
    stage="hypothesis",
    code_template="edacore.assumptions.check_symmetry({df}, col={col!r}, by={by!r})",
)
def check_symmetry(df: pd.DataFrame, col: str, by: str | None = None) -> list[AssumptionCheck]:
    """Symmetry of a distribution about its centre, read off the sample
    skewness (`moments::skewness`, already this module's reference for
    `check_normality_descriptive`).

    This is what the Wilcoxon tests actually assume: the signed-rank test
    needs the distribution (of values, or of paired differences) to be
    symmetric for its result to read as a statement about the median.
    Normality is sufficient but far stronger than needed, which is why this
    is a separate check rather than reusing a normality test.

    Rule: |skew| < 0.5 -> PASS, < 1 -> BORDERLINE, else FAIL -- the
    conventional "approximately / moderately / highly skewed" bands. Like
    every threshold in Section 6.8, they are a convention, not a law.
    """
    checks = []
    for level, values in _series_by(df, col, by).items():
        skew = float(stats.skew(values, bias=True))
        magnitude = abs(skew)
        if magnitude < 0.5:
            status = CheckStatus.PASS
        elif magnitude < 1.0:
            status = CheckStatus.BORDERLINE
        else:
            status = CheckStatus.FAIL
        scope = {"variable": col} | ({"group": level} if by else {})
        checks.append(
            AssumptionCheck(
                fact_id=f"symmetry.skewness.{col}" + (f".group={level}" if by else ""),
                assumption="symmetry",
                method="skewness",
                scope=scope,
                statistic=skew,
                p_value=None,
                threshold="|skew| < 0.5 -> pass; < 1 -> borderline; else fail",
                status=status,
                consequence=(
                    "A signed-rank test on a skewed distribution still tests something, "
                    "but it stops being readable as a statement about the median."
                ),
            )
        )
    return checks


@register(
    name="check_influential_outliers",
    kind="check",
    stage="hypothesis",
    code_template="edacore.assumptions.check_influential_outliers({df}, x={x!r}, y={y!r})",
)
def check_influential_outliers(df: pd.DataFrame, x: str, y: str) -> AssumptionCheck:
    """Whether any single point dominates the x-y relationship, via Cook's
    distance from the simple OLS fit (`stats::cooks.distance` in R).

    Section 6.7 lists "no influential outliers" as a soft assumption of
    `pearson`, and it is the one that most often turns a correlation into
    an artefact: one point far out in x can create or destroy an r on its
    own. This measures influence (effect on the fit), not outlyingness
    (distance from the middle) -- a point can be extreme and harmless, or
    unremarkable and decisive. General outlier DETECTION is Section 6.4,
    which lands in M11; this is only the diagnostic Pearson needs.

    Rule: max Cook's D > 1 -> FAIL, > 4/n -> BORDERLINE, else PASS (the
    two conventional cutoffs).
    """
    clean = df[[x, y]].dropna()
    n = len(clean)
    fitted = OLS(clean[y].to_numpy(dtype=float), add_constant(clean[x].to_numpy(dtype=float))).fit()
    max_cooks = float(np.max(fitted.get_influence().cooks_distance[0]))

    # 1.0 and 0.5 are Cook & Weisberg's conventional cutoffs. The other
    # common rule, 4/n, is a screen for which INDIVIDUAL points to look at,
    # not a verdict on the dataset: at any real n some point exceeds it by
    # chance, so using it here would make almost every correlation a caveat
    # and train the user to ignore the warning.
    if max_cooks > 1.0:
        status = CheckStatus.FAIL
    elif max_cooks > 0.5:
        status = CheckStatus.BORDERLINE
    else:
        status = CheckStatus.PASS

    return AssumptionCheck(
        fact_id=f"influential_outliers.cooks_d.{x}.{y}",
        assumption="no_influential_outliers",
        method="cooks_distance",
        scope={"x": x, "y": y},
        statistic=max_cooks,
        p_value=None,
        threshold=(
            f"max Cook's D > 1 -> fail; > 0.5 -> borderline; else pass "
            f"(n={n}; the 4/n screening rule would be {4 / n:.4g})"
        ),
        status=status,
        consequence=(
            "A single influential point can create or erase a correlation on its own; "
            "the coefficient then describes that point rather than the relationship."
        ),
    )


@register(
    name="check_balanced_design",
    kind="check",
    stage="hypothesis",
    code_template=(
        "edacore.assumptions.check_balanced_design({df}, subject={subject!r}, within={within!r})"
    ),
)
def check_balanced_design(df: pd.DataFrame, subject: str, within: str) -> AssumptionCheck:
    """Every subject observed exactly once at every level of `within` --
    the "balanced" hard assumption of `repeated_measures_anova`.

    Deterministic counting, so no test statistic and no R fixture beyond
    the cell counts themselves (same footing as `check_sample_size`).
    """
    counts = pd.crosstab(df[subject], df[within]).to_numpy()
    min_cell, max_cell = int(counts.min()), int(counts.max())
    balanced = min_cell == 1 and max_cell == 1

    if min_cell == 0:
        detail = "some subject is missing at least one condition"
    elif max_cell > 1:
        detail = f"some subject has up to {max_cell} rows for one condition"
    else:
        detail = "every subject appears exactly once per condition"

    return AssumptionCheck(
        fact_id=f"balanced_design.{subject}.{within}",
        assumption="balanced",
        method="subject_by_condition_crosstab",
        scope={"subject": subject, "within": within},
        statistic=float(min_cell),
        p_value=None,
        threshold=f"every subject x condition cell holds exactly 1 row ({detail})",
        status=CheckStatus.PASS if balanced else CheckStatus.FAIL,
        consequence=(
            "A repeated-measures ANOVA partitions each subject's variance across all "
            "conditions; with a condition missing or duplicated for some subjects, that "
            "partition -- and the within-subject error term built from it -- is not defined."
        ),
    )


@register(
    name="check_proportion_counts",
    kind="check",
    stage="hypothesis",
    code_template=(
        "edacore.assumptions.check_proportion_counts("
        "{df}, outcome={outcome!r}, group={group!r}, event={event!r})"
    ),
)
def check_proportion_counts(
    df: pd.DataFrame, outcome: str, group: str, event: object
) -> AssumptionCheck:
    """At least 10 successes AND 10 failures in every group -- the
    normal-approximation condition Section 6.7 lists for
    `two_proportion_z` (n*p >= 10 and n*(1-p) >= 10).

    Deterministic counting, like `check_sample_size`.
    """
    clean = df[[outcome, group]].dropna()
    worst = None
    for _, rows in clean.groupby(group, observed=True):
        successes = int((rows[outcome] == event).sum())
        failures = len(rows) - successes
        smallest = min(successes, failures)
        worst = smallest if worst is None else min(worst, smallest)
    if worst is None:
        raise ValueError(f"'{group}' has no non-missing groups to count")

    return AssumptionCheck(
        fact_id=f"proportion_counts.{outcome}.{group}",
        assumption="np_at_least_10",
        method="min_successes_or_failures",
        scope={"outcome": outcome, "group": group, "event": str(event)},
        statistic=float(worst),
        p_value=None,
        threshold="every group has >= 10 successes and >= 10 failures",
        status=CheckStatus.PASS if worst >= 10 else CheckStatus.FAIL,
        consequence=(
            "With very few events (or non-events) in a group, the normal approximation "
            "behind a z-test for proportions breaks down; an exact test (Fisher) is the "
            "honest alternative."
        ),
    )
