"""Factorial hypothesis tests (ARCHITECTURE.md, Section 6.7).

Verified against R (tests/fixtures/r_reference/).

Type III SS trap: Type III sums of squares are only correct with
sum-to-zero contrasts (R: `options(contrasts=c("contr.sum","contr.poly"))`
+ `car::Anova(fit, type=3)`; here: `C(factor, Sum)` in the formula, passed
to `statsmodels.stats.anova.anova_lm(fit, typ=3)`). With the default
treatment contrasts, Type III main-effect p-values are simply wrong
whenever the design is unbalanced -- confirmed directly: on this module's
own unbalanced test fixture, factor_b's Type III p-value is 0.034 with
default (treatment) contrasts vs. 0.0006 with sum contrasts, a
qualitatively different conclusion, not a rounding difference. Type II SS
doesn't have this problem (it's contrast-invariant by construction), so
`typ=2` (this function's default, matching Section 6.7's own signature)
doesn't need the sum-contrast fix; only `typ=3` does.

aligned_rank_transform_anova ports the ART procedure (Wobbrock et al.
2011) exactly, verified against ARTool::art + anova: for a two-way
factorial design (this function's scope -- ARTool supports deeper
factorial nesting than is implemented here), each effect (A, B, A:B) gets
its own "aligned" response (residual-from-the-saturated-model plus that
effect's own estimated contribution, with every other effect's
contribution removed), which is then rank-transformed and tested with an
ordinary Type III F-test, keeping only that effect's own row. Confirmed
to match R to ~1e-9 on all three terms.
"""

from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from edacore.contracts import TestResult
from edacore.registry import register
from edacore.stattests._shared import NanPolicy


def _clean_factorial(
    df: pd.DataFrame, outcome: str, factors: list[str], nan_policy: NanPolicy, warnings: list[str]
) -> pd.DataFrame:
    sub = df[[outcome, *factors]]
    n_before = len(sub)
    clean = sub.dropna()
    n_dropped = n_before - len(clean)
    if n_dropped > 0:
        if nan_policy == "raise":
            raise ValueError(f"{n_dropped} row(s) missing data, nan_policy=raise")
        warnings.append(f"{n_dropped} row(s) dropped: missing data (nan_policy='omit')")
    return clean


@register(
    name="two_way_anova",
    kind="test",
    stage="hypothesis",
    tags={"parametric", "factorial"},
    assumptions={
        "hard": ["numeric_outcome", "independent"],
        "soft": ["normality", "equal_variance"],
    },
    estimand="main effects and interaction of two independent factors",
    code_template=(
        "edacore.stattests.factorial.two_way_anova("
        "{df}, outcome={outcome!r}, factors={factors!r}, typ={typ}, nan_policy={nan_policy!r})"
    ),
)
def two_way_anova(
    df: pd.DataFrame,
    outcome: str,
    factors: list[str],
    typ: int = 2,
    nan_policy: NanPolicy = "omit",
) -> list[TestResult]:
    """Returns one TestResult per term (both main effects and the
    interaction) -- a factorial design has more than one F-test's worth
    of results, unlike every other function in this module."""
    if len(factors) != 2:
        raise ValueError(f"two_way_anova supports exactly 2 factors, got {len(factors)}")
    if typ not in (2, 3):
        raise ValueError(f"typ must be 2 or 3, got {typ}")
    warnings: list[str] = []
    clean = _clean_factorial(df, outcome, factors, nan_policy, warnings)

    a, b = factors
    a_term = f"C({a}, Sum)" if typ == 3 else f"C({a})"
    b_term = f"C({b}, Sum)" if typ == 3 else f"C({b})"
    model = smf.ols(f"{outcome} ~ {a_term} * {b_term}", data=clean).fit()
    aov = sm.stats.anova_lm(model, typ=typ)

    term_labels = {f"{a_term}": a, f"{b_term}": b, f"{a_term}:{b_term}": f"{a}:{b}"}
    cell_sizes = clean.groupby(factors, observed=True).size()
    unbalanced = cell_sizes.nunique() > 1
    validity_notes = (
        [f"unbalanced design (cell sizes: {cell_sizes.to_dict()})"] if unbalanced else []
    )

    ss_type = "III" if typ == 3 else "II"
    results = []
    for row_name, label in term_labels.items():
        row = aov.loc[row_name]
        results.append(
            TestResult(
                fact_id=f"two_way_anova.{outcome}.{label}",
                function="two_way_anova",
                estimand=f"effect of '{label}' on '{outcome}' (Type {ss_type} SS)",
                statistic=float(row["F"]),
                statistic_name="F",
                df=(float(row["df"]), float(aov.loc["Residual", "df"])),
                p_value=float(row["PR(>F)"]),
                n={"total": len(clean)},
                warnings=list(warnings) if label == factors[0] else [],
                validity_notes=validity_notes,
            )
        )
    return results


@register(
    name="aligned_rank_transform_anova",
    kind="test",
    stage="hypothesis",
    tags={"nonparametric", "factorial"},
    assumptions={"hard": ["ordinal_or_higher", "independent"], "soft": []},
    estimand="main effects and interaction of two independent factors (rank-based)",
    code_template=(
        "edacore.stattests.factorial.aligned_rank_transform_anova("
        "{df}, outcome={outcome!r}, factors={factors!r}, nan_policy={nan_policy!r})"
    ),
)
def aligned_rank_transform_anova(
    df: pd.DataFrame, outcome: str, factors: list[str], nan_policy: NanPolicy = "omit"
) -> list[TestResult]:
    """Aligned Rank Transform (Wobbrock et al. 2011), ported exactly from
    ARTool::art + anova. Scope: two factors only (main effects + one
    interaction) -- see module docstring."""
    if len(factors) != 2:
        raise ValueError(
            f"aligned_rank_transform_anova supports exactly 2 factors, got {len(factors)}"
        )
    warnings: list[str] = []
    clean = _clean_factorial(df, outcome, factors, nan_policy, warnings)
    a, b = factors

    grand_mean = clean[outcome].mean()
    a_effect = clean.groupby(a)[outcome].transform("mean") - grand_mean
    b_effect = clean.groupby(b)[outcome].transform("mean") - grand_mean
    cell_mean = clean.groupby([a, b])[outcome].transform("mean")
    ab_effect = cell_mean - grand_mean - a_effect - b_effect
    residual = clean[outcome] - cell_mean

    aligned = {a: residual + a_effect, b: residual + b_effect, f"{a}:{b}": residual + ab_effect}
    row_for_term = {a: f"C({a}, Sum)", b: f"C({b}, Sum)", f"{a}:{b}": f"C({a}, Sum):C({b}, Sum)"}

    work = clean.copy()
    results = []
    for term, aligned_response in aligned.items():
        work["_art_rank"] = stats.rankdata(aligned_response.to_numpy())
        model = smf.ols(f"_art_rank ~ C({a}, Sum) * C({b}, Sum)", data=work).fit()
        aov = sm.stats.anova_lm(model, typ=3)
        row = aov.loc[row_for_term[term]]
        results.append(
            TestResult(
                fact_id=f"aligned_rank_transform_anova.{outcome}.{term}",
                function="aligned_rank_transform_anova",
                estimand=f"effect of '{term}' on '{outcome}' (aligned rank transform)",
                statistic=float(row["F"]),
                statistic_name="F",
                df=(float(row["df"]), float(aov.loc["Residual", "df"])),
                p_value=float(row["PR(>F)"]),
                n={"total": len(clean)},
                warnings=list(warnings) if term == a else [],
            )
        )
    return results
