"""k-related-group hypothesis tests (ARCHITECTURE.md, Section 6.7).

Verified against R (tests/fixtures/r_reference/).

repeated_measures_anova matches afex::aov_ez's *default* output exactly:
the primary statistic/df/p_value are Greenhouse-Geisser-corrected (afex's
own default `correction="GG"`), with the uncorrected values, Huynh-Feldt
correction, and Mauchly's sphericity test carried in `validity_notes` for
transparency. GG/HF epsilon are computed the same way car's
summary.Anova.mlm computes them internally (read from car's source, the
same way check_sphericity_mauchly ported Mauchly's exact formula): the
eigenvalues of the covariance matrix of Helmert-contrast-transformed
scores -- reusing assumptions._helmert_contrasts, the same orthonormal
contrast basis already verified for Mauchly's W.

cochran_q needs no separate implementation: DescTools::CochranQTest
delegates directly to stats::friedman.test (Cochran's Q is exactly
Friedman's test applied to binary data), confirmed by reading
DescTools::CochranQTest's source.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from edacore.assumptions import _helmert_contrasts, check_sphericity_mauchly
from edacore.contracts import AssumptionCheck, TestResult
from edacore.effect_sizes import kendalls_w, magnitude_label, partial_eta_squared_from_f
from edacore.registry import register
from edacore.stattests._shared import NanPolicy


def _validity_notes(
    mauchly: AssumptionCheck,
    df1: int,
    df2: int,
    f_stat: float,
    p_uncorrected: float,
    eps_gg: float,
    eps_hf: float,
    df1_hf: float,
    df2_hf: float,
    p_hf: float,
) -> list[str]:
    sphericity = "holds" if mauchly.p_value is not None and mauchly.p_value >= 0.05 else "violated"
    hf_capped = " (>1, capped at 1 for the correction)" if eps_hf > 1 else ""
    return [
        f"Mauchly's sphericity test: W={mauchly.statistic:.6g}, "
        f"p={mauchly.p_value:.6g} (sphericity {sphericity})",
        f"Uncorrected (sphericity-assuming): F({df1},{df2})={f_stat:.6g}, p={p_uncorrected:.6g}",
        f"Greenhouse-Geisser epsilon={eps_gg:.6g} (used for the primary statistic/df/p_value)",
        f"Huynh-Feldt epsilon={eps_hf:.6g}{hf_capped}: "
        f"F({df1_hf:.6g},{df2_hf:.6g})={f_stat:.6g}, p={p_hf:.6g}",
    ]


def _clean_wide(
    df: pd.DataFrame, dv: str, subject: str, within: str, nan_policy: NanPolicy, warnings: list[str]
) -> pd.DataFrame:
    sub = df[[subject, within, dv]].dropna(subset=[within, subject])
    n_before = len(sub)
    sub = sub.dropna(subset=[dv])
    n_dropped = n_before - len(sub)
    if n_dropped > 0:
        if nan_policy == "raise":
            raise ValueError(f"{n_dropped} row(s) missing '{dv}', nan_policy=raise")
        warnings.append(f"{n_dropped} row(s) dropped: missing '{dv}' (nan_policy='omit')")
    wide = sub.pivot(index=subject, columns=within, values=dv)
    n_before_wide = len(wide)
    wide = wide.dropna()
    n_incomplete = n_before_wide - len(wide)
    if n_incomplete > 0:
        warnings.append(
            f"{n_incomplete} subject(s) dropped: incomplete across all levels of '{within}' "
            "(design must be balanced)"
        )
    return wide


@register(
    name="repeated_measures_anova",
    kind="test",
    stage="hypothesis",
    tags={"parametric", "k_related"},
    assumptions={"hard": ["balanced", "numeric_outcome"], "soft": ["normality", "sphericity"]},
    estimand="mean of the outcome across k related (within-subject) conditions",
    code_template=(
        "edacore.stattests.k_related.repeated_measures_anova("
        "{df}, dv={dv!r}, subject={subject!r}, within={within!r}, ci={ci}, "
        "nan_policy={nan_policy!r})"
    ),
)
def repeated_measures_anova(
    df: pd.DataFrame,
    dv: str,
    subject: str,
    within: str,
    ci: float = 0.95,
    nan_policy: NanPolicy = "omit",
) -> TestResult:
    warnings: list[str] = []
    wide = _clean_wide(df, dv, subject, within, nan_policy, warnings)
    mat = wide.to_numpy(dtype=float)
    n_subjects, k = mat.shape

    grand_mean = mat.mean()
    subject_means = mat.mean(axis=1)
    condition_means = mat.mean(axis=0)
    ss_total = float(np.sum((mat - grand_mean) ** 2))
    ss_subject = float(k * np.sum((subject_means - grand_mean) ** 2))
    ss_condition = float(n_subjects * np.sum((condition_means - grand_mean) ** 2))
    ss_error = ss_total - ss_subject - ss_condition

    df1 = k - 1
    df2 = (n_subjects - 1) * (k - 1)
    f_stat = (ss_condition / df1) / (ss_error / df2)
    pes = ss_condition / (ss_condition + ss_error)
    p_uncorrected = float(stats.f.sf(f_stat, df1, df2))
    # The sphericity correction changes the test's df, not the effect
    # size: effectsize::eta_squared on the afex fit returns exactly
    # F_to_eta2 at the UNCORRECTED df (checked against R).
    pes_ci = partial_eta_squared_from_f(f_stat, df1, df2, ci=ci)

    contrasts = _helmert_contrasts(k)
    y = mat @ contrasts.T
    s = np.cov(y, rowvar=False, ddof=1)
    lam = np.linalg.eigvalsh(s)
    lam = lam[lam > 0]
    pp = k - 1
    eps_gg = float((lam.mean()) ** 2 / (lam**2).mean())
    error_df = n_subjects - 1
    eps_hf = ((error_df + 1) * pp * eps_gg - 2) / (pp * (error_df - pp * eps_gg))
    eps_hf_capped = min(1.0, eps_hf)

    df1_gg, df2_gg = eps_gg * df1, eps_gg * df2
    p_gg = float(stats.f.sf(f_stat, df1_gg, df2_gg))
    df1_hf, df2_hf = eps_hf_capped * df1, eps_hf_capped * df2
    p_hf = float(stats.f.sf(f_stat, df1_hf, df2_hf))

    # Reuse the already-cleaned, balanced `wide` (not the raw df) so
    # Mauchly's test sees exactly the same subjects/conditions as the
    # ANOVA above.
    long = wide.reset_index().melt(id_vars=subject, var_name=within, value_name=dv)
    mauchly = check_sphericity_mauchly(long, subject=subject, within=within, dv=dv)

    return TestResult(
        fact_id=f"repeated_measures_anova.{dv}",
        function="repeated_measures_anova",
        estimand=f"mean of '{dv}' across levels of '{within}' (within-subject)",
        statistic=float(f_stat),
        statistic_name="F",
        df=(float(df1_gg), float(df2_gg)),
        p_value=p_gg,
        effect_size=float(pes),
        effect_size_name="partial_eta_squared",
        effect_size_ci=(pes_ci["ci_low"], pes_ci["ci_high"]),
        effect_magnitude=magnitude_label(float(pes), "partial_eta_squared"),
        n={subject: n_subjects},
        warnings=warnings,
        validity_notes=_validity_notes(
            mauchly, df1, df2, f_stat, p_uncorrected, eps_gg, eps_hf, df1_hf, df2_hf, p_hf
        ),
    )


@register(
    name="friedman",
    kind="test",
    stage="hypothesis",
    tags={"nonparametric", "k_related"},
    assumptions={"hard": ["ordinal_or_higher"], "soft": []},
    estimand="stochastic dominance across k related (within-subject) conditions",
    code_template=(
        "edacore.stattests.k_related.friedman("
        "{df}, dv={dv!r}, subject={subject!r}, within={within!r}, nan_policy={nan_policy!r})"
    ),
)
def friedman(
    df: pd.DataFrame, dv: str, subject: str, within: str, nan_policy: NanPolicy = "omit"
) -> TestResult:
    warnings: list[str] = []
    wide = _clean_wide(df, dv, subject, within, nan_policy, warnings)
    mat = wide.to_numpy(dtype=float)
    n_subjects, k = mat.shape

    result = stats.friedmanchisquare(*[mat[:, j] for j in range(k)])
    long = wide.reset_index().melt(id_vars=subject, var_name=within, value_name=dv)
    w = kendalls_w(long, dv, within, subject)

    return TestResult(
        fact_id=f"friedman.{dv}",
        function="friedman",
        estimand=f"stochastic dominance of '{dv}' across levels of '{within}' (within-subject)",
        statistic=float(result.statistic),
        statistic_name="chi_squared",
        df=float(k - 1),
        p_value=float(result.pvalue),
        effect_size=w["estimate"],
        effect_size_name="kendalls_w",
        effect_size_ci=(w["ci_low"], w["ci_high"]),
        effect_magnitude=magnitude_label(w["estimate"], "kendalls_w"),
        n={subject: n_subjects},
        warnings=warnings,
    )


@register(
    name="cochran_q",
    kind="test",
    stage="hypothesis",
    tags={"nonparametric", "k_related"},
    assumptions={"hard": ["binary_outcome"], "soft": []},
    estimand="proportion of successes across k related (within-subject) binary conditions",
    code_template=(
        "edacore.stattests.k_related.cochran_q("
        "{df}, dv={dv!r}, subject={subject!r}, within={within!r}, nan_policy={nan_policy!r})"
    ),
)
def cochran_q(
    df: pd.DataFrame, dv: str, subject: str, within: str, nan_policy: NanPolicy = "omit"
) -> TestResult:
    warnings: list[str] = []
    wide = _clean_wide(df, dv, subject, within, nan_policy, warnings)
    mat = wide.to_numpy(dtype=float)
    n_subjects, k = mat.shape

    result = stats.friedmanchisquare(*[mat[:, j] for j in range(k)])

    return TestResult(
        fact_id=f"cochran_q.{dv}",
        function="cochran_q",
        estimand=f"proportion of successes for '{dv}' across levels of '{within}' (within-subject)",
        statistic=float(result.statistic),
        statistic_name="Q",
        df=float(k - 1),
        p_value=float(result.pvalue),
        n={subject: n_subjects},
        warnings=warnings,
    )
