"""Post-hoc pairwise comparisons for k-group tests (ARCHITECTURE.md,
Section 6.7's k-group tables). Returns `PostHocResult` (Section 5) rather
than `TestResult`: these procedures produce multiple pairwise comparisons
per call, which doesn't fit TestResult's one-result shape.

Verified against R (tests/fixtures/r_reference/). Package choices, each
confirmed to reproduce R's default output exactly (ARCHITECTURE.md Section
18's M3 part 2a entry has the full writeup):

- tukey_hsd: statsmodels.stats.multicomp.pairwise_tukeyhsd (matches R's
  stats::TukeyHSD directly; scikit-posthocs' own posthoc_tukey_hsd is a
  thin wrapper around the same statsmodels function but only exposes the
  p-value matrix, not diff/lwr/upr).
- games_howell, dunn_test, nemenyi (Friedman): scikit-posthocs, matched at
  each function's own default p-value adjustment.
- conover (Friedman): scikit-posthocs with p_adjust='single-step' --
  its plain default (p_adjust=None) does NOT match PMCMRplus's default
  output (confirmed: neither raw nor any monotonic p.adjust of
  scikit-posthocs' un-adjusted p-values reproduces R's numbers), but
  'single-step' is a real option in scikit-posthocs' own p_adjust
  parameter and reproduces R's PMCMRplus default exactly.
- permutation, paired-t, McNemar pairwise post-hocs: no R package needed,
  built directly on this project's own already-R-verified
  permutation_test_2s / paired_t / stats.mcnemar, pairwise + Holm.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

from edacore.contracts import PairwiseComparison, PostHocResult
from edacore.registry import register
from edacore.stattests._shared import NanPolicy
from edacore.stattests.k_independent import _clean_k_groups
from edacore.stattests.k_related import _clean_wide
from edacore.stattests.two_sample import paired_t, permutation_test_2s


@register(
    name="tukey_hsd",
    kind="posthoc",
    stage="hypothesis",
    tags={"posthoc", "k_independent"},
    estimand="pairwise mean differences across k independent groups",
    code_template=(
        "edacore.stattests.posthoc.tukey_hsd({df}, outcome={outcome!r}, group={group!r}, "
        "nan_policy={nan_policy!r})"
    ),
)
def tukey_hsd(
    df: pd.DataFrame, outcome: str, group: str, nan_policy: NanPolicy = "omit"
) -> PostHocResult:
    warnings: list[str] = []
    clean = _clean_k_groups(df, outcome, group, nan_policy, warnings)

    result = pairwise_tukeyhsd(clean[outcome].to_numpy(), clean[group].to_numpy())
    # `.group_t`/`.group_c` aren't available on statsmodels 0.14.0 (this
    # project's declared floor); `meandiffs`/`confint`/`pvalues` are
    # documented to follow itertools.combinations(groupsunique, 2) order
    # instead, which is stable across versions. Those arrays are
    # mean(second) - mean(first) for each (first, second) pair, so
    # group_a=second/group_b=first keeps `estimate` = mean(group_a) -
    # mean(group_b), matching R's own "later - earlier" labeling.
    pairs = list(itertools.combinations(result.groupsunique, 2))
    comparisons = [
        PairwiseComparison(
            group_a=str(b),
            group_b=str(a),
            estimate=float(diff),
            p_value=float(p),
            p_adjusted=float(p),
            ci=(float(lo), float(hi)),
        )
        for (a, b), diff, p, lo, hi in zip(
            pairs,
            result.meandiffs,
            result.pvalues,
            result.confint[:, 0],
            result.confint[:, 1],
            strict=True,
        )
    ]
    return PostHocResult(
        fact_id=f"tukey_hsd.{outcome}",
        function="tukey_hsd",
        method="Tukey HSD",
        p_adjust_method="tukey (studentized range)",
        comparisons=comparisons,
        n={str(k): int(v) for k, v in clean.groupby(group, observed=True).size().items()},
        warnings=warnings,
    )


@register(
    name="games_howell",
    kind="posthoc",
    stage="hypothesis",
    tags={"posthoc", "k_independent"},
    estimand="pairwise mean differences across k independent groups, unequal variance",
    code_template=(
        "edacore.stattests.posthoc.games_howell({df}, outcome={outcome!r}, group={group!r}, "
        "nan_policy={nan_policy!r})"
    ),
)
def games_howell(
    df: pd.DataFrame, outcome: str, group: str, nan_policy: NanPolicy = "omit"
) -> PostHocResult:
    warnings: list[str] = []
    clean = _clean_k_groups(df, outcome, group, nan_policy, warnings)
    groups = {
        str(name): g[outcome].to_numpy(dtype=float)
        for name, g in clean.groupby(group, observed=True)
    }

    p_matrix = sp.posthoc_games_howell(clean, val_col=outcome, group_col=group)

    comparisons = []
    for a, b in itertools.combinations(groups.keys(), 2):
        na, nb = len(groups[a]), len(groups[b])
        va, vb = groups[a].var(ddof=1), groups[b].var(ddof=1)
        ma, mb = groups[a].mean(), groups[b].mean()
        se = np.sqrt(va / na + vb / nb)
        t_stat = (ma - mb) / se
        q_stat = t_stat * np.sqrt(2)
        comparisons.append(
            PairwiseComparison(
                group_a=a,
                group_b=b,
                estimate=float(ma - mb),
                statistic=float(abs(q_stat)),
                p_value=float(p_matrix.loc[a, b]),
                p_adjusted=float(p_matrix.loc[a, b]),
            )
        )
    return PostHocResult(
        fact_id=f"games_howell.{outcome}",
        function="games_howell",
        method="Games-Howell",
        p_adjust_method="studentized range (built into the test)",
        comparisons=comparisons,
        n={k: len(v) for k, v in groups.items()},
        warnings=warnings,
    )


@register(
    name="dunn_test",
    kind="posthoc",
    stage="hypothesis",
    tags={"posthoc", "k_independent"},
    estimand="pairwise rank-sum differences across k independent groups",
    code_template=(
        "edacore.stattests.posthoc.dunn_test({df}, outcome={outcome!r}, group={group!r}, "
        "p_adjust={p_adjust!r}, nan_policy={nan_policy!r})"
    ),
)
def dunn_test(
    df: pd.DataFrame,
    outcome: str,
    group: str,
    p_adjust: str = "holm",
    nan_policy: NanPolicy = "omit",
) -> PostHocResult:
    """Matched at FSA::dunnTest's default: two-sided p-values, Holm-adjusted."""
    warnings: list[str] = []
    clean = _clean_k_groups(df, outcome, group, nan_policy, warnings)
    groups = {
        str(name): g[outcome].to_numpy(dtype=float)
        for name, g in clean.groupby(group, observed=True)
    }

    ranks = stats.rankdata(clean[outcome].to_numpy())
    n_total = len(clean)
    ranked = clean.assign(_rank=ranks)
    mean_rank = ranked.groupby(group, observed=True)["_rank"].mean()
    _, counts = np.unique(ranks, return_counts=True)
    tie_correction = 1 - (np.sum(counts**3 - counts)) / (n_total**3 - n_total)

    names = list(groups.keys())
    z_list = []
    p_unadj = []
    for a, b in itertools.combinations(names, 2):
        na, nb = len(groups[a]), len(groups[b])
        se = np.sqrt(tie_correction * n_total * (n_total + 1) / 12 * (1 / na + 1 / nb))
        z = (mean_rank[a] - mean_rank[b]) / se
        z_list.append(z)
        p_unadj.append(2 * stats.norm.sf(abs(z)))

    _, p_adj, _, _ = multipletests(p_unadj, method=p_adjust)

    comparisons = [
        PairwiseComparison(
            group_a=a, group_b=b, statistic=float(z), p_value=float(p0), p_adjusted=float(pa)
        )
        for (a, b), z, p0, pa in zip(
            itertools.combinations(names, 2), z_list, p_unadj, p_adj, strict=True
        )
    ]
    return PostHocResult(
        fact_id=f"dunn_test.{outcome}",
        function="dunn_test",
        method="Dunn",
        p_adjust_method=p_adjust,
        comparisons=comparisons,
        n={k: len(v) for k, v in groups.items()},
        warnings=warnings,
    )


@register(
    name="permutation_posthoc",
    kind="posthoc",
    stage="hypothesis",
    tags={"posthoc", "k_independent", "resampling"},
    estimand="pairwise mean differences across k independent groups, permutation-based",
    code_template=(
        "edacore.stattests.posthoc.permutation_posthoc({df}, outcome={outcome!r}, group={group!r}, "
        "n_resamples={n_resamples}, random_state={random_state}, p_adjust={p_adjust!r}, "
        "nan_policy={nan_policy!r})"
    ),
)
def permutation_posthoc(
    df: pd.DataFrame,
    outcome: str,
    group: str,
    n_resamples: int = 10_000,
    random_state: int = 0,
    p_adjust: str = "holm",
    nan_policy: NanPolicy = "omit",
) -> PostHocResult:
    warnings: list[str] = []
    clean = _clean_k_groups(df, outcome, group, nan_policy, warnings)
    names = sorted(clean[group].astype(str).unique())

    p_unadj = []
    raw: list[tuple[str, str, float]] = []
    for a, b in itertools.combinations(names, 2):
        pair_df = clean[clean[group].astype(str).isin([a, b])]
        result = permutation_test_2s(
            pair_df, group, outcome, n_resamples=n_resamples, random_state=random_state
        )
        assert result.p_value is not None
        p_unadj.append(result.p_value)
        raw.append(
            (a, b, result.estimate if result.estimate is not None else float(result.statistic))
        )
        if any("monte_carlo" in w for w in result.warnings):
            warnings.append(
                f"{a} vs {b}: {[w for w in result.warnings if 'permutation mode' in w][0]}"
            )

    _, p_adj, _, _ = multipletests(p_unadj, method=p_adjust)
    comparisons = [
        PairwiseComparison(
            group_a=a, group_b=b, estimate=float(diff), p_value=float(p0), p_adjusted=float(pa)
        )
        for (a, b, diff), p0, pa in zip(raw, p_unadj, p_adj, strict=True)
    ]
    return PostHocResult(
        fact_id=f"permutation_posthoc.{outcome}",
        function="permutation_posthoc",
        method="pairwise permutation",
        p_adjust_method=p_adjust,
        comparisons=comparisons,
        n={str(k): int(v) for k, v in clean.groupby(group, observed=True).size().items()},
        warnings=warnings,
    )


@register(
    name="paired_posthoc",
    kind="posthoc",
    stage="hypothesis",
    tags={"posthoc", "k_related"},
    estimand="pairwise mean differences across k related (within-subject) conditions",
    code_template=(
        "edacore.stattests.posthoc.paired_posthoc({df}, dv={dv!r}, subject={subject!r}, "
        "within={within!r}, p_adjust={p_adjust!r}, nan_policy={nan_policy!r})"
    ),
)
def paired_posthoc(
    df: pd.DataFrame,
    dv: str,
    subject: str,
    within: str,
    p_adjust: str = "holm",
    nan_policy: NanPolicy = "omit",
) -> PostHocResult:
    warnings: list[str] = []
    wide = _clean_wide(df, dv, subject, within, nan_policy, warnings)
    names = list(wide.columns)

    p_unadj = []
    raw: list[tuple[str, str, float, float]] = []
    for a, b in itertools.combinations(names, 2):
        pair_df = pd.DataFrame({str(a): wide[a], str(b): wide[b]})
        result = paired_t(pair_df, str(a), str(b))
        assert result.p_value is not None
        p_unadj.append(result.p_value)
        raw.append((str(a), str(b), result.estimate or 0.0, float(result.statistic)))

    _, p_adj, _, _ = multipletests(p_unadj, method=p_adjust)
    comparisons = [
        PairwiseComparison(
            group_a=a,
            group_b=b,
            statistic=stat,
            estimate=diff,
            p_value=float(p0),
            p_adjusted=float(pa),
        )
        for (a, b, diff, stat), p0, pa in zip(raw, p_unadj, p_adj, strict=True)
    ]
    return PostHocResult(
        fact_id=f"paired_posthoc.{dv}",
        function="paired_posthoc",
        method="pairwise paired t-test",
        p_adjust_method=p_adjust,
        comparisons=comparisons,
        n={subject: len(wide)},
        warnings=warnings,
    )


@register(
    name="nemenyi_friedman",
    kind="posthoc",
    stage="hypothesis",
    tags={"posthoc", "k_related"},
    estimand="pairwise rank differences across k related (within-subject) conditions",
    code_template=(
        "edacore.stattests.posthoc.nemenyi_friedman({df}, dv={dv!r}, subject={subject!r}, "
        "within={within!r}, nan_policy={nan_policy!r})"
    ),
)
def nemenyi_friedman(
    df: pd.DataFrame, dv: str, subject: str, within: str, nan_policy: NanPolicy = "omit"
) -> PostHocResult:
    warnings: list[str] = []
    wide = _clean_wide(df, dv, subject, within, nan_policy, warnings)
    names = list(wide.columns)

    p_matrix = sp.posthoc_nemenyi_friedman(wide.to_numpy())
    p_matrix.index = names
    p_matrix.columns = names

    comparisons = [
        PairwiseComparison(
            group_a=str(a),
            group_b=str(b),
            p_value=float(p_matrix.loc[a, b]),
            p_adjusted=float(p_matrix.loc[a, b]),
        )
        for a, b in itertools.combinations(names, 2)
    ]
    return PostHocResult(
        fact_id=f"nemenyi_friedman.{dv}",
        function="nemenyi_friedman",
        method="Nemenyi",
        p_adjust_method="single-step (studentized range)",
        comparisons=comparisons,
        n={subject: len(wide)},
        warnings=warnings,
    )


@register(
    name="conover_friedman",
    kind="posthoc",
    stage="hypothesis",
    tags={"posthoc", "k_related"},
    estimand="pairwise rank differences across k related (within-subject) conditions",
    code_template=(
        "edacore.stattests.posthoc.conover_friedman({df}, dv={dv!r}, subject={subject!r}, "
        "within={within!r}, nan_policy={nan_policy!r})"
    ),
)
def conover_friedman(
    df: pd.DataFrame, dv: str, subject: str, within: str, nan_policy: NanPolicy = "omit"
) -> PostHocResult:
    """p_adjust='single-step' matches PMCMRplus::frdAllPairsConoverTest's
    default exactly; scikit-posthocs' own plain default (no adjustment)
    does not (see module docstring)."""
    warnings: list[str] = []
    wide = _clean_wide(df, dv, subject, within, nan_policy, warnings)
    names = list(wide.columns)

    p_matrix = sp.posthoc_conover_friedman(wide.to_numpy(), p_adjust="single-step")
    p_matrix.index = names
    p_matrix.columns = names

    comparisons = [
        PairwiseComparison(
            group_a=str(a),
            group_b=str(b),
            p_value=float(p_matrix.loc[a, b]),
            p_adjusted=float(p_matrix.loc[a, b]),
        )
        for a, b in itertools.combinations(names, 2)
    ]
    return PostHocResult(
        fact_id=f"conover_friedman.{dv}",
        function="conover_friedman",
        method="Conover",
        p_adjust_method="single-step",
        comparisons=comparisons,
        n={subject: len(wide)},
        warnings=warnings,
    )


@register(
    name="mcnemar_posthoc",
    kind="posthoc",
    stage="hypothesis",
    tags={"posthoc", "k_related"},
    estimand="pairwise proportion differences across k related binary conditions",
    code_template=(
        "edacore.stattests.posthoc.mcnemar_posthoc({df}, dv={dv!r}, subject={subject!r}, "
        "within={within!r}, p_adjust={p_adjust!r}, nan_policy={nan_policy!r})"
    ),
)
def mcnemar_posthoc(
    df: pd.DataFrame,
    dv: str,
    subject: str,
    within: str,
    p_adjust: str = "holm",
    nan_policy: NanPolicy = "omit",
) -> PostHocResult:
    warnings: list[str] = []
    wide = _clean_wide(df, dv, subject, within, nan_policy, warnings)
    names = list(wide.columns)

    p_unadj = []
    raw: list[tuple[str, str, float]] = []
    for a, b in itertools.combinations(names, 2):
        table = (
            pd.crosstab(wide[a], wide[b])
            .reindex(index=[0, 1], columns=[0, 1], fill_value=0)
            .to_numpy()
        )
        b01, b10 = table[0, 1], table[1, 0]
        n_disc = b01 + b10
        if b01 == b10:
            # stats::mcnemar.test only applies the continuity correction
            # when the off-diagonal counts differ (its source: `any(x -
            # t(x) != 0)`) -- applying it unconditionally would make an
            # exactly-symmetric table (statistic should be exactly 0)
            # come out nonzero instead, since (|0| - 1)^2 = 1 != 0.
            stat, p_val = 0.0, 1.0
        else:
            stat = float((abs(b01 - b10) - 1) ** 2 / n_disc)
            p_val = float(stats.chi2.sf(stat, 1))
        p_unadj.append(p_val)
        raw.append((str(a), str(b), stat))

    _, p_adj, _, _ = multipletests(p_unadj, method=p_adjust)
    comparisons = [
        PairwiseComparison(
            group_a=a, group_b=b, statistic=stat, p_value=float(p0), p_adjusted=float(pa)
        )
        for (a, b, stat), p0, pa in zip(raw, p_unadj, p_adj, strict=True)
    ]
    return PostHocResult(
        fact_id=f"mcnemar_posthoc.{dv}",
        function="mcnemar_posthoc",
        method="pairwise McNemar",
        p_adjust_method=p_adjust,
        comparisons=comparisons,
        n={subject: len(wide)},
        warnings=warnings,
    )
