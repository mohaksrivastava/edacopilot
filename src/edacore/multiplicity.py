"""Multiple-testing correction (ARCHITECTURE.md, Section 6.8).

Matches R's stats::p.adjust to float precision for all four methods
(verified against tests/fixtures/r_reference/adjust_pvalues__raw_pvalues.json).
"""

from __future__ import annotations

from typing import Literal

from statsmodels.stats.multitest import multipletests

from edacore.registry import register

AdjustMethod = Literal["holm", "bonferroni", "fdr_bh", "fdr_by"]


@register(
    name="adjust_pvalues",
    kind="effect",
    stage="hypothesis",
    code_template="edacore.multiplicity.adjust_pvalues({pvals!r}, method={method!r})",
    takes_df=False,
)
def adjust_pvalues(pvals: list[float], method: AdjustMethod = "holm") -> list[float]:
    """Adjust a list of p-values for multiple comparisons.

    `method` matches R's stats::p.adjust naming: "holm" (step-down
    Holm-Bonferroni, the default per personas/professor.yaml and
    personas/consultant.yaml), "bonferroni", "fdr_bh" (Benjamini-Hochberg),
    or "fdr_by" (Benjamini-Yekutieli, valid under arbitrary dependence).
    """
    _, adjusted, _, _ = multipletests(pvals, method=method)
    return [float(p) for p in adjusted]
