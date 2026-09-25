"""Regenerate docs/eligibility_table.md from the live registry and rules.

The table is a review artefact: it is how the maintainer checks that what
the engine actually enforces matches Section 6.7's tables. Hand-writing it
would mean reviewing a document that can drift away from the code the
moment an assumption changes, so it is generated, and
`test_eligibility_table_doc.py` fails if the committed file and a fresh
render disagree.

    python scripts/generate_eligibility_table.py          # rewrite the file
    python scripts/generate_eligibility_table.py --check  # exit 1 if stale
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from edacopilot.eligibility.checks import COCHRAN_SKEW_FACTOR, RESOLVERS
from edacopilot.eligibility.rules import ALL_FAMILIES, PENDING_GOALS, RULES
from edacopilot.eligibility.rules.distribution_fit import TS_STATIONARITY_METHODS
from edacore.registry import registry

DOC_PATH = Path(__file__).resolve().parents[1] / "docs" / "eligibility_table.md"

# One line per assumption name: which registered check decides it, and how.
# Keyed by the name as it appears in a function's `assumptions`.
DECIDED_BY: dict[str, tuple[str, str]] = {
    "continuous": ("check_measurement_level", "semantic type must be continuous"),
    "numeric": ("check_measurement_level", "continuous or discrete"),
    "numeric_outcome": ("check_measurement_level", "continuous or discrete"),
    "numeric_or_ordinal": ("check_measurement_level", "continuous, discrete or ordinal"),
    "ordinal_or_higher": (
        "check_measurement_level",
        "continuous, discrete, ordinal or binary",
    ),
    "binary": ("check_measurement_level", "binary"),
    "binary_outcome": ("check_measurement_level", "binary"),
    "categorical": ("check_measurement_level", "nominal, ordinal or binary"),
    "categorical_outcome": ("check_measurement_level", "nominal, ordinal or binary"),
    "ordinal_exposure": ("check_measurement_level", "ordinal"),
    "paired_binary": ("check_measurement_level", "every paired column is binary"),
    "independent": (
        "confirmed design + check_design_crossing",
        "the user's confirmed design must be `independent`, and no subject id may appear "
        "under more than one group. With no id column, `check_independence_design` is "
        "carried as UNTESTABLE so the candidate shows what it is assuming",
    ),
    "paired": ("confirmed design", "the user's confirmed design must be `paired`"),
    "repeated": ("confirmed design", "the user's confirmed design must be `repeated`"),
    "balanced": (
        "check_balanced_design",
        "every subject appears exactly once at every level of the within factor",
    ),
    "normality": ("check_normality_shapiro + check_normality_descriptive", "per group"),
    "normality_or_large_n": (
        "check_normality_shapiro + check_normality_descriptive",
        f"as `normality`, except that Cochran's rule decides it: the large-sample "
        f"condition is met when n > {COCHRAN_SKEW_FACTOR:.0f}*skew^2 in every group, and "
        f"meeting it waives the normality failure. Cochran's rule is about SKEW only -- it "
        f"says nothing about heavy tails, which a near-symmetric heavy-tailed sample can "
        f"have while satisfying it. The normality results are still reported as evidence",
    ),
    "normality_of_differences": (
        "check_normality_shapiro",
        "on the per-subject differences, pivoted from long format when needed",
    ),
    "normality_within_groups": ("check_normality_shapiro", "split by the binary column"),
    "bivariate_normality": (
        "check_normality_shapiro",
        "on each margin. Marginal normality does not imply joint normality; it is the "
        "standard practical proxy, and only a FAIL is conclusive",
    ),
    "equal_variance": ("check_equal_variance_levene", "median-centred (Brown-Forsythe)"),
    "same_shape": (
        "check_same_shape",
        "KS on centred/scaled groups; with k > 2 every pair, Holm-adjusted, worst pair decides",
    ),
    "symmetry": ("check_symmetry", "\\|skew\\| < 0.5 pass, < 1 borderline"),
    "symmetry_of_differences": ("check_symmetry", "on the per-subject differences"),
    "sphericity": ("check_sphericity_mauchly", "Mauchly's W"),
    "linearity": ("check_linearity", "RESET test"),
    "monotonicity": (
        "check_monotonicity",
        "fraction of the lowess smooth's movement that reverses direction",
    ),
    "no_influential_outliers": (
        "check_influential_outliers",
        "max Cook's D > 1 fail, > 0.5 borderline",
    ),
    "expected_counts": (
        "check_expected_counts + check_expected_counts_gof",
        "all expected >= 1 and <= 20% of cells < 5; from the table's margins for a two-way "
        "question, from the reference proportions for a goodness-of-fit one",
    ),
    "np_at_least_10": ("check_proportion_counts", ">= 10 successes and >= 10 failures per group"),
    "min_n_per_group>=2": ("check_sample_size", "every group n >= 2"),
    "exchangeability": ("-- (UNTESTABLE)", "a property of how the data was collected"),
    "ordered_sequence": ("-- (UNTESTABLE)", "whether the stored row order is the meaningful one"),
}


def _assumption_cell(names: list[str]) -> str:
    return ", ".join(f"`{name}`" for name in names) if names else "--"


def _family_section(family) -> list[str]:  # type: ignore[no-untyped-def]
    lines = [f"### `{family.name}`", ""]
    if family.note:
        lines += [f"> {family.note}", ""]
    lines += [
        "| Method | Hard assumptions | Soft assumptions |",
        "|---|---|---|",
    ]
    for method in family.methods:
        spec = registry.get(method.function)
        lines.append(
            f"| `{method.function}` | {_assumption_cell(spec.assumptions.hard)} "
            f"| {_assumption_cell(spec.assumptions.soft)} |"
        )
    lines.append("")
    return lines


def render() -> str:
    lines: list[str] = [
        "# Eligibility table",
        "",
        "**Generated** by `scripts/generate_eligibility_table.py` from the live function",
        "registry and the Section 7.3 rules. Do not edit by hand:",
        "`tests/unit/edacopilot/test_eligibility_table_doc.py` fails if this file and a",
        "fresh render disagree, so an assumption cannot change without this table changing",
        "with it.",
        "",
        "This is what the eligibility engine actually enforces, per method family. Read it",
        "against ARCHITECTURE.md Section 6.7's own tables: where they differ, one of the two",
        "is wrong.",
        "",
        "## How a status is decided (Section 7.2)",
        "",
        "| Outcome | Rule |",
        "|---|---|",
        "| `INELIGIBLE` | any **hard** assumption's deciding check returns FAIL |",
        "| `CAVEAT` | no hard failure, but a **soft** assumption returns FAIL or BORDERLINE |",
        "| `ELIGIBLE` | neither |",
        "",
        "UNTESTABLE never changes a status. Independence of sampling and exchangeability are",
        "facts about how the data was collected, not about its values (Section 5.2), so they",
        "are surfaced as reasons on every candidate that rests on them and left for the user",
        "to vouch for. A method whose *question parameters* are missing (equivalence bounds, a",
        "reference value, a covariate) is reported under `unavailable` rather than as",
        "INELIGIBLE -- that is a question to ask, not a verdict about the data.",
        "",
        "## Which check decides each assumption",
        "",
        "| Assumption | Decided by | Rule |",
        "|---|---|---|",
    ]
    for name in sorted(DECIDED_BY):
        decided_by, rule = DECIDED_BY[name]
        rendered = (
            decided_by
            if decided_by.startswith("--")
            else " + ".join(f"`{part.strip()}`" for part in decided_by.split("+"))
        )
        lines.append(f"| `{name}` | {rendered} | {rule} |")
    lines.append("")

    lines += [
        "## Method families (Section 7.3)",
        "",
    ]
    for family in ALL_FAMILIES:
        lines += _family_section(family)

    lines += [
        "## Goals with no methods yet",
        "",
        "| Goal | Served by |",
        "|---|---|",
    ]
    for goal, milestone in sorted(PENDING_GOALS.items(), key=lambda kv: kv[0].value):
        lines.append(f"| `{goal.value}` | {milestone} |")
    lines.append(
        f"| `trend` | M12 (Section 6.9): the `ts_stationarity` family will offer "
        f"{', '.join(f'`{name}`' for name in TS_STATIONARITY_METHODS)} |"
    )
    lines += [
        "",
        "Asking for one of these raises `UnsupportedQuestionError` naming the milestone,",
        "rather than reporting perfectly good methods as INELIGIBLE: they are not ineligible,",
        "they do not exist yet.",
        "",
    ]

    goals_with_rules = ", ".join(f"`{goal.value}`" for goal in sorted(RULES, key=lambda g: g.value))
    lines += [
        "## Coverage",
        "",
        f"- Goals with rules: {goals_with_rules}",
        f"- Families: {len(ALL_FAMILIES)}",
        f"- Distinct assumption names in use: {len(RESOLVERS)}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if the file is stale")
    args = parser.parse_args()

    rendered = render()
    if args.check:
        current = DOC_PATH.read_text(encoding="utf-8") if DOC_PATH.exists() else ""
        if current != rendered:
            print(f"{DOC_PATH} is stale; run: python {Path(__file__).name}", file=sys.stderr)
            return 1
        return 0

    DOC_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {DOC_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
