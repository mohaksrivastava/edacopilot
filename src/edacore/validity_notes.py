"""Deterministic interpretation guards (ARCHITECTURE.md, Section 6.12).

A junior analyst reading a p-value is where most of this project's value
lands or is lost. The nine guards here are the sentences that have to be
attached to a result whether or not anyone thinks to ask for them: that a
non-significant test is not evidence of no effect, that a significant test
on a million rows may be describing nothing, that an association is not a
cause. They are templates filled from numbers `edacore` already computed —
rule 1 holds here as everywhere, and there is no arithmetic in this module
beyond picking the smallest group and summing `TestResult.n`.

Each guard is a function `(result, context) -> str | None`, exactly as
Section 6.12 specifies, and `attach(result, context)` runs all nine in a
fixed order and returns a copy of the result carrying the notes it
produced. The order is the table's own, so two results with the same
problems read the same way.

**Why a context object.** Four of the nine guards can be decided from the
`TestResult` alone. The other five cannot: whether this is the eighth test
of the session, whether the same-shape check failed, whether the outcome
column was imputed, whether rows were dropped in the outlier stage, and
whether this particular comparison was chosen after looking at the data are
all facts about the *session*, not about the result. `edacore` has no
session and must not grow one (Section 3.1: it is usable on its own as a
plain statistics library), so the caller assembles a `ValidityContext` and
passes it in. The agent layer builds one in the HYPOTHESIS stage; a user
calling `edacore` directly gets the four result-only guards from the
default context and can fill in the rest.

**Notes append, they do not replace.** Several `edacore` test functions
already use `TestResult.validity_notes` to carry test-specific facts that
have nowhere else to go — Mauchly's test and the Huynh-Feldt correction on
a repeated-measures ANOVA, the run count on a runs test. `attach` appends
after those and skips any note already present, so calling it twice is
harmless.

Two conventions Section 6.12 leaves open, decided with the maintainer and
recorded in Section 18:

- **"n small"** for `large_effect_small_n` means *the smallest group* has
  fewer than 30 observations, not the total. The note's claim is that the
  CI is wide, and CI width is driven by the smallest cell: a 200-vs-12
  comparison is preliminary whatever its total.
- **`mann_whitney_shape` also covers Kruskal-Wallis**, which declares the
  same `same_distribution_shape` assumption. The misreading it prevents —
  reporting a rank test as a difference in medians — is identical there,
  and `check_same_shape` already handles k > 2 groups.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

from edacore.contracts import AssumptionCheck, CheckStatus, TestResult

DEFAULT_ALPHA: Final = 0.05

# How small the smallest group may be before `large_effect_small_n` fires.
SMALL_N: Final = 30

# Magnitudes that count as "at least medium" for `large_effect_small_n`.
_AT_LEAST_MEDIUM: Final = frozenset({"medium", "large"})

# Tests whose result is an association and nothing more. The `correlation`
# tag covers pearson/spearman/kendall_tau/point_biserial/partial_correlation/
# distance_correlation/mutual_information; these are the contingency-table
# tests, which carry the same caveat and do not share that tag.
_ASSOCIATION_FUNCTIONS: Final = frozenset(
    {
        "chi2_independence",
        "fisher_exact",
        "g_test",
        "cochran_armitage_trend",
    }
)

# Rank-based tests read as "one group tends to be larger" unless the groups
# have the same shape. See the module docstring for why Kruskal-Wallis is
# here alongside Mann-Whitney.
_SHAPE_DEPENDENT_FUNCTIONS: Final = frozenset({"mann_whitney", "kruskal_wallis"})

_SAME_SHAPE_ASSUMPTION: Final = "same_distribution_shape"


@dataclass(frozen=True)
class ValidityContext:
    """Everything a guard needs that the `TestResult` does not carry.

    Every field has a default, so `attach(result)` works on a bare result
    and simply produces fewer notes. A guard whose evidence is absent stays
    silent rather than guessing — an interpretation guard that fired on an
    assumption about the session would be worse than one that did not fire.
    """

    alpha: float = DEFAULT_ALPHA

    # --- from the eligibility engine -------------------------------------
    # The checks that were run for this test. `mann_whitney_shape` reads the
    # same-shape check out of this list.
    checks: Sequence[AssumptionCheck] = ()
    # The QuestionSpec's goal, as a plain string so edacore stays free of
    # any dependency on the agent layer's enums.
    goal: str | None = None
    # Which columns played which role, e.g. {"outcome": "income"}. Used to
    # decide whether imputation or outlier removal touched *this* test.
    variables: Mapping[str, str] = field(default_factory=dict)

    # --- from the test ledger (Section 12.3) -----------------------------
    # The session test count *including* this test. 1 means this is the
    # only one, and `multiple_testing` stays silent.
    test_number: int = 1
    adjust_method: str | None = None
    p_adjusted: float | None = None

    # --- from the session's history --------------------------------------
    # column -> fraction of its values that are imputed (0-1). Filled by
    # the missingness stage (M10).
    imputed_fractions: Mapping[str, float] = field(default_factory=dict)
    # column -> number of rows removed as outliers. Filled by the outlier
    # stage (M11).
    outliers_removed: Mapping[str, int] = field(default_factory=dict)
    # Set when this specific comparison was chosen after looking at the
    # data — the pair of groups with the biggest visual gap, a subgroup
    # picked from a plot. The orchestrator sets it; nothing can infer it.
    chosen_after_exploring: bool = False


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _total_n(result: TestResult) -> int | None:
    if not result.n:
        return None
    return sum(result.n.values())


def _smallest_group_n(result: TestResult) -> int | None:
    """The smallest group's n, or the single n for a one-sample test.

    `TestResult.n` is `{group_label: count}` for a group comparison and
    `{"total": n}` for a one-sample or correlation test, so the minimum is
    the right reading in both cases.
    """
    if not result.n:
        return None
    return min(result.n.values())


def _interval(result: TestResult) -> tuple[float, float] | None:
    """The interval to quote for "the effect could still be this big".

    `result.ci` is the interval on the estimand itself (a difference in
    means, a correlation) and is the one to prefer; a rank test may have no
    such interval but still report one on its effect size, which answers
    the same question in standardised units.
    """
    return result.ci or result.effect_size_ci


def _columns_of_interest(result: TestResult, context: ValidityContext) -> list[str]:
    """The columns this test's conclusion actually rests on.

    Imputing an unrelated column does not weaken this result, so the
    imputation and outlier guards are scoped to the outcome and the
    variables being related — not to every column in the frame.
    """
    roles = ("outcome", "variable", "x", "y")
    return [column for role in roles if (column := context.variables.get(role)) is not None]


def _is_association(result: TestResult, context: ValidityContext) -> bool:
    if context.goal == "association":
        return True
    if result.function in _ASSOCIATION_FUNCTIONS:
        return True
    # Imported here rather than at module scope: `edacore.registry` imports
    # `edacore.codegen`, and a module-level cycle through the registry would
    # make `validity_notes` unimportable before the registry is populated.
    from edacore.registry import registry

    try:
        return "correlation" in registry.get(result.function).tags
    except KeyError:
        return False


def _same_shape_check(context: ValidityContext) -> AssumptionCheck | None:
    for check in context.checks:
        if check.assumption == _SAME_SHAPE_ASSUMPTION:
            return check
    return None


# --------------------------------------------------------------------------
# the nine guards (Section 6.12, in the table's order)
# --------------------------------------------------------------------------


def non_significance(result: TestResult, context: ValidityContext) -> str | None:
    """p >= alpha: absence of evidence is not evidence of absence."""
    if result.p_value is None or result.p_value < context.alpha:
        return None
    head = "Not significant does not mean no effect."
    interval = _interval(result)
    if interval is None:
        return (
            f"{head} No interval was computed here, so the result does not bound how "
            f"large the effect could still be; consider an equivalence test if "
            f"'no difference' is the claim."
        )
    lo, hi = interval
    return (
        f"{head} The CI for the difference ranges from {lo:.3g} to {hi:.3g}; "
        f"consider an equivalence test if 'no difference' is the claim."
    )


def tiny_effect_large_n(result: TestResult, context: ValidityContext) -> str | None:
    """Significant and negligible: the p-value is reporting the sample size."""
    if result.p_value is None or result.p_value >= context.alpha:
        return None
    if result.effect_magnitude != "negligible":
        return None
    n = _total_n(result)
    if n is None:
        return None
    return (
        f"Statistically significant but negligible in size; with n={n}, very small "
        f"differences become significant."
    )


def large_effect_small_n(result: TestResult, context: ValidityContext) -> str | None:
    """A big estimate from a small sample is an imprecise estimate.

    "n small" is the smallest group, not the total — see the module
    docstring.
    """
    if result.effect_magnitude not in _AT_LEAST_MEDIUM:
        return None
    smallest = _smallest_group_n(result)
    if smallest is None or smallest >= SMALL_N:
        return None
    return (
        f"Large estimated effect with a wide CI; treat as preliminary "
        f"(smallest group n={smallest})."
    )


def correlation_not_causation(result: TestResult, context: ValidityContext) -> str | None:
    if not _is_association(result, context):
        return None
    return "Association only; it does not show that one variable causes the other."


def multiple_testing(result: TestResult, context: ValidityContext) -> str | None:
    """More than one test in the session: say which one this is."""
    if context.test_number <= 1:
        return None
    method = context.adjust_method or "holm"
    adjusted = context.p_adjusted if context.p_adjusted is not None else result.p_adjusted
    if adjusted is None:
        return (
            f"This is test #{context.test_number} in this session; p-values should be "
            f"read against that count."
        )
    return (
        f"This is test #{context.test_number} in this session; the adjusted p-value "
        f"({method}) is {adjusted:.4g}."
    )


def mann_whitney_shape(result: TestResult, context: ValidityContext) -> str | None:
    """A rank test on differently-shaped groups is not a median comparison."""
    if result.function not in _SHAPE_DEPENDENT_FUNCTIONS:
        return None
    check = _same_shape_check(context)
    if check is None or check.status is not CheckStatus.FAIL:
        return None
    return (
        "Groups differ in shape, so this tests whether one group tends to have larger "
        "values, not a difference in medians."
    )


def post_selection(result: TestResult, context: ValidityContext) -> str | None:
    """The comparison was picked after looking at the same data."""
    if not context.chosen_after_exploring:
        return None
    return "This comparison was chosen after exploring the data, so its p-value is optimistic."


def imputed_data(result: TestResult, context: ValidityContext) -> str | None:
    """The outcome contains values that were filled in, not observed."""
    if not context.imputed_fractions:
        return None
    relevant = {
        column: fraction
        for column, fraction in context.imputed_fractions.items()
        if column in _columns_of_interest(result, context) and fraction > 0
    }
    if not relevant:
        return None
    worst = max(relevant.values())
    return f"{worst * 100:.0f}% of values are imputed; single imputation understates uncertainty."


def outliers_removed(result: TestResult, context: ValidityContext) -> str | None:
    """Rows dropped in the outlier stage change what this test saw."""
    if not context.outliers_removed:
        return None
    relevant = {
        column: count
        for column, count in context.outliers_removed.items()
        if column in _columns_of_interest(result, context) and count > 0
    }
    if not relevant:
        return None
    total = sum(relevant.values())
    return f"{total} outlier(s) were removed earlier; results may differ with them included."


Guard = Callable[[TestResult, ValidityContext], "str | None"]

# Section 6.12's order, so two results with the same problems read the same.
GUARDS: Final[tuple[tuple[str, Guard], ...]] = (
    ("non_significance", non_significance),
    ("tiny_effect_large_n", tiny_effect_large_n),
    ("large_effect_small_n", large_effect_small_n),
    ("correlation_not_causation", correlation_not_causation),
    ("multiple_testing", multiple_testing),
    ("mann_whitney_shape", mann_whitney_shape),
    ("post_selection", post_selection),
    ("imputed_data", imputed_data),
    ("outliers_removed", outliers_removed),
)

GUARD_NAMES: Final[tuple[str, ...]] = tuple(name for name, _ in GUARDS)


def evaluate(result: TestResult, context: ValidityContext | None = None) -> dict[str, str]:
    """Every guard that fires, keyed by its Section 6.12 name."""
    context = context or ValidityContext()
    fired: dict[str, str] = {}
    for name, guard in GUARDS:
        note = guard(result, context)
        if note:
            fired[name] = note
    return fired


def notes_for(result: TestResult, context: ValidityContext | None = None) -> list[str]:
    """The notes that apply to this result, in Section 6.12's order."""
    return list(evaluate(result, context).values())


def attach(result: TestResult, context: ValidityContext | None = None) -> TestResult:
    """Return a copy of `result` carrying its validity notes.

    Appends to whatever the test function already put there (Mauchly's
    test, a runs count) and skips duplicates, so this is idempotent.
    """
    existing = list(result.validity_notes)
    added = [note for note in notes_for(result, context) if note not in existing]
    if not added:
        return result
    return result.model_copy(update={"validity_notes": [*existing, *added]})
