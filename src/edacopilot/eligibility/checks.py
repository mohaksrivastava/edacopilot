"""Resolving assumption names to actual checks (ARCHITECTURE.md, Section 7.2).

Section 5.4 lets every registered function declare its assumptions as
strings ("normality_or_large_n", "independent", ...). Section 7.2's engine
then has to turn each of those names into `AssumptionCheck` evidence about
*this* dataset. This module is that translation layer, plus the cache
Section 7.2 asks for ("checks, cached per data version").

Two rules shape it:

- **Nothing is invented.** Every resolver delegates to a registered
  `edacore.assumptions` check; this module decides *which* check to run on
  *which* columns, never how to compute one.
- **An assumption with no evidence is UNTESTABLE, not PASS.** Independence
  of sampling, exchangeability and the ordering of a sequence are facts
  about how data was collected (Section 5.2). Reporting them as passing
  because nothing contradicted them would be the exact failure mode this
  tool exists to prevent, so they resolve to UNTESTABLE and surface as a
  reason on every candidate that rests on them.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from edacore.contracts import AssumptionCheck, CheckStatus, SemanticType
from edacore.profiling import infer_semantic_types
from edacore.registry import registry

from .spec import Design, QuestionSpec

# Per-group n at which the CLT is taken to carry a mean-based test even
# though a formal normality test rejects. Above this (with no extreme
# skew), Shapiro-Wilk rejects almost any real sample for reasons that do
# not matter to a t-test's own behaviour.
#
# This threshold is a CONVENTION, not a result -- like Section 6.8's
# effect-size bands. It is set high deliberately: Section 7.4's worked
# example has n = 38/41 with normality FAIL and expects a CAVEAT, so the
# escape must not fire at moderate n. Persona policies (Section 8.1) apply
# their own, stricter rules on top; this is only the floor below which the
# engine will not soften a normality failure at all.
LARGE_N_FOR_CLT = 100
MAX_SKEW_FOR_CLT = 2.0

# A "variable map": canonical role -> column name. Each method family builds
# one from a QuestionSpec (which of `x`/`y` is the binary column, where the
# subject id lives, and so on), so resolvers never have to re-derive it.
VariableMap = dict[str, str]


class CheckCache:
    """Runs registered `edacore.assumptions` checks, once per
    (dataset version, check, scope).

    Section 7.2 requires the cache; the reason is that a single candidate
    set runs the same normality check for several methods, and a check is a
    fact about the data, not about the method that asked for it. The
    version is a content hash by default, so a transformed DataFrame
    (Section 12.1's version DAG) never reads a stale fact.
    """

    def __init__(self, df: pd.DataFrame, dataset_version: str | None = None) -> None:
        self.df = df
        self.dataset_version = dataset_version or self._content_hash(df)
        self._cache: dict[tuple[Any, ...], list[AssumptionCheck]] = {}
        self.n_runs = 0

    @staticmethod
    def _content_hash(df: pd.DataFrame) -> str:
        row_hash = int(pd.util.hash_pandas_object(df, index=True).sum())
        return f"{df.shape[0]}x{df.shape[1]}:{hash(tuple(df.columns)) ^ row_hash:x}"

    def run(self, check_name: str, **params: Any) -> list[AssumptionCheck]:
        key = (self.dataset_version, check_name, tuple(sorted(params.items(), key=str)))
        cached = self._cache.get(_hashable(key))
        if cached is not None:
            return cached
        result = registry.get(check_name).func(self.df, **params)
        checks = list(result) if isinstance(result, list) else [result]
        self._cache[_hashable(key)] = checks
        self.n_runs += 1
        return checks


def _hashable(key: tuple[Any, ...]) -> tuple[Any, ...]:
    """Cache keys must be hashable; a param value occasionally is not
    (a list of covariate names, say), so fall back to its repr."""
    out: list[Any] = []
    for part in key:
        if isinstance(part, tuple):
            out.append(_hashable(part))
        elif isinstance(part, Hashable):
            out.append(part)
        else:
            out.append(repr(part))
    return tuple(out)


@dataclass(frozen=True)
class ResolutionContext:
    """Everything a resolver needs: the data, the confirmed question, the
    family's role assignment, and the shared cache."""

    df: pd.DataFrame
    spec: QuestionSpec
    variables: VariableMap
    cache: CheckCache
    semantic_types: dict[str, tuple[SemanticType, float]] = field(default_factory=dict)

    def column(self, *roles: str) -> str | None:
        for role in roles:
            col = self.variables.get(role)
            if col is not None:
                return col
        return None

    def columns(self, *roles: str) -> list[str]:
        return [self.variables[role] for role in roles if role in self.variables]


@dataclass(frozen=True)
class Resolution:
    """What one assumption name resolved to.

    The split matters. `graded` decides the assumption: a FAIL there makes a
    hard assumption INELIGIBLE and a soft one a CAVEAT. `evidence` is shown
    to the user and recorded, but never graded, because the same fact can
    support opposite verdicts depending on what was asked:
    `check_design_crossing` FAILing ("these rows are not independent") is
    fatal to an independent-samples test and is exactly what a paired test
    needs. Grading it directly made every paired test ineligible on paired
    data -- the bug this type exists to prevent.
    """

    graded: list[AssumptionCheck]
    evidence: list[AssumptionCheck] = field(default_factory=list)

    @property
    def all_checks(self) -> list[AssumptionCheck]:
        return [*self.graded, *self.evidence]


Resolver = Callable[[ResolutionContext], "Resolution"]


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------


def _untestable(assumption: str, reason: str, scope: dict[str, str] | None = None) -> Resolution:
    """An assumption that data alone cannot settle (Section 5.2's UNTESTABLE)."""
    return Resolution(
        [
            AssumptionCheck(
                fact_id=f"{assumption}.untestable",
                assumption=assumption,
                method="design_question",
                scope=scope or {},
                statistic=None,
                p_value=None,
                threshold="cannot be decided from the data; depends on how it was collected",
                status=CheckStatus.UNTESTABLE,
                consequence=reason,
            )
        ]
    )


def _verdict(
    assumption: str,
    method: str,
    passes: bool,
    threshold: str,
    consequence: str,
    scope: dict[str, str],
    statistic: float | None = None,
) -> AssumptionCheck:
    """A deterministic PASS/FAIL fact the engine derives from the confirmed
    design or from an inferred type, rather than from a test statistic."""
    return AssumptionCheck(
        fact_id=f"{assumption}." + ".".join(f"{k}={v}" for k, v in sorted(scope.items())),
        assumption=assumption,
        method=method,
        scope=scope,
        statistic=statistic,
        p_value=None,
        threshold=threshold,
        status=CheckStatus.PASS if passes else CheckStatus.FAIL,
        consequence=consequence,
    )


_SCALE_SETS: dict[str, frozenset[SemanticType]] = {
    "continuous": frozenset({SemanticType.CONTINUOUS}),
    "numeric": frozenset({SemanticType.CONTINUOUS, SemanticType.DISCRETE}),
    "numeric_outcome": frozenset({SemanticType.CONTINUOUS, SemanticType.DISCRETE}),
    "numeric_or_ordinal": frozenset(
        {SemanticType.CONTINUOUS, SemanticType.DISCRETE, SemanticType.ORDINAL}
    ),
    "ordinal_or_higher": frozenset(
        {
            SemanticType.CONTINUOUS,
            SemanticType.DISCRETE,
            SemanticType.ORDINAL,
            SemanticType.BINARY,
        }
    ),
    "binary": frozenset({SemanticType.BINARY}),
    "binary_outcome": frozenset({SemanticType.BINARY}),
    "categorical": frozenset({SemanticType.NOMINAL, SemanticType.ORDINAL, SemanticType.BINARY}),
    "categorical_outcome": frozenset(
        {SemanticType.NOMINAL, SemanticType.ORDINAL, SemanticType.BINARY}
    ),
    "ordinal_exposure": frozenset({SemanticType.ORDINAL}),
}

# Which roles each scale assumption applies to. Only roles actually present
# in the family's variable map are checked, so a family that has no `x`/`y`
# simply skips them.
_SCALE_ROLES: dict[str, tuple[str, ...]] = {
    "continuous": ("outcome", "variable", "x", "y"),
    "numeric": ("x", "y", "outcome", "variable", "covariate"),
    "numeric_outcome": ("outcome", "variable"),
    "numeric_or_ordinal": ("outcome", "variable", "x", "y"),
    "ordinal_or_higher": ("outcome", "variable", "x", "y"),
    "binary": ("binary",),
    "binary_outcome": ("outcome", "variable", "binary"),
    "categorical": ("a", "b", "x", "y", "outcome", "group"),
    "categorical_outcome": ("outcome", "variable"),
    "ordinal_exposure": ("ordinal",),
}


def _scale_resolver(assumption: str) -> Resolver:
    """Resolve a measurement-level assumption by delegating to
    `check_measurement_level` once per acceptable semantic type, and
    accepting the column if any of them matches.

    The per-type calls are real, cached facts (their `fact_id` carries the
    required scale), so the evidence trail shows exactly what was tried.
    """
    accepted = _SCALE_SETS[assumption]
    roles = _SCALE_ROLES[assumption]

    def resolve(ctx: ResolutionContext) -> Resolution:
        graded: list[AssumptionCheck] = []
        evidence: list[AssumptionCheck] = []
        for col in ctx.columns(*roles):
            attempts = [
                ctx.cache.run("check_measurement_level", col=col, required=scale)[0]
                for scale in sorted(accepted, key=lambda s: s.value)
            ]
            matched = next((c for c in attempts if c.status is CheckStatus.PASS), None)
            decisive = matched if matched is not None else attempts[0]
            graded.append(decisive)
            # The scales that did not match are evidence, not failures: an
            # ordinal column legitimately fails "is it continuous?" while
            # still satisfying "ordinal or higher".
            evidence.extend(c for c in attempts if c is not decisive)
        if not graded:
            return _untestable(
                assumption,
                f"no column in this question plays one of the roles {roles} that "
                f"'{assumption}' applies to",
            )
        return Resolution(graded, evidence)

    return resolve


def _paired_differences(ctx: ResolutionContext) -> pd.Series | None:
    """The per-subject differences a paired test operates on, from either
    layout: two measurement columns (wide), or outcome/group/subject (long).

    Returns None when the data cannot be paired up, which the caller turns
    into a FAIL rather than guessing.
    """
    a, b = ctx.variables.get("a"), ctx.variables.get("b")
    if a is not None and b is not None:
        pair = ctx.df[[a, b]].dropna()
        return pair[a] - pair[b]

    outcome = ctx.column("outcome")
    group = ctx.column("group")
    subject = ctx.column("subject", "id", "entity")
    if outcome is None or group is None or subject is None:
        return None
    levels = list(pd.unique(ctx.df[group].dropna()))
    if len(levels) != 2:
        return None
    wide = ctx.df.pivot_table(index=subject, columns=group, values=outcome, aggfunc="mean").dropna()
    if wide.empty:
        return None
    differences: pd.Series = wide[levels[0]] - wide[levels[1]]
    return differences


def _normality_evidence(ctx: ResolutionContext) -> list[AssumptionCheck]:
    """Shapiro-Wilk per group, plus the descriptive (skew/kurtosis) check.

    Section 6.6's own note on `check_normality_descriptive` is "always run
    alongside a formal test", and the two answer different questions: the
    formal test asks whether the deviation is larger than sampling noise,
    the descriptive one asks whether it is large enough to matter. A
    borderline skew with a passing Shapiro is a real caveat, and only the
    descriptive check can see it.
    """
    outcome = ctx.column("outcome", "variable")
    if outcome is None:
        return _untestable("normality", "no outcome column to test for normality").graded
    group = ctx.column("group")
    return [
        *ctx.cache.run("check_normality_shapiro", col=outcome, by=group),
        *ctx.cache.run("check_normality_descriptive", col=outcome, by=group),
    ]


def _group_sizes(ctx: ResolutionContext) -> list[int]:
    outcome, group = ctx.column("outcome", "variable"), ctx.column("group")
    if outcome is None:
        return []
    if group is None:
        return [int(ctx.df[outcome].notna().sum())]
    return [int(rows[outcome].notna().sum()) for _, rows in ctx.df.groupby(group, observed=True)]


# --------------------------------------------------------------------------
# Resolvers
# --------------------------------------------------------------------------


def _resolve_normality(ctx: ResolutionContext) -> Resolution:
    return Resolution(_normality_evidence(ctx))


def _resolve_normality_or_large_n(ctx: ResolutionContext) -> Resolution:
    """Normality, with the CLT escape that Section 6.7's "normality or large
    n" actually names.

    Shapiro-Wilk's null is *exact* normality, which no real sample satisfies;
    at large n it therefore rejects on deviations a mean-based test does not
    care about. So above `LARGE_N_FOR_CLT` per group, and with no extreme
    skew, the failure is reported as a PASS carrying the reason -- but the
    underlying Shapiro checks stay in the evidence, so nothing is hidden.
    """
    checks = _normality_evidence(ctx)
    if all(c.status is not CheckStatus.FAIL for c in checks):
        return Resolution(checks)

    sizes = _group_sizes(ctx)
    outcome = ctx.column("outcome", "variable")
    if not sizes or outcome is None or min(sizes) < LARGE_N_FOR_CLT:
        return Resolution(checks)

    descriptive = [c for c in checks if c.method == "descriptive_skew_kurtosis"]
    skews = [abs(c.statistic) for c in descriptive if c.statistic is not None]
    if skews and max(skews) >= MAX_SKEW_FOR_CLT:
        return Resolution(checks)

    # The CLT verdict REPLACES the Shapiro failures as the graded fact --
    # it does not sit alongside them, or the failure it exists to excuse
    # would still make the method a caveat. The Shapiro checks stay as
    # evidence, so the user still sees exactly what was rejected.
    return Resolution(
        graded=[
            _verdict(
                "normality_or_large_n",
                "central_limit_theorem",
                passes=True,
                threshold=(
                    f"every group n >= {LARGE_N_FOR_CLT} and |skew| < {MAX_SKEW_FOR_CLT}, so the "
                    "sampling distribution of the mean is approximately normal even though the "
                    "data is not"
                ),
                consequence=(
                    "At this sample size a formal normality test rejects deviations that do "
                    "not affect a mean-based test; the CLT covers them."
                ),
                scope={"variable": outcome},
                statistic=float(min(sizes)),
            )
        ],
        evidence=checks,
    )


def _resolve_normality_of_differences(ctx: ResolutionContext) -> Resolution:
    diffs = _paired_differences(ctx)
    if diffs is None:
        return _untestable(
            "normality_of_differences",
            "the paired differences cannot be formed: no matching subject id across groups",
        )
    frame = pd.DataFrame({"_diff": diffs.to_numpy(dtype=float)})
    return Resolution(
        [
            c.model_copy(update={"fact_id": f"{c.fact_id}.paired_differences"})
            for c in registry.get("check_normality_shapiro").func(frame, col="_diff", by=None)
        ]
    )


def _resolve_bivariate_normality(ctx: ResolutionContext) -> Resolution:
    """Marginal normality of both variables.

    Marginal normality does not imply joint (bivariate) normality -- it is
    the standard practical proxy, and it is one-directional: if a margin is
    non-normal the pair cannot be bivariate normal, so a FAIL here is
    conclusive while a PASS is only suggestive. Pearson's point estimate
    does not need this at all; its CI and p-value do.
    """
    out: list[AssumptionCheck] = []
    for col in ctx.columns("x", "y"):
        out.extend(ctx.cache.run("check_normality_shapiro", col=col, by=None))
    return Resolution(out) if out else _untestable("bivariate_normality", "no x/y pair to test")


def _resolve_normality_within_groups(ctx: ResolutionContext) -> Resolution:
    binary, numeric = ctx.column("binary"), ctx.column("x", "outcome")
    if binary is None or numeric is None:
        return Resolution(_normality_evidence(ctx))
    return Resolution(ctx.cache.run("check_normality_shapiro", col=numeric, by=binary))


def _resolve_equal_variance(ctx: ResolutionContext) -> Resolution:
    outcome, group = ctx.column("outcome", "variable"), ctx.column("group")
    if outcome is None or group is None:
        return _untestable("equal_variance", "no outcome/group pair to compare variances across")
    return Resolution(ctx.cache.run("check_equal_variance_levene", col=outcome, group=group))


def _resolve_same_shape(ctx: ResolutionContext) -> Resolution:
    outcome, group = ctx.column("outcome", "variable"), ctx.column("group")
    if outcome is None or group is None:
        return _untestable("same_shape", "no outcome/group pair to compare shapes across")
    return Resolution(ctx.cache.run("check_same_shape", col=outcome, group=group))


def _resolve_symmetry(ctx: ResolutionContext) -> Resolution:
    col = ctx.column("variable", "outcome")
    if col is None:
        return _untestable("symmetry", "no variable to test for symmetry")
    return Resolution(ctx.cache.run("check_symmetry", col=col, by=None))


def _resolve_symmetry_of_differences(ctx: ResolutionContext) -> Resolution:
    diffs = _paired_differences(ctx)
    if diffs is None:
        return _untestable(
            "symmetry_of_differences",
            "the paired differences cannot be formed: no matching subject id across groups",
        )
    frame = pd.DataFrame({"_diff": diffs.to_numpy(dtype=float)})
    return Resolution(
        [
            c.model_copy(update={"fact_id": f"{c.fact_id}.paired_differences"})
            for c in registry.get("check_symmetry").func(frame, col="_diff", by=None)
        ]
    )


def _resolve_sphericity(ctx: ResolutionContext) -> Resolution:
    dv, subject, within = (
        ctx.column("outcome", "variable"),
        ctx.column("subject", "id", "entity"),
        ctx.column("group", "within"),
    )
    if dv is None or subject is None or within is None:
        return _untestable("sphericity", "sphericity needs an outcome, a subject id and a factor")
    return Resolution(
        ctx.cache.run("check_sphericity_mauchly", subject=subject, within=within, dv=dv)
    )


def _resolve_balanced(ctx: ResolutionContext) -> Resolution:
    subject, within = (
        ctx.column("subject", "id", "entity"),
        ctx.column("group", "within"),
    )
    if subject is None or within is None:
        return _untestable("balanced", "balance needs a subject id and a within-subject factor")
    return Resolution(ctx.cache.run("check_balanced_design", subject=subject, within=within))


def _resolve_linearity(ctx: ResolutionContext) -> Resolution:
    x, y = ctx.column("x"), ctx.column("y")
    if x is None or y is None:
        return _untestable("linearity", "no x/y pair to test for linearity")
    return Resolution(ctx.cache.run("check_linearity", x=x, y=y))


def _resolve_monotonicity(ctx: ResolutionContext) -> Resolution:
    x, y = ctx.column("x"), ctx.column("y")
    if x is None or y is None:
        return _untestable("monotonicity", "no x/y pair to test for monotonicity")
    return Resolution(ctx.cache.run("check_monotonicity", x=x, y=y))


def _resolve_no_influential_outliers(ctx: ResolutionContext) -> Resolution:
    x, y = ctx.column("x"), ctx.column("y")
    if x is None or y is None:
        return _untestable("no_influential_outliers", "no x/y pair to measure influence on")
    return Resolution(ctx.cache.run("check_influential_outliers", x=x, y=y))


def _resolve_expected_counts(ctx: ResolutionContext) -> Resolution:
    """Two-way tables use the margins; a goodness-of-fit question uses the
    reference proportions it was asked against."""
    variable = ctx.column("variable")
    if variable is not None and ctx.spec.reference_proportions is not None:
        return Resolution(
            ctx.cache.run(
                "check_expected_counts_gof",
                col=variable,
                expected=dict(ctx.spec.reference_proportions),
            )
        )

    a, b = ctx.column("a", "x", "outcome"), ctx.column("b", "y", "group")
    if a is None or b is None:
        return _untestable(
            "expected_counts",
            "expected counts need either a two-way table or the reference proportions a "
            "goodness-of-fit question is asked against, and this question has neither yet",
        )
    return Resolution(ctx.cache.run("check_expected_counts", a=a, b=b))


def _resolve_np_at_least_10(ctx: ResolutionContext) -> Resolution:
    outcome, group = ctx.column("outcome", "variable"), ctx.column("group")
    if outcome is None or group is None:
        return _untestable("np_at_least_10", "no outcome/group pair to count events in")
    event = _majority_event(ctx.df[outcome])
    return Resolution(
        ctx.cache.run("check_proportion_counts", outcome=outcome, group=group, event=event)
    )


def _majority_event(series: pd.Series) -> Any:
    """Which level counts as the "event" for a binary outcome.

    The check is symmetric -- it takes the smaller of successes and
    failures in each group -- so the choice cannot change its verdict; it
    only has to be deterministic.
    """
    values = sorted(pd.unique(series.dropna()), key=str)
    return values[-1] if values else None


def _resolve_min_n_per_group(ctx: ResolutionContext) -> Resolution:
    outcome, group = ctx.column("outcome", "variable"), ctx.column("group")
    if outcome is None or group is None:
        return _untestable("min_n_per_group>=2", "no outcome/group pair to count")
    return Resolution(ctx.cache.run("check_sample_size", col=outcome, group=group, min_n=2))


def _resolve_paired_binary(ctx: ResolutionContext) -> Resolution:
    out: list[AssumptionCheck] = []
    for col in ctx.columns("a", "b", "outcome"):
        attempts = ctx.cache.run("check_measurement_level", col=col, required=SemanticType.BINARY)
        out.extend(attempts)
    if not out:
        return _untestable("paired_binary", "no columns to check for a paired binary outcome")
    return Resolution(out)


# --- design ---------------------------------------------------------------

_DESIGN_CONSEQUENCE = {
    "independent": (
        "An independent-samples test treats every row as a separate subject; if the same "
        "subject appears more than once, it counts one person's repeated readings as "
        "independent evidence and reports more certainty than the data supports."
    ),
    "paired": (
        "A paired test subtracts each subject's own two measurements; without a real "
        "one-to-one link between rows, that subtraction pairs unrelated observations."
    ),
    "repeated": (
        "A repeated-measures method partitions each subject's variance across conditions; "
        "without repeated observations per subject there is nothing to partition."
    ),
}


def _design_resolver(required: Design) -> Resolver:
    """Resolve a design assumption against the design the USER confirmed,
    corroborated by the id evidence.

    The confirmed design is authoritative -- Section 7.1 makes design a
    user decision, and `validate_spec` will not let an unconfirmed or
    contradicted one reach the engine. This resolver's extra job is to
    catch the case where the confirmed design still conflicts with the ids
    (a user override), and fail rather than go along with it.
    """
    name = required.value

    def resolve(ctx: ResolutionContext) -> Resolution:
        design = ctx.spec.design
        group = ctx.column("group")
        scope = {"design": design.value, "required": name}

        if design is Design.UNKNOWN:
            return _untestable(
                name,
                "the design has not been confirmed yet; whether these rows are independent "
                "cannot be read off the values alone",
                scope=scope,
            )

        matches = design is required
        evidence: list[AssumptionCheck] = []
        extra_graded: list[AssumptionCheck] = []
        id_col = ctx.column("subject", "id", "entity")
        if group is not None and id_col is not None:
            evidence = ctx.cache.run("check_design_crossing", group=group, id_col=id_col)
            crossing_fails = any(c.status is CheckStatus.FAIL for c in evidence)
            if required is Design.INDEPENDENT and crossing_fails:
                matches = False
        elif required is Design.INDEPENDENT:
            # No id column to corroborate with, so independence rests on the
            # user's confirmation alone. Section 6.6's check_independence_design
            # says exactly that, and it is GRADED rather than evidence: unlike
            # the crossing check, "nobody can tell from the values" means the
            # same thing whichever design was claimed, and grading it is what
            # puts it on every candidate's reasons. UNTESTABLE never blocks.
            extra_graded = ctx.cache.run("check_independence_design", id_col=None)

        # The crossing check is EVIDENCE, never the verdict: "these rows
        # are not independent" is fatal to an independent-samples test and
        # is precisely what a paired one requires, so the same FAIL must
        # not be graded the same way for both.
        return Resolution(
            graded=[
                *extra_graded,
                _verdict(
                    name,
                    "confirmed_design",
                    passes=matches,
                    threshold=(
                        f"the confirmed design is '{design.value}'; this method needs '{name}'"
                    ),
                    consequence=_DESIGN_CONSEQUENCE[name],
                    scope=scope,
                ),
            ],
            evidence=evidence,
        )

    return resolve


RESOLVERS: dict[str, Resolver] = {
    # measurement level
    **{name: _scale_resolver(name) for name in _SCALE_SETS},
    "paired_binary": _resolve_paired_binary,
    # design
    "independent": _design_resolver(Design.INDEPENDENT),
    "paired": _design_resolver(Design.PAIRED),
    "repeated": _design_resolver(Design.REPEATED),
    "balanced": _resolve_balanced,
    # distributional
    "normality": _resolve_normality,
    "normality_or_large_n": _resolve_normality_or_large_n,
    "normality_of_differences": _resolve_normality_of_differences,
    "normality_within_groups": _resolve_normality_within_groups,
    "bivariate_normality": _resolve_bivariate_normality,
    "equal_variance": _resolve_equal_variance,
    "same_shape": _resolve_same_shape,
    "symmetry": _resolve_symmetry,
    "symmetry_of_differences": _resolve_symmetry_of_differences,
    "sphericity": _resolve_sphericity,
    "linearity": _resolve_linearity,
    "monotonicity": _resolve_monotonicity,
    "no_influential_outliers": _resolve_no_influential_outliers,
    # counts
    "expected_counts": _resolve_expected_counts,
    "np_at_least_10": _resolve_np_at_least_10,
    "min_n_per_group>=2": _resolve_min_n_per_group,
    # untestable by construction (Section 5.2)
    "exchangeability": lambda ctx: _untestable(
        "exchangeability",
        "a permutation test assumes the group labels are exchangeable under the null; "
        "that follows from how the data was collected, not from its values",
    ),
    "ordered_sequence": lambda ctx: _untestable(
        "ordered_sequence",
        "a runs test reads the rows in the order they are stored; whether that order is "
        "the meaningful one (time, position) is something only you can confirm",
    ),
}


def resolve(assumption: str, ctx: ResolutionContext) -> Resolution:
    """Evidence for one assumption name, or a hard failure if it is unknown.

    Unknown names are a bug, not a caveat: a method whose assumption the
    engine cannot resolve would otherwise be silently reported as eligible.
    """
    resolver = RESOLVERS.get(assumption)
    if resolver is None:
        raise KeyError(
            f"no resolver for assumption '{assumption}'. Every name used in a registered "
            f"function's assumptions must be resolvable, or the eligibility engine would "
            f"silently ignore it. Add a resolver in edacopilot.eligibility.checks."
        )
    return resolver(ctx)


def infer_types(df: pd.DataFrame) -> dict[str, tuple[SemanticType, float]]:
    result: dict[str, tuple[SemanticType, float]] = infer_semantic_types(df)
    return result


def numeric_like(semantic_type: SemanticType) -> bool:
    return semantic_type in (SemanticType.CONTINUOUS, SemanticType.DISCRETE)


def n_levels(df: pd.DataFrame, col: str) -> int:
    return int(df[col].nunique(dropna=True))


def is_constant(values: np.ndarray) -> bool:
    return bool(np.ptp(values) == 0)
