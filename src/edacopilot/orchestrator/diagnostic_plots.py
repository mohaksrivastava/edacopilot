"""The plot that shows what a failed assumption looks like (Section 9.4).

Section 1.3's "transparent" principle says every recommendation shows the
diagnostics behind it, and Section 13.2 has plots rendering inline in
cards. A Shapiro-Wilk p-value of 0.00009 tells a junior analyst that
normality failed; a Q-Q plot tells them *how* — a long right tail is a
different problem from two outliers, and they lead to different methods.
That is the gap this mapping closes.

**Only FAIL and BORDERLINE checks get a plot.** Rendering all of them
would put eight images on a card where two matter, and Section 1.3's other
principle ("show disagreement, hide noise") applies to evidence as much as
to personas: a card the user learns to scroll past is worse than a shorter
one, because the one time the Q-Q plot matters it gets scrolled past too.
`plot_all=True` overrides this for anyone who wants the full set.

**The group column comes from the question, not from the check's scope.**
`equal_variance` records `scope["group"]` as the grouping *column* while
`normality_or_large_n` records the failing *level*, so reading the scope
would silently plot by a column called "F". The `QuestionSpec` knows which
column was being grouped by; the scope is only consulted for the variable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from edacopilot.eligibility import QuestionSpec
from edacore.contracts import AssumptionCheck, CheckStatus

# Statuses whose checks are worth illustrating (see the module docstring).
PLOTTED_STATUSES = frozenset({CheckStatus.FAIL, CheckStatus.BORDERLINE})

PlotSpec = tuple[str, dict[str, Any]]
Builder = Callable[[AssumptionCheck, QuestionSpec], PlotSpec | None]


def _outcome(check: AssumptionCheck, spec: QuestionSpec) -> str | None:
    """The column the check is about."""
    return (
        check.scope.get("variable")
        or spec.variables.get("outcome")
        or spec.variables.get("variable")
    )


def _group(spec: QuestionSpec) -> str | None:
    """The grouping column, from the question (see the module docstring)."""
    return spec.variables.get("group")


def _qq(check: AssumptionCheck, spec: QuestionSpec) -> PlotSpec | None:
    """Normality: the shape of the departure is the whole point."""
    column = _outcome(check, spec)
    if column is None:
        return None
    return "qq_plot", {"col": column, "group": _group(spec)}


def _spread(check: AssumptionCheck, spec: QuestionSpec) -> PlotSpec | None:
    """Equal variance: boxes side by side make unequal spread obvious."""
    column, group = _outcome(check, spec), _group(spec)
    if column is None or group is None:
        return None
    return "boxplot", {"col": column, "group": group}


def _shape(check: AssumptionCheck, spec: QuestionSpec) -> PlotSpec | None:
    """Same shape: a violin shows shape, which is what the check is about
    and what decides whether a rank test reads as a median difference."""
    column, group = _outcome(check, spec), _group(spec)
    if column is None or group is None:
        return None
    return "violin", {"col": column, "group": group}


def _symmetry(check: AssumptionCheck, spec: QuestionSpec) -> PlotSpec | None:
    column = _outcome(check, spec)
    return None if column is None else ("histogram", {"col": column})


def _relationship(check: AssumptionCheck, spec: QuestionSpec) -> PlotSpec | None:
    """Linearity and monotonicity: the LOWESS curve against the scatter is
    exactly the comparison the check performs."""
    x, y = spec.variables.get("x"), spec.variables.get("y")
    if x is None or y is None:
        return None
    return "scatter_lowess", {"x": x, "y": y}


def _table(check: AssumptionCheck, spec: QuestionSpec) -> PlotSpec | None:
    """Expected counts: a mosaic shows which cell is thin."""
    a = spec.variables.get("outcome") or spec.variables.get("x")
    b = spec.variables.get("group") or spec.variables.get("y")
    if a is None or b is None:
        return None
    return "mosaic", {"a": a, "b": b}


def _outliers(check: AssumptionCheck, spec: QuestionSpec) -> PlotSpec | None:
    column = _outcome(check, spec)
    return None if column is None else ("outlier_plot", {"col": column})


def _counts(check: AssumptionCheck, spec: QuestionSpec) -> PlotSpec | None:
    """Sample size: a strip plot shows n, which a box plot hides."""
    column, group = _outcome(check, spec), _group(spec)
    if column is None or group is None:
        return None
    return "strip_by_group", {"col": column, "group": group}


BUILDERS: dict[str, Builder] = {
    "normality": _qq,
    "normality_or_large_n": _qq,
    "normality_of_differences": _qq,
    "normality_within_groups": _qq,
    "bivariate_normality": _qq,
    "equal_variance": _spread,
    "homoscedasticity": _spread,
    "same_distribution_shape": _shape,
    "same_shape": _shape,
    "symmetry": _symmetry,
    "symmetry_of_differences": _symmetry,
    "linearity": _relationship,
    "monotonicity": _relationship,
    "expected_counts": _table,
    "np_at_least_10": _table,
    "no_influential_outliers": _outliers,
    "adequate_n": _counts,
    "sample_size": _counts,
    "min_n_per_group>=2": _counts,
}


def plot_for_check(check: AssumptionCheck, spec: QuestionSpec) -> PlotSpec | None:
    """The plot that illustrates this check, or None if there isn't one.

    None is a legitimate answer and the common one: independence and
    exchangeability are facts about data collection with nothing to draw,
    and a measurement-level check is a statement about a column's type.
    """
    builder = BUILDERS.get(check.assumption)
    return None if builder is None else builder(check, spec)


def illustrate(
    checks: list[AssumptionCheck],
    spec: QuestionSpec,
    render: Callable[[str, dict[str, Any]], str],
    *,
    plot_all: bool = False,
) -> tuple[list[AssumptionCheck], list[str]]:
    """Attach plots to the checks that warrant them.

    Returns the checks with `plot_ref` filled where a plot was drawn, and
    the refs in card order. `render` does the drawing and storing, so this
    module stays free of the session.

    A plot that fails to render is skipped rather than raised: a card
    without its picture is worse than a card with one, but both are far
    better than a turn that dies because a degenerate column broke a
    smoother.
    """
    illustrated: list[AssumptionCheck] = []
    refs: list[str] = []
    for check in checks:
        wanted = plot_all or check.status in PLOTTED_STATUSES
        plot = plot_for_check(check, spec) if wanted else None
        if plot is None:
            illustrated.append(check)
            continue
        function, params = plot
        try:
            ref = render(function, params)
        except Exception:  # noqa: BLE001 - see docstring
            illustrated.append(check)
            continue
        illustrated.append(check.model_copy(update={"plot_ref": ref}))
        if ref not in refs:
            refs.append(ref)
    return illustrated, refs
