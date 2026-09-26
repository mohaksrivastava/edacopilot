"""The EXPLORE stage (ARCHITECTURE.md, Section 9.3), read-only.

Section 9.3's row: "Univariate summaries and bivariate plots; read-only."
Read-only is load-bearing rather than incidental. Looking at the data
changes what the user will ask next, and Section 6.12's `post_selection`
guard exists because of it — so this stage records that exploration
happened and never creates a step, a version or a test.

Section 6.11's plotting functions are not in this build (no milestone has
claimed them yet), so the bivariate half is numeric: a summary of each
variable, and the association between a pair, computed with functions that
already exist. When `viz.py` lands, the plots attach to the same cards
through `Card.plots`, which is why that field is populated with the refs
this stage would produce rather than left as a promise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from edacopilot.orchestrator.cards import Card, Suggestion
from edacopilot.orchestrator.intents import Intent, IntentType, button
from edacore.contracts import SemanticType
from edacore.profiling import summarize_categorical, summarize_numeric

from .base import BaseStage

if TYPE_CHECKING:  # pragma: no cover
    from edacopilot.session.session import Session

# How many columns an entry summary describes before it stops listing them.
# A 500-column frame's entry card should be readable.
MAX_COLUMNS_SHOWN = 12


class ExploreStage(BaseStage):
    name = "explore"
    summary = "univariate and bivariate summaries, read-only"
    milestone = "M7"
    allowed_functions = [
        "summarize_numeric",
        "summarize_categorical",
        "summarize_datetime",
        "infer_semantic_types",
    ]

    def entry_summary(self, session: Session) -> Card:
        profile = session.store.profile()
        df = session.data
        shown = profile.columns[:MAX_COLUMNS_SHOWN]
        lines = [
            "Read-only. Nothing here changes the data or records a step — and nothing "
            "here is a test, so none of it enters the test ledger.",
            "",
        ]
        for column in shown:
            lines.append(_column_line(df, column.name, column.semantic_type))
        if len(profile.columns) > MAX_COLUMNS_SHOWN:
            lines.append("")
            lines.append(
                f"({len(profile.columns) - MAX_COLUMNS_SHOWN} more columns not shown; "
                f"ask about one by name.)"
            )
        lines.append("")
        lines.append(
            "Anything you notice here was found by looking, so a test chosen because of "
            "it carries the `post_selection` note (Section 6.12): its p-value is optimistic."
        )
        return Card(
            kind="info",
            title="Explore",
            stage=self.name,
            body_md="\n".join(lines),
            plots=self._overview_plots(session, profile),
            actions=[suggestion.as_button() for suggestion in self.next_suggestions(session)],
        )

    def _overview_plots(self, session: Session, profile: Any) -> list[str]:
        """One plot for the whole frame, when there is one worth drawing.

        A correlation heatmap needs at least two numeric columns; below
        that there is nothing to relate and the panel would be an empty
        square with a colourbar.
        """
        numeric = [
            column.name
            for column in profile.columns
            if column.semantic_type in (SemanticType.CONTINUOUS, SemanticType.DISCRETE)
        ]
        if len(numeric) < 2:
            return []
        return _safe_plots(
            session, [("correlation_heatmap", {"cols": numeric[:MAX_COLUMNS_SHOWN]})]
        )

    def describe(self, session: Session, column: str) -> Card:
        """One variable, summarised. Still read-only."""
        df = session.data
        if column not in df.columns:
            return Card(
                kind="warning",
                title=f"No column called '{column}'",
                stage=self.name,
                body_md=f"The columns are: {', '.join(f'`{name}`' for name in df.columns)}.",
            )
        profile = session.store.profile()
        semantic_type = next(
            (c.semantic_type for c in profile.columns if c.name == column), SemanticType.MIXED
        )
        summary = _summarize(df, column, semantic_type)
        lines = [f"`{column}` — read as **{semantic_type.value}**.", ""]
        lines += [f"- {key}: {_format(value)}" for key, value in summary.items()]
        return Card(
            kind="info",
            title=f"Summary of {column}",
            stage=self.name,
            body_md="\n".join(lines),
            plots=_safe_plots(session, _univariate_plots(column, semantic_type)),
            actions=[suggestion.as_button() for suggestion in self.next_suggestions(session)],
        )

    def relate(self, session: Session, x: str, y: str) -> Card:
        """Two variables together — Section 9.3's "bivariate plots".

        Which plot depends on what the pair is, because the same picture
        does not work for two quantities, a quantity by a group, and two
        sets of labels. Still read-only: nothing here is a test, and
        nothing enters the ledger.
        """
        df = session.data
        missing = [name for name in (x, y) if name not in df.columns]
        if missing:
            return Card(
                kind="warning",
                title=f"No column called {', '.join(repr(n) for n in missing)}",
                stage=self.name,
                body_md=f"The columns are: {', '.join(f'`{c}`' for c in df.columns)}.",
            )

        profile = session.store.profile()
        types = {column.name: column.semantic_type for column in profile.columns}
        plots, how = _bivariate_plots(x, y, types)
        return Card(
            kind="info",
            title=f"{x} and {y}",
            stage=self.name,
            body_md=(
                f"{how}\n\nRead-only: this is a look, not a test. A hypothesis chosen "
                f"because of what you see here carries Section 6.12's "
                f"`data_driven_comparison` note — the p-value will be optimistic."
            ),
            plots=_safe_plots(session, plots),
            actions=[suggestion.as_button() for suggestion in self.next_suggestions(session)],
        )

    def next_suggestions(self, session: Session) -> list[Suggestion]:
        return [
            Suggestion(
                label="Ask a question of the data",
                reason="the hypothesis stage proposes a valid test for it",
                intent=Intent(type=IntentType.GOTO_STAGE, target_stage="hypothesis"),
                call='session.ask(goal="compare_groups", outcome=..., group=...)',
            ),
            Suggestion(
                label="Back to the profile",
                reason="column types, duplicates, leakage and personal data",
                intent=button(IntentType.GOTO_STAGE, target_stage="profile"),
                call='session.goto_stage("profile")',
            ),
        ]


_NUMERIC = (SemanticType.CONTINUOUS, SemanticType.DISCRETE)
_CATEGORICAL = (SemanticType.NOMINAL, SemanticType.ORDINAL, SemanticType.BINARY)


def _univariate_plots(column: str, semantic_type: SemanticType) -> list[tuple[str, dict[str, Any]]]:
    """Section 9.3's "univariate summaries", as pictures.

    A histogram and an ECDF for a quantity: the histogram shows shape and
    the ECDF shows it without a binning choice, and disagreement between
    them is usually the binning. A bar chart for labels, where neither
    applies.
    """
    if semantic_type in _NUMERIC:
        return [("histogram", {"col": column}), ("ecdf", {"col": column})]
    if semantic_type is SemanticType.TEXT:
        return [("text_length_hist", {"col": column}), ("top_terms_bar", {"col": column})]
    return [("bar_counts", {"col": column})]


def _bivariate_plots(
    x: str, y: str, types: dict[str, SemanticType]
) -> tuple[list[tuple[str, dict[str, Any]]], str]:
    """The right picture for this pair, and a line saying why."""
    x_type = types.get(x, SemanticType.MIXED)
    y_type = types.get(y, SemanticType.MIXED)

    if x_type in _NUMERIC and y_type in _NUMERIC:
        return (
            [("scatter_lowess", {"x": x, "y": y})],
            "Both are quantities, so a scatter with a LOWESS curve: the curve shows "
            "whether a straight line is the right description, which is what Pearson's "
            "correlation assumes and Spearman's does not.",
        )
    if x_type in _CATEGORICAL and y_type in _NUMERIC:
        return (
            [("boxplot", {"col": y, "group": x}), ("strip_by_group", {"col": y, "group": x})],
            f"`{y}` is a quantity and `{x}` is a grouping, so boxes for the spread and a "
            f"strip plot for the individual points — the strip shows how many "
            f"observations each box rests on, which the box hides.",
        )
    if x_type in _NUMERIC and y_type in _CATEGORICAL:
        return (
            [("boxplot", {"col": x, "group": y}), ("strip_by_group", {"col": x, "group": y})],
            f"`{x}` is a quantity and `{y}` is a grouping, so boxes for the spread and a "
            f"strip plot for the individual points.",
        )
    return (
        [("mosaic", {"a": x, "b": y})],
        "Both are labels, so a mosaic: tile area is cell frequency, and an association "
        "shows as tiles that fail to line up.",
    )


def _safe_plots(session: Session, plots: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """Render each plot, skipping any that will not draw.

    A degenerate column should cost its own panel, not the whole card.
    """
    refs: list[str] = []
    for function, params in plots:
        try:
            refs.append(session.plot(function, **params))
        except Exception:  # noqa: BLE001 - see docstring
            continue
    return refs


def _summarize(df: pd.DataFrame, column: str, semantic_type: SemanticType) -> dict[str, object]:
    if semantic_type in (SemanticType.CONTINUOUS, SemanticType.DISCRETE):
        return dict(summarize_numeric(df, column))
    return dict(summarize_categorical(df, column))


def _column_line(df: pd.DataFrame, column: str, semantic_type: SemanticType) -> str:
    summary = _summarize(df, column, semantic_type)
    keys: tuple[str, ...]
    if semantic_type in (SemanticType.CONTINUOUS, SemanticType.DISCRETE):
        keys = ("n", "mean", "median", "std", "skew")
    else:
        keys = ("n", "n_unique", "top", "top_freq")
    parts = [f"{key}={_format(summary[key])}" for key in keys if key in summary]
    return f"- `{column}` ({semantic_type.value}): " + ", ".join(parts)


def _format(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)
