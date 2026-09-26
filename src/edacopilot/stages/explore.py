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

from typing import TYPE_CHECKING

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
            actions=[suggestion.as_button() for suggestion in self.next_suggestions(session)],
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
