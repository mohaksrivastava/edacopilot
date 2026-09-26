"""The PROFILE stage (ARCHITECTURE.md, Sections 6.1 and 9.3).

Section 9.3's row for PROFILE: confirm low-confidence semantic types,
confirm the target, confirm id/time/entity columns; and "blocks tests on
columns with unconfirmed type < 0.8 confidence".

The stage is read-only. Everything it shows is already in the
`DatasetProfile` the DatasetStore computed, so entering PROFILE creates no
step, no version and no test — which is the point of rule 3 as it applies
to a stage that only looks.

The blocking rule is *not* enforced from here, and that is deliberate: it
is already enforced where it can see the question, in
`eligibility/spec.py`'s `_type_ambiguities`, which turns a low-confidence
column in a named role into an ambiguity the user must answer before any
candidate is built. Enforcing it a second time here would let the two
copies disagree. What this stage does is surface the same columns *before*
the user trips over them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from edacopilot.orchestrator.cards import Card, Suggestion
from edacopilot.orchestrator.intents import Intent, IntentType, button
from edacore.contracts import DatasetProfile, SemanticType

from .base import BaseStage

if TYPE_CHECKING:  # pragma: no cover
    from edacopilot.session.session import Session

# Section 9.3: a column whose semantic type was inferred with less
# confidence than this needs the user's confirmation before a test may rest
# on it.
CONFIDENCE_FLOOR = 0.8


class ProfileStage(BaseStage):
    name = "profile"
    summary = "what is in this dataset, and what each column means"
    milestone = "M1"
    allowed_functions = [
        "profile_dataset",
        "infer_semantic_types",
        "detect_duplicates",
        "detect_structure",
        "detect_leakage_candidates",
        "detect_pii",
        "detect_sentinel_values",
        "summarize_numeric",
        "summarize_categorical",
        "summarize_datetime",
    ]

    def entry_summary(self, session: Session) -> Card:
        profile = session.store.profile()
        body = "\n".join(
            [
                f"{profile.n_rows} rows x {profile.n_cols} columns, "
                f"read as **{profile.structure.replace('_', ' ')}**.",
                "",
                *_column_lines(profile),
                *_flag_lines(profile),
                *_question_lines(profile),
            ]
        )
        return Card(
            kind="info",
            title="Dataset profile",
            stage=self.name,
            body_md=body,
            actions=[suggestion.as_button() for suggestion in self.next_suggestions(session)],
        )

    def next_suggestions(self, session: Session) -> list[Suggestion]:
        profile = session.store.profile()
        suggestions: list[Suggestion] = []
        if unconfirmed_columns(profile):
            suggestions.append(
                Suggestion(
                    label="Confirm the uncertain column types",
                    reason="a test on a column read wrongly answers a different question",
                    intent=button(IntentType.EXPLAIN, topic="semantic_type"),
                    call='session.explain("semantic type")',
                )
            )
        if any(column.n_missing for column in profile.columns):
            suggestions.append(
                Suggestion(
                    label="Review missing values",
                    reason="tests drop incomplete rows silently",
                    intent=button(IntentType.GOTO_STAGE, target_stage="missingness"),
                    call='session.goto_stage("missingness")',
                )
            )
        suggestions.append(
            Suggestion(
                label="Look at the variables",
                reason="univariate and bivariate summaries, read-only",
                intent=button(IntentType.GOTO_STAGE, target_stage="explore"),
                call='session.goto_stage("explore")',
            )
        )
        suggestions.append(
            Suggestion(
                label="Ask a question of the data",
                reason="the hypothesis stage proposes a valid test for it",
                intent=Intent(type=IntentType.GOTO_STAGE, target_stage="hypothesis"),
                call='session.ask(goal="compare_groups", outcome=..., group=...)',
            )
        )
        return suggestions[:4]


def unconfirmed_columns(profile: DatasetProfile) -> list[str]:
    """Columns whose semantic type the user has not confirmed and the
    profiler is not sure of (Section 9.3's blocking rule)."""
    return [
        column.name
        for column in profile.columns
        if column.semantic_confidence < CONFIDENCE_FLOOR
        and column.semantic_type is not SemanticType.CONSTANT
    ]


def _column_lines(profile: DatasetProfile) -> list[str]:
    lines = ["| Column | Type | Confidence | Missing | Unique |", "|---|---|---|---|---|"]
    for column in profile.columns:
        missing = (
            "—"
            if column.n_missing == 0
            else f"{column.n_missing} ({column.n_missing / max(column.n, 1):.0%})"
        )
        lines.append(
            f"| `{column.name}` | {column.semantic_type.value} | "
            f"{column.semantic_confidence:.0%} | {missing} | {column.n_unique} |"
        )
    lines.append("")
    return lines


def _flag_lines(profile: DatasetProfile) -> list[str]:
    lines: list[str] = []
    if profile.duplicate_rows:
        lines.append(
            f"- **{profile.duplicate_rows} exact duplicate rows.** They count twice in "
            f"every test, which makes results look more certain than they are."
        )
    sentinels = [column for column in profile.columns if column.sentinel_candidates]
    for column in sentinels:
        values = ", ".join(str(value) for value in column.sentinel_candidates)
        lines.append(
            f"- **`{column.name}` contains {values}**, which looks like a missing-value "
            f"code rather than a measurement."
        )
    if profile.leakage_candidates:
        joined = ", ".join(f"`{name}`" for name in profile.leakage_candidates)
        lines.append(f"- **Possible target leakage:** {joined} looks derived from the target.")
    if profile.pii_columns:
        joined = ", ".join(f"`{name}`" for name in profile.pii_columns)
        lines.append(
            f"- **Personal data:** {joined}. Nothing from these columns is sent anywhere; "
            f"strict privacy mode is worth turning on (Section 11)."
        )
    if lines:
        lines.append("")
    return lines


def _question_lines(profile: DatasetProfile) -> list[str]:
    """Section 9.3's PROFILE questions, as things to confirm rather than a
    blocking prompt — the user may not care about a column they never use."""
    lines: list[str] = []
    unconfirmed = unconfirmed_columns(profile)
    if unconfirmed:
        joined = ", ".join(f"`{name}`" for name in unconfirmed)
        lines.append(
            f"**To confirm:** {joined} were read with less than "
            f"{CONFIDENCE_FLOOR:.0%} confidence. A test that uses one of them will ask "
            f"before it runs."
        )
    if profile.entity_id:
        lines.append(
            f"**To confirm:** `{profile.entity_id}` looks like a subject identifier. If it "
            f"is, group comparisons on this data are paired rather than independent."
        )
    if profile.time_index:
        lines.append(f"**To confirm:** `{profile.time_index}` looks like the time column.")
    if profile.target:
        lines.append(f"**To confirm:** the target is `{profile.target}`.")
    return lines
