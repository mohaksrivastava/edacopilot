"""Stages this build does not implement yet (ARCHITECTURE.md, Section 16).

QUALITY, MISSINGNESS, OUTLIERS, TRANSFORM, TIMESERIES and TEXT belong to
M10-M13. They exist here as stubs so the state machine is complete: the
user can walk to them, the jump warnings can name them, and the export
stage can say what was never reviewed.

A stub returns an info card that says what the stage will do and which
milestone brings it, and refuses to build a spec or execute anything. It
does not return an empty result: a stage that quietly produced nothing
would let the user believe missing data had been reviewed when it had not,
which is precisely the failure this project exists to prevent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from edacopilot.orchestrator.cards import Card, Suggestion
from edacopilot.orchestrator.intents import IntentType, button

from .base import BaseStage

if TYPE_CHECKING:  # pragma: no cover
    from edacopilot.session.session import Session


class StubStage(BaseStage):
    """A stage whose functions land in a later milestone."""

    allowed_functions: list[str] = []
    # What the user can do instead, right now.
    alternative: str = ""

    def entry_summary(self, session: Session) -> Card:
        body = [
            f"**Available in a later version ({self.milestone}).** "
            f"This stage will cover {self.summary}.",
            "",
            "Nothing in this stage has run, so anything it would have caught is still "
            "in the data. Jumping to a later stage will say so.",
        ]
        if self.alternative:
            body += ["", self.alternative]
        return Card(
            kind="info",
            title=f"{self.name.title()} — not in this build",
            stage=self.name,
            body_md="\n".join(body),
            actions=[suggestion.as_button() for suggestion in self.next_suggestions(session)],
        )

    def next_suggestions(self, session: Session) -> list[Suggestion]:
        return [
            Suggestion(
                label="Back to the profile",
                reason="what is in the data, and what needs confirming",
                intent=button(IntentType.GOTO_STAGE, target_stage="profile"),
                call='session.goto_stage("profile")',
            ),
            Suggestion(
                label="Ask a question of the data",
                reason="the hypothesis stage is implemented",
                intent=button(IntentType.GOTO_STAGE, target_stage="hypothesis"),
                call='session.goto_stage("hypothesis")',
            ),
        ]


class QualityStage(StubStage):
    name = "quality"
    summary = "sentinel values, label merges, range rules and duplicate rows (Section 6.2)"
    milestone = "M11"
    alternative = (
        "The profile already flags sentinel candidates and duplicate rows, so you can see "
        "what this stage would ask about — it just cannot fix it yet."
    )


class MissingnessStage(StubStage):
    name = "missingness"
    summary = "MCAR evidence, MAR predictors, the MNAR conversation, and imputation (Section 6.3)"
    milestone = "M10"
    alternative = (
        "Until then, every test drops rows with missing values in the columns it uses, and "
        "the result card reports the n it actually had."
    )


class OutliersStage(StubStage):
    name = "outliers"
    summary = "detecting extreme values and deciding whether they are errors (Section 6.4)"
    milestone = "M11"
    alternative = (
        "The `no_influential_outliers` assumption check still runs where a method declares "
        "it, so an outlier that would change a conclusion shows up as a caveat."
    )


class TransformStage(StubStage):
    name = "transform"
    summary = "log and Box-Cox transforms, encoding, scaling and binning (Section 6.5)"
    milestone = "M11"
    alternative = (
        "Robust and rank-based methods are available now and often make a transform "
        "unnecessary: the Maverick persona proposes one whenever a soft assumption fails."
    )


class TimeSeriesStage(StubStage):
    name = "timeseries"
    summary = "stationarity, decomposition, autocorrelation and change points (Section 6.9)"
    milestone = "M12"
    alternative = (
        "A TREND question raises `UnsupportedQuestionError` naming this milestone rather "
        "than reporting those methods as ineligible: they are not ineligible, they do not "
        "exist yet."
    )


class TextStage(StubStage):
    name = "text"
    summary = "text column profiling, language detection and term frequencies (Section 6.10)"
    milestone = "M13"
    alternative = (
        "Text columns are already detected in the profile and, per rule 5, no raw text is "
        "ever sent to a model."
    )


class ExportStage(StubStage):
    name = "export"
    summary = "the reproducible notebook and the markdown report (Section 14)"
    milestone = "M14"
    alternative = (
        "`session.code()` already returns every accepted step as runnable Python, which is "
        "the content the export will wrap in a notebook."
    )
