"""Stage modules (ARCHITECTURE.md, Section 9.3).

`STAGES` maps every `Stage` in the state machine to its module, including
the stubs — so the orchestrator never has to ask whether a stage exists,
only what it says when you get there.
"""

from __future__ import annotations

from typing import Final

from edacopilot.orchestrator.state_machine import Stage

from .base import (
    BaseStage,
    FunctionNotAllowedError,
    StageModule,
    StageNotAvailableError,
    StepOutcome,
)
from .explore import ExploreStage
from .hypothesis import POST_HOC_FOR, HypothesisStage, alpha_for
from .profile import ProfileStage, unconfirmed_columns
from .stubs import (
    ExportStage,
    MissingnessStage,
    OutliersStage,
    QualityStage,
    StubStage,
    TextStage,
    TimeSeriesStage,
    TransformStage,
)


class IngestStage(BaseStage):
    """A placeholder for the one stage that has already happened.

    `Session.start(df)` *is* ingestion, so this stage exists only so the
    state machine has a module for every stage and nothing has to special-case
    the first one.
    """

    name = "ingest"
    summary = "loading the dataset"
    milestone = "M6"
    allowed_functions: list[str] = []

    def entry_summary(self, session):  # type: ignore[no-untyped-def]
        from edacopilot.orchestrator.cards import Card

        return Card(
            kind="info",
            title="Dataset loaded",
            stage=self.name,
            body_md=(
                f"{len(session.data)} rows, {len(session.data.columns)} columns, "
                f"version `{session.version}` on branch `{session.active_branch}`. "
                f"The original data is kept read-only as `v0`, so any change can be undone."
            ),
        )


STAGES: Final[dict[Stage, BaseStage]] = {
    Stage.INGEST: IngestStage(),
    Stage.PROFILE: ProfileStage(),
    Stage.QUALITY: QualityStage(),
    Stage.MISSINGNESS: MissingnessStage(),
    Stage.OUTLIERS: OutliersStage(),
    Stage.TRANSFORM: TransformStage(),
    Stage.EXPLORE: ExploreStage(),
    Stage.HYPOTHESIS: HypothesisStage(),
    Stage.TIMESERIES: TimeSeriesStage(),
    Stage.TEXT: TextStage(),
    Stage.EXPORT: ExportStage(),
}

IMPLEMENTED: Final[frozenset[Stage]] = frozenset(
    stage for stage, module in STAGES.items() if not isinstance(module, StubStage)
)


def get_stage(stage: Stage) -> BaseStage:
    return STAGES[stage]


__all__ = [
    "IMPLEMENTED",
    "POST_HOC_FOR",
    "STAGES",
    "BaseStage",
    "ExploreStage",
    "ExportStage",
    "FunctionNotAllowedError",
    "HypothesisStage",
    "IngestStage",
    "MissingnessStage",
    "OutliersStage",
    "ProfileStage",
    "QualityStage",
    "StageModule",
    "StageNotAvailableError",
    "StepOutcome",
    "StubStage",
    "TextStage",
    "TimeSeriesStage",
    "TransformStage",
    "alpha_for",
    "get_stage",
    "unconfirmed_columns",
]
