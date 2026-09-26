"""The stage state machine (ARCHITECTURE.md, Section 9.2).

Section 9.2's first bullet is the whole design: the machine defines the
**suggested** order, not a forced one. So nothing here can refuse a
transition. What it can do is notice that the user has jumped past a stage
they never visited and say, in one line, what that costs — with a number
from the profile rather than a general caution, because "you skipped
missingness" teaches nothing and "12% of `income` is missing and tests will
drop those rows" teaches the thing that matters.

The warning is one line by design. A wall of warnings at every jump is a
wall the user learns to scroll past, which is the failure mode Section 1.3
calls out ("Show disagreement, hide noise"). So the line names the single
most consequential concrete fact, and lists the other skipped stages in a
parenthetical.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Final

from edacore.contracts import DatasetProfile, SemanticType


class Stage(StrEnum):
    INGEST = "ingest"
    PROFILE = "profile"
    QUALITY = "quality"
    MISSINGNESS = "missingness"
    OUTLIERS = "outliers"
    TRANSFORM = "transform"
    EXPLORE = "explore"
    HYPOTHESIS = "hypothesis"
    EXPORT = "export"
    TIMESERIES = "timeseries"
    TEXT = "text"


# Section 9.2's main line. TIMESERIES and TEXT are conditional and sit
# outside it, so jumping to HYPOTHESIS never warns about not having looked
# at text columns in a frame that has none.
STAGE_ORDER: Final[tuple[Stage, ...]] = (
    Stage.INGEST,
    Stage.PROFILE,
    Stage.QUALITY,
    Stage.MISSINGNESS,
    Stage.OUTLIERS,
    Stage.TRANSFORM,
    Stage.EXPLORE,
    Stage.HYPOTHESIS,
    Stage.EXPORT,
)

CONDITIONAL_STAGES: Final[tuple[Stage, ...]] = (Stage.TIMESERIES, Stage.TEXT)

# Stages a jump never warns about. INGEST happens by construction (there is
# a dataset, so it was ingested), and EXPLORE is read-only: not having
# looked at a scatter plot cannot invalidate a later test.
_NEVER_WARNS: Final[frozenset[Stage]] = frozenset({Stage.INGEST, Stage.EXPLORE})


def parse_stage(name: str) -> Stage:
    try:
        return Stage(name.strip().lower())
    except ValueError:
        known = ", ".join(stage.value for stage in Stage)
        raise ValueError(f"no stage called '{name}'; the stages are: {known}") from None


def position(stage: Stage) -> int:
    """Where a stage sits in the suggested order.

    A conditional stage has no position in the main line, so it can never
    be "jumped past" — returning -1 keeps it out of `skipped_stages`.
    """
    return STAGE_ORDER.index(stage) if stage in STAGE_ORDER else -1


def skipped_stages(target: Stage, visited: set[Stage]) -> list[Stage]:
    """Stages before `target` in the suggested order that were never visited."""
    target_position = position(target)
    if target_position < 0:
        return []
    return [
        stage
        for stage in STAGE_ORDER[:target_position]
        if stage not in visited and stage not in _NEVER_WARNS
    ]


def jump_warning(target: Stage, visited: set[Stage], profile: DatasetProfile) -> str | None:
    """Section 9.2's one-line warning, or None if nothing was skipped.

    The concrete fact comes from the profile, so the line is about this
    dataset rather than about the idea of missing data.
    """
    skipped = skipped_stages(target, visited)
    if not skipped:
        return None

    detail = _first_concrete_warning(skipped, profile)
    names = ", ".join(stage.value for stage in skipped)
    if detail is None:
        return (
            f"{_stage_list_phrase(skipped)} not been reviewed, so nothing has been "
            f"checked or corrected in the data yet. (Skipped: {names}.)"
        )
    return f"{detail} (Skipped: {names}.)"


def _stage_list_phrase(skipped: list[Stage]) -> str:
    if len(skipped) == 1:
        return f"The {skipped[0].value} stage has"
    return "Earlier stages have"


# Which skipped stage gets to speak, when several were skipped. Not the
# order they appear in the pipeline: the line should name whatever most
# quietly changes a result. Missing values are first because they are
# dropped without being mentioned anywhere the user looks; a sentinel or a
# duplicate at least leaves a visible number behind.
_WARNING_PRIORITY: Final[tuple[Stage, ...]] = (
    Stage.MISSINGNESS,
    Stage.QUALITY,
    Stage.PROFILE,
    Stage.OUTLIERS,
    Stage.TRANSFORM,
)


def _first_concrete_warning(skipped: list[Stage], profile: DatasetProfile) -> str | None:
    """The most consequential thing that is actually true of this dataset."""
    was_skipped = set(skipped)
    for stage in _WARNING_PRIORITY:
        if stage not in was_skipped:
            continue
        detail = _WARNING_BUILDERS[stage](profile)
        if detail is not None:
            return detail
    return None


def _missingness_warning(profile: DatasetProfile) -> str | None:
    worst = max(
        (column for column in profile.columns if column.n_missing > 0),
        key=lambda column: column.n_missing / max(column.n, 1),
        default=None,
    )
    if worst is None:
        return None
    share = worst.n_missing / max(worst.n, 1)
    return (
        f"Missing values haven't been reviewed; {share:.0%} of `{worst.name}` is missing "
        f"and tests will drop those rows."
    )


def _quality_warning(profile: DatasetProfile) -> str | None:
    sentinels = [column for column in profile.columns if column.sentinel_candidates]
    if sentinels:
        first = sentinels[0]
        values = ", ".join(str(value) for value in first.sentinel_candidates)
        return (
            f"Data quality hasn't been reviewed; `{first.name}` contains {values}, which "
            f"looks like a missing-value code and would be averaged as a real number."
        )
    if profile.duplicate_rows:
        share = profile.duplicate_rows / max(profile.n_rows, 1)
        return (
            f"Data quality hasn't been reviewed; {profile.duplicate_rows} rows "
            f"({share:.0%}) are exact duplicates and count twice in every test."
        )
    return None


def _profile_warning(profile: DatasetProfile) -> str | None:
    unconfirmed = [
        column
        for column in profile.columns
        if column.semantic_confidence < 0.8 and column.semantic_type is not SemanticType.CONSTANT
    ]
    if not unconfirmed:
        return None
    first = unconfirmed[0]
    return (
        f"The profile hasn't been confirmed; `{first.name}` was read as "
        f"{first.semantic_type.value} with only {first.semantic_confidence:.0%} confidence."
    )


def _outliers_warning(profile: DatasetProfile) -> str | None:
    # Section 6.4's detection lands in M11. Until then the profile carries
    # no outlier count, so this stays silent rather than inventing one: a
    # warning that fires on every dataset is a warning nobody reads.
    return None


def _transform_warning(profile: DatasetProfile) -> str | None:
    # Transforms are optional by nature -- skipping them invalidates
    # nothing, so there is no consequence to state.
    return None


_WARNING_BUILDERS: Final[dict[Stage, Callable[[DatasetProfile], str | None]]] = {
    Stage.PROFILE: _profile_warning,
    Stage.QUALITY: _quality_warning,
    Stage.MISSINGNESS: _missingness_warning,
    Stage.OUTLIERS: _outliers_warning,
    Stage.TRANSFORM: _transform_warning,
    Stage.HYPOTHESIS: lambda profile: None,
    Stage.EXPORT: lambda profile: None,
}


def applicable_stages(profile: DatasetProfile) -> list[Stage]:
    """The stages this dataset actually has (Section 9.2's conditionals)."""
    stages = list(STAGE_ORDER)
    if profile.structure in ("time_series", "panel"):
        stages.insert(stages.index(Stage.HYPOTHESIS), Stage.TIMESERIES)
    if any(column.semantic_type is SemanticType.TEXT for column in profile.columns):
        stages.insert(stages.index(Stage.HYPOTHESIS), Stage.TEXT)
    return stages


def next_stage(current: Stage, profile: DatasetProfile) -> Stage | None:
    """The stage the suggested order would move to next."""
    stages = applicable_stages(profile)
    if current not in stages:
        return None
    index = stages.index(current)
    return stages[index + 1] if index + 1 < len(stages) else None
