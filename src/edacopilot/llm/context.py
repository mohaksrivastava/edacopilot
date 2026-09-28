"""`ContextBuilder`: the only path from session data to any prompt (Section 11).

Nothing else in this package may put a dataset value in a message sent to
a model. Every field name in `Section11FIELDS` below traces back to
Section 11's table; a field this module does not build is a field no
prompt can contain, which is the property Section 11 exists for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from edacore.contracts import ColumnProfile, DatasetProfile, SemanticType
from edacore.profiling import summarize_categorical

if TYPE_CHECKING:
    from edacopilot.session import Session

# Section 10.4: "truncate to the most relevant 40 columns for the current
# spec". "Most relevant" = named in the spec first, then the rest in
# profile order, so a 500-column frame doesn't blow the context budget on
# columns nobody asked about.
MAX_COLUMNS = 40

# Section 11's table: "top 10 levels sent" in standard mode.
MAX_CATEGORY_LEVELS = 10

_CATEGORICAL_TYPES = frozenset({SemanticType.NOMINAL, SemanticType.ORDINAL, SemanticType.BINARY})

PrivacyLevel = Literal["standard", "strict"]


@dataclass(frozen=True)
class ColumnContext:
    """One column, exactly as Section 11 allows it to appear in a prompt."""

    name: str  # real name (standard) or alias (strict)
    semantic_type: str
    n: int
    n_missing: int
    summary: dict[str, Any]
    top_levels: dict[str, float] | None = None  # None in strict mode
    is_pii: bool = False


@dataclass(frozen=True)
class PromptContext:
    """Everything `ContextBuilder` assembled for one call."""

    privacy_level: PrivacyLevel
    dataset_summary: dict[str, Any]
    columns: list[ColumnContext]
    stage: str
    session_summary: list[str]
    recent_turns: list[tuple[str, str]]
    column_alias_map: dict[str, str] = field(default_factory=dict)  # real -> alias, strict only

    def unalias(self, name: str) -> str:
        """A model's alias back to the real column name (strict mode only)."""
        reverse = {alias: real for real, alias in self.column_alias_map.items()}
        return reverse.get(name, name)


class ContextBuilder:
    """Builds `PromptContext` from a `Session` (Section 11).

    `level="strict"` is forced whenever the dataset's profile has any PII
    column, regardless of what was asked for, until the caller confirms
    otherwise -- callers do that by calling again with
    `pii_confirmed=True`, which is the one thing that can override the
    force.
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    def build(
        self,
        *,
        level: PrivacyLevel = "standard",
        pii_confirmed: bool = False,
        relevant_columns: list[str] | None = None,
    ) -> PromptContext:
        profile = self.session.store.profile()
        effective_level: PrivacyLevel = level
        if profile.pii_columns and not pii_confirmed:
            effective_level = "strict"

        columns = _select_columns(profile, relevant_columns)
        alias_map = _alias_map(columns) if effective_level == "strict" else {}

        built = [self._column_context(col, profile, effective_level, alias_map) for col in columns]

        return PromptContext(
            privacy_level=effective_level,
            dataset_summary={
                "n_rows": profile.n_rows,
                "n_cols": profile.n_cols,
                "structure": profile.structure,
            },
            columns=built,
            stage=self.session.stage,
            session_summary=[_one_line(step) for step in self.session.steps()],
            recent_turns=self.session.recent_turns(),
            column_alias_map=alias_map,
        )

    def _column_context(
        self,
        col: ColumnProfile,
        profile: DatasetProfile,
        level: PrivacyLevel,
        alias_map: dict[str, str],
    ) -> ColumnContext:
        is_pii = col.name in profile.pii_columns
        name = alias_map.get(col.name, col.name) if level == "strict" else col.name

        if is_pii:
            # Section 11: PII gets "name + type only" (standard) or "alias +
            # type only" (strict) -- no summary, no levels, nothing else.
            return ColumnContext(
                name=name,
                semantic_type=col.semantic_type.value,
                n=0,
                n_missing=0,
                summary={},
                is_pii=True,
            )

        top_levels = None
        if (
            level == "standard"
            and col.semantic_type in _CATEGORICAL_TYPES
            and "high_cardinality" not in col.flags
        ):
            # A NOMINAL/ORDINAL column flagged high-cardinality is, in
            # practice, closer to free text or an identifier than to a
            # handful of category labels -- "top levels" of a column where
            # almost every value is unique is almost every raw value,
            # which is exactly what Section 11 says never leaves the
            # session ("raw text values: never").
            top_levels = _top_levels(self.session.data, col.name)

        return ColumnContext(
            name=name,
            semantic_type=col.semantic_type.value,
            n=col.n,
            n_missing=col.n_missing,
            summary=dict(col.summary),
            top_levels=top_levels,
        )


def _select_columns(profile: DatasetProfile, relevant: list[str] | None) -> list[ColumnProfile]:
    """The most relevant columns, capped at `MAX_COLUMNS` (Section 10.4)."""
    if len(profile.columns) <= MAX_COLUMNS:
        return list(profile.columns)
    relevant = relevant or []
    named = [col for col in profile.columns if col.name in relevant]
    rest = [col for col in profile.columns if col.name not in relevant]
    return (named + rest)[:MAX_COLUMNS]


def _alias_map(columns: list[ColumnProfile]) -> dict[str, str]:
    """`col_01`, `col_02`, ... in a stable order (Section 11's strict mode)."""
    return {col.name: f"col_{index:02d}" for index, col in enumerate(columns, start=1)}


def _top_levels(df: Any, col: str) -> dict[str, float]:
    """Top `MAX_CATEGORY_LEVELS` levels by share (Section 11's "standard" row).

    `ColumnProfile.summary` never carries the raw `frequencies` dict --
    it is a nested structure and `_coerce_summary` (edacore) drops those --
    so this reaches into the frame directly, through the same deterministic
    function the profile itself was built from.
    """
    frequencies = summarize_categorical(df, col).get("frequencies", {})
    ranked = sorted(frequencies.items(), key=lambda item: item[1], reverse=True)
    total = sum(frequencies.values()) or 1
    return {str(level): count / total for level, count in ranked[:MAX_CATEGORY_LEVELS]}


def _one_line(step: dict[str, Any]) -> str:
    """One line per accepted step (Section 10.4's session summary)."""
    chosen = step.get("chosen") or "?"
    stage = step.get("stage") or "?"
    return f"{stage}: {chosen}"
