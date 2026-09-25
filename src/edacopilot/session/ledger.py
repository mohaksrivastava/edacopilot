"""The test ledger (ARCHITECTURE.md, Section 12.3).

The ledger exists because of a specific way analyses go wrong: run twenty
tests, report the one with p < 0.05, and the result is noise dressed as a
finding. So every executed hypothesis test is recorded, adjusted p-values
are recomputed across the whole session after each new test, and the card
always shows the raw p, the adjusted p and the count.

The one subtle rule, spelled out in Section 12.3 and implemented here:

- An **omnibus** test is exactly **one** entry, using its own `p_value` —
  never one entry per comparison it implies. A one-way ANOVA over five
  groups is one conclusion, not ten.
- A **post-hoc** procedure does not enter the session adjustment at all.
  Its `PairwiseComparison.p_adjusted` values are already adjusted within
  its own family by its own procedure (Tukey's studentized range, Dunn's
  Holm step-down). Pooling those into a session-wide correction would
  adjust them twice and make the family's own numbers uninterpretable. The
  omnibus test that triggered the post-hoc still counts normally, because
  running a follow-up does not change how many session-level conclusions
  that omnibus result supports.

Adjustment itself is delegated to `edacore.multiplicity.adjust_pvalues`, so
the ledger cannot drift from the function the rest of the project uses.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from edacore.contracts import PostHocResult, TestResult
from edacore.multiplicity import adjust_pvalues
from edacore.registry import registry

AdjustMethod = Literal["holm", "bonferroni", "fdr_bh", "fdr_by"]

DEFAULT_FAMILY = "session"

# Section 12.3's post-hoc procedures: recorded, but never pooled into the
# session-wide adjustment.
POSTHOC_FUNCTIONS = frozenset(
    {
        "tukey_hsd",
        "games_howell",
        "dunn_test",
        "nemenyi_friedman",
        "conover_friedman",
        "paired_posthoc",
        "mcnemar_posthoc",
        "permutation_posthoc",
        "repeated_measures_posthoc",
        "cochran_q_posthoc",
    }
)


class LedgerEntry(BaseModel):
    """One executed hypothesis test."""

    model_config = ConfigDict(frozen=True)

    entry_id: str
    fact_id: str
    function: str
    family: str = DEFAULT_FAMILY
    branch: str = "main"
    step_id: str | None = None
    p_value: float | None = None
    p_adjusted: float | None = None
    p_adjusted_in_family: float | None = None
    estimand: str = ""
    # True for a post-hoc procedure: recorded for provenance, excluded from
    # the session adjustment and from the session test count (Section 12.3).
    counts_toward_session: bool = True
    # A post-hoc's own within-family adjustment, as its procedure reported
    # it. Carried so the ledger can show it without re-adjusting it.
    internal_adjust_method: str | None = None
    n_comparisons: int = 1


class PostHocSummary(BaseModel):
    """What a post-hoc contributed, without entering the adjustment."""

    model_config = ConfigDict(frozen=True)

    entry_id: str
    function: str
    method: str
    p_adjust_method: str | None
    n_comparisons: int
    smallest_p_adjusted: float | None


class TestLedger(BaseModel):
    """Every test in the session, with adjusted p-values kept current."""

    model_config = ConfigDict(frozen=False)

    entries: list[LedgerEntry] = Field(default_factory=list)
    method: AdjustMethod = "holm"

    # ---- recording ------------------------------------------------------

    def _next_id(self) -> str:
        return f"t{len(self.entries)}"

    def record(
        self,
        result: TestResult,
        *,
        family: str = DEFAULT_FAMILY,
        branch: str = "main",
        step_id: str | None = None,
    ) -> LedgerEntry:
        """Record one omnibus or pairwise `TestResult` as a single entry.

        A test function that returns several results (two-way ANOVA's one
        per term, `correlation_matrix`'s one per pair) is recorded by
        calling this once per result: each is a separate conclusion the
        session is drawing, so each counts.
        """
        entry = LedgerEntry(
            entry_id=self._next_id(),
            fact_id=result.fact_id,
            function=result.function,
            family=family,
            branch=branch,
            step_id=step_id,
            p_value=result.p_value,
            estimand=result.estimand,
            counts_toward_session=True,
        )
        self.entries.append(entry)
        self._readjust()
        return self.by_id(entry.entry_id)

    def record_posthoc(
        self,
        result: PostHocResult,
        *,
        family: str = DEFAULT_FAMILY,
        branch: str = "main",
        step_id: str | None = None,
    ) -> LedgerEntry:
        """Record a post-hoc procedure as one entry that does *not* count.

        Section 12.3's rationale: its comparisons are already adjusted
        within their own family by the procedure itself, and re-adjusting
        them against unrelated session tests would correct twice.
        """
        entry = LedgerEntry(
            entry_id=self._next_id(),
            fact_id=result.fact_id,
            function=result.function,
            family=family,
            branch=branch,
            step_id=step_id,
            p_value=None,
            estimand=f"{result.method} post-hoc comparisons",
            counts_toward_session=False,
            internal_adjust_method=result.p_adjust_method,
            n_comparisons=len(result.comparisons),
        )
        self.entries.append(entry)
        self._readjust()
        return self.by_id(entry.entry_id)

    # ---- adjustment -----------------------------------------------------

    def _counting(self) -> list[LedgerEntry]:
        return [e for e in self.entries if e.counts_toward_session and e.p_value is not None]

    def _readjust(self) -> None:
        """Recompute session-wide and per-family adjusted p-values.

        Section 12.3: "recomputed for the whole session (and per family)
        after each new test". Recomputed rather than incrementally updated,
        because a step-down method like Holm depends on the whole set --
        adding a test changes the adjusted p-value of every earlier one.
        """
        counting = self._counting()
        session_adjusted = self._adjust([e.p_value for e in counting])  # type: ignore[misc]

        per_family: dict[str, list[float]] = {}
        for entry in counting:
            per_family.setdefault(entry.family, []).append(entry.p_value)  # type: ignore[arg-type]
        family_adjusted = {name: self._adjust(values) for name, values in per_family.items()}
        family_cursor = dict.fromkeys(family_adjusted, 0)

        updated: list[LedgerEntry] = []
        cursor = 0
        for entry in self.entries:
            if entry.counts_toward_session and entry.p_value is not None:
                index = family_cursor[entry.family]
                family_cursor[entry.family] = index + 1
                updated.append(
                    entry.model_copy(
                        update={
                            "p_adjusted": session_adjusted[cursor],
                            "p_adjusted_in_family": family_adjusted[entry.family][index],
                        }
                    )
                )
                cursor += 1
            else:
                updated.append(entry.model_copy(update={"p_adjusted": None}))
        self.entries = updated

    def _adjust(self, p_values: list[float]) -> list[float]:
        if not p_values:
            return []
        return [float(p) for p in adjust_pvalues(p_values, method=self.method)]

    def set_method(self, method: AdjustMethod) -> None:
        """Change the adjustment method and recompute (Section 12.3: "using
        the active persona method or the user's session setting")."""
        self.method = method
        self._readjust()

    # ---- reading --------------------------------------------------------

    @property
    def n_tests(self) -> int:
        """The session test count shown on every card. Post-hoc procedures
        are excluded (Section 12.3)."""
        return len(self._counting())

    def by_id(self, entry_id: str) -> LedgerEntry:
        for entry in self.entries:
            if entry.entry_id == entry_id:
                return entry
        raise KeyError(f"no ledger entry '{entry_id}'")

    def families(self) -> list[str]:
        return sorted({e.family for e in self.entries})

    def for_family(self, family: str) -> list[LedgerEntry]:
        return [e for e in self.entries if e.family == family]

    def rows(self) -> list[dict[str, Any]]:
        """`session.ledger()`'s table: every test with raw and adjusted p."""
        return [
            {
                "entry_id": e.entry_id,
                "function": e.function,
                "family": e.family,
                "estimand": e.estimand,
                "p_value": e.p_value,
                "p_adjusted": e.p_adjusted,
                "p_adjusted_in_family": e.p_adjusted_in_family,
                "counts": e.counts_toward_session,
            }
            for e in self.entries
        ]

    def posthoc_summaries(self) -> list[PostHocSummary]:
        return [
            PostHocSummary(
                entry_id=e.entry_id,
                function=e.function,
                method=e.estimand,
                p_adjust_method=e.internal_adjust_method,
                n_comparisons=e.n_comparisons,
                smallest_p_adjusted=None,
            )
            for e in self.entries
            if not e.counts_toward_session
        ]

    def __len__(self) -> int:
        return len(self.entries)


def is_posthoc(function: str) -> bool:
    """Whether a registered function is a post-hoc procedure.

    Checks the registry's own `kind` first and falls back to Section 12.3's
    list, so a new post-hoc registered with `kind="posthoc"` is handled
    without editing this module.
    """
    if function in POSTHOC_FUNCTIONS:
        return True
    try:
        return registry.get(function).kind == "posthoc"
    except KeyError:
        return False
