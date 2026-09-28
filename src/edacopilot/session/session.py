"""The session object and persistence (ARCHITECTURE.md, Section 12.4).

A `Session` ties the three pieces of Section 12 together — the dataset
version DAG, the provenance log, the test ledger — and makes them survive a
kernel restart, which is not a nicety: a Jupyter session that loses its
history on restart loses the audit trail that every result depends on.

`.edacopilot/<session_id>/` holds `session.json` plus one parquet per
version. **That directory can contain the user's data**, which is why
`edacopilot.session.GITIGNORE_WARNING` exists and the README says so: a
session directory committed to a shared repository is a data leak with no
warning attached.

The eligibility engine's `dataset_version` hook is wired here. Section 7.2
caches assumption checks per data version, and the version id it caches
under is now the DatasetStore's own — so a transform that produces `v3`
cannot read a fact computed about `v2`. That is the correctness property
the cache exists for, and `candidates()` is the only place it is
established.
"""

from __future__ import annotations

import inspect
import json
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pandas as pd

from edacopilot.eligibility import CandidateSet, CheckCache, QuestionSpec, select_candidates
from edacopilot.orchestrator.cards import Card
from edacopilot.orchestrator.loop import Orchestrator, TurnState
from edacopilot.personas import PersonaVerdict, propose_with_rationales
from edacore.contracts import Candidate, Eligibility, PostHocResult, TestResult
from edacore.registry import registry

from .dataset_store import DatasetStore, VersionRecord
from .ledger import AdjustMethod, LedgerEntry, TestLedger, is_posthoc
from .plots import PlotStore
from .provenance import ProposalView, ProvenanceLog, Step

if TYPE_CHECKING:
    # Not a module-level import: edacopilot.llm's __init__ imports
    # llm.fallback, which imports edacopilot.orchestrator.intents, which
    # this module's own orchestrator.loop import must reach *first* for
    # that chain to finish cleanly. A top-level `from edacopilot.llm...`
    # here, ahead or behind, either races that or fights ruff's import
    # sort into recreating the race -- so the real import is lazy, inside
    # the `llm` property below, same as `Orchestrator.module()` does for
    # its own cycle.
    from edacopilot.llm.client import LLMClient

SESSION_DIR = ".edacopilot"
STATE_FILE = "session.json"
STATE_VERSION = 3

GITIGNORE_WARNING = (
    f"{SESSION_DIR}/ stores your dataset as parquet alongside the session history. "
    f"Add it to .gitignore before committing anything from this project."
)


class SessionNotFoundError(FileNotFoundError):
    pass


@dataclass
class Session:
    """One analysis session (Section 12)."""

    session_id: str
    store: DatasetStore
    provenance: ProvenanceLog = field(default_factory=ProvenanceLog)
    test_ledger: TestLedger = field(default_factory=TestLedger)
    config: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    _caches: dict[str, CheckCache] = field(default_factory=dict, repr=False)
    plot_store: PlotStore | None = None
    _turn_state: TurnState = field(default_factory=TurnState, repr=False)
    _orchestrator: Orchestrator | None = field(default=None, repr=False)
    # Section 10.4's "last 6 conversational turns (text only)", for the
    # ContextBuilder (M9). Transient like `TurnState`'s own scratch fields
    # (see that class's docstring): a resumed session re-asks rather than
    # having the LLM reason about turns from a context the user no longer
    # has, so this is never persisted to session.json.
    _turns: deque[tuple[str, str]] = field(default_factory=lambda: deque(maxlen=6), repr=False)
    # Section 11: "keep a local copy of every outgoing prompt ... this log
    # never leaves the machine." In-memory only, for the same reason.
    _prompt_log: list[dict[str, Any]] = field(default_factory=list, repr=False)
    _llm_client: LLMClient | None = field(default=None, repr=False)

    # ---- lifecycle ------------------------------------------------------

    @classmethod
    def start(
        cls,
        df: pd.DataFrame,
        *,
        session_id: str | None = None,
        root: Path | str = SESSION_DIR,
        config: dict[str, Any] | None = None,
    ) -> Session:
        session_id = session_id or uuid4().hex[:12]
        directory = Path(root) / session_id
        if (directory / STATE_FILE).exists():
            raise FileExistsError(
                f"session '{session_id}' already exists at {directory}; use resume() to reopen it"
            )
        session = cls(
            session_id=session_id,
            store=DatasetStore.create(df, directory),
            config=dict(config or {}),
        )
        session.save()
        return session

    @property
    def directory(self) -> Path:
        return self.store.directory

    @property
    def plots(self) -> PlotStore:
        """Where this session's rendered plots live (Section 6.11).

        Created lazily rather than in `__post_init__` so a resumed session
        can hand in the records it read from disk.
        """
        if self.plot_store is None:
            self.plot_store = PlotStore(directory=self.directory)
        return self.plot_store

    def plot(self, function: str, /, **params: Any) -> str:
        """Render a registered plot and return its `plot_ref`.

        The two halves of Section 6.11 meet here: `edacore.viz` draws a
        `Figure` from the data, and the session writes the PNG and mints
        the id. Keyed on the dataset version, so a plot of `v1` is never
        served for a question about `v2`.
        """
        spec = registry.get(function)
        if spec.kind != "viz":
            raise ValueError(f"'{function}' is not a plot function (kind={spec.kind!r})")
        figure = spec.func(self.data, **params) if spec.takes_df else spec.func(**params)
        before = len(self.plots)
        plot_ref = self.plots.save(
            figure, function=function, params=params, version_id=self.store.head
        )
        if len(self.plots) != before:
            # Persisted immediately: the PNG is already on disk, and a
            # session.json that did not know about it would leave an
            # orphan file and a card referring to a ref that resume()
            # cannot resolve.
            self.save()
        return plot_ref

    @property
    def state_path(self) -> Path:
        return self.directory / STATE_FILE

    # ---- data -----------------------------------------------------------

    @property
    def data(self) -> pd.DataFrame:
        """The active head's data."""
        return self.store.get()

    @property
    def version(self) -> str:
        return self.store.head

    @property
    def active_branch(self) -> str:
        return self.store.active_branch

    def original(self) -> pd.DataFrame:
        return self.store.get("v0")

    # ---- the eligibility hook -------------------------------------------

    def check_cache(self, version_id: str | None = None) -> CheckCache:
        """The assumption-check cache for one dataset version.

        One cache per version, keyed by the DatasetStore's own version id.
        A new version therefore gets a new cache and cannot inherit a fact
        computed about its parent -- which is the whole point of Section
        7.2's "cached per data version".
        """
        version_id = version_id or self.store.head
        if version_id not in self._caches:
            self._caches[version_id] = CheckCache(
                self.store.get(version_id), dataset_version=version_id
            )
        return self._caches[version_id]

    def candidates(self, spec: QuestionSpec) -> CandidateSet:
        """Eligible methods for a question, against the active head."""
        version_id = self.store.head
        return select_candidates(
            spec,
            self.store.get(version_id),
            dataset_version=version_id,
            cache=self.check_cache(version_id),
        )

    def propose(self, spec: QuestionSpec) -> PersonaVerdict:
        return propose_with_rationales(self.candidates(spec))

    # ---- accepting a step ----------------------------------------------

    def record_step(
        self,
        *,
        stage: str,
        chosen: Candidate | None = None,
        chosen_via: str = "consensus",
        spec: QuestionSpec | None = None,
        candidates: CandidateSet | None = None,
        verdict: PersonaVerdict | None = None,
        override_reason: str | None = None,
        new_data: pd.DataFrame | None = None,
        result: TestResult | PostHocResult | None = None,
        family: str | None = None,
        label: str = "",
        code_preamble: str = "",
        df_var: str = "df",
    ) -> Step:
        """Record an accepted step, and any new dataset version or test
        result that came with it (Section 12.2).

        `code` is rendered here rather than passed in, from the registry, so
        the log's reproducibility claim cannot drift from what the function
        actually is (rule 7).
        """
        if (
            chosen_via == "override"
            and chosen is not None
            and chosen.eligibility is not Eligibility.ELIGIBLE
            and not (override_reason or "").strip()
        ):
            raise ValueError(
                f"overriding to '{chosen.function}' needs a typed reason: it is "
                f"{chosen.eligibility.value}, not eligible (Section 7.5). The reason is "
                f"stored in the provenance log and emitted as a code comment on export"
            )

        step_id = self.provenance.next_step_id()
        input_version = self.store.head
        output_version: str | None = None
        if new_data is not None:
            record = self.store.add_version(new_data, created_by_step=step_id, label=label or stage)
            output_version = record.version_id

        entry: LedgerEntry | None = None
        if result is not None:
            entry = self._record_result(result, family=family, step_id=step_id)

        step = Step(
            step_id=step_id,
            stage=stage,
            branch=self.store.active_branch,
            question=spec,
            checks=[check.fact_id for check in (candidates.checks if candidates else [])],
            proposals=_proposal_views(verdict),
            chosen=chosen,
            chosen_via=chosen_via,
            override_reason=override_reason,
            input_version=input_version,
            output_version=output_version,
            result_fact_id=entry.fact_id if entry else None,
            code=_render_code(chosen, override_reason, code_preamble, df_var),
        )
        self.provenance.append(step)
        self.save()
        return step

    def _record_result(
        self,
        result: TestResult | PostHocResult,
        *,
        family: str | None,
        step_id: str,
    ) -> LedgerEntry:
        family_label = family or "session"
        if isinstance(result, PostHocResult) or is_posthoc(result.function):
            return self.test_ledger.record_posthoc(
                result,  # type: ignore[arg-type]
                family=family_label,
                branch=self.store.active_branch,
                step_id=step_id,
            )
        return self.test_ledger.record(
            result, family=family_label, branch=self.store.active_branch, step_id=step_id
        )

    # ---- the conversational API (Section 9) ------------------------------
    #
    # Every method here returns a `Card`: what the user would see. That is
    # the whole surface M8's panel will drive, so a golden transcript in
    # `tests/conversations/` exercises the same loop a button click will,
    # and the UI can only introduce UI bugs.
    #
    # Rule 3 is the property to preserve when adding to this list: only
    # `accept` and `override` may reach `record_step`. Everything else
    # looks, asks or navigates.

    @property
    def orchestrator(self) -> Orchestrator:
        """The turn loop for this session, created on first use."""
        if self._orchestrator is None:
            self._orchestrator = Orchestrator(self, self._turn_state)
        return self._orchestrator

    @property
    def stage(self) -> str:
        return self.orchestrator.stage.value

    def ask(self, text: str | None = None, /, **fields: Any) -> Card:
        """Pose a question. Proposes; never runs (rule 3)."""
        return self.orchestrator.ask(text, **fields)

    def answer(self, **fields: Any) -> Card:
        """Answer the clarification the last card asked for."""
        return self.orchestrator.answer(**fields)

    def accept(self, persona: str | None = None) -> Card:
        """Run the proposal on screen, optionally naming whose."""
        return self.orchestrator.accept(persona)

    def override(
        self,
        candidate: str | Candidate | None = None,
        reason: str | None = None,
        *,
        confirm: bool = False,
    ) -> Card:
        """Run a method the personas did not propose (Section 7.5)."""
        return self.orchestrator.override(candidate, reason, confirm=confirm)

    def modify(self, **changes: Any) -> Card:
        """Change the question (alpha, design, a column) and re-propose."""
        return self.orchestrator.modify(**changes)

    def explain(self, topic: str) -> Card:
        """Explain a term, or why a method is or is not eligible here."""
        return self.orchestrator.explain(topic)

    def show_diagnostic_plot(self, fact_id: str) -> Card:
        """Render the plot for a passing check the card didn't draw (m8.1)."""
        return self.orchestrator.show_diagnostic_plot(fact_id)

    def goto_stage(self, name: str) -> Card:
        """Jump to a stage, with Section 9.2's warning if it skips one."""
        return self.orchestrator.goto_stage(name)

    def skip(self) -> Card:
        """Move to the next suggested stage without doing anything here."""
        return self.orchestrator.skip()

    def undo(self) -> Card:
        """Move the branch head back one version (Section 12.1)."""
        return self.orchestrator.undo()

    def branch(self, name: str, *, version: str | None = None) -> Card:
        """Start a new branch from the current head, or from `version`."""
        return self.orchestrator.branch(name, version=version)

    def switch_branch(self, name: str) -> Card:
        """Make another branch active."""
        return self.orchestrator.switch_branch(name)

    # ---- version navigation --------------------------------------------

    def branch_from(self, name: str, *, version: str | None = None) -> VersionRecord:
        record = self.store.branch(name, from_version=version)
        self.save()
        return record

    def switch_to_branch(self, name: str) -> VersionRecord:
        record = self.store.switch_branch(name)
        self.save()
        return record

    @property
    def branches(self) -> list[str]:
        return self.store.branches

    # ---- reading --------------------------------------------------------

    def ledger(self) -> list[dict[str, Any]]:
        """Section 12.3's table: every test with raw and adjusted p-values."""
        return self.test_ledger.rows()

    def steps(self) -> list[dict[str, Any]]:
        return [step.summary() for step in self.provenance.steps]

    def code(self) -> str:
        """The session as a runnable script (rule 7)."""
        return self.provenance.code()

    # ---- LLM layer plumbing (Sections 10.4, 11) --------------------------

    def record_turn(self, role: str, text: str) -> None:
        """Append to the last-6-turns window the `ContextBuilder` reads.

        Blank text is not a turn -- most button clicks carry no free text,
        and an empty entry would just be padding the window with nothing.
        """
        if text.strip():
            self._turns.append((role, text.strip()))

    def recent_turns(self) -> list[tuple[str, str]]:
        """Section 10.4's "last 6 conversational turns (text only)"."""
        return list(self._turns)

    def log_prompt(self, call: str, messages: list[dict[str, Any]], **meta: Any) -> None:
        """Keep a local copy of an outgoing prompt (Section 11).

        In-memory only, per-session, never written to `session.json` and
        never sent anywhere else -- `llm_audit()` is the only reader.
        """
        self._prompt_log.append({"call": call, "messages": messages, **meta})

    def llm_audit(self) -> list[dict[str, Any]]:
        """Every prompt this session has sent, for the user to inspect."""
        return list(self._prompt_log)

    @property
    def llm(self) -> LLMClient:
        """This session's `LLMClient`, built once from `self.config`.

        `self.config` is the same plain dict `start(..., config={...})`
        already merges (Section 10.2): an `"llm"` and/or `"privacy"` key in
        it override `edacopilot.toml`'s values exactly the way the rest of
        `config` already overrides other defaults.
        """
        if self._llm_client is None:
            from edacopilot.llm.client import LLMClient
            from edacopilot.llm.config import load_config

            overrides = {
                "llm": self.config.get("llm", {}),
                "privacy": self.config.get("privacy", {}),
            }
            self._llm_client = LLMClient(load_config(overrides).llm)
        return self._llm_client

    def set_adjust_method(self, method: AdjustMethod) -> None:
        self.test_ledger.set_method(method)
        self.save()

    # ---- persistence (Section 12.4) -------------------------------------

    def to_state(self) -> dict[str, Any]:
        return {
            "state_version": STATE_VERSION,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "config": self.config,
            "store": self.store.to_state(),
            "provenance": self.provenance.model_dump(mode="json"),
            "ledger": self.test_ledger.model_dump(mode="json"),
            # Which stage we are in, and which have been visited. Durable
            # because Section 9.2's warning ("missing values haven't been
            # reviewed") must survive a kernel restart, or it becomes a
            # lie. The transient half -- a proposal on screen, a question
            # awaiting an answer -- is deliberately not persisted: a
            # resumed session re-asks rather than accepting into a context
            # the user no longer has in front of them.
            "orchestrator": self._turn_state.to_state(),
            "plots": self.plots.to_state(),
        }

    def save(self) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        # Written via a temporary file and replaced atomically: a kernel
        # that dies mid-write would otherwise leave a truncated session.json
        # and lose the whole history rather than the last step.
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.to_state(), indent=2), encoding="utf-8")
        tmp.replace(self.state_path)
        return self.state_path

    @classmethod
    def resume(cls, session_id: str, *, root: Path | str = SESSION_DIR) -> Session:
        """Restore a session after a kernel restart (Section 12.4)."""
        directory = Path(root) / session_id
        state_path = directory / STATE_FILE
        if not state_path.exists():
            raise SessionNotFoundError(f"no session state at {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("state_version") != STATE_VERSION:
            raise ValueError(
                f"session '{session_id}' was written by state version "
                f"{state.get('state_version')}; this build reads {STATE_VERSION}"
            )

        store_state = state["store"]
        store = DatasetStore.restore(
            directory=directory,
            versions={
                record["version_id"]: VersionRecord.model_validate(record)
                for record in store_state["versions"]
            },
            branch_heads=store_state["branch_heads"],
            active_branch=store_state["active_branch"],
        )
        missing = [
            version_id for version_id in store.versions if not store.path_for(version_id).exists()
        ]
        if missing:
            raise SessionNotFoundError(
                f"session '{session_id}' is missing parquet file(s) for {missing}; the "
                f"directory has been moved or partly deleted"
            )
        return cls(
            session_id=state["session_id"],
            store=store,
            provenance=ProvenanceLog.model_validate(state["provenance"]),
            test_ledger=TestLedger.model_validate(state["ledger"]),
            config=state.get("config", {}),
            created_at=datetime.fromisoformat(state["created_at"]),
            _turn_state=TurnState.from_state(state.get("orchestrator", {})),
            plot_store=PlotStore.from_state(directory, state.get("plots", [])),
        )


def _proposal_views(verdict: PersonaVerdict | None) -> list[ProposalView]:
    if verdict is None:
        return []
    return [
        ProposalView(
            persona=pick.persona,
            function=pick.function,
            eligibility=pick.eligibility.value if pick.eligibility else None,
            rationale=pick.rationale,
            concurs_with=pick.concurs_with,
        )
        for pick in verdict.picks
    ]


def _render_code(
    chosen: Candidate | None,
    override_reason: str | None = None,
    preamble: str = "",
    df_var: str = "df",
) -> str:
    """The runnable line for a step, from the registry (rule 7).

    A `Candidate` carries only the params that identify the analysis -- the
    columns and the question's own values -- because that is what the
    eligibility engine and the personas reason about. A code template also
    references the parameters that have defaults (`ci`, `nan_policy`,
    `random_state`), so those are filled in from the function's own
    signature. Rendering them explicitly rather than relying on defaults is
    deliberate: the exported notebook should say what it ran, including the
    seed, so someone reading it later can reproduce the number without
    knowing this version's defaults.

    A template that still cannot be rendered becomes a comment rather than
    an exception. A step that ran is a fact, and losing the whole log
    because one line would not render would be worse than recording that it
    did not.
    """
    if chosen is None:
        return ""
    try:
        params = {**_default_params(chosen.function), **dict(chosen.params)}
        call = registry.to_code(chosen.function, params, df_var=df_var)
    except Exception as exc:  # noqa: BLE001 - see docstring
        call = f"# could not render code for {chosen.function}: {exc}"
    if preamble:
        # Some methods run against a reshaped frame (a paired test needs the
        # two measurements side by side). Rule 7 says the recorded step must
        # be *runnable*, so the reshape is part of the step rather than
        # something the exported notebook would be missing.
        call = f"{preamble}\n{call}"
    reason = (override_reason or "").strip()
    if not reason:
        return call
    # Section 7.5: the reason is emitted as a code comment on export, so
    # someone reading the notebook later sees why a flagged method was used
    # at the point where it was used.
    return f"# Override: {reason}\n{call}"


def _default_params(function: str) -> dict[str, Any]:
    """Every parameter of a registered function that has a default."""
    signature = inspect.signature(registry.get(function).func)
    return {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }


def resume(session_id: str, *, root: Path | str = SESSION_DIR) -> Session:
    """Module-level alias, so `edacopilot.resume(...)` reads as Section 12.4
    writes it."""
    return Session.resume(session_id, root=root)


def list_sessions(root: Path | str = SESSION_DIR) -> list[str]:
    base = Path(root)
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if (p / STATE_FILE).exists())
