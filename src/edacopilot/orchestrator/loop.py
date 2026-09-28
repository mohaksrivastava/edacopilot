"""The turn loop (ARCHITECTURE.md, Sections 3.2 and 9).

Section 3.2's diagram, implemented literally. A turn is: intent in, route to
a stage, build a spec, run diagnostics, rank candidates, ask the personas,
render a card — and then **wait**. The wait is not a detail of the UI; it is
rule 3, and it is the only reason the system can claim not to be an
autonomous analyst.

So the whole of this module is arranged around one property: *no code path
from `ask`, `explain`, `goto_stage`, `skip`, `undo`, `branch` or
`switch_branch` reaches `Session.record_step`*. Only `accept` and
`override` do, and both are reached only by the user saying so. A property
test drives random sequences of the first group and asserts the provenance
log stays empty.

Section 7.5's friction lives here too, in the one place it can be enforced:

- An **ELIGIBLE** candidate the user prefers needs nothing. The engine
  already ruled it valid; choosing among valid methods is a preference, not
  an override of anything.
- A **CAVEAT** candidate the chosen persona did not pick needs a typed
  reason.
- An **INELIGIBLE** candidate additionally shows the hard assumption that
  failed and what that failure does to the result, and needs a second,
  separate confirmation.

The reason is stored in `ProvenanceLog` and emitted as a comment above the
call in `Step.code`, so it survives into the exported notebook — which is
where someone reading the analysis later will actually see it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from edacopilot.eligibility import (
    AmbiguousSpecError,
    CandidateSet,
    ColumnMatchConfig,
    Design,
    Goal,
    InvalidSpecError,
    QuestionSpec,
    UnsupportedQuestionError,
    validate_spec,
)
from edacopilot.llm.fallback import (
    answer_free_question,
    build_question_spec_form,
    parse_intent,
)
from edacopilot.personas import PersonaVerdict, propose_with_rationales, render_card
from edacore.contracts import (
    AssumptionCheck,
    Candidate,
    CheckStatus,
    Eligibility,
    PostHocResult,
    TestResult,
)
from edacore.registry import registry

from .cards import ActionButton, Card, ProposalView, Suggestion
from .diagnostic_plots import illustrate, plot_for_check
from .intents import Intent, IntentType, button
from .state_machine import Stage, jump_warning, next_stage, parse_stage

if TYPE_CHECKING:  # pragma: no cover
    from edacopilot.session.session import Session
    from edacopilot.stages.base import BaseStage
    from edacopilot.stages.hypothesis import HypothesisStage


class OverrideRequired(RuntimeError):
    """Raised when an override is executed without what Section 7.5 demands.

    The Python API returns an `override_confirm` card rather than raising,
    because a card is what a UI needs. This exists for the internal path
    that must not be reachable without both pieces, so a bug there fails
    loudly instead of running an ineligible method.
    """


@dataclass
class Pending:
    """A proposal on screen, waiting for the user (Section 3.2, step 8)."""

    spec: QuestionSpec
    candidates: CandidateSet
    verdict: PersonaVerdict
    kind: str = "test"  # "test" or "post_hoc"
    # For a post-hoc: the single candidate, which no family produced.
    candidate: Candidate | None = None
    licensed_by: str = ""


@dataclass
class OpenQuestion:
    """A clarification the orchestrator asked and is waiting on."""

    spec: QuestionSpec | None
    ambiguities: list[str]
    # "design", "form" or "intent": what kind of answer resolves it.
    kind: str = "design"
    options: list[str] = field(default_factory=list)


@dataclass
class LastResult:
    """The most recent executed test, for post-hoc and suggestion rules."""

    candidate: Candidate
    result: TestResult
    spec: QuestionSpec
    checks: list[AssumptionCheck]


@dataclass
class TurnState:
    """Everything the loop remembers between turns.

    The durable part — which stage we are in and which have been visited —
    is persisted with the session, because "have you looked at missing
    data?" must survive a kernel restart or the warning in Section 9.2
    becomes a lie. The transient part (a proposal on screen, a question
    awaiting an answer) is not persisted: a resumed session re-asks rather
    than silently accepting into a context the user no longer has.
    """

    stage: Stage = Stage.PROFILE
    visited: set[Stage] = field(default_factory=lambda: {Stage.INGEST})
    pending: Pending | None = None
    open_question: OpenQuestion | None = None
    pending_override: Candidate | None = None
    last_result: LastResult | None = None

    def to_state(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            # Sorted: a set's iteration order is not a function of its value
            # (see `Candidate.tags`), and this goes into session.json.
            "visited": sorted(stage.value for stage in self.visited),
        }

    @classmethod
    def from_state(cls, state: dict[str, Any]) -> TurnState:
        return cls(
            stage=Stage(state.get("stage", Stage.PROFILE.value)),
            visited={Stage(name) for name in state.get("visited", [Stage.INGEST.value])},
        )


class Orchestrator:
    """The only component that talks to the user (Section 3.3)."""

    def __init__(self, session: Session, state: TurnState | None = None) -> None:
        self.session = session
        self.state = state or TurnState()

    # ---- stage plumbing --------------------------------------------------

    @property
    def stage(self) -> Stage:
        return self.state.stage

    def module(self, stage: Stage | None = None) -> BaseStage:
        """The stage module for a stage.

        Imported inside the function because `edacopilot.stages` imports
        this package for `Stage` and `Card`, and a module-level import here
        would close the cycle.
        """
        from edacopilot.stages import get_stage

        return get_stage(stage or self.state.stage)

    def hypothesis(self) -> HypothesisStage:
        """The HYPOTHESIS stage, typed.

        Its flow needs more than the `StageModule` protocol declares --
        the post-hoc rules, the validity context, the result suggestions --
        and those belong to that stage rather than to every stage.
        """
        from edacopilot.stages.hypothesis import HypothesisStage as _HypothesisStage

        module = self.module(Stage.HYPOTHESIS)
        assert isinstance(module, _HypothesisStage)
        return module

    def _enter(self, stage: Stage) -> None:
        """Record the stage, and persist it.

        Saved rather than held in memory because Section 9.2's warning
        ("missing values haven't been reviewed") has to survive a kernel
        restart. A user who reviewed missingness yesterday being told they
        did not would make every warning worth less.
        """
        changed = stage is not self.state.stage or stage not in self.state.visited
        self.state.stage = stage
        self.state.visited.add(stage)
        if changed:
            self.session.save()

    # ---- the turn loop ---------------------------------------------------

    def handle(self, intent: Intent) -> Card:
        """One turn (Section 3.2). Routing only — the work is in the handlers."""
        text = str(intent.payload.get("text", ""))
        self.session.record_turn("user", text)
        card = self._route(intent)
        self.session.record_turn("assistant", card.title)
        return card

    def _route(self, intent: Intent) -> Card:
        if not intent.is_confident:
            return self._clarify(intent)

        handlers: dict[IntentType, Callable[[], Card]] = {
            IntentType.ASK_ANALYSIS: lambda: self._ask_intent(intent),
            IntentType.ANSWER_CLARIFICATION: lambda: self._answer_intent(intent),
            IntentType.ACCEPT: lambda: self.accept(intent.persona),
            IntentType.OVERRIDE: lambda: self.override(
                intent.payload.get("function"),
                reason=intent.payload.get("reason"),
                confirm=bool(intent.payload.get("confirm")),
            ),
            IntentType.MODIFY: lambda: self.modify(**_modify_kwargs(intent.payload)),
            IntentType.EXPLAIN: lambda: self.explain(
                str(intent.payload.get("topic") or intent.payload.get("text", ""))
            ),
            IntentType.GOTO_STAGE: lambda: self.goto_stage(intent.target_stage or ""),
            IntentType.UNDO: self.undo,
            IntentType.BRANCH: lambda: self.branch(str(intent.payload.get("name", "alternative"))),
            IntentType.SWITCH_BRANCH: lambda: self.switch_branch(
                str(intent.payload.get("name", "main"))
            ),
            IntentType.SKIP: self.skip,
            IntentType.SHOW: lambda: self._show(intent),
            IntentType.SHOW_DIAGNOSTIC_PLOT: lambda: self.show_diagnostic_plot(
                str(intent.payload.get("fact_id", ""))
            ),
            IntentType.SETTINGS: lambda: self._settings(intent),
            IntentType.EXPORT: lambda: self.goto_stage(Stage.EXPORT.value),
        }
        handler = handlers.get(intent.type)
        if handler is None:
            return self._clarify(intent)
        return handler()

    def _clarify(self, intent: Intent) -> Card:
        """Section 9.1: below the confidence floor, ask with 2-3 buttons."""
        text = str(intent.payload.get("text", "")).strip()
        quoted = f' "{text}"' if text else ""
        options = [
            (
                "Ask this as an analysis question",
                IntentType.ASK_ANALYSIS,
                "session.ask(goal=..., outcome=..., group=...)",
            ),
            ("Explain a term", IntentType.EXPLAIN, f'session.explain("{text or "..."}")'),
            ("Look at the variables", IntentType.GOTO_STAGE, 'session.goto_stage("explore")'),
        ]
        actions = [
            ActionButton(
                label=label,
                intent=button(kind, text=text, topic=text, target_stage="explore")
                if kind is IntentType.GOTO_STAGE
                else button(kind, text=text, topic=text),
                call=call,
            )
            for label, kind, call in options
        ]
        return Card(
            kind="question",
            title="I'm not sure what you meant",
            stage=self.state.stage.value,
            body_md=(
                f"With the LLM switched off, intent is matched by keyword rules "
                f"(Section 10.5), and nothing matched{quoted} confidently enough to act on. "
                f"Which of these did you mean?"
            ),
            actions=actions,
        )

    # ---- ask -------------------------------------------------------------

    def ask(self, text: str | None = None, /, **fields: Any) -> Card:
        """Pose a question (Section 3.2, steps 1-7).

        Three ways in, all landing in the same place:

        - a structured `QuestionSpec` (`spec=`),
        - keyword arguments (`goal=`, `outcome=`, `group=`, `design=`),
        - free text, which goes through the keyword intent parser and, if
          that cannot produce a spec, returns the form-style card Section
          10.5 calls for.
        """
        spec = fields.pop("spec", None)
        if isinstance(spec, QuestionSpec):
            return self._propose(spec)

        if fields:
            return self._propose(self._spec_from_fields(fields))

        if text is None:
            return self._form_card("Tell me what to compare")

        return self.handle(self._parse_intent(text))

    def _parse_intent(self, text: str) -> Intent:
        """Section 3.4's `parse_intent`: the LLM first, Section 10.5's
        keyword parser as the fallback. One of the two calls Section
        15.5 names as where a golden transcript's mocked LLM slots in."""
        from edacopilot.llm.calls import parse_intent as parse_intent_llm
        from edacopilot.llm.client import LLMFallbackRequired

        awaiting_answer = self.state.open_question is not None
        try:
            return parse_intent_llm(text, self.session, awaiting_answer=awaiting_answer)
        except LLMFallbackRequired:
            return parse_intent(text, awaiting_answer=awaiting_answer)

    def _ask_intent(self, intent: Intent) -> Card:
        if intent.payload.get("post_hoc"):
            return self._propose_post_hoc()
        fields = {
            key: value
            for key, value in intent.payload.items()
            if key not in ("text", "post_hoc") and value is not None
        }
        if fields:
            return self._propose(self._spec_from_fields(fields))

        text = str(intent.payload.get("text", ""))
        spec = self._build_question_spec(text) if text else None
        if spec is not None:
            return self._propose(spec)
        return self._form_card(
            "I can see this is an analysis question, but not which columns it is about"
        )

    def _build_question_spec(self, text: str) -> QuestionSpec | None:
        """Section 3.4's `build_question_spec`. `None` means no LLM
        answered it -- the caller falls back to the form card, Section
        10.5's own fallback for this call (a spec built without a model
        would be a guess, so the fallback shows fields instead of one)."""
        from edacopilot.llm.calls import build_question_spec
        from edacopilot.llm.client import LLMFallbackRequired

        try:
            return build_question_spec(text, self.session)
        except LLMFallbackRequired:
            return None

    def _spec_from_fields(self, fields: dict[str, Any]) -> QuestionSpec:
        stage = self.hypothesis()
        payload = dict(fields)
        payload.setdefault("goal", Goal.COMPARE_GROUPS.value)
        payload.setdefault("design", Design.UNKNOWN.value)
        return stage.build_spec(Intent(type=IntentType.ASK_ANALYSIS, payload=payload), self.session)

    def _propose(self, spec: QuestionSpec) -> Card:
        """Steps 3-7 of Section 3.2: validate, diagnose, rank, propose."""
        self._enter(Stage.HYPOTHESIS)
        self.state.pending = None
        self.state.pending_override = None

        try:
            validated = validate_spec(spec, self.session.data, self._column_matching_config())
        except InvalidSpecError as exc:
            return Card(
                kind="warning",
                title="That question cannot be built",
                stage=self.state.stage.value,
                body_md=(
                    f"{exc}\n\nThis is not something you can answer by clarifying — the "
                    f"question names something the data does not have."
                ),
            )

        if validated.ambiguities:
            # Section 7.1 step 3, and the invariant that matters most here:
            # this returns *before* any persona is consulted. A column
            # ambiguity (Section 7.1's fuzzy-match step) renders candidate
            # buttons; a design ambiguity renders the paired/independent
            # choice -- `column_ambiguity_candidates` tells the two apart
            # (validate_spec never mixes them in one call).
            kind = "column" if validated.column_ambiguity_candidates else "design"
            self.state.open_question = OpenQuestion(
                spec=validated, ambiguities=list(validated.ambiguities), kind=kind
            )
            return self._question_card(validated)

        self.state.open_question = None
        try:
            candidates = self.session.candidates(validated)
        except UnsupportedQuestionError as exc:
            return Card(
                kind="info",
                title="Available in a later version",
                stage=self.state.stage.value,
                body_md=str(exc),
            )
        except AmbiguousSpecError as exc:  # pragma: no cover - defensive
            self.state.open_question = OpenQuestion(spec=validated, ambiguities=exc.ambiguities)
            return self._question_card(validated)

        verdict = propose_with_rationales(candidates)
        self.state.pending = Pending(spec=validated, candidates=candidates, verdict=verdict)
        card = self._proposal_card(self.state.pending)
        return _with_corrections(card, validated)

    def _column_matching_config(self) -> ColumnMatchConfig:
        """Section 10.2's `[column_matching]` table, read the same
        ad-hoc way `plot_all_diagnostics` already is -- a plain dict on
        `session.config`, no dedicated settings object for something this
        small."""
        overrides = self.session.config.get("column_matching", {})
        return ColumnMatchConfig(**overrides)

    def _question_card(self, spec: QuestionSpec) -> Card:
        """Section 7.1's ambiguities, as the question that blocks the turn.

        Two distinct kinds share this one card: a design ambiguity offers
        the paired/independent choice; a column ambiguity (the fuzzy-match
        step) offers one button per candidate column, keyed off
        `column_ambiguity_candidates` set by `validate_spec`.
        """
        if spec.column_ambiguity_candidates:
            return self._column_question_card(spec)

        lines = [
            "Before any method can be proposed, this needs an answer. "
            "The design is the single most consequential choice in a group comparison: "
            "it decides which tests are valid at all.",
            "",
        ]
        lines += [f"- {question}" for question in spec.ambiguities]
        actions = [
            ActionButton(
                label="They are paired / repeated measurements",
                intent=button(IntentType.ANSWER_CLARIFICATION, design=Design.PAIRED.value),
                call='session.answer(design="paired")',
            ),
            ActionButton(
                label="They are independent groups",
                intent=button(IntentType.ANSWER_CLARIFICATION, design=Design.INDEPENDENT.value),
                call='session.answer(design="independent")',
            ),
            ActionButton(
                label="Explain the difference",
                intent=button(IntentType.EXPLAIN, topic="paired"),
                call='session.explain("paired")',
            ),
        ]
        return Card(
            kind="question",
            title="One question first",
            stage=self.state.stage.value,
            body_md="\n".join(lines),
            actions=actions,
        )

    def _column_question_card(self, spec: QuestionSpec) -> Card:
        """A named column had no confident match: one button per candidate,
        per role. Picking one answers the same way a design choice does --
        `_answer_intent` already sets `spec.variables[role]` from a
        keyword payload naming that role, so a candidate button is just
        `button(ANSWER_CLARIFICATION, **{role: candidate})`, nothing new."""
        lines = [
            "Before any method can be proposed, this needs an answer.",
            "",
        ]
        lines += [f"- {question}" for question in spec.ambiguities]
        actions = [
            ActionButton(
                label=f"{role}: use `{candidate}`",
                intent=button(IntentType.ANSWER_CLARIFICATION, **{role: candidate}),
                call=f'session.answer({role}="{candidate}")',
            )
            for role, candidates in spec.column_ambiguity_candidates.items()
            for candidate in candidates
        ]
        return Card(
            kind="question",
            title="Which column did you mean?",
            stage=self.state.stage.value,
            body_md="\n".join(lines),
            actions=actions,
        )

    def _form_card(self, title: str) -> Card:
        """Section 10.5's form-style fallback for `build_question_spec`."""
        fields = build_question_spec_form(self.session.data)
        lines = [
            "With the LLM switched off there is no honest way to turn a sentence into a "
            "question spec, so rather than guess, here are the fields to fill in "
            "(Section 10.5).",
            "",
        ]
        for spec_field in fields:
            options = ", ".join(f"`{option}`" for option in spec_field.options)
            required = "" if spec_field.required else " *(optional)*"
            lines.append(f"- **{spec_field.name}**{required} — {spec_field.prompt}")
            if options:
                lines.append(f"  - one of: {options}")
        lines += [
            "",
            "For example:",
            "",
            "```python",
            'session.ask(goal="compare_groups", outcome="income", group="gender",',
            '            design="independent")',
            "```",
        ]
        return Card(
            kind="question",
            title=title,
            stage=self.state.stage.value,
            body_md="\n".join(lines),
            actions=[
                ActionButton(
                    label="Look at the variables first",
                    intent=button(IntentType.GOTO_STAGE, target_stage=Stage.EXPLORE.value),
                    call='session.goto_stage("explore")',
                )
            ],
        )

    # ---- answering a clarification ---------------------------------------

    def answer(self, **fields: Any) -> Card:
        """Answer the open question and resume the turn it blocked."""
        return self._answer_intent(button(IntentType.ANSWER_CLARIFICATION, **fields))

    def _answer_intent(self, intent: Intent) -> Card:
        question = self.state.open_question
        if question is None or question.spec is None:
            return Card(
                kind="info",
                title="Nothing to answer",
                stage=self.state.stage.value,
                body_md="There is no open question right now. Ask something first.",
            )

        payload = {k: v for k, v in intent.payload.items() if k != "text"}
        spec = question.spec
        updates: dict[str, Any] = {}
        confirmed = set(spec.confirmed_by_user)

        if design := payload.get("design"):
            updates["design"] = Design(design)
            confirmed.add("design")
        for role in ("outcome", "group", "x", "y", "variable", "subject"):
            if role in payload:
                updates.setdefault("variables", dict(spec.variables))[role] = payload[role]
                confirmed.add(role)
        if payload.get("confirmed") is True:
            # A bare "yes" confirms exactly what the question was about.
            confirmed.update(_roles_in_question(question.ambiguities, spec))
        for role in payload.get("confirm_roles", ()):
            confirmed.add(role)

        updates["confirmed_by_user"] = confirmed
        # The answer clears the old ambiguities; validate_spec recomputes them.
        updates["ambiguities"] = []
        self.state.open_question = None
        return self._propose(spec.model_copy(update=updates))

    # ---- accept ----------------------------------------------------------

    def accept(self, persona: str | None = None) -> Card:
        """Execute the proposal on screen. The only path to a step, with
        `override` (rule 3)."""
        pending = self.state.pending
        if pending is None:
            return Card(
                kind="info",
                title="Nothing to accept",
                stage=self.state.stage.value,
                body_md="There is no proposal on screen. Ask a question first.",
            )

        if pending.kind == "post_hoc":
            assert pending.candidate is not None
            return self._run_post_hoc(pending, pending.candidate)

        candidate, chosen_via, error = self._resolve_accept(pending, persona)
        if error is not None:
            return error
        assert candidate is not None
        return self._run(pending, candidate, chosen_via=chosen_via)

    def _resolve_accept(
        self, pending: Pending, persona: str | None
    ) -> tuple[Candidate | None, str, Card | None]:
        verdict = pending.verdict
        if persona is not None:
            pick = verdict.pick_for(persona)
            if pick.candidate is None:
                return (
                    None,
                    "",
                    Card(
                        kind="warning",
                        title=f"{pick.display_name} has nothing to propose here",
                        stage=self.state.stage.value,
                        body_md=pick.rationale,
                    ),
                )
            return pending.candidates.get(pick.candidate.function), persona, None

        if verdict.is_consensus:
            function = verdict.proposals[0].function
            if not function:
                return (
                    None,
                    "",
                    Card(
                        kind="warning",
                        title="No persona has a method to offer",
                        stage=self.state.stage.value,
                        body_md=render_card(verdict),
                    ),
                )
            return pending.candidates.get(function), "consensus", None

        # Section 1.3: show disagreement. The user picks; the system does not
        # pick for them by quietly defaulting to the anchor persona.
        names = " or ".join(
            f'`session.accept("{p.personas[0]}")` for {" + ".join(p.display_names)} ({p.function})'
            for p in verdict.proposals
        )
        return (
            None,
            "",
            Card(
                kind="warning",
                title="The personas disagree — which one?",
                stage=self.state.stage.value,
                body_md=(
                    f"{render_card(verdict)}\n\nName the persona whose proposal to run: {names}."
                ),
                proposals=_proposal_views(pending),
                # The same buttons the divergence card offered. A card that
                # says "choose one" and then offers no way to choose is the
                # kind of dead end that sends a user back to a raw t-test.
                actions=[
                    ActionButton(
                        label=f"{' + '.join(proposal.display_names)}: {proposal.function}",
                        intent=button(IntentType.ACCEPT, persona=proposal.personas[0]),
                        call=f'session.accept("{proposal.personas[0]}")',
                    )
                    for proposal in verdict.proposals
                    if proposal.function
                ],
            ),
        )

    # ---- override (Section 7.5) ------------------------------------------

    def override(
        self, candidate: str | Candidate | None, reason: str | None = None, *, confirm: bool = False
    ) -> Card:
        """Choose a method the personas did not propose (Section 7.5)."""
        pending = self.state.pending
        if pending is None:
            return Card(
                kind="info",
                title="Nothing to override",
                stage=self.state.stage.value,
                body_md="There is no proposal on screen. Ask a question first.",
            )
        if candidate is None:
            return Card(
                kind="warning",
                title="Which method?",
                stage=self.state.stage.value,
                body_md='Name the method to use, e.g. `session.override("student_t", reason=...)`.',
            )

        function = candidate if isinstance(candidate, str) else candidate.function
        try:
            chosen = pending.candidates.get(function)
        except KeyError:
            available = ", ".join(f"`{c.function}`" for c in pending.candidates.candidates)
            return Card(
                kind="warning",
                title=f"'{function}' is not a candidate for this question",
                stage=self.state.stage.value,
                body_md=(
                    f"The methods considered for `{pending.candidates.family}` are: {available}."
                ),
            )

        typed = (reason or "").strip()
        proposed = {pick.function for pick in pending.verdict.picks if pick.function}

        if chosen.eligibility is Eligibility.ELIGIBLE and function not in proposed:
            # Section 7.5 asks for friction on CAVEAT and INELIGIBLE only.
            # The engine already ruled this valid; preferring it is a choice
            # among valid methods, not an override of a judgement.
            return self._run(pending, chosen, chosen_via="override", reason=typed or None)

        if chosen.eligibility is Eligibility.INELIGIBLE:
            if not typed or not confirm:
                self.state.pending_override = chosen
                return self._ineligible_card(pending, chosen, typed)
            self.state.pending_override = None
            return self._run(pending, chosen, chosen_via="override", reason=typed)

        # CAVEAT, and not what the persona picked.
        if function in proposed:
            return self._run(pending, chosen, chosen_via="override", reason=typed or None)
        if not typed:
            return self._caveat_card(pending, chosen)
        return self._run(pending, chosen, chosen_via="override", reason=typed)

    def _caveat_card(self, pending: Pending, chosen: Candidate) -> Card:
        checks = {check.fact_id: check for check in pending.candidates.checks}
        failing = [checks[fact_id] for fact_id in chosen.reasons if fact_id in checks]
        lines = [
            f"`{chosen.function}` is **{chosen.eligibility.value}** here: it can be run, but "
            f"at least one assumption it makes is not comfortably met, and no persona chose "
            f"it. Section 7.5 asks for a typed reason, which is stored in the provenance log "
            f"and written into the exported code as a comment.",
            "",
        ]
        for check in failing:
            lines.append(f"- **{check.fact_id}** is {check.status.value}: {check.consequence}")
        lines += [
            "",
            f'```python\nsession.override("{chosen.function}", reason="...")\n```',
        ]
        return Card(
            kind="override_confirm",
            title=f"A reason is needed to use {chosen.function}",
            stage=self.state.stage.value,
            body_md="\n".join(lines),
            diagnostics=failing,
            actions=[
                ActionButton(
                    label=f"Use {chosen.function} with a reason",
                    intent=button(IntentType.OVERRIDE, function=chosen.function),
                    call=f'session.override("{chosen.function}", reason="...")',
                )
            ],
        )

    def _ineligible_card(self, pending: Pending, chosen: Candidate, typed: str) -> Card:
        """Section 7.5's second gate: the hard failure, its consequence, and
        a confirmation that is separate from the reason."""
        checks = {check.fact_id: check for check in pending.candidates.checks}
        hard = _hard_failures(chosen, checks)
        lines = [
            f"`{chosen.function}` is **ineligible** for this data. This is not a caveat: "
            f"a hard assumption has failed, so the number it returns will not mean what "
            f"the method claims it means.",
            "",
            "**What failed, and what it does to the result:**",
            "",
        ]
        for check in hard:
            lines.append(
                f"- **{check.assumption}** ({check.fact_id}) is {check.status.value.upper()}"
                + (f", {check.method} p={check.p_value:.4g}" if check.p_value is not None else "")
            )
            lines.append(f"  - {check.consequence}")
        lines += ["", "**Valid alternatives the engine found:**", ""]
        for alternative in pending.candidates.eligible[:3]:
            lines.append(f"- `{alternative.function}` — {alternative.estimand}")
        lines += ["", "To go ahead anyway, both of these are required:", ""]
        lines.append(
            f"1. a typed reason {'(given: ' + typed + ')' if typed else '(not yet given)'}"
        )
        lines.append("2. a second, separate confirmation")
        lines += [
            "",
            "```python",
            f'session.override("{chosen.function}", reason="...", confirm=True)',
            "```",
        ]
        return Card(
            kind="override_confirm",
            title=f"{chosen.function} is ineligible — confirm to run it anyway",
            stage=self.state.stage.value,
            body_md="\n".join(lines),
            diagnostics=hard,
            actions=[
                ActionButton(
                    label=f"Yes, run {chosen.function} anyway",
                    intent=button(
                        IntentType.OVERRIDE,
                        function=chosen.function,
                        reason=typed,
                        confirm=True,
                    ),
                    call=f'session.override("{chosen.function}", reason="...", confirm=True)',
                ),
                *(
                    [
                        # Names what `accept()` would actually run -- the
                        # persona proposal -- not merely the first eligible
                        # method, which need not be the same thing.
                        ActionButton(
                            label=f"Use {_proposed_function(pending)} instead",
                            intent=button(IntentType.ACCEPT),
                            call="session.accept()",
                        )
                    ]
                    if _proposed_function(pending)
                    else []
                ),
            ],
        )

    # ---- running ----------------------------------------------------------

    def _run(
        self,
        pending: Pending,
        candidate: Candidate,
        *,
        chosen_via: str,
        reason: str | None = None,
    ) -> Card:
        """Section 3.2, step 8: execute, record, render, suggest, wait."""
        if candidate.eligibility is Eligibility.INELIGIBLE and not (reason or "").strip():
            raise OverrideRequired(
                f"'{candidate.function}' is ineligible and reached execution without a "
                f"typed reason; Section 7.5 forbids that"
            )

        stage = self.hypothesis()
        outcome = stage.execute(candidate, self.session, pending.spec)
        result = outcome.result
        assert isinstance(result, TestResult)

        step = self.session.record_step(
            stage=Stage.HYPOTHESIS.value,
            chosen=candidate,
            chosen_via=chosen_via,
            spec=pending.spec,
            candidates=pending.candidates,
            verdict=pending.verdict,
            override_reason=reason,
            result=result,
            family=outcome.family,
            code_preamble=outcome.code_preamble,
            df_var=outcome.df_var,
        )

        entry = self.session.test_ledger.by_id(
            next(e.entry_id for e in self.session.test_ledger.entries if e.step_id == step.step_id)
        )
        annotated = stage.annotate(
            result.model_copy(update={"p_adjusted": entry.p_adjusted}),
            stage.validity_context(
                self.session,
                spec=pending.spec,
                checks=list(pending.candidates.checks),
                test_number=self.session.test_ledger.n_tests,
                p_adjusted=entry.p_adjusted,
                chosen_after_exploring=Stage.EXPLORE in self.state.visited,
            ),
        )

        self.state.pending = None
        self.state.last_result = LastResult(
            candidate=candidate,
            result=annotated,
            spec=pending.spec,
            checks=list(pending.candidates.checks),
        )

        suggestions = stage.result_suggestions(
            self.session,
            candidate=candidate,
            result=annotated,
            spec=pending.spec,
            checks=list(pending.candidates.checks),
            graded=list(pending.candidates.graded),
        )
        return self._result_card(candidate, annotated, step, reason, suggestions, outcome.note)

    def _run_post_hoc(self, pending: Pending, candidate: Candidate) -> Card:
        stage = self.hypothesis()
        outcome = stage.execute_post_hoc(candidate, self.session)
        assert isinstance(outcome.result, PostHocResult)
        step = self.session.record_step(
            stage=Stage.HYPOTHESIS.value,
            chosen=candidate,
            chosen_via="consensus",
            spec=pending.spec,
            candidates=pending.candidates,
            verdict=pending.verdict,
            result=outcome.result,
            family=outcome.family,
        )
        self.state.pending = None
        return Card(
            kind="result",
            title=f"{candidate.function} — pairwise comparisons",
            stage=self.state.stage.value,
            body_md="\n".join(
                [
                    f"Licensed by the significant `{pending.licensed_by}` above.",
                    "",
                    outcome.note,
                    "",
                    f"Recorded as step `{step.step_id}`.",
                ]
            ),
            result=outcome.result,
            actions=[s.as_button() for s in stage.next_suggestions(self.session)],
        )

    def _result_card(
        self,
        candidate: Candidate,
        result: TestResult,
        step: Any,
        reason: str | None,
        suggestions: list[Suggestion],
        note: str,
    ) -> Card:
        lines: list[str] = []
        if reason:
            lines += [f"**Override.** Reason given: “{reason}”", ""]
        lines += [
            f"Test #{self.session.test_ledger.n_tests} in this session, recorded as step "
            f"`{step.step_id}` on branch `{self.session.active_branch}`.",
        ]
        if note:
            lines += ["", note]
        lines += ["", "Code:", "", "```python", step.code, "```"]
        return Card(
            kind="result",
            title=f"{candidate.function}: {candidate.estimand}",
            stage=self.state.stage.value,
            body_md="\n".join(lines),
            result=result,
            actions=[suggestion.as_button() for suggestion in suggestions],
        )

    # ---- post-hoc proposal ------------------------------------------------

    def _propose_post_hoc(self) -> Card:
        last = self.state.last_result
        if last is None:
            return Card(
                kind="info",
                title="No test to follow up",
                stage=self.state.stage.value,
                body_md="A post-hoc follows a significant omnibus test; none has run yet.",
            )
        stage = self.hypothesis()
        candidate = stage.post_hoc_for(last.candidate, last.result, last.spec, self.session)
        if candidate is None:
            return Card(
                kind="info",
                title="No post-hoc applies here",
                stage=self.state.stage.value,
                body_md=(
                    f"`{last.candidate.function}` either is not an omnibus test or was not "
                    f"significant. Running pairwise comparisons after a non-significant "
                    f"omnibus is the inflation the omnibus exists to prevent."
                ),
            )
        pending = Pending(
            spec=last.spec,
            candidates=self.session.candidates(last.spec),
            verdict=propose_with_rationales(self.session.candidates(last.spec)),
            kind="post_hoc",
            candidate=candidate,
            licensed_by=last.candidate.function,
        )
        self.state.pending = pending
        from edacopilot.stages.hypothesis import POST_HOC_FOR

        option = POST_HOC_FOR[last.candidate.function]
        return Card(
            kind="consensus",
            title=f"Post-hoc: {candidate.function}",
            stage=self.state.stage.value,
            body_md="\n".join(
                [
                    f"`{last.candidate.function}` was significant, which says the groups differ "
                    f"but not which ones. `{candidate.function}` answers that: {option.why}.",
                    "",
                    "Its p-values are adjusted within its own family of pairs and are "
                    "deliberately kept out of the session-wide adjustment, which would "
                    "correct them twice (Section 12.3). It does not add to the session "
                    "test count.",
                    "",
                    f"Its eligibility is inherited from the `{last.candidate.function}` result "
                    f"you accepted (**{candidate.eligibility.value}**): the assumptions it "
                    f"rests on are the ones already checked there.",
                ]
            ),
            actions=[
                ActionButton(
                    label=f"Run {candidate.function}",
                    intent=button(IntentType.ACCEPT),
                    call="session.accept()",
                )
            ],
        )

    # ---- modify ------------------------------------------------------------

    def modify(self, **changes: Any) -> Card:
        """Change something about the question and re-propose (Section 9.1).

        Re-proposing rather than patching the card matters: changing alpha
        or the design changes which methods are eligible, and a card that
        showed the old ranking under the new setting would be wrong.
        """
        pending = self.state.pending
        spec = (
            pending.spec
            if pending
            else (self.state.last_result.spec if self.state.last_result else None)
        )
        if spec is None:
            return Card(
                kind="info",
                title="Nothing to modify",
                stage=self.state.stage.value,
                body_md="There is no question on screen. Ask one first.",
            )

        updates: dict[str, Any] = {}
        variables = dict(spec.variables)
        for key, value in changes.items():
            if key == "design":
                updates["design"] = Design(value)
            elif key == "goal":
                updates["goal"] = Goal(value)
            elif key in ("alpha", "alternative", "equivalence_bounds", "reference_value"):
                updates[key] = value
            else:
                variables[key] = value
        if variables != spec.variables:
            updates["variables"] = variables
        updates["ambiguities"] = []
        return self._propose(spec.model_copy(update=updates))

    # ---- explain -----------------------------------------------------------

    def explain(self, topic: str) -> Card:
        """Answer a "what is X" or "why not X" question (Section 10.5).

        A method named in the current proposal is explained from *this
        session's* checks rather than from the glossary, because "why not a
        t-test" is a question about this data.
        """
        pending = self.state.pending
        function = _named_candidate(topic, pending)
        if function is not None and pending is not None:
            return self._explain_candidate(pending, function)

        answer = answer_free_question(topic)
        return Card(
            kind="info",
            title=f"About: {topic.strip()}",
            stage=self.state.stage.value,
            body_md=answer,
            actions=[
                ActionButton(
                    label="Back to the proposal" if pending else "Ask a question",
                    intent=button(IntentType.ASK_ANALYSIS),
                    call="session.ask(...)",
                )
            ],
        )

    def _explain_candidate(self, pending: Pending, function: str) -> Card:
        candidate = pending.candidates.get(function)
        checks = {check.fact_id: check for check in pending.candidates.checks}
        relevant = [checks[fact_id] for fact_id in candidate.reasons if fact_id in checks]
        spec = registry.get(function)
        lines = [
            f"`{function}` is **{candidate.eligibility.value}** for this question.",
            "",
            f"It estimates: {candidate.estimand or spec.estimand or 'n/a'}.",
            "",
            f"Hard assumptions: {_join_code(spec.assumptions.hard)}.",
            f"Soft assumptions: {_join_code(spec.assumptions.soft)}.",
        ]
        if relevant:
            lines += ["", "Here, these are what decided it:", ""]
            for check in relevant:
                lines.append(
                    f"- **{check.fact_id}** — {check.status.value.upper()} "
                    f"({check.method}, threshold: {check.threshold})"
                )
                lines.append(f"  - {check.consequence}")
        else:
            lines += ["", "Every assumption it declares is met here."]
        return Card(
            kind="info",
            title=f"Why {function} is {candidate.eligibility.value}",
            stage=self.state.stage.value,
            body_md="\n".join(lines),
            diagnostics=relevant,
        )

    # ---- on-demand diagnostic plots (m8.1) ----------------------------------

    def show_diagnostic_plot(self, fact_id: str) -> Card:
        """Render the plot for a check that passed and so was never drawn.

        Only reachable from a "Show plot" button on the proposal still on
        screen (Section 1.3's "hide noise" keeps PASSes out of the card by
        default); if the proposal has moved on, there is nothing stale to
        show, so this says so rather than guessing at an old check.
        """
        pending = self.state.pending
        stale = Card(
            kind="info",
            title="That diagnostic is no longer on screen",
            stage=self.state.stage.value,
            body_md="Ask a new question to see its checks and plots again.",
        )
        if pending is None:
            return stale
        check = next((c for c in pending.candidates.checks if c.fact_id == fact_id), None)
        if check is None or check.status is not CheckStatus.PASS:
            return stale
        plot = plot_for_check(check, pending.spec)
        if plot is None:
            return Card(
                kind="info",
                title=f"No plot for {check.assumption}",
                stage=self.state.stage.value,
                body_md="This check has nothing to draw.",
            )
        function, params = plot
        ref = self._render_plot(function, params)
        return Card(
            kind="info",
            title=f"Plot: {check.assumption} ({check.method})",
            stage=self.state.stage.value,
            body_md=f"**{check.status.value.upper()}** — {check.consequence}",
            diagnostics=[check.model_copy(update={"plot_ref": ref})],
            plots=[ref],
        )

    # ---- navigation ---------------------------------------------------------

    def goto_stage(self, name: str) -> Card:
        """Jump to a stage, warning if it skips an unvisited one (Section 9.2)."""
        try:
            target = parse_stage(name)
        except ValueError as exc:
            return Card(
                kind="warning",
                title="No such stage",
                stage=self.state.stage.value,
                body_md=str(exc),
            )

        warning = jump_warning(target, self.state.visited, self.session.store.profile())
        self._enter(target)
        entry = self.module(target).entry_summary(self.session)
        if warning is None:
            return entry
        return entry.model_copy(
            update={
                "kind": "warning",
                "body_md": f"⚠ {warning}\n\n{entry.body_md}",
            }
        )

    def skip(self) -> Card:
        """Move to the next suggested stage without doing anything here."""
        following = next_stage(self.state.stage, self.session.store.profile())
        if following is None:
            return Card(
                kind="info",
                title="Nothing after this stage",
                stage=self.state.stage.value,
                body_md="This is the last stage in the suggested order.",
            )
        skipped = self.state.stage
        card = self.goto_stage(following.value)
        return card.model_copy(
            update={
                "body_md": (
                    f"Skipped **{skipped.value}** — nothing from it was run or recorded.\n\n"
                    f"{card.body_md}"
                )
            }
        )

    def undo(self) -> Card:
        """Move the branch head back one version (Section 12.1).

        The step stays in the provenance log: an undo changes which data is
        active, it does not erase the fact that something happened.
        """
        before = self.session.version
        try:
            record = self.session.store.undo()
        except Exception as exc:  # noqa: BLE001 - store raises its own types
            return Card(
                kind="warning",
                title="Nothing to undo",
                stage=self.state.stage.value,
                body_md=str(exc),
            )
        self.session.save()
        self.state.pending = None
        return Card(
            kind="info",
            title=f"Undone: back to {record.version_id}",
            stage=self.state.stage.value,
            body_md=(
                f"The head of branch `{self.session.active_branch}` moved from `{before}` to "
                f"`{record.version_id}` ({len(self.session.data)} rows).\n\n"
                f"`{before}` is not deleted — `session.branch(...)` from it, or redo by "
                f"switching, if you change your mind. The step that created it is still in "
                f"the provenance log."
            ),
        )

    def branch(self, name: str, *, version: str | None = None) -> Card:
        try:
            record = self.session.branch_from(name, version=version)
        except Exception as exc:  # noqa: BLE001 - store raises its own types
            return Card(
                kind="warning",
                title="Could not create that branch",
                stage=self.state.stage.value,
                body_md=str(exc),
            )
        self.state.pending = None
        return Card(
            kind="info",
            title=f"On branch {name}",
            stage=self.state.stage.value,
            body_md=(
                f"Branched from `{record.version_id}`. Work here does not affect "
                f"{_join_branches(b for b in self.session.branches if b != name)}; "
                f"switch with `session.switch_branch(name)`.\n\n"
                f"Branches: {_join_branches(self.session.branches)}."
            ),
        )

    def switch_branch(self, name: str) -> Card:
        try:
            record = self.session.switch_to_branch(name)
        except Exception as exc:  # noqa: BLE001 - store raises its own types
            return Card(
                kind="warning",
                title="Could not switch branch",
                stage=self.state.stage.value,
                body_md=str(exc),
            )
        self.state.pending = None
        return Card(
            kind="info",
            title=f"On branch {name}",
            stage=self.state.stage.value,
            body_md=(
                f"Head is `{record.version_id}` ({len(self.session.data)} rows). "
                f"Branches: {', '.join(f'`{b}`' for b in self.session.branches)}."
            ),
        )

    # ---- read-only ----------------------------------------------------------

    def _show(self, intent: Intent) -> Card:
        """Ad-hoc read-only request (Section 9.1's SHOW).

        "plot age against income" names two columns and wants them
        related; "plot income" names one and wants it described. Which
        columns were named is read off the text against the frame's own
        column list, since with no LLM there is nothing else to read it
        with (Section 10.5).
        """
        from edacopilot.stages.explore import ExploreStage

        self._enter(Stage.EXPLORE)
        explore = self.module(Stage.EXPLORE)
        assert isinstance(explore, ExploreStage)

        text = str(intent.payload.get("text", ""))
        named = [str(column) for column in self.session.data.columns if str(column) in text]
        if len(named) >= 2:
            return explore.relate(self.session, named[0], named[1])
        if len(named) == 1:
            return explore.describe(self.session, named[0])
        return explore.entry_summary(self.session)

    def _settings(self, intent: Intent) -> Card:
        if intent.payload.get("show") == "ledger":
            return self.ledger_card()
        return Card(
            kind="info",
            title="Settings",
            stage=self.state.stage.value,
            body_md=(
                f"alpha: {self.session.config.get('alpha', 0.05)}; "
                f"multiple-testing method: {self.session.test_ledger.method}; "
                f"deterministic mode: on (no LLM configured)."
            ),
        )

    def ledger_card(self) -> Card:
        rows = self.session.ledger()
        if not rows:
            return Card(
                kind="info",
                title="Test ledger",
                stage=self.state.stage.value,
                body_md="No tests have run yet.",
            )
        lines = [
            f"{self.session.test_ledger.n_tests} test(s) counted toward the session "
            f"adjustment, using **{self.session.test_ledger.method}**.",
            "",
            "| # | test | family | p (raw) | p (adj, session) | p (adj, family) | counts |",
            "|---|---|---|---|---|---|---|",
        ]
        for row in rows:
            lines.append(
                f"| {row['entry_id']} | `{row['function']}` | {row['family']} "
                f"| {_p(row['p_value'])} | {_p(row['p_adjusted'])} "
                f"| {_p(row['p_adjusted_in_family'])} | {'yes' if row['counts'] else 'no'} |"
            )
        lines += [
            "",
            "A post-hoc procedure is recorded but does not count: its comparisons are "
            "already adjusted within their own family, and pooling them here would correct "
            "them twice (Section 12.3).",
        ]
        return Card(
            kind="info", title="Test ledger", stage=self.state.stage.value, body_md="\n".join(lines)
        )

    # ---- cards --------------------------------------------------------------

    def _proposal_card(self, pending: Pending) -> Card:
        verdict = pending.verdict
        consensus = verdict.is_consensus
        statuses = pending.candidates.statuses()
        lines = [
            render_card(verdict),
            "",
            f"Family: `{pending.candidates.family}`. "
            f"{len(pending.candidates.candidates)} method(s) considered, "
            f"{len(pending.candidates.eligible)} eligible.",
            "",
            "| Method | Status | Estimates |",
            "|---|---|---|",
        ]
        for candidate in pending.candidates.candidates:
            lines.append(
                f"| `{candidate.function}` | {candidate.eligibility.value} | {candidate.estimand} |"
            )
        if pending.candidates.unavailable:
            lines += ["", "Not available for this question (a missing parameter, not a verdict):"]
            lines += [
                f"- `{function}`: {why}"
                for function, why in sorted(pending.candidates.unavailable.items())
            ]
        lines += [
            "",
            "Nothing has run. Accept a proposal, or override with a method of your own.",
        ]

        actions: list[ActionButton] = []
        for proposal in verdict.proposals:
            if not proposal.function:
                continue
            persona_id = proposal.personas[0]
            label = (
                f"Run {proposal.function}"
                if consensus
                else f"{' + '.join(proposal.display_names)}: {proposal.function}"
            )
            actions.append(
                ActionButton(
                    label=label,
                    intent=button(IntentType.ACCEPT, persona=None if consensus else persona_id),
                    call="session.accept()" if consensus else f'session.accept("{persona_id}")',
                )
            )
        first_other = next(
            (
                c.function
                for c in pending.candidates.candidates
                if c.function not in {p.function for p in verdict.proposals}
            ),
            None,
        )
        if first_other is not None:
            actions.append(
                ActionButton(
                    label=f"Use something else (e.g. {first_other})",
                    intent=button(IntentType.OVERRIDE, function=first_other),
                    call=f'session.override("{first_other}")',
                )
            )
        actions.append(
            ActionButton(
                label="Why this method?",
                intent=button(
                    IntentType.EXPLAIN, topic=verdict.proposals[0].function or "effect_size"
                ),
                call=f'session.explain("{verdict.proposals[0].function or "effect size"}")',
            )
        )

        # Section 13.2: plots render inline in cards. A Shapiro p-value
        # says normality failed; the Q-Q plot says how, and a long tail
        # and two outliers lead to different methods.
        diagnostics, plots = illustrate(
            list(pending.candidates.checks),
            pending.spec,
            self._render_plot,
            plot_all=bool(self.session.config.get("plot_all_diagnostics")),
        )

        # A PASSing check gets no plot inline (Section 1.3: hide noise), but
        # its picture is a click away if one exists to draw (m8.1) -- a
        # button, not an eager render, so a card full of PASSes costs
        # nothing until someone actually wants to look. UNTESTABLE/N-A
        # checks are excluded even when a builder matches their assumption
        # name: there is no computed check behind them, so a plot would
        # show data without the evidence to back what it's illustrating.
        for check in diagnostics:
            if check.status is not CheckStatus.PASS:
                continue
            if plot_for_check(check, pending.spec) is None:
                continue
            actions.append(
                ActionButton(
                    label=f"Show plot: {check.assumption}",
                    intent=button(IntentType.SHOW_DIAGNOSTIC_PLOT, fact_id=check.fact_id),
                    call=f'session.show_diagnostic_plot("{check.fact_id}")',
                )
            )

        return Card(
            kind="consensus" if consensus else "divergence",
            title=(
                "All three personas agree"
                if consensus
                else f"The personas disagree ({len(verdict.proposals)} proposals)"
            ),
            stage=self.state.stage.value,
            body_md="\n".join(lines),
            diagnostics=diagnostics,
            graded=list(pending.candidates.graded),
            proposals=_proposal_views(pending, statuses),
            plots=plots,
            actions=actions,
        )

    def _render_plot(self, function: str, params: dict[str, Any]) -> str:
        """Draw and store one plot, returning its `plot_ref`."""
        return self.session.plot(function, **params)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _with_corrections(card: Card, spec: QuestionSpec) -> Card:
    """Prepend Section 7.1's column-correction notes to a card's body.

    Never silent (the module docstring on `column_matching.py` says why):
    a fuzzy-matched column name is corrected automatically, but the
    correction is the first thing on the card that follows, not something
    folded invisibly into the proposal.
    """
    if not spec.column_corrections:
        return card
    notes = "\n".join(f"*{note}*" for note in spec.column_corrections)
    body = f"{notes}\n\n{card.body_md}" if card.body_md else notes
    return card.model_copy(update={"body_md": body})


def _proposal_views(
    pending: Pending, statuses: dict[str, Eligibility] | None = None
) -> list[ProposalView]:
    statuses = statuses if statuses is not None else pending.candidates.statuses()
    return [
        ProposalView(
            persona=pick.persona,
            display_name=pick.display_name,
            function=pick.function,
            eligibility=(statuses[pick.function].value if pick.function in statuses else None),
            rationale=pick.rationale,
            concurs_with=pick.concurs_with,
        )
        for pick in pending.verdict.picks
    ]


def _join_branches(names: Any) -> str:
    listed = [f"`{name}`" for name in names]
    return ", ".join(listed) if listed else "any other branch"


def _proposed_function(pending: Pending) -> str | None:
    """What a bare `accept()` would run, if that is unambiguous."""
    if not pending.verdict.is_consensus:
        return None
    return pending.verdict.proposals[0].function or None


def _hard_failures(
    candidate: Candidate, checks: dict[str, AssumptionCheck]
) -> list[AssumptionCheck]:
    """The hard assumptions that made a candidate ineligible."""
    hard = set(registry.get(candidate.function).assumptions.hard)
    failing = [
        check
        for fact_id in candidate.reasons
        if (check := checks.get(fact_id)) is not None
        and check.status is CheckStatus.FAIL
        and check.assumption in hard
    ]
    if failing:
        return failing
    # Assumption names and check names do not always coincide; fall back to
    # every failing reason rather than showing none.
    return [
        check
        for fact_id in candidate.reasons
        if (check := checks.get(fact_id)) is not None and check.status is CheckStatus.FAIL
    ]


def _named_candidate(topic: str, pending: Pending | None) -> str | None:
    if pending is None:
        return None
    normalised = topic.lower().replace(" ", "_").replace("-", "_")
    matches = [c.function for c in pending.candidates.candidates if c.function in normalised]
    return max(matches, key=len) if matches else None


def _roles_in_question(ambiguities: list[str], spec: QuestionSpec) -> set[str]:
    """Which spec roles a bare "yes" is confirming.

    Read from the question text, which names the column, so a "yes" confirms
    the thing that was asked about rather than everything at once.
    """
    joined = " ".join(ambiguities)
    confirmed = {role for role, column in spec.variables.items() if f"'{column}'" in joined}
    if "paired" in joined or "independent" in joined or "subject" in joined:
        confirmed.add("design")
    return confirmed


def _modify_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "text"}


def _join_code(names: list[str]) -> str:
    return ", ".join(f"`{name}`" for name in names) if names else "none"


def _p(value: float | None) -> str:
    return "—" if value is None else f"{value:.4g}"
