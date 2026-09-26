"""The HYPOTHESIS stage (ARCHITECTURE.md, Sections 9.3, 7 and 8).

This is the stage the rest of the system exists for, and the one where
getting it wrong is expensive. Section 9.3's row is short — "Confirm design
(independent/paired/repeated); full eligibility + persona flow; TestLedger
updated" — and every clause is load-bearing:

- **Confirm design.** Not inferred, never inferred. `validate_spec` raises
  the question and `select_candidates` refuses to proceed while it is open
  (Section 7.1), so the design question reaches the user *before* any
  persona card. That ordering is an invariant with its own test.
- **Full eligibility + persona flow.** The stage does not choose a method.
  It hands a validated spec to Section 7's engine and Section 8's personas
  and renders what comes back.
- **TestLedger updated.** Every executed test enters the ledger, adjusted
  p-values are recomputed session-wide, and the result card shows raw p,
  adjusted p and the count (Section 12.3).

**Post-hoc candidates.** A post-hoc is not in any Section 7.3 family: it is
not something the user asks for, it is something a significant omnibus
result licenses. Section 7.3 therefore has no eligibility rule that would
produce one. The rule this stage uses instead, recorded in Section 18: a
post-hoc candidate inherits the eligibility status and reasons of the
omnibus candidate the user accepted. That is the honest reading — the
assumptions a Dunn test rests on are the ones Kruskal-Wallis already had
checked, and inventing a fresh ELIGIBLE verdict for it would claim a check
that never ran.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import pandas as pd

from edacopilot.eligibility import (
    CandidateSet,
    Design,
    Goal,
    QuestionSpec,
    UnsupportedQuestionError,
    subject_column,
)
from edacopilot.orchestrator.cards import Card, Suggestion
from edacopilot.orchestrator.intents import Intent, IntentType, button
from edacore.contracts import (
    AssumptionCheck,
    Candidate,
    CheckStatus,
    Eligibility,
    PostHocResult,
    TestResult,
)
from edacore.registry import registry
from edacore.validity_notes import ValidityContext, attach

from .base import BaseStage, StepOutcome

if TYPE_CHECKING:  # pragma: no cover
    from edacopilot.session.session import Session

DEFAULT_ALPHA: Final = 0.05


@dataclass(frozen=True)
class PostHocOption:
    """The post-hoc a given omnibus licenses, and how to call it."""

    function: str
    # Which QuestionSpec roles map to which of the post-hoc's parameters.
    roles: dict[str, str]
    why: str


# Section 12.3's omnibus list, each paired with the procedure whose own
# family-wise correction matches its design. Games-Howell rather than Tukey
# for the unequal-variance omnibus tests, because Tukey's studentized range
# assumes the equal variance those tests were chosen to avoid assuming.
POST_HOC_FOR: Final[dict[str, PostHocOption]] = {
    "one_way_anova": PostHocOption(
        "tukey_hsd",
        {"outcome": "outcome", "group": "group"},
        "all pairs, corrected together by the studentized range",
    ),
    "welch_anova": PostHocOption(
        "games_howell",
        {"outcome": "outcome", "group": "group"},
        "the unequal-variance counterpart of Tukey's test",
    ),
    "alexander_govern": PostHocOption(
        "games_howell",
        {"outcome": "outcome", "group": "group"},
        "the unequal-variance counterpart of Tukey's test",
    ),
    "kruskal_wallis": PostHocOption(
        "dunn_test",
        {"outcome": "outcome", "group": "group"},
        "pairwise rank comparisons, Holm-adjusted within the family of pairs",
    ),
    "permutation_anova": PostHocOption(
        "permutation_posthoc",
        {"outcome": "outcome", "group": "group"},
        "pairwise permutation tests, Holm-adjusted within the family of pairs",
    ),
    "friedman": PostHocOption(
        "nemenyi_friedman",
        {"outcome": "dv", "subject": "subject", "group": "within"},
        "pairwise comparisons of mean ranks after a significant Friedman test",
    ),
    "repeated_measures_anova": PostHocOption(
        "paired_posthoc",
        {"outcome": "dv", "subject": "subject", "group": "within"},
        "pairwise paired comparisons, Holm-adjusted within the family of pairs",
    ),
    "cochran_q": PostHocOption(
        "mcnemar_posthoc",
        {"outcome": "dv", "subject": "subject", "group": "within"},
        "pairwise McNemar tests, Holm-adjusted within the family of pairs",
    ),
}


class HypothesisStage(BaseStage):
    name = "hypothesis"
    summary = "choosing and running a valid test for a question"
    milestone = "M7"

    @property
    def allowed_functions(self) -> list[str]:  # type: ignore[override]
        """Every registered test, post-hoc and check.

        Listed from the registry rather than written out, so a test added in
        a later milestone is in scope without editing this file — and so the
        scope cannot silently include a transform.
        """
        return sorted(
            spec.name
            for spec in registry.list()
            if spec.kind in ("test", "posthoc", "check", "effect")
        )

    # ---- entry -----------------------------------------------------------

    def entry_summary(self, session: Session) -> Card:
        profile = session.store.profile()
        lines = [
            "Ask a question and this stage will check the assumptions, rank every valid "
            "method, and show what each persona would choose. Nothing runs until you accept.",
            "",
            "What you can ask about:",
        ]
        numeric = [
            c.name for c in profile.columns if c.semantic_type.value in ("continuous", "discrete")
        ]
        grouping = [
            c.name
            for c in profile.columns
            if c.semantic_type.value in ("nominal", "ordinal", "binary")
        ]
        lines.append(f"- outcomes: {_join(numeric)}")
        lines.append(f"- groupings: {_join(grouping)}")
        if profile.entity_id:
            lines += [
                "",
                f"`{profile.entity_id}` looks like a subject identifier, so the design "
                f"question — independent groups or repeated measurements — will be asked "
                f"before any test is proposed.",
            ]
        return Card(
            kind="info",
            title="Hypothesis testing",
            stage=self.name,
            body_md="\n".join(lines),
            actions=[s.as_button() for s in self.next_suggestions(session)],
        )

    # ---- the flow --------------------------------------------------------

    def build_spec(self, intent: Intent, session: Session) -> QuestionSpec:
        """A spec from a structured intent (Section 10.5's fallback).

        With no LLM there is nothing here that maps prose to columns; the
        orchestrator has already turned keyword arguments or a form into
        the payload this reads. A payload that does not name what the goal
        needs raises, and the orchestrator turns that into the form card.
        """
        payload = dict(intent.payload)
        goal = Goal(payload.pop("goal", Goal.COMPARE_GROUPS.value))
        design = Design(payload.pop("design", Design.UNKNOWN.value))
        variables = {
            role: column
            for role, column in payload.items()
            if role in _VARIABLE_ROLES and isinstance(column, str)
        }
        confirmed = set(payload.get("confirmed_by_user") or ())
        return QuestionSpec(
            goal=goal,
            variables=variables,
            design=design,
            alternative=payload.get("alternative", "two-sided"),
            alpha=payload.get("alpha"),
            equivalence_bounds=payload.get("equivalence_bounds"),
            reference_value=payload.get("reference_value"),
            reference_proportions=payload.get("reference_proportions"),
            user_text=str(payload.get("text", "")),
            confirmed_by_user=confirmed,
        )

    def execute(
        self, candidate: Candidate, session: Session, spec: QuestionSpec | None = None
    ) -> StepOutcome:
        """Run the chosen method and report what it took to run it.

        Validity notes are attached by the caller rather than here, because
        five of Section 6.12's nine guards are about the session -- how many
        tests have run, what the same-shape check said, whether the outcome
        was imputed -- and the stage has the session but not the ledger
        entry this very call is about to create.
        """
        self.check_allowed(candidate.function)
        frame, preamble, df_var = self.frame_for(candidate, session, spec)
        result = registry.get(candidate.function).func(frame, **candidate.params)
        return StepOutcome(
            result=result,
            family=_family_label(candidate, session),
            code_preamble=preamble,
            df_var=df_var,
        )

    def frame_for(
        self, candidate: Candidate, session: Session, spec: QuestionSpec | None
    ) -> tuple[pd.DataFrame, str, str]:
        """The frame this method runs against, and the code that produced it.

        `edacore`'s paired tests take two *columns* (Section 6.7), while a
        paired question arrives in long form: outcome, group, subject. M4
        recorded the two level names as the candidate's `a` and `b` params
        and left the reshape to the orchestrator deliberately -- the
        eligibility engine reasons about the user's own columns, and
        pivoting inside it would make every check report against a frame the
        user never saw.

        The pivot is returned as source, not merely performed, because rule
        7 requires the recorded step to be runnable: an exported notebook
        whose paired t-test ran against a frame the notebook never built
        would reproduce nothing.
        """
        df = session.data
        a, b = candidate.params.get("a"), candidate.params.get("b")
        needs_pivot = (
            isinstance(a, str)
            and isinstance(b, str)
            and a not in df.columns
            and b not in df.columns
        )
        if not needs_pivot:
            return df, "", "df"
        if spec is None:  # pragma: no cover - defensive
            raise UnsupportedQuestionError(
                f"'{candidate.function}' needs the question's columns to reshape the data, "
                f"and none were passed"
            )

        outcome, group = spec.variables.get("outcome"), spec.variables.get("group")
        subject = subject_column(spec, df)
        if outcome is None or group is None or subject is None:
            raise UnsupportedQuestionError(
                f"'{candidate.function}' needs each subject's two measurements side by "
                f"side, which means pivoting on a subject column; none was named and none "
                f"could be inferred from the data"
            )
        wide = df.pivot(index=subject, columns=group, values=outcome)
        preamble = f"wide = df.pivot(index={subject!r}, columns={group!r}, values={outcome!r})"
        return wide, preamble, "wide"

    def validity_context(
        self,
        session: Session,
        *,
        spec: QuestionSpec | None,
        checks: list[AssumptionCheck],
        test_number: int,
        p_adjusted: float | None,
        chosen_after_exploring: bool = False,
    ) -> ValidityContext:
        return ValidityContext(
            alpha=alpha_for(spec, session),
            checks=tuple(checks),
            goal=spec.goal.value if spec else None,
            variables=dict(spec.variables) if spec else {},
            test_number=test_number,
            adjust_method=session.test_ledger.method,
            p_adjusted=p_adjusted,
            chosen_after_exploring=chosen_after_exploring,
        )

    def annotate(self, result: TestResult, context: ValidityContext) -> TestResult:
        return attach(result, context)

    # ---- post-hoc ---------------------------------------------------------

    def post_hoc_for(
        self, candidate: Candidate, result: TestResult, spec: QuestionSpec, session: Session
    ) -> Candidate | None:
        """The post-hoc a significant omnibus licenses, or None.

        Silent when the omnibus was not significant: running pairwise
        comparisons after a non-significant omnibus is the inflation the
        omnibus was there to prevent.
        """
        option = POST_HOC_FOR.get(candidate.function)
        if option is None:
            return None
        if result.p_value is None or result.p_value >= alpha_for(spec, session):
            return None
        try:
            params = {
                parameter: spec.variables[role]
                for role, parameter in option.roles.items()
                if role in spec.variables
            }
        except KeyError:  # pragma: no cover - defensive
            return None
        if len(params) != len(option.roles):
            return None
        function_spec = registry.get(option.function)
        return Candidate(
            function=option.function,
            params=params,
            # Inherited from the omnibus the user accepted -- see the module
            # docstring. A post-hoc's assumptions are the omnibus's.
            eligibility=candidate.eligibility,
            reasons=list(candidate.reasons),
            tags=set(function_spec.tags),
            estimand=function_spec.estimand or "",
        )

    def execute_post_hoc(self, candidate: Candidate, session: Session) -> StepOutcome:
        self.check_allowed(candidate.function)
        result: PostHocResult = registry.get(candidate.function).func(
            session.data, **candidate.params
        )
        return StepOutcome(
            result=result,
            family=f"{candidate.function} comparisons",
            note=(
                f"These {len(result.comparisons)} p-values are adjusted within this "
                f"procedure's own family by {result.p_adjust_method or 'the procedure itself'}. "
                f"They are deliberately *not* pooled into the session-wide adjustment, which "
                f"would correct them twice (Section 12.3), and they do not change the "
                f"session test count."
            ),
        )

    # ---- suggestions (Section 9.5) ----------------------------------------

    def next_suggestions(self, session: Session) -> list[Suggestion]:
        suggestions = [
            Suggestion(
                label="Ask another question",
                reason="each new test is added to the ledger and adjusts the rest",
                intent=Intent(type=IntentType.ASK_ANALYSIS),
                call="session.ask(goal=..., outcome=..., group=...)",
            ),
            Suggestion(
                label="Show the test ledger",
                reason="every test so far, with raw and adjusted p-values",
                intent=button(IntentType.SETTINGS, show="ledger"),
                call="session.ledger()",
            ),
        ]
        if session.test_ledger.n_tests:
            suggestions.append(
                Suggestion(
                    label="Branch to try a different path",
                    reason="compare two treatments of the same data side by side",
                    intent=button(IntentType.BRANCH, name="alternative"),
                    call='session.branch("alternative")',
                )
            )
        return suggestions

    def result_suggestions(
        self,
        session: Session,
        *,
        candidate: Candidate,
        result: TestResult,
        spec: QuestionSpec,
        checks: list[AssumptionCheck],
        graded: list[str] | None = None,
    ) -> list[Suggestion]:
        """Section 9.5's deterministic rules, 2-4 buttons, never executed.

        Ordered by what the result actually calls for, so the first button
        is the one the user most likely wants.
        """
        suggestions: list[Suggestion] = []
        alpha = alpha_for(spec, session)
        significant = result.p_value is not None and result.p_value < alpha

        post_hoc = self.post_hoc_for(candidate, result, spec, session)
        if post_hoc is not None:
            option = POST_HOC_FOR[candidate.function]
            suggestions.append(
                Suggestion(
                    label=f"Run post-hoc comparisons ({post_hoc.function})",
                    reason=(f"the omnibus says the groups differ but not which; {option.why}"),
                    intent=Intent(
                        type=IntentType.ASK_ANALYSIS,
                        payload={"post_hoc": True, "function": post_hoc.function},
                    ),
                    call='session.ask("run post-hoc comparisons")',
                )
            )

        if not significant and result.p_value is not None:
            suggestions.append(
                Suggestion(
                    label="Test for equivalence instead",
                    reason="a non-significant result is not evidence of no difference",
                    intent=Intent(
                        type=IntentType.ASK_ANALYSIS,
                        payload={"goal": Goal.EQUIVALENCE.value, **dict(spec.variables)},
                    ),
                    call='session.ask(goal="equivalence", ..., equivalence_bounds=(lo, hi))',
                )
            )

        # Only a check that *graded* an assumption is worth offering to
        # explain; evidence that reads FAIL while deciding nothing would
        # send the user to look at something that changed no verdict.
        decisive = set(graded) if graded else None
        failing = [
            check
            for check in checks
            if check.status is CheckStatus.FAIL and (decisive is None or check.fact_id in decisive)
        ]
        if failing:
            suggestions.append(
                Suggestion(
                    label=f"Explain {failing[0].assumption}",
                    reason=f"{failing[0].fact_id} failed, and it shaped which methods were valid",
                    intent=button(IntentType.EXPLAIN, topic=failing[0].assumption),
                    call=f'session.explain("{failing[0].assumption}")',
                )
            )

        if session.test_ledger.n_tests > 1:
            suggestions.append(
                Suggestion(
                    label="Show the test ledger",
                    reason=(
                        f"{session.test_ledger.n_tests} tests so far; the adjusted p-values "
                        f"all moved when this one was added"
                    ),
                    intent=button(IntentType.SETTINGS, show="ledger"),
                    call="session.ledger()",
                )
            )

        suggestions.append(
            Suggestion(
                label="Ask another question",
                reason="each new test enters the ledger and adjusts the rest",
                intent=Intent(type=IntentType.ASK_ANALYSIS),
                call="session.ask(goal=..., outcome=..., group=...)",
            )
        )

        # Section 9.5 asks for 2-4 buttons. A clean, significant result
        # with no failed assumption and no post-hoc would otherwise offer
        # one, and a lone "ask another question" reads as a dead end at the
        # moment the user most needs somewhere to go: the effect size is
        # what a significant p-value leaves unanswered.
        if len(suggestions) < 2 and result.effect_size_name:
            suggestions.append(
                Suggestion(
                    label=f"Explain {result.effect_size_name}",
                    reason="the p-value says there is an effect; this says how big",
                    intent=button(IntentType.EXPLAIN, topic=result.effect_size_name),
                    call=f'session.explain("{result.effect_size_name}")',
                )
            )
        if len(suggestions) < 2:
            suggestions.append(
                Suggestion(
                    label="Show the test ledger",
                    reason="every test so far, with raw and adjusted p-values",
                    intent=button(IntentType.SETTINGS, show="ledger"),
                    call="session.ledger()",
                )
            )
        return suggestions[:4]


_VARIABLE_ROLES: Final = frozenset(
    {"outcome", "group", "x", "y", "variable", "subject", "id", "entity", "covariate", "factor_b"}
)


def alpha_for(spec: QuestionSpec | None, session: Session) -> float:
    """The alpha in force: the question's, else the session's, else 0.05.

    Section 7.1 has `alpha=None` mean "persona/session default", so the
    resolution order lives in one place rather than at each call site.
    """
    if spec is not None and spec.alpha is not None:
        return float(spec.alpha)
    configured = session.config.get("alpha")
    return float(configured) if configured is not None else DEFAULT_ALPHA


def _family_label(candidate: Candidate, session: Session) -> str:
    """Section 12.3's family label, e.g. "income comparisons"."""
    outcome = candidate.params.get("value_col") or candidate.params.get("outcome")
    outcome = outcome or candidate.params.get("dv") or candidate.params.get("col")
    return f"{outcome} comparisons" if outcome else "session"


def _join(names: list[str]) -> str:
    return ", ".join(f"`{name}`" for name in names) if names else "none"


def is_significant(result: TestResult, alpha: float) -> bool:
    return result.p_value is not None and result.p_value < alpha


def status_of(candidates: CandidateSet, function: str) -> Eligibility:
    return candidates.get(function).eligibility
