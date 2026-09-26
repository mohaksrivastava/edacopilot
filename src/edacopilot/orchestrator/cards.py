"""Cards: the orchestrator's output payloads (ARCHITECTURE.md, Section 9.4).

A `Card` is plain data. It knows nothing about ipywidgets, matplotlib or
HTML — M8's `ui/cards.py` renders one, M7 only builds them. That split is
what lets the conversation tests in Section 15.5 assert on what the user
sees without a notebook: `Card.to_text()` is a complete rendering, so a
golden transcript is a real transcript rather than a summary of one.

Two additions to Section 9.4's field list, both the same kind of addition
`PostHocResult` itself was:

- `result` also accepts a `PostHocResult`. Section 9.4 predates that model,
  and a post-hoc's comparison table is exactly a result card's content.
- `stage` records which stage produced the card. Section 9.2's warnings and
  Section 9.5's suggestions are both stage-scoped, and a transcript that
  did not say which stage it was in would be hard to read.
- `graded` names which of the `diagnostics` actually decided an assumption.
  Section 7.2's engine deliberately records *evidence* alongside verdicts
  (`Resolution.evidence`), and some of that evidence reads FAIL while
  deciding nothing: a `measurement_level` attempt against a scale no method
  required, for instance. Both belong on the card — Section 7.4's worked
  example lists Shapiro's result, which is evidence under
  `normality_or_large_n` — but they must not look alike, or a reader who
  notices the diagnostics are padded with irrelevant failures stops reading
  them, which is the one thing this project cannot afford.

Everything else is Section 9.4 verbatim. In particular, next-step
suggestions (Section 9.5) are `actions`, not a separate field: they are
buttons, and the rule that they are never executed automatically is a
property of the loop, not of the card.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from edacore.contracts import (
    AssumptionCheck,
    CheckStatus,
    PostHocResult,
    TestResult,
    TransformRecord,
)

from .intents import Intent

CardKind = Literal[
    "info",
    "question",
    "consensus",
    "divergence",
    "result",
    "warning",
    "override_confirm",
    "export",
]

# How each status prints in a diagnostics table. Fixed strings rather than
# colour, because a card has to survive being read as plain text.
_STATUS_MARK = {
    CheckStatus.PASS: "PASS",
    CheckStatus.BORDERLINE: "BORDERLINE",
    CheckStatus.FAIL: "FAIL",
    CheckStatus.NOT_APPLICABLE: "n/a",
    CheckStatus.UNTESTABLE: "UNTESTABLE",
}


class ActionButton(BaseModel):
    """One button, and the intent clicking it produces (Section 9.4)."""

    model_config = ConfigDict(frozen=True)

    label: str
    intent: Intent
    # What the user would type to get the same effect through the Python
    # API. Shown in the plain-text rendering so a transcript doubles as
    # documentation of the API, and so M8's buttons and this API cannot
    # drift apart unnoticed.
    call: str = ""


class ProposalView(BaseModel):
    """One persona's proposal, flattened for display (Section 9.4).

    Mirrors `session.provenance.ProposalView`, which records the same thing
    for the log. Kept separate because a card also shows the persona's
    display name and the eligibility status the *engine* assigned, and the
    log should not grow display concerns.
    """

    model_config = ConfigDict(frozen=True)

    persona: str
    display_name: str
    function: str | None
    eligibility: str | None = None
    rationale: str = ""
    concurs_with: str | None = None


class Suggestion(BaseModel):
    """A next step the system offers but does not take (Section 9.5)."""

    model_config = ConfigDict(frozen=True)

    label: str
    reason: str = ""
    intent: Intent
    call: str = ""

    def as_button(self) -> ActionButton:
        return ActionButton(label=self.label, intent=self.intent, call=self.call)


class Card(BaseModel):
    """A UI payload (Section 9.4)."""

    model_config = ConfigDict(frozen=True)

    kind: CardKind
    title: str
    body_md: str = ""
    stage: str = ""
    diagnostics: list[AssumptionCheck] = Field(default_factory=list)
    graded: list[str] = Field(default_factory=list)
    proposals: list[ProposalView] = Field(default_factory=list)
    result: TestResult | PostHocResult | TransformRecord | None = None
    plots: list[str] = Field(default_factory=list)
    actions: list[ActionButton] = Field(default_factory=list)

    # ---- plain-text rendering -------------------------------------------

    def to_text(self) -> str:
        """The whole card as plain text (Section 15.5).

        Everything the card carries appears here. A rendering that dropped
        a field would make a golden transcript agree with a card the user
        would not recognise.
        """
        lines: list[str] = [f"[{self.kind}] {self.title}"]
        if self.stage:
            lines.append(f"stage: {self.stage}")
        lines.append("")
        if self.body_md.strip():
            lines.extend(self.body_md.strip().splitlines())
            lines.append("")
        lines.extend(_diagnostics_lines(self.diagnostics, self.graded))
        lines.extend(_proposal_lines(self.proposals))
        lines.extend(_result_lines(self.result))
        lines.extend(_plot_lines(self.plots))
        lines.extend(_action_lines(self.actions))
        while lines and not lines[-1].strip():
            lines.pop()
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_text()


def _diagnostics_lines(checks: list[AssumptionCheck], graded: list[str]) -> list[str]:
    """Verdicts first, then evidence, each labelled for what it is."""
    if not checks:
        return []
    decisive = set(graded)
    verdicts = [check for check in checks if not decisive or check.fact_id in decisive]
    evidence = [check for check in checks if decisive and check.fact_id not in decisive]

    lines = ["Diagnostics (these decided which methods are valid):"]
    lines.extend(_check_lines(verdicts))
    if evidence:
        lines.append("")
        lines.append("Also computed (evidence, not a verdict on any method):")
        lines.extend(_check_lines(evidence))
    lines.append("")
    return lines


def _check_lines(checks: list[AssumptionCheck]) -> list[str]:
    lines: list[str] = []
    for check in checks:
        stat = "" if check.statistic is None else f" statistic={check.statistic:.4g}"
        p = "" if check.p_value is None else f" p={check.p_value:.4g}"
        lines.append(f"  {_STATUS_MARK[check.status]:<11} {check.fact_id}{stat}{p}")
        lines.append(f"              -> {check.consequence}")
    return lines


def _proposal_lines(proposals: list[ProposalView]) -> list[str]:
    if not proposals:
        return []
    lines = ["Proposals:"]
    for proposal in proposals:
        method = proposal.function or "(nothing to offer)"
        status = f" [{proposal.eligibility}]" if proposal.eligibility else ""
        lines.append(f"  {proposal.display_name} -> {method}{status}")
        if proposal.rationale:
            lines.append(f"      {proposal.rationale}")
    lines.append("")
    return lines


def _result_lines(result: TestResult | PostHocResult | TransformRecord | None) -> list[str]:
    if result is None:
        return []
    if isinstance(result, TestResult):
        return _test_result_lines(result)
    if isinstance(result, PostHocResult):
        return _posthoc_lines(result)
    return _transform_lines(result)


def _test_result_lines(result: TestResult) -> list[str]:
    lines = ["Result:", f"  test        {result.function} ({result.estimand})"]
    lines.append(f"  {result.statistic_name + ' ':<11} {result.statistic:.4g}")
    if result.df is not None:
        lines.append(f"  df          {_format_df(result.df)}")
    if result.p_value is not None:
        lines.append(f"  p (raw)     {result.p_value:.4g}")
    if result.p_adjusted is not None:
        lines.append(f"  p (adj)     {result.p_adjusted:.4g}")
    if result.estimate is not None:
        lines.append(f"  estimate    {result.estimate:.4g}{_ci(result.ci, result.ci_level)}")
    if result.effect_size is not None:
        magnitude = f" ({result.effect_magnitude})" if result.effect_magnitude else ""
        lines.append(
            f"  effect      {result.effect_size_name} = {result.effect_size:.4g}"
            f"{_ci(result.effect_size_ci, result.ci_level)}{magnitude}"
        )
    if result.n:
        lines.append(f"  n used      {_format_n(result.n)}")
    lines.append("")
    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in result.warnings)
        lines.append("")
    if result.validity_notes:
        lines.append("Validity notes:")
        lines.extend(f"  - {note}" for note in result.validity_notes)
        lines.append("")
    return lines


def _posthoc_lines(result: PostHocResult) -> list[str]:
    lines = [
        "Result:",
        f"  procedure   {result.function} ({result.method})",
        f"  adjustment  {result.p_adjust_method or 'none'} "
        f"within this family of {len(result.comparisons)}",
        "",
        "Comparisons:",
    ]
    for comparison in result.comparisons:
        adjusted = "" if comparison.p_adjusted is None else f"  p(adj)={comparison.p_adjusted:.4g}"
        estimate = "" if comparison.estimate is None else f"  estimate={comparison.estimate:.4g}"
        lines.append(
            f"  {comparison.group_a} vs {comparison.group_b}:"
            f"  p={comparison.p_value:.4g}{adjusted}{estimate}"
        )
    lines.append("")
    if result.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in result.warnings)
        lines.append("")
    return lines


def _transform_lines(record: TransformRecord) -> list[str]:
    lines = [
        "Result:",
        f"  transform   {record.function}",
        f"  columns     {', '.join(record.columns_affected)}",
        f"  rows        {record.rows_before} -> {record.rows_after}",
        "",
    ]
    if record.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in record.warnings)
        lines.append("")
    return lines


def _plot_lines(plots: list[str]) -> list[str]:
    if not plots:
        return []
    return ["Plots:", *(f"  - {ref}" for ref in plots), ""]


def _action_lines(actions: list[ActionButton]) -> list[str]:
    if not actions:
        return []
    lines = ["Next:"]
    for action in actions:
        call = f"   ({action.call})" if action.call else ""
        lines.append(f"  [{action.label}]{call}")
    lines.append("")
    return lines


def _ci(ci: tuple[float, float] | None, level: float) -> str:
    if ci is None:
        return ""
    return f"  {level:.0%} CI [{ci[0]:.4g}, {ci[1]:.4g}]"


def _format_df(df: float | tuple[float, float]) -> str:
    if isinstance(df, tuple):
        return f"{df[0]:.4g}, {df[1]:.4g}"
    return f"{df:.4g}"


def _format_n(n: dict[str, int]) -> str:
    return ", ".join(f"{label}={count}" for label, count in n.items())
