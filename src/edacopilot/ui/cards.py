"""Rendering a `Card` as ipywidgets (ARCHITECTURE.md, Sections 9.4 and 13.2).

M7 made a `Card` plain data with a `to_text()` rendering; this draws the
same object. Nothing here decides anything — no card is built here, no
eligibility is read, no intent is invented. A widget renders what the
orchestrator already put in the card, and a button carries the `Intent`
the card already attached. That is what keeps Section 13.1's promise that
"a pure-Python API mirrors every action": the UI cannot do anything the
API cannot, because it has nothing to do it with.

**Colour is never the only signal.** Section 13.2 is explicit, and it
matters beyond accessibility guidelines: these cards are read by someone
deciding whether an assumption failed, and a red-green distinction is
invisible to roughly one man in twelve. Every status carries an icon and
the word as well as the colour, and `STATUS_MARKS` is the single place
that mapping lives so no renderer can drift from it.

**No JupyterLab-only APIs.** Section 13.2 requires VS Code notebooks and
Notebook 7 too, so this uses core ipywidgets (`Box`, `HTML`, `Button`,
`Accordion`, `Tab`, `Output`) and nothing from `ipylab`, `jupyterlab-*` or
the Lab-specific `display` machinery. Plots go in as PNG bytes in an
`Image` widget rather than through a matplotlib backend, which is the one
approach that behaves the same in all three.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import ipywidgets as widgets

from edacopilot.orchestrator.cards import ActionButton, Card
from edacopilot.orchestrator.intents import Intent
from edacore.contracts import (
    AssumptionCheck,
    CheckStatus,
    PostHocResult,
    TestResult,
    TransformRecord,
)

from .markup import to_html

# Section 13.2's palette, each paired with an icon and a word so the
# distinction survives greyscale, colour-blindness and a plain-text copy.
STATUS_MARKS: dict[CheckStatus, tuple[str, str, str]] = {
    CheckStatus.PASS: ("#2e7d32", "✓", "PASS"),
    CheckStatus.BORDERLINE: ("#ef6c00", "△", "BORDERLINE"),
    CheckStatus.FAIL: ("#c62828", "✕", "FAIL"),
    CheckStatus.NOT_APPLICABLE: ("#757575", "–", "N/A"),
    CheckStatus.UNTESTABLE: ("#757575", "?", "UNTESTABLE"),
}

# Border colour per card kind, so a result and a warning are distinguishable
# at a glance in a long scroll. Again never the only signal: the kind is in
# the header text too.
KIND_COLOURS: dict[str, str] = {
    "info": "#1f77b4",
    "question": "#7b1fa2",
    "consensus": "#2e7d32",
    "divergence": "#ef6c00",
    "result": "#1f77b4",
    "warning": "#c62828",
    "override_confirm": "#c62828",
    "export": "#00695c",
}

KIND_LABELS: dict[str, str] = {
    "info": "Info",
    "question": "Question",
    "consensus": "All personas agree",
    "divergence": "Personas disagree",
    "result": "Result",
    "warning": "Warning",
    "override_confirm": "Confirm override",
    "export": "Export",
}

Emit = Callable[[Intent], Any]
"""What a button does: hand an `Intent` back to the panel."""


def render_card(
    card: Card,
    emit: Emit,
    *,
    png: Callable[[str], bytes] | None = None,
    define: Callable[[str], str | None] | None = None,
) -> widgets.Widget:
    """One card as a widget tree.

    `png` resolves a `plot_ref` to image bytes and `define` resolves a
    glossary term; both are injected rather than imported so this module
    never reaches into a session.
    """
    children: list[widgets.Widget] = [_header(card)]
    if card.body_md.strip():
        children.append(widgets.HTML(to_html(card.body_md, define=define)))
    if card.diagnostics:
        children.append(diagnostics_widget(card.diagnostics, card.graded, define=define))
    if card.proposals:
        children.append(_proposals(card, emit))
    if card.result is not None:
        children.append(result_widget(card.result, define=define))
    if card.plots and png is not None:
        children.append(_plots(card.plots, png))
    if card.actions:
        children.append(action_row(card.actions, emit))

    box = widgets.VBox(children)
    box.add_class("edacopilot-card")
    box.layout = widgets.Layout(
        border=f"1px solid {KIND_COLOURS.get(card.kind, '#999999')}",
        border_left=f"5px solid {KIND_COLOURS.get(card.kind, '#999999')}",
        padding="10px 12px",
        margin="6px 0",
        width="100%",
    )
    return box


def _header(card: Card) -> widgets.HTML:
    colour = KIND_COLOURS.get(card.kind, "#999999")
    kind = KIND_LABELS.get(card.kind, card.kind)
    stage = f" · stage: {card.stage}" if card.stage else ""
    return widgets.HTML(
        f'<div style="margin-bottom:4px">'
        f'<span style="color:{colour};font-weight:600">{kind}</span>'
        f'<span style="color:#666">{stage}</span><br>'
        f'<span style="font-size:1.05em;font-weight:600">{_escape(card.title)}</span>'
        f"</div>"
    )


# --------------------------------------------------------------------------
# diagnostics (Section 13.2's collapsible, colour-coded table)
# --------------------------------------------------------------------------


def diagnostics_widget(
    checks: list[AssumptionCheck],
    graded: list[str] | None = None,
    *,
    define: Callable[[str], str | None] | None = None,
) -> widgets.Accordion:
    """The diagnostics table, collapsed by default.

    Collapsed because a proposal card carries a dozen checks and the
    decision is above them; open on the first click, because the whole
    argument of Section 1.3 is that the evidence must be one gesture away
    rather than absent.

    Verdicts and evidence are separated for the reason M7 recorded: some
    evidence reads FAIL while deciding nothing, and a reader who notices
    the table is padded with irrelevant failures stops reading it.
    """
    decisive = set(graded or [])
    verdicts = [c for c in checks if not decisive or c.fact_id in decisive]
    evidence = [c for c in checks if decisive and c.fact_id not in decisive]

    rows = [_diagnostics_table(verdicts, define=define)]
    if evidence:
        rows.append(
            widgets.HTML(
                '<div style="margin-top:8px;color:#666;font-size:0.9em">'
                "Also computed (evidence, not a verdict on any method)</div>"
            )
        )
        rows.append(_diagnostics_table(evidence, define=define))

    failing = sum(1 for c in verdicts if c.status is CheckStatus.FAIL)
    borderline = sum(1 for c in verdicts if c.status is CheckStatus.BORDERLINE)
    summary = f"Diagnostics ({len(verdicts)} checks"
    if failing or borderline:
        summary += f", {failing} failed, {borderline} borderline"
    summary += ")"

    accordion = widgets.Accordion(children=[widgets.VBox(rows)])
    accordion.set_title(0, summary)
    accordion.selected_index = None
    return accordion


def _diagnostics_table(
    checks: list[AssumptionCheck], *, define: Callable[[str], str | None] | None
) -> widgets.HTML:
    header = (
        "<tr>"
        '<th style="text-align:left;padding:2px 8px">Status</th>'
        '<th style="text-align:left;padding:2px 8px">Check</th>'
        '<th style="text-align:left;padding:2px 8px">Statistic</th>'
        '<th style="text-align:left;padding:2px 8px">What it means if ignored</th>'
        "</tr>"
    )
    rows = "".join(_diagnostics_row(check, define=define) for check in checks)
    return widgets.HTML(
        f'<table style="border-collapse:collapse;font-size:0.9em;width:100%">{header}{rows}</table>'
    )


def _diagnostics_row(check: AssumptionCheck, *, define: Callable[[str], str | None] | None) -> str:
    colour, icon, word = STATUS_MARKS[check.status]
    numbers = []
    if check.statistic is not None:
        numbers.append(f"{check.statistic:.4g}")
    if check.p_value is not None:
        numbers.append(f"p={check.p_value:.4g}")
    return (
        "<tr>"
        f'<td style="padding:2px 8px;color:{colour};white-space:nowrap">'
        f"<b>{icon} {word}</b></td>"
        f'<td style="padding:2px 8px"><code>{_escape(check.fact_id)}</code></td>'
        f'<td style="padding:2px 8px">{_escape(", ".join(numbers))}</td>'
        f'<td style="padding:2px 8px;color:#444">'
        f"{to_html(check.consequence, define=define, inline=True)}</td>"
        "</tr>"
    )


# --------------------------------------------------------------------------
# proposals (Section 13.2's side-by-side panels with one Accept each)
# --------------------------------------------------------------------------


def _proposals(card: Card, emit: Emit) -> widgets.Widget:
    """One panel per persona, each with its own Accept.

    Side by side rather than stacked, because Section 1.3's "show
    disagreement" only works if the alternatives can be compared without
    scrolling between them.
    """
    accepts = {
        action.intent.persona: action
        for action in card.actions
        if action.intent.type.value == "accept"
    }
    panels = [_proposal_panel(proposal, accepts, emit) for proposal in card.proposals]
    return widgets.HBox(panels, layout=widgets.Layout(flex_flow="row wrap", width="100%"))


def _proposal_panel(proposal: Any, accepts: dict[Any, Any], emit: Emit) -> widgets.Widget:
    status = proposal.eligibility or ""
    colour = {
        "eligible": "#2e7d32",
        "caveat": "#ef6c00",
        "ineligible": "#c62828",
    }.get(status, "#666666")
    method = proposal.function or "nothing to offer"
    concurs = (
        f'<div style="color:#666;font-size:0.85em">concurs with {proposal.concurs_with}</div>'
        if proposal.concurs_with
        else ""
    )
    head = widgets.HTML(
        f'<div style="font-weight:600">{_escape(proposal.display_name)}</div>'
        f'<div style="font-size:1.05em"><code>{_escape(method)}</code></div>'
        f'<div style="color:{colour};font-size:0.85em">{_escape(status)}</div>'
        f"{concurs}"
        f'<div style="margin-top:4px;font-size:0.9em">{_escape(proposal.rationale)}</div>'
    )

    children: list[widgets.Widget] = [head]
    action = accepts.get(proposal.persona) or accepts.get(None)
    if action is not None and proposal.function:
        children.append(_button(action, emit, style="success"))

    panel = widgets.VBox(children)
    panel.layout = widgets.Layout(
        border="1px solid #ddd", padding="8px", margin="4px", width="290px"
    )
    return panel


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------


def result_widget(
    result: TestResult | PostHocResult | TransformRecord,
    *,
    define: Callable[[str], str | None] | None = None,
) -> widgets.Widget:
    if isinstance(result, TestResult):
        return _test_result(result, define=define)
    if isinstance(result, PostHocResult):
        return _posthoc_result(result)
    return _transform_result(result)


def _test_result(
    result: TestResult, *, define: Callable[[str], str | None] | None
) -> widgets.Widget:
    rows: list[tuple[str, str]] = [("test", f"{result.function} — {result.estimand}")]
    rows.append((result.statistic_name, f"{result.statistic:.4g}"))
    if result.df is not None:
        rows.append(("df", _format_df(result.df)))
    if result.p_value is not None:
        rows.append(("p (raw)", f"{result.p_value:.4g}"))
    if result.p_adjusted is not None:
        rows.append(("p (adjusted)", f"{result.p_adjusted:.4g}"))
    if result.estimate is not None:
        rows.append(("estimate", f"{result.estimate:.4g}{_ci(result.ci, result.ci_level)}"))
    if result.effect_size is not None:
        magnitude = f" ({result.effect_magnitude})" if result.effect_magnitude else ""
        rows.append(
            (
                "effect size",
                f"{result.effect_size_name} = {result.effect_size:.4g}"
                f"{_ci(result.effect_size_ci, result.ci_level)}{magnitude}",
            )
        )
    if result.n:
        rows.append(("n used", ", ".join(f"{k}={v}" for k, v in result.n.items())))

    children: list[widgets.Widget] = [_definition_table(rows, define=define)]
    if result.warnings:
        children.append(_note_list("Warnings", result.warnings, "#ef6c00", define=define))
    if result.validity_notes:
        children.append(
            _note_list("Validity notes", result.validity_notes, "#1f77b4", define=define)
        )
    return widgets.VBox(children)


def _posthoc_result(result: PostHocResult) -> widgets.Widget:
    header = (
        "<tr>"
        '<th style="text-align:left;padding:2px 8px">Comparison</th>'
        '<th style="text-align:left;padding:2px 8px">p</th>'
        '<th style="text-align:left;padding:2px 8px">p (adjusted)</th>'
        '<th style="text-align:left;padding:2px 8px">estimate</th>'
        "</tr>"
    )
    rows = "".join(
        "<tr>"
        f'<td style="padding:2px 8px">{_escape(c.group_a)} vs {_escape(c.group_b)}</td>'
        f'<td style="padding:2px 8px">{c.p_value:.4g}</td>'
        f'<td style="padding:2px 8px">'
        f"{'—' if c.p_adjusted is None else format(c.p_adjusted, '.4g')}</td>"
        f'<td style="padding:2px 8px">'
        f"{'—' if c.estimate is None else format(c.estimate, '.4g')}</td>"
        "</tr>"
        for c in result.comparisons
    )
    return widgets.HTML(
        f'<div style="font-size:0.9em"><b>{_escape(result.method)}</b> — '
        f"{len(result.comparisons)} comparisons, adjusted by "
        f"{_escape(result.p_adjust_method or 'the procedure itself')} within this family."
        f'</div><table style="border-collapse:collapse;font-size:0.9em">{header}{rows}</table>'
    )


def _transform_result(record: TransformRecord) -> widgets.Widget:
    return _definition_table(
        [
            ("transform", record.function),
            ("columns", ", ".join(record.columns_affected)),
            ("rows", f"{record.rows_before} → {record.rows_after}"),
        ],
        define=None,
    )


def _definition_table(
    rows: list[tuple[str, str]], *, define: Callable[[str], str | None] | None
) -> widgets.HTML:
    body = "".join(
        f'<tr><td style="padding:2px 10px 2px 0;color:#666;white-space:nowrap">'
        f"{to_html(label, define=define, inline=True)}</td>"
        f'<td style="padding:2px 0">{to_html(value, define=define, inline=True)}</td></tr>'
        for label, value in rows
    )
    return widgets.HTML(f'<table style="font-size:0.95em">{body}</table>')


def _note_list(
    title: str,
    notes: list[str],
    colour: str,
    *,
    define: Callable[[str], str | None] | None,
) -> widgets.HTML:
    items = "".join(f"<li>{to_html(note, define=define, inline=True)}</li>" for note in notes)
    return widgets.HTML(
        f'<div style="margin-top:6px"><b style="color:{colour}">{title}</b>'
        f'<ul style="margin:2px 0 0 18px;font-size:0.9em">{items}</ul></div>'
    )


# --------------------------------------------------------------------------
# plots and actions
# --------------------------------------------------------------------------


def _plots(refs: list[str], png: Callable[[str], bytes]) -> widgets.Widget:
    """Plots as `Image` widgets holding PNG bytes.

    Not an `Output` widget capturing a matplotlib figure: that depends on
    an active kernel display hook and behaves differently in VS Code than
    in Lab. Bytes in an `Image` render identically in all three.
    """
    images: list[widgets.Widget] = []
    for ref in refs:
        try:
            data = png(ref)
        except Exception:  # noqa: BLE001 - a missing PNG must not kill the card
            images.append(widgets.HTML(f'<div style="color:#999">[plot {_escape(ref)}]</div>'))
            continue
        images.append(
            widgets.Image(value=data, format="png", layout=widgets.Layout(max_width="460px"))
        )
    return widgets.HBox(images, layout=widgets.Layout(flex_flow="row wrap"))


def action_row(actions: list[ActionButton], emit: Emit) -> widgets.Widget:
    """Section 9.5's next-step buttons. Shown, never pressed for the user."""
    return widgets.HBox(
        [_button(action, emit) for action in actions],
        layout=widgets.Layout(flex_flow="row wrap", margin="6px 0 0 0"),
    )


def _button(action: ActionButton, emit: Emit, *, style: str = "") -> widgets.Button:
    button = widgets.Button(
        description=action.label,
        button_style=style,
        tooltip=action.call or action.label,
        layout=widgets.Layout(width="auto", margin="2px 4px 2px 0"),
    )
    # The intent is carried on the widget so a test can assert what a
    # button would emit without clicking it, and so the click handler has
    # nothing to reconstruct.
    button.intent = action.intent
    button.on_click(lambda _button, intent=action.intent: emit(intent))
    return button


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _ci(ci: tuple[float, float] | None, level: float) -> str:
    return "" if ci is None else f"  {level:.0%} CI [{ci[0]:.4g}, {ci[1]:.4g}]"


def _format_df(df: float | tuple[float, float]) -> str:
    return f"{df[0]:.4g}, {df[1]:.4g}" if isinstance(df, tuple) else f"{df:.4g}"


def _escape(text: str) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
