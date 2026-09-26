"""The ipywidgets panel (ARCHITECTURE.md, Section 13.2).

The layout Section 13.2 draws: a header with session, branch, version and
stage; stage tabs showing visited state; a scrolling history of cards; and
a text box at the bottom.

**The panel owns no logic.** Every button hands an `Intent` to
`Orchestrator.handle`, which is the same method the Python API calls, and
the card that comes back is appended to the history. There is no code path
in this module that inspects eligibility, picks a persona or decides
anything — which is what makes Section 13.1's claim true rather than
aspirational, and what makes the conversation tests in
`tests/conversations/` cover the UI as well as the API.

**Three hosts, one set of widgets.** JupyterLab, Notebook 7 and VS Code
notebooks all render core ipywidgets; they do not all render the same
JupyterLab extensions, custom MIME types or script in HTML. So this uses
`Box`, `HTML`, `Button`, `Tab`, `Accordion`, `Textarea`, `Checkbox` and
`Image`, and nothing else. Plots arrive as PNG bytes rather than through a
matplotlib display hook for the same reason.

**The override flow is the one place the UI adds a step** (Section 7.5).
The API takes the reason as an argument; a panel has to collect it, so an
`override_confirm` card grows a text box, Accept stays disabled until it
is non-empty, and an INELIGIBLE override grows a second confirmation
checkbox that is also required. The gates are still enforced in the
orchestrator — this only makes them reachable with a mouse.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import ipywidgets as widgets

from edacopilot.llm.fallback import load_glossary, normalise_topic
from edacopilot.orchestrator.cards import Card
from edacopilot.orchestrator.intents import Intent, IntentType, button
from edacopilot.orchestrator.state_machine import STAGE_ORDER, Stage, applicable_stages

from .cards import KIND_COLOURS, action_row, diagnostics_widget, render_card
from .markup import to_html

if TYPE_CHECKING:  # pragma: no cover
    from edacopilot.session.session import Session

# How many cards the history keeps. A Jupyter output cell with hundreds of
# widget trees becomes unusable long before it becomes useful, and the
# provenance log is the real history.
MAX_HISTORY = 40


class Panel:
    """Section 13.2's panel, driving one `Session`."""

    def __init__(self, session: Session, *, name: str = "") -> None:
        self.session = session
        self.name = name or session.session_id
        self.history = widgets.VBox([])
        self.header = widgets.HTML()
        self.tabs = widgets.HBox([])
        self.entry = widgets.Text(
            placeholder="type a question, or a term to explain...",
            # `continuous_update=False` makes Enter (and blur) fire a single
            # value change, which is ipywidgets 8's replacement for the
            # deprecated `Text.on_submit`.
            continuous_update=False,
            layout=widgets.Layout(flex="1 1 auto"),
        )
        self.entry.observe(self._on_entry_change, names="value")
        self.send = widgets.Button(description="Send", button_style="primary")
        self.send.on_click(lambda _button: self._submit())

        self.root = widgets.VBox(
            [
                self.header,
                self.tabs,
                widgets.Box(
                    [self.history],
                    layout=widgets.Layout(max_height="640px", overflow="auto", width="100%"),
                ),
                widgets.HBox([self.entry, self.send], layout=widgets.Layout(width="100%")),
            ],
            layout=widgets.Layout(width="100%", border="1px solid #ccc", padding="8px"),
        )
        self._refresh_chrome()

    # ---- the turn loop, from the UI side --------------------------------

    def emit(self, intent: Intent) -> Card:
        """Hand an intent to the orchestrator and show what comes back.

        The entire UI goes through here, and it does nothing an API caller
        could not: `session.orchestrator.handle(intent)` is the same call.
        """
        card = self.session.orchestrator.handle(intent)
        self.show(card)
        return card

    def show(self, card: Card) -> widgets.Widget:
        """Append a card to the history and update the chrome."""
        widget = self.render(card)
        children = [*self.history.children, widget][-MAX_HISTORY:]
        self.history.children = tuple(children)
        self._refresh_chrome()
        return widget

    def render(self, card: Card) -> widgets.Widget:
        """One card as widgets, with this session's plots and glossary."""
        if card.kind == "override_confirm":
            return self._override_card(card)
        return render_card(card, self.emit, png=self._png, define=self._define)

    # ---- the override flow (Section 7.5) --------------------------------

    def _override_card(self, card: Card) -> widgets.Widget:
        """A reason box, and for an ineligible method a second confirmation.

        Accept is disabled until the reason is non-empty — and, when the
        method is ineligible, until the confirmation is ticked too. The
        orchestrator enforces both regardless; disabling the button is so
        the user is not invited to press something that will be refused.
        """
        target = _override_action(card)
        needs_confirmation = bool(target and target.intent.payload.get("confirm"))
        function = str(target.intent.payload.get("function", "")) if target else ""

        children: list[widgets.Widget] = [
            widgets.HTML(
                f'<div style="color:{KIND_COLOURS["override_confirm"]};font-weight:600">'
                f"Confirm override</div>"
                f'<div style="font-size:1.05em;font-weight:600">{card.title}</div>'
            ),
            widgets.HTML(to_html(card.body_md, define=self._define)),
        ]
        if card.diagnostics:
            children.append(diagnostics_widget(card.diagnostics, card.graded, define=self._define))

        reason = widgets.Textarea(
            placeholder="Why are you choosing this method? This is stored in the "
            "provenance log and written into the exported code as a comment.",
            layout=widgets.Layout(width="100%", height="70px"),
        )
        confirm = widgets.Checkbox(
            value=False,
            description="I understand the failed assumption above and want to run it anyway",
            indent=False,
            layout=widgets.Layout(width="100%"),
        )
        accept = widgets.Button(
            description=f"Run {function}" if function else "Run it",
            button_style="danger" if needs_confirmation else "warning",
            disabled=True,
        )

        def _update(_change: Any = None) -> None:
            accept.disabled = not reason.value.strip() or (needs_confirmation and not confirm.value)

        reason.observe(_update, names="value")
        confirm.observe(_update, names="value")
        accept.on_click(
            lambda _b: self.emit(
                button(
                    IntentType.OVERRIDE,
                    function=function,
                    reason=reason.value.strip(),
                    confirm=True,
                )
            )
        )
        # Exposed so a test can drive the flow without a front end.
        accept.intent_source = (reason, confirm)

        children.append(reason)
        if needs_confirmation:
            children.append(confirm)
        alternatives = [a for a in card.actions if a is not target]
        children.append(
            widgets.HBox(
                [accept, *action_row(alternatives, self.emit).children]
                if alternatives
                else [accept],
                layout=widgets.Layout(flex_flow="row wrap", margin="6px 0 0 0"),
            )
        )

        box = widgets.VBox(children)
        box.layout = widgets.Layout(
            border=f"1px solid {KIND_COLOURS['override_confirm']}",
            border_left=f"5px solid {KIND_COLOURS['override_confirm']}",
            padding="10px 12px",
            margin="6px 0",
            width="100%",
        )
        return box

    # ---- chrome ----------------------------------------------------------

    def _refresh_chrome(self) -> None:
        self.header.value = (
            f'<div style="font-weight:600">edacopilot · {self.name} · '
            f"branch: {self.session.active_branch} · {self.session.version} · "
            f"stage: {self.session.stage.upper()}</div>"
        )
        self.tabs.children = tuple(self._stage_buttons())

    def _stage_buttons(self) -> list[widgets.Button]:
        """Stage tabs with visited state (Section 13.2).

        Visited state is shown with a tick and a weight change as well as
        a colour, for the same reason the diagnostics table is: colour
        alone is not a signal everyone receives.
        """
        state = self.session.orchestrator.state
        try:
            stages = applicable_stages(self.session.store.profile())
        except Exception:  # noqa: BLE001 - a panel must render before profiling succeeds
            stages = list(STAGE_ORDER)

        buttons: list[widgets.Button] = []
        for stage in stages:
            if stage is Stage.INGEST:
                continue
            visited = stage in state.visited
            current = stage is state.stage
            label = f"{'✓ ' if visited else ''}{stage.value}"
            tab = widgets.Button(
                description=label,
                button_style="info" if current else "",
                tooltip=(
                    f"{stage.value}: "
                    f"{'visited' if visited else 'not visited yet'}"
                    f"{' (current)' if current else ''}"
                ),
                layout=widgets.Layout(width="auto", margin="0 2px"),
            )
            if current:
                tab.style.font_weight = "bold"
            tab.intent = button(IntentType.GOTO_STAGE, target_stage=stage.value)
            tab.on_click(
                lambda _b, name=stage.value: self.emit(
                    button(IntentType.GOTO_STAGE, target_stage=name)
                )
            )
            buttons.append(tab)
        return buttons

    # ---- input -----------------------------------------------------------

    def _on_entry_change(self, change: Any) -> None:
        """Enter in the text box sends, and only then.

        Guarded against the change `_submit` itself causes when it clears
        the box, which would otherwise recurse.
        """
        if str(change.get("new", "")).strip():
            self._submit()

    def _submit(self) -> Card | None:
        text = self.entry.value.strip()
        if not text:
            return None
        self.entry.value = ""
        return self.ask(text)

    def ask(self, text: str) -> Card:
        """What pressing Send does, callable directly (and from a test)."""
        card = self.session.ask(text)
        self.show(card)
        return card

    # ---- injected lookups -------------------------------------------------

    def _png(self, plot_ref: str) -> bytes:
        return self.session.plots.png_bytes(plot_ref)

    def _define(self, word: str) -> str | None:
        """A glossary definition for a word in card text, or None.

        Matched on the normalised word so `mann-whitney` and
        `mann_whitney` both resolve, and only on terms long enough to be
        worth a tooltip: underlining every "the" would make the card
        unreadable and teach nothing.
        """
        if len(word) < 4:
            return None
        terms, aliases = load_glossary()
        key = normalise_topic(word).replace(" ", "_")
        if key in terms:
            return terms[key]
        if key in aliases:
            return terms[aliases[key]]
        return None

    # ---- display ----------------------------------------------------------

    def display(self) -> None:  # pragma: no cover - needs a live kernel
        from IPython.display import display

        display(self.root)  # type: ignore[no-untyped-call]

    def _ipython_display_(self) -> None:  # pragma: no cover - host callback
        self.display()


def _override_action(card: Card) -> Any:
    """The card's own "run it anyway" action, if it has one.

    `confirm=True` in the payload is what distinguishes the INELIGIBLE
    card from the CAVEAT one — the orchestrator sets it on the former only
    (Section 7.5), and it is the contract between the two modules.
    """
    for action in card.actions:
        if action.intent.type is IntentType.OVERRIDE:
            return action
    return None
