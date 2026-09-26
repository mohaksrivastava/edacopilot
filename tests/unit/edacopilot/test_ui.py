"""The Jupyter UI (ARCHITECTURE.md, Section 13).

The UI's whole claim is that it adds a view and no behaviour: every button
emits the `Intent` the card already carried, into the same
`Orchestrator.handle` the Python API calls. So the central test here is
not that a widget looks right — it is that **clicking a button leaves the
session in exactly the state the equivalent API call would**. If that
holds, M7's conversation tests cover the UI too, and the only bugs M8 can
introduce are rendering bugs.

Everything runs in-process. ipywidgets objects are ordinary Python objects
whose `on_click` handlers can be invoked directly; no kernel, no browser
and no front end is involved, which is why this can run in CI at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ipywidgets as widgets
import numpy as np
import pandas as pd
import pytest

import edacopilot as eda
from edacopilot.orchestrator import Card, Intent, IntentType
from edacopilot.orchestrator.cards import ActionButton, ProposalView
from edacopilot.session import Session
from edacopilot.ui import (
    KIND_COLOURS,
    STATUS_MARKS,
    Panel,
    run_magic,
    to_html,
)
from edacopilot.ui.magics import EdaMagicError
from edacore.contracts import (
    AssumptionCheck,
    CheckStatus,
    PairwiseComparison,
    PostHocResult,
    TestResult,
    TransformRecord,
)
from tests.scenarios.generators import ordinal_as_numeric, worked_example_7_4


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "v": np.concatenate([rng.normal(0, 1, 30), rng.normal(1.2, 1, 30)]),
            "g": ["a"] * 30 + ["b"] * 30,
        }
    )


@pytest.fixture
def session(tmp_path: Path) -> Session:
    return Session.start(_frame(), session_id="ui", root=tmp_path)


@pytest.fixture
def panel(session: Session) -> Panel:
    return Panel(session, name="test")


def _buttons(widget: Any) -> list[widgets.Button]:
    """Every Button anywhere in a widget tree."""
    found: list[widgets.Button] = []
    if isinstance(widget, widgets.Button):
        found.append(widget)
    for child in getattr(widget, "children", ()):
        found.extend(_buttons(child))
    if isinstance(widget, widgets.Accordion):
        for child in widget.children:
            found.extend(_buttons(child))
    return found


def _html(widget: Any) -> str:
    """All HTML text anywhere in a widget tree."""
    parts: list[str] = []
    if isinstance(widget, widgets.HTML):
        parts.append(widget.value)
    for child in getattr(widget, "children", ()):
        parts.append(_html(child))
    return " ".join(parts)


# --------------------------------------------------------------------------
# Every card kind renders
# --------------------------------------------------------------------------


def _check(status: CheckStatus = CheckStatus.FAIL) -> AssumptionCheck:
    return AssumptionCheck(
        fact_id="normality.shapiro.v.g=a",
        assumption="normality",
        method="shapiro",
        statistic=0.86,
        p_value=0.0003,
        threshold="p < 0.05 -> fail",
        status=status,
        consequence="Parametric tests that assume normality may mislead.",
    )


def _result() -> TestResult:
    return TestResult(
        fact_id="t.1",
        function="welch_t",
        estimand="difference in means",
        statistic=2.1,
        statistic_name="t",
        df=57.2,
        p_value=0.04,
        p_adjusted=0.08,
        estimate=0.9,
        ci=(0.05, 1.75),
        effect_size=0.54,
        effect_size_name="hedges_g",
        effect_size_ci=(0.02, 1.05),
        effect_magnitude="medium",
        n={"a": 30, "b": 30},
        warnings=["a warning"],
        validity_notes=["Association only; it does not show causation."],
    )


def _card(kind: str) -> Card:
    """One card of each kind, each carrying the fields that kind uses."""
    common: dict[str, Any] = {"kind": kind, "title": f"A {kind} card", "stage": "hypothesis"}
    if kind in ("consensus", "divergence"):
        return Card(
            **common,
            body_md="Professor + Consultant → Mann-Whitney.",
            diagnostics=[_check(), _check(CheckStatus.PASS)],
            graded=["normality.shapiro.v.g=a"],
            proposals=[
                ProposalView(
                    persona="professor",
                    display_name="Professor",
                    function="mann_whitney",
                    eligibility="eligible",
                    rationale="Its assumptions hold here.",
                ),
                ProposalView(
                    persona="maverick",
                    display_name="Maverick",
                    function="yuen_trimmed_t",
                    eligibility="eligible",
                    rationale="Robust to the skew.",
                ),
            ],
            actions=[
                ActionButton(
                    label="Professor: mann_whitney",
                    intent=Intent(type=IntentType.ACCEPT, persona="professor"),
                    call='session.accept("professor")',
                ),
                ActionButton(
                    label="Maverick: yuen_trimmed_t",
                    intent=Intent(type=IntentType.ACCEPT, persona="maverick"),
                    call='session.accept("maverick")',
                ),
            ],
        )
    if kind == "result":
        return Card(**common, body_md="Test #1.", result=_result())
    if kind == "override_confirm":
        return Card(
            **common,
            body_md="A hard assumption failed.",
            diagnostics=[_check()],
            actions=[
                ActionButton(
                    label="Yes, run it anyway",
                    intent=Intent(
                        type=IntentType.OVERRIDE,
                        payload={"function": "student_t", "reason": "", "confirm": True},
                    ),
                    call="session.override(...)",
                )
            ],
        )
    if kind == "export":
        return Card(**common, body_md="The notebook is ready.", result=None)
    return Card(**common, body_md="Some **body** text with `code`.")


ALL_KINDS = [
    "info",
    "question",
    "consensus",
    "divergence",
    "result",
    "warning",
    "override_confirm",
    "export",
]


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_every_card_kind_renders_to_a_widget_tree(kind: str, panel: Panel) -> None:
    """Section 9.4 names eight kinds; each has to draw."""
    widget = panel.render(_card(kind))
    assert isinstance(widget, widgets.Widget)
    assert _html(widget).strip(), f"the {kind} card rendered no text"


@pytest.mark.parametrize("kind", ALL_KINDS)
def test_every_card_kind_shows_its_title_and_kind(kind: str, panel: Panel) -> None:
    html = _html(panel.render(_card(kind)))
    assert f"A {kind} card" in html
    assert KIND_COLOURS[kind] in html


def test_a_transform_result_renders(panel: Panel) -> None:
    record = TransformRecord(
        function="log_transform",
        columns_affected=["v"],
        rows_before=60,
        rows_after=60,
    )
    html = _html(panel.render(Card(kind="result", title="t", result=record)))
    assert "log_transform" in html


def test_a_posthoc_result_renders_its_comparisons(panel: Panel) -> None:
    posthoc = PostHocResult(
        fact_id="ph.1",
        function="dunn_test",
        method="Dunn",
        p_adjust_method="holm",
        comparisons=[PairwiseComparison(group_a="a", group_b="b", p_value=0.01, p_adjusted=0.03)],
    )
    html = _html(panel.render(Card(kind="result", title="t", result=posthoc)))
    assert "a vs b" in html or ("a" in html and "b" in html)
    assert "holm" in html


def test_a_result_card_shows_everything_section_13_2_requires(panel: Panel) -> None:
    """Effect size, CI, raw and adjusted p, n used, warnings, validity notes."""
    html = _html(panel.render(_card("result")))
    for expected in [
        "hedges_g",
        "0.54",
        "0.02",  # effect-size CI low
        "0.04",  # raw p
        "0.08",  # adjusted p
        "a=30",
        "medium",
        "a warning",
        "does not show causation",
    ]:
        assert expected in html, f"{expected!r} missing from the result card"


# --------------------------------------------------------------------------
# Status is never signalled by colour alone (Section 13.2)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("status", list(CheckStatus))
def test_every_status_carries_an_icon_and_a_word_as_well_as_a_colour(
    status: CheckStatus, panel: Panel
) -> None:
    """Roughly one man in twelve cannot separate the red from the green,
    and these cards exist to tell someone whether an assumption failed."""
    colour, icon, word = STATUS_MARKS[status]
    card = Card(kind="info", title="t", diagnostics=[_check(status)])
    html = _html(panel.render(card))
    assert colour in html
    assert icon in html, f"{status.value} has no icon"
    assert word in html, f"{status.value} has no word"


def test_the_status_marks_cover_every_status() -> None:
    assert set(STATUS_MARKS) == set(CheckStatus)


def test_icons_and_words_are_distinct_between_statuses() -> None:
    """An icon shared by PASS and FAIL would be no better than colour."""
    icons = [icon for _, icon, _ in STATUS_MARKS.values()]
    words = [word for _, _, word in STATUS_MARKS.values()]
    assert len(set(words)) == len(words)
    # N/A and UNTESTABLE may share a colour; their icons must still differ.
    assert len(set(icons)) == len(icons)


def test_diagnostics_are_collapsed_by_default(panel: Panel) -> None:
    """The decision is above the evidence; the evidence is one click away."""
    widget = panel.render(_card("divergence"))
    accordions = _find(widget, widgets.Accordion)
    assert accordions, "the diagnostics table is not collapsible"
    assert accordions[0].selected_index is None


def test_evidence_is_shown_apart_from_verdicts(panel: Panel) -> None:
    card = Card(
        kind="consensus",
        title="t",
        diagnostics=[_check(), _check(CheckStatus.PASS).model_copy(update={"fact_id": "e.1"})],
        graded=["normality.shapiro.v.g=a"],
    )
    html = _html(panel.render(card))
    assert "Also computed" in html


def _find(widget: Any, kind: type) -> list[Any]:
    found = [widget] if isinstance(widget, kind) else []
    for child in getattr(widget, "children", ()):
        found.extend(_find(child, kind))
    return found


# --------------------------------------------------------------------------
# The property that matters: a click equals the API call
# --------------------------------------------------------------------------


def _api_state(session: Session) -> dict[str, Any]:
    """What a session looks like afterwards, for comparing two routes."""
    return {
        "stage": session.stage,
        "version": session.version,
        "branch": session.active_branch,
        "steps": session.steps(),
        "ledger": session.ledger(),
        "code": session.code(),
    }


def test_every_action_button_carries_a_valid_intent(panel: Panel, session: Session) -> None:
    session.ask(
        goal="compare_groups",
        outcome="v",
        group="g",
        design="independent",
        confirmed_by_user=["design"],
    )
    for kind in ALL_KINDS:
        for button in _buttons(panel.render(_card(kind))):
            intent = getattr(button, "intent", None)
            if intent is None:
                continue
            assert isinstance(intent, Intent)
            assert intent.type in set(IntentType)
            assert intent.is_confident, f"{button.description} emits an unconfident intent"


def test_clicking_accept_leaves_the_same_state_as_the_api_call(tmp_path: Path) -> None:
    """The whole claim of Section 13.1, as a test.

    Two sessions, identical data. One is driven through the API, the other
    by invoking the button's own click handler. The resulting sessions must
    be indistinguishable.
    """
    fields = {
        "goal": "compare_groups",
        "outcome": "v",
        "group": "g",
        "design": "independent",
        "confirmed_by_user": ["design"],
    }

    through_ui = Session.start(_frame(), session_id="ui2", root=tmp_path)
    ui = Panel(through_ui)
    card = through_ui.ask(**fields)
    ui.show(card)
    accept = next(
        b
        for b in _buttons(ui.render(card))
        if getattr(b, "intent", None) is not None and b.intent.type is IntentType.ACCEPT
    )
    # Whatever this card offered -- a consensus button carries no persona,
    # a divergence button carries one -- the API call it stands for is the
    # same one with the same argument.
    persona = accept.intent.persona
    accept.click()

    through_api = Session.start(_frame(), session_id="api", root=tmp_path)
    through_api.ask(**fields)
    through_api.accept(persona)

    assert through_ui.provenance.steps, "clicking Accept recorded nothing"
    assert _api_state(through_ui) == _api_state(through_api)


@pytest.mark.parametrize(
    ("label", "api", "intent"),
    [
        (
            "goto",
            lambda s: s.goto_stage("explore"),
            Intent(type=IntentType.GOTO_STAGE, target_stage="explore"),
        ),
        ("skip", lambda s: s.skip(), Intent(type=IntentType.SKIP)),
        (
            "explain",
            lambda s: s.explain("sphericity"),
            Intent(type=IntentType.EXPLAIN, payload={"topic": "sphericity"}),
        ),
        (
            "branch",
            lambda s: s.branch("side"),
            Intent(type=IntentType.BRANCH, payload={"name": "side"}),
        ),
    ],
)
def test_each_intent_routes_the_same_through_the_panel(
    tmp_path: Path, label: str, api: Any, intent: Intent
) -> None:
    through_api = Session.start(_frame(), session_id=f"api-{label}", root=tmp_path)
    api(through_api)

    through_ui = Session.start(_frame(), session_id=f"ui-{label}", root=tmp_path)
    Panel(through_ui).emit(intent)

    assert _api_state(through_ui) == _api_state(through_api)


def test_a_stage_tab_click_is_a_goto_stage(tmp_path: Path) -> None:
    session = Session.start(_frame(), session_id="tabs", root=tmp_path)
    panel = Panel(session)
    tab = next(b for b in panel.tabs.children if "explore" in b.description)
    tab.click()
    assert session.stage == "explore"


def test_stage_tabs_show_visited_state_without_relying_on_colour(tmp_path: Path) -> None:
    session = Session.start(_frame(), session_id="visited", root=tmp_path)
    panel = Panel(session)
    session.goto_stage("profile")
    panel._refresh_chrome()

    labels = {b.description.lstrip("✓ ").strip(): b.description for b in panel.tabs.children}
    assert labels["profile"].startswith("✓"), "a visited stage is not marked"
    assert not labels["hypothesis"].startswith("✓")
    tooltips = {b.description: b.tooltip for b in panel.tabs.children}
    assert any("visited" in tip for tip in tooltips.values())
    assert any("not visited yet" in tip for tip in tooltips.values())


def test_no_button_executes_anything_by_being_rendered(tmp_path: Path) -> None:
    """Rule 3 through the UI: drawing a card is not accepting it."""
    session = Session.start(_frame(), session_id="render", root=tmp_path)
    panel = Panel(session)
    card = session.ask(
        goal="compare_groups",
        outcome="v",
        group="g",
        design="independent",
        confirmed_by_user=["design"],
    )
    panel.show(card)
    for kind in ALL_KINDS:
        panel.render(_card(kind))
    assert len(session.provenance) == 0
    assert session.test_ledger.n_tests == 0


# --------------------------------------------------------------------------
# The override flow in the UI (Section 7.5)
# --------------------------------------------------------------------------


@pytest.fixture
def ineligible(tmp_path: Path) -> tuple[Session, Panel, Card]:
    session = Session.start(ordinal_as_numeric(), session_id="inelig", root=tmp_path)
    panel = Panel(session)
    session.ask(
        goal="association",
        x="satisfaction",
        y="tenure_months",
        design="independent",
        confirmed_by_user=["design", "x", "y"],
    )
    card = session.override("cochran_armitage_trend")
    assert card.kind == "override_confirm"
    return session, panel, card


def test_accept_is_disabled_until_a_reason_is_typed(
    ineligible: tuple[Session, Panel, Card],
) -> None:
    _session, panel, card = ineligible
    widget = panel.render(card)
    accept = _buttons(widget)[0]
    reason, confirm = accept.intent_source

    assert accept.disabled, "Accept was enabled with no reason"
    reason.value = "the reviewer asked for it"
    assert accept.disabled, "an ineligible override needs the confirmation too"
    confirm.value = True
    assert not accept.disabled


def test_a_whitespace_reason_does_not_count(
    ineligible: tuple[Session, Panel, Card],
) -> None:
    _session, panel, card = ineligible
    accept = _buttons(panel.render(card))[0]
    reason, confirm = accept.intent_source
    confirm.value = True
    reason.value = "   "
    assert accept.disabled


def test_the_ineligible_card_shows_the_failed_assumption_and_its_consequence(
    ineligible: tuple[Session, Panel, Card],
) -> None:
    _session, panel, card = ineligible
    html = _html(panel.render(card))
    assert card.diagnostics
    for check in card.diagnostics:
        assert check.consequence[:40] in html


def test_completing_the_ui_override_matches_the_api(tmp_path: Path) -> None:
    reason = "the reviewer asked for a trend test"

    through_api = Session.start(ordinal_as_numeric(), session_id="oapi", root=tmp_path)
    fields = {
        "goal": "association",
        "x": "satisfaction",
        "y": "tenure_months",
        "design": "independent",
        "confirmed_by_user": ["design", "x", "y"],
    }
    through_api.ask(**fields)
    through_api.override("cochran_armitage_trend", reason, confirm=True)

    through_ui = Session.start(ordinal_as_numeric(), session_id="oui", root=tmp_path)
    panel = Panel(through_ui)
    through_ui.ask(**fields)
    card = through_ui.override("cochran_armitage_trend")
    accept = _buttons(panel.render(card))[0]
    text, confirm = accept.intent_source
    text.value = reason
    confirm.value = True
    accept.click()

    assert _api_state(through_ui) == _api_state(through_api)
    assert through_ui.provenance.steps[-1].override_reason == reason


def test_a_caveat_override_needs_no_second_confirmation(tmp_path: Path) -> None:
    from tests.scenarios.generators import heteroscedastic_groups

    session = Session.start(heteroscedastic_groups(), session_id="cav", root=tmp_path)
    panel = Panel(session)
    session.ask(
        goal="compare_groups",
        outcome="value",
        group="group",
        design="independent",
        confirmed_by_user=["design"],
    )
    pending = session.orchestrator.state.pending
    assert pending is not None
    caveated = next(
        c.function
        for c in pending.candidates.candidates
        if c.eligibility.value == "caveat"
        and c.function not in {p.function for p in pending.verdict.picks}
    )
    card = session.override(caveated)
    accept = _buttons(panel.render(card))[0]
    reason, _confirm = accept.intent_source

    assert accept.disabled
    reason.value = "the team reports this test by convention"
    assert not accept.disabled, "a caveat override should need only the reason"


# --------------------------------------------------------------------------
# Glossary terms are clickable (Section 13)
# --------------------------------------------------------------------------


def test_a_glossary_term_in_card_text_gets_its_definition(panel: Panel) -> None:
    html = to_html("The sphericity assumption failed.", define=panel._define)
    assert "<abbr" in html
    assert "repeated-measures" in html


def test_an_unknown_word_is_left_alone(panel: Panel) -> None:
    html = to_html("The frobnicator failed.", define=panel._define)
    assert "<abbr" not in html


def test_short_words_are_not_linked(panel: Panel) -> None:
    """Underlining every "the" would make a card unreadable."""
    html = to_html("the p is low", define=panel._define)
    assert html.count("<abbr") <= 1


def test_terms_inside_code_spans_are_not_linked(panel: Panel) -> None:
    """A term already shown as code is already labelled."""
    html = to_html("Use `mann_whitney` here.", define=panel._define)
    assert "<code" in html
    assert "<abbr" not in html.split("<code")[1].split("</code>")[0]


def test_definitions_are_escaped_into_the_tooltip(panel: Panel) -> None:
    html = to_html("sphericity", define=panel._define)
    assert 'title="' in html
    assert "<script" not in html


def test_diagnostics_consequences_get_glossary_links(panel: Panel) -> None:
    html = _html(panel.render(_card("divergence")))
    assert "<abbr" in html, "no term in the diagnostics table was linked"


# --------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------


def test_markup_renders_the_constructs_cards_use() -> None:
    html = to_html(
        "# Heading\n\n"
        "Some **bold** and `code`.\n\n"
        "- one\n- two\n\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n\n"
        "```python\nx = 1\n```"
    )
    for fragment in ["<b>bold</b>", "<code", "<ul", "<li>one</li>", "<table", "<pre"]:
        assert fragment in html, f"{fragment} missing"


def test_markup_escapes_interpolated_data() -> None:
    """Card text is written by this project; the values in it are not.

    A column literally called `<script>` is far-fetched, but an override
    reason typed by a user is not, and it lands in card text.
    """
    html = to_html("Reason: <script>alert(1)</script>")
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_markup_escapes_inside_a_table_cell() -> None:
    html = to_html("| col |\n|---|\n| <b>x</b> |")
    assert "&lt;b&gt;x&lt;/b&gt;" in html


# --------------------------------------------------------------------------
# Plots in cards (Section 13.2)
# --------------------------------------------------------------------------


def test_plots_render_as_image_widgets(tmp_path: Path) -> None:
    """PNG bytes in an `Image`, not a matplotlib display hook: that is the
    one approach that behaves the same in Lab, Notebook 7 and VS Code."""
    session = Session.start(worked_example_7_4(), session_id="plots", root=tmp_path)
    panel = Panel(session)
    card = session.ask(
        goal="compare_groups",
        outcome="income",
        group="gender",
        design="independent",
        confirmed_by_user=["design"],
    )
    assert card.plots, "the proposal card carried no plots"
    images = _find(panel.render(card), widgets.Image)
    assert len(images) == len(card.plots)
    assert all(image.value[:4] == b"\x89PNG" for image in images)


def test_a_missing_plot_does_not_break_the_card(panel: Panel) -> None:
    card = Card(kind="info", title="t", plots=["nope"])
    html = _html(panel.render(card))
    assert "nope" in html


# --------------------------------------------------------------------------
# Entry points and magics (Section 13.1)
# --------------------------------------------------------------------------


def test_start_returns_a_session_with_a_panel(tmp_path: Path) -> None:
    session = eda.start(_frame(), session_id="s", root=tmp_path, display=False, name="demo")
    assert isinstance(session, Session)
    assert isinstance(session._panel, Panel)
    assert session._panel.name == "demo"
    assert len(session._panel.history.children) == 1, "the profile card was not shown"
    assert session.stage == "profile"


def test_start_records_the_target_in_config(tmp_path: Path) -> None:
    session = eda.start(_frame(), session_id="t", root=tmp_path, display=False, target="v")
    assert session.config["target"] == "v"


def test_the_magics_mirror_the_api(tmp_path: Path) -> None:
    namespace: dict[str, Any] = {"df": _frame()}
    session = run_magic("start df --name demo", {**namespace, "_eda_session": None})
    assert isinstance(session, Session)


def test_every_magic_command_resolves_to_a_session_call(tmp_path: Path) -> None:
    session = Session.start(_frame(), session_id="magic", root=tmp_path)
    namespace: dict[str, Any] = {"_eda_session": session, "df": _frame()}

    assert run_magic("explain sphericity", namespace).kind == "info"
    assert run_magic("goto explore", namespace).stage == "explore"
    assert run_magic("skip", namespace) is not None
    assert run_magic("branch side", namespace).kind == "info"
    assert run_magic("switch main", namespace).kind == "info"
    assert run_magic("ledger", namespace).kind == "info"
    assert isinstance(run_magic("steps", namespace), list)
    assert isinstance(run_magic("code", namespace), str)
    assert len(session.provenance) == 0, "a magic executed something"


def test_a_magic_without_a_session_says_so() -> None:
    with pytest.raises(EdaMagicError, match="no session yet"):
        run_magic("ask 'anything'", {})


def test_an_unknown_magic_command_says_so(tmp_path: Path) -> None:
    session = Session.start(_frame(), session_id="bad", root=tmp_path)
    with pytest.raises(EdaMagicError, match="unknown command"):
        run_magic("frobnicate", {"_eda_session": session})


def test_start_magic_needs_a_dataframe_in_scope() -> None:
    with pytest.raises(EdaMagicError, match="no variable called"):
        run_magic("start nope", {})


def test_the_ask_magic_matches_the_api(tmp_path: Path) -> None:
    through_api = Session.start(_frame(), session_id="aapi", root=tmp_path)
    through_api.explain("what is a p-value?")

    through_magic = Session.start(_frame(), session_id="amag", root=tmp_path)
    run_magic('explain "what is a p-value?"', {"_eda_session": through_magic})

    assert _api_state(through_magic) == _api_state(through_api)


# --------------------------------------------------------------------------
# Host portability (Section 13.2)
# --------------------------------------------------------------------------


def test_only_core_ipywidgets_are_used() -> None:
    """Section 13.2: "avoid JupyterLab-only APIs".

    Checked against the source rather than at runtime, because the failure
    it guards against is an import that works on the developer's machine
    and not in VS Code.
    """
    import ast

    import edacopilot.ui.cards as cards_module
    import edacopilot.ui.magics as magics_module
    import edacopilot.ui.markup as markup_module
    import edacopilot.ui.panel as panel_module

    # Lab-only or host-specific packages. Checked as *imports* rather than
    # as text, so a module that merely explains why it avoids one does not
    # trip its own guard.
    forbidden = {"ipylab", "jupyterlab", "jupyterlab_widgets", "jupyter_server", "notebook"}
    for module in (cards_module, panel_module, markup_module, magics_module):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        offending = imported & forbidden
        assert not offending, f"{module.__name__} imports {sorted(offending)}"


def test_the_panel_builds_without_a_kernel(session: Session) -> None:
    """Every test in this file already proves this, but stating it makes
    the CI requirement explicit: no kernel, no browser, no front end."""
    panel = Panel(session)
    assert isinstance(panel.root, widgets.VBox)
    assert panel.root.children


def test_no_deprecated_widget_apis_are_used(session: Session, recwarn: Any) -> None:
    """ipywidgets 8 deprecated `Text.on_submit` and rejects unknown Layout
    traits. Both fail loudly enough in a kernel to be worth pinning."""
    panel = Panel(session)
    panel.show(_card("info"))
    messages = [str(w.message) for w in recwarn]
    assert not [m for m in messages if "on_submit" in m or "unrecognized arguments" in m], messages


# --------------------------------------------------------------------------
# The strongest form: every real card from every golden transcript renders
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "worked_example_7_4",
        "paired_as_independent",
        "ineligible_override",
        "undo_then_branch",
        "kruskal_then_posthoc",
        "jump_with_missing_data",
    ],
)
def test_every_card_a_real_transcript_produces_renders(name: str, tmp_path: Path) -> None:
    """The synthetic cards above cover each *kind*; these cover each card
    the system actually builds, with the fields it actually fills.

    A card kind that renders on a hand-made example and not on the real
    thing — a result with no CI, a question with no actions, a divergence
    whose proposal has `function=None` — would pass the kind test and fail
    in front of a user.
    """
    from tests.conversations.transcripts import by_name

    transcript = by_name(name)
    session = transcript.start(tmp_path)
    panel = Panel(session)
    for step in transcript.steps:
        card = step.run(session)
        widget = panel.show(card)
        assert isinstance(widget, widgets.Widget)
        assert _html(widget).strip(), f"{name}: a {card.kind} card rendered no text"


def test_a_whole_transcript_driven_through_the_panel_matches_the_api(tmp_path: Path) -> None:
    """Section 13.1's claim at full length: the same six turns, once
    through the API and once emitting each card's own buttons, end in the
    same session state."""
    fields = {
        "goal": "compare_groups",
        "outcome": "income",
        "group": "gender",
        "design": "independent",
        "confirmed_by_user": ["design"],
    }

    through_api = Session.start(worked_example_7_4(), session_id="tapi", root=tmp_path)
    through_api.goto_stage("profile")
    through_api.ask(**fields)
    through_api.accept("professor")

    through_ui = Session.start(worked_example_7_4(), session_id="tui", root=tmp_path)
    panel = Panel(through_ui)
    profile_tab = next(b for b in panel.tabs.children if "profile" in b.description)
    profile_tab.click()
    card = through_ui.ask(**fields)
    panel.show(card)
    accept = next(
        b
        for b in _buttons(panel.render(card))
        if getattr(b, "intent", None) is not None
        and b.intent.type is IntentType.ACCEPT
        and b.intent.persona == "professor"
    )
    accept.click()

    assert _api_state(through_ui) == _api_state(through_api)


# --------------------------------------------------------------------------
# Does it actually render in the three hosts? (Section 13.2)
# --------------------------------------------------------------------------

WIDGET_MIME = "application/vnd.jupyter.widget-view+json"


def test_the_panel_emits_the_mime_type_all_three_hosts_render(session: Session) -> None:
    """JupyterLab, Notebook 7 and VS Code all render widgets through
    `application/vnd.jupyter.widget-view+json`. A panel that produced
    anything else — a custom MIME type, an HTML blob, a Lab-only view —
    would show in one host and not the others, and that is the failure
    this milestone most needs to not have.

    Checked in-process here; the nightly test below does it through a real
    kernel, which is what a host actually talks to.
    """
    bundle = Panel(session).root._repr_mimebundle_()
    assert WIDGET_MIME in bundle
    assert "text/plain" in bundle, "no fallback for a host without widget support"


@pytest.mark.slow
def test_the_panel_renders_inside_a_real_kernel(tmp_path: Path) -> None:
    """The same check through a live kernel, which is what VS Code, Lab and
    Notebook 7 each talk to.

    Marked slow: it starts a subprocess kernel, so it runs nightly rather
    than on every push. It is the only test here that exercises the
    ipykernel path at all, which is the one piece the in-process tests
    cannot stand in for.
    """
    import json
    import queue
    import sys

    jupyter_client = pytest.importorskip("jupyter_client")
    pytest.importorskip("ipykernel")

    # A kernelspec pointing at *this* interpreter, so the test does not
    # depend on one having been installed on the machine.
    spec_root = tmp_path / "jupyter"
    spec_dir = spec_root / "kernels" / "edacopilot-test"
    spec_dir.mkdir(parents=True)
    (spec_dir / "kernel.json").write_text(
        json.dumps(
            {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "edacopilot test",
                "language": "python",
            }
        ),
        encoding="utf-8",
    )

    manager = jupyter_client.kernelspec.KernelSpecManager()
    manager.kernel_dirs.insert(0, str(spec_root / "kernels"))
    km = jupyter_client.manager.KernelManager(
        kernel_name="edacopilot-test", kernel_spec_manager=manager
    )
    km.start_kernel()
    client = km.client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=90)
        client.execute(
            "import matplotlib; matplotlib.use('Agg')\n"
            "import pandas as pd, numpy as np, tempfile\n"
            "import edacopilot as eda\n"
            "rng = np.random.default_rng(0)\n"
            "df = pd.DataFrame({'v': rng.normal(size=40), 'g': ['a']*20+['b']*20})\n"
            "s = eda.start(df, session_id='k', root=tempfile.mkdtemp(), display=False)\n"
            "from IPython.display import display\n"
            "display(s._panel.root)\n"
        )
        mimes: set[str] = set()
        errors: list[str] = []
        while True:
            try:
                message = client.get_iopub_msg(timeout=120)
            except queue.Empty:
                break
            kind = message["msg_type"]
            if kind == "display_data":
                mimes |= set(message["content"]["data"])
            elif kind == "error":
                errors.append("\n".join(message["content"]["traceback"]))
            elif kind == "status" and message["content"]["execution_state"] == "idle":
                break
        assert not errors, errors[0][-2000:]
        assert WIDGET_MIME in mimes, f"the kernel emitted {sorted(mimes)}"
    finally:
        client.stop_channels()
        km.shutdown_kernel(now=True)
