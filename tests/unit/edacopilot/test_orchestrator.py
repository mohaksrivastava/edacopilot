"""The orchestrator's parts: state machine, cards, and the intent parser.

The invariants live in `test_orchestrator_invariants.py` and the end-to-end
behaviour in `tests/conversations/`. What is here is the machinery those
two rest on, tested where a failure is easiest to read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edacopilot.llm import NOT_IN_GLOSSARY, build_question_spec_form, parse_intent
from edacopilot.orchestrator import (
    STAGE_ORDER,
    ActionButton,
    Card,
    CardKind,
    Intent,
    IntentType,
    Stage,
    applicable_stages,
    button,
    jump_warning,
    next_stage,
    parse_stage,
    skipped_stages,
)
from edacopilot.orchestrator.cards import ProposalView
from edacopilot.session import Session
from edacopilot.stages import IMPLEMENTED, STAGES, StubStage
from edacore.contracts import AssumptionCheck, CheckStatus, TestResult
from edacore.profiling import profile_dataset

# --------------------------------------------------------------------------
# state machine (Section 9.2)
# --------------------------------------------------------------------------


def _profile(df: pd.DataFrame):  # type: ignore[no-untyped-def]
    return profile_dataset(df)


def _clean() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"v": rng.normal(size=40), "g": ["a"] * 20 + ["b"] * 20})


def _with_missing(share: float = 0.2) -> pd.DataFrame:
    df = _clean()
    n = int(len(df) * share)
    df.loc[: n - 1, "v"] = np.nan
    return df


def test_the_suggested_order_is_section_9_2s() -> None:
    assert [stage.value for stage in STAGE_ORDER] == [
        "ingest",
        "profile",
        "quality",
        "missingness",
        "outliers",
        "transform",
        "explore",
        "hypothesis",
        "export",
    ]


def test_parse_stage_names_the_stages_it_knows() -> None:
    assert parse_stage("Hypothesis") is Stage.HYPOTHESIS
    with pytest.raises(ValueError, match="no stage called 'nowhere'"):
        parse_stage("nowhere")


def test_skipping_nothing_produces_no_warning() -> None:
    visited = {
        Stage.INGEST,
        Stage.PROFILE,
        Stage.QUALITY,
        Stage.MISSINGNESS,
        Stage.OUTLIERS,
        Stage.TRANSFORM,
    }
    assert skipped_stages(Stage.HYPOTHESIS, visited) == []
    assert jump_warning(Stage.HYPOTHESIS, visited, _profile(_with_missing())) is None


def test_explore_is_never_warned_about() -> None:
    """It is read-only: not having looked at a summary cannot invalidate a
    later test, and a warning that fires for nothing trains the user to
    ignore the ones that matter."""
    assert Stage.EXPLORE not in skipped_stages(Stage.HYPOTHESIS, {Stage.INGEST})


def test_the_warning_names_a_number_from_this_dataset() -> None:
    warning = jump_warning(Stage.HYPOTHESIS, {Stage.INGEST}, _profile(_with_missing(0.25)))
    assert warning is not None
    assert "25% of `v` is missing" in warning
    assert "drop those rows" in warning


def test_the_warning_is_one_line() -> None:
    warning = jump_warning(Stage.HYPOTHESIS, {Stage.INGEST}, _profile(_with_missing()))
    assert warning is not None and "\n" not in warning


def test_missingness_outranks_quality_when_both_were_skipped() -> None:
    """Missing rows are dropped without appearing anywhere the user looks;
    a duplicate at least leaves a visible count behind."""
    df = pd.concat([_with_missing(), _with_missing().head(4)], ignore_index=True)
    warning = jump_warning(Stage.HYPOTHESIS, {Stage.INGEST}, _profile(df))
    assert warning is not None
    assert warning.startswith("Missing values haven't been reviewed")


def test_a_dataset_with_nothing_wrong_still_names_the_skipped_stages() -> None:
    warning = jump_warning(Stage.HYPOTHESIS, {Stage.INGEST}, _profile(_clean()))
    assert warning is not None
    assert "Skipped: profile, quality, missingness, outliers, transform" in warning
    assert "nothing has been checked or corrected" in warning


def test_conditional_stages_appear_only_when_they_apply() -> None:
    assert Stage.TIMESERIES not in applicable_stages(_profile(_clean()))
    ts = pd.DataFrame(
        {
            "t": pd.date_range("2024-01-01", periods=60, freq="D"),
            "v": np.arange(60, dtype=float),
        }
    )
    profile = _profile(ts)
    if profile.structure in ("time_series", "panel"):
        assert Stage.TIMESERIES in applicable_stages(profile)


def test_next_stage_follows_the_suggested_order() -> None:
    profile = _profile(_clean())
    assert next_stage(Stage.PROFILE, profile) is Stage.QUALITY
    assert next_stage(Stage.EXPORT, profile) is None


# --------------------------------------------------------------------------
# stage modules (Section 9.3)
# --------------------------------------------------------------------------


def test_every_stage_has_a_module() -> None:
    assert set(STAGES) == set(Stage)


def test_only_profile_explore_hypothesis_and_ingest_are_implemented() -> None:
    """M7's scope, asserted so a stub cannot quietly start claiming to work."""
    assert IMPLEMENTED == {Stage.INGEST, Stage.PROFILE, Stage.EXPLORE, Stage.HYPOTHESIS}


@pytest.mark.parametrize(
    "stage",
    sorted(set(Stage) - {Stage.INGEST, Stage.PROFILE, Stage.EXPLORE, Stage.HYPOTHESIS}),
    ids=lambda stage: stage.value,
)
def test_a_stub_says_which_version_brings_it(tmp_path: Path, stage: Stage) -> None:
    session = Session.start(_clean(), session_id=f"stub-{stage.value}", root=tmp_path)
    module = STAGES[stage]
    assert isinstance(module, StubStage)
    card = module.entry_summary(session)
    assert card.kind == "info"
    assert "available in a later version" in card.body_md.lower()
    assert module.milestone in card.body_md
    assert card.actions, "a stub with no way out of it"


@pytest.mark.parametrize(
    "stage",
    sorted(set(Stage) - {Stage.INGEST, Stage.PROFILE, Stage.EXPLORE, Stage.HYPOTHESIS}),
    ids=lambda stage: stage.value,
)
def test_a_stub_refuses_to_build_a_spec_or_execute(tmp_path: Path, stage: Stage) -> None:
    """A stub that returned something empty would let a user believe the
    stage had run."""
    from edacopilot.stages import StageNotAvailableError

    session = Session.start(_clean(), session_id=f"refuse-{stage.value}", root=tmp_path)
    module = STAGES[stage]
    with pytest.raises(StageNotAvailableError):
        module.build_spec(Intent(type=IntentType.ASK_ANALYSIS), session)


def test_explore_records_nothing(tmp_path: Path) -> None:
    session = Session.start(_clean(), session_id="ex", root=tmp_path)
    session.goto_stage("explore")
    session.ask("show me v")
    assert len(session.provenance) == 0
    assert list(session.store.versions) == ["v0"]


# --------------------------------------------------------------------------
# cards (Section 9.4)
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


def test_a_card_renders_every_field_it_carries() -> None:
    """A rendering that dropped a field would make a golden transcript
    agree with a card the user would not recognise."""
    card = Card(
        kind="result",
        title="A title",
        stage="hypothesis",
        body_md="Some **body**.",
        diagnostics=[_check()],
        proposals=[
            ProposalView(
                persona="professor",
                display_name="Professor",
                function="mann_whitney",
                eligibility="eligible",
                rationale="Because its assumptions hold.",
            )
        ],
        result=TestResult(
            fact_id="t.1",
            function="mann_whitney",
            estimand="stochastic dominance",
            statistic=603.0,
            statistic_name="U",
            p_value=0.085,
            p_adjusted=0.17,
            effect_size=-0.23,
            effect_size_name="rank_biserial",
            effect_size_ci=(-0.45, 0.03),
            effect_magnitude="small",
            n={"F": 38, "M": 41},
            warnings=["a warning"],
            validity_notes=["a validity note"],
        ),
        plots=["plot_01"],
        actions=[
            ActionButton(label="Run it", intent=button(IntentType.ACCEPT), call="session.accept()")
        ],
    )
    text = card.to_text()
    for expected in [
        "[result] A title",
        "stage: hypothesis",
        "Some **body**.",
        "normality.shapiro.v.g=a",
        "Parametric tests that assume normality may mislead.",
        "Professor -> mann_whitney [eligible]",
        "Because its assumptions hold.",
        "0.085",
        "0.17",
        "rank_biserial",
        "(small)",
        "F=38, M=41",
        "a warning",
        "a validity note",
        "plot_01",
        "[Run it]",
        "session.accept()",
    ]:
        assert expected in text, f"{expected!r} missing from the rendered card"


def test_an_empty_card_still_renders() -> None:
    assert Card(kind="info", title="Nothing").to_text() == "[info] Nothing"


def test_evidence_is_rendered_apart_from_verdicts() -> None:
    verdict = _check(CheckStatus.PASS)
    evidence = _check(CheckStatus.FAIL).model_copy(update={"fact_id": "measurement_level.v.binary"})
    card = Card(
        kind="consensus",
        title="t",
        diagnostics=[verdict, evidence],
        graded=[verdict.fact_id],
    )
    text = card.to_text()
    verdict_section, evidence_section = text.split("Also computed")
    assert verdict.fact_id in verdict_section
    assert evidence.fact_id not in verdict_section
    assert evidence.fact_id in evidence_section


def test_without_a_graded_list_every_check_is_shown_as_a_verdict() -> None:
    """A card built by hand (a stage that runs one check) should not have
    its only diagnostic demoted to evidence."""
    card = Card(kind="info", title="t", diagnostics=[_check()])
    assert "Also computed" not in card.to_text()
    assert _check().fact_id in card.to_text()


def test_card_kinds_are_section_9_4s() -> None:
    import typing

    assert set(typing.get_args(CardKind)) == {
        "info",
        "question",
        "consensus",
        "divergence",
        "result",
        "warning",
        "override_confirm",
        "export",
    }


# --------------------------------------------------------------------------
# the fallback intent parser (Section 10.5)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("undo that", IntentType.UNDO),
        ("go back", IntentType.UNDO),
        ("skip this", IntentType.SKIP),
        ("export the notebook", IntentType.EXPORT),
        ("go to hypothesis", IntentType.GOTO_STAGE),
        ("use alpha 0.01", IntentType.MODIFY),
        ("accept the professor's proposal", IntentType.ACCEPT),
        ("use student_t anyway", IntentType.OVERRIDE),
        ("run post-hoc comparisons", IntentType.ASK_ANALYSIS),
        ("which groups differ?", IntentType.ASK_ANALYSIS),
        ("what is sphericity?", IntentType.EXPLAIN),
        ("explain the normality check", IntentType.EXPLAIN),
        ("plot age against income", IntentType.SHOW),
        ("show me the ledger", IntentType.SETTINGS),
        ("branch alternative", IntentType.BRANCH),
        ("switch to branch main", IntentType.SWITCH_BRANCH),
        ("is income different by region?", IntentType.ASK_ANALYSIS),
        ("how are age and income related?", IntentType.ASK_ANALYSIS),
    ],
)
def test_the_keyword_rules_route_the_common_phrasings(text: str, expected: IntentType) -> None:
    intent = parse_intent(text)
    assert intent.type is expected
    assert intent.is_confident, f"{text!r} routed to {expected} but below the confidence floor"


def test_an_unmatched_message_is_honest_about_it() -> None:
    """Section 9.1's floor only helps if the confidence is truthful. A
    keyword ruleset that stretched to cover everything would misroute
    confidently, which is worse than not routing."""
    intent = parse_intent("wibble the frobnicator")
    assert intent.type is IntentType.OTHER
    assert not intent.is_confident


def test_a_bare_answer_is_read_as_one_only_when_a_question_is_open() -> None:
    assert parse_intent("paired").type is not IntentType.ANSWER_CLARIFICATION
    answered = parse_intent("paired", awaiting_answer=True)
    assert answered.type is IntentType.ANSWER_CLARIFICATION
    assert answered.payload["design"] == "paired"


def test_a_named_persona_is_carried_on_the_intent() -> None:
    assert parse_intent("go with the maverick").persona == "maverick"
    assert parse_intent("accept").persona is None


def test_an_override_names_the_method_from_the_registry() -> None:
    """Matched against the registry, so an override can name any method the
    catalogue actually has."""
    assert parse_intent("use welch_anova anyway").payload["function"] == "welch_anova"
    assert parse_intent("use mann whitney anyway").payload["function"] == "mann_whitney"


def test_a_button_is_certain_by_construction() -> None:
    intent = button(IntentType.ACCEPT, persona="professor")
    assert intent.confidence == 1.0
    assert intent.persona == "professor"
    assert intent.is_confident


# --------------------------------------------------------------------------
# the form-style fallback (Section 10.5)
# --------------------------------------------------------------------------


def test_the_form_offers_only_columns_the_data_has() -> None:
    fields = {field.name: field for field in build_question_spec_form(_clean())}
    assert fields["outcome"].options == ["v"]
    assert fields["group"].options == ["g"]
    assert "compare_groups" in fields["goal"].options
    assert not fields["alpha"].required


def test_asking_with_nothing_returns_the_form(tmp_path: Path) -> None:
    session = Session.start(_clean(), session_id="form", root=tmp_path)
    card = session.ask()
    assert card.kind == "question"
    assert "fields to fill in" in card.body_md
    assert "session.ask(" in card.body_md


def test_free_text_that_cannot_be_resolved_returns_the_form(tmp_path: Path) -> None:
    session = Session.start(_clean(), session_id="form2", root=tmp_path)
    card = session.ask("is v different between the groups?")
    assert card.kind == "question"
    assert "not which columns it is about" in card.title


def test_an_unparseable_message_asks_rather_than_guessing(tmp_path: Path) -> None:
    session = Session.start(_clean(), session_id="huh", root=tmp_path)
    card = session.ask("wibble the frobnicator")
    assert card.kind == "question"
    assert "not sure what you meant" in card.title
    assert 2 <= len(card.actions) <= 3


# --------------------------------------------------------------------------
# explain (Section 10.5)
# --------------------------------------------------------------------------


def test_explaining_a_method_in_the_proposal_uses_this_session(tmp_path: Path) -> None:
    """ "Why not a t-test" is a question about this data, not a definition."""
    from tests.scenarios.generators import worked_example_7_4

    session = Session.start(worked_example_7_4(), session_id="why", root=tmp_path)
    session.ask(
        goal="compare_groups",
        outcome="income",
        group="gender",
        design="independent",
        confirmed_by_user=["design"],
    )
    card = session.explain("student_t")
    assert "is caveat" in card.title
    assert card.diagnostics, "no check was cited"
    assert "variance.levene" in card.body_md


def test_explaining_a_term_with_no_proposal_open_uses_the_glossary(tmp_path: Path) -> None:
    session = Session.start(_clean(), session_id="gloss", root=tmp_path)
    card = session.explain("sphericity")
    assert "repeated-measures" in card.body_md
    assert card.diagnostics == []


def test_explaining_something_unknown_says_so(tmp_path: Path) -> None:
    session = Session.start(_clean(), session_id="unk", root=tmp_path)
    assert session.explain("frobnicator").body_md == NOT_IN_GLOSSARY


# --------------------------------------------------------------------------
# modify (Section 9.1)
# --------------------------------------------------------------------------


def test_modifying_alpha_re_proposes_rather_than_patching(tmp_path: Path) -> None:
    """Changing alpha changes which methods are caveated, so the card has
    to be rebuilt; a patched card would show the old ranking under the new
    setting."""
    from tests.scenarios.generators import worked_example_7_4

    session = Session.start(worked_example_7_4(), session_id="mod", root=tmp_path)
    session.ask(
        goal="compare_groups",
        outcome="income",
        group="gender",
        design="independent",
        confirmed_by_user=["design"],
    )
    card = session.modify(alpha=0.01)
    assert card.kind in ("consensus", "divergence")
    pending = session.orchestrator.state.pending
    assert pending is not None and pending.spec.alpha == 0.01
    assert len(session.provenance) == 0


def test_modify_with_nothing_open_says_so(tmp_path: Path) -> None:
    session = Session.start(_clean(), session_id="mod2", root=tmp_path)
    assert "Nothing to modify" in session.modify(alpha=0.01).title


# --------------------------------------------------------------------------
# persistence of the durable turn state (Section 12.4)
# --------------------------------------------------------------------------


def test_visited_stages_survive_a_resume(tmp_path: Path) -> None:
    """Section 9.2's warning must survive a kernel restart, or it becomes a
    lie: a user who reviewed missingness yesterday should not be told they
    did not."""
    from edacopilot.session import resume

    session = Session.start(_with_missing(), session_id="stages", root=tmp_path)
    session.goto_stage("profile")
    session.goto_stage("missingness")
    session.goto_stage("quality")

    restored = resume("stages", root=tmp_path)
    assert restored.stage == "quality"
    card = restored.goto_stage("hypothesis")
    assert card.kind == "warning"
    assert "missingness" not in card.body_md.split("Skipped:")[1]


def test_a_resumed_session_does_not_keep_a_proposal_on_screen(tmp_path: Path) -> None:
    """The transient half is deliberately not persisted: accepting into a
    context the user no longer has in front of them is the failure mode."""
    from edacopilot.session import resume

    session = Session.start(_clean(), session_id="pend", root=tmp_path)
    session.ask(
        goal="compare_groups",
        outcome="v",
        group="g",
        design="independent",
        confirmed_by_user=["design"],
    )
    assert session.orchestrator.state.pending is not None

    restored = resume("pend", root=tmp_path)
    assert restored.orchestrator.state.pending is None
    assert "Nothing to accept" in restored.accept().title


# --------------------------------------------------------------------------
# next-step suggestions (Section 9.5)
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scenario",
    ["clean_significant", "not_significant", "significant_omnibus"],
)
def test_a_result_card_always_offers_between_two_and_four_next_steps(
    tmp_path: Path, scenario: str
) -> None:
    """Section 9.5: "2-4 suggested next actions as buttons".

    A lone button reads as a dead end at the moment the user most needs
    somewhere to go, and more than four is a menu nobody reads.
    """
    if scenario == "significant_omnibus":
        rng = np.random.default_rng(4)
        df = pd.DataFrame(
            {
                "v": np.concatenate(
                    [rng.exponential(2, 40), rng.exponential(3, 40), rng.exponential(6, 40)]
                ),
                "g": ["a"] * 40 + ["b"] * 40 + ["c"] * 40,
            }
        )
    elif scenario == "not_significant":
        df = _clean()
    else:
        rng = np.random.default_rng(1)
        df = pd.DataFrame(
            {
                "v": np.concatenate([rng.normal(0, 1, 40), rng.normal(2.5, 1, 40)]),
                "g": ["a"] * 40 + ["b"] * 40,
            }
        )

    session = Session.start(df, session_id=f"sug-{scenario}", root=tmp_path)
    session.ask(
        goal="compare_groups",
        outcome="v",
        group="g",
        design="independent",
        confirmed_by_user=["design"],
    )
    card = session.accept("professor")
    assert card.kind == "result"
    assert 2 <= len(card.actions) <= 4, [action.label for action in card.actions]


def test_a_significant_omnibus_offers_its_post_hoc_first(tmp_path: Path) -> None:
    """Section 9.5's own example, and the rule that a post-hoc follows a
    significant omnibus rather than being offered after every test."""
    rng = np.random.default_rng(4)
    df = pd.DataFrame(
        {
            "v": np.concatenate(
                [rng.exponential(2, 40), rng.exponential(3, 40), rng.exponential(6, 40)]
            ),
            "g": ["a"] * 40 + ["b"] * 40 + ["c"] * 40,
        }
    )
    session = Session.start(df, session_id="omni", root=tmp_path)
    session.ask(
        goal="compare_groups",
        outcome="v",
        group="g",
        design="independent",
        confirmed_by_user=["design"],
    )
    card = session.accept("professor")
    assert card.actions[0].label.startswith("Run post-hoc comparisons")
    assert "dunn_test" in card.actions[0].label


def test_a_two_group_test_is_never_offered_a_post_hoc(tmp_path: Path) -> None:
    """There are no pairs left to compare, so the button would be noise."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "v": np.concatenate([rng.normal(0, 1, 40), rng.normal(2.5, 1, 40)]),
            "g": ["a"] * 40 + ["b"] * 40,
        }
    )
    session = Session.start(df, session_id="two", root=tmp_path)
    session.ask(
        goal="compare_groups",
        outcome="v",
        group="g",
        design="independent",
        confirmed_by_user=["design"],
    )
    card = session.accept("professor")
    assert not any("post-hoc" in action.label.lower() for action in card.actions)


def test_a_non_significant_omnibus_is_not_offered_a_post_hoc(tmp_path: Path) -> None:
    """Running pairwise comparisons after a non-significant omnibus is the
    inflation the omnibus exists to prevent."""
    rng = np.random.default_rng(9)
    df = pd.DataFrame(
        {
            "v": rng.normal(0, 1, 90),
            "g": ["a"] * 30 + ["b"] * 30 + ["c"] * 30,
        }
    )
    session = Session.start(df, session_id="null", root=tmp_path)
    session.ask(
        goal="compare_groups",
        outcome="v",
        group="g",
        design="independent",
        confirmed_by_user=["design"],
    )
    card = session.accept("professor")
    assert card.result is not None
    assert not any("post-hoc" in action.label.lower() for action in card.actions)
    follow_up = session.ask("run post-hoc comparisons")
    assert "No post-hoc applies here" in follow_up.title
    assert len(session.provenance) == 1
