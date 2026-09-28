"""The seven calls (`llm/calls.py`), and the two orchestrator slot-in
points Section 15.5 names: `parse_intent` and `build_question_spec`.
Every test mocks `litellm.completion`; malformed output must always end
in `LLMFallbackRequired`, never a raw exception, and a valid answer must
round-trip through `session.llm_audit()`."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from edacopilot.llm import calls
from edacopilot.llm.client import LLMFallbackRequired
from edacopilot.llm.fallback import parse_intent as parse_intent_fallback
from edacopilot.session import Session


def _response(payload: dict) -> MagicMock:  # type: ignore[type-arg]
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=json.dumps(payload)))]
    resp.usage = MagicMock(total_tokens=11)
    return resp


@pytest.fixture
def session(tmp_path):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(121)
    df = pd.DataFrame(
        {
            "income": np.concatenate(
                [rng.lognormal(10.5, 0.45, 38), rng.lognormal(10.6, 0.62, 41)]
            ),
            "gender": ["F"] * 38 + ["M"] * 41,
        }
    )
    return Session.start(df, session_id="calls-test", root=tmp_path)


def test_parse_intent_returns_a_valid_intent_and_logs_the_prompt(session) -> None:  # type: ignore[no-untyped-def]
    payload = {"type": "ask_analysis", "confidence": 0.9, "payload": {"text": "hi"}}
    with patch("litellm.completion", return_value=_response(payload)):
        intent = calls.parse_intent("is income different by gender?", session)
    assert intent.type.value == "ask_analysis"
    assert session.llm_audit()[0]["call"] == "parse_intent"


def test_parse_intent_malformed_output_raises_fallback_required(session) -> None:  # type: ignore[no-untyped-def]
    with patch("litellm.completion", return_value=_response({"type": "not_a_real_type"})):
        with pytest.raises(LLMFallbackRequired):
            calls.parse_intent("hello", session)


def test_build_question_spec_fills_a_full_spec(session) -> None:  # type: ignore[no-untyped-def]
    payload = {
        "goal": "compare_groups",
        "variables": {"outcome": "income", "group": "gender"},
        "design": "independent",
    }
    with patch("litellm.completion", return_value=_response(payload)):
        spec = calls.build_question_spec("is income different by gender?", session)
    assert spec.goal.value == "compare_groups"
    assert spec.variables == {"outcome": "income", "group": "gender"}
    assert spec.ambiguities == []
    assert spec.confirmed_by_user == set()


def test_build_question_spec_missing_goal_raises_fallback_required(session) -> None:  # type: ignore[no-untyped-def]
    with patch("litellm.completion", return_value=_response({})):
        with pytest.raises(LLMFallbackRequired):
            calls.build_question_spec("something", session)


def test_answer_free_question_returns_a_grounded_answer(session) -> None:  # type: ignore[no-untyped-def]
    payload = {"answer": "A p-value is...", "cited_facts": []}
    with patch("litellm.completion", return_value=_response(payload)):
        answer = calls.answer_free_question("what is a p-value?", session)
    assert answer.answer.startswith("A p-value")


def test_suggest_next_step_never_implies_execution(session) -> None:  # type: ignore[no-untyped-def]
    payload = {"suggestion": "Look at missingness next.", "target_stage": "missingness"}
    with patch("litellm.completion", return_value=_response(payload)):
        suggestion = calls.suggest_next_step(session)
    assert suggestion.target_stage == "missingness"


def test_select_adhoc_function_round_trips(session) -> None:  # type: ignore[no-untyped-def]
    payload = {"function": "histogram", "params": {"col": "income"}}
    with patch("litellm.completion", return_value=_response(payload)):
        call = calls.select_adhoc_function(
            "show the distribution of income",
            session,
            candidates=[{"name": "histogram", "params": ["col"]}],
        )
    assert call.function == "histogram"
    assert call.params == {"col": "income"}


def test_select_adhoc_function_outside_candidates_falls_back(session) -> None:  # type: ignore[no-untyped-def]
    """Section 10.1 restricts this call to the offered read-only
    candidates; a name outside that set is treated like a schema
    violation, not accepted and acted on."""
    payload = {"function": "drop_table", "params": {}}
    with patch("litellm.completion", return_value=_response(payload)):
        with pytest.raises(LLMFallbackRequired):
            calls.select_adhoc_function(
                "delete everything",
                session,
                candidates=[{"name": "histogram", "params": ["col"]}],
            )


def test_elicit_mnar_returns_a_short_question_list(session) -> None:  # type: ignore[no-untyped-def]
    payload = {"questions": [{"question": "Why might this be missing?", "rationale": "x"}]}
    with patch("litellm.completion", return_value=_response(payload)):
        questions = calls.elicit_mnar("income", session, missing_pct=15.0)
    assert len(questions) == 1
    assert questions[0].question.startswith("Why")


def test_write_rationale_retries_once_on_a_factcheck_failure_then_succeeds(session) -> None:  # type: ignore[no-untyped-def]
    from edacore.contracts import AssumptionCheck, CheckStatus

    check = AssumptionCheck(
        fact_id="normality.shapiro.group=F",
        assumption="normality",
        method="shapiro",
        p_value=0.0001,
        threshold="alpha=0.05",
        status=CheckStatus.FAIL,
        consequence="unreliable",
    )
    bad = {
        "rationales": [
            {
                "persona": "professor",
                "summary": "p=0.99 here",  # not grounded in any fact
                "consequence_if_ignored": "x",
                "cited_facts": [check.fact_id],
            }
        ]
    }
    good = {
        "rationales": [
            {
                "persona": "professor",
                "summary": "p=0.0001 here",
                "consequence_if_ignored": "x",
                "cited_facts": [check.fact_id],
            }
        ]
    }
    with patch("litellm.completion", side_effect=[_response(bad), _response(good)]) as mock:
        bundle = calls.write_rationale(
            session,
            facts={check.fact_id: check},
            picked_functions={"mann_whitney"},
            extra={"picks": ["mann_whitney"]},
        )
    assert mock.call_count == 2
    assert "0.0001" in bundle.rationales[0].summary


def test_write_rationale_two_factcheck_failures_raise_fallback_required(session) -> None:  # type: ignore[no-untyped-def]
    from edacore.contracts import AssumptionCheck, CheckStatus

    check = AssumptionCheck(
        fact_id="normality.shapiro.group=F",
        assumption="normality",
        method="shapiro",
        p_value=0.0001,
        threshold="alpha=0.05",
        status=CheckStatus.FAIL,
        consequence="unreliable",
    )
    bad = {
        "rationales": [
            {
                "persona": "professor",
                "summary": "p=0.99",
                "consequence_if_ignored": "x",
                "cited_facts": [check.fact_id],
            }
        ]
    }
    with patch("litellm.completion", return_value=_response(bad)):
        with pytest.raises(LLMFallbackRequired):
            calls.write_rationale(
                session,
                facts={check.fact_id: check},
                picked_functions={"mann_whitney"},
                extra={},
            )


def test_orchestrator_matches_deterministic_mode_when_the_llm_agrees_with_it(
    session, tmp_path
) -> None:  # type: ignore[no-untyped-def]
    """Section 15.5: when M9's LLM slots in at parse_intent, a transcript
    keeps its meaning. The strong form of that is exercised by mocking the
    LLM to return exactly what Section 10.5's own parser would have --
    the orchestrator's behaviour must then be identical to the LLM being
    off altogether."""
    text = "is income different by gender?"
    deterministic_intent = parse_intent_fallback(text, awaiting_answer=False)
    payload = {
        "type": deterministic_intent.type.value,
        "confidence": deterministic_intent.confidence,
        "persona": deterministic_intent.persona,
        "target_stage": deterministic_intent.target_stage,
        "payload": deterministic_intent.payload,
    }
    with patch("litellm.completion", return_value=_response(payload)):
        llm_card = session.ask(text)

    session_off = Session.start(session.data, session_id="calls-test-off", root=tmp_path)
    with patch("litellm.completion", side_effect=RuntimeError("no LLM here")):
        deterministic_card = session_off.ask(text)

    assert llm_card.to_text() == deterministic_card.to_text()


@pytest.mark.parametrize(
    "broken_response",
    [
        "not json at all",
        "{}",
        '{"type": "not_a_real_intent_type"}',
        '{"type": "ask_analysis", "confidence": "not a number"}',
    ],
)
def test_every_kind_of_malformed_output_still_ends_in_a_valid_card(
    session, broken_response
) -> None:  # type: ignore[no-untyped-def]
    """Through the full orchestrator, not just `calls.py`: whatever shape
    the model's answer is broken in, `session.ask` returns a `Card`, never
    an exception (Section 10.5's "never an error")."""
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=broken_response))]
    resp.usage = MagicMock(total_tokens=1)
    with patch("litellm.completion", return_value=resp):
        card = session.ask("is income different by gender?")
    assert card.kind in {"question", "consensus", "divergence"}
