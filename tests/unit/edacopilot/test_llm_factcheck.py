"""Section 8.4's fact-check: every number, every cited fact_id, every
named method must trace back to something the model was actually given."""

from __future__ import annotations

from edacopilot.llm.contracts import PersonaRationale, RationaleBundle
from edacopilot.llm.factcheck import check_rationale
from edacore.contracts import AssumptionCheck, CheckStatus

CHECK = AssumptionCheck(
    fact_id="normality.shapiro.group=F",
    assumption="normality",
    method="shapiro",
    statistic=0.86,
    p_value=0.0001,
    threshold="alpha=0.05",
    status=CheckStatus.FAIL,
    consequence="A t-test's p-value would be unreliable.",
)
FACTS = {CHECK.fact_id: CHECK}


def _bundle(**rationale_kwargs: object) -> RationaleBundle:
    defaults = {
        "persona": "professor",
        "summary": "ok",
        "consequence_if_ignored": "x",
        "cited_facts": [CHECK.fact_id],
    }
    defaults.update(rationale_kwargs)
    return RationaleBundle(rationales=[PersonaRationale(**defaults)])  # type: ignore[arg-type]


def test_a_grounded_rationale_passes() -> None:
    bundle = _bundle(summary="Normality failed (p=0.0001).")
    assert check_rationale(bundle, facts=FACTS, picked_functions={"mann_whitney"}) is None


def test_a_number_absent_from_any_fact_fails() -> None:
    bundle = _bundle(summary="Normality failed (p=0.5).")
    failure = check_rationale(bundle, facts=FACTS, picked_functions={"mann_whitney"})
    assert failure is not None
    assert "0.5" in failure.reason


def test_rounding_tolerance_allows_a_re_quoted_number() -> None:
    bundle = _bundle(summary="The Shapiro statistic was about 0.8601.")
    assert check_rationale(bundle, facts=FACTS, picked_functions={"mann_whitney"}) is None


def test_an_unknown_cited_fact_id_fails() -> None:
    bundle = _bundle(cited_facts=["not_a_real_fact"])
    failure = check_rationale(bundle, facts=FACTS, picked_functions={"mann_whitney"})
    assert failure is not None
    assert "not_a_real_fact" in failure.reason


def test_naming_a_method_that_was_not_picked_fails() -> None:
    bundle = _bundle(summary="Use student_t here for the comparison.")
    failure = check_rationale(bundle, facts=FACTS, picked_functions={"mann_whitney"})
    assert failure is not None
    assert "student_t" in failure.reason


def test_naming_the_actual_pick_does_not_fail() -> None:
    bundle = _bundle(summary="mann_whitney is the clean choice here.")
    assert check_rationale(bundle, facts=FACTS, picked_functions={"mann_whitney"}) is None


def test_prose_with_no_underscored_words_never_false_positives() -> None:
    bundle = _bundle(
        summary="This method avoids assuming a normal distribution, which the data does not have."
    )
    assert check_rationale(bundle, facts=FACTS, picked_functions={"mann_whitney"}) is None


def test_the_comparison_field_is_checked_too() -> None:
    bundle = RationaleBundle(
        rationales=[
            PersonaRationale(
                persona="professor",
                summary="ok",
                consequence_if_ignored="x",
                cited_facts=[CHECK.fact_id],
            )
        ],
        comparison="This holds even when p is nowhere near 0.99999.",
    )
    failure = check_rationale(bundle, facts=FACTS, picked_functions={"mann_whitney"})
    assert failure is not None
    assert "comparison" in failure.reason
