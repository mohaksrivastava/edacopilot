"""Trap scenarios and the Section 7.4 worked example.

Section 16's acceptance criteria for M4 are "table-driven eligibility tests
pass" (test_eligibility_families.py) and "`paired_as_independent` raises
ambiguity" (here). Section 15.3's other three engine-facing traps --
`heteroscedastic_groups`, `heavy_tails_small_n`, `ordinal_as_numeric` --
are here too, each asserting the behaviour that table names.

These are end-to-end: raw generated data in, ranked candidates out, no
mocking. A trap test that stubbed the checks would only prove the engine
plumbing works, and the plumbing is not where these mistakes happen.
"""

from __future__ import annotations

import pytest
from scipy import stats

from edacopilot.eligibility import (
    AmbiguousSpecError,
    Design,
    Goal,
    QuestionSpec,
    select_candidates,
    validate_spec,
)
from edacore.contracts import CheckStatus, Eligibility
from tests.scenarios.generators import (
    heavy_tails_small_n,
    heteroscedastic_groups,
    ordinal_as_numeric,
    paired_as_independent,
    worked_example_7_4,
)


def _compare(outcome: str, group: str, design: Design, **extra: object) -> QuestionSpec:
    variables = {"outcome": outcome, "group": group}
    variables.update({k: v for k, v in extra.items() if isinstance(v, str)})
    return QuestionSpec(
        goal=Goal.COMPARE_GROUPS,
        variables=variables,
        design=design,
        confirmed_by_user={"design"},
    )


# --------------------------------------------------------------------------
# paired_as_independent -- M4's named acceptance criterion
# --------------------------------------------------------------------------


def test_paired_as_independent_raises_the_design_ambiguity() -> None:
    df = paired_as_independent()
    spec = QuestionSpec(
        goal=Goal.COMPARE_GROUPS,
        variables={"outcome": "score", "group": "condition", "subject": "subject_id"},
        design=Design.INDEPENDENT,
    )
    validated = validate_spec(spec, df)

    assert validated.ambiguities, "the planted design error was not detected"
    assert validated.design is Design.INDEPENDENT, "the engine changed the design by itself"
    with pytest.raises(AmbiguousSpecError):
        select_candidates(validated, df)


def test_paired_as_independent_confirmed_paired_offers_only_paired_methods() -> None:
    df = paired_as_independent()
    spec = _compare("score", "condition", Design.PAIRED, subject="subject_id")
    candidates = select_candidates(validate_spec(spec, df), df)

    assert candidates.family == "two_paired_numeric"
    assert {c.function for c in candidates.candidates} == {
        "paired_t",
        "wilcoxon_signed_rank",
        "sign_test_paired",
        "permutation_test_paired",
    }
    assert all(c.eligibility is Eligibility.ELIGIBLE for c in candidates.candidates)


def test_overriding_to_independent_makes_every_independent_test_ineligible() -> None:
    """Section 15.3: "paired IDs -> all independent tests INELIGIBLE once
    the design is confirmed". A user may override (Section 7.5), but the
    engine still has to say the method is invalid, and say why."""
    df = paired_as_independent()
    overridden = _compare("score", "condition", Design.INDEPENDENT, subject="subject_id")
    candidates = select_candidates(overridden.model_copy(update={"ambiguities": []}), df)

    assert candidates.family == "two_independent_numeric"
    assert all(c.eligibility is Eligibility.INELIGIBLE for c in candidates.candidates)

    crossing = next(c for c in candidates.checks if c.fact_id.startswith("design_crossing"))
    assert crossing.status is CheckStatus.FAIL
    assert "same subject" in crossing.consequence


# --------------------------------------------------------------------------
# heteroscedastic_groups
# --------------------------------------------------------------------------


def test_heteroscedastic_groups_caveats_student_t_and_prefers_welch() -> None:
    """Section 15.3: "Student t -> CAVEAT; Welch preferred"."""
    df = heteroscedastic_groups()
    candidates = select_candidates(
        validate_spec(_compare("value", "group", Design.INDEPENDENT), df), df
    )

    assert candidates.get("student_t").eligibility is Eligibility.CAVEAT
    variance = next(c for c in candidates.checks if c.fact_id.startswith("variance.levene"))
    assert variance.status is CheckStatus.FAIL
    assert any("variance.levene" in r for r in candidates.get("student_t").reasons)

    order = [c.function for c in candidates.candidates]
    assert order.index("welch_t") < order.index("student_t")


# --------------------------------------------------------------------------
# heavy_tails_small_n
# --------------------------------------------------------------------------


def test_heavy_tails_small_n_caveats_the_parametric_methods() -> None:
    """Section 15.3: "Parametric -> CAVEAT". At n=12 per group with df=2
    tails there is no large-n escape, so the caveat must come from the
    normality check itself rather than being waived."""
    df = heavy_tails_small_n()
    candidates = select_candidates(
        validate_spec(_compare("value", "group", Design.INDEPENDENT), df), df
    )

    for parametric in ("student_t", "welch_t"):
        assert candidates.get(parametric).eligibility is Eligibility.CAVEAT, parametric
        assert any("normality" in r for r in candidates.get(parametric).reasons), parametric

    # The robust and rank-based alternatives are the point of flagging it.
    assert candidates.get("yuen_trimmed_t").eligibility is Eligibility.ELIGIBLE
    assert candidates.get("mann_whitney").eligibility is Eligibility.ELIGIBLE


def test_cochrans_rule_does_not_waive_normality_at_this_n_and_skew() -> None:
    """`normality_or_large_n` is decided by Cochran's rule: n > 25*skew^2
    per group. Here the skew is large enough that the requirement is an
    order of magnitude above the actual n, so the escape cannot fire --
    which is why the parametric caveat above is a statement about the
    engine and not an accident of the seed."""
    from edacopilot.eligibility.checks import cochran_requirement

    df = heavy_tails_small_n()
    for _, rows in df.groupby("group"):
        values = rows["value"].to_numpy(float)
        assert len(values) < cochran_requirement(float(stats.skew(values, bias=True)))


# --------------------------------------------------------------------------
# ordinal_as_numeric
# --------------------------------------------------------------------------


def test_ordinal_as_numeric_blocks_pearson_until_the_user_confirms() -> None:
    """Section 15.3: "Flag as ordinal candidate; block Pearson until
    confirmed". The question must name the consequence (equal gaps), not
    the jargon."""
    df = ordinal_as_numeric()
    spec = QuestionSpec(
        goal=Goal.ASSOCIATION,
        variables={"x": "satisfaction", "y": "tenure_months"},
        design=Design.INDEPENDENT,
        confirmed_by_user={"design"},
    )
    validated = validate_spec(spec, df)
    assert any("gaps between values" in a for a in validated.ambiguities)
    with pytest.raises(AmbiguousSpecError):
        select_candidates(validated, df)


def test_ordinal_as_numeric_routes_away_from_pearson_once_confirmed() -> None:
    df = ordinal_as_numeric()
    spec = QuestionSpec(
        goal=Goal.ASSOCIATION,
        variables={"x": "satisfaction", "y": "tenure_months"},
        design=Design.INDEPENDENT,
        confirmed_by_user={"design", "x", "y"},
    )
    candidates = select_candidates(validate_spec(spec, df), df)

    assert candidates.family == "ordinal_any"
    assert "pearson" not in {c.function for c in candidates.candidates}
    assert candidates.get("spearman").eligibility is not Eligibility.INELIGIBLE


# --------------------------------------------------------------------------
# Section 7.4's worked example
# --------------------------------------------------------------------------

# Section 7.4's check table, as statuses. The spec quotes statistics from a
# dataset it does not ship, so what is reproduced is every check's verdict.
_EXPECTED_CHECKS = {
    "normality.shapiro.income.gender=F": CheckStatus.FAIL,
    "normality.shapiro.income.gender=M": CheckStatus.FAIL,
    "normality.descriptive.income.gender=F": CheckStatus.BORDERLINE,
    "normality.descriptive.income.gender=M": CheckStatus.BORDERLINE,
    "variance.levene.income.gender": CheckStatus.FAIL,
    "sample_size.income.gender": CheckStatus.PASS,
    "same_shape.income.gender": CheckStatus.PASS,
}

# Section 7.4's candidate table verbatim. The family (Section 7.3) also
# offers bootstrap_diff and ks_two_sample, which that table does not list;
# they are asserted separately rather than quietly folded in.
_EXPECTED_CANDIDATES = {
    "student_t": Eligibility.CAVEAT,
    "welch_t": Eligibility.CAVEAT,
    "yuen_trimmed_t": Eligibility.ELIGIBLE,
    "mann_whitney": Eligibility.ELIGIBLE,
    "brunner_munzel": Eligibility.ELIGIBLE,
    "permutation_test_2s": Eligibility.ELIGIBLE,
}


def test_worked_example_7_4_reproduces_the_check_table() -> None:
    df = worked_example_7_4()
    candidates = select_candidates(
        validate_spec(_compare("income", "gender", Design.INDEPENDENT), df), df
    )
    by_id = {c.fact_id: c.status for c in candidates.checks}
    for fact_id, expected in _EXPECTED_CHECKS.items():
        assert by_id.get(fact_id) is expected, f"{fact_id}: {by_id.get(fact_id)} != {expected}"


def test_worked_example_7_4_reproduces_the_candidate_table() -> None:
    df = worked_example_7_4()
    candidates = select_candidates(
        validate_spec(_compare("income", "gender", Design.INDEPENDENT), df), df
    )
    statuses = candidates.statuses()
    for function, expected in _EXPECTED_CANDIDATES.items():
        assert statuses[function] is expected, f"{function}: {statuses[function]} != {expected}"


def test_worked_example_7_4_reasons_match_the_spec_table() -> None:
    """Section 7.4's "Why" column: Student's t is caveated for normality AND
    equal variance, Welch for normality alone."""
    df = worked_example_7_4()
    candidates = select_candidates(
        validate_spec(_compare("income", "gender", Design.INDEPENDENT), df), df
    )
    student = candidates.get("student_t").reasons
    welch = candidates.get("welch_t").reasons

    assert any("normality" in r for r in student)
    assert any("variance.levene" in r for r in student)
    assert any("normality" in r for r in welch)
    assert not any("variance.levene" in r for r in welch)


def test_worked_example_7_4_family_carries_two_more_candidates_than_the_spec_table() -> None:
    """Section 7.3's `two_independent_numeric` family has eight methods;
    Section 7.4's table shows six. The two it omits are eligible here, and
    recording that keeps the difference deliberate rather than a silent
    mismatch between two parts of the spec."""
    df = worked_example_7_4()
    candidates = select_candidates(
        validate_spec(_compare("income", "gender", Design.INDEPENDENT), df), df
    )
    extra = set(candidates.statuses()) - set(_EXPECTED_CANDIDATES)
    assert extra == {"bootstrap_diff", "ks_two_sample"}
    assert all(candidates.statuses()[f] is Eligibility.ELIGIBLE for f in extra)
