"""QuestionSpec validation, and the design cross-check it exists for
(ARCHITECTURE.md, Section 7.1).

The design cross-check is the single most consequential thing this layer
does, so most of this file is about one property: when the data and the
proposed design disagree, the system asks. It never picks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edacopilot.eligibility import (
    AmbiguousSpecError,
    Design,
    Goal,
    InvalidSpecError,
    QuestionSpec,
    describe_spec,
    select_candidates,
    validate_spec,
)
from tests.scenarios.generators import paired_as_independent


def _independent(n_per_group: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "subject_id": range(1, 2 * n_per_group + 1),
            "group": ["A"] * n_per_group + ["B"] * n_per_group,
            "score": np.concatenate(
                [rng.normal(10, 2, n_per_group), rng.normal(12, 2, n_per_group)]
            ),
        }
    )


def _spec(**overrides: object) -> QuestionSpec:
    base: dict[str, object] = {
        "goal": Goal.COMPARE_GROUPS,
        "variables": {"outcome": "score", "group": "group", "subject": "subject_id"},
        "design": Design.INDEPENDENT,
        "confirmed_by_user": {"design"},
    }
    return QuestionSpec(**{**base, **overrides})  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Section 7.1 step 1: columns and roles
# --------------------------------------------------------------------------


def test_unknown_column_is_an_error_not_a_question() -> None:
    """A column that does not exist is a spec-building bug. Asking the user
    about it would be asking them to debug the system."""
    df = _independent()
    with pytest.raises(InvalidSpecError, match="not in the data"):
        validate_spec(_spec(variables={"outcome": "nope", "group": "group"}), df)


def test_missing_required_role_is_an_error() -> None:
    df = _independent()
    with pytest.raises(InvalidSpecError, match="needs"):
        validate_spec(_spec(variables={"outcome": "score"}), df)


def test_single_level_grouping_column_is_an_error() -> None:
    df = _independent()
    df["group"] = "only_one"
    with pytest.raises(InvalidSpecError, match="at least 2"):
        validate_spec(_spec(), df)


# --------------------------------------------------------------------------
# Section 7.1 step 2: the design cross-check
# --------------------------------------------------------------------------


def test_repeated_ids_across_groups_raise_an_ambiguity_not_a_design_change() -> None:
    """Section 15.3's `paired_as_independent` trap, and the acceptance
    criterion for M4: the engine must surface the conflict, and `design`
    must come back exactly as it went in."""
    df = paired_as_independent()
    spec = QuestionSpec(
        goal=Goal.COMPARE_GROUPS,
        variables={"outcome": "score", "group": "condition", "subject": "subject_id"},
        design=Design.INDEPENDENT,
    )
    validated = validate_spec(spec, df)

    assert len(validated.ambiguities) == 1
    question = validated.ambiguities[0]
    assert "subject_id" in question and "condition" in question
    assert question.rstrip().endswith("?")
    # The system asked. It did not decide.
    assert validated.design is Design.INDEPENDENT


def test_the_engine_refuses_to_run_on_an_unresolved_spec() -> None:
    """Section 7.1 step 3, enforced rather than trusted: an ambiguity cannot
    be bypassed by calling the engine directly."""
    df = paired_as_independent()
    validated = validate_spec(
        QuestionSpec(
            goal=Goal.COMPARE_GROUPS,
            variables={"outcome": "score", "group": "condition", "subject": "subject_id"},
            design=Design.INDEPENDENT,
        ),
        df,
    )
    with pytest.raises(AmbiguousSpecError) as excinfo:
        select_candidates(validated, df)
    assert excinfo.value.ambiguities == validated.ambiguities


def test_paired_claimed_but_no_shared_ids_also_asks() -> None:
    """The mirror-image error: the design says paired, but nothing links a
    row in one group to a row in the other."""
    df = _independent()
    validated = validate_spec(_spec(design=Design.PAIRED), df)
    assert any("no value of" in a for a in validated.ambiguities)
    assert validated.design is Design.PAIRED


def test_ids_repeating_within_one_group_is_clustering_not_pairing() -> None:
    """Repeated ids break independence without making the design paired.
    Conflating the two would push the user toward a paired test that has no
    pairs to work with."""
    df = _independent(10)
    df.loc[0, "subject_id"] = df.loc[1, "subject_id"]
    validated = validate_spec(_spec(), df)
    assert any("clustered" in a for a in validated.ambiguities)
    assert not any("paired measurements" in a for a in validated.ambiguities)


def test_clean_independent_data_with_confirmed_design_has_no_ambiguity() -> None:
    assert validate_spec(_spec(), _independent()).ambiguities == []


def test_unconfirmed_design_is_asked_about_even_when_the_ids_agree() -> None:
    """Section 7.1: "Design is always confirmed by the user". Ids that look
    independent are evidence, not confirmation."""
    validated = validate_spec(_spec(design=Design.UNKNOWN), _independent())
    assert any("confirm" in a for a in validated.ambiguities)


def test_design_is_questioned_even_with_no_id_column_at_all() -> None:
    df = _independent().drop(columns=["subject_id"])
    spec = QuestionSpec(
        goal=Goal.COMPARE_GROUPS,
        variables={"outcome": "score", "group": "group"},
        design=Design.UNKNOWN,
    )
    validated = validate_spec(spec, df)
    assert any("no id column" in a for a in validated.ambiguities)


def test_an_unnamed_id_column_is_still_found_and_disclosed() -> None:
    """The LLM may not name the subject column at all. `detect_structure`
    finds it, and the question says so, because a user asked about a column
    they did not mention deserves to know where it came from."""
    df = paired_as_independent()
    spec = QuestionSpec(
        goal=Goal.COMPARE_GROUPS,
        variables={"outcome": "score", "group": "condition"},
        design=Design.INDEPENDENT,
    )
    validated = validate_spec(spec, df)
    assert validated.ambiguities
    assert "did not name it" in validated.ambiguities[0]


# --------------------------------------------------------------------------
# Section 7.1 step 1: measurement level
# --------------------------------------------------------------------------


def test_numeric_coded_ordinal_outcome_is_questioned_once_and_then_accepted() -> None:
    """Section 15.3's `ordinal_as_numeric`. The question names the concrete
    consequence (equal gaps), not the word "ordinal"."""
    rng = np.random.default_rng(4)
    df = pd.DataFrame(
        {
            "satisfaction": rng.integers(1, 6, 120),
            "group": ["A"] * 60 + ["B"] * 60,
            "subject_id": range(120),
        }
    )
    spec = _spec(variables={"outcome": "satisfaction", "group": "group", "subject": "subject_id"})
    validated = validate_spec(spec, df)
    assert any("gaps between values" in a for a in validated.ambiguities)

    confirmed = validate_spec(
        spec.model_copy(update={"confirmed_by_user": {"design", "outcome"}}), df
    )
    assert confirmed.ambiguities == []


def test_describe_spec_is_json_ready() -> None:
    described = describe_spec(validate_spec(_spec(), _independent()))
    assert described["goal"] == "compare_groups"
    assert described["design"] == "independent"
    assert described["confirmed_by_user"] == ["design"]
