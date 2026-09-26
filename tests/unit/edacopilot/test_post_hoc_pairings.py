"""Which post-hoc follows which omnibus (ARCHITECTURE.md, Sections 9.5 and 12.3).

The pairing is fixed, and it is fixed for a statistical reason rather than
a stylistic one: a post-hoc's family-wise correction has to match the
design its omnibus was chosen for. Tukey's studentized range assumes the
equal variance that Welch's ANOVA was picked precisely to avoid assuming,
so offering Tukey after Welch would hand back the assumption the user had
just been steered away from — with the correction silently wrong rather
than visibly absent.

So each pairing gets a test, at two levels:

- the table itself, against the specification, so a typo in the mapping
  fails on its own terms rather than as a surprising transcript diff;
- each pairing end to end, from a real significant omnibus through the
  suggested button to the executed procedure, because a mapping that is
  right but unreachable (a role the params builder cannot fill, a family
  that never produces that candidate) would pass the first test and help
  nobody.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from edacopilot.session import Session
from edacopilot.stages import POST_HOC_FOR
from edacore.contracts import PostHocResult

# The specification, written out rather than imported, so this test states
# the requirement instead of restating the implementation.
PAIRINGS: dict[str, str] = {
    "one_way_anova": "tukey_hsd",
    "welch_anova": "games_howell",
    "alexander_govern": "games_howell",
    "kruskal_wallis": "dunn_test",
    "repeated_measures_anova": "paired_posthoc",
    "friedman": "nemenyi_friedman",
    "cochran_q": "mcnemar_posthoc",
    "permutation_anova": "permutation_posthoc",
}

# Section 9.5 permits either rank procedure after Friedman. Nemenyi is the
# current choice; Conover has more power but leans on the omnibus having
# been significant, which is a stronger dependency to take on by default.
FRIEDMAN_ALTERNATIVES = {"nemenyi_friedman", "conover_friedman"}


# --------------------------------------------------------------------------
# datasets, one per family
# --------------------------------------------------------------------------


def k_independent_frame() -> pd.DataFrame:
    """Three normal, equal-variance groups that really differ, so every
    k-independent omnibus is eligible and significant at once."""
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "v": np.concatenate(
                [rng.normal(0, 1, 35), rng.normal(1.0, 1, 35), rng.normal(2.0, 1, 35)]
            ),
            "g": ["a"] * 35 + ["b"] * 35 + ["c"] * 35,
        }
    )


def repeated_frame() -> pd.DataFrame:
    """25 subjects measured under three conditions, with a real within-subject
    effect and a subject term large enough for the pairing to matter."""
    rng = np.random.default_rng(11)
    n = 25
    base = rng.normal(0, 1, n).repeat(3)
    return pd.DataFrame(
        {
            "subject_id": np.repeat(np.arange(n), 3),
            "condition": np.tile(["t1", "t2", "t3"], n),
            "score": base + np.tile([0.0, 0.8, 1.8], n) + rng.normal(0, 0.4, 3 * n),
        }
    )


def repeated_binary_frame() -> pd.DataFrame:
    """The same shape with a yes/no outcome, which is Cochran's Q's family."""
    rng = np.random.default_rng(3)
    n = 30
    probabilities = np.tile([0.25, 0.5, 0.85], n)
    return pd.DataFrame(
        {
            "subject_id": np.repeat(np.arange(n), 3),
            "condition": np.tile(["t1", "t2", "t3"], n),
            "passed": (rng.random(3 * n) < probabilities).astype(int),
        }
    )


def _session_for(omnibus: str, tmp_path: Path) -> Session:
    """A session whose pending proposal contains `omnibus`, already asked."""
    if omnibus == "cochran_q":
        frame, outcome, group, design, subject = (
            repeated_binary_frame(),
            "passed",
            "condition",
            "repeated",
            "subject_id",
        )
    elif omnibus in ("repeated_measures_anova", "friedman"):
        frame, outcome, group, design, subject = (
            repeated_frame(),
            "score",
            "condition",
            "repeated",
            "subject_id",
        )
    else:
        frame, outcome, group, design, subject = (
            k_independent_frame(),
            "v",
            "g",
            "independent",
            None,
        )

    session = Session.start(frame, session_id=f"ph-{omnibus}", root=tmp_path)
    fields: dict[str, object] = {
        "goal": "compare_groups",
        "outcome": outcome,
        "group": group,
        "design": design,
        "confirmed_by_user": ["design"],
    }
    if subject is not None:
        fields["subject"] = subject
    session.ask(**fields)
    return session


def _run_omnibus(session: Session, omnibus: str):  # type: ignore[no-untyped-def]
    """Run one named omnibus, whatever the personas would have picked.

    `override` with a reason clears Section 7.5's gates for a CAVEAT
    candidate; an ELIGIBLE one ignores the reason. Either way the test
    controls which omnibus ran, which is the point.
    """
    pending = session.orchestrator.state.pending
    assert pending is not None, "no proposal on screen"
    assert omnibus in {c.function for c in pending.candidates.candidates}, (
        f"'{omnibus}' is not a candidate for family '{pending.candidates.family}'; "
        f"the fixture no longer exercises this pairing"
    )
    return session.override(omnibus, reason="pinning this omnibus for the pairing test")


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------


def test_the_pairing_table_is_exactly_the_specified_one() -> None:
    actual = {omnibus: option.function for omnibus, option in POST_HOC_FOR.items()}
    assert actual == PAIRINGS


def test_every_omnibus_section_12_3_names_has_a_pairing() -> None:
    """Section 12.3 lists the omnibus tests by name. One without a paired
    post-hoc would leave the user at "the groups differ" with no next step.
    """
    section_12_3 = {
        "one_way_anova",
        "welch_anova",
        "kruskal_wallis",
        "friedman",
        "repeated_measures_anova",
        "cochran_q",
        "alexander_govern",
        "permutation_anova",
    }
    assert section_12_3 <= set(POST_HOC_FOR)


def test_the_friedman_pairing_is_one_of_the_two_rank_procedures() -> None:
    assert POST_HOC_FOR["friedman"].function in FRIEDMAN_ALTERNATIVES


@pytest.mark.parametrize(("omnibus", "expected"), sorted(PAIRINGS.items()))
def test_each_pairing_reaches_the_right_procedure(
    tmp_path: Path, omnibus: str, expected: str
) -> None:
    """End to end: significant omnibus -> suggested button -> executed post-hoc."""
    session = _session_for(omnibus, tmp_path)
    result_card = _run_omnibus(session, omnibus)
    assert result_card.kind == "result", result_card.to_text()
    assert result_card.result is not None
    assert result_card.result.p_value is not None and result_card.result.p_value < 0.05, (
        f"the fixture for {omnibus} is no longer significant, so no post-hoc is licensed"
    )

    post_hoc_buttons = [a for a in result_card.actions if "post-hoc" in a.label.lower()]
    assert post_hoc_buttons, f"{omnibus} was significant but suggested no post-hoc"
    assert expected in post_hoc_buttons[0].label

    proposal = session.ask("run post-hoc comparisons")
    assert expected in proposal.title

    ran = session.accept()
    assert ran.kind == "result"
    assert isinstance(ran.result, PostHocResult)
    assert ran.result.function == expected


# --------------------------------------------------------------------------
# the pairing that matters most
# --------------------------------------------------------------------------


@pytest.mark.parametrize("omnibus", ["welch_anova", "alexander_govern"])
def test_tukey_is_never_suggested_after_an_unequal_variance_omnibus(
    tmp_path: Path, omnibus: str
) -> None:
    """Tukey's studentized range assumes the equal variance these tests were
    chosen to avoid assuming. Offering it here would hand back the
    assumption the user had just been steered away from, with a correction
    that is silently wrong rather than visibly absent.
    """
    session = _session_for(omnibus, tmp_path)
    card = _run_omnibus(session, omnibus)

    labels = " ".join(action.label for action in card.actions)
    assert "tukey" not in labels.lower()
    assert "games_howell" in labels

    proposal = session.ask("run post-hoc comparisons")
    assert "games_howell" in proposal.title
    ran = session.accept()
    assert isinstance(ran.result, PostHocResult)
    assert ran.result.function == "games_howell"


def test_tukey_is_suggested_after_the_equal_variance_anova(tmp_path: Path) -> None:
    """The contrast that makes the test above mean something."""
    session = _session_for("one_way_anova", tmp_path)
    card = _run_omnibus(session, "one_way_anova")
    labels = " ".join(action.label for action in card.actions)
    assert "tukey_hsd" in labels
    assert "games_howell" not in labels


def test_the_repeated_measures_pairing_adjusts_with_holm(tmp_path: Path) -> None:
    """Section 9.5's pairing for repeated_measures_anova is paired_posthoc
    *with Holm*, which is the procedure's own family-wise correction."""
    session = _session_for("repeated_measures_anova", tmp_path)
    _run_omnibus(session, "repeated_measures_anova")
    session.ask("run post-hoc comparisons")
    ran = session.accept()
    assert isinstance(ran.result, PostHocResult)
    assert ran.result.p_adjust_method == "holm"
    assert all(c.p_adjusted is not None for c in ran.result.comparisons)


@pytest.mark.parametrize("omnibus", sorted(PAIRINGS))
def test_no_post_hoc_is_offered_when_the_omnibus_is_not_significant(
    tmp_path: Path, omnibus: str
) -> None:
    """Running pairwise comparisons after a non-significant omnibus is the
    inflation the omnibus exists to prevent, so the pairing must not fire."""
    from edacopilot.stages import HypothesisStage
    from edacore.contracts import Candidate, Eligibility

    session = _session_for(omnibus, tmp_path)
    pending = session.orchestrator.state.pending
    assert pending is not None
    candidate = pending.candidates.get(omnibus)

    stage = HypothesisStage()
    null_result = stage.execute(candidate, session, pending.spec).result
    assert null_result is not None
    # Same result, forced non-significant: the pairing is a rule about the
    # p-value, so this isolates that rule from the data.
    not_significant = null_result.model_copy(update={"p_value": 0.42})
    assert stage.post_hoc_for(candidate, not_significant, pending.spec, session) is None

    significant = null_result.model_copy(update={"p_value": 0.001})
    offered = stage.post_hoc_for(candidate, significant, pending.spec, session)
    assert offered is not None
    assert offered.function == PAIRINGS[omnibus]
    assert isinstance(offered, Candidate)
    assert offered.eligibility in set(Eligibility)
