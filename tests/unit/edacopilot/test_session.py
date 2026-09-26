"""The session object, provenance, and persistence (Sections 12.2 and 12.4).

M6's acceptance criteria are covered here: undo/branch/switch round-trips,
and a session resumed after a restart produces identical state. The resume
test is a real round-trip through the filesystem rather than a
reconstruction check, because "after a kernel restart" is the case that
matters and an in-process copy would not exercise it.

Rule 4 gets its own test. The log records the analysis and nothing about
the user, and that is a property of the model rather than of how it happens
to be used, so it is asserted against the model itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import edacore
from edacopilot.eligibility import Design, Goal, QuestionSpec
from edacopilot.session import (
    GITIGNORE_WARNING,
    ROOT_VERSION,
    Session,
    SessionNotFoundError,
    Step,
    assert_no_user_evaluation,
    list_sessions,
    resume,
)
from edacore.contracts import Eligibility, PairwiseComparison, PostHocResult
from edacore.stattests import k_independent, two_sample
from tests.scenarios.generators import heteroscedastic_groups


def _frame(seed: int = 1, n: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "value": np.concatenate([rng.normal(10, 2, n), rng.normal(12, 2, n)]),
            "group": ["A"] * n + ["B"] * n,
        }
    )


def _spec() -> QuestionSpec:
    return QuestionSpec(
        goal=Goal.COMPARE_GROUPS,
        variables={"outcome": "value", "group": "group"},
        design=Design.INDEPENDENT,
        confirmed_by_user={"design"},
    )


@pytest.fixture
def session(tmp_path: Path) -> Session:
    return Session.start(_frame(), session_id="test", root=tmp_path)


@pytest.fixture
def caveated_session(tmp_path: Path) -> Session:
    """A session whose candidates are not all eligible.

    The clean fixture above is deliberately unproblematic, so every method
    comes back ELIGIBLE and Section 7.5's friction never engages. Unequal
    variances give `student_t` a caveat to be overridden past.
    """
    return Session.start(heteroscedastic_groups(), session_id="caveat", root=tmp_path)


# --------------------------------------------------------------------------
# The eligibility hook (Section 7.2's cache, keyed by DatasetStore version)
# --------------------------------------------------------------------------


def test_candidates_are_computed_against_the_active_head_version(session: Session) -> None:
    candidates = session.candidates(_spec())
    assert candidates.dataset_version == ROOT_VERSION == session.version


def test_the_check_cache_is_reused_within_a_version(session: Session) -> None:
    session.candidates(_spec())
    runs = session.check_cache().n_runs
    assert runs > 0
    session.candidates(_spec())
    assert session.check_cache().n_runs == runs, "a second call recomputed cached checks"


def test_a_new_version_gets_a_new_cache_and_cannot_inherit_stale_facts(
    session: Session,
) -> None:
    """The correctness property Section 7.2's cache exists for. A transform
    that halves the data must not read a normality result computed about the
    data before it."""
    before = session.candidates(_spec())
    session.record_step(stage="quality", new_data=_frame(seed=2, n=15), label="resampled")
    after = session.candidates(_spec())

    assert after.dataset_version != before.dataset_version
    assert session.check_cache(before.dataset_version) is not session.check_cache(
        after.dataset_version
    )
    by_id_before = {c.fact_id: c for c in before.checks}
    by_id_after = {c.fact_id: c for c in after.checks}
    shared = set(by_id_before) & set(by_id_after)
    assert shared, "the two versions should share at least one fact_id"
    assert any(
        by_id_before[fact_id].statistic != by_id_after[fact_id].statistic
        for fact_id in shared
        if by_id_before[fact_id].statistic is not None
    ), "the same fact_id returned an identical statistic on different data: stale cache"


# --------------------------------------------------------------------------
# Accepting steps (Section 12.2)
# --------------------------------------------------------------------------


def test_accepting_a_test_records_the_step_and_the_ledger_entry(session: Session) -> None:
    candidates = session.candidates(_spec())
    verdict = session.propose(_spec())
    result = two_sample.welch_t(session.data, "group", "value")

    step = session.record_step(
        stage="hypothesis",
        chosen=candidates.get("welch_t"),
        chosen_via="consensus",
        spec=_spec(),
        candidates=candidates,
        verdict=verdict,
        result=result,
        family="value comparisons",
    )

    assert step.step_id == "s0"
    assert step.branch == "main"
    assert step.input_version == ROOT_VERSION
    assert step.output_version is None
    assert step.checks, "the step should record the fact_ids it rested on"
    assert {p.persona for p in step.proposals} == {"professor", "consultant", "maverick"}
    assert session.test_ledger.n_tests == 1
    assert session.ledger()[0]["family"] == "value comparisons"


def test_the_recorded_code_runs_and_reproduces_the_number(session: Session) -> None:
    """Rule 7, end to end. `Candidate.params` carries only the identifying
    arguments, so the defaults are filled from the function's signature --
    which means the recorded line is complete and says what it ran,
    including the seed."""
    candidates = session.candidates(_spec())
    result = two_sample.welch_t(session.data, "group", "value")
    step = session.record_step(stage="hypothesis", chosen=candidates.get("welch_t"), result=result)

    assert step.code.startswith("edacore.")
    assert "ci=0.95" in step.code
    replayed = eval(step.code, {"df": session.data, "edacore": edacore})  # noqa: S307
    assert replayed.p_value == result.p_value
    assert replayed.effect_size == result.effect_size


def test_accepting_a_transform_creates_a_version_linked_to_the_step(
    session: Session,
) -> None:
    step = session.record_step(
        stage="quality", new_data=session.data.head(20), label="first 20 rows"
    )
    assert step.output_version == "v1"
    assert session.version == "v1"
    assert session.store.record("v1").created_by_step == step.step_id
    assert len(session.data) == 20


def test_an_override_to_a_caveated_method_needs_a_typed_reason(caveated_session: Session) -> None:
    """Section 7.5: choosing a caveated or ineligible method requires a
    typed reason, stored in provenance and emitted on export."""
    candidates = caveated_session.candidates(_spec())
    caveated = next(c for c in candidates.candidates if c.eligibility is not Eligibility.ELIGIBLE)

    with pytest.raises(ValueError, match="typed reason"):
        caveated_session.record_step(stage="hypothesis", chosen=caveated, chosen_via="override")

    step = caveated_session.record_step(
        stage="hypothesis",
        chosen=caveated,
        chosen_via="override",
        override_reason="reviewers expect the textbook test",
    )
    assert step.is_override
    assert caveated_session.provenance.overrides() == [step]


def test_the_reason_is_emitted_as_a_code_comment(caveated_session: Session) -> None:
    """Section 7.5's last clause. The reason has to reach whoever reads the
    exported notebook, at the line it applies to -- not only the log."""
    candidates = caveated_session.candidates(_spec())
    caveated = next(c for c in candidates.candidates if c.eligibility is not Eligibility.ELIGIBLE)
    step = caveated_session.record_step(
        stage="hypothesis",
        chosen=caveated,
        chosen_via="override",
        override_reason="reviewers expect the textbook test",
    )
    assert step.code.startswith("# Override: reviewers expect the textbook test\n")
    assert caveated.function in step.code.splitlines()[1]


def test_choosing_an_eligible_method_needs_no_reason(session: Session) -> None:
    """Section 7.5 puts friction on CAVEAT and INELIGIBLE, and only those.

    The engine already ruled an ELIGIBLE method valid, so preferring one the
    personas did not name is a choice among valid methods -- not an override
    of a judgement, and not something to demand a justification for. Asking
    for one anyway would train the user to type anything to get past it,
    which would devalue the reason on the overrides that matter.
    """
    candidates = session.candidates(_spec())
    eligible = next(c for c in candidates.candidates if c.eligibility is Eligibility.ELIGIBLE)
    step = session.record_step(stage="hypothesis", chosen=eligible, chosen_via="override")
    assert step.override_reason is None
    assert not step.code.startswith("#")


def test_the_provenance_log_is_append_only_across_an_undo(session: Session) -> None:
    """An undo moves the dataset head; it does not erase the fact that the
    step happened. "What did I do and then undo?" is a fair question."""
    session.record_step(stage="quality", new_data=session.data.head(30))
    assert len(session.provenance) == 1
    session.undo()
    assert session.version == ROOT_VERSION
    assert len(session.provenance) == 1


def test_the_session_renders_as_a_runnable_script(session: Session) -> None:
    candidates = session.candidates(_spec())
    session.record_step(
        stage="hypothesis",
        chosen=candidates.get("welch_t"),
        result=two_sample.welch_t(session.data, "group", "value"),
    )
    session.record_step(
        stage="hypothesis",
        chosen=candidates.get("mann_whitney"),
        result=two_sample.mann_whitney(session.data, "group", "value"),
    )
    lines = [line for line in session.code().splitlines() if line.strip()]
    assert len(lines) == 2
    for line in lines:
        assert eval(line, {"df": session.data, "edacore": edacore}) is not None  # noqa: S307


# --------------------------------------------------------------------------
# Rule 4: the log records the analysis, not the user
# --------------------------------------------------------------------------


def test_the_step_model_has_no_field_that_evaluates_the_user() -> None:
    assert_no_user_evaluation(Step)


def test_the_guard_catches_a_field_that_would_evaluate_the_user() -> None:
    """Proving the guard works, rather than trusting it."""
    from pydantic import BaseModel

    class Tempting(BaseModel):
        step_id: str
        warnings_ignored: int = 0

    with pytest.raises(AssertionError, match="rule 4"):
        assert_no_user_evaluation(Tempting)


# --------------------------------------------------------------------------
# Section 12.3's rule, through the session
# --------------------------------------------------------------------------


def test_a_posthoc_accepted_through_the_session_does_not_count(session: Session) -> None:
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "value": np.concatenate([rng.normal(m, 8, 25) for m in (50, 55, 62)]),
            "group": sum(([label] * 25 for label in "ABC"), []),
        }
    )
    three = Session.start(df, session_id="three", root=session.directory.parent)
    omnibus = k_independent.one_way_anova(df, "value", "group")
    three.record_step(stage="hypothesis", result=omnibus, family="value")

    posthoc = PostHocResult(
        fact_id="tukey_hsd.value",
        function="tukey_hsd",
        method="tukey_hsd",
        p_adjust_method="tukey",
        comparisons=[PairwiseComparison(group_a="A", group_b="B", p_value=0.01, p_adjusted=0.03)],
        n={"total": 75},
    )
    three.record_step(stage="hypothesis", result=posthoc, family="value")

    assert three.test_ledger.n_tests == 1
    assert len(three.test_ledger) == 2


# --------------------------------------------------------------------------
# Persistence (Section 12.4) -- M6's acceptance criterion
# --------------------------------------------------------------------------


def _busy_session(root: Path) -> Session:
    """A session with something of every kind in it, to round-trip."""
    session = Session.start(_frame(), session_id="busy", root=root)
    candidates = session.candidates(_spec())
    verdict = session.propose(_spec())
    session.record_step(
        stage="hypothesis",
        chosen=candidates.get("welch_t"),
        chosen_via="consensus",
        spec=_spec(),
        candidates=candidates,
        verdict=verdict,
        result=two_sample.welch_t(session.data, "group", "value"),
        family="value comparisons",
    )
    session.record_step(stage="quality", new_data=session.data.head(60), label="trimmed")
    session.record_step(
        stage="hypothesis",
        chosen=candidates.get("mann_whitney"),
        chosen_via="override",
        override_reason="the reviewer asked for a rank-based test",
        result=two_sample.mann_whitney(session.data, "group", "value"),
        family="value comparisons",
    )
    session.branch_from("alternative")
    session.record_step(stage="quality", new_data=session.data.head(40), label="trimmed further")
    session.switch_branch("main")
    return session


def test_serialising_a_candidate_is_canonical() -> None:
    """A set's dump order is not a function of its value.

    CPython iterates a set in hash-table order, and for two sets with the
    same members that order can differ when their insertion histories
    collided differently -- which depends on the per-process string hash
    seed. That made a serialised session non-canonical: writing it to JSON
    and reading it back produced an equal model whose dump differed, in
    roughly one process in ten. It surfaced as an intermittent failure of
    the resume round-trip below, which is exactly the kind of bug that gets
    dismissed as flaky.

    Pinned here because the round-trip test cannot catch it reliably: it
    passes in ~90% of processes either way.
    """
    from edacore.contracts import Candidate

    members = {"parametric", "independent", "two_sample"}
    candidate = Candidate(
        function="welch_t", eligibility=Eligibility.ELIGIBLE, estimand="x", tags=members
    )
    assert candidate.model_dump(mode="json")["tags"] == sorted(members)

    # Built from a different insertion order, the dump must still match.
    reversed_order = Candidate(
        function="welch_t",
        eligibility=Eligibility.ELIGIBLE,
        estimand="x",
        tags=set(reversed(sorted(members))),
    )
    assert reversed_order.model_dump(mode="json") == candidate.model_dump(mode="json")

    round_tripped = Candidate.model_validate(candidate.model_dump(mode="json"))
    assert round_tripped.model_dump(mode="json") == candidate.model_dump(mode="json")


def test_serialising_a_question_spec_is_canonical() -> None:
    confirmed = {"design", "outcome", "x"}
    spec = QuestionSpec(
        goal=Goal.COMPARE_GROUPS,
        variables={"outcome": "value", "group": "group"},
        design=Design.INDEPENDENT,
        confirmed_by_user=confirmed,
    )
    assert spec.model_dump(mode="json")["confirmed_by_user"] == sorted(confirmed)


def test_a_resumed_session_has_identical_state(tmp_path: Path) -> None:
    """M6's acceptance criterion, as a round-trip equality test through the
    filesystem -- which is what "after a kernel restart" actually means."""
    original = _busy_session(tmp_path)
    before = original.to_state()

    restored = resume("busy", root=tmp_path)
    assert restored.to_state() == before
    # And byte-identical as JSON, which is what actually persists: equal
    # dicts whose serialisation differs would still corrupt the next save.
    assert json.dumps(restored.to_state(), indent=2) == json.dumps(before, indent=2)


def test_a_resumed_session_has_the_same_data_versions_and_ledger(tmp_path: Path) -> None:
    original = _busy_session(tmp_path)
    restored = resume("busy", root=tmp_path)

    assert restored.version == original.version
    assert restored.active_branch == original.active_branch
    assert restored.branches == original.branches
    assert restored.ledger() == original.ledger()
    assert restored.steps() == original.steps()
    assert restored.code() == original.code()
    for version_id in original.store.versions:
        pd.testing.assert_frame_equal(
            restored.store.get(version_id), original.store.get(version_id)
        )


def test_a_resumed_session_can_be_continued(tmp_path: Path) -> None:
    """Restoring is not read-only: the point is to carry on working."""
    original = _busy_session(tmp_path)
    steps_before = len(original.provenance)
    tests_before = original.test_ledger.n_tests

    restored = resume("busy", root=tmp_path)
    candidates = restored.candidates(_spec())
    restored.record_step(
        stage="hypothesis",
        chosen=candidates.get("yuen_trimmed_t"),
        result=two_sample.yuen_trimmed_t(restored.data, "group", "value"),
    )
    assert len(restored.provenance) == steps_before + 1
    assert restored.test_ledger.n_tests == tests_before + 1

    again = resume("busy", root=tmp_path)
    assert len(again.provenance) == steps_before + 1


def test_resuming_a_missing_session_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(SessionNotFoundError, match="no session state"):
        resume("nope", root=tmp_path)


def test_resuming_a_session_whose_parquet_is_gone_is_a_clear_error(tmp_path: Path) -> None:
    """Better to say the directory was moved than to fail later with a
    pandas read error on a path the user never typed."""
    session = _busy_session(tmp_path)
    session.store.path_for("v1").unlink()
    with pytest.raises(SessionNotFoundError, match="missing parquet"):
        resume("busy", root=tmp_path)


def test_starting_a_session_that_already_exists_is_refused(tmp_path: Path) -> None:
    Session.start(_frame(), session_id="dup", root=tmp_path)
    with pytest.raises(FileExistsError, match="use resume"):
        Session.start(_frame(), session_id="dup", root=tmp_path)


def test_sessions_can_be_listed(tmp_path: Path) -> None:
    Session.start(_frame(), session_id="a", root=tmp_path)
    Session.start(_frame(), session_id="b", root=tmp_path)
    assert list_sessions(tmp_path) == ["a", "b"]
    assert list_sessions(tmp_path / "nothing-here") == []


def test_the_state_file_is_written_atomically(session: Session) -> None:
    """A kernel that dies mid-write should cost the last step, not the whole
    history."""
    session.record_step(stage="quality", new_data=session.data.head(10))
    assert session.state_path.exists()
    assert not session.state_path.with_suffix(".json.tmp").exists()


def test_the_gitignore_warning_names_the_directory_and_the_risk() -> None:
    """Section 12.4 asks for this warning because the directory holds the
    user's data; a session committed to a shared repo is a silent leak."""
    assert ".edacopilot/" in GITIGNORE_WARNING
    assert "gitignore" in GITIGNORE_WARNING.lower()
    assert "parquet" in GITIGNORE_WARNING


def test_the_readme_carries_the_warning() -> None:
    readme = Path(__file__).parents[3] / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert ".edacopilot/" in text
    assert ".gitignore" in text


# --------------------------------------------------------------------------
# Undo / branch / switch through the session
# --------------------------------------------------------------------------


def test_undo_branch_switch_round_trip_through_the_session(session: Session) -> None:
    original = session.data.copy()
    session.record_step(stage="quality", new_data=session.data.head(50))
    session.record_step(stage="quality", new_data=session.data.head(30))
    assert session.version == "v2"

    session.branch_from("side", version="v1")
    assert session.active_branch == "side"
    assert session.version == "v1"
    session.record_step(stage="quality", new_data=session.data.head(25))

    session.switch_branch("main")
    assert session.version == "v2"
    session.undo()
    session.undo()
    assert session.version == ROOT_VERSION
    pd.testing.assert_frame_equal(session.data, original)

    session.switch_branch("side")
    assert len(session.data) == 25


def test_only_v0_and_the_head_are_in_memory_after_a_long_session(session: Session) -> None:
    for size in (60, 50, 40, 30):
        session.record_step(stage="quality", new_data=session.data.head(size))
    assert session.store.in_memory() == {ROOT_VERSION, session.version}


def test_a_candidate_set_from_a_resumed_session_still_respects_eligibility(
    tmp_path: Path,
) -> None:
    """Nothing about persistence may weaken rule 2."""
    original = _busy_session(tmp_path)
    del original
    restored = resume("busy", root=tmp_path)
    candidates = restored.candidates(_spec())
    verdict = restored.propose(_spec())
    ineligible = {
        c.function for c in candidates.candidates if c.eligibility is Eligibility.INELIGIBLE
    }
    for persona_pick in verdict.picks:
        assert persona_pick.function not in ineligible
