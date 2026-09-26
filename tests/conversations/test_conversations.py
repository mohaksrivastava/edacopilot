"""Golden conversation tests (ARCHITECTURE.md, Section 15.5).

Six transcripts, each replayed against a real session with no LLM (rule 9)
and compared to a committed file. The comparison is over the whole rendered
transcript — every card, plus the provenance, versions, ledger count and
recorded code afterwards — so this fails on a behaviour change and on a
wording change alike. That strictness is intended: the wording of a card is
most of what this product is, and it should not be able to change without
someone looking at the diff.

Regenerate with:

    python scripts/generate_transcripts.py

`docs/sample_conversation.md` is rendered from the same transcript by the
same renderer, so the document cannot drift from what is asserted here.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.generate_transcripts import SAMPLE, SAMPLE_DOC, sample_document
from tests.conversations.runner import golden_path, render, run
from tests.conversations.transcripts import Transcript, by_name, transcripts


@pytest.mark.parametrize("transcript", transcripts(), ids=lambda t: t.name)
def test_transcript_matches_its_golden_file(transcript: Transcript, tmp_path: Path) -> None:
    path = golden_path(transcript.name)
    assert path.exists(), (
        f"no golden file for '{transcript.name}'; run `python scripts/generate_transcripts.py`"
    )
    assert render(transcript, tmp_path) == path.read_text(encoding="utf-8"), (
        f"'{transcript.name}' no longer renders as its golden file. If the change is "
        f"intended, run `python scripts/generate_transcripts.py` and review the diff."
    )


def test_there_are_at_least_six_transcripts() -> None:
    """Section 15.5's coverage, named rather than counted.

    A count alone would pass if six transcripts all covered the same path.
    """
    names = {transcript.name for transcript in transcripts()}
    assert names >= {
        "worked_example_7_4",
        "paired_as_independent",
        "ineligible_override",
        "undo_then_branch",
        "kruskal_then_posthoc",
        "jump_with_missing_data",
    }


def test_the_sample_document_is_the_worked_example(tmp_path: Path) -> None:
    """M7's deliverable for review, kept in step with the test above."""
    expected = sample_document(render(by_name(SAMPLE), tmp_path))
    assert SAMPLE_DOC.exists(), "run `python scripts/generate_transcripts.py`"
    assert SAMPLE_DOC.read_text(encoding="utf-8") == expected


def test_a_transcript_replays_identically_twice(tmp_path: Path) -> None:
    """Rule 7's other half: the same actions produce the same numbers.

    Resampling methods carry fixed seeds and the ranking is total, so two
    runs of one transcript must be byte-identical. A difference here would
    mean something in the path is reading the clock, the hash seed or an
    unseeded RNG.
    """
    transcript = by_name("kruskal_then_posthoc")
    first = render(transcript, tmp_path / "a")
    second = render(transcript, tmp_path / "b")
    assert first == second


# --------------------------------------------------------------------------
# What each transcript is actually for, asserted separately from its text.
#
# The golden comparison above would catch any of these changing, but it
# would not say *what* broke. These name the property, so a failure reads
# as "the design question stopped being asked" rather than "line 41 of a
# text file differs".
# --------------------------------------------------------------------------


def test_the_worked_example_reproduces_section_7_4(tmp_path: Path) -> None:
    session, _ = run(by_name("worked_example_7_4"), tmp_path)
    step = session.provenance.steps[-1]
    assert step.chosen is not None
    assert step.chosen.function == "mann_whitney"
    assert step.chosen_via == "professor"
    proposed = {view.persona: view.function for view in step.proposals}
    assert proposed == {
        "professor": "mann_whitney",
        "consultant": "mann_whitney",
        "maverick": "yuen_trimmed_t",
    }


def test_the_paired_trap_asks_before_it_proposes(tmp_path: Path) -> None:
    transcript = by_name("paired_as_independent")
    session, blocks = run(transcript, tmp_path)
    assert "[question] One question first" in blocks[0]
    assert "Proposals:" not in blocks[0], "a persona was consulted before the design was settled"
    assert "[consensus]" in blocks[1]
    assert session.provenance.steps[0].chosen is not None
    assert session.provenance.steps[0].chosen.function == "paired_t"


def test_the_ineligible_override_needs_both_gates(tmp_path: Path) -> None:
    session, blocks = run(by_name("ineligible_override"), tmp_path)
    assert "[override_confirm]" in blocks[1]
    assert "second, separate confirmation" in blocks[1]
    # The first override produced no step; only the confirmed one did.
    assert len(session.provenance) == 1
    step = session.provenance.steps[0]
    assert step.override_reason == "the reviewer asked for a trend test"
    assert step.code.startswith("# Override: the reviewer asked for a trend test")


def test_undo_keeps_the_version_and_the_step(tmp_path: Path) -> None:
    session, _ = run(by_name("undo_then_branch"), tmp_path)
    assert "v1" in session.store.versions, "an undo deleted the version it moved off"
    assert len(session.provenance) == 1, "an undo erased a step from the log"
    assert session.active_branch == "main"
    assert session.version == "v0"


def test_the_posthoc_does_not_enter_the_session_adjustment(tmp_path: Path) -> None:
    session, _ = run(by_name("kruskal_then_posthoc"), tmp_path)
    assert session.test_ledger.n_tests == 1, "the post-hoc was counted as a session test"
    assert len(session.test_ledger.entries) == 2, "the post-hoc was not recorded at all"
    posthoc = session.test_ledger.entries[-1]
    assert posthoc.function == "dunn_test"
    assert not posthoc.counts_toward_session
    assert posthoc.p_adjusted is None


def test_jumping_past_missingness_warns_with_a_real_number(tmp_path: Path) -> None:
    _, blocks = run(by_name("jump_with_missing_data"), tmp_path)
    warning = blocks[1]
    assert "[warning]" in warning
    assert "Missing values haven't been reviewed" in warning
    assert "`income`" in warning
    assert "Skipped: quality, missingness, outliers, transform" in warning
