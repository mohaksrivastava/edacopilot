"""`docs/persona_picks.md` must match what the personas actually do.

Same reasoning as `test_eligibility_table_doc.py`: the file is the artefact
the maintainer reviews persona behaviour against, so a stale one would be
read as a statement about current behaviour. It is generated, and this test
fails if the committed file and a fresh render disagree.
"""

from __future__ import annotations

from edacopilot.personas import PERSONA_ORDER
from scripts.generate_persona_picks import DOC_PATH, render, scenarios


def test_the_committed_doc_matches_a_fresh_render() -> None:
    assert DOC_PATH.exists(), "docs/persona_picks.md is missing"
    assert DOC_PATH.read_text(encoding="utf-8") == render(), (
        "docs/persona_picks.md is stale; regenerate it with "
        "`python scripts/generate_persona_picks.py`"
    )


def test_every_persona_appears_in_every_scenario() -> None:
    """A persona silently missing from a row would read as agreement."""
    text = DOC_PATH.read_text(encoding="utf-8")
    for section in text.split("\n### ")[1:]:
        heading = section.split("\n", 1)[0]
        if "The engine stops here" in section:
            continue
        for persona_id in PERSONA_ORDER:
            display = persona_id.title()
            assert f"**{display}**" in section, f"{display} missing from '{heading}'"


def test_the_doc_covers_the_worked_example_and_the_trap_scenarios() -> None:
    names = {scenario.name for scenario in scenarios()}
    assert "worked example 7.4" in names
    for trap in ("heteroscedastic_groups", "heavy_tails_small_n"):
        assert any(trap in name for name in names), trap
    assert any("paired_as_independent" in name for name in names)
    assert any("ordinal_as_numeric" in name for name in names)


def test_every_scenario_row_carries_a_rationale() -> None:
    """The deliverable is the table *with* rationales -- a pick with no
    reason attached is not reviewable."""
    text = DOC_PATH.read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        if not line.startswith("| **"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        # The persona summary table at the top has five columns; the pick
        # tables have four. Only the latter carry rationales.
        if len(cells) == 4:
            rows.append(cells)

    assert rows, "no pick rows found in the document"
    for cells in rows:
        pick_cell, rationale = cells[1], cells[-1]
        assert pick_cell.startswith("`") or pick_cell == "\u2014", cells
        assert rationale.endswith("."), cells
