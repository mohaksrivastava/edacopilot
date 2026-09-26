"""Every catalogue subsection is delivered by some milestone (Section 16).

A Section 6 subsection with no milestone does not announce itself. It
simply never gets built, and the gap surfaces much later as a milestone
that cannot meet its own acceptance criteria — which is exactly how 6.12
(validity notes) went missing until M6.1 and 6.11 (visualisation) until
M7.1. Both were in Section 4.2's repository layout, which is why they read
as planned.

So the cross-check is a test rather than a thing to remember. It parses
ARCHITECTURE.md itself, because the specification is the artefact being
checked: a coverage table hand-maintained beside the milestone table would
be one more thing to forget.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ARCHITECTURE = Path(__file__).resolve().parents[3] / "ARCHITECTURE.md"
SPEC = ARCHITECTURE.read_text(encoding="utf-8")

# Milestones that exist in Section 16's table.
MILESTONE = re.compile(r"^\| (M\d+) \| ", re.MULTILINE)
# "### 6.4 Outliers (`outliers.py`) — stage `outliers`"
SUBSECTION = re.compile(r"^### (6\.\d+) (.+)$", re.MULTILINE)
# A row of the Section 6 coverage table: "| 6.4 Outliers | M11 | |"
COVERAGE_ROW = re.compile(r"^\| (6\.\d+) [^|]*\| (M\d+) \|", re.MULTILINE)


def subsections() -> dict[str, str]:
    return {number: title.strip() for number, title in SUBSECTION.findall(SPEC)}


def coverage() -> dict[str, str]:
    return dict(COVERAGE_ROW.findall(SPEC))


def milestones() -> set[str]:
    return set(MILESTONE.findall(SPEC))


def test_the_spec_parses_at_all() -> None:
    """Without this, every assertion below could pass on an empty match."""
    assert len(subsections()) >= 12
    assert len(milestones()) >= 16
    assert len(coverage()) >= 12


def test_every_section_6_subsection_is_claimed_by_a_milestone() -> None:
    uncovered = sorted(set(subsections()) - set(coverage()))
    assert not uncovered, (
        f"Section 6 subsection(s) {uncovered} appear in the function catalogue but in no "
        f"milestone's coverage table. A subsection nobody delivers is not a gap anyone "
        f"notices until the milestone that needs it fails its acceptance criteria."
    )


def test_the_coverage_table_names_no_subsection_that_does_not_exist() -> None:
    """The mirror-image error: a row left behind after a renumbering."""
    phantom = sorted(set(coverage()) - set(subsections()))
    assert not phantom, f"the coverage table names {phantom}, which Section 6 does not have"


@pytest.mark.parametrize("subsection", sorted(subsections()))
def test_each_subsection_is_assigned_to_a_real_milestone(subsection: str) -> None:
    assigned = coverage()[subsection]
    assert assigned in milestones(), (
        f"{subsection} is assigned to '{assigned}', which is not a row of Section 16's "
        f"milestone table"
    )


def test_the_two_subsections_that_were_missing_are_the_ones_recorded() -> None:
    """Pins the finding itself, so the audit's conclusion stays readable
    rather than becoming a table nobody remembers the reason for."""
    assert coverage()["6.11"] == "M8"
    assert coverage()["6.12"] == "M7"
