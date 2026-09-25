"""`docs/eligibility_table.md` must match what the engine enforces.

The table is the artefact the maintainer reviews Section 7's behaviour
against. A hand-maintained one would drift the first time an assumption
changed, and a stale table is worse than no table: it would be read as a
statement about the current code. So it is generated, and this test fails
if the committed file and a fresh render disagree.
"""

from __future__ import annotations

import pytest

from edacopilot.eligibility.checks import RESOLVERS
from edacore.registry import registry
from scripts.generate_eligibility_table import DECIDED_BY, DOC_PATH, render


def test_the_committed_table_matches_a_fresh_render() -> None:
    assert DOC_PATH.exists(), "docs/eligibility_table.md is missing"
    assert DOC_PATH.read_text(encoding="utf-8") == render(), (
        "docs/eligibility_table.md is stale; regenerate it with "
        "`python scripts/generate_eligibility_table.py`"
    )


def test_every_assumption_the_engine_resolves_is_documented() -> None:
    """A resolver with no row in the table would enforce something the
    reviewer never sees."""
    assert set(RESOLVERS) == set(DECIDED_BY), {
        "resolved_but_undocumented": sorted(set(RESOLVERS) - set(DECIDED_BY)),
        "documented_but_unresolved": sorted(set(DECIDED_BY) - set(RESOLVERS)),
    }


def test_every_check_the_table_names_is_registered() -> None:
    """The "Decided by" column has to name real functions, or the table
    cannot be used to audit anything."""
    for decided_by, _ in DECIDED_BY.values():
        if decided_by.startswith("--"):
            continue
        for part in decided_by.split("+"):
            name = part.strip()
            if name == "confirmed design":
                continue
            registry.get(name)  # raises if it is not registered


@pytest.mark.parametrize("function_spec", list(registry.list(stage="hypothesis", kind="test")))
def test_every_registered_tests_assumptions_appear_in_the_table(function_spec: object) -> None:
    declared = set(function_spec.assumptions.hard) | set(function_spec.assumptions.soft)  # type: ignore[attr-defined]
    assert declared <= set(DECIDED_BY), sorted(declared - set(DECIDED_BY))
