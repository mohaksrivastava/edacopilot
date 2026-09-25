"""Every declared assumption must resolve to something real.

Section 5.4 lets a registered function declare its assumptions as free
strings. Nothing in the type system connects `"normality_or_large_n"` to
code, so a typo, a rename, or a new method copied from an old one can
declare an assumption the eligibility engine has never heard of. The
engine raises on an unknown name rather than skipping it (Section 7.2), but
that only surfaces when someone happens to run that method's family on
real data.

This file makes it a build-time property instead: for every registered
function, every hard and soft assumption resolves either to a registered
`edacore.assumptions` check or to an explicit ask-user handler that returns
UNTESTABLE. CI fails otherwise.

The distinction matters and is asserted, not assumed. An assumption backed
by a check is decided from the data. An assumption backed by an ask-user
handler can never be decided from the data at all (Section 5.2) -- it is
surfaced on every candidate that rests on it and left to the user. What
must never exist is a third category: a name that silently resolves to
nothing, which would let a method be reported as eligible without that
assumption ever having been looked at.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from edacopilot.eligibility import CheckCache, Design, Goal, QuestionSpec
from edacopilot.eligibility.checks import (
    RESOLVERS,
    ResolutionContext,
    resolve,
)
from edacore.contracts import CheckStatus
from edacore.registry import registry

# Assumptions that cannot be decided from data, by their nature (Section
# 5.2). Each must resolve to UNTESTABLE, and nothing else may.
ASK_USER_ASSUMPTIONS = frozenset({"exchangeability", "ordered_sequence"})


def _declared_assumptions() -> dict[str, set[str]]:
    """Every assumption name in the registry, mapped to the functions that
    declare it. Covers every kind and stage, not just hypothesis tests, so
    a future transform or check declaring assumptions is covered too."""
    declared: dict[str, set[str]] = {}
    for spec in registry.list():
        for name in (*spec.assumptions.hard, *spec.assumptions.soft):
            declared.setdefault(name, set()).add(spec.name)
    return declared


DECLARED = _declared_assumptions()


def _context() -> ResolutionContext:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "value": np.concatenate([rng.normal(10, 2, 30), rng.normal(12, 2, 30)]),
            "group": ["A"] * 30 + ["B"] * 30,
            "subject": list(range(60)),
            "x": rng.normal(0, 1, 60),
            "y": rng.normal(0, 1, 60),
        }
    )
    spec = QuestionSpec(
        goal=Goal.COMPARE_GROUPS,
        variables={"outcome": "value", "group": "group"},
        design=Design.INDEPENDENT,
        confirmed_by_user={"design"},
    )
    return ResolutionContext(
        df=df,
        spec=spec,
        variables={"outcome": "value", "group": "group", "x": "x", "y": "y"},
        cache=CheckCache(df),
    )


def test_every_declared_assumption_has_a_resolver() -> None:
    """The invariant itself. An unresolvable assumption is worse than a
    missing one: the method would be reported as eligible without it ever
    being checked."""
    unresolvable = {
        name: sorted(functions) for name, functions in DECLARED.items() if name not in RESOLVERS
    }
    assert not unresolvable, (
        "these assumptions are declared by registered functions but have no resolver "
        f"in edacopilot.eligibility.checks: {unresolvable}"
    )


def test_no_resolver_exists_for_an_assumption_nobody_declares() -> None:
    """The other direction: a resolver with no declaring function is dead
    code that a reader would take for live behaviour."""
    orphans = sorted(set(RESOLVERS) - set(DECLARED))
    assert not orphans, (
        f"these resolvers are defined but no registered function declares them: {orphans}"
    )


@pytest.mark.parametrize("assumption", sorted(DECLARED))
def test_each_assumption_resolves_to_a_check_or_an_ask_user_handler(assumption: str) -> None:
    """Resolving must produce real `AssumptionCheck` evidence -- never an
    empty result, which would read as "nothing was wrong"."""
    resolution = resolve(assumption, _context())
    assert resolution.graded, (
        f"'{assumption}' resolved to no graded checks at all; an assumption that produces "
        "no evidence cannot be distinguished from one that passed"
    )
    for check in resolution.all_checks:
        assert check.fact_id, f"'{assumption}' produced a check with no fact_id"
        assert check.consequence, (
            f"'{assumption}' produced a check with no consequence; Section 6.6 requires "
            "one sentence saying what goes wrong if it is ignored"
        )
        assert check.threshold, f"'{assumption}' produced a check with no threshold"


@pytest.mark.parametrize("assumption", sorted(ASK_USER_ASSUMPTIONS))
def test_ask_user_assumptions_resolve_to_untestable(assumption: str) -> None:
    """These are facts about how the data was collected. Reporting them as
    PASS because nothing contradicted them is the exact failure this tool
    exists to prevent."""
    assert assumption in DECLARED, f"'{assumption}' is no longer declared by any function"
    graded = resolve(assumption, _context()).graded
    assert all(c.status is CheckStatus.UNTESTABLE for c in graded), [
        (c.fact_id, c.status.value) for c in graded
    ]


def test_an_unknown_assumption_raises_rather_than_resolving_to_nothing() -> None:
    with pytest.raises(KeyError, match="no resolver"):
        resolve("an_assumption_nobody_implemented", _context())


def test_every_check_named_by_a_resolver_is_registered() -> None:
    """Resolvers reach edacore through the registry by name. A renamed check
    would otherwise fail only when that branch happened to run."""
    context = _context()
    for assumption in sorted(DECLARED):
        resolve(assumption, context)  # raises KeyError from the registry if a name is stale
