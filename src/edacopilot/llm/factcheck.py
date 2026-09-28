"""Fact-check for `write_rationale` (ARCHITECTURE.md, Section 8.4).

Deterministic, and it has to be: rule 1 says the LLM never computes a
statistic, so nothing here trusts a number the model wrote unless that
exact number (rounding-tolerant) already exists in a fact the model was
given. A rationale that fails this check is not shown -- it goes back for
one retry with the failure appended, then falls back to
`personas/rationale.py`'s templates (Section 10.5).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from edacore.contracts import AssumptionCheck, PostHocResult, TestResult

from .contracts import PersonaRationale, RationaleBundle

_NUMBER = re.compile(r"-?\d+\.?\d*")

# Rounding tolerance: a fact quoted to 3 s.f. (Section 11's own rounding
# rule for what leaves the session) must still match a rationale that
# re-quotes it to 2 or 4.
_TOLERANCE = 5e-3


@dataclass(frozen=True)
class FactCheckFailure:
    reason: str


Fact = AssumptionCheck | TestResult | PostHocResult


_SCALAR_ATTRS = ("statistic", "p_value", "p_adjusted", "estimate", "effect_size")
_INTERVAL_ATTRS = ("ci", "effect_size_ci")


def _fact_numbers(fact: object) -> list[float]:
    numbers: list[float] = []
    for attr in _SCALAR_ATTRS:
        value = getattr(fact, attr, None)
        if isinstance(value, int | float):
            numbers.append(float(value))
    for attr in _INTERVAL_ATTRS:
        interval = getattr(fact, attr, None)
        if isinstance(interval, tuple | list):
            numbers.extend(float(v) for v in interval if isinstance(v, int | float))
    comparisons = getattr(fact, "comparisons", None)
    if isinstance(comparisons, list):
        for comparison in comparisons:
            numbers.extend(_fact_numbers(comparison))
    return numbers


def _known_numbers(facts: dict[str, Fact]) -> list[float]:
    numbers: list[float] = []
    for fact in facts.values():
        numbers.extend(_fact_numbers(fact))
    return numbers


def _text_numbers(text: str) -> list[float]:
    return [float(match) for match in _NUMBER.findall(text)]


def _numbers_are_grounded(text: str, known: list[float]) -> str | None:
    for number in _text_numbers(text):
        if not any(abs(number - k) <= _TOLERANCE * max(1.0, abs(k)) for k in known):
            return f"{number} does not match any cited fact (within tolerance)"
    return None


def check_rationale(
    bundle: RationaleBundle,
    *,
    facts: dict[str, Fact],
    picked_functions: set[str],
) -> FactCheckFailure | None:
    """Section 8.4's three checks, in the order that fails cheapest first.

    `facts` is the union of this turn's `AssumptionCheck`s and any
    `TestResult`/`PostHocResult` already on the ledger, keyed by `fact_id`
    -- exactly what `cited_facts` is allowed to name. `picked_functions` is
    the set of methods the personas actually picked, for the "method names
    must match the picks" rule.
    """
    known_numbers = _known_numbers(facts)
    for rationale in bundle.rationales:
        failure = _check_one(rationale, facts, known_numbers, picked_functions)
        if failure is not None:
            return failure
    comparison_failure = _numbers_are_grounded(bundle.comparison, known_numbers)
    if comparison_failure is not None:
        return FactCheckFailure(f"comparison: {comparison_failure}")
    return None


def _check_one(
    rationale: PersonaRationale,
    facts: dict[str, Fact],
    known_numbers: list[float],
    picked_functions: set[str],
) -> FactCheckFailure | None:
    for fact_id in rationale.cited_facts:
        if fact_id not in facts:
            return FactCheckFailure(f"{rationale.persona}: cited unknown fact_id {fact_id!r}")

    text = " ".join(
        [rationale.summary, rationale.consequence_if_ignored, *rationale.pros, *rationale.cons]
    )
    number_failure = _numbers_are_grounded(text, known_numbers)
    if number_failure is not None:
        return FactCheckFailure(f"{rationale.persona}: {number_failure}")

    if picked_functions:
        named = {
            function
            for function in picked_functions
            if function.replace("_", " ") in text.lower() or function in text
        }
        # Not every rationale has to name a method by its function name
        # (some read as prose), so this only flags the failure mode
        # Section 8.4 actually warns about: naming a function that is not
        # one of the picks at all.
        mentioned_functions = {
            word for word in re.findall(r"[a-z][a-z0-9_]{2,}", text.lower()) if "_" in word
        }
        bogus = mentioned_functions - picked_functions - named
        # Only treat it as a bogus method name if it collides with a
        # registered function name; ordinary words with underscores don't
        # occur in prose, so this is conservative rather than clever.
        from edacore.registry import registry

        registered = {spec.name for spec in registry.list()}
        bogus &= registered
        if bogus:
            return FactCheckFailure(
                f"{rationale.persona}: named {sorted(bogus)}, "
                f"not among the picks {sorted(picked_functions)}"
            )
    return None
