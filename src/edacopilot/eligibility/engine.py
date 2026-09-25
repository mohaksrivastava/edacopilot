"""The eligibility engine (ARCHITECTURE.md, Section 7.2).

`select_candidates` turns a confirmed `QuestionSpec` into a ranked
`CandidateSet`. It is deterministic end to end -- no LLM, no randomness, no
persona input -- because of rule 2: *no persona can propose an ineligible
method*. Personas (Section 8) choose among what this returns; they cannot
add to it.

Section 7.2's rule, verbatim:

    INELIGIBLE if any hard assumption FAILs
    CAVEAT     if any soft assumption FAILs or is BORDERLINE
    ELIGIBLE   otherwise

Three consequences of reading that literally, which this module does:

- **UNTESTABLE never blocks.** Independence and exchangeability are facts
  about data collection, not values (Section 5.2). They are surfaced as
  reasons on every candidate that rests on them, but they cannot make a
  method ineligible -- the user is the only one who can settle them.
- **A hard assumption's BORDERLINE does not downgrade anything**, because
  the rule names FAIL only. No hard check can currently return BORDERLINE
  (they are type, design and count checks), and a test asserts that stays
  true, so this is a documented invariant rather than a silent gap.
- **A method whose params cannot be built is not INELIGIBLE.** It is
  missing a parameter of the question (equivalence bounds, a reference
  value, a covariate), which is a question to ask, not a verdict about the
  data.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from edacore.contracts import AssumptionCheck, Candidate, CheckStatus, Eligibility
from edacore.registry import registry

from .checks import CheckCache, ResolutionContext, infer_types, resolve
from .rules import PENDING_GOALS, RULES, MethodFamily, UnsupportedQuestionError
from .spec import QuestionSpec, require_resolved

# Statuses that make a soft assumption a caveat (Section 7.2).
_CAVEAT_STATUSES = frozenset({CheckStatus.FAIL, CheckStatus.BORDERLINE})


class CandidateSet(BaseModel):
    """What Section 7.2 returns: the question, the evidence, the ranking."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    spec: QuestionSpec
    family: str
    checks: list[AssumptionCheck] = Field(default_factory=list)
    candidates: list[Candidate] = Field(default_factory=list)
    unavailable: dict[str, str] = Field(default_factory=dict)
    dataset_version: str = ""

    def by_status(self, status: Eligibility) -> list[Candidate]:
        return [c for c in self.candidates if c.eligibility is status]

    @property
    def eligible(self) -> list[Candidate]:
        return self.by_status(Eligibility.ELIGIBLE)

    def get(self, function: str) -> Candidate:
        for candidate in self.candidates:
            if candidate.function == function:
                return candidate
        raise KeyError(f"'{function}' is not a candidate in family '{self.family}'")

    def statuses(self) -> dict[str, Eligibility]:
        return {c.function: c.eligibility for c in self.candidates}


@dataclass(frozen=True)
class _Evaluation:
    eligibility: Eligibility
    reasons: list[str]
    checks: list[AssumptionCheck]
    n_caveats: int


def _evaluate(function: str, ctx: ResolutionContext) -> _Evaluation:
    spec = registry.get(function)
    hard_checks: list[AssumptionCheck] = []
    soft_checks: list[AssumptionCheck] = []
    reasons: list[str] = []
    blocking: list[str] = []
    caveats: list[str] = []
    untestable: list[str] = []

    for assumption in spec.assumptions.hard:
        found = resolve(assumption, ctx)
        hard_checks.extend(found.all_checks)
        for check in found.graded:
            if check.status is CheckStatus.FAIL:
                blocking.append(check.fact_id)
            elif check.status is CheckStatus.UNTESTABLE:
                untestable.append(check.fact_id)

    for assumption in spec.assumptions.soft:
        found = resolve(assumption, ctx)
        soft_checks.extend(found.all_checks)
        for check in found.graded:
            if check.status in _CAVEAT_STATUSES:
                caveats.append(check.fact_id)
            elif check.status is CheckStatus.UNTESTABLE:
                untestable.append(check.fact_id)

    if blocking:
        eligibility = Eligibility.INELIGIBLE
        reasons = blocking
    elif caveats:
        eligibility = Eligibility.CAVEAT
        reasons = caveats
    else:
        eligibility = Eligibility.ELIGIBLE
        reasons = []

    # Untestable assumptions ride along on every status: they are the ones
    # the user has to vouch for, and hiding them on an ELIGIBLE candidate is
    # exactly how a junior analyst ends up assuming independence.
    #
    # Deduplicated, order preserved: one check can satisfy two assumptions
    # (a Shapiro result answers both `normality` and `normality_or_large_n`),
    # and listing the same fact twice reads like two separate problems.
    reasons = list(dict.fromkeys([*reasons, *untestable]))
    return _Evaluation(eligibility, reasons, [*hard_checks, *soft_checks], len(set(caveats)))


_RANK_ORDER = {Eligibility.ELIGIBLE: 0, Eligibility.CAVEAT: 1, Eligibility.INELIGIBLE: 2}


def _rank_key(
    candidate: Candidate, n_caveats: int, interpretability: int
) -> tuple[int, int, int, int, str]:
    """Section 7.2's ranking: within a status, fewer caveats first, then
    broader validity (the `robust` tag), then interpretability.

    The function name is the final tie-break so the order is total and
    stable -- two methods with identical metadata must not swap places
    between runs (rule 7).
    """
    robust = 0 if "robust" in candidate.tags else 1
    return (
        _RANK_ORDER[candidate.eligibility],
        n_caveats,
        robust,
        interpretability,
        candidate.function,
    )


def select_candidates(
    spec: QuestionSpec,
    df: pd.DataFrame,
    *,
    dataset_version: str | None = None,
    cache: CheckCache | None = None,
) -> CandidateSet:
    """Rank every method in this question's family (Section 7.2).

    `spec` must already be validated (`validate_spec`) and free of
    ambiguities; a spec that still has any raises `AmbiguousSpecError`
    rather than proceeding with a guessed design.
    """
    require_resolved(spec)

    rules = RULES.get(spec.goal)
    if rules is None:
        pending = PENDING_GOALS.get(spec.goal, "a later milestone")
        raise UnsupportedQuestionError(
            f"goal '{spec.goal.value}' has no hypothesis-testing methods; it is served by {pending}"
        )

    family: MethodFamily = rules.family(spec, df)
    # A family may derive a column its assumptions are about (see
    # MethodFamily.prepare). Checks run against that frame; the methods
    # themselves are still described in terms of the user's own columns.
    prepared = family.prepared(spec, df)
    cache = cache or CheckCache(prepared, dataset_version)
    ctx = ResolutionContext(
        df=prepared,
        spec=spec,
        variables=family.variables(spec, prepared),
        cache=cache,
        semantic_types=infer_types(prepared),
    )

    candidates: list[Candidate] = []
    sort_keys: list[tuple[int, int, int, int, str]] = []
    all_checks: dict[str, AssumptionCheck] = {}
    unavailable: dict[str, str] = {}

    for method in family.methods:
        try:
            params = method.params(spec, df)
        except UnsupportedQuestionError as exc:
            unavailable[method.function] = str(exc)
            continue

        evaluation = _evaluate(method.function, ctx)
        for check in evaluation.checks:
            all_checks.setdefault(check.fact_id, check)

        function_spec = registry.get(method.function)
        candidate = Candidate(
            function=method.function,
            params=params,
            eligibility=evaluation.eligibility,
            reasons=evaluation.reasons,
            tags=set(function_spec.tags),
            estimand=function_spec.estimand or "",
        )
        candidates.append(candidate)
        sort_keys.append(_rank_key(candidate, evaluation.n_caveats, method.interpretability))

    ranked = [c for _, c in sorted(zip(sort_keys, candidates, strict=True), key=lambda p: p[0])]

    return CandidateSet(
        spec=spec,
        family=family.name,
        checks=sorted(all_checks.values(), key=lambda c: c.fact_id),
        candidates=ranked,
        unavailable=unavailable,
        dataset_version=cache.dataset_version,
    )
