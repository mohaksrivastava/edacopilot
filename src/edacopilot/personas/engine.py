"""The persona engine: pick (8.2) and divergence detection (8.3).

Rule 2 is the constraint everything here is built around: *no persona can
propose an ineligible method*. The eligibility engine (Section 7) decides
what is valid before any persona sees the question, and a persona only ever
chooses among what it is handed. Nothing in this module can add a candidate,
and the one thing it can change -- a CAVEAT becoming ELIGIBLE -- is limited
to named, bounded rules that the persona's own YAML has to declare.

The three relaxation and tightening rules, in full:

- **`clt_shortcut: cochran`** may clear a normality caveat, and only when
  Cochran's rule (n > 25*skew^2 per group) is met. This is the one place a
  persona overrides a formal test, and it is defensible precisely because
  the test is answering a different question from the one that matters.
- **`borderline_is: pass`** may clear a caveat *all* of whose causes are
  BORDERLINE. A single FAIL anywhere in the cause list blocks it. A
  persona cannot reach eligibility by declining to look at a check.
- **`variance.strategy: always_robust`** removes methods that assume equal
  variance, full stop -- not only when the variance check happens to fail.
  That is what "always" means in the YAML comment ("always use Welch-type
  methods; no variance test needed"), and it is the defensible reading:
  Welch's t-test costs almost nothing when variances *are* equal, so a
  persona optimising for fewest defensible steps has no reason to run a
  variance test first and then decide. It is a tightening either way: it
  never waives the caveat on such a method, it declines to propose it.

Hard assumptions are untouchable throughout. An INELIGIBLE candidate is
filtered out before any persona rule runs, and a test asserts no persona
ever returns one.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from edacopilot.eligibility.engine import CandidateSet
from edacore.contracts import AssumptionCheck, Candidate, CheckStatus, Eligibility
from edacore.registry import registry

from .policy import EquivalenceRule, PersonaPolicy, load_equivalences, load_personas

# Assumption names whose caveats the CLT shortcut is allowed to clear.
_NORMALITY_ASSUMPTIONS = frozenset(
    {
        "normality",
        "normality_or_large_n",
        "normality_of_differences",
        "normality_within_groups",
        "bivariate_normality",
    }
)


class PersonaPick(BaseModel):
    """What one persona proposes, and why."""

    model_config = ConfigDict(frozen=True)

    persona: str
    display_name: str
    candidate: Candidate | None
    # The status the persona sees, after its own rules. Equal to
    # `candidate.eligibility` unless a rule relaxed it.
    eligibility: Eligibility | None = None
    relaxations: list[str] = Field(default_factory=list)
    excluded: dict[str, str] = Field(default_factory=dict)
    rationale: str = ""
    # Set when this persona deferred to another rather than proposing its
    # own method (Section 8.1's `proposal_policy`). Deferring is not
    # silence: the persona still appears, with this note saying so.
    concurs_with: str | None = None
    concur_note: str = ""

    @property
    def function(self) -> str | None:
        return self.candidate.function if self.candidate else None


class Proposal(BaseModel):
    """One distinct proposal, and the personas making it (Section 8.3)."""

    model_config = ConfigDict(frozen=True)

    function: str
    params: dict[str, object]
    personas: list[str]
    display_names: list[str]
    rationale: str = ""
    # When personas named *different* methods that a Section 8.3
    # equivalence rule collapsed into this one proposal: who named what,
    # and why the conclusion is the same.
    equivalent_methods: dict[str, str] = Field(default_factory=dict)
    equivalence_note: str = ""

    @property
    def is_equivalence(self) -> bool:
        return bool(self.equivalent_methods)


class PersonaVerdict(BaseModel):
    """All three picks, grouped into proposals (Section 8.3)."""

    model_config = ConfigDict(frozen=True)

    picks: list[PersonaPick]
    proposals: list[Proposal]

    @property
    def is_consensus(self) -> bool:
        return len(self.proposals) == 1

    @property
    def card(self) -> str:
        return "consensus" if self.is_consensus else "divergence"

    def pick_for(self, persona_id: str) -> PersonaPick:
        for pick in self.picks:
            if pick.persona == persona_id:
                return pick
        raise KeyError(f"no pick for persona '{persona_id}'")

    def functions(self) -> dict[str, str | None]:
        return {pick.persona: pick.function for pick in self.picks}


# --------------------------------------------------------------------------
# Reclassification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Reclassified:
    candidate: Candidate
    eligibility: Eligibility
    relaxation: str | None
    n_caveats: int


def _causes(candidate: Candidate, checks: dict[str, AssumptionCheck]) -> list[AssumptionCheck]:
    """The checks behind this candidate's status, in the order given."""
    return [checks[fact_id] for fact_id in candidate.reasons if fact_id in checks]


def _reclassify(
    policy: PersonaPolicy, candidate: Candidate, checks: dict[str, AssumptionCheck]
) -> _Reclassified:
    causes = _causes(candidate, checks)
    blocking = [c for c in causes if c.status in (CheckStatus.FAIL, CheckStatus.BORDERLINE)]
    n_caveats = len(blocking)

    if candidate.eligibility is not Eligibility.CAVEAT or not blocking:
        return _Reclassified(candidate, candidate.eligibility, None, n_caveats)

    normality_only = all(c.assumption in _NORMALITY_ASSUMPTIONS for c in blocking)

    # Rule 1: the CLT shortcut, which only ever applies to normality and
    # only when Cochran's rule is met (the engine already evaluated it).
    if policy.normality.clt_shortcut == "cochran" and normality_only:
        cochran = [c for c in checks.values() if c.method == "cochran_rule"]
        if cochran and all(c.status is CheckStatus.PASS for c in cochran):
            return _Reclassified(
                candidate,
                Eligibility.ELIGIBLE,
                f"Cochran's rule is met, so the normality caveat is waived "
                f"({cochran[0].threshold})",
                0,
            )

    # Rule 2: borderline-only caveats, for a persona that reads BORDERLINE
    # as passing. One FAIL anywhere blocks this.
    if policy.relaxes_borderline and all(c.status is CheckStatus.BORDERLINE for c in blocking):
        names = ", ".join(sorted({c.assumption for c in blocking}))
        return _Reclassified(
            candidate,
            Eligibility.ELIGIBLE,
            f"reads borderline as passing, and {names} is borderline rather than failing",
            0,
        )

    return _Reclassified(candidate, Eligibility.CAVEAT, None, n_caveats)


# --------------------------------------------------------------------------
# Preferences (the `prefer` keys a policy may declare)
# --------------------------------------------------------------------------

# Each returns a sort key, lowest first.
PreferenceFn = Callable[[_Reclassified, CandidateSet], float]


def _fewer_caveats(item: _Reclassified, _: CandidateSet) -> float:
    return float(item.n_caveats)


def _has_tag(tag: str) -> PreferenceFn:
    def key(item: _Reclassified, _: CandidateSet) -> float:
        return 0.0 if tag in item.candidate.tags else 1.0

    return key


def _exact_over_asymptotic(item: _Reclassified, _: CandidateSet) -> float:
    return 0.0 if "exact" in item.candidate.tags else 1.0


def _fewest_steps(item: _Reclassified, _: CandidateSet) -> float:
    """Methods that need a follow-up step rank below ones that do not.

    An omnibus test whose result only becomes actionable after a post-hoc
    procedure is two decisions, not one.
    """
    spec = registry.get(item.candidate.function)
    return 1.0 if "k_independent" in spec.tags or "k_related" in spec.tags else 0.0


def _interpretability(item: _Reclassified, candidates: CandidateSet) -> float:
    """The family's own ordering, which already encodes how directly each
    method answers the user's question (see `rules._base.Method`)."""
    order = [c.function for c in candidates.candidates]
    return float(order.index(item.candidate.function))


def _informative_effect_sizes(item: _Reclassified, _: CandidateSet) -> float:
    """Prefer a method whose effect size comes with a confidence interval.

    M3.4 made that true of all but nine, so this mostly separates the
    exempt ones (a KS D statistic, a distance correlation) from the rest.
    """
    exempt = {"ks_two_sample", "distance_correlation", "mutual_information", "anderson_ksamp"}
    return 1.0 if item.candidate.function in exempt else 0.0


PREFERENCE_KEYS: dict[str, PreferenceFn] = {
    "assumptions_clearly_met": _fewer_caveats,
    "exact_over_asymptotic": _exact_over_asymptotic,
    "fewest_steps": _fewest_steps,
    "interpretability": _interpretability,
    "robust_default": _has_tag("robust"),
    "robust": _has_tag("robust"),
    "resampling": _has_tag("resampling"),
    "informative_effect_sizes": _informative_effect_sizes,
}


# --------------------------------------------------------------------------
# pick (Section 8.2)
# --------------------------------------------------------------------------


def pick(
    policy: PersonaPolicy,
    candidates: CandidateSet,
    already: dict[str, PersonaPick] | None = None,
) -> PersonaPick:
    """Section 8.2's algorithm, with every exclusion recorded.

    `already` holds the picks of personas resolved earlier in
    `PERSONA_ORDER`, which a `proposal_policy` needs in order to defer to
    one of them. Absent it, a persona that would have deferred simply
    proposes its own method -- so `pick` stays usable on its own.

    A pick with `candidate=None` means the persona's method pool is empty
    for this family and it has nobody to defer to. That is a real answer
    ("I have nothing to offer here"), not an error.
    """
    checks = {check.fact_id: check for check in candidates.checks}
    excluded: dict[str, str] = {}
    pool: list[_Reclassified] = []

    for candidate in candidates.candidates:
        if candidate.eligibility is Eligibility.INELIGIBLE:
            excluded[candidate.function] = "ineligible: a hard assumption fails"
            continue
        if not (candidate.tags & policy.method_pool_tags):
            excluded[candidate.function] = (
                f"outside this persona's method pool ({', '.join(sorted(policy.method_pool_tags))})"
            )
            continue
        if policy.avoids_equal_variance_methods and _assumes_equal_variance(candidate):
            excluded[candidate.function] = (
                "assumes equal variance, and this persona always uses a variance-robust "
                "method instead"
            )
            continue
        pool.append(_reclassify(policy, candidate, checks))

    if not pool:
        empty = PersonaPick(
            persona=policy.id, display_name=policy.display_name, candidate=None, excluded=excluded
        )
        deferred = _maybe_defer(
            policy,
            candidates,
            already or {},
            reason="nothing in this family is inside this persona's method pool",
        )
        return deferred.model_copy(update={"excluded": excluded}) if deferred else empty

    eligible = [item for item in pool if item.eligibility is Eligibility.ELIGIBLE]
    if not eligible:
        # Section 8.2: "if empty, keep least-caveated".
        fewest = min(item.n_caveats for item in pool)
        eligible = [item for item in pool if item.n_caveats == fewest]

    chosen = min(eligible, key=lambda item: _sort_key(policy, item, candidates))
    own = PersonaPick(
        persona=policy.id,
        display_name=policy.display_name,
        candidate=chosen.candidate,
        eligibility=chosen.eligibility,
        relaxations=[chosen.relaxation] if chosen.relaxation else [],
        excluded=excluded,
    )
    deferred = _maybe_defer(policy, candidates, already or {}, reason=None)
    return deferred or own


def _has_unmet_soft_assumption(candidates: CandidateSet) -> bool:
    """Is anything the family depends on less than clean?

    Read across the whole candidate set, not one method: the Maverick's
    value is in noticing that *this data* is awkward, which is a property
    of the data rather than of whichever method happens to be ranked first.
    """
    soft_names = {
        name
        for candidate in candidates.candidates
        for name in registry.get(candidate.function).assumptions.soft
    }
    return any(
        check.assumption in soft_names
        and check.status in (CheckStatus.FAIL, CheckStatus.BORDERLINE)
        for check in candidates.checks
    )


def _has_flagged_outliers(candidates: CandidateSet) -> bool:
    """Are outliers flagged for the variables in play?

    For now this reads `check_influential_outliers`, the only outlier
    diagnostic that exists (M4 added it for Pearson). General outlier
    detection is Section 6.4, which lands in M11; when it does, its flags
    feed this same trigger and nothing else here has to change.
    """
    return any(
        check.assumption == "no_influential_outliers"
        and check.status in (CheckStatus.FAIL, CheckStatus.BORDERLINE)
        for check in candidates.checks
    )


_TRIGGERS = {
    "soft_assumption_not_met": _has_unmet_soft_assumption,
    "outliers_flagged": _has_flagged_outliers,
}


def _maybe_defer(
    policy: PersonaPolicy,
    candidates: CandidateSet,
    already: dict[str, PersonaPick],
    reason: str | None,
) -> PersonaPick | None:
    """Apply `proposal_policy`: concur with another persona instead of
    proposing, when nothing about this data calls for an alternative."""
    proposal_policy = policy.proposal_policy
    if proposal_policy is None:
        return None
    if reason is None and any(
        _TRIGGERS[trigger](candidates) for trigger in proposal_policy.propose_when
    ):
        return None

    target = already.get(proposal_policy.otherwise_concur_with)
    if target is None or target.candidate is None:
        return None

    note = (
        reason
        or "every assumption these methods make is met, so an alternative would "
        "add unfamiliarity without adding information"
    )
    return PersonaPick(
        persona=policy.id,
        display_name=policy.display_name,
        candidate=target.candidate,
        eligibility=target.eligibility,
        concurs_with=target.persona,
        concur_note=note,
    )


def _sort_key(
    policy: PersonaPolicy, item: _Reclassified, candidates: CandidateSet
) -> tuple[float, ...]:
    """The persona's declared preferences, in order, then the engine's own
    ranking as a final tie-break.

    The tie-break matters for reproducibility (rule 7): two methods a
    persona genuinely has no preference between must not swap places
    between runs.
    """
    keys = [PREFERENCE_KEYS[name](item, candidates) for name in policy.prefer]
    return (*keys, _interpretability(item, candidates))


def _assumes_equal_variance(candidate: Candidate) -> bool:
    spec = registry.get(candidate.function)
    return "equal_variance" in (*spec.assumptions.hard, *spec.assumptions.soft)


# --------------------------------------------------------------------------
# Divergence detection (Section 8.3)
# --------------------------------------------------------------------------


def _same_proposal(a: PersonaPick, b: PersonaPick) -> bool:
    """Section 8.3: "the same" means same function and materially identical
    params. Params are compared as given, not by identity -- two personas
    reaching the same method by different routes are still one proposal."""
    if a.candidate is None or b.candidate is None:
        return a.candidate is None and b.candidate is None
    return a.candidate.function == b.candidate.function and a.candidate.params == b.candidate.params


def _satisfied(rule: EquivalenceRule, checks: Sequence[AssumptionCheck]) -> bool:
    """A rule applies only when its named check actually PASSes.

    An absent check is not a pass: if the condition was never evaluated,
    there is no evidence the two methods agree, and merging them would
    hide a difference rather than reveal there is none.
    """
    relevant = [c for c in checks if c.assumption == rule.when_passes]
    return bool(relevant) and all(c.status is CheckStatus.PASS for c in relevant)


def _merge_equivalent(
    groups: list[list[PersonaPick]], checks: Sequence[AssumptionCheck]
) -> tuple[list[list[PersonaPick]], dict[int, EquivalenceRule]]:
    """Collapse groups whose methods a Section 8.3 rule makes equivalent.

    Only ever collapses: a rule can turn a divergence into a consensus, and
    never the reverse.
    """
    rules = [rule for rule in load_equivalences() if _satisfied(rule, checks)]
    if not rules or len(groups) < 2:
        return groups, {}

    merged: list[list[PersonaPick]] = []
    applied: dict[int, EquivalenceRule] = {}
    for group in groups:
        functions = {p.candidate.function for p in group if p.candidate}
        for index, existing in enumerate(merged):
            existing_functions = {p.candidate.function for p in existing if p.candidate}
            rule = next((r for r in rules if r.applies_to(existing_functions | functions)), None)
            if rule is not None:
                merged[index] = [*existing, *group]
                applied[index] = rule
                break
        else:
            merged.append(list(group))
    return merged, applied


def detect_divergence(
    picks: Sequence[PersonaPick], checks: Sequence[AssumptionCheck] = ()
) -> PersonaVerdict:
    """Group picks into distinct proposals, preserving persona order.

    Two personas naming different methods that answer the same question
    under a condition the data meets are one proposal, not two (Section
    8.3's practical equivalence). Presenting that as a choice would ask the
    user to decide between two answers that are not different, which
    teaches them to read the card as noise.
    """
    groups: list[list[PersonaPick]] = []
    for item in picks:
        for group in groups:
            if _same_proposal(group[0], item):
                group.append(item)
                break
        else:
            groups.append([item])

    groups, applied = _merge_equivalent(groups, checks)

    proposals = []
    for index, group in enumerate(groups):
        rule = applied.get(index)
        methods = {p.persona: p.candidate.function for p in group if p.candidate}
        distinct = len(set(methods.values())) > 1
        proposals.append(
            Proposal(
                function=group[0].candidate.function if group[0].candidate else "",
                params=dict(group[0].candidate.params) if group[0].candidate else {},
                personas=[p.persona for p in group],
                display_names=[p.display_name for p in group],
                equivalent_methods=methods if (rule and distinct) else {},
                equivalence_note=rule.note if (rule and distinct) else "",
            )
        )
    return PersonaVerdict(picks=list(picks), proposals=proposals)


def propose(candidates: CandidateSet) -> PersonaVerdict:
    """Every persona's pick for this candidate set, grouped (8.2 + 8.3).

    Personas resolve in `PERSONA_ORDER`, so a persona whose
    `proposal_policy` defers to another always has that other pick
    available. The order is fixed, which also keeps the result stable
    (rule 7).
    """
    picks: dict[str, PersonaPick] = {}
    for policy in load_personas():
        picks[policy.id] = pick(policy, candidates, already=picks)
    return detect_divergence(list(picks.values()), candidates.checks)
