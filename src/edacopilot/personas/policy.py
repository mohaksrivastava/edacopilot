"""Persona policies: the schema, and loading the three YAML files
(ARCHITECTURE.md, Section 8.1).

A persona is *data*, not code. That is the whole point: what a persona will
and will not propose has to be inspectable and reviewable, and a policy
expressed as a YAML file can be read by someone who does not read Python.
This module is the contract that keeps those files honest -- a typo in a key
is a load-time error, not a persona that silently stops applying one of its
own rules.

`extra="forbid"` throughout is deliberate. A misspelled `method_pool_tag`
would otherwise be ignored, leaving the persona with an empty method pool
and no indication why it stopped proposing anything.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

PERSONA_DIR = Path(__file__).parent

# The order personas are shown in, everywhere. Fixed rather than
# alphabetical or filesystem order, so a card never reorders itself.
PERSONA_ORDER: tuple[str, ...] = ("professor", "consultant", "maverick")

BorderlineReading = Literal["pass", "fail"]
CltShortcut = Literal["none", "cochran"]


class NormalityPolicy(BaseModel):
    """How a persona reads the normality evidence.

    `method` is what the persona *cites* when it explains itself -- not a
    licence to ignore the checks it leaves out. A persona that could
    discard a Shapiro FAIL by not listing `shapiro` would be skipping an
    assumption silently, which is the failure mode this whole system
    exists to prevent. Relaxation happens only through `clt_shortcut` and
    `borderline_is`, both of which are explicit and bounded.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: list[str] = Field(min_length=1)
    borderline_is: BorderlineReading = "fail"
    clt_shortcut: CltShortcut = "none"


class VariancePolicy(BaseModel):
    """How a persona reads the equal-variance evidence.

    `strategy: always_robust` is a *tightening*, not a relaxation: it means
    the persona will not choose a method that assumes equal variance when
    the variance check is failing, because a Welch-type method is always
    available. It never waives the caveat on a method that does assume it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    method: str | None = None
    borderline_is: BorderlineReading = "fail"
    strategy: Literal["always_robust"] | None = None


ProposalTrigger = Literal["soft_assumption_not_met", "outliers_flagged"]


class ProposalPolicy(BaseModel):
    """When a persona offers its own method, and whom it defers to otherwise.

    The Maverick is the persona this exists for. Its value is in the cases
    where the standard choice is compromised; on data that meets every
    assumption, an unfamiliar method is a cost its own
    `must_state_tradeoff` constraint names ("less familiar to reviewers")
    with no matching benefit -- and it puts a divergence card in front of
    the user where the personas do not really disagree.

    Deferring is not silence: the concurring persona still appears, with a
    note saying it examined the data and had nothing to add.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    propose_when: list[ProposalTrigger] = Field(min_length=1)
    otherwise_concur_with: str


class EquivalenceRule(BaseModel):
    """Two methods that answer the same question under a stated condition
    (Section 8.3).

    `when_passes` is mandatory by design: a rule that merged methods
    unconditionally could hide a real disagreement. Student's t and Welch's
    t agree only when the variances really are equal, and the check has to
    say so before the rule applies.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    functions: frozenset[str] = Field(min_length=2)
    when_passes: str
    note: str

    def applies_to(self, functions: set[str]) -> bool:
        return len(functions) > 1 and functions <= self.functions


class MultipleTestingPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    default: str
    apply: Literal["always"] | dict[str, int]


class PersonaPolicy(BaseModel):
    """One persona's complete policy (Section 8.1).

    The `missing`, `outliers` and `transforms` sections are validated as
    present and well-shaped but not interpreted here: they belong to M10
    (missingness) and M11 (outliers, transforms). Typing them precisely now
    would only have to be redone when those milestones define what the keys
    mean.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    display_name: str
    alpha: float = Field(gt=0, lt=1)
    normality: NormalityPolicy
    variance: VariancePolicy | None = None
    method_pool_tags: set[str] = Field(min_length=1)
    prefer: list[str] = Field(min_length=1)
    multiple_testing: MultipleTestingPolicy
    explanation_style: str
    proposal_policy: ProposalPolicy | None = None
    missing: dict[str, Any] = Field(default_factory=dict)
    outliers: dict[str, Any] = Field(default_factory=dict)
    transforms: dict[str, Any] = Field(default_factory=dict)
    constraints: dict[str, Any] = Field(default_factory=dict)

    @field_validator("prefer")
    @classmethod
    def _known_preferences(cls, value: list[str]) -> list[str]:
        """Every preference key must be one the pick algorithm implements.

        An unrecognised key would be silently skipped, leaving the persona
        ranking by something other than what its file says.
        """
        from .engine import PREFERENCE_KEYS

        unknown = [key for key in value if key not in PREFERENCE_KEYS]
        if unknown:
            raise ValueError(
                f"unknown preference key(s) {unknown}; the pick algorithm implements "
                f"{sorted(PREFERENCE_KEYS)}"
            )
        return value

    @property
    def relaxes_borderline(self) -> bool:
        return self.normality.borderline_is == "pass"

    @property
    def avoids_equal_variance_methods(self) -> bool:
        return self.variance is not None and self.variance.strategy == "always_robust"


@lru_cache(maxsize=1)
def load_equivalences(directory: Path | None = None) -> tuple[EquivalenceRule, ...]:
    """Practical-equivalence rules (Section 8.3), from `equivalences.yaml`.

    Config rather than code so a rule can be reviewed, and added, without
    touching the pick algorithm -- the same reason the personas themselves
    are YAML.
    """
    path = (directory or PERSONA_DIR) / "equivalences.yaml"
    if not path.exists():
        return ()
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return tuple(EquivalenceRule.model_validate(rule) for rule in raw.get("rules", []))


def load_policy(path: Path) -> PersonaPolicy:
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name} does not contain a YAML mapping")
    return PersonaPolicy.model_validate(raw)


@lru_cache(maxsize=1)
def load_personas(directory: Path | None = None) -> tuple[PersonaPolicy, ...]:
    """The three personas, always in `PERSONA_ORDER`.

    Cached: the files do not change during a session, and re-parsing them
    per turn would make the pick's cost depend on how often it is called.
    """
    base = directory or PERSONA_DIR
    policies = {
        path.stem: load_policy(path)
        for path in sorted(base.glob("*.yaml"))
        if path.stem != "equivalences"
    }

    missing = [name for name in PERSONA_ORDER if name not in policies]
    if missing:
        raise FileNotFoundError(f"persona policy file(s) missing from {base}: {missing}")
    unexpected = sorted(set(policies) - set(PERSONA_ORDER))
    if unexpected:
        raise ValueError(
            f"unexpected persona file(s) in {base}: {unexpected}. Section 8 defines three "
            f"personas; adding a fourth changes the divergence card and needs a spec change."
        )
    for name, policy in policies.items():
        if policy.id != name:
            raise ValueError(f"{name}.yaml declares id '{policy.id}'; they must match")
        concur = policy.proposal_policy.otherwise_concur_with if policy.proposal_policy else None
        if concur is not None and concur not in PERSONA_ORDER:
            raise ValueError(
                f"{name}.yaml defers to '{concur}', which is not one of {PERSONA_ORDER}"
            )
        if concur == name:
            raise ValueError(f"{name}.yaml defers to itself")
    return tuple(policies[name] for name in PERSONA_ORDER)


def get_persona(persona_id: str) -> PersonaPolicy:
    for policy in load_personas():
        if policy.id == persona_id:
            return policy
    raise KeyError(f"no persona '{persona_id}'; have {PERSONA_ORDER}")
