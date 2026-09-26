"""Every serialised model dumps as a function of its value (M6.1).

M6 found one instance of this bug and fixed it in place: `Candidate.tags`
and `QuestionSpec.confirmed_by_user` are `set`s, and CPython iterates a set
in hash-table order, which for two sets with the same members can differ
depending on how their insertion histories collided -- which in turn
depends on the per-process string hash seed. A session written to JSON and
read back therefore produced an *equal* model whose dump differed, in
roughly one process in ten.

Fixing the two fields that happened to be caught is not the same as fixing
the class of bug, so this module does two things:

1. **A guard over every model in both packages.** Any field whose
   annotation involves a `set` or `frozenset` must be covered by a
   `field_serializer`. A new model with a bare set fails here rather than
   intermittently in a resume test months later.
2. **An insertion-order test per field**, since a serializer that exists is
   not the same as a serializer that sorts.

Only sets are unordered. `dict` preserves insertion order in CPython 3.7+,
so a dict field is canonical as long as it is *built* deterministically;
every dict on these models is built by iterating a list (the family's
methods, `PERSONA_ORDER`, the dataset's columns, the store's versions in
creation order), so its order is a function of the value that built it.
"""

from __future__ import annotations

import importlib
import json
import pkgutil
import typing
from typing import Any

import pytest
from pydantic import BaseModel

import edacopilot
import edacore
from edacopilot.eligibility import Design, Goal, QuestionSpec
from edacopilot.personas.policy import EquivalenceRule, get_persona
from edacore.contracts import Candidate, Eligibility
from edacore.registry import registry


def _import_everything() -> None:
    """Import both packages fully, so `BaseModel.__subclasses__` is complete.

    A model in a module nothing has imported yet is exactly the one that
    would slip past this guard.
    """
    for package in (edacore, edacopilot):
        for module in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
            importlib.import_module(module.name)


def _is_unordered(annotation: Any) -> bool:
    """Whether a type annotation contains a set anywhere inside it."""
    if typing.get_origin(annotation) in (set, frozenset):
        return True
    return any(_is_unordered(arg) for arg in typing.get_args(annotation))


def _models() -> list[type[BaseModel]]:
    _import_everything()

    def descendants(cls: type[BaseModel]) -> list[type[BaseModel]]:
        out: list[type[BaseModel]] = []
        for sub in cls.__subclasses__():
            out.append(sub)
            out.extend(descendants(sub))
        return out

    found = {
        model
        for model in descendants(BaseModel)
        if model.__module__.startswith(("edacore.", "edacopilot."))
    }
    return sorted(found, key=lambda m: (m.__module__, m.__name__))


def _unordered_fields() -> list[tuple[type[BaseModel], str]]:
    return [
        (model, name)
        for model in _models()
        for name, field in model.model_fields.items()
        if _is_unordered(field.annotation)
    ]


def _serialised_fields(model: type[BaseModel]) -> set[str]:
    decorators = model.__pydantic_decorators__.field_serializers
    return {field for decorator in decorators.values() for field in decorator.info.fields}


def test_the_audit_finds_the_models_it_is_meant_to_cover() -> None:
    """The guard below is only worth anything if it sees the whole tree.

    An import error or a renamed package would make `_unordered_fields()`
    return nothing, and a test that asserts a property of an empty list
    passes for the wrong reason.
    """
    found = {f"{model.__name__}.{name}" for model, name in _unordered_fields()}
    assert found == {
        "Candidate.tags",
        "QuestionSpec.confirmed_by_user",
        "FunctionSpec.tags",
        "EquivalenceRule.functions",
        "PersonaPolicy.method_pool_tags",
    }


@pytest.mark.parametrize(
    ("model", "field"),
    _unordered_fields(),
    ids=lambda arg: arg.__name__ if isinstance(arg, type) else str(arg),
)
def test_every_set_field_has_a_sorting_serializer(model: type[BaseModel], field: str) -> None:
    assert field in _serialised_fields(model), (
        f"{model.__module__}.{model.__name__}.{field} is a set and has no field_serializer, "
        f"so its JSON dump is not a function of its value"
    )


def _dumps_identically(build: Any, members: list[str]) -> None:
    """Build the same value from two insertion orders; require one dump."""
    forwards = build(set(members))
    backwards = build(set(reversed(members)))
    assert forwards.model_dump(mode="json") == backwards.model_dump(mode="json")
    assert json.dumps(forwards.model_dump(mode="json"), sort_keys=True) == json.dumps(
        backwards.model_dump(mode="json"), sort_keys=True
    )


def test_candidate_tags_are_canonical() -> None:
    members = ["parametric", "independent", "two_sample", "robust"]
    _dumps_identically(
        lambda tags: Candidate(
            function="welch_t", eligibility=Eligibility.ELIGIBLE, estimand="x", tags=tags
        ),
        members,
    )
    built = Candidate(
        function="welch_t", eligibility=Eligibility.ELIGIBLE, estimand="x", tags=set(members)
    )
    assert built.model_dump(mode="json")["tags"] == sorted(members)


def test_question_spec_confirmations_are_canonical() -> None:
    members = ["design", "outcome", "x", "group"]
    _dumps_identically(
        lambda confirmed: QuestionSpec(
            goal=Goal.COMPARE_GROUPS,
            variables={"outcome": "value", "group": "group"},
            design=Design.INDEPENDENT,
            confirmed_by_user=confirmed,
        ),
        members,
    )


def test_function_spec_tags_are_canonical() -> None:
    """A real registered function, not a synthetic one: the dump that
    matters is the one `docs/eligibility_table.md` and the M9 tool metadata
    are rendered from."""
    spec = registry.get("welch_t")
    assert spec.model_dump(mode="json")["tags"] == sorted(spec.tags)

    members = sorted(spec.tags)
    _dumps_identically(lambda tags: spec.model_copy(update={"tags": tags}), members)


def test_equivalence_rule_functions_are_canonical() -> None:
    members = ["student_t", "welch_t", "yuen_trimmed_t"]
    _dumps_identically(
        lambda functions: EquivalenceRule(
            id="t_family",
            functions=frozenset(functions),
            when_passes="equal_variance",
            note="the same conclusion when variances are equal",
        ),
        members,
    )


def test_persona_policy_method_pool_tags_are_canonical() -> None:
    policy = get_persona("maverick")
    assert policy.model_dump(mode="json")["method_pool_tags"] == sorted(policy.method_pool_tags)

    members = sorted(policy.method_pool_tags)
    _dumps_identically(lambda tags: policy.model_copy(update={"method_pool_tags": tags}), members)
