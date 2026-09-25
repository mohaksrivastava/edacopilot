"""Method-family plumbing shared by every goal's rules (Section 7.3).

A family answers two questions the engine cannot answer generically:

- **Which methods are even in the conversation?** Section 7.3's table.
- **What does each spec role mean here?** `pearson(x, y)` and
  `point_biserial(binary, x)` both come from an ASSOCIATION question with
  roles `x` and `y`, but which column is which differs. The family resolves
  that once, into a `VariableMap` the assumption resolvers read.

Families never decide eligibility. They list candidates; Section 7.2's
engine decides, from the registry's declared assumptions and the checks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..checks import VariableMap
from ..spec import QuestionSpec

ParamBuilder = Callable[[QuestionSpec, pd.DataFrame], dict[str, Any]]


@dataclass(frozen=True)
class Method:
    """One candidate method within a family."""

    function: str
    params: ParamBuilder
    # Tie-breaker for Section 7.2's ranking, lowest first: how directly the
    # result answers the user's question in their own terms. A difference in
    # means beats a difference in mean ranks, which beats a probability of
    # superiority, which beats "the distributions differ somehow".
    interpretability: int = 5


@dataclass(frozen=True)
class MethodFamily:
    """A Section 7.3 family: its methods and its role assignment."""

    name: str
    methods: tuple[Method, ...]
    variables: Callable[[QuestionSpec, pd.DataFrame], VariableMap]
    note: str = ""
    # Some assumptions are about a grouping the raw data does not have a
    # column for. A factorial design's homoscedasticity assumption is about
    # its CELLS (factor_a x factor_b), not about either factor's margin, so
    # the family derives that column and the checks run against it. The
    # derived frame is what the cache keys on, so a family that adds a
    # column cannot read another family's cached facts by accident.
    prepare: Callable[[QuestionSpec, pd.DataFrame], pd.DataFrame] | None = None

    def prepared(self, spec: QuestionSpec, df: pd.DataFrame) -> pd.DataFrame:
        return df if self.prepare is None else self.prepare(spec, df)


@dataclass(frozen=True)
class RuleSet:
    """The rules for one `Goal` (Section 7.2's `RULES[spec.goal]`)."""

    goal_name: str
    family: Callable[[QuestionSpec, pd.DataFrame], MethodFamily]
    families: tuple[str, ...] = field(default_factory=tuple)


class UnsupportedQuestionError(NotImplementedError):
    """The question is well-formed but this milestone cannot serve it.

    Kept distinct from `InvalidSpecError` (a broken spec) and from
    `INELIGIBLE` (a method whose hard assumption the data fails): neither
    of those is true here. The methods simply do not exist yet, and saying
    "ineligible" would misreport a missing feature as a statistical verdict.
    """


def levels(df: pd.DataFrame, col: str) -> list[Any]:
    """Distinct non-missing levels of a grouping column, in a stable order:
    a categorical's own category order when it has one, else order of first
    appearance (which is what edacore's test functions use)."""
    series = df[col]
    if isinstance(series.dtype, pd.CategoricalDtype):
        present = set(series.dropna().unique())
        return [level for level in series.cat.categories if level in present]
    return list(pd.unique(series.dropna()))


def two_levels(df: pd.DataFrame, col: str) -> tuple[Any, Any]:
    found = levels(df, col)
    if len(found) != 2:
        raise ValueError(f"'{col}' has {len(found)} levels; expected exactly 2")
    return found[0], found[1]
