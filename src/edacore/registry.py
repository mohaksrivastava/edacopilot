"""The edacore function registry (ARCHITECTURE.md, Section 5.4).

Every public edacore function is registered with the ``@register`` decorator,
which attaches a :class:`FunctionSpec` describing its stage, eligibility
metadata, and code template. The eligibility engine, persona engine, and
ad-hoc LLM function selection (Section 3.4) all work against this metadata
rather than against raw Python callables.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Collection, Iterable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from edacore.codegen import render_call

FunctionKind = Literal[
    "profile", "check", "test", "effect", "transform", "impute", "detect", "viz", "export"
]


class AssumptionSpec(BaseModel):
    """A function's hard (eligibility-blocking) and soft (caveat) assumptions."""

    model_config = ConfigDict(frozen=True)

    hard: list[str] = Field(default_factory=list)
    soft: list[str] = Field(default_factory=list)


class FunctionSpec(BaseModel):
    """Registration metadata for one edacore function."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str
    kind: FunctionKind
    stage: str
    tags: set[str] = Field(default_factory=set)
    assumptions: AssumptionSpec = Field(default_factory=AssumptionSpec)
    estimand: str | None = None
    code_template: str
    read_only: bool = True
    func: Callable[..., Any] = Field(exclude=True)


class Registry:
    """A lookup of :class:`FunctionSpec` by name, populated via :meth:`register`."""

    def __init__(self) -> None:
        self._functions: dict[str, FunctionSpec] = {}

    def register(
        self,
        *,
        name: str,
        kind: FunctionKind,
        stage: str,
        code_template: str,
        tags: Collection[str] = (),
        assumptions: dict[str, list[str]] | None = None,
        estimand: str | None = None,
        read_only: bool = True,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator that registers a function and returns it unchanged."""

        def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
            if name in self._functions:
                raise ValueError(f"function '{name}' is already registered")
            spec = FunctionSpec(
                name=name,
                kind=kind,
                stage=stage,
                tags=set(tags),
                assumptions=AssumptionSpec(**(assumptions or {})),
                estimand=estimand,
                code_template=code_template,
                read_only=read_only,
                func=func,
            )
            self._functions[name] = spec
            return func

        return decorator

    def get(self, name: str) -> FunctionSpec:
        try:
            return self._functions[name]
        except KeyError:
            raise KeyError(f"no function registered under '{name}'") from None

    def list(
        self,
        *,
        stage: str | None = None,
        kind: FunctionKind | None = None,
        tags: Iterable[str] | None = None,
    ) -> list[FunctionSpec]:
        results = list(self._functions.values())
        if stage is not None:
            results = [s for s in results if s.stage == stage]
        if kind is not None:
            results = [s for s in results if s.kind == kind]
        if tags is not None:
            wanted = set(tags)
            results = [s for s in results if wanted <= s.tags]
        return results

    def json_schema(self, name: str) -> dict[str, Any]:
        """JSON schema of a function's params, excluding ``df`` (Section 3.4)."""
        return _params_json_schema(self.get(name).func)

    def to_code(self, name: str, params: dict[str, Any], df_var: str = "df") -> str:
        """Render a function call as runnable Python source (Section 5.4)."""
        return render_call(self.get(name).code_template, params, df_var=df_var)

    def clear(self) -> None:
        self._functions.clear()


def _params_json_schema(func: Callable[..., Any]) -> dict[str, Any]:
    sig = inspect.signature(func, eval_str=True)
    fields: dict[str, Any] = {}
    for position, (param_name, param) in enumerate(sig.parameters.items()):
        if position == 0:
            continue  # the first argument is always `df`
        annotation = param.annotation if param.annotation is not inspect.Parameter.empty else Any
        default = param.default if param.default is not inspect.Parameter.empty else ...
        fields[param_name] = (annotation, default)
    model = create_model(f"{func.__name__}_Params", **fields)
    schema: dict[str, Any] = model.model_json_schema()
    return schema


# The module-level registry all `edacore` functions register into.
registry = Registry()
register = registry.register
