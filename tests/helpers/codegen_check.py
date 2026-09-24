"""Shared test helper enforcing ARCHITECTURE.md rule 7.

Rule 7: "Every accepted step must be reproducible as plain Python code in
the exported notebook." `assert_codegen_matches` proves that for a given
registered function: the code `registry.to_code(...)` renders, when
executed against a fixture DataFrame, evaluates to the same result as
calling the function directly. Every edacore function's test should call
this at least once.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from edacore.registry import Registry


def assert_codegen_matches(
    registry: Registry,
    name: str,
    params: dict[str, Any],
    df: pd.DataFrame,
    *,
    df_var: str = "df",
    namespace: dict[str, Any] | None = None,
) -> None:
    """Assert that `registry.to_code(name, params, df_var)`, when exec'd
    against `df`, reproduces the result of calling the registered function
    directly with the same `df` and `params`.

    `namespace` supplies any names the code_template references besides
    `df_var` (e.g. an imported module like `scipy`).
    """
    spec = registry.get(name)
    code = registry.to_code(name, params, df_var=df_var)

    exec_namespace: dict[str, Any] = {df_var: df, **(namespace or {})}
    codegen_result = eval(code, exec_namespace)
    direct_result = spec.func(df, **params)

    assert codegen_result == direct_result, (
        f"registry.to_code({name!r}, {params!r}) rendered {code!r}, which evaluated to "
        f"{codegen_result!r}, but calling the function directly gave {direct_result!r}"
    )
