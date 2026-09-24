"""Render a registered function's code template into runnable Python source.

Every registered edacore function carries a ``code_template`` — a
``str.format``-style string written against public libraries (or edacore
itself) so an accepted step can be exported as plain, reproducible code
(ARCHITECTURE.md, rule 7 and Section 5.4). A template may reference ``{df}``
for the dataframe variable name and ``{param_name}`` for each of the
function's own parameters; the ``!r`` conversion is used where a parameter
must be rendered as a Python literal (e.g. ``{col!r}`` for a column name).
"""

from __future__ import annotations

from typing import Any


class CodeRenderError(ValueError):
    """A code template could not be rendered from the given params."""


def render_call(code_template: str, params: dict[str, Any], df_var: str | None = "df") -> str:
    """Render ``code_template`` against ``params`` and ``df_var``.

    ``df_var`` is ``None`` for functions that don't operate on a dataset at
    all (e.g. ``adjust_pvalues``, which takes a list of p-values from the
    test ledger, or power-analysis functions) — their code_template has no
    ``{df}`` placeholder and none is injected. A ``"df"`` key in ``params``
    is then just a normal parameter (e.g. power analysis's degrees-of-
    freedom argument), not a collision.

    Raises:
        CodeRenderError: if ``df_var`` is given and ``params`` also contains
            a ``"df"`` key (ambiguous: which one is ``{df}``?), or if the
            template references a placeholder missing from ``params``.
    """
    if df_var is not None and "df" in params:
        raise CodeRenderError("'df' is reserved for df_var and must not appear in params")

    values: dict[str, Any] = {**params} if df_var is None else {"df": df_var, **params}
    try:
        return code_template.format(**values)
    except KeyError as exc:
        missing = exc.args[0]
        raise CodeRenderError(
            f"code_template references '{{{missing}}}' which is not in params or df_var"
        ) from exc
