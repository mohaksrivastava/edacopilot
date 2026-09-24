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


def render_call(code_template: str, params: dict[str, Any], df_var: str = "df") -> str:
    """Render ``code_template`` against ``params`` and ``df_var``.

    Raises:
        CodeRenderError: if ``params`` contains a ``"df"`` key (reserved for
            ``df_var``), or if the template references a placeholder that is
            missing from ``params``.
    """
    if "df" in params:
        raise CodeRenderError("'df' is reserved for df_var and must not appear in params")

    values: dict[str, Any] = {"df": df_var, **params}
    try:
        return code_template.format(**values)
    except KeyError as exc:
        missing = exc.args[0]
        raise CodeRenderError(
            f"code_template references '{{{missing}}}' which is not in params or df_var"
        ) from exc
