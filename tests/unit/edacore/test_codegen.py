from __future__ import annotations

import pytest

from edacore.codegen import CodeRenderError, render_call


def test_render_call_substitutes_params_and_df_var() -> None:
    code = render_call("{df}[{col!r}].mean()", {"col": "income"}, df_var="sales")
    assert code == "sales['income'].mean()"


def test_render_call_defaults_df_var_to_df() -> None:
    code = render_call("{df}.shape", {})
    assert code == "df.shape"


def test_render_call_missing_param_raises() -> None:
    with pytest.raises(CodeRenderError):
        render_call("{df}[{col!r}].mean()", {})


def test_render_call_rejects_df_in_params() -> None:
    with pytest.raises(CodeRenderError):
        render_call("{df}.head()", {"df": "oops"})
