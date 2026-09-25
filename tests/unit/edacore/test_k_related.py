"""edacore.stattests.k_related vs R (tests/fixtures/r_reference/).

repeated_measures_anova matches afex::aov_ez's *default* output exactly:
the primary statistic/df/p_value are Greenhouse-Geisser-corrected (afex's
own default), with the uncorrected values, Huynh-Feldt correction, and
Mauchly's test carried in validity_notes. See k_related.py's module
docstring for the full writeup (ARCHITECTURE.md Section 18, M3 part 2a).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import edacore
from edacore.registry import registry
from tests.helpers.codegen_check import assert_codegen_matches

FIXTURES = Path(__file__).parents[2] / "fixtures" / "r_reference"
_NAMESPACE = {"edacore": edacore}


def _data(name: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "data" / f"{name}.csv")


def _ref(name: str) -> dict[str, Any]:
    with (FIXTURES / f"{name}.json").open() as f:
        result: dict[str, Any] = json.load(f)
        return result


def _close(got: float | None, want: float, *, abs_tol: float = 1e-6) -> None:
    assert got is not None
    assert got == pytest.approx(want, abs=abs_tol, rel=1e-6)


def _binary_long(name: str) -> pd.DataFrame:
    wide = _data(name)
    return (
        wide.reset_index()
        .rename(columns={"index": "subject"})
        .melt(id_vars="subject", var_name="condition", value_name="value")
    )


# --------------------------------------------------------------------------
# repeated_measures_anova
# --------------------------------------------------------------------------


def test_repeated_measures_anova_matches_afex_default() -> None:
    df = _data("repeated_measures_long")
    ref = _ref("repeated_measures_anova__repeated_measures")
    result = edacore.stattests.k_related.repeated_measures_anova(
        df, "value", "subject", "condition"
    )
    _close(result.statistic, ref["statistic"])
    assert result.df is not None
    assert result.df[0] == pytest.approx(ref["df1_gg"])
    assert result.df[1] == pytest.approx(ref["df2_gg"])
    _close(result.p_value, ref["p_value_gg"])
    _close(result.effect_size, ref["pes"])
    assert result.effect_size_name == "partial_eta_squared"
    assert_codegen_matches(
        registry,
        "repeated_measures_anova",
        {
            "dv": "value",
            "subject": "subject",
            "within": "condition",
            "ci": 0.95,
            "nan_policy": "omit",
        },
        df,
        namespace=_NAMESPACE,
    )


def test_repeated_measures_anova_validity_notes_carry_uncorrected_and_hf() -> None:
    df = _data("repeated_measures_long")
    ref = _ref("repeated_measures_anova__repeated_measures")
    result = edacore.stattests.k_related.repeated_measures_anova(
        df, "value", "subject", "condition"
    )
    notes = " ".join(result.validity_notes)
    assert "Mauchly" in notes
    assert "Greenhouse-Geisser" in notes
    assert "Huynh-Feldt" in notes
    assert f"{ref['df1_uncorrected']}" in notes or str(int(ref["df1_uncorrected"])) in notes


# --------------------------------------------------------------------------
# friedman
# --------------------------------------------------------------------------


def test_friedman_matches_r() -> None:
    df = _data("repeated_measures_long")
    ref = _ref("friedman__repeated_measures")
    result = edacore.stattests.k_related.friedman(df, "value", "subject", "condition")
    _close(result.statistic, ref["statistic"])
    assert result.df == pytest.approx(ref["df"])
    _close(result.p_value, ref["p_value"])
    assert result.effect_size_name == "kendalls_w"
    assert_codegen_matches(
        registry,
        "friedman",
        {"dv": "value", "subject": "subject", "within": "condition", "nan_policy": "omit"},
        df,
        namespace=_NAMESPACE,
    )


# --------------------------------------------------------------------------
# cochran_q
# --------------------------------------------------------------------------


def test_cochran_q_matches_r() -> None:
    df = _binary_long("cochran_q_binary")
    ref = _ref("cochran_q__cochran_q_binary")
    result = edacore.stattests.k_related.cochran_q(df, "value", "subject", "condition")
    _close(result.statistic, ref["statistic"])
    assert result.df == pytest.approx(ref["df"])
    _close(result.p_value, ref["p_value"])
    assert_codegen_matches(
        registry,
        "cochran_q",
        {"dv": "value", "subject": "subject", "within": "condition", "nan_policy": "omit"},
        df,
        namespace=_NAMESPACE,
    )


# --------------------------------------------------------------------------
# balanced-design handling
# --------------------------------------------------------------------------


def test_friedman_drops_subjects_incomplete_across_conditions() -> None:
    # kendalls_w's BCa bootstrap needs more than a couple of subjects to
    # be well-defined; use enough complete subjects that dropping the one
    # incomplete subject still leaves a workable sample.
    rows = []
    for subj in range(1, 6):
        for cond, val in zip(["a", "b", "c"], [1.0, 2.0, 3.0], strict=True):
            if subj == 5 and cond == "c":
                continue  # subject 5 missing condition c
            rows.append({"subject": subj, "condition": cond, "value": val + subj})
    df = pd.DataFrame(rows)

    result = edacore.stattests.k_related.friedman(df, "value", "subject", "condition")
    assert result.n == {"subject": 4}
    assert any("incomplete" in w for w in result.warnings)


# --------------------------------------------------------------------------
# M3.4: repeated_measures_anova's partial eta-squared CI
# --------------------------------------------------------------------------


def test_repeated_measures_anova_partial_eta_squared_ci_matches_r() -> None:
    """effectsize::eta_squared on the afex fit uses the UNCORRECTED F and
    df: the sphericity correction changes the test's df, not the effect
    size. Checked on a dataset with a real effect, so the lower bound is
    strictly positive."""
    df = _data("repeated_measures_effect_long")
    ref = _ref("repeated_measures_anova__repeated_measures_effect")
    result = edacore.stattests.k_related.repeated_measures_anova(
        df, "value", "subject", "condition"
    )
    _close(result.statistic, ref["statistic"])
    _close(result.p_value, ref["p_value_gg"])
    assert result.effect_size_name == "partial_eta_squared"
    _close(result.effect_size, ref["pes"])
    assert result.effect_size_ci is not None
    _close(result.effect_size_ci[0], ref["ci_low"])
    _close(result.effect_size_ci[1], ref["ci_high"])
    assert result.effect_size_ci[0] > 0
    # The reported df ARE the GG-corrected ones, so the CI is not simply
    # reading them back off the result.
    assert result.df == (pytest.approx(ref["df1_gg"]), pytest.approx(ref["df2_gg"]))
    assert ref["df1_gg"] != pytest.approx(ref["df1_uncorrected"])


def test_repeated_measures_anova_partial_eta_squared_ci_on_the_null_dataset() -> None:
    df = _data("repeated_measures_long")
    ref = _ref("partial_eta_squared_rm__repeated_measures")
    result = edacore.stattests.k_related.repeated_measures_anova(
        df, "value", "subject", "condition"
    )
    _close(result.effect_size, ref["estimate"])
    assert result.effect_size_ci == (ref["ci_low"], ref["ci_high"])
    # effectsize::eta_squared(afex fit) == effectsize::F_to_eta2 at the
    # uncorrected df -- the identity this implementation relies on.
    assert ref["estimate"] == pytest.approx(ref["estimate_from_f"])
    assert ref["ci_low"] == pytest.approx(ref["ci_low_from_f"])
