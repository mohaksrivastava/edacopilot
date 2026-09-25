"""M3's acceptance criterion, enforced rather than audited.

ARCHITECTURE.md Section 16 requires that every Section 6.7 hypothesis test
"returns effect size + CI", and Section 6.7 repeats it ("All return
`TestResult` including effect size with CI"). The m3.3 audit
(docs/m3_effect_size_audit.md) found 9 of 45 test functions actually did;
m3.4 fixed the other 27 and exempted 9 with reasons.

This file calls every registered `kind="test"` function in stage
`hypothesis` on data suited to its design and asserts the criterion holds,
so a new test function (or a regression in an existing one) fails CI
instead of silently reopening the gap. `_EXEMPT` is the list of accepted
exemptions from the audit; adding a name to it is a deliberate,
reviewable act, and the test also fails if an exempt function starts
returning an effect size (the exemption would then be stale).

The numbers themselves are checked against R in each module's own test
file; what this file checks is coverage, shape, and self-consistency.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import edacore
from edacore.contracts import TestResult
from edacore.effect_sizes import _MAGNITUDE_THRESHOLDS, UNLABELLED_MEASURES
from edacore.registry import registry

# Accepted exemptions (docs/m3_effect_size_audit.md, confirmed for m3.4).
# Each has no standard effect size, or no standard CI for the one it has.
_EXEMPT: dict[str, str] = {
    "ks_two_sample": "D is the effect size; no standard CI exists for it",
    "alexander_govern": "Section 6.7 lists '-' for this test's effect size",
    "cochran_q": "Section 6.7 lists '-'",
    "cochran_armitage_trend": "Section 6.7 lists '-'",
    "anderson_ksamp": "distribution-equality test; no standard effect size",
    "runs_test": "randomness test; no standard effect size",
    "distance_correlation": "no analytic CI; a bootstrap of dCor is biased near 0",
    "mutual_information": "no standard CI for the KSG estimator",
    "bootstrap_one_sample": "Section 6.7 specifies 'CI only' -- the estimate IS the result",
    "bootstrap_diff": "Section 6.7 specifies 'CI of diff'",
}

# fisher_exact is exempt for r x c tables only (no odds ratio beyond 2x2)
# and is therefore called below on a 2x2 table, where it is NOT exempt.


def _normal(n: int, mean: float, sd: float, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(mean, sd, n)


def _two_groups() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "value": np.concatenate([_normal(40, 100, 15, 1), _normal(40, 108, 12, 2)]),
            "group": ["A"] * 40 + ["B"] * 40,
        }
    )


def _k_groups() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "value": np.concatenate(
                [_normal(25, 50, 10, 3), _normal(25, 58, 12, 4), _normal(25, 66, 9, 5)]
            ),
            "group": ["A"] * 25 + ["B"] * 25 + ["C"] * 25,
        }
    )


def _paired() -> pd.DataFrame:
    before = _normal(30, 70, 12, 6)
    return pd.DataFrame({"before": before, "after": before + _normal(30, 5, 6, 7)})


def _repeated() -> pd.DataFrame:
    base = _normal(20, 50, 8, 8)
    wide = pd.DataFrame(
        {
            "subject": np.arange(20),
            "t1": base + _normal(20, 0, 3, 9),
            "t2": base + _normal(20, 6, 3, 10),
            "t3": base + _normal(20, 11, 3, 11),
        }
    )
    return wide.melt(id_vars="subject", var_name="condition", value_name="value")


def _repeated_binary() -> pd.DataFrame:
    rng = np.random.default_rng(12)
    wide = pd.DataFrame(
        {
            "subject": np.arange(25),
            "t1": rng.binomial(1, 0.4, 25),
            "t2": rng.binomial(1, 0.7, 25),
            "t3": rng.binomial(1, 0.5, 25),
        }
    )
    return wide.melt(id_vars="subject", var_name="condition", value_name="value")


def _factorial() -> pd.DataFrame:
    sizes = [8, 5, 10, 6]
    means = [10.0, 12.0, 9.0, 14.0]
    return pd.DataFrame(
        {
            "value": np.concatenate(
                [
                    _normal(n, m, 2, 13 + i)
                    for i, (n, m) in enumerate(zip(sizes, means, strict=True))
                ]
            ),
            "factor_a": np.repeat(["A1", "A1", "A2", "A2"], sizes),
            "factor_b": np.repeat(["B1", "B2", "B1", "B2"], sizes),
        }
    )


def _table(counts: dict[tuple[str, str], int]) -> pd.DataFrame:
    rows = [{"a": a, "b": b} for (a, b), k in counts.items() for _ in range(k)]
    return pd.DataFrame(rows)


def _two_by_two() -> pd.DataFrame:
    return _table(
        {
            ("exposed", "yes"): 30,
            ("exposed", "no"): 20,
            ("unexposed", "yes"): 15,
            ("unexposed", "no"): 35,
        }
    )


def _paired_binary() -> pd.DataFrame:
    rng = np.random.default_rng(14)
    before = rng.binomial(1, 0.5, 60)
    after = np.where(rng.random(60) < 0.3, 1 - before, before)
    return pd.DataFrame({"before": before, "after": after})


def _numeric_pair() -> pd.DataFrame:
    x = _normal(40, 0, 1, 15)
    return pd.DataFrame({"x": x, "y": 0.6 * x + _normal(40, 0, 1, 16), "z": _normal(40, 0, 1, 17)})


_TWO = _two_groups()
_K = _k_groups()
_PAIR = _paired()
_RM = _repeated()
_RMB = _repeated_binary()
_FAC = _factorial()
_2X2 = _two_by_two()
_PB = _paired_binary()
_NUM = _numeric_pair()
_ONE = pd.DataFrame({"x": _normal(50, 50, 10, 18), "binary": [1] * 20 + [0] * 30})
_CAT = pd.DataFrame({"c": ["A"] * 18 + ["B"] * 25 + ["C"] * 30 + ["D"] * 12})

# Every registered hypothesis test, with arguments matching its design.
_CALLS: dict[str, tuple[pd.DataFrame, dict[str, Any]]] = {
    "one_sample_t": (_ONE, {"col": "x", "mu0": 48.0}),
    "wilcoxon_one_sample": (_ONE, {"col": "x", "mu0": 48.0}),
    "sign_test": (_ONE, {"col": "x", "mu0": 48.0}),
    "binomial_test": (_ONE, {"col": "binary", "p0": 0.5}),
    "chi2_goodness_of_fit": (
        _CAT,
        {"col": "c", "expected": {"A": 0.2, "B": 0.3, "C": 0.3, "D": 0.2}},
    ),
    "bootstrap_one_sample": (_ONE, {"col": "x", "n_boot": 200}),
    "student_t": (_TWO, {"group_col": "group", "value_col": "value"}),
    "welch_t": (_TWO, {"group_col": "group", "value_col": "value"}),
    "yuen_trimmed_t": (_TWO, {"group_col": "group", "value_col": "value"}),
    "mann_whitney": (_TWO, {"group_col": "group", "value_col": "value"}),
    "brunner_munzel": (_TWO, {"group_col": "group", "value_col": "value"}),
    "permutation_test_2s": (
        _TWO,
        {"group_col": "group", "value_col": "value", "n_resamples": 200, "n_boot": 200},
    ),
    "bootstrap_diff": (_TWO, {"group_col": "group", "value_col": "value", "n_boot": 200}),
    "ks_two_sample": (_TWO, {"group_col": "group", "value_col": "value"}),
    "paired_t": (_PAIR, {"a": "after", "b": "before"}),
    "wilcoxon_signed_rank": (_PAIR, {"a": "after", "b": "before"}),
    "sign_test_paired": (_PAIR, {"a": "after", "b": "before"}),
    "permutation_test_paired": (
        _PAIR,
        {"a": "after", "b": "before", "n_resamples": 200, "n_boot": 200},
    ),
    "one_way_anova": (_K, {"outcome": "value", "group": "group"}),
    "welch_anova": (_K, {"outcome": "value", "group": "group"}),
    "alexander_govern": (_K, {"outcome": "value", "group": "group"}),
    "kruskal_wallis": (_K, {"outcome": "value", "group": "group"}),
    "permutation_anova": (_K, {"outcome": "value", "group": "group", "n_resamples": 200}),
    "repeated_measures_anova": (_RM, {"dv": "value", "subject": "subject", "within": "condition"}),
    "friedman": (_RM, {"dv": "value", "subject": "subject", "within": "condition"}),
    "cochran_q": (_RMB, {"dv": "value", "subject": "subject", "within": "condition"}),
    "two_way_anova": (_FAC, {"outcome": "value", "factors": ["factor_a", "factor_b"]}),
    "aligned_rank_transform_anova": (
        _FAC,
        {"outcome": "value", "factors": ["factor_a", "factor_b"]},
    ),
    "chi2_independence": (_2X2, {"a": "a", "b": "b"}),
    "fisher_exact": (_2X2, {"a": "a", "b": "b"}),
    "g_test": (_2X2, {"a": "a", "b": "b"}),
    "mcnemar": (_PB, {"a": "before", "b": "after"}),
    "two_proportion_z": (
        _2X2,
        {"outcome": "b", "group": "a", "groups": ("exposed", "unexposed"), "event": "yes"},
    ),
    "cochran_armitage_trend": (
        _table(
            {
                ("d0", "yes"): 45,
                ("d0", "no"): 5,
                ("d1", "yes"): 38,
                ("d1", "no"): 12,
                ("d2", "yes"): 28,
                ("d2", "no"): 22,
                ("d3", "yes"): 15,
                ("d3", "no"): 35,
            }
        ),
        {"binary": "b", "ordinal": "a", "event": "yes"},
    ),
    "pearson": (_NUM, {"x": "x", "y": "y"}),
    "spearman": (_NUM, {"x": "x", "y": "y"}),
    "kendall_tau": (_NUM, {"x": "x", "y": "y"}),
    "point_biserial": (_ONE, {"binary": "binary", "x": "x"}),
    "partial_correlation": (_NUM, {"x": "x", "y": "y", "covars": ["z"]}),
    "distance_correlation": (_NUM, {"x": "x", "y": "y", "n_resamples": 100}),
    "mutual_information": (_NUM, {"x": "x", "y": "y"}),
    "correlation_matrix": (_NUM, {"cols": ["x", "y", "z"]}),
    "anderson_ksamp": (_K, {"outcome": "value", "group": "group"}),
    "tost_equivalence": (_TWO, {"outcome": "value", "group": "group", "low": -20.0, "high": 20.0}),
    "runs_test": (_ONE, {"col": "x"}),
}


def _registered_tests() -> list[str]:
    return sorted(spec.name for spec in registry.list(stage="hypothesis", kind="test"))


def _flatten(result: Any) -> list[TestResult]:
    return list(result) if isinstance(result, list) else [result]


def test_every_registered_hypothesis_test_is_exercised_here() -> None:
    """If a new test function is added to Section 6.7 without a row in
    `_CALLS`, this fails -- the criterion below cannot silently skip it."""
    assert set(_registered_tests()) == set(_CALLS), {
        "missing_from_calls": sorted(set(_registered_tests()) - set(_CALLS)),
        "not_registered": sorted(set(_CALLS) - set(_registered_tests())),
    }


@pytest.mark.parametrize("name", _registered_tests())
def test_effect_size_and_ci_present_or_exempt(name: str) -> None:
    df, params = _CALLS[name]
    results = _flatten(registry.get(name).func(df, **params))
    assert results, f"{name} returned no TestResult"

    for result in results:
        if name in _EXEMPT:
            assert result.effect_size is None, (
                f"{name} is listed as exempt ({_EXEMPT[name]}) but now returns an effect size "
                f"({result.effect_size_name}); remove it from _EXEMPT and from "
                f"docs/m3_effect_size_audit.md"
            )
            continue
        assert result.effect_size is not None, f"{name} returns no effect size"
        assert result.effect_size_name is not None, f"{name}'s effect size is unnamed"
        assert result.effect_size_ci is not None, (
            f"{name} returns {result.effect_size_name} with no CI -- M3's acceptance "
            f"criterion (ARCHITECTURE.md Section 16) requires effect size AND CI"
        )
        low, high = result.effect_size_ci
        assert low <= high, f"{name}: inverted CI {result.effect_size_ci}"
        assert low <= result.effect_size <= high, (
            f"{name}: {result.effect_size_name}={result.effect_size} lies outside its own "
            f"CI {result.effect_size_ci}"
        )


@pytest.mark.parametrize("name", _registered_tests())
def test_effect_magnitude_is_set_exactly_when_thresholds_exist(name: str) -> None:
    """A magnitude label is only meaningful for a standardized measure.
    Every reported effect_size_name must either have thresholds (and then
    carry a label) or be declared unlabelled -- never silently neither."""
    df, params = _CALLS[name]
    for result in _flatten(registry.get(name).func(df, **params)):
        measure = result.effect_size_name
        if measure is None:
            continue
        assert measure in _MAGNITUDE_THRESHOLDS or measure in UNLABELLED_MEASURES, (
            f"{name} reports effect size '{measure}', which is neither in "
            f"_MAGNITUDE_THRESHOLDS nor declared in UNLABELLED_MEASURES"
        )
        if measure in UNLABELLED_MEASURES:
            assert result.effect_magnitude is None, (
                f"{name} reports '{measure}' (declared unlabelled) with magnitude "
                f"'{result.effect_magnitude}'"
            )
        else:
            assert result.effect_magnitude is not None, (
                f"{name} reports '{measure}', which has thresholds, but no effect_magnitude"
            )


def test_exemption_list_matches_the_audit_document() -> None:
    """The audit is the reviewed artefact; this keeps code and document from
    drifting apart."""
    audit = edacore.__file__.rsplit("src", 1)[0] + "docs/m3_effect_size_audit.md"
    with open(audit, encoding="utf-8") as f:
        text = f.read()
    for name, reason in _EXEMPT.items():
        assert f"`{name}`" in text, f"{name} is exempt in code but absent from the audit"
        assert reason, f"{name} has an empty exemption reason"
