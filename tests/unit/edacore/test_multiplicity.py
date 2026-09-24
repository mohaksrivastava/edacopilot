"""edacore.multiplicity vs R's stats::p.adjust
(tests/fixtures/r_reference/adjust_pvalues__raw_pvalues.json).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

import edacore
from edacore.registry import registry
from tests.helpers.codegen_check import assert_codegen_matches_no_df

FIXTURES = Path(__file__).parents[2] / "fixtures" / "r_reference"


def _load_raw_pvalues() -> list[float]:
    with (FIXTURES / "data" / "raw_pvalues.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        return [float(row["p"]) for row in reader]


def _load_reference() -> dict[str, object]:
    with (FIXTURES / "adjust_pvalues__raw_pvalues.json").open() as f:
        result: dict[str, object] = json.load(f)
        return result


@pytest.mark.parametrize("method", ["holm", "bonferroni", "fdr_bh", "fdr_by"])
def test_adjust_pvalues_matches_r(method: str) -> None:
    pvals = _load_raw_pvalues()
    reference = _load_reference()

    result = edacore.multiplicity.adjust_pvalues(pvals, method=method)  # type: ignore[arg-type]

    expected = reference[method]
    assert isinstance(expected, list)
    for got, want in zip(result, expected, strict=True):
        assert got == pytest.approx(want, abs=1e-6)


def test_adjust_pvalues_passes_codegen_check() -> None:
    pvals = _load_raw_pvalues()
    assert_codegen_matches_no_df(
        registry,
        "adjust_pvalues",
        {"pvals": pvals, "method": "holm"},
        namespace={"edacore": edacore},
    )


def test_holm_is_never_more_conservative_than_bonferroni() -> None:
    """Sanity check on ordering: Holm is uniformly at least as powerful as
    Bonferroni, so its adjusted p-values are never larger."""
    pvals = _load_raw_pvalues()
    holm = edacore.multiplicity.adjust_pvalues(pvals, method="holm")
    bonferroni = edacore.multiplicity.adjust_pvalues(pvals, method="bonferroni")
    assert all(h <= b + 1e-12 for h, b in zip(holm, bonferroni, strict=True))
