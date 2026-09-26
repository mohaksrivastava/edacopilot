"""Visualisation (ARCHITECTURE.md, Section 6.11).

Three properties, one of which is the milestone's acceptance criterion.

**Every function in 6.11 exists and is registered.** Checked against the
specification's own list, so a function that was never written fails here
rather than as an `AttributeError` in a card months later — which is
exactly how 6.11 went missing from the milestone table in the first place.

**Rule 7 holds for plots.** The code each function renders, when executed,
draws the same picture. Compared as *plotted data* — the artists' own
arrays — not as pixels: an image comparison would fail on a matplotlib
point release or a font substitution without anything being wrong, and
nobody can diagnose a pixel diff.

**Nothing mutates its input and nothing leaks.** Rule 6, plus the practical
matter that 25 plot functions called repeatedly would otherwise fill the
kernel with figures nobody closes.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

import edacore
from edacore import viz
from edacore.registry import registry
from edacore.stattests import two_sample
from tests.helpers.plot_check import (
    assert_figures_match,
    assert_plot_codegen_matches,
    figure_signature,
)

# Section 6.11's list, verbatim.
SECTION_6_11 = [
    "histogram",
    "kde",
    "ecdf",
    "boxplot",
    "violin",
    "strip_by_group",
    "qq_plot",
    "scatter_lowess",
    "pair_plot",
    "correlation_heatmap",
    "bar_counts",
    "mosaic",
    "missing_matrix",
    "missing_heatmap",
    "missing_by_group",
    "outlier_plot",
    "before_after",
    "ts_line",
    "stl_plot",
    "acf_plot",
    "pacf_plot",
    "change_point_plot",
    "text_length_hist",
    "top_terms_bar",
    "effect_size_forest",
]


def frame() -> pd.DataFrame:
    """One frame exercising every shape the plots need: numeric, grouped,
    categorical, text, and a column with real missing values."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x": rng.normal(size=60),
            "y": np.concatenate([rng.normal(0, 1, 30), rng.normal(1.5, 2, 30)]),
            "g": ["a"] * 30 + ["b"] * 30,
            "cat": rng.choice(["p", "q", "r"], 60),
            "z": rng.normal(size=60),
            "txt": ["the quick brown fox", "the lazy dog sleeps"] * 30,
        }
    )
    # Two columns with missing values, overlapping but not identical: one
    # column alone would send `missing_heatmap` down its degenerate branch
    # and leave the real one untested.
    df.loc[:9, "x"] = np.nan
    df.loc[5:14, "z"] = np.nan
    return df


def seasonal_frame() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    index = np.arange(120)
    return pd.DataFrame(
        {"v": np.sin(index * 0.5) + index * 0.01 + rng.normal(0, 0.1, 120), "t": index}
    )


def results() -> list[Any]:
    df = frame().dropna(subset=["y"])
    return [two_sample.welch_t(df, "g", "y"), two_sample.mann_whitney(df, "g", "y")]


# (name, params, which frame). One entry per Section 6.11 function, so the
# codegen check below covers all 25 rather than a sample.
CASES: list[tuple[str, dict[str, Any], str]] = [
    ("histogram", {"col": "y", "bins": 12}, "main"),
    ("kde", {"col": "y", "points": 50}, "main"),
    ("ecdf", {"col": "y"}, "main"),
    ("boxplot", {"col": "y", "group": "g"}, "main"),
    ("violin", {"col": "y", "group": "g"}, "main"),
    ("strip_by_group", {"col": "y", "group": "g", "jitter": 0.08, "random_state": 0}, "main"),
    ("qq_plot", {"col": "y", "group": "g"}, "main"),
    ("scatter_lowess", {"x": "x", "y": "y", "frac": 0.6}, "main"),
    ("pair_plot", {"cols": ["x", "y"], "max_cols": 6}, "main"),
    ("correlation_heatmap", {"cols": ["x", "y"], "method": "spearman"}, "main"),
    ("bar_counts", {"col": "cat", "top": 20}, "main"),
    ("mosaic", {"a": "g", "b": "cat"}, "main"),
    ("missing_matrix", {"max_rows": 500}, "main"),
    ("missing_heatmap", {}, "main"),
    ("missing_by_group", {"col": "x", "group": "g"}, "main"),
    ("outlier_plot", {"col": "y", "flagged": None, "whis": 1.5}, "main"),
    ("ts_line", {"col": "v", "time": "t", "entity": None}, "seasonal"),
    ("stl_plot", {"col": "v", "period": 12}, "seasonal"),
    ("acf_plot", {"col": "v", "lags": 10}, "seasonal"),
    ("pacf_plot", {"col": "v", "lags": 10}, "seasonal"),
    ("change_point_plot", {"col": "v", "change_points": [40, 80], "time": "t"}, "seasonal"),
    ("text_length_hist", {"col": "txt", "unit": "characters"}, "main"),
    ("top_terms_bar", {"col": "txt", "top": 10}, "main"),
]


def _frame_for(which: str) -> pd.DataFrame:
    return frame() if which == "main" else seasonal_frame()


# --------------------------------------------------------------------------
# the catalogue
# --------------------------------------------------------------------------


def test_every_function_in_section_6_11_exists() -> None:
    missing = [name for name in SECTION_6_11 if not hasattr(viz, name)]
    assert not missing, f"Section 6.11 lists {missing}, which edacore.viz does not have"


def test_every_function_in_section_6_11_is_registered() -> None:
    """Rule 6: every public edacore function is registered with metadata
    and a code template."""
    registered = {spec.name for spec in registry.list(kind="viz")}
    assert registered == set(SECTION_6_11)


@pytest.mark.parametrize("name", SECTION_6_11)
def test_each_plot_function_is_registered_as_a_read_only_viz(name: str) -> None:
    spec = registry.get(name)
    assert spec.kind == "viz"
    assert spec.read_only, "a plot must not change the data"
    assert spec.code_template, "rule 6 requires a code template"


def test_the_case_table_covers_every_function() -> None:
    """The codegen check below is only worth what its coverage is."""
    covered = {name for name, _, _ in CASES} | {"before_after", "effect_size_forest"}
    assert covered == set(SECTION_6_11)


# --------------------------------------------------------------------------
# rule 7: the rendered code draws the same picture
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "params", "which"), CASES, ids=[c[0] for c in CASES])
def test_generated_code_reproduces_the_plot(name: str, params: dict[str, Any], which: str) -> None:
    assert_plot_codegen_matches(
        registry, name, params, _frame_for(which), namespace={"edacore": edacore}
    )


def test_generated_code_reproduces_before_after() -> None:
    """Two frames, so it does not fit the single-`df` case table."""
    before = frame()
    after = before.assign(y=np.log1p(before["y"].abs()))
    code = registry.to_code("before_after", {"after": "after", "col": "y"}, df_var="before")
    from_code = eval(code, {"edacore": edacore, "before": before, "after": after})
    direct = viz.before_after(before, after, "y")
    assert_figures_match(from_code, direct, context="before_after")


def test_generated_code_reproduces_the_forest_plot() -> None:
    """`takes_df=False`: its input is a list of results, not a frame."""
    computed = results()
    assert_plot_codegen_matches(
        registry,
        "effect_size_forest",
        {"results": computed},
        None,
        namespace={"edacore": edacore, "TestResult": type(computed[0])},
    )


# --------------------------------------------------------------------------
# rule 6, and not leaking
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("name", "params", "which"), CASES, ids=[c[0] for c in CASES])
def test_no_plot_mutates_its_input(name: str, params: dict[str, Any], which: str) -> None:
    df = _frame_for(which)
    before = df.copy()
    registry.get(name).func(df, **params)
    pd.testing.assert_frame_equal(df, before)


@pytest.mark.parametrize(("name", "params", "which"), CASES, ids=[c[0] for c in CASES])
def test_every_plot_returns_a_figure_with_content(
    name: str, params: dict[str, Any], which: str
) -> None:
    fig = registry.get(name).func(_frame_for(which), **params)
    assert isinstance(fig, Figure)
    assert fig.axes, f"{name} returned a figure with no axes"
    drawn = figure_signature(fig)
    assert any(
        axes["lines"] or axes["patches"] or axes["collections"] or axes["images"] for axes in drawn
    ), f"{name} returned an empty figure"


def test_plots_do_not_enter_pyplots_global_registry() -> None:
    """A library that drew through pyplot would leak a figure per call into
    the host kernel and start warning at twenty. These build a bare
    `Figure`, which the caller owns."""
    import matplotlib.pyplot as plt

    plt.close("all")
    before = len(plt.get_fignums())
    for name, params, which in CASES:
        registry.get(name).func(_frame_for(which), **params)
    assert len(plt.get_fignums()) == before


# --------------------------------------------------------------------------
# degenerate input: a card must still get a plot
# --------------------------------------------------------------------------


def test_a_constant_column_does_not_break_the_density_plot() -> None:
    """A KDE of a constant is singular. Losing the whole card to that would
    be worse than a panel that explains it."""
    df = pd.DataFrame({"c": [3.0] * 20})
    fig = viz.kde(df, "c")
    assert "no variation" in fig.axes[0].texts[0].get_text()


def test_a_frame_with_no_missing_values_still_draws_the_heatmap() -> None:
    fig = viz.missing_heatmap(pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}))
    assert "fewer than two columns" in fig.axes[0].texts[0].get_text()


def test_a_forest_plot_of_nothing_says_so() -> None:
    fig = viz.effect_size_forest([])
    assert "no effect sizes" in fig.axes[0].texts[0].get_text()


def test_pair_plot_respects_max_cols() -> None:
    """Section 6.11's cap. It is about legibility and about not inviting a
    multiplicity problem, not about cost."""
    df = pd.DataFrame({f"c{i}": np.arange(10, dtype=float) + i for i in range(10)})
    fig = viz.pair_plot(df, list(df.columns), max_cols=3)
    assert len(fig.axes) == 9


def test_strip_jitter_is_reproducible() -> None:
    """An exported notebook that redrew the plot differently would
    undermine rule 7, so the jitter is seeded."""
    df = frame()
    first = figure_signature(viz.strip_by_group(df, "y", "g"))
    second = figure_signature(viz.strip_by_group(df, "y", "g"))
    for a, b in zip(first, second, strict=True):
        for line_a, line_b in zip(a["lines"], b["lines"], strict=True):
            assert np.array_equal(line_a, line_b)


def test_group_order_does_not_depend_on_row_order() -> None:
    """A plot whose categories reordered under a row shuffle would be
    reporting something about the row order, not about the data."""
    df = frame()
    shuffled = df.sample(frac=1.0, random_state=3)
    assert (
        figure_signature(viz.boxplot(df, "y", "g"))[0]["xticklabels"]
        == figure_signature(viz.boxplot(shuffled, "y", "g"))[0]["xticklabels"]
    )
