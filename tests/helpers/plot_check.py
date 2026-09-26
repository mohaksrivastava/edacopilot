"""Rule 7 for plots: the rendered code draws the same picture.

`assert_codegen_matches` compares return values, which does not work for a
`Figure` — two figures drawn from identical data are different objects and
compare unequal. Comparing *pixels* would work but is the wrong test: it
fails on a matplotlib point release, a font substitution or a DPI
difference, none of which mean the plot changed. It is also the kind of
failure nobody can diagnose from the assertion message.

So the comparison is over what the artists were actually given: line
vertices, bar rectangles, scatter offsets, image arrays, and the text that
labels them. That is the content of the plot, it is exactly reproducible,
and a mismatch points at the data path rather than at the renderer.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from edacore.registry import Registry


def _patch_geometry(patch: Any) -> np.ndarray:
    """A patch's position and size in data coordinates.

    `Patch.get_path()` is in unit coordinates for a Rectangle and data
    coordinates for a Polygon, so the transform has to be applied to get
    one comparable answer for both.
    """
    if isinstance(patch, Rectangle):
        return np.asarray(patch.get_bbox().bounds, dtype=float)
    path = patch.get_patch_transform().transform_path(patch.get_path())
    return np.asarray(path.vertices, dtype=float)


def axes_signature(ax: Any) -> dict[str, Any]:
    """Everything an axes was told to draw, as plain arrays."""
    return {
        "lines": [np.asarray(line.get_xydata(), dtype=float) for line in ax.get_lines()],
        "patches": [_patch_geometry(patch) for patch in ax.patches],
        "collections": [
            np.asarray(collection.get_offsets(), dtype=float) for collection in ax.collections
        ],
        "images": [np.asarray(image.get_array(), dtype=float) for image in ax.images],
        "title": ax.get_title(),
        "xlabel": ax.get_xlabel(),
        "ylabel": ax.get_ylabel(),
        "xticklabels": [t.get_text() for t in ax.get_xticklabels()],
        "yticklabels": [t.get_text() for t in ax.get_yticklabels()],
    }


def figure_signature(fig: Figure) -> list[dict[str, Any]]:
    """The whole figure's plotted content, one entry per axes."""
    return [axes_signature(ax) for ax in fig.axes]


def _arrays_equal(a: Any, b: Any, *, where: str) -> None:
    assert type(a) is type(b), f"{where}: {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, np.ndarray):
        assert a.shape == b.shape, f"{where}: shape {a.shape} vs {b.shape}"
        # equal_nan: a masked or absent value is a legitimate thing to plot,
        # and it must compare equal to itself.
        assert np.allclose(a, b, equal_nan=True), f"{where}: values differ"
    elif isinstance(a, list):
        assert len(a) == len(b), f"{where}: {len(a)} vs {len(b)} items"
        for index, (x, y) in enumerate(zip(a, b, strict=True)):
            _arrays_equal(x, y, where=f"{where}[{index}]")
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), f"{where}: keys differ"
        for key in a:
            _arrays_equal(a[key], b[key], where=f"{where}.{key}")
    else:
        assert a == b, f"{where}: {a!r} vs {b!r}"


def assert_figures_match(actual: Figure, expected: Figure, *, context: str = "") -> None:
    """Two figures plot the same data with the same labels."""
    left, right = figure_signature(actual), figure_signature(expected)
    assert len(left) == len(right), f"{context}: {len(left)} axes vs {len(right)}"
    for index, (a, b) in enumerate(zip(left, right, strict=True)):
        _arrays_equal(a, b, where=f"{context or 'figure'}.axes[{index}]")


def assert_plot_codegen_matches(
    registry: Registry,
    name: str,
    params: dict[str, Any],
    df: pd.DataFrame | None = None,
    *,
    df_var: str = "df",
    namespace: dict[str, Any] | None = None,
) -> None:
    """Rule 7 for a registered plot function.

    Executes `registry.to_code(name, params)` and asserts the figure it
    produces plots the same data as calling the function directly. `df` is
    None for a `takes_df=False` plot (`effect_size_forest`).
    """
    spec = registry.get(name)
    code = (
        registry.to_code(name, params, df_var=df_var)
        if spec.takes_df
        else registry.to_code(name, params)
    )

    exec_namespace: dict[str, Any] = dict(namespace or {})
    if spec.takes_df:
        exec_namespace[df_var] = df
    from_code = eval(code, exec_namespace)
    direct = spec.func(df, **params) if spec.takes_df else spec.func(**params)

    assert isinstance(from_code, Figure), f"{name}'s rendered code did not return a Figure"
    try:
        assert_figures_match(from_code, direct, context=f"{name} (code: {code})")
    finally:
        from_code.clear()
        direct.clear()
