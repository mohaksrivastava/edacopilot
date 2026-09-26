"""Visualisation (ARCHITECTURE.md, Section 6.11).

Every function here returns a bare `matplotlib.figure.Figure` and nothing
else. Two consequences worth stating, because both are deliberate:

**No pyplot.** These construct `Figure()` directly rather than going
through `plt.subplots()`. `pyplot` keeps a global registry of every figure
it creates and never releases one on its own, so a library that used it
would leak a figure per call into the host kernel and start warning at
twenty. A bare `Figure` is owned by its caller and collected when dropped.
It also means nothing here depends on a backend being chosen, which is
what makes these safe to call from a headless script.

**No session.** Section 6.11 says each function returns "a Figure and a
`plot_ref` ID stored in the session", but `edacore` has no session and must
not grow one (Section 3.1: it is usable on its own as a plain statistics
library). So the split is: `edacore.viz` draws, and
`edacopilot.session.PlotStore` writes the PNG and mints the ref. A user
calling `edacore` directly gets a Figure and does what they like with it.

Rule 6 applies as everywhere else: no function mutates its input frame, and
every one is registered with a code template that reproduces the plot.
`tests/helpers/plot_check.py` verifies that by executing the rendered code
and comparing the *plotted data* — the artists' own arrays — rather than
pixels, which would fail on a font or a version bump without anything
being wrong.

Three signatures depart from Section 6.11's list, each because the
milestone that owns the *detection* has not happened yet. In every case the
function takes the thing it would have been given, so the plot exists now
and the detector plugs in later without a signature change:

- `outlier_plot(df, col, flagged=None)` — M11's detectors pass indices;
  the default marks Tukey-fence points so the plot is useful today.
- `change_point_plot(df, col, change_points, time=None)` — M12 passes the
  detected points.
- `before_after(before, after, col, record=None)` — Section 6.11 writes
  `before_after(col, record)`, but a `TransformRecord` carries summaries,
  not the data, so the two frames have to be passed. The record is still
  accepted, and used for the annotation when given.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy import stats
from statsmodels.graphics.mosaicplot import mosaic as sm_mosaic
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import STL

from edacore.contracts import TestResult, TransformRecord
from edacore.registry import register

# One size for every single-panel plot, so a notebook of cards does not
# jump about. Multi-panel figures scale from this.
FIGSIZE: tuple[float, float] = (6.0, 4.0)

# Section 13.2's status palette, kept here so a plot drawn into a card
# matches the card's own colour-coding rather than approximating it.
STATUS_COLOURS = {
    "pass": "#2e7d32",
    "borderline": "#ef6c00",
    "fail": "#c62828",
    "untestable": "#757575",
}

_PRIMARY = "#1f77b4"
_SECONDARY = "#ff7f0e"
_GRID = {"alpha": 0.3, "linewidth": 0.6}


def _figure(
    nrows: int = 1, ncols: int = 1, figsize: tuple[float, float] | None = None
) -> tuple[Figure, Any]:
    """A figure and its axes, without touching pyplot's global registry."""
    width, height = figsize or FIGSIZE
    fig = Figure(figsize=(width * ncols, height * nrows), layout="constrained")
    axes = fig.subplots(nrows, ncols, squeeze=False)
    return fig, axes


def _numeric(df: pd.DataFrame, col: str) -> np.ndarray:
    values: np.ndarray = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
    finite: np.ndarray = values[~np.isnan(values)]
    return finite


def _groups(df: pd.DataFrame, col: str, group: str) -> tuple[list[str], list[np.ndarray]]:
    """Group labels and their finite values, in a fixed (sorted) order.

    Sorted rather than first-seen: a plot whose category order depended on
    row order would change under an operation that reorders rows without
    changing anything the plot is about.
    """
    frame = df[[col, group]].dropna()
    labels = sorted(frame[group].astype(str).unique())
    values = [
        pd.to_numeric(frame.loc[frame[group].astype(str) == label, col], errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
        for label in labels
    ]
    return labels, values


def _finish(ax: Any, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, **_GRID)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------
# Univariate distribution
# --------------------------------------------------------------------------


@register(
    name="histogram",
    kind="viz",
    stage="explore",
    code_template="edacore.viz.histogram({df}, col={col!r}, bins={bins!r})",
)
def histogram(df: pd.DataFrame, col: str, bins: int | str = "auto") -> Figure:
    """Distribution of one numeric column."""
    fig, axes = _figure()
    ax = axes[0][0]
    ax.hist(_numeric(df, col), bins=bins, color=_PRIMARY, edgecolor="white")
    _finish(ax, f"Distribution of {col}", col, "count")
    return fig


@register(
    name="kde",
    kind="viz",
    stage="explore",
    code_template="edacore.viz.kde({df}, col={col!r}, points={points!r})",
)
def kde(df: pd.DataFrame, col: str, points: int = 200) -> Figure:
    """Gaussian kernel density estimate.

    Degenerate input (fewer than two distinct values) makes the KDE
    singular; the panel says so rather than raising, because a card that
    lost its whole plot to one constant column would be worse than a card
    that explains it.
    """
    values = _numeric(df, col)
    fig, axes = _figure()
    ax = axes[0][0]
    if len(np.unique(values)) < 2:
        ax.text(0.5, 0.5, f"{col} has no variation to smooth", ha="center", va="center")
    else:
        grid = np.linspace(values.min(), values.max(), points)
        ax.plot(grid, stats.gaussian_kde(values)(grid), color=_PRIMARY)
    _finish(ax, f"Density of {col}", col, "density")
    return fig


@register(
    name="ecdf",
    kind="viz",
    stage="explore",
    code_template="edacore.viz.ecdf({df}, col={col!r})",
)
def ecdf(df: pd.DataFrame, col: str) -> Figure:
    """Empirical cumulative distribution: every value, no binning choice."""
    values = np.sort(_numeric(df, col))
    fig, axes = _figure()
    ax = axes[0][0]
    ax.step(
        values, np.arange(1, len(values) + 1) / max(len(values), 1), where="post", color=_PRIMARY
    )
    _finish(ax, f"ECDF of {col}", col, "proportion <= x")
    return fig


@register(
    name="qq_plot",
    kind="viz",
    stage="hypothesis",
    code_template="edacore.viz.qq_plot({df}, col={col!r}, group={group!r})",
)
def qq_plot(df: pd.DataFrame, col: str, group: str | None = None) -> Figure:
    """Normal Q-Q plot, the visual half of a normality check (Section 6.6).

    One panel per group when `group` is given, because normality is a
    within-group assumption: a combined plot can look straight while both
    groups are skewed in opposite directions.
    """
    if group is None:
        labels, samples = [col], [_numeric(df, col)]
    else:
        labels, samples = _groups(df, col, group)

    fig, axes = _figure(1, max(len(samples), 1), figsize=(4.2, 4.0))
    for ax, label, values in zip(axes[0], labels, samples, strict=True):
        if len(values) < 2:
            ax.text(0.5, 0.5, "too few values", ha="center", va="center")
        else:
            theoretical = stats.norm.ppf((np.arange(1, len(values) + 1) - 0.5) / len(values))
            observed = np.sort(values)
            ax.plot(theoretical, observed, "o", color=_PRIMARY, markersize=4)
            slope, intercept = np.polyfit(theoretical, observed, 1)
            ax.plot(theoretical, slope * theoretical + intercept, color=_SECONDARY, linewidth=1)
        _finish(ax, f"Q-Q: {label}", "theoretical quantiles", "observed")
    return fig


# --------------------------------------------------------------------------
# Distribution by group
# --------------------------------------------------------------------------


@register(
    name="boxplot",
    kind="viz",
    stage="explore",
    code_template="edacore.viz.boxplot({df}, col={col!r}, group={group!r})",
)
def boxplot(df: pd.DataFrame, col: str, group: str | None = None) -> Figure:
    """Box plot, the visual half of an equal-variance check (Section 6.6)."""
    labels, samples = ([col], [_numeric(df, col)]) if group is None else _groups(df, col, group)
    fig, axes = _figure()
    ax = axes[0][0]
    ax.boxplot(samples, tick_labels=labels)
    _finish(ax, f"{col} by {group}" if group else f"{col}", group or "", col)
    return fig


@register(
    name="violin",
    kind="viz",
    stage="explore",
    code_template="edacore.viz.violin({df}, col={col!r}, group={group!r})",
)
def violin(df: pd.DataFrame, col: str, group: str | None = None) -> Figure:
    """Violin plot: shape as well as spread, which is what decides whether
    a rank test reads as a difference in medians (Section 6.12)."""
    labels, samples = ([col], [_numeric(df, col)]) if group is None else _groups(df, col, group)
    usable = [s for s in samples if len(np.unique(s)) > 1]
    fig, axes = _figure()
    ax = axes[0][0]
    if usable:
        ax.violinplot(samples, showmedians=True)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    _finish(ax, f"{col} by {group}" if group else f"{col}", group or "", col)
    return fig


@register(
    name="strip_by_group",
    kind="viz",
    stage="explore",
    code_template=(
        "edacore.viz.strip_by_group({df}, col={col!r}, group={group!r}, jitter={jitter!r}, "
        "random_state={random_state!r})"
    ),
)
def strip_by_group(
    df: pd.DataFrame,
    col: str,
    group: str,
    jitter: float = 0.08,
    random_state: int = 0,
) -> Figure:
    """Every observation, jittered. The plot that shows n as well as spread,
    which a box plot hides — and small n is what makes a large effect
    preliminary (Section 6.12's `large_effect_small_n`).

    The jitter is seeded, so the same data draws the same picture; an
    exported notebook that redrew it differently would undermine rule 7.
    """
    labels, samples = _groups(df, col, group)
    rng = np.random.default_rng(random_state)
    fig, axes = _figure()
    ax = axes[0][0]
    for position, values in enumerate(samples, start=1):
        offsets = rng.uniform(-jitter, jitter, size=len(values))
        ax.plot(position + offsets, values, "o", color=_PRIMARY, markersize=4, alpha=0.7)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    _finish(ax, f"{col} by {group} (n = {', '.join(str(len(s)) for s in samples)})", group, col)
    return fig


# --------------------------------------------------------------------------
# Bivariate
# --------------------------------------------------------------------------


@register(
    name="scatter_lowess",
    kind="viz",
    stage="explore",
    code_template="edacore.viz.scatter_lowess({df}, x={x!r}, y={y!r}, frac={frac!r})",
)
def scatter_lowess(df: pd.DataFrame, x: str, y: str, frac: float = 0.6) -> Figure:
    """Scatter with a LOWESS curve: the check on whether Pearson's
    straight-line assumption is the right description (Section 6.6)."""
    frame = df[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
    xs, ys = frame[x].to_numpy(dtype=float), frame[y].to_numpy(dtype=float)
    fig, axes = _figure()
    ax = axes[0][0]
    ax.plot(xs, ys, "o", color=_PRIMARY, markersize=4, alpha=0.7)
    if len(xs) >= 3 and len(np.unique(xs)) > 1:
        smoothed = lowess(ys, xs, frac=frac, return_sorted=True)
        ax.plot(smoothed[:, 0], smoothed[:, 1], color=_SECONDARY, linewidth=2)
    _finish(ax, f"{y} vs {x}", x, y)
    return fig


@register(
    name="pair_plot",
    kind="viz",
    stage="explore",
    code_template="edacore.viz.pair_plot({df}, cols={cols!r}, max_cols={max_cols!r})",
)
def pair_plot(df: pd.DataFrame, cols: Sequence[str], max_cols: int = 6) -> Figure:
    """Scatter matrix, capped at `max_cols` columns.

    The cap is Section 6.11's, and it is about legibility rather than
    cost: a 20x20 grid of 40-pixel panels shows nothing, and the number of
    pairwise comparisons it invites is its own multiplicity problem.
    """
    chosen = list(cols)[:max_cols]
    n = len(chosen)
    fig, axes = _figure(n, n, figsize=(2.0, 2.0))
    frame = df[chosen].apply(pd.to_numeric, errors="coerce")
    for row, y in enumerate(chosen):
        for column, x in enumerate(chosen):
            ax = axes[row][column]
            pair = frame[[x, y]].dropna()
            if row == column:
                ax.hist(pair[x].to_numpy(dtype=float), bins="auto", color=_PRIMARY)
            else:
                ax.plot(
                    pair[x].to_numpy(dtype=float),
                    pair[y].to_numpy(dtype=float),
                    "o",
                    color=_PRIMARY,
                    markersize=2,
                    alpha=0.6,
                )
            if row == n - 1:
                ax.set_xlabel(x)
            if column == 0:
                ax.set_ylabel(y)
    fig.suptitle(f"Pairwise relationships ({n} columns)")
    return fig


@register(
    name="correlation_heatmap",
    kind="viz",
    stage="explore",
    code_template=("edacore.viz.correlation_heatmap({df}, cols={cols!r}, method={method!r})"),
)
def correlation_heatmap(
    df: pd.DataFrame, cols: Sequence[str] | None = None, method: str = "spearman"
) -> Figure:
    """Pairwise correlations as a matrix.

    Spearman by default, not Pearson: a heatmap is a scan for anything
    worth looking at, and a monotonic-but-curved relationship that Pearson
    reads as near zero is precisely what should not be missed at the
    scanning stage.
    """
    frame = df[list(cols)] if cols is not None else df.select_dtypes("number")
    matrix = frame.corr(method=method)  # type: ignore[arg-type]
    fig, axes = _figure(figsize=(5.5, 5.0))
    ax = axes[0][0]
    image = ax.imshow(matrix.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_title(f"{method.title()} correlations")
    return fig


@register(
    name="bar_counts",
    kind="viz",
    stage="explore",
    code_template="edacore.viz.bar_counts({df}, col={col!r}, top={top!r})",
)
def bar_counts(df: pd.DataFrame, col: str, top: int = 20) -> Figure:
    """Category frequencies, most common first, capped at `top`."""
    counts = df[col].astype(str).value_counts().head(top)
    fig, axes = _figure()
    ax = axes[0][0]
    ax.bar(range(len(counts)), counts.to_numpy(dtype=float), color=_PRIMARY)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=45, ha="right")
    _finish(ax, f"Counts of {col}", col, "count")
    return fig


@register(
    name="mosaic",
    kind="viz",
    stage="explore",
    code_template="edacore.viz.mosaic({df}, a={a!r}, b={b!r})",
)
def mosaic(df: pd.DataFrame, a: str, b: str) -> Figure:
    """Mosaic plot of a contingency table: tile area is cell frequency, so
    an association shows as tiles that fail to line up."""
    frame = df[[a, b]].dropna().astype(str)
    fig, axes = _figure(figsize=(5.5, 4.5))
    ax = axes[0][0]
    sm_mosaic(frame, index=[a, b], ax=ax, gap=0.02, title="")
    ax.set_title(f"{a} by {b}")
    return fig


# --------------------------------------------------------------------------
# Missingness (Section 6.3's plots; the analysis is M10)
# --------------------------------------------------------------------------


@register(
    name="missing_matrix",
    kind="viz",
    stage="missingness",
    code_template="edacore.viz.missing_matrix({df}, max_rows={max_rows!r})",
)
def missing_matrix(df: pd.DataFrame, max_rows: int = 500) -> Figure:
    """Row-by-column map of what is missing, in row order.

    Row order is kept deliberately: missingness that arrives in blocks is
    a different mechanism from missingness scattered at random, and that
    is visible here and nowhere else.
    """
    frame = df.head(max_rows)
    fig, axes = _figure(figsize=(6.5, 4.0))
    ax = axes[0][0]
    ax.imshow(frame.isna().to_numpy(dtype=float), aspect="auto", cmap="Greys", vmin=0, vmax=1)
    ax.set_xticks(range(len(frame.columns)))
    ax.set_xticklabels(frame.columns, rotation=45, ha="right")
    ax.set_title(f"Missing values by row (first {len(frame)} rows; dark = missing)")
    ax.set_ylabel("row")
    return fig


@register(
    name="missing_heatmap",
    kind="viz",
    stage="missingness",
    code_template="edacore.viz.missing_heatmap({df})",
)
def missing_heatmap(df: pd.DataFrame) -> Figure:
    """Correlation of missingness between columns.

    Two columns that go missing together point at one mechanism rather
    than two, which is the first evidence for MAR over MCAR.
    """
    indicators = df.isna().astype(float)
    varying = indicators.loc[:, indicators.std() > 0]
    fig, axes = _figure(figsize=(5.5, 5.0))
    ax = axes[0][0]
    if varying.shape[1] < 2:
        ax.text(
            0.5, 0.5, "fewer than two columns have any missing values", ha="center", va="center"
        )
        ax.set_title("Missingness correlation")
        return fig
    matrix = varying.corr()
    image = ax.imshow(matrix.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_title("Missingness correlation")
    return fig


@register(
    name="missing_by_group",
    kind="viz",
    stage="missingness",
    code_template="edacore.viz.missing_by_group({df}, col={col!r}, group={group!r})",
)
def missing_by_group(df: pd.DataFrame, col: str, group: str) -> Figure:
    """Share of `col` missing within each level of `group`.

    A difference across groups is evidence for MAR: the missingness
    depends on something observed, so a model conditioned on it can impute
    honestly (Section 15.3's `mar_by_group` trap).
    """
    shares = df.groupby(df[group].astype(str), observed=True)[col].apply(
        lambda values: float(values.isna().mean())
    )
    shares = shares.sort_index()
    fig, axes = _figure()
    ax = axes[0][0]
    ax.bar(range(len(shares)), shares.to_numpy(dtype=float), color=_PRIMARY)
    ax.set_xticks(range(len(shares)))
    ax.set_xticklabels(shares.index, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    _finish(ax, f"Share of {col} missing, by {group}", group, "proportion missing")
    return fig


# --------------------------------------------------------------------------
# Outliers and transforms (the detectors are M11)
# --------------------------------------------------------------------------


@register(
    name="outlier_plot",
    kind="viz",
    stage="outliers",
    code_template=(
        "edacore.viz.outlier_plot({df}, col={col!r}, flagged={flagged!r}, whis={whis!r})"
    ),
)
def outlier_plot(
    df: pd.DataFrame,
    col: str,
    flagged: Sequence[int] | None = None,
    whis: float = 1.5,
) -> Figure:
    """Values in row order with the flagged ones marked.

    `flagged` is row labels, which M11's detectors will supply. Until then
    it defaults to the Tukey fence, so the plot is useful now — but the
    fence is a rule of thumb, not a verdict, and the title says which rule
    drew the marks.
    """
    values = pd.to_numeric(df[col], errors="coerce")
    if flagged is None:
        q1, q3 = values.quantile([0.25, 0.75])
        span = float(q3 - q1)
        low, high = float(q1) - whis * span, float(q3) + whis * span
        marked = values.index[(values < low) | (values > high)]
        rule = f"Tukey fence, whis={whis}"
    else:
        marked = pd.Index(list(flagged))
        rule = "flagged by the outlier stage"

    fig, axes = _figure()
    ax = axes[0][0]
    ax.plot(
        range(len(values)),
        values.to_numpy(dtype=float),
        "o",
        color=_PRIMARY,
        markersize=4,
        alpha=0.6,
    )
    positions = [values.index.get_loc(label) for label in marked if label in values.index]
    if positions:
        ax.plot(
            positions,
            values.loc[marked].to_numpy(dtype=float),
            "o",
            color=STATUS_COLOURS["fail"],
            markersize=7,
            fillstyle="none",
            markeredgewidth=1.5,
        )
    _finish(ax, f"{col}: {len(positions)} flagged ({rule})", "row", col)
    return fig


@register(
    name="before_after",
    kind="viz",
    stage="transform",
    code_template="edacore.viz.before_after({df}, after={after}, col={col!r})",
)
def before_after(
    before: pd.DataFrame,
    after: pd.DataFrame,
    col: str,
    record: TransformRecord | None = None,
) -> Figure:
    """One column's distribution either side of a transform, with the
    skewness that usually motivated it.

    Section 6.11 writes this as `before_after(col, record)`, but a
    `TransformRecord` carries summaries rather than the data, so both
    frames are passed. The record is used for the subtitle when given.
    """
    fig, axes = _figure(1, 2, figsize=(4.5, 4.0))
    for ax, frame, label in ((axes[0][0], before, "before"), (axes[0][1], after, "after")):
        values = _numeric(frame, col)
        ax.hist(values, bins="auto", color=_PRIMARY, edgecolor="white")
        skew = float(stats.skew(values)) if len(values) > 2 else float("nan")
        _finish(ax, f"{col} {label} (skew {skew:.2f})", col, "count")
    fig.suptitle(record.function if record is not None else f"{col}: before and after")
    return fig


# --------------------------------------------------------------------------
# Time series (Section 6.9's plots; the analysis is M12)
# --------------------------------------------------------------------------


@register(
    name="ts_line",
    kind="viz",
    stage="timeseries",
    code_template=("edacore.viz.ts_line({df}, col={col!r}, time={time!r}, entity={entity!r})"),
)
def ts_line(df: pd.DataFrame, col: str, time: str, entity: str | None = None) -> Figure:
    """A series over time; one line per entity for panel data."""
    frame = df[[c for c in (col, time, entity) if c is not None]].dropna().sort_values(time)
    fig, axes = _figure(figsize=(7.0, 3.5))
    ax = axes[0][0]
    if entity is None:
        ax.plot(frame[time], frame[col].to_numpy(dtype=float), color=_PRIMARY, linewidth=1)
    else:
        for label in sorted(frame[entity].astype(str).unique()):
            part = frame[frame[entity].astype(str) == label]
            ax.plot(part[time], part[col].to_numpy(dtype=float), linewidth=1, label=label)
        if frame[entity].nunique() <= 10:
            ax.legend(fontsize="small")
    _finish(ax, f"{col} over {time}", time, col)
    return fig


@register(
    name="stl_plot",
    kind="viz",
    stage="timeseries",
    code_template="edacore.viz.stl_plot({df}, col={col!r}, period={period!r})",
)
def stl_plot(df: pd.DataFrame, col: str, period: int = 12) -> Figure:
    """STL decomposition: observed, trend, seasonal, residual."""
    values = _numeric(df, col)
    result = STL(values, period=period, robust=True).fit()
    fig, axes = _figure(4, 1, figsize=(7.0, 1.8))
    for ax, series, label in zip(
        (axes[0][0], axes[1][0], axes[2][0], axes[3][0]),
        (values, result.trend, result.seasonal, result.resid),
        ("observed", "trend", "seasonal", "residual"),
        strict=True,
    ):
        ax.plot(
            np.arange(len(series)), np.asarray(series, dtype=float), color=_PRIMARY, linewidth=1
        )
        ax.set_ylabel(label)
        ax.grid(True, **_GRID)
    axes[0][0].set_title(f"STL decomposition of {col} (period {period})")
    axes[3][0].set_xlabel("index")
    return fig


@register(
    name="acf_plot",
    kind="viz",
    stage="timeseries",
    code_template="edacore.viz.acf_plot({df}, col={col!r}, lags={lags!r})",
)
def acf_plot(df: pd.DataFrame, col: str, lags: int = 20) -> Figure:
    """Autocorrelation by lag: how far back the series remembers, and the
    direct evidence against treating its rows as independent."""
    fig, axes = _figure()
    plot_acf(_numeric(df, col), lags=lags, ax=axes[0][0], title=f"ACF of {col}")
    return fig


@register(
    name="pacf_plot",
    kind="viz",
    stage="timeseries",
    code_template="edacore.viz.pacf_plot({df}, col={col!r}, lags={lags!r})",
)
def pacf_plot(df: pd.DataFrame, col: str, lags: int = 20) -> Figure:
    """Partial autocorrelation: each lag's own contribution, with the
    shorter lags' influence removed."""
    fig, axes = _figure()
    plot_pacf(_numeric(df, col), lags=lags, ax=axes[0][0], title=f"PACF of {col}")
    return fig


@register(
    name="change_point_plot",
    kind="viz",
    stage="timeseries",
    code_template=(
        "edacore.viz.change_point_plot({df}, col={col!r}, change_points={change_points!r}, "
        "time={time!r})"
    ),
)
def change_point_plot(
    df: pd.DataFrame,
    col: str,
    change_points: Sequence[int],
    time: str | None = None,
) -> Figure:
    """A series with detected change points marked, and each segment's mean.

    The points are passed in rather than detected here: detection is
    Section 6.9's, which lands in M12.
    """
    values = _numeric(df, col)
    positions = [int(p) for p in change_points if 0 < int(p) < len(values)]
    fig, axes = _figure(figsize=(7.0, 3.5))
    ax = axes[0][0]
    ax.plot(np.arange(len(values)), values, color=_PRIMARY, linewidth=1)
    for start, end in zip([0, *positions], [*positions, len(values)], strict=True):
        segment = values[start:end]
        if len(segment):
            ax.hlines(float(segment.mean()), start, end, color=_SECONDARY, linewidth=2)
    for point in positions:
        ax.axvline(point, color=STATUS_COLOURS["fail"], linestyle="--", linewidth=1)
    _finish(ax, f"{col}: {len(positions)} change point(s)", time or "index", col)
    return fig


# --------------------------------------------------------------------------
# Text (Section 6.10's plots; the analysis is M13)
# --------------------------------------------------------------------------


@register(
    name="text_length_hist",
    kind="viz",
    stage="text",
    code_template="edacore.viz.text_length_hist({df}, col={col!r}, unit={unit!r})",
)
def text_length_hist(df: pd.DataFrame, col: str, unit: str = "characters") -> Figure:
    """Length distribution of a text column, in characters or words."""
    text = df[col].dropna().astype(str)
    lengths = (text.str.len() if unit == "characters" else text.str.split().map(len)).to_numpy(
        dtype=float
    )
    fig, axes = _figure()
    ax = axes[0][0]
    if len(lengths):
        ax.hist(lengths, bins="auto", color=_PRIMARY, edgecolor="white")
    _finish(ax, f"Length of {col} ({unit})", unit, "count")
    return fig


@register(
    name="top_terms_bar",
    kind="viz",
    stage="text",
    code_template="edacore.viz.top_terms_bar({df}, col={col!r}, top={top!r})",
)
def top_terms_bar(df: pd.DataFrame, col: str, top: int = 20) -> Figure:
    """The most frequent terms in a text column.

    Tokenising is a plain lowercase word split. Nothing about the text
    leaves this process either way (rule 5), but the counts are what a
    card would show, and a card should not depend on an optional
    dependency's tokeniser being installed.
    """
    tokens: list[str] = []
    for value in df[col].dropna().astype(str):
        tokens.extend(re.findall(r"[a-z0-9']+", value.lower()))
    counts = pd.Series(tokens, dtype="object").value_counts().head(top)
    fig, axes = _figure(figsize=(6.0, 4.5))
    ax = axes[0][0]
    ax.barh(range(len(counts)), counts.to_numpy(dtype=float), color=_PRIMARY)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index)
    ax.invert_yaxis()
    _finish(ax, f"Most frequent terms in {col}", "count", "")
    return fig


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@register(
    name="effect_size_forest",
    kind="viz",
    stage="hypothesis",
    takes_df=False,
    code_template="edacore.viz.effect_size_forest(results={results!r})",
)
def effect_size_forest(results: Sequence[TestResult]) -> Figure:
    """Every effect size in the session with its interval, on one axis.

    The plot that makes Section 6.12's two size guards visible at a
    glance: a significant result whose interval straddles zero next to one
    that does not, and a large estimate whose interval is far wider than
    the rest.
    """
    usable = [r for r in results if r.effect_size is not None]
    fig, axes = _figure(figsize=(6.5, max(2.0, 0.5 * len(usable) + 1.0)))
    ax = axes[0][0]
    if not usable:
        ax.text(0.5, 0.5, "no effect sizes to show", ha="center", va="center")
        ax.set_title("Effect sizes")
        return fig

    for position, result in enumerate(usable):
        estimate = float(result.effect_size)  # type: ignore[arg-type]
        ax.plot([estimate], [position], "o", color=_PRIMARY, markersize=6)
        if result.effect_size_ci is not None:
            low, high = result.effect_size_ci
            ax.hlines(position, float(low), float(high), color=_PRIMARY, linewidth=1.5)
    ax.axvline(0.0, color=STATUS_COLOURS["untestable"], linestyle="--", linewidth=1)
    ax.set_yticks(range(len(usable)))
    ax.set_yticklabels([f"{r.function} ({r.effect_size_name})" for r in usable])
    ax.invert_yaxis()
    _finish(ax, "Effect sizes with confidence intervals", "effect size", "")
    return fig
