"""Matplotlib visualization helpers (ports of the Julia inspection tutorial).

Every function returns ``(fig, ax)``, accepts an optional ``ax=`` (to compose
into a subplot grid) and ``save=`` (a path to write a PNG), and handles both 2D
and 3D (volumetric) runs via a uniform ``reduce=`` argument:

- ``reduce="maxproj"`` (default) collapses Z by maximum projection;
- ``reduce=<int>`` shows a single Z plane.

For 2D runs ``reduce`` is ignored.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import hsv_to_rgb

from .cells import paint_cells

if TYPE_CHECKING:
    from .model import Run


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────
def _axes(ax=None, figsize=(6, 6)):
    """Return ``(fig, ax)``, creating a new figure if ``ax`` is None."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    return fig, ax


def _finish(fig, save):
    """Save the figure if a path was given."""
    if save is not None:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig


def _reduce_image(img: np.ndarray, reduce) -> np.ndarray:
    """Collapse a possibly-3D spatial image to 2D for display."""
    if img.ndim <= 2:
        return img
    if isinstance(reduce, (int, np.integer)):
        return img[:, :, int(reduce)]
    return img.max(axis=2)  # "maxproj" / default


def _reduce_paint(value: np.ndarray, alpha: np.ndarray, reduce, hue=None):
    """Collapse 3D paint outputs to 2D, choosing the plane of maximum alpha."""
    if value.ndim <= 2:
        return value, alpha, hue
    if isinstance(reduce, (int, np.integer)):
        z = int(reduce)
        h = None if hue is None else hue[:, :, z]
        return value[:, :, z], alpha[:, :, z], h
    zi = np.argmax(alpha, axis=2)  # dominant plane per pixel
    yy, xx = np.indices(zi.shape)
    v = value[yy, xx, zi]
    a = alpha[yy, xx, zi]
    h = None if hue is None else hue[yy, xx, zi]
    return v, a, h


def _clip_limits(img: np.ndarray, clip):
    """Percentile-based intensity limits, ignoring NaNs."""
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(finite, [100 * clip[0], 100 * clip[1]])
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


# ──────────────────────────────────────────────────────────────────────────────
# image
# ──────────────────────────────────────────────────────────────────────────────
def show_image(img, *, clip=(0.02, 0.98), cmap="gray", reduce="maxproj", ax=None, save=None,
               title=None):
    """Display a spatial image with percentile contrast clipping.

    Parameters
    ----------
    img : array_like
        2D ``(Y, X)`` or 3D ``(Y, X, Z)`` image.
    clip : (float, float), optional
        Lower/upper percentiles (in ``[0, 1]``) for contrast.
    cmap : str, optional
    reduce : {'maxproj', int}, optional
        Z reduction for 3D input.
    ax : matplotlib.axes.Axes, optional
    save : str, optional
    title : str, optional

    Returns
    -------
    (matplotlib.figure.Figure, matplotlib.axes.Axes)
    """
    img = _reduce_image(np.asarray(img), reduce)
    vmin, vmax = _clip_limits(img, clip)
    fig, ax = _axes(ax)
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    return _finish(fig, save), ax


# ──────────────────────────────────────────────────────────────────────────────
# cells located (random color, alpha by weight)
# ──────────────────────────────────────────────────────────────────────────────
def show_cells_located(run: Run, *, colorby="random", seed=0, anatomy=None,
                        reduce="maxproj", ax=None, save=None, title=None):
    """Show all cells as colored footprints, alpha proportional to spatial weight.

    Parameters
    ----------
    run : Run
    colorby : {'random', 'component'}, optional
        Hue assignment per cell.
    seed : int, optional
        RNG seed for ``colorby='random'``.
    anatomy : array_like, optional
        Grayscale background to draw cells over; defaults to ``run.bg.S``.
    reduce : {'maxproj', int}, optional
        Z reduction for 3D runs.
    ax, save, title
        As elsewhere.

    Returns
    -------
    (Figure, Axes)
    """
    paint = paint_cells(run, colorby=colorby, seed=seed)
    value, alpha, hue = _reduce_paint(paint["value"], paint["alpha"], reduce, paint.get("hue"))

    fig, ax = _axes(ax)
    if anatomy is None:
        anatomy = run.bg.S
    bg = _reduce_image(np.asarray(anatomy), reduce)
    vmin, vmax = _clip_limits(bg, (0.02, 0.98))
    ax.imshow(bg, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")

    rgb = hsv_to_rgb(np.stack([hue, np.ones_like(hue), np.ones_like(hue)], axis=-1))
    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
    ax.imshow(rgba, interpolation="nearest")
    ax.set_axis_off()
    ax.set_title(title or f"{run.ncells} cells")
    return _finish(fig, save), ax


# ──────────────────────────────────────────────────────────────────────────────
# project cells by scalar value
# ──────────────────────────────────────────────────────────────────────────────
def project_cells(run: Run, values, *, label=None, cmap="viridis", reduce="maxproj",
                  ax=None, save=None, title=None):
    """Paint cells into anatomical space, colored by a per-cell scalar.

    Parameters
    ----------
    run : Run
    values : callable or array_like
        ``Cell -> float`` (e.g. ``lambda c: c.T.var()``) or a length-``ncells`` array.
    label : str, optional
        Colorbar label.
    cmap : str, optional
    reduce : {'maxproj', int}, optional
    ax, save, title

    Returns
    -------
    (Figure, Axes)
    """
    paint = paint_cells(run, values=values)
    value, alpha, _ = _reduce_paint(paint["value"], paint["alpha"], reduce)

    fig, ax = _axes(ax)
    masked = np.ma.masked_where(alpha <= 0, value)
    im = ax.imshow(masked, cmap=cmap, interpolation="nearest")
    ax.set_axis_off()
    ax.set_title(title or (label or "per-cell value"))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if label:
        cb.set_label(label)
    return _finish(fig, save), ax


# ──────────────────────────────────────────────────────────────────────────────
# single cell
# ──────────────────────────────────────────────────────────────────────────────
def show_cell(run: Run, k: int, *, anatomy=None, reduce="maxproj", ax=None, save=None):
    """Show a single cell ``k``: its footprint over the anatomy, with its trace inset.

    Parameters
    ----------
    run : Run
    k : int
        Canonical cell index.
    anatomy : array_like, optional
        Background image; defaults to ``run.bg.S``.
    reduce, ax, save

    Returns
    -------
    (Figure, Axes)
    """
    cell = run.cells[k]
    frame = run.fill_subset([k], [1.0], weighted=True, reduce="add")
    fp = _reduce_image(frame, reduce)

    fig, ax = _axes(ax)
    if anatomy is None:
        anatomy = run.bg.S
    bg = _reduce_image(np.asarray(anatomy), reduce)
    vmin, vmax = _clip_limits(bg, (0.02, 0.98))
    ax.imshow(bg, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.imshow(np.ma.masked_where(fp <= 0, fp), cmap="inferno", alpha=0.85,
              interpolation="nearest")
    cy, cx = cell.centroid[0], cell.centroid[1]
    ax.plot(cx, cy, "c+", markersize=12, markeredgewidth=2)
    ax.set_axis_off()
    ax.set_title(f"cell {k} (tile {cell.tileid}, comp {cell.comp})")

    # trace inset
    iax = ax.inset_axes([0.0, -0.22, 1.0, 0.18])
    iax.plot(cell.T, color="k", lw=0.8)
    iax.set_xlabel("time")
    iax.set_yticks([])
    return _finish(fig, save), ax


# ──────────────────────────────────────────────────────────────────────────────
# raster
# ──────────────────────────────────────────────────────────────────────────────
def cell_raster(run: Run, *, normalize=True, cmap="magma", ax=None, save=None, title=None):
    """Raster image of the data matrix (cells x time).

    Parameters
    ----------
    run : Run
    normalize : bool, optional
        Scale each cell's trace to ``[0, 1]`` (peak-normalized) for display.
    cmap, ax, save, title

    Returns
    -------
    (Figure, Axes)
    """
    M = run.traces.astype(np.float32)
    if normalize and M.size:
        peak = M.max(axis=1, keepdims=True)
        peak[peak == 0] = 1.0
        M = M / peak
    fig, ax = _axes(ax, figsize=(8, 5))
    im = ax.imshow(M, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xlabel("time")
    ax.set_ylabel("cell")
    ax.set_title(title or f"data matrix ({M.shape[0]} cells x {M.shape[1]} time)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _finish(fig, save), ax


# ──────────────────────────────────────────────────────────────────────────────
# merge penalties
# ──────────────────────────────────────────────────────────────────────────────
def plot_merge_penalties(run: Run, *, ax=None, save=None):
    """Plot the pairwise merge penalties in the order they were applied.

    Parameters
    ----------
    run : Run
    ax, save

    Returns
    -------
    (Figure, Axes)
    """
    fig, ax = _axes(ax, figsize=(6, 4))
    sched = run.merge_schedule
    if sched is None or len(sched) == 0:
        ax.text(0.5, 0.5, "no merge schedule", ha="center", va="center")
        ax.set_axis_off()
        return _finish(fig, save), ax
    penalties = np.asarray(sched[sched.dtype.names[-1]], dtype=float)
    ax.plot(penalties, marker=".", lw=0.8)
    ax.set_xlabel("merge step")
    ax.set_ylabel("penalty")
    ax.set_title("merge penalties")
    return _finish(fig, save), ax


__all__ = [
    "show_image",
    "show_cells_located",
    "project_cells",
    "show_cell",
    "cell_raster",
    "plot_merge_penalties",
]
