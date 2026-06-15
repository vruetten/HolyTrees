"""Cells, the data matrix, and the vectorized spatial "fill" engine.

A **cell** is one component of one tile: a spatial footprint ``S`` plus a
temporal trace ``T``. :func:`eachcell` flattens a :class:`~holytrees.model.TileTree`
into the canonical, ordered list of cells used everywhere (``cells[k]`` <->
``traces[k]``).

Spatial reconstruction and per-cell painting all funnel through a single
**packed footprint index** (a flat COO layout of every cell's nonzero footprint
pixels) so there is exactly one place that touches pixels, and it is fully
vectorized — no Python per-pixel loops, correct for overlapping footprints.

Coordinate frame: everything here is in the analysis-window frame ``(Y, X[, Z])``
(0-based). Use ``Cell.centroid_global`` / ``Cell.box_global`` to map back to the
original recording via the run's ``origin``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .model import Box, TileTree

if TYPE_CHECKING:
    from .model import Run


# ──────────────────────────────────────────────────────────────────────────────
# Cell
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Cell:
    """One factorized cell: a spatial footprint plus a temporal trace.

    Attributes
    ----------
    S : numpy.ndarray
        Footprint, spatial shape ``box.dims`` (``(Y, X[, Z])``), ``float32``.
    T : numpy.ndarray
        Trace, shape ``(ntimes,)``, ``float32``.
    box : Box
        Footprint location in the analysis-window frame.
    tileid : int
        Original tile index this cell came from.
    comp : int
        Component index within that tile.
    index : int
        Canonical cell index (position in ``run.cells`` / row of ``run.traces``).
    origin : tuple of int
        0-based window offset into the original image (from the run).
    """

    S: np.ndarray
    T: np.ndarray
    box: Box
    tileid: int
    comp: int
    index: int
    origin: tuple[int, ...]

    @property
    def amplitude(self) -> float:
        """Peak trace value over time (a natural NMF "brightness")."""
        return float(self.T.max()) if self.T.size else 0.0

    @property
    def area(self) -> int:
        """Number of nonzero footprint pixels."""
        return int(np.count_nonzero(self.S))

    @property
    def centroid(self) -> tuple[float, ...]:
        """Footprint-weighted centroid, **0-based** window coordinates ``(Y, X[, Z])``."""
        w = np.abs(self.S)
        total = w.sum()
        starts = [sl.start for sl in self.box.slices]
        if total == 0:
            return tuple(float(s) + d / 2.0 for s, d in zip(starts, self.box.dims))
        idx = np.indices(self.S.shape, dtype=np.float64)
        return tuple(
            float((idx[a] * w).sum() / total) + starts[a] for a in range(self.S.ndim)
        )

    @property
    def centroid_global(self) -> tuple[float, ...]:
        """Centroid mapped into the original-image frame (0-based)."""
        return tuple(c + o for c, o in zip(self.centroid, self.origin))

    @property
    def box_global(self) -> Box:
        """This cell's box translated into the original-image frame."""
        return self.box.shifted(self.origin)

    def add_to(self, frame: np.ndarray, value: float = 1.0) -> None:
        """Add ``value * S`` into ``frame`` at this cell's window-frame slices (in place)."""
        frame[self.box.slices] += (self.S * value).astype(frame.dtype, copy=False)


# ──────────────────────────────────────────────────────────────────────────────
# eachcell + data matrix
# ──────────────────────────────────────────────────────────────────────────────
def eachcell(ttree: TileTree, *, origin: tuple[int, ...] = (0, 0)) -> list[Cell]:
    """Flatten a :class:`~holytrees.model.TileTree` into the canonical cell list.

    Parameters
    ----------
    ttree : TileTree
        The valid tiles.
    origin : tuple of int, optional
        0-based window offset into the original image (stored on each cell).

    Returns
    -------
    list of Cell
        One entry per tile component, in tile order then component order. This
        ordering is canonical — used by :func:`data_matrix`, :func:`fill`, and
        all visualizations.
    """
    cells: list[Cell] = []
    k = 0
    for tid, tile in ttree.eachtile():
        for c in range(tile.ncomponents):
            cells.append(
                Cell(
                    S=np.asarray(tile.component_S(c), dtype=np.float32),
                    T=np.asarray(tile.component_T(c), dtype=np.float32),
                    box=tile.box,
                    tileid=int(tid),
                    comp=c,
                    index=k,
                    origin=origin,
                )
            )
            k += 1
    return cells


def data_matrix(run: Run) -> np.ndarray:
    """Stack the cell traces into the ``(ncells, ntimes)`` data matrix.

    Parameters
    ----------
    run : Run

    Returns
    -------
    numpy.ndarray
        Shape ``(ncells, ntimes)``, ``float32``. Row ``k`` is ``run.cells[k].T``.
        For an empty run, shape ``(0, ntimes)``.
    """
    cells = run.cells
    if not cells:
        return np.zeros((0, run.ntimes), dtype=np.float32)
    return np.stack([c.T for c in cells]).astype(np.float32, copy=False)


# ──────────────────────────────────────────────────────────────────────────────
# Packed footprint index (the single vectorized engine)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class FootprintIndex:
    """Flat COO layout of every cell's nonzero footprint pixels.

    Attributes
    ----------
    flat_index : numpy.ndarray
        ``int64`` flattened (C-order) frame index of each pixel, length ``npix``.
    cell : numpy.ndarray
        ``int32`` owning cell index per pixel.
    weight : numpy.ndarray
        ``float32`` footprint value ``S`` at each pixel.
    norm_weight : numpy.ndarray
        ``float32`` per-cell-normalized ``|S| / max|S|`` (in ``[0, 1]``), for alpha.
    shape : tuple of int
        Spatial frame shape ``(Y, X[, Z])``.
    ncells : int
        Number of cells indexed.
    """

    flat_index: np.ndarray
    cell: np.ndarray
    weight: np.ndarray
    norm_weight: np.ndarray
    shape: tuple[int, ...]
    ncells: int

    @property
    def npix(self) -> int:
        """Total number of footprint pixels across all cells."""
        return int(self.flat_index.size)

    @property
    def framesize(self) -> int:
        """Number of pixels in a flattened spatial frame."""
        return int(np.prod(self.shape)) if self.shape else 0


def build_footprint_index(run: Run) -> FootprintIndex:
    """Build the packed footprint index for all cells in ``run`` (cached on the run).

    Loops once over cells (hundreds–thousands), never over pixels; only nonzero
    footprint pixels are stored, so overlapping/variable-size tiles cost nothing
    extra and reconstruction stays exact for additive reduction.
    """
    shape = tuple(int(s) for s in run.spacesize)
    cells = run.cells
    flat_parts, cell_parts, w_parts, nw_parts = [], [], [], []
    for c in cells:
        S = np.asarray(c.S)
        nz = np.flatnonzero(S.ravel())
        if nz.size == 0:
            continue
        starts = np.array([sl.start for sl in c.box.slices], dtype=np.int64)
        # local (within-box) C-order multi-index of nonzero pixels -> global coords
        local = np.array(np.unravel_index(nz, S.shape), dtype=np.int64)  # (ndim, n)
        glob = local + starts[:, None]
        flat_parts.append(np.ravel_multi_index(tuple(glob), shape))
        wv = S.ravel()[nz].astype(np.float32)
        w_parts.append(wv)
        aw = np.abs(wv)
        m = aw.max()
        nw_parts.append(aw / m if m > 0 else aw)
        cell_parts.append(np.full(nz.size, c.index, dtype=np.int32))

    if flat_parts:
        flat_index = np.concatenate(flat_parts)
        cell = np.concatenate(cell_parts)
        weight = np.concatenate(w_parts)
        norm_weight = np.concatenate(nw_parts)
    else:
        flat_index = np.empty(0, dtype=np.int64)
        cell = np.empty(0, dtype=np.int32)
        weight = np.empty(0, dtype=np.float32)
        norm_weight = np.empty(0, dtype=np.float32)
    return FootprintIndex(flat_index, cell, weight, norm_weight, shape, len(cells))


def _scatter(index: np.ndarray, contrib: np.ndarray, n: int, reduce: str) -> np.ndarray:
    """Scatter ``contrib`` into a length-``n`` zero frame at flat positions ``index``."""
    out = np.zeros(n, dtype=np.float32)
    if reduce == "add":
        np.add.at(out, index, contrib)
    elif reduce == "overwrite":
        out[index] = contrib
    elif reduce == "max":
        # assign in ascending-contribution order so the largest writes last
        order = np.argsort(contrib, kind="stable")
        out[index[order]] = contrib[order]
    else:
        raise ValueError(f"reduce must be 'add', 'overwrite', or 'max'; got {reduce!r}")
    return out


def fill(run: Run, values, *, weighted: bool = False, reduce: str = "overwrite") -> np.ndarray:
    """Scatter per-cell ``values`` into a spatial frame — one vectorized pass.

    Parameters
    ----------
    run : Run
    values : array_like
        Either shape ``(ncells,)`` -> a single frame, or ``(nfeatures, ncells)``
        -> a stack of frames. ``values[k]`` is the value for ``run.cells[k]``.
    weighted : bool, optional
        If True, multiply each pixel contribution by the footprint value ``S``
        at that pixel (so a cell paints its spatial profile, not a flat blob).
    reduce : {'overwrite', 'add', 'max'}, optional
        How overlapping footprints combine. ``'add'`` uses an unbuffered
        scatter-add (correct for overlaps; used by reconstruction). ``'overwrite'``
        lets an arbitrary cell win; ``'max'`` lets the largest contribution win.

    Returns
    -------
    numpy.ndarray
        Shape ``(Y, X[, Z])`` for 1-D ``values``, or ``(nfeatures, Y, X[, Z])``
        for 2-D ``values``; ``float32``.
    """
    idx = run.footprints
    values = np.asarray(values, dtype=np.float32)
    shape = idx.shape
    fsize = idx.framesize
    w = idx.weight if weighted else None

    if values.ndim == 1:
        _check_len(values.shape[0], idx.ncells)
        contrib = values[idx.cell]
        if w is not None:
            contrib = contrib * w
        return _scatter(idx.flat_index, contrib, fsize, reduce).reshape(shape)

    if values.ndim == 2:
        nf = values.shape[0]
        _check_len(values.shape[1], idx.ncells)
        contrib = values[:, idx.cell]  # (nf, npix)
        if w is not None:
            contrib = contrib * w[None, :]
        # combine feature offset + pixel index into one flat scatter
        offsets = (np.arange(nf, dtype=np.int64) * fsize)[:, None]
        full_index = (offsets + idx.flat_index[None, :]).ravel()
        out = _scatter(full_index, contrib.ravel(), nf * fsize, reduce)
        return out.reshape((nf, *shape))

    raise ValueError(f"values must be 1-D or 2-D; got ndim={values.ndim}")


def fill_subset(
    run: Run, indices, values, *, weighted: bool = False, reduce: str = "overwrite"
) -> np.ndarray:
    """Like :func:`fill`, but only the cells in ``indices`` contribute.

    Parameters
    ----------
    run : Run
    indices : array_like of int
        Canonical cell indices to include.
    values : array_like
        Length ``len(indices)`` (or ``(nfeatures, len(indices))``); aligned to
        ``indices``.
    weighted, reduce
        As in :func:`fill`.

    Returns
    -------
    numpy.ndarray
        Same shape convention as :func:`fill`.
    """
    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=np.float32)
    full = np.zeros(
        (run.ncells,) if values.ndim == 1 else (values.shape[0], run.ncells),
        dtype=np.float32,
    )
    if values.ndim == 1:
        full[indices] = values
    else:
        full[:, indices] = values
    return fill(run, full, weighted=weighted, reduce=reduce)


def _check_len(got: int, want: int) -> None:
    if got != want:
        raise ValueError(f"values has {got} entries but run has {want} cells")


# ──────────────────────────────────────────────────────────────────────────────
# Reconstruction (thin wrappers over fill)
# ──────────────────────────────────────────────────────────────────────────────
def reconstruct_frame(run: Run, t: int) -> np.ndarray:
    """Reconstruct the spatial frame at time ``t``: ``sum_k S_k * T_k[t]``.

    Parameters
    ----------
    run : Run
    t : int
        Time index.

    Returns
    -------
    numpy.ndarray
        Shape ``(Y, X[, Z])``, ``float32``.
    """
    if run.ncells == 0:
        return np.zeros(run.spacesize, dtype=np.float32)
    return fill(run, run.traces[:, t], weighted=True, reduce="add")


def reconstruct_maxproj(run: Run) -> np.ndarray:
    """Amplitude-weighted spatial reconstruction (a max-projection surrogate).

    Uses each cell's peak trace value as its amplitude and scatter-adds the
    weighted footprints, so no ``(Y, X[, Z], ntimes)`` cube is ever built.

    Returns
    -------
    numpy.ndarray
        Shape ``(Y, X[, Z])``, ``float32``.
    """
    if run.ncells == 0:
        return np.zeros(run.spacesize, dtype=np.float32)
    amplitude = run.traces.max(axis=1)
    return fill(run, amplitude, weighted=True, reduce="add")


# ──────────────────────────────────────────────────────────────────────────────
# Painting cells by value / color
# ──────────────────────────────────────────────────────────────────────────────
def resolve_values(run: Run, values) -> np.ndarray:
    """Resolve ``values`` (a callable ``Cell -> float`` or an array) to a length-ncells array."""
    if callable(values):
        arr = np.array([float(values(c)) for c in run.cells], dtype=np.float32)
    else:
        arr = np.asarray(values, dtype=np.float32)
        _check_len(arr.shape[0], run.ncells)
    return arr


def paint_cells(
    run: Run,
    values=None,
    *,
    colorby: str | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Paint cells into anatomical space, each pixel taking its **dominant** cell.

    For every frame pixel the cell with the largest normalized footprint weight
    "wins"; that cell supplies the painted ``value`` (and ``hue``), and ``alpha``
    is its normalized weight. This is a single vectorized pass over footprint
    pixels (no per-pixel loop).

    Parameters
    ----------
    run : Run
    values : callable or array_like, optional
        Per-cell scalar to paint (callable ``Cell -> float`` or length-``ncells``
        array). Defaults to each cell's amplitude.
    colorby : {'random', 'component'}, optional
        If given, also return a per-pixel ``hue`` in ``[0, 1)``: random per cell
        (seeded) or sequential by cell index.
    seed : int, optional
        RNG seed for ``colorby='random'``.

    Returns
    -------
    dict
        ``{'value': (Y, X[, Z]), 'alpha': (Y, X[, Z])}`` and, if ``colorby`` is
        set, ``'hue': (Y, X[, Z])``. All ``float32`` in the window frame.
    """
    idx = run.footprints
    shape = idx.shape
    fsize = idx.framesize
    out: dict[str, np.ndarray] = {}

    if run.ncells == 0 or idx.npix == 0:
        out["value"] = np.zeros(shape, dtype=np.float32)
        out["alpha"] = np.zeros(shape, dtype=np.float32)
        if colorby is not None:
            out["hue"] = np.zeros(shape, dtype=np.float32)
        return out

    if values is None:
        vals = np.array([c.amplitude for c in run.cells], dtype=np.float32)
    else:
        vals = resolve_values(run, values)

    order = np.argsort(idx.norm_weight, kind="stable")  # dominant cell writes last
    fi = idx.flat_index[order]
    cell_ord = idx.cell[order]

    value_flat = np.zeros(fsize, dtype=np.float32)
    value_flat[fi] = vals[cell_ord]
    out["value"] = value_flat.reshape(shape)

    alpha_flat = np.zeros(fsize, dtype=np.float32)
    alpha_flat[fi] = idx.norm_weight[order]
    out["alpha"] = alpha_flat.reshape(shape)

    if colorby is not None:
        hue = _cell_hues(run.ncells, colorby, seed)
        hue_flat = np.zeros(fsize, dtype=np.float32)
        hue_flat[fi] = hue[cell_ord]
        out["hue"] = hue_flat.reshape(shape)
    return out


def _cell_hues(ncells: int, colorby: str, seed: int) -> np.ndarray:
    """Per-cell hue in ``[0, 1)``."""
    if colorby == "random":
        return np.random.default_rng(seed).random(ncells).astype(np.float32)
    if colorby == "component":
        return (np.arange(ncells, dtype=np.float32) / max(ncells, 1)) % 1.0
    raise ValueError(f"colorby must be 'random' or 'component'; got {colorby!r}")


__all__ = [
    "Cell",
    "FootprintIndex",
    "eachcell",
    "data_matrix",
    "build_footprint_index",
    "fill",
    "fill_subset",
    "reconstruct_frame",
    "reconstruct_maxproj",
    "paint_cells",
    "resolve_values",
]
