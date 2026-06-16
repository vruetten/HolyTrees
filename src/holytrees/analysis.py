"""Spatial-connectivity diagnostics for cells (over-merge detection).

A correctly-merged cell should have a **spatially connected** footprint. The
pairwise merge step can, however, fuse regions that only share a time course
even when their supports do not touch, producing a "cell" whose footprint is a
patchwork of disconnected *lobes*. These helpers quantify that: count the
connected components of a footprint and flag cells with more than one.

Everything here is pure NumPy (no SciPy): connected components are found with a
vectorized-edge union-find that works for 2D (4-connectivity) and 3D
(6-connectivity) footprints alike.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .cells import Cell
    from .model import Run


def label_components(mask: np.ndarray, axes: tuple[int, ...] | None = None) -> tuple[np.ndarray, int]:
    """Label face-connected components of a boolean ``mask``.

    Face-connectivity means 4-connectivity in 2D and 6-connectivity in 3D (no
    diagonals). Implemented with a weighted union-find over neighbor pairs, which
    are gathered vectorized per axis, so only the (small) union loop is in Python.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean array of any dimensionality.
    axes : tuple of int, optional
        Restrict connectivity to these axes (others become "stacking" axes across
        which voxels never connect). Defaults to all axes. Used for anisotropic
        stacks (e.g. a spaced z) so slices are analyzed independently.

    Returns
    -------
    labels : numpy.ndarray
        ``int64`` array of the same shape; background is ``0`` and each component
        gets a label in ``1..nlabels``.
    nlabels : int
        Number of connected components.
    """
    mask = np.asarray(mask, dtype=bool)
    ax_iter = tuple(range(mask.ndim)) if axes is None else tuple(axes)
    flat = np.flatnonzero(mask)
    if flat.size == 0:
        return np.zeros(mask.shape, dtype=np.int64), 0

    # Dense local id (0..n-1) for each foreground voxel, -1 elsewhere.
    local = np.full(mask.size, -1, dtype=np.int64)
    local[flat] = np.arange(flat.size)
    idxgrid = local.reshape(mask.shape)

    parent = np.arange(flat.size, dtype=np.int64)

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:  # path compression
            parent[a], a = root, parent[a]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for ax in ax_iter:
        sl_a = [slice(None)] * mask.ndim
        sl_b = [slice(None)] * mask.ndim
        sl_a[ax] = slice(None, -1)
        sl_b[ax] = slice(1, None)
        edge = mask[tuple(sl_a)] & mask[tuple(sl_b)]
        if not edge.any():
            continue
        ia = idxgrid[tuple(sl_a)][edge]
        ib = idxgrid[tuple(sl_b)][edge]
        for a, b in zip(ia.tolist(), ib.tolist()):
            union(a, b)

    roots = np.array([find(i) for i in range(flat.size)], dtype=np.int64)
    _, inv = np.unique(roots, return_inverse=True)
    labels = np.zeros(mask.size, dtype=np.int64)
    labels[flat] = inv + 1
    return labels.reshape(mask.shape), int(inv.max()) + 1


def cell_lobes(cell: "Cell", *, min_frac: float = 0.0) -> int:
    """Number of connected lobes in a cell's footprint.

    Parameters
    ----------
    cell : Cell
        The cell whose footprint ``S`` is analyzed.
    min_frac : float, optional
        Only count lobes that carry at least this fraction of the footprint's
        total mass (``sum |S|``). The default ``0.0`` counts every connected
        component; a small value (e.g. ``0.02``) ignores tiny speckle so the
        count reflects genuine, separated blobs.

    Returns
    -------
    int
        Number of qualifying lobes (``0`` for an all-zero footprint, ``1`` for a
        normal connected cell, ``>= 2`` for a disconnected/over-merged cell).
    """
    S = np.abs(np.asarray(cell.S, dtype=np.float64))
    mask = S > 0
    labels, n = label_components(mask)
    if n == 0:
        return 0
    if min_frac <= 0.0:
        return n
    total = S.sum()
    if total == 0:
        return 0
    mass = np.bincount(labels.ravel(), weights=S.ravel(), minlength=n + 1)[1:]
    return int((mass >= min_frac * total).sum())


def _flood_axes(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Axes with extent ``> 1`` — the contiguous spatial sheet to flood over.

    Mirrors the ``nohole`` merge gate: an axis of thickness 1 (e.g. a single
    z-plane of a widely-spaced stack) is a *stacking* axis and is excluded from
    connectivity, so enclosure is judged independently within each slice.
    """
    return tuple(d for d in range(len(shape)) if shape[d] > 1)


def _full_offsets(ndim: int, axes: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Half of the full-connectivity neighbor offsets restricted to ``axes``.

    Full connectivity is 8 in 2D / 26 in 3D (the dual of the 4-/6-connected
    foreground). Steps on non-``axes`` are forced to 0. Only offsets whose first
    nonzero step is ``+1`` are returned, since an offset and its negation
    describe the same neighbor pairs.
    """
    axset = set(axes)
    half: list[tuple[int, ...]] = []
    for combo in itertools.product((-1, 0, 1), repeat=ndim):
        if any(combo[d] != 0 and d not in axset for d in range(ndim)):
            continue
        nz = [c for c in combo if c != 0]
        if nz and nz[0] == 1:
            half.append(combo)
    return half


def label_components_full(mask: np.ndarray, axes: tuple[int, ...]) -> tuple[np.ndarray, int]:
    """Label components of ``mask`` under **full** connectivity over ``axes``.

    Like :func:`label_components` but with diagonals included (8-connectivity in
    2D, 26 in 3D), restricted to the given ``axes``. Used to flood the
    *background* when detecting enclosed holes, so the empty interior is the
    topological dual of the 4-/6-connected foreground.
    """
    mask = np.asarray(mask, dtype=bool)
    flat = np.flatnonzero(mask)
    if flat.size == 0:
        return np.zeros(mask.shape, dtype=np.int64), 0

    local = np.full(mask.size, -1, dtype=np.int64)
    local[flat] = np.arange(flat.size)
    idxgrid = local.reshape(mask.shape)
    parent = np.arange(flat.size, dtype=np.int64)

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    ndim = mask.ndim
    for off in _full_offsets(ndim, axes):
        sl_a, sl_b = [], []
        for d in range(ndim):
            if off[d] == 1:
                sl_a.append(slice(None, -1))
                sl_b.append(slice(1, None))
            elif off[d] == -1:
                sl_a.append(slice(1, None))
                sl_b.append(slice(None, -1))
            else:
                sl_a.append(slice(None))
                sl_b.append(slice(None))
        edge = mask[tuple(sl_a)] & mask[tuple(sl_b)]
        if not edge.any():
            continue
        ia = idxgrid[tuple(sl_a)][edge]
        ib = idxgrid[tuple(sl_b)][edge]
        for a, b in zip(ia.tolist(), ib.tolist()):
            union(a, b)

    roots = np.array([find(i) for i in range(flat.size)], dtype=np.int64)
    _, inv = np.unique(roots, return_inverse=True)
    labels = np.zeros(mask.size, dtype=np.int64)
    labels[flat] = inv + 1
    return labels.reshape(mask.shape), int(inv.max()) + 1


def _enclosed_background(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label background components of ``mask`` that are fully enclosed.

    Background (``~mask``) is labeled with **face** connectivity over the flood
    axes; any component that touches the array border along a flood axis is "open"
    (connected to the exterior) and dropped. What remains are the holes. Face
    connectivity means an empty cell whose face-neighbors are all occupied is
    enclosed even if a diagonal corner is open (occupancy is correspondingly
    8-/26-connected) — a box hemmed in on all sides counts as a hole.

    Returns ``(labels, nlabels)`` where ``labels`` is nonzero only on enclosed
    background voxels (relabeled ``1..nlabels``).
    """
    mask = np.asarray(mask, dtype=bool)
    fa = _flood_axes(mask.shape)
    if not fa:
        return np.zeros(mask.shape, dtype=np.int64), 0
    bg = ~mask
    labels, n = label_components(bg, fa)
    if n == 0:
        return labels, 0
    open_ids: set[int] = set()
    for d in fa:
        for end in (0, mask.shape[d] - 1):
            face = labels.take(end, axis=d)
            open_ids.update(int(v) for v in np.unique(face) if v != 0)
    enclosed = labels.copy()
    if open_ids:
        enclosed[np.isin(enclosed, list(open_ids))] = 0
    keep = np.unique(enclosed)
    keep = keep[keep != 0]
    if keep.size == 0:
        return np.zeros(mask.shape, dtype=np.int64), 0
    remap = np.zeros(int(labels.max()) + 1, dtype=np.int64)
    remap[keep] = np.arange(1, keep.size + 1)
    return remap[enclosed], int(keep.size)


def cell_holes(cell: "Cell", *, min_hole_px: int = 1, threshold: float = 0.0) -> int:
    """Number of enclosed background holes in a cell's footprint.

    A hole is a connected region of empty pixels fully surrounded by the cell's
    footprint (a ring/loop wrapping a void) — the pixel-level analogue of what
    the ``nohole`` merge gate prevents at box granularity. Background is flooded
    with **face** connectivity over the flood axes (4-conn in 2D; per-plane 2D for
    a thin/stacked z), so a void hemmed in on all sides is enclosed even if a
    diagonal corner is open (occupancy is correspondingly 8-/26-connected),
    matching the Julia gate.

    Parameters
    ----------
    cell : Cell
    min_hole_px : int, optional
        Minimum size (in pixels) for an enclosed region to count, ignoring
        single-pixel speckle holes when raised (default ``1``).
    threshold : float, optional
        Footprint magnitude above which a pixel is "occupied" (default ``0.0``,
        i.e. any nonzero pixel).

    Returns
    -------
    int
        Number of enclosed holes of at least ``min_hole_px`` pixels (``0`` for a
        simply-connected or empty footprint).
    """
    S = np.abs(np.asarray(cell.S, dtype=np.float64))
    mask = S > threshold
    labels, n = _enclosed_background(mask)
    if n == 0:
        return 0
    if min_hole_px <= 1:
        return n
    sizes = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    return int((sizes >= min_hole_px).sum())


def flag_holed(
    run: "Run",
    *,
    min_holes: int = 1,
    min_hole_px: int = 4,
    threshold: float = 0.0,
    indices=None,
) -> list[dict]:
    """Find cells whose footprint encloses one or more holes.

    An enclosed hole is the signature of a ring/loop over-merge: the merge step
    wrapped the footprint around empty space. This scans the run (or a subset)
    and returns the offending cells sorted by enclosed-hole area (largest first).
    It is an independent, pixel-level check on the box-level ``nohole`` merge
    gate, computed on the realized (post-``trim!``/``absorb_bg!``) footprints, so
    it may also surface small sub-tile holes the box gate does not target.

    Parameters
    ----------
    run : Run
    min_holes : int, optional
        Report a cell only if it has at least this many qualifying holes
        (default ``1``).
    min_hole_px : int, optional
        Minimum pixels for an enclosed region to count (default ``4``, ignoring
        tiny speckle voids).
    threshold : float, optional
        Footprint magnitude above which a pixel is "occupied" (default ``0.0``).
    indices : iterable of int, optional
        Restrict the scan to these cell indices (e.g. only the largest cells).
        Defaults to all cells.

    Returns
    -------
    list of dict
        One dict per flagged cell,
        ``{"index": int, "nholes": int, "hole_px": int, "area": int}``, sorted by
        ``hole_px`` descending. Empty if no cell encloses a hole.
    """
    cells = run.cells
    idxs = range(len(cells)) if indices is None else [int(i) for i in indices]
    flagged: list[dict] = []
    for k in idxs:
        c = cells[k]
        S = np.abs(np.asarray(c.S, dtype=np.float64))
        labels, n = _enclosed_background(S > threshold)
        if n == 0:
            continue
        sizes = np.bincount(labels.ravel(), minlength=n + 1)[1:]
        big = sizes[sizes >= min_hole_px]
        if big.size < min_holes:
            continue
        flagged.append(
            {
                "index": int(k),
                "nholes": int(big.size),
                "hole_px": int(big.sum()),
                "area": int(c.area),
            }
        )
    flagged.sort(key=lambda d: d["hole_px"], reverse=True)
    return flagged


def flag_overmerged(
    run: "Run",
    *,
    min_lobes: int = 2,
    min_frac: float = 0.02,
    indices=None,
) -> list[dict]:
    """Find cells whose footprint splits into multiple disconnected lobes.

    A multi-lobe footprint is the signature of an over-merge: the merge step
    combined spatially-separated regions into one cell. This scans the run (or a
    subset) and returns the offending cells, sorted by footprint area (largest
    first), so the worst cases surface immediately.

    Parameters
    ----------
    run : Run
    min_lobes : int, optional
        Report a cell only if it has at least this many qualifying lobes
        (default ``2``).
    min_frac : float, optional
        Mass fraction a lobe must carry to count, passed to :func:`cell_lobes`
        (default ``0.02``, i.e. ignore lobes holding < 2% of the footprint mass).
    indices : iterable of int, optional
        Restrict the scan to these cell indices (e.g. only the largest cells).
        Defaults to all cells.

    Returns
    -------
    list of dict
        One dict per flagged cell, ``{"index": int, "nlobes": int, "area": int}``,
        sorted by ``area`` descending. Empty if no cell is over-merged.
    """
    cells = run.cells
    idxs = range(len(cells)) if indices is None else [int(i) for i in indices]
    flagged: list[dict] = []
    for k in idxs:
        c = cells[k]
        n = cell_lobes(c, min_frac=min_frac)
        if n >= min_lobes:
            flagged.append({"index": int(k), "nlobes": int(n), "area": int(c.area)})
    flagged.sort(key=lambda d: d["area"], reverse=True)
    return flagged


__all__ = [
    "label_components",
    "label_components_full",
    "cell_lobes",
    "flag_overmerged",
    "cell_holes",
    "flag_holed",
]
