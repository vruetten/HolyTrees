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

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .cells import Cell
    from .model import Run


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label face-connected components of a boolean ``mask``.

    Face-connectivity means 4-connectivity in 2D and 6-connectivity in 3D (no
    diagonals). Implemented with a weighted union-find over neighbor pairs, which
    are gathered vectorized per axis, so only the (small) union loop is in Python.

    Parameters
    ----------
    mask : numpy.ndarray
        Boolean array of any dimensionality.

    Returns
    -------
    labels : numpy.ndarray
        ``int64`` array of the same shape; background is ``0`` and each component
        gets a label in ``1..nlabels``.
    nlabels : int
        Number of connected components.
    """
    mask = np.asarray(mask, dtype=bool)
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

    for ax in range(mask.ndim):
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


__all__ = ["label_components", "cell_lobes", "flag_overmerged"]
