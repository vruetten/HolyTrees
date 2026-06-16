"""Object model for a tiled-NMF run: ``Box``, ``Tile``, ``TileTree``, ``Background``, ``Run``.

These mirror the Julia types from Tim Holy's tiled-factorization stack, but hold
plain numpy arrays (the run is fully decoded and the HDF5 file closed before a
:class:`Run` is returned, so it is picklable and carries no open file handles).

Coordinate conventions (read this once and the rest follows):

- Every spatial array is in **Julia logical order** ``(Y, X[, Z])`` (the decoder
  reverses h5py's column-major axes for us).
- A :class:`Box` stores **1-based inclusive** intervals exactly as Julia does;
  :pyattr:`Box.slices` converts to **0-based half-open** numpy slices, and that
  conversion is the *only* place index arithmetic happens.
- Boxes/footprints/``bg.S``/``dev``/``meanimg`` live in the **cropped
  analysis-window** frame, not the original recording. :pyattr:`Run.origin` and
  the ``*_global`` helpers map back to the original image.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import cached_property
from glob import glob
from typing import TYPE_CHECKING, Any

import numpy as np

from ._jld2 import read_jld2
from .errors import RunFormatError

if TYPE_CHECKING:  # avoid an import cycle at runtime (cells imports model)
    from .cells import Cell

logger = logging.getLogger("holytrees")

_SUPPORTED_SCHEMA = {None, 1}


# ──────────────────────────────────────────────────────────────────────────────
# Box
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Box:
    """An axis-aligned spatial box, stored as 1-based inclusive intervals.

    Parameters
    ----------
    intervals : tuple of (int, int)
        One ``(lo, hi)`` pair per spatial axis, **1-based inclusive** (Julia
        convention), in ``(Y, X[, Z])`` order.

    Examples
    --------
    >>> b = Box(((1, 120), (182, 301)))
    >>> b.dims
    (120, 120)
    >>> b.slices
    (slice(0, 120, None), slice(181, 301, None))
    """

    intervals: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "intervals", tuple((int(lo), int(hi)) for lo, hi in self.intervals)
        )

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions (2 or 3)."""
        return len(self.intervals)

    @property
    def dims(self) -> tuple[int, ...]:
        """Per-axis size ``hi - lo + 1`` (the shape of a footprint in this box)."""
        return tuple(hi - lo + 1 for lo, hi in self.intervals)

    @property
    def slices(self) -> tuple[slice, ...]:
        """0-based half-open numpy slices that place a footprint into a frame."""
        return tuple(slice(lo - 1, hi) for lo, hi in self.intervals)

    @property
    def ranges_1based(self) -> tuple[tuple[int, int], ...]:
        """The raw 1-based inclusive intervals (Julia parity)."""
        return self.intervals

    @property
    def centroid(self) -> tuple[float, ...]:
        """Box center as **0-based** coordinates (matches numpy arrays / imshow)."""
        return tuple((lo + hi) / 2.0 - 1.0 for lo, hi in self.intervals)

    def shifted(self, origin: tuple[int, ...]) -> Box:
        """Return a copy translated by ``origin`` (0-based offset per axis)."""
        return Box(
            tuple((lo + o, hi + o) for (lo, hi), o in zip(self.intervals, origin))
        )

    def global_slices(self, origin: tuple[int, ...]) -> tuple[slice, ...]:
        """0-based numpy slices in the **original-image** frame, given ``origin``."""
        return tuple(slice(lo - 1 + o, hi + o) for (lo, hi), o in zip(self.intervals, origin))


# ──────────────────────────────────────────────────────────────────────────────
# Tile
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Tile:
    """One tile of a :class:`TileTree`: a low-rank ``S · Tᵀ`` patch.

    Attributes
    ----------
    box : Box
        Location of the tile in the analysis-window frame.
    maxsize : tuple of int
        The tile's maximum allowed spatial size (bookkeeping from the fit).
    S : numpy.ndarray
        Spatial footprints, shape ``box.dims`` (single component) or
        ``box.dims + (ncomponents,)``; ``float32``.
    T : numpy.ndarray
        Temporal traces, shape ``(ntimes,)`` (single component) or
        ``(ntimes, ncomponents)``; ``float32``.
    """

    box: Box
    maxsize: tuple[int, ...]
    S: np.ndarray
    T: np.ndarray

    @property
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions (from the box)."""
        return self.box.ndim

    @property
    def ncomponents(self) -> int:
        """Number of components (cells) packed into this tile."""
        return 1 if self.S.ndim == self.spatial_ndim else int(self.S.shape[-1])

    @property
    def spacesize(self) -> tuple[int, ...]:
        """Spatial shape of a footprint (``== box.dims``)."""
        return self.box.dims

    @property
    def ntimes(self) -> int:
        """Number of time points."""
        return int(self.T.shape[0])

    def component_S(self, c: int) -> np.ndarray:
        """Spatial footprint of component ``c``, shape ``box.dims``."""
        return self.S if self.S.ndim == self.spatial_ndim else self.S[..., c]

    def component_T(self, c: int) -> np.ndarray:
        """Temporal trace of component ``c``, shape ``(ntimes,)``."""
        return self.T if self.T.ndim == 1 else self.T[:, c]

    def array(self) -> np.ndarray:
        """Reconstruct ``S · Tᵀ`` for **this tile only**.

        Returns
        -------
        numpy.ndarray
            Shape ``box.dims + (ntimes,)`` (small — one tile, not the whole
            frame). Summed over components for multi-component tiles.
        """
        if self.S.ndim == self.spatial_ndim:  # single component
            return self.S[..., None] * self.T[None]
        # multi-component: contract the component axis of S with that of T
        return np.tensordot(self.S, self.T, axes=([self.S.ndim - 1], [1]))

    def maxprojectT(self) -> np.ndarray:
        """Max projection over time of :meth:`array`, shape ``box.dims``."""
        return self.array().max(axis=-1)


# ──────────────────────────────────────────────────────────────────────────────
# TileTree
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class TileTree:
    """The set of **valid** tiles produced by the factorization.

    Attributes
    ----------
    tiles : list of Tile
        Only the valid tiles (invalid/merged-away slots are dropped).
    ids : numpy.ndarray
        Original 0-based indices of the valid tiles in the full tile array.
    dims : tuple of int
        Full domain shape ``(spatial..., ntimes)`` in the analysis-window frame.
    """

    tiles: list[Tile]
    ids: np.ndarray
    dims: tuple[int, ...]

    @classmethod
    def from_decoded(cls, d: dict[str, Any]) -> TileTree:
        """Build a :class:`TileTree` from the decoded ``ttree_final`` dict."""
        raw_tiles = d["tiles"]
        valid = np.asarray(d["valid"], dtype=bool)
        dims = tuple(int(x) for x in d["dims"])
        ids = np.flatnonzero(valid)
        tiles = [_tile_from_decoded(raw_tiles[i]) for i in ids]
        return cls(tiles=tiles, ids=ids, dims=dims)

    @property
    def ntiles(self) -> int:
        """Number of valid tiles."""
        return len(self.tiles)

    @property
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions (2 or 3)."""
        return len(self.dims) - 1

    @property
    def spacesize(self) -> tuple[int, ...]:
        """Spatial shape of the full domain, ``(Y, X[, Z])``."""
        return self.dims[:-1]

    @property
    def ntimes(self) -> int:
        """Number of time points."""
        return self.dims[-1]

    def eachtile(self):
        """Iterate ``(id, tile)`` over valid tiles, in canonical order."""
        return zip(self.ids.tolist(), self.tiles)


# ──────────────────────────────────────────────────────────────────────────────
# Background
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Background:
    """The full-frame background / anatomy component.

    Attributes
    ----------
    box : Box
        Covers the whole analysis window.
    S : numpy.ndarray
        Spatial anatomy image, shape ``(Y, X[, Z])``.
    T : numpy.ndarray
        Background time course, shape ``(ntimes,)``.
    maxsize : tuple of int
        Bookkeeping from the fit.
    """

    box: Box
    S: np.ndarray
    T: np.ndarray
    maxsize: tuple[int, ...]

    @classmethod
    def from_decoded(cls, d: dict[str, Any]) -> Background:
        """Build a :class:`Background` from a decoded ``bg``/``bg_absorbed`` dict."""
        return cls(
            box=Box(d["box"]),
            S=np.asarray(d["S"]),
            T=np.asarray(d["T"]),
            maxsize=tuple(int(x) for x in d.get("maxsize", d["S"].shape)),
        )


def _tile_from_decoded(d: dict[str, Any]) -> Tile:
    return Tile(
        box=Box(d["box"]),
        maxsize=tuple(int(x) for x in d.get("maxsize", ())),
        S=np.asarray(d["S"]),
        T=np.asarray(d["T"]),
    )


def _unwrap(x: Any) -> Any:
    """Unwrap a possibly ref-wrapped single value (JLD2 stores some as ``[v]``)."""
    if isinstance(x, list) and len(x) == 1:
        return x[0]
    return x


def _origin(window: dict | None, params: dict | None, ndim: int) -> tuple[int, ...]:
    """0-based offset of the analysis window within the original recording.

    Reads ``yrng``/``xrng`` (and a z selection if present) from ``window`` when
    available (newer pipeline) or from ``params`` (older producer). Defaults to
    zeros when nothing is available.
    """
    src = window or params or {}
    yr = src.get("yrng")
    xr = src.get("xrng")
    y0 = (int(yr[0]) - 1) if yr else 0
    x0 = (int(xr[0]) - 1) if xr else 0
    if ndim >= 3:
        zsel = src.get("zsel") or src.get("zrng")
        z0 = (int(zsel[0]) - 1) if zsel else 0
        return (y0, x0, z0)
    return (y0, x0)


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Run:
    """A fully-loaded tiled-NMF run — the main entry point.

    Build one with :func:`holytrees.load`. The headline accessors are
    :pyattr:`cells` (the canonical, ordered list of :class:`~holytrees.cells.Cell`),
    :pyattr:`traces` (the ``(ncells, ntimes)`` data matrix), :meth:`fill` /
    :meth:`reconstruct_maxproj` (vectorized spatial reconstruction), and the
    ``show_*``/:meth:`project` visualization shortcuts.

    Attributes
    ----------
    ttree : TileTree
        The valid tiles.
    bg : Background
        Full-frame background / anatomy.
    bg_absorbed : Background or None
        Background with absorbed components, when present.
    dev, meanimg : numpy.ndarray or None
        Per-pixel deviation image and mean image (analysis-window frame).
    boxes_active : list of Box
        The active tile boxes at fit time.
    merge_schedule : numpy.ndarray or None
        Structured array of pairwise merges ``(id1, id2, penalty)``.
    window, params, provenance, timings : dict / structures or None
        Run metadata as decoded from the file.
    deviation_method : str or None
    elapsed_s : float or None
    schema_version : int or None
    path : str
        Path the run was loaded from.
    origin : tuple of int
        0-based offset ``(y0, x0[, z0])`` of the window in the original image.
    """

    ttree: TileTree
    bg: Background
    bg_absorbed: Background | None
    dev: np.ndarray | None
    meanimg: np.ndarray | None
    boxes_active: list[Box]
    merge_schedule: np.ndarray | None
    window: dict | None
    params: dict | None
    provenance: dict | None
    timings: Any
    deviation_method: str | None
    elapsed_s: float | None
    schema_version: int | None
    path: str
    origin: tuple[int, ...]

    # ── loading ───────────────────────────────────────────────────────────────
    @classmethod
    def load(cls, path: str) -> Run:
        """Load a run from a ``run_*.jld2`` file or a ``run_<id>/`` directory.

        Parameters
        ----------
        path : str
            Path to a ``.jld2`` run file, or a directory containing exactly one
            (the newest ``run_*.jld2`` is chosen if several are present).

        Returns
        -------
        Run

        Raises
        ------
        holytrees.RunFormatError
            If the path is not an HDF5 run, or required keys (``ttree_final``,
            ``bg``) are missing.
        """
        jld2 = _resolve_path(path)
        try:
            d = read_jld2(jld2)
        except OSError as e:
            raise RunFormatError(f"{jld2!r} is not a readable HDF5/.jld2 file: {e}") from e

        for required in ("ttree_final", "bg"):
            if required not in d:
                raise RunFormatError(
                    f"{jld2!r} is missing required key {required!r}; "
                    "is this a tiled-NMF pipeline run?"
                )

        schema = d.get("schema_version")
        if schema is not None:
            schema = int(schema)
        if schema not in _SUPPORTED_SCHEMA:
            logger.warning(
                "run %r has schema_version=%s, newer than this holytrees supports "
                "(%s); reading best-effort.",
                jld2, schema, sorted(v for v in _SUPPORTED_SCHEMA if v is not None),
            )

        ttree = TileTree.from_decoded(d["ttree_final"])
        bg = Background.from_decoded(d["bg"])
        ba_raw = _unwrap(d.get("bg_absorbed"))
        bg_absorbed = Background.from_decoded(ba_raw) if isinstance(ba_raw, dict) else None

        params = d.get("params")
        window = d.get("window")
        boxes_active = [Box(b) for b in (d.get("boxes_active") or [])]

        return cls(
            ttree=ttree,
            bg=bg,
            bg_absorbed=bg_absorbed,
            dev=_as_array(d.get("dev")),
            meanimg=_as_array(d.get("meanimg")),
            boxes_active=boxes_active,
            merge_schedule=d.get("merge_schedule"),
            window=window,
            params=params,
            provenance=d.get("provenance"),
            timings=d.get("timings"),
            deviation_method=d.get("deviation_method"),
            elapsed_s=_as_float(d.get("elapsed_s")),
            schema_version=schema,
            path=jld2,
            origin=_origin(window, params, ttree.spatial_ndim),
        )

    # ── cheap summary ───────────────────────────────────────────────────────────
    @property
    def ntiles(self) -> int:
        """Number of valid tiles."""
        return self.ttree.ntiles

    @property
    def ntimes(self) -> int:
        """Number of time points."""
        return self.ttree.ntimes

    @property
    def spacesize(self) -> tuple[int, ...]:
        """Spatial shape of the analysis-window frame, ``(Y, X[, Z])``."""
        return self.ttree.spacesize

    @property
    def is3d(self) -> bool:
        """Whether the run is volumetric (3 spatial dimensions)."""
        return self.ttree.spatial_ndim == 3

    @property
    def ncells(self) -> int:
        """Total number of cells (components across all valid tiles)."""
        return len(self.cells)

    # ── canonical cells + data matrix (cached) ──────────────────────────────────
    @cached_property
    def cells(self) -> list[Cell]:
        """The canonical, ordered list of cells (one per tile component).

        This list defines the cell ordering used everywhere else: ``cells[k]``
        corresponds to ``traces[k]`` and to column/row ``k`` of any per-cell array.
        """
        from .cells import eachcell

        return eachcell(self.ttree, origin=self.origin)

    @cached_property
    def traces(self) -> np.ndarray:
        """The ``(ncells, ntimes)`` data matrix, ``float32``; row ``k`` is ``cells[k]``."""
        from .cells import data_matrix

        return data_matrix(self)

    @cached_property
    def footprints(self):
        """Cached packed-COO footprint index over all cells (the fill engine's core)."""
        from .cells import build_footprint_index

        return build_footprint_index(self)

    # ── vectorized spatial mapping (delegates to the cells fill engine) ──────────
    def fill(self, values, *, weighted: bool = False, reduce: str = "overwrite"):
        """Scatter per-cell ``values`` into a spatial frame (single vectorized op).

        See :func:`holytrees.cells.fill` for full semantics.
        """
        from .cells import fill

        return fill(self, values, weighted=weighted, reduce=reduce)

    def fill_subset(self, indices, values, *, weighted: bool = False, reduce: str = "overwrite"):
        """Like :meth:`fill`, but only for the cells in ``indices``."""
        from .cells import fill_subset

        return fill_subset(self, indices, values, weighted=weighted, reduce=reduce)

    def reconstruct_frame(self, t: int) -> np.ndarray:
        """Reconstruct the spatial frame at time ``t`` (``(Y, X[, Z])``)."""
        from .cells import reconstruct_frame

        return reconstruct_frame(self, t)

    def reconstruct_maxproj(self) -> np.ndarray:
        """Max-over-time reconstruction (``(Y, X[, Z])``); never builds a cube."""
        from .cells import reconstruct_maxproj

        return reconstruct_maxproj(self)

    # ── visualization shortcuts (delegate to viz) ───────────────────────────────
    def show_cells(self, *, colorby: str = "random", **kw):
        """Show all cells colored, alpha ~ weight. See :func:`holytrees.viz.show_cells_located`."""
        from .viz import show_cells_located

        return show_cells_located(self, colorby=colorby, **kw)

    def project(self, values, **kw):
        """Paint cells by a scalar (callable ``Cell -> float`` or length-``ncells`` array)."""
        from .viz import project_cells

        return project_cells(self, values, **kw)

    def show_cell(self, k: int, **kw):
        """Show a single cell ``k`` over the anatomy. See :func:`holytrees.viz.show_cell`."""
        from .viz import show_cell

        return show_cell(self, k, **kw)

    def raster(self, **kw):
        """Raster plot of the data matrix. See :func:`holytrees.viz.cell_raster`."""
        from .viz import cell_raster

        return cell_raster(self, **kw)

    # ── connectivity diagnostics (delegate to analysis) ─────────────────────────
    def cell_lobes(self, k: int, *, min_frac: float = 0.0) -> int:
        """Number of connected lobes in cell ``k``'s footprint.

        ``1`` is a normal connected cell; ``>= 2`` flags a disconnected
        (over-merged) footprint. See :func:`holytrees.analysis.cell_lobes`.
        """
        from .analysis import cell_lobes

        return cell_lobes(self.cells[k], min_frac=min_frac)

    def flag_overmerged(self, *, min_lobes: int = 2, min_frac: float = 0.02, indices=None):
        """List cells with disconnected (multi-lobe) footprints, largest first.

        See :func:`holytrees.analysis.flag_overmerged`.
        """
        from .analysis import flag_overmerged

        return flag_overmerged(self, min_lobes=min_lobes, min_frac=min_frac, indices=indices)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        kind = "3D" if self.is3d else "2D"
        return (
            f"<Run {os.path.basename(self.path)!r} {kind} "
            f"ntiles={self.ntiles} ncells={self.ncells} "
            f"spacesize={self.spacesize} ntimes={self.ntimes}>"
        )


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────
def _resolve_path(path: str) -> str:
    """Resolve a run path: a file is returned as-is; a dir -> newest run_*.jld2."""
    if os.path.isdir(path):
        candidates = sorted(glob(os.path.join(path, "run_*.jld2")))
        if not candidates:
            candidates = sorted(glob(os.path.join(path, "*.jld2")))
        if not candidates:
            raise RunFormatError(f"no run_*.jld2 file found in directory {path!r}")
        return candidates[-1]
    if not os.path.exists(path):
        raise RunFormatError(f"path does not exist: {path!r}")
    return path


def _as_array(x: Any) -> np.ndarray | None:
    return np.asarray(x) if isinstance(x, np.ndarray) else None


def _as_float(x: Any) -> float | None:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


__all__ = ["Box", "Tile", "TileTree", "Background", "Run"]
