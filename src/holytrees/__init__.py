"""holytrees — pure-Python inspection of Tim Holy's tiled-NMF pipeline output.

Read a saved run (``run_*.jld2``) with no Julia, pull out cells (footprint +
trace), build the cells x time **data matrix**, **reconstruct** data spatially,
and **visualize** cells.

Quickstart
----------
>>> import holytrees as ht
>>> run = ht.load("path/to/run_20260614_182728.jld2")  # or a run_<id>/ dir
>>> run.ntiles, run.ncells, run.ntimes               # doctest: +SKIP
>>> traces = run.traces                               # (ncells, ntimes) data matrix
>>> frame = run.reconstruct_maxproj()                 # (Y, X[, Z]) spatial reconstruction
>>> fig, ax = run.show_cells(colorby="random")        # doctest: +SKIP

Conventions
-----------
- Spatial arrays are ``(Y, X[, Z])`` (Julia logical order).
- ``run.cells[k]`` and ``run.traces[k]`` are aligned (canonical ordering).
- Coordinates live in the cropped analysis-window frame; use ``Cell.centroid_global``
  / ``Cell.box_global`` (with ``run.origin``) to map back to the original image.
"""

from __future__ import annotations

import logging

from .analysis import cell_lobes, flag_overmerged, label_components
from .cells import (
    Cell,
    data_matrix,
    eachcell,
    fill,
    fill_subset,
    paint_cells,
    reconstruct_frame,
    reconstruct_maxproj,
)
from .errors import HolyTreesError, RunFormatError
from .model import Background, Box, Run, Tile, TileTree
from .viz import (
    cell_raster,
    plot_merge_penalties,
    project_cells,
    show_cell,
    show_cells_located,
    show_image,
)

__version__ = "0.1.0"

# Library code should not configure logging handlers; attach a no-op so importing
# holytrees never emits "No handlers found" warnings, but leave config to the app.
logging.getLogger("holytrees").addHandler(logging.NullHandler())


def load(path: str) -> Run:
    """Load a tiled-NMF run from a ``run_*.jld2`` file or a ``run_<id>/`` directory.

    Parameters
    ----------
    path : str
        Path to a ``.jld2`` run file, or a directory containing one.

    Returns
    -------
    Run
        A fully-decoded, picklable run object (no open file handles).

    Raises
    ------
    holytrees.RunFormatError
        If the path is not a readable pipeline run.

    Examples
    --------
    >>> import holytrees as ht
    >>> run = ht.load("tests/data/run_20260614_182728.jld2")  # doctest: +SKIP
    """
    return Run.load(path)


__all__ = [
    "load",
    "Run",
    "Box",
    "Tile",
    "TileTree",
    "Background",
    "Cell",
    "eachcell",
    "data_matrix",
    "fill",
    "fill_subset",
    "reconstruct_frame",
    "reconstruct_maxproj",
    "paint_cells",
    "cell_lobes",
    "flag_overmerged",
    "label_components",
    "show_image",
    "show_cells_located",
    "project_cells",
    "show_cell",
    "cell_raster",
    "plot_merge_penalties",
    "HolyTreesError",
    "RunFormatError",
    "__version__",
]
