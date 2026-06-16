"""Interactive walkthrough of a tiled-NMF run with ``holytrees``.

Run it section-by-section (the ``# %%`` cell markers work in VS Code / Cursor /
Jupyter), or top-to-bottom as a script::

    python demo.py
    HOLYTREES_RUN=/path/to/run_<id>.jld2 python demo.py   # point at a specific run

Each section explains what it shows; the headline figures are saved to
``figures/`` (a small curated set is committed so the README gallery survives a
fresh clone).
"""

# %% imports & run selection
import os
import time

import matplotlib


def _in_ipython() -> bool:
    """True when running inside an IPython/Jupyter kernel (incl. Cursor's interactive window)."""
    try:
        from IPython import get_ipython

        return get_ipython() is not None
    except Exception:
        return False


# In Jupyter/interactive sessions, leave the backend alone so the inline backend
# renders figures in the window (run `%matplotlib inline` once if needed). As a
# plain script with no display (e.g. over SSH), fall back to headless Agg so the
# figures/ PNGs still get written. Never force a GUI backend like TkAgg remotely.
if _in_ipython():
    from IPython import get_ipython

    get_ipython().run_line_magic("matplotlib", "inline")  # render figures in the window
elif not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")  # headless: still writes the figures/ PNGs

import numpy as np
import matplotlib.pyplot as pl
import holytrees as ht
from importlib import reload
import holytrees.viz
reload(ht)
reload(holytrees.viz)

HERE = os.path.dirname(os.path.abspath(__file__))


# The whole-body zebrafish run this demo was built around; override with $HOLYTREES_RUN,
# else fall back to the committed toy fixture so the demo always runs.
DEFAULT_RUN = (
    "/nrs/ahrens/Virginia_nrs/wVT/221124_f338_ubi_gCaMP7f_bactin_mCherry_CAAX_"
    "8505_7dpf_hypoxia_t23/exp0/nmf_holy/run_20260614_175659.jld2"
)
RUN_PATH = os.environ.get("HOLYTREES_RUN") or DEFAULT_RUN
if not os.path.exists(RUN_PATH):
    RUN_PATH = os.path.join(HERE, "tests", "data", "run_20260614_182728.jld2")
print(f"loading {RUN_PATH}")

PYTHON_PLOT_DIR = "/nrs/ahrens/Virginia_nrs/wVT/221124_f338_ubi_gCaMP7f_bactin_mCherry_CAAX_8505_7dpf_hypoxia_t23/exp0/nmf_holy/plots_python/"
FIGDIR = os.path.join(PYTHON_PLOT_DIR, "figures")
os.makedirs(FIGDIR, exist_ok=True)


# %% load & summarize
_t0 = time.perf_counter()
run = ht.load(RUN_PATH)
_load_s = time.perf_counter() - _t0
_mb = os.path.getsize(RUN_PATH) / 1e6
print(f"ht.load took {_load_s:.2f} s  ({_mb:.0f} MB on disk)")
#%%
print(run)
print(f"  tiles      : {run.ntiles}")
print(f"  cells      : {run.ncells}")
print(f"  time points: {run.ntimes}")
print(f"  space      : {run.spacesize}  (3D={run.is3d})")
print(f"  window orig: {run.origin}")
if run.provenance:
    print(f"  produced by: julia {run.provenance.get('julia_version')} "
          f"@ {run.provenance.get('git_commit', '')[:10]}")
    print(f"  pkg versions: {run.provenance.get('pkg_versions')}")
#%% # accessible attributes
run.bg.S # background image
run.cells                     # list[Cell], length = ncells
run.cells[0]                  # a single Cell
run.cells[0].S                # footprint, shape == box.dims: (Y, X) [or (Y, X, Z) if 3D], float32
# np.sqrt(np.sum(run.cells[0].S**2))
run.cells[0].T                # trace, shape (ntimes,), float32
run.cells[0].box              # Box: ndim (2 or 3) 1-based inclusive (lo, hi) intervals
run.cells[0].tileid           # int scalar (original tile index)
run.cells[0].comp             # int scalar (component index within the tile)
run.cells[0].centroid         # tuple of len ndim: (y, x) [or (y, x, z)], 0-based window coords
run.cells[0].centroid_global  # tuple of len ndim: same, mapped to original-image coords


run.traces # data matrix
run.ntiles
run.ncells # number of cells
run.ntimes # number of time points
run.spacesize # size of the space   
run.is3d
run.origin
run.provenance
run.provenance.get('julia_version')
run.provenance.get('git_commit')
run.provenance.get('pkg_versions')


# %% the anatomy / background image
# bg.S is the full-frame background (anatomy) the cells sit on top of.
ht.show_image(run.bg.S, title="background / anatomy (bg.S)",
              save=os.path.join(FIGDIR, "bg_anatomy.png"))


# %% all cells located (random color, alpha proportional to spatial weight)
run.show_cells(colorby="random", save=os.path.join(FIGDIR, "cells_located.png"))


# %% the data matrix
# run.traces is the (ncells x ntimes) matrix; each row is one cell's trace.
%load_ext autoreload
%autoreload 2
reload(holytrees.viz)
reload(ht)
traces = run.traces
traces_norm = (traces-np.mean(traces, axis=1, keepdims=True)) / np.sqrt(np.sum(traces**2, axis=1, keepdims=True))
lo, hi = np.nanpercentile(traces_norm, [5, 95])
print("data matrix:", traces.shape, traces.dtype)
run.raster(M=traces_norm, vmin=lo, vmax=hi, save=os.path.join(FIGDIR, "cell_raster.png"))
#%%

# rank cells by footprint size (number of nonzero pixels in the spatial map)
cell_sizes = np.array([c.area for c in run.cells])
order = np.argsort(cell_sizes)[::-1]  # largest footprint first
print("footprint size: min", cell_sizes.min(), "median", int(np.median(cell_sizes)),
      "max", cell_sizes.max())
print("5 largest cells (index, size):",
      [(int(i), int(cell_sizes[i])) for i in order[:5]])
#%%
for i in order[:5]:
    print(i)
    run.show_cell(i)
    

# %% a single, highly active cell over the anatomy (with its trace)
k = int(np.argmax([c.amplitude for c in run.cells]))
print(f"most active cell: index {k}, tile {run.cells[k].tileid}, "
      f"centroid(global) {tuple(round(x, 1) for x in run.cells[k].centroid_global)}")

run.show_cell(k, save=os.path.join(FIGDIR, "single_cell.png"))




# %% paint cells by a scalar — here, trace variance
# Pass a callable Cell -> float, or a precomputed length-ncells array.
run.project(lambda c: c.T.var(), label="trace variance",
            save=os.path.join(FIGDIR, "cells_var.png"))


# %% spatial reconstruction (amplitude-weighted max projection)
# One vectorized scatter over all cells; never materializes a (Y, X, T) cube.
proj = run.reconstruct_maxproj()
ht.show_image(proj, cmap="inferno", title="reconstruction (max projection)",
              save=os.path.join(FIGDIR, "reconstruction.png"))
print("reconstruction:", proj.shape, "max", float(proj.max()))


# %% (bonus) get any single time frame back, in space
frame0 = run.reconstruct_frame(0)
print("frame 0:", frame0.shape)

print("done — figures written to", FIGDIR)
