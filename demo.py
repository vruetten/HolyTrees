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

import matplotlib

if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")  # headless-safe; remove for an interactive backend

import numpy as np

import holytrees as ht

HERE = os.path.dirname(os.path.abspath(__file__))
FIGDIR = os.path.join(HERE, "figures")
os.makedirs(FIGDIR, exist_ok=True)

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


# %% load & summarize
run = ht.load(RUN_PATH)
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


# %% the anatomy / background image
# bg.S is the full-frame background (anatomy) the cells sit on top of.
ht.show_image(run.bg.S, title="background / anatomy (bg.S)",
              save=os.path.join(FIGDIR, "bg_anatomy.png"))


# %% all cells located (random color, alpha proportional to spatial weight)
run.show_cells(colorby="random", save=os.path.join(FIGDIR, "cells_located.png"))


# %% the data matrix
# run.traces is the (ncells x ntimes) matrix; each row is one cell's trace.
traces = run.traces
print("data matrix:", traces.shape, traces.dtype)
run.raster(save=os.path.join(FIGDIR, "cell_raster.png"))


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
