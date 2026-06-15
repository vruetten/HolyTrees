"""Self-contained tests against the committed toy run fixture.

The fixture ``tests/data/run_20260614_182728.jld2`` is a small (~1.9 MB) real
run, so these tests need no external paths. The larger f338 run is exercised by
``test_f338`` only when its absolute path happens to be present (auto-skipped).
"""

from __future__ import annotations

import os
import pickle

import matplotlib

matplotlib.use("Agg")  # headless rendering for viz smoke tests

import numpy as np
import pytest

import holytrees as ht
from holytrees.model import Background, Box, Run, TileTree

HERE = os.path.dirname(__file__)
FIXTURE = os.path.join(HERE, "data", "run_20260614_182728.jld2")
F338 = (
    "/nrs/ahrens/Virginia_nrs/wVT/221124_f338_ubi_gCaMP7f_bactin_mCherry_CAAX_"
    "8505_7dpf_hypoxia_t23/exp0/nmf_holy/run_20260614_175659.jld2"
)


@pytest.fixture(scope="module")
def run() -> Run:
    return ht.load(FIXTURE)


# ── loading & summary ─────────────────────────────────────────────────────────
def test_load_summary(run):
    assert run.ntiles == 5
    assert run.ncells == 5
    assert run.ntimes == 8
    assert run.spacesize == (301, 301)
    assert run.is3d is False
    assert run.origin == (699, 999)


def test_load_directory():
    """Passing the containing directory resolves to the newest run_*.jld2."""
    r = ht.load(os.path.dirname(FIXTURE))
    assert r.ntiles == 5


def test_bad_path_raises():
    with pytest.raises(ht.RunFormatError):
        ht.load(os.path.join(HERE, "does_not_exist.jld2"))


def test_picklable(run):
    r2 = pickle.loads(pickle.dumps(run))
    assert r2.ncells == run.ncells
    assert np.array_equal(r2.traces, run.traces)


# ── cells & data matrix ─────────────────────────────────────────────────────────
def test_eachcell_count(run):
    assert len(run.cells) == run.ncells
    assert [c.index for c in run.cells] == list(range(run.ncells))


def test_data_matrix_shape_and_alignment(run):
    assert run.traces.shape == (run.ncells, run.ntimes)
    assert run.traces.dtype == np.float32
    for k, c in enumerate(run.cells):
        assert np.allclose(run.traces[k], c.T)


def test_footprint_shape_matches_box(run):
    """Axis-order guard: a footprint's spatial shape must equal its box dims."""
    for c in run.cells:
        assert c.S.shape == c.box.dims


def test_box_slices_roundtrip():
    b = Box(((1, 120), (182, 301)))
    assert b.dims == (120, 120)
    assert b.slices == (slice(0, 120), slice(181, 301))
    assert b.ranges_1based == ((1, 120), (182, 301))


# ── reconstruction / fill ───────────────────────────────────────────────────────
def test_reconstruct_frame_matches_reference(run):
    t = 3
    frame = run.reconstruct_frame(t)
    ref = np.zeros(run.spacesize, dtype=np.float64)
    for c in run.cells:
        ref[c.box.slices] += c.S * c.T[t]
    assert frame.shape == run.spacesize
    assert np.abs(frame - ref).max() < 1e-3


def test_fill_additive_overlap():
    """Two overlapping cells must scatter-add correctly (no lost contributions)."""
    tt = TileTree(
        tiles=[
            _tile(((1, 2), (1, 2)), np.ones((2, 2), np.float32)),
            _tile(((2, 3), (2, 3)), np.ones((2, 2), np.float32)),
        ],
        ids=np.array([0, 1]),
        dims=(4, 4, 1),
    )
    run = _bare_run(tt, spacesize=(4, 4))
    out = run.fill(np.array([1.0, 1.0]), weighted=True, reduce="add")
    assert out[1, 1] == 2.0  # the shared corner gets both
    assert out[0, 0] == 1.0
    assert out[2, 2] == 1.0
    assert out.sum() == 8.0


def test_fill_stacked_matches_single(run):
    feats = np.random.default_rng(0).random((3, run.ncells)).astype(np.float32)
    vols = run.fill(feats, weighted=True, reduce="add")
    assert vols.shape == (3, *run.spacesize)
    for i in range(3):
        assert np.allclose(vols[i], run.fill(feats[i], weighted=True, reduce="add"), atol=1e-4)


def test_fill_length_check(run):
    with pytest.raises(ValueError):
        run.fill(np.ones(run.ncells + 1))


def test_maxproj_shape(run):
    assert run.reconstruct_maxproj().shape == run.spacesize


# ── painting / coloring ─────────────────────────────────────────────────────────
def test_paint_alpha_bounded(run):
    p = ht.paint_cells(run, values=lambda c: c.T.var(), colorby="random")
    assert set(p) == {"value", "alpha", "hue"}
    assert p["value"].shape == run.spacesize
    assert 0.0 <= p["alpha"].max() <= 1.0


# ── coordinate mapping ──────────────────────────────────────────────────────────
def test_global_mapping(run):
    c = run.cells[0]
    g = c.centroid_global
    for a in range(len(g)):
        assert g[a] == pytest.approx(c.centroid[a] + run.origin[a])
    assert c.box_global.intervals[0][0] == c.box.intervals[0][0] + run.origin[0]


# ── empty run ────────────────────────────────────────────────────────────────────
def test_empty_run():
    tt = TileTree(tiles=[], ids=np.array([], dtype=int), dims=(10, 10, 4))
    run = _bare_run(tt, spacesize=(10, 10))
    assert run.ncells == 0
    assert run.cells == []
    assert run.traces.shape == (0, 4)
    assert np.all(run.reconstruct_frame(0) == 0)
    assert np.all(run.reconstruct_maxproj() == 0)
    p = ht.paint_cells(run, colorby="random")
    assert np.all(p["alpha"] == 0)
    # viz must degrade gracefully
    ht.show_cells_located(run)


# ── viz smoke ────────────────────────────────────────────────────────────────────
def test_viz_smoke(run, tmp_path):
    ht.show_image(run.bg.S, save=str(tmp_path / "img.png"))
    run.show_cells(save=str(tmp_path / "cells.png"))
    run.project(lambda c: c.T.var(), label="var", save=str(tmp_path / "proj.png"))
    run.show_cell(0, save=str(tmp_path / "cell.png"))
    run.raster(save=str(tmp_path / "raster.png"))
    ht.plot_merge_penalties(run, save=str(tmp_path / "merge.png"))
    for name in ("img", "cells", "proj", "cell", "raster", "merge"):
        assert (tmp_path / f"{name}.png").exists()


# ── optional large run ──────────────────────────────────────────────────────────
@pytest.mark.skipif(not os.path.exists(F338), reason="f338 run not available")
def test_f338():
    r = ht.load(F338)
    assert r.ncells > 0
    assert r.bg.S.shape == r.spacesize
    assert r.reconstruct_maxproj().shape == r.spacesize


# ── helpers ──────────────────────────────────────────────────────────────────────
def _tile(intervals, S):
    from holytrees.model import Tile

    return Tile(box=Box(intervals), maxsize=S.shape, S=S, T=np.ones(1, np.float32))


def _bare_run(tt: TileTree, spacesize) -> Run:
    bg = Background(
        box=Box(tuple((1, s) for s in spacesize)),
        S=np.zeros(spacesize, np.float32),
        T=np.zeros(tt.ntimes, np.float32),
        maxsize=spacesize,
    )
    return Run(
        ttree=tt, bg=bg, bg_absorbed=None, dev=None, meanimg=None, boxes_active=[],
        merge_schedule=None, window=None, params=None, provenance=None, timings=None,
        deviation_method=None, elapsed_s=None, schema_version=1, path="<synthetic>",
        origin=tuple(0 for _ in spacesize),
    )
