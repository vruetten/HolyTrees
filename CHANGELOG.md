# Changelog

All notable changes to `holytrees` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning.

## [0.1.0] - 2026-06-15

### Added
- Initial release.
- `holytrees.load` — read a tiled-NMF pipeline run (`run_*.jld2` or a `run_<id>/`
  directory) directly from HDF5, no Julia required.
- Object model: `Box`, `Tile`, `TileTree`, `Background`, `Run`, `Cell`.
- Data matrix (`Run.traces`) and a cached packed-footprint `fill` engine for fast
  spatial reconstruction / per-cell painting.
- Visualization helpers: `show_image`, `show_cells_located`, `project_cells`,
  `show_cell`, `cell_raster`, `plot_merge_penalties`.
