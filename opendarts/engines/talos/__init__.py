"""Talos -- 2D observation is the outward silhouette edge at the tip,
then the same multi-view ray triangulation as score_dart(). As a proper
subpackage (not a single file) since 2026-08-13.

**What lives here, and why (2026-08-13 restructure)**: same discipline
as `opendarts.engines.apollo`'s own 2026-08-12 subpackage move -- this
was a pure MOVE of the real, working 807-line `talos.py` module (every
tuned constant/threshold preserved exactly, nothing reimplemented), not
a rewrite. `opendarts/pipeline.py` and `opendarts/calibration/`/
`opendarts/triangulation/`/`opendarts/geometry/` stay genuinely shared --
Talos calls into them unchanged. Blob/tip pixel detection is
deliberately NOT shared: `opendarts/detection/` no longer exists (moved
into `opendarts.engines.apollo` on 2026-08-12, kept evolving there
since), so Talos depends on its own frozen local copy,
`blob_detection.py` -- a verbatim snapshot of that machinery taken when
it still lived at `opendarts.detection.tip_detection`, kept intentionally
decoupled from Apollo's copy so it can't silently drift out from under
Talos's proven 88.8%/95.3% numbers (see `blob_detection.py`'s own
"Frozen-copy note" for the full reasoning). Everything in THIS package
is Talos's own opinion about how to turn raw images into a score:

- `shaft_line.py` -- `fit_shaft_line_px()` and its private helpers
  (`_shaft_line_through_tip`, `_direction_from_shaft_pixels`): finds the
  dart's blob (reusing this package's own frozen `blob_detection.py`
  copy, NOT a shared/current module) and fits a 2D shaft line through
  the detected tip. `MIN_SHAFT_SPAN_PX` / `_SHAFT_TIPWARD_FRACTION`.
- `plane_geometry.py` -- pure 3D line/plane math with no pixel selection
  opinion of its own: `plane_from_image_line()`, `_line_from_planes()`,
  `_hit_z0()`, `_residuals_at()`, `_geom_for()`,
  `intersect_planes_with_board()`, and the single-pixel ∩ Z=0 helper
  `_pixel_board_xy()` (shared by `engine.py` and `consensus.py`).
  `CONCURRENCE_GATE_MM` / `MAX_PLAUSIBLE_TILT_DEG`.
- `consensus.py` -- Talos's own opinion about WHICH pixel is the
  observation and how to combine cameras that disagree:
  `_outward_edge_pixel()` (the outward-cap-centroid 2D pick),
  `_pair_if_sector_compromise()` (the 2-of-3 sector-compromise pair
  logic), `_rescue_outside_from_centerlines()` (result-outside rescue
  from >=2 on-board centerlines), and
  `_lock_unanimous_centerline_sector()` (3/3 CL sector lock),
  `_lock_cap_walked_radial()` (onboard caps walked a radial wire;
  CLs + dart axis did not), `_lock_shaft_snap_consensus()` (snap mean,
  CL majority, and dart axis agree).
  `OUTWARD_EDGE_WINDOW_PX` / `OUTWARD_CAP_MM`.
- `engine.py` -- `TalosEngine`, the registry entry (`name = "Talos"`),
  and the `score()` orchestration function that calls the three pieces
  above in sequence (per-camera shaft line -> outward-edge pixel pick ->
  ray triangulation via the shared `opendarts.pipeline.score_dart()`, with
  the sector-compromise pair and line∩plane fallback), plus the small
  `EngineResult`-shaping helpers (`_engine_result_from_score_dart`,
  `_plane_miss`). Optional `prior_board_xy_mm` (2026-08-17) erases a
  ghost leftover-dart tip on one poisoned camera; see `prior_dart.py`.
- `prior_dart.py` -- visit-prior board-XY lookup + gated erase of a
  motion-diff blob that is actually last throw's dart (T15, +1/-0).

Zero-behavior-change proof: this engine was run against the full real
`data/archive/clean/` (and `important/`) corpus and confirmed the same
150/169 = 88.8% sector+ring match against Talos's own AD-referenced
ground truth as the pre-restructure flat module, plus a field-for-field
identical-output
comparison against a captured pre-refactor baseline.

Public re-export: `TalosEngine` (so `from opendarts.engines.talos import
TalosEngine` -- e.g. `opendarts/engines/registry.py` -- keeps working
exactly as it did when this was a single module, not a package). Every
other name above (constants, private helpers) is imported from its own
submodule directly by tests/callers that need it -- same convention
`opendarts.engines.apollo`/`opendarts.engines.athena` already use.
"""
from __future__ import annotations

from opendarts.engines.talos.engine import TalosEngine

__all__ = ["TalosEngine"]
