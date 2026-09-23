"""Apollo -- today's real, production scoring algorithm, as a proper
subpackage (not a single file) since 2026-08-12.

**What lives here, and why (2026-08-12 refactor)**: per the real
architectural split agreed for this project, `opendarts/pipeline.py` and
`opendarts/calibration/`/`opendarts/triangulation/`/`opendarts/geometry/` hold
ONLY the genuinely shared, single-correct-answer camera geometry/
calibration machinery every engine (today's `Apollo`, and any future
one) must call into UNCHANGED -- there is only one real camera rig and
one real board, so engines may only ever disagree about detection/
scoring ALGORITHM, never about basic geometry. Everything in THIS
package, by contrast, is Apollo's own opinion about how to turn raw
images into a score -- one algorithm among potentially several, not a
universal truth -- and previously lived scattered across generic-
sounding, shared-looking module paths that wrongly implied otherwise.
Moved here, verbatim (every real tuned constant/threshold preserved
exactly, this was a MOVE not a reimplementation, proven at the time by
a real corpus-wide before/after comparison):

- `tip_detection.py` -- ALL of `opendarts/detection/tip_detection.py`'s
  former contents: `detect_tip()` and every private helper
  (`_component_stats_in_bbox`, `_diff_mask`, `_locate_tip_in_component`,
  `_find_companion_end`, `_local_bg_texture`) and every measured constant
  (`DIFF_THRESHOLD`, `MIN_ELONGATION_RATIO`, `OPEN_KERNEL_PX`,
  `DILATE_KERNEL_PX`, `TOP_K_AREA_CANDIDATES`, `N_TIP_POINTS_AVERAGED`,
  `WIDTH_AMBIGUITY_RATIO_MIN`, `TEXTURE_TIEBREAK_RATIO_MIN`,
  `COMPANION_MAX_GAP_PX`, `COMPANION_MAX_PERP_PX`, and the rest --
  see that module's own docstring for the full, real incident history
  each one is tuned from).
- `board_roi.py` -- ALL of `opendarts/detection/board_roi.py`'s former
  contents: `reject_outside_roi()`, `detect_tip_in_roi()`, and the
  board-ROI-mask-building machinery.
- `scoring.py` -- `opendarts/pipeline.py`'s former `score_dart()` and
  everything specific to Apollo's scoring STRATEGY (as opposed to
  calibration-solving, which stayed in `opendarts/pipeline.py`): the 2-of-3
  RANSAC fallback-pair logic, the alt-tip-pixels brute-force combination
  search, and the `MAX_RAY_DISAGREEMENT_MM`/
  `MAX_RAY_DISAGREEMENT_MM_FALLBACK_PAIR` scoring-strategy constants.
  `opendarts.pipeline.calibrate_camera()` (calibration-SOLVING, wraps
  `opendarts.calibration.pnp.solve_extrinsics()`) stayed in
  `opendarts/pipeline.py`, genuinely shared -- only the SCORING strategy
  moved.
- `evaluate_tip_detection.py` / `evaluate_board_roi.py` -- the offline
  real-corpus measurement tooling for the two modules above, moved
  alongside what they measure (formerly `opendarts/detection/
  evaluate_tip_detection.py` / `evaluate_board_roi.py`).
- `engine.py` -- `ApolloEngine`, the registry entry (`name =
  "Apollo"`) implementing the `opendarts.engines.base.Engine` protocol by
  calling the three pieces above in sequence (detect per camera -> ROI
  gate -> score), plus the two adapters between `opendarts.pipeline.
  ScoreResult` (the on-disk/legacy shape) and `opendarts.engines.base.
  EngineResult` (the generic engine-framework shape): forward
  (`score_result_to_engine_result`) and, added 2026-08-12 alongside the
  live/replay bypass removal below, the full-fidelity inverse
  (`engine_result_to_score_result`) -- see `engine.py`'s own module
  docstring for why a second, Apollo-specific inverse was needed
  instead of reusing `opendarts.engines.base.engine_result_to_score_result`'s
  generic (deliberately lossy for an arbitrary engine) one.

**Also 2026-08-12 -- the live/replay direct-call bypass removed.**
Before this date, `opendarts/live/capture_daemon.py`'s
`handle_ready_to_capture()` special-cased `primary="Apollo"` to call
`detect_tip()`/`reject_outside_roi()`/`score_dart()` directly rather than
through `ApolloEngine` -- a deliberate, documented "byte-identical by
construction" caution kept only until the wrapping was actually proven
identical. `opendarts/capture/replay.py`'s `replay_throw()` had the exact
same kind of bypass (duplicating the same three-call sequence rather than
going through `replay_throw_with_engine()`). Both are now proven
byte-identical (full real corpus, zero mismatches), so both call sites
now route through
`get_engine("Apollo").score()` like any other primary engine -- no
special-cased bypass left anywhere in the live or offline call paths.
`opendarts/capture/rescore_all.py` had no SEPARATE bypass of its own (it
always went through `opendarts.capture.replay`'s functions), so it needed no
direct change -- it inherits the fix via `replay.py`.

Public re-export: `ApolloEngine` (so `from opendarts.engines.apollo
import ApolloEngine` -- e.g. `opendarts/engines/registry.py` -- keeps
working exactly as it did when this was a single module, not a package).
"""
from __future__ import annotations

from opendarts.engines.apollo.engine import ApolloEngine, engine_result_to_score_result, score_result_to_engine_result

__all__ = ["ApolloEngine", "engine_result_to_score_result", "score_result_to_engine_result"]
