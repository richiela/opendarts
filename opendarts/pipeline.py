"""End-to-end pipeline, wiring together calibration (opendarts.calibration)
and triangulation (opendarts.triangulation) -- the genuinely shared,
single-correct-answer geometry/calibration machinery every engine (today's
`Apollo`, and any future one) calls into UNCHANGED, so engines can only
ever disagree about detection/scoring ALGORITHM, never about basic camera
geometry. See docs/DESIGN.md for the full design this implements.

**2026-08-12 -- `score_dart()` and its scoring-strategy constants
(`MAX_RAY_DISAGREEMENT_MM`, `MAX_RAY_DISAGREEMENT_MM_FALLBACK_PAIR`, the
2-of-3 RANSAC fallback, the alt-tip-candidate combination search) MOVED
to `opendarts/engines/apollo/scoring.py`.** Per the real architectural
split this project settled on: this module (`opendarts/pipeline.py`) keeps
only calibration-SOLVING (`calibrate_camera()`, wrapping
`opendarts.calibration.pnp.solve_extrinsics()`) and the dataclasses that are
genuinely shared framework currency regardless of which engine is
primary -- `CameraCalibration` (every engine's `score()` signature takes
one, per docs/ENGINES.md's Engine interface), `CalibrationAttempt`
(`calibrate_camera()`'s own return type), and `ScoreResult` (the on-disk
"primary result" schema `opendarts.capture.throw_package.
save_throw_package()` was built around before the engine framework
existed -- every engine's `EngineResult` ultimately gets adapted into one
of these for that schema, see `opendarts.engines.base.
engine_result_to_score_result()` for the generic adapter and
`opendarts.engines.apollo.engine.engine_result_to_score_result()` for
Apollo's own full-fidelity one). The actual SCORING STRATEGY --
`score_dart()`'s ray-triangulation-with-fallback opinion about how to
turn tip pixels into a sector/ring -- is Apollo-specific, not universal
(a future engine is free to score completely differently), so it now
lives with the rest of Apollo's algorithm in
`opendarts/engines/apollo/`. See that package's own module docstring for
the full move record (what came from where, and why) and
`opendarts/engines/apollo/scoring.py`'s own docstring for the real
incident history (`throw_1786580447119`) the alt-tip-candidate
combination search and the 2-of-3 RANSAC fallback were both built from --
preserved verbatim there, not summarized away.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from opendarts.calibration.pnp import PnpResult, solve_extrinsics
from opendarts.triangulation.rays import TriangulationResult


@dataclass
class CameraCalibration:
    """A single camera's known intrinsics + solved extrinsics -- the
    output of Phase 3 (opendarts.calibration.pnp), input to Phase 4
    (opendarts.triangulation.rays)."""

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rvec: np.ndarray
    tvec: np.ndarray
    pnp_result: PnpResult | None = None  # kept for diagnostics/confidence
    # Mirrors pnp_result.landmark_spread_ok -- pulled up to a first-class
    # field (not just buried in pnp_result) because
    # opendarts.engines.apollo.scoring.score_dart() actually consumes it,
    # see opendarts/calibration/pnp.py's landmark_hull_area_fraction. None only
    # if pnp_result itself is missing (shouldn't happen via
    # calibrate_camera(), only possible if a CameraCalibration is
    # hand-constructed without going through it).
    landmark_spread_ok: bool | None = None


@dataclass
class CalibrationAttempt:
    """2026-08-12: calibrate_camera() used to return
    bare None on failure, silently discarding PnpResult.reason -- an
    asymmetry with opendarts.engines.apollo.scoring.score_dart(), which
    correctly preserves
    TriangulationResult.reason on failure. A caller "expected to handle
    some cameras failing to calibrate" (the old docstring's own words)
    got nothing to handle WITH. This wrapper fixes that; calibrate_camera
    now returns one of these instead of CameraCalibration | None."""

    ok: bool
    calibration: CameraCalibration | None
    pnp_result: PnpResult | None
    reason: str = ""


# Correlated calibration bias -- score_dart() (opendarts/engines/
# apollo/scoring.py, moved from here 2026-08-12) rejects outright if
# ANY contributing camera's calibration had a poorly-spread landmark
# quad, since ray agreement (checked post-triangulation) structurally
# cannot catch CORRELATED calibration bias across cameras. See that
# module's own MAX_RAY_DISAGREEMENT_MM docstring, and
# opendarts/calibration/pnp.py's landmark_hull_area_fraction /
# MIN_LANDMARK_HULL_AREA_FRACTION, for the full measured basis --
# unchanged by the move, just relocated with the scoring strategy that
# consumes it.


@dataclass
class ScoreResult:
    ok: bool
    sector: str | None
    ring: str | None
    board_xy_mm: tuple[float, float] | None
    triangulation: TriangulationResult | None
    n_cameras_used: int = 0
    reason: str = ""
    max_ray_disagreement_mm: float | None = None
    # Populated whenever triangulation was attempted (whether accepted or
    # rejected) -- the specific camera indices whose rays were actually
    # used for the returned `triangulation`/`board_xy_mm`. On a
    # successful 2-of-3 RANSAC fallback this is the winning PAIR, not
    # every camera passed in -- n_cameras_used already reflects len(this).
    cameras_used: tuple[int, ...] | None = None
    # Set only when the 2-of-3 fallback fired and succeeded: the camera
    # excluded as the apparent outlier. None in every other case
    # (including outright rejection -- an excluded camera is only
    # meaningful once a result was actually accepted from the fallback).
    outlier_camera: int | None = None
    # Added 2026-08-12 -- real incident, throw
    # throw_1786580447119 (see opendarts/engines/apollo/tip_detection.py's
    # module docstring dated entry for the full write-up: two darts'
    # shafts merged into one diff blob, cam1/cam2 each picked the wrong
    # end). Populated only when `alt_tip_pixels` was passed to
    # opendarts.engines.apollo.scoring.score_dart() AND the WINNING
    # combination actually used at least one camera's alt candidate
    # instead of its primary tip_px -- the camera indices that did, or
    # None otherwise (no alt available, or the winning combination never
    # needed one -- including every outright-rejected result, since
    # "winning" only means something once a result was actually
    # accepted).
    alt_candidates_used: tuple[int, ...] | None = None


def calibrate_camera(
    object_points_mm: np.ndarray,
    image_points_px: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_width: float | None = None,
    image_height: float | None = None,
) -> CalibrationAttempt:
    """Wraps opendarts.calibration.pnp.solve_extrinsics into a
    CalibrationAttempt ready for opendarts.engines.apollo.scoring.
    score_dart() (via .calibration) or for
    inspecting why calibration failed (via .reason) -- callers building a
    multi-camera rig should expect some cameras to fail calibration and
    handle that explicitly, not have it hidden inside an exception
    handler or silently discarded (see CalibrationAttempt's docstring for
    why this used to return a bare None on failure, since fixed).

    image_width/image_height: forwarded to solve_extrinsics() for the
    landmark-spread conditioning check.
    A 2026-08-12 review found this pipeline-level entry point had
    NO way to pass them at all -- solve_extrinsics() silently inferred
    image dimensions as 2*camera_matrix[0,2]/2*camera_matrix[1,2]
    (assumes a centered principal point), and a real camera with an
    off-center principal point could silently defeat the landmark-spread
    gate (demonstrated: an extreme off-center cx flips landmark_spread_ok
    from False to True on the known-bad CLUSTERED_QUAD). Not exploitable
    today (every intrinsics producer in this repo centers the principal
    point), but it sat directly in the path of real
    intrinsics sourcing, with no way to route around the inference once
    real, possibly-off-center intrinsics exist. Pass explicit dimensions
    here once real image dimensions are known, rather than relying on the
    inference -- the inference remains only as a fallback for callers
    (like the existing tests) that don't have real dimensions to pass.
    """
    result = solve_extrinsics(
        object_points_mm,
        image_points_px,
        camera_matrix,
        dist_coeffs,
        image_width=image_width,
        image_height=image_height,
    )
    if not result.ok:
        return CalibrationAttempt(
            ok=False, calibration=None, pnp_result=result, reason=result.reason
        )
    calibration = CameraCalibration(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvec=result.rvec,
        tvec=result.tvec,
        pnp_result=result,
        landmark_spread_ok=result.landmark_spread_ok,
    )
    return CalibrationAttempt(ok=True, calibration=calibration, pnp_result=result)
