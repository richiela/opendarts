"""Athena's own calibrated board-region gate. NOT a port of
`opendarts.engines.apollo.board_roi` (different padding, different test
point, different rejection policy -- see module docstrings for the
difference) -- but the same underlying, obviously-correct CV operation
any calibration-aware ROI check has to do: project the board's known 3D
boundary circle through a camera's solved pose (`cv2.projectPoints`, the
forward-projection inverse of `back_project_ray`'s `cv2.undistortPoints`)
into that camera's pixel space, and test whether a candidate detection
pixel falls near it.

**Why this exists**: the first real Athena run against
`data/archive/clean/` found `crossing_detection.py`'s blob selection
picking the wrong diff component (a bright reflection/light-strip
artifact, an occasional dart-shaped occlusion, etc.) often enough to drag
the corpus-wide mean tip-delta to 60-80mm per camera even though the
MEDIAN was a healthy 5-10mm -- a heavy tail of confidently-wrong reads,
not a uniformly-noisy detector. `opendarts.engines.apollo.tip_detection`'s
own docstring documents the identical failure mode; that engine's fix
(`board_roi.reject_outside_roi()`) is calibration-aware for exactly this
reason, and Athena needs its own version of the same idea for the same
reason -- ROI gating is universal CV practice here, not something to
avoid just because Apollo already has a differently-shaped version of
it.
"""
from __future__ import annotations

import math

import numpy as np

from opendarts.geometry.board import DOUBLE_OUTER_RADIUS_MM
from opendarts.pipeline import CameraCalibration

# How far past the board's own physical double-outer radius the gate's
# projected boundary circle is drawn, before converting to pixels -- a
# real dart's FLIGHT can sit well outside DOUBLE_OUTER_RADIUS_MM even
# when the TIP is a valid on-board (or near-miss/just-outside) score, and
# this gate is checking the CROSSING pixel (near the tip end), not the
# whole blob, so a generous but not unlimited margin. Measured
# (full Athena pipeline, real data/archive/clean/
# corpus, real sector+ring match vs AD): swept 0.95..5.0. A real, if
# mild, plateau from 1.15 to 1.3 (all four measured 72.8%, the
# corpus-wide best found), falling off in both directions (1.0: 71.0%,
# 1.6: 70.4%, 2.0: 65.1%, 3.0: 51.5%, 5.0: 0.6% -- an unbounded ROI stops
# rejecting anything, strictly worse than no gate at all since it also
# disables the re-ranking-by-plausibility benefit). 1.2 chosen as the
# middle of the measured plateau, not its edge.
BOARD_ROI_RADIUS_PAD_FACTOR = 1.2

# Number of points sampled around the projected boundary circle -- plenty
# for a smooth polygon at typical rig camera distances/resolutions;
# not swept independently (secondary to the radius pad above).
_N_BOUNDARY_SAMPLES = 96


def project_board_roi_polygon(calib: CameraCalibration) -> np.ndarray:
    """The board's (padded) double-outer boundary circle, projected into
    this camera's pixel space -- an (N, 2) float32 array, ready for
    `cv2.pointPolygonTest`."""
    import cv2

    radius = DOUBLE_OUTER_RADIUS_MM * BOARD_ROI_RADIUS_PAD_FACTOR
    pts3d = np.array(
        [
            (radius * math.cos(t), radius * math.sin(t), 0.0)
            for t in np.linspace(0.0, 2.0 * math.pi, _N_BOUNDARY_SAMPLES, endpoint=False)
        ],
        dtype=np.float64,
    )
    projected, _ = cv2.projectPoints(
        pts3d, calib.rvec, calib.tvec, calib.camera_matrix, calib.dist_coeffs
    )
    return projected.reshape(-1, 2).astype(np.float32)


def point_in_board_roi(pixel_xy: tuple[float, float], polygon: np.ndarray) -> bool:
    """True if `pixel_xy` is inside (or on) the projected board ROI
    polygon. `cv2.pointPolygonTest` returns a signed distance (positive
    = inside); >= 0 accepts, matching cv2's own convention rather than
    inventing a separate margin -- the padding already lives in
    `BOARD_ROI_RADIUS_PAD_FACTOR` above, not duplicated here."""
    import cv2

    result = cv2.pointPolygonTest(polygon, (float(pixel_xy[0]), float(pixel_xy[1])), False)
    return result >= 0
