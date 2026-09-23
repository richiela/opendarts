"""Tests for opendarts/engines/apollo/board_roi.py -- moved here
2026-08-12 from opendarts/detection/board_roi.py (see that package's own
module docstring for the full move record) -- the calibrated board
region-of-interest (ROI) mask that was named long before it was built. See that module's own docstring for the
real-corpus measurement that decided which of its two integration
approaches (`reject_outside_roi()`/`detect_tip_in_roi()`) is
recommended.

Synthetic and provable throughout: known camera pose, known 3D point,
prove the forward-projection math (`board_roi_polygon_px`) is correct
against a hand-computed ground truth AND against
`tests.support.synthetic.project_points` (an independent,
already-tested `cv2.projectPoints` call used throughout this project's
other calibration tests) -- two different checks, not the same
computation asserted against itself.

A second track of real-corpus plausibility checks lived here until
2026-09-17, pinning measured numbers against reference case images that
no longer exist anywhere; they went with the images.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tests.support.synthetic import (
    SyntheticCamera,
    make_camera_matrix,
    make_ring_camera,
    project_points,
)
from opendarts.engines.apollo.board_roi import (
    BOARD_FACE_RADIUS_MM,
    ROI_MARGIN_MM,
    ROI_RADIUS_MM,
    board_roi_mask,
    board_roi_polygon_px,
    board_roi_world_points_mm,
    detect_tip_in_roi,
    point_in_board_roi,
    reject_outside_roi,
)
from opendarts.engines.apollo.tip_detection import TipDetectionResult, detect_tip
from opendarts.geometry.board import polar_to_xy_mm
from opendarts.pipeline import CameraCalibration

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720


def _calib_from_synthetic(cam: SyntheticCamera) -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=cam.camera_matrix,
        dist_coeffs=cam.dist_coeffs,
        rvec=cam.rvec,
        tvec=cam.tvec,
        pnp_result=None,
        landmark_spread_ok=True,
    )


def _straight_down_camera(height_mm: float = 1000.0, fov_deg: float = 90.0) -> SyntheticCamera:
    """A camera directly above the board center, looking straight down,
    with no rotation about its own optical axis -- image X aligned with
    world X, image Y aligned with world -Y (OpenCV image Y grows
    downward; world +Y is "up"/12 o'clock in board.py's convention). The
    simplest possible non-degenerate pose to hand-verify projection math
    against."""
    import cv2

    camera_matrix = make_camera_matrix(image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT, fov_deg=fov_deg)
    # Camera center at (0, 0, height_mm), looking straight down (-Z).
    # World X -> image X, world Y -> image -Y (down), world Z -> image Z
    # (away from camera, i.e. -world Z direction since camera looks down).
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ]
    )
    tvec = -R @ np.array([0.0, 0.0, height_mm])
    rvec, _ = cv2.Rodrigues(R)
    return SyntheticCamera(
        camera_matrix=camera_matrix,
        dist_coeffs=np.zeros(5),
        rvec=rvec.reshape(3, 1),
        tvec=tvec.reshape(3, 1),
    )


# --------------------------------------------------------------------------
# 1. Synthetic, provable.
# --------------------------------------------------------------------------

def test_board_roi_world_points_mm_matches_board_polar_convention():
    """board_roi_world_points_mm must use the SAME clockwise-from-+Y
    angle convention as opendarts.geometry.board.polar_to_xy_mm -- pinned
    directly against it, not just internally self-consistent."""
    pts = board_roi_world_points_mm(radius_mm=170.0, n_samples=36)
    for i, angle_deg in enumerate(np.linspace(0.0, 360.0, 36, endpoint=False)):
        expected_x, expected_y = polar_to_xy_mm(170.0, angle_deg)
        assert pts[i, 0] == pytest.approx(expected_x, abs=1e-9)
        assert pts[i, 1] == pytest.approx(expected_y, abs=1e-9)
        assert pts[i, 2] == 0.0


def test_board_roi_world_points_mm_all_at_given_radius():
    pts = board_roi_world_points_mm(radius_mm=255.5, n_samples=100)
    radii = np.hypot(pts[:, 0], pts[:, 1])
    np.testing.assert_allclose(radii, 255.5, atol=1e-9)


def test_straight_down_camera_projects_board_center_to_principal_point():
    """Ground-truth hand check: a camera directly above the board center
    looking straight down must project the board center (0,0,0) to
    exactly the image's principal point (cx, cy) -- no calibration
    solver involved, pure forward-projection geometry."""
    cam = _straight_down_camera(height_mm=1000.0)
    calib = _calib_from_synthetic(cam)

    import cv2

    img_pts, _ = cv2.projectPoints(
        np.array([[0.0, 0.0, 0.0]]), calib.rvec, calib.tvec, calib.camera_matrix, calib.dist_coeffs
    )
    px = img_pts.reshape(2)
    assert px[0] == pytest.approx(IMAGE_WIDTH / 2.0, abs=1e-6)
    assert px[1] == pytest.approx(IMAGE_HEIGHT / 2.0, abs=1e-6)


def test_straight_down_camera_roi_polygon_matches_hand_computed_radius_px():
    """A circle of known board-mm radius, viewed by a straight-down
    camera at known height, projects to a circle of known PIXEL radius:
    r_px = f * (r_mm / height_mm) for a pinhole camera with no
    distortion, since every boundary point is at the SAME depth
    (height_mm) under this specific pose. Hand-derived expected value,
    not just "matches cv2.projectPoints" (which is what board_roi.py
    itself calls) -- this is an independent formula."""
    height_mm = 1000.0
    fov_deg = 90.0
    radius_mm = 255.5
    cam = _straight_down_camera(height_mm=height_mm, fov_deg=fov_deg)
    calib = _calib_from_synthetic(cam)

    roi = board_roi_polygon_px(calib, radius_mm=radius_mm, n_samples=180)
    assert roi.ok
    center = np.array([IMAGE_WIDTH / 2.0, IMAGE_HEIGHT / 2.0])
    px_radii = np.linalg.norm(roi.polygon_px - center, axis=1)

    fx = calib.camera_matrix[0, 0]
    expected_r_px = fx * (radius_mm / height_mm)
    np.testing.assert_allclose(px_radii, expected_r_px, rtol=1e-6)


@pytest.mark.parametrize("cam_index", [0, 1, 2])
def test_board_roi_polygon_matches_independent_project_points(cam_index):
    """Cross-check board_roi_polygon_px's projection against
    tests.support.synthetic.project_points -- an independently
    written (if ultimately also cv2.projectPoints-based) helper already
    used and trusted throughout this project's other calibration tests
    (test_pnp_calibration.py). Uses this project's real 3-camera-ring
    geometry (make_ring_camera), not just the straight-down toy case."""
    camera_matrix = make_camera_matrix(image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT, fov_deg=90.0)
    cam = make_ring_camera(cam_index, n_cameras=3, camera_matrix=camera_matrix)
    calib = _calib_from_synthetic(cam)

    roi = board_roi_polygon_px(calib, radius_mm=255.5, n_samples=72)
    assert roi.ok

    expected = project_points(cam, board_roi_world_points_mm(radius_mm=255.5, n_samples=72))
    np.testing.assert_allclose(roi.polygon_px, expected, atol=1e-6)


def test_board_roi_polygon_ok_false_when_boundary_behind_camera():
    """A camera positioned INSIDE the ROI radius, facing outward (away
    from the board center rather than at it), should have its ROI
    boundary land behind it -- board_roi_polygon_px must detect this
    rather than returning a silently-nonsensical polygon (same class of
    guard as opendarts.triangulation.rays.TriangulationResult.
    all_positive_depth)."""
    import cv2

    camera_matrix = make_camera_matrix(image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT, fov_deg=90.0)
    # Camera at board center, facing +X (world) -- the entire ROI circle
    # (radius 255.5mm, centered at world origin, Z=0) surrounds this
    # camera; large portions of it are behind whichever way it faces.
    R = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    tvec = -R @ np.array([0.0, 0.0, 0.0])
    rvec, _ = cv2.Rodrigues(R)
    calib = CameraCalibration(
        camera_matrix=camera_matrix,
        dist_coeffs=np.zeros(5),
        rvec=rvec.reshape(3, 1),
        tvec=tvec.reshape(3, 1),
        landmark_spread_ok=True,
    )
    roi = board_roi_polygon_px(calib, radius_mm=255.5, n_samples=72)
    assert not roi.ok
    assert roi.polygon_px is None
    assert "behind the camera" in roi.reason.lower()


def test_point_in_board_roi_center_inside_far_point_outside():
    cam = _straight_down_camera(height_mm=1000.0)
    calib = _calib_from_synthetic(cam)
    roi = board_roi_polygon_px(calib, radius_mm=255.5, n_samples=144)
    assert roi.ok

    center_px = (IMAGE_WIDTH / 2.0, IMAGE_HEIGHT / 2.0)
    assert point_in_board_roi(center_px, roi.polygon_px) is True

    far_px = (-5000.0, -5000.0)
    assert point_in_board_roi(far_px, roi.polygon_px) is False


def test_board_roi_mask_agrees_with_point_in_board_roi():
    """The rasterized mask and the float polygon test are two
    independently-computed approximations of the same ROI (see
    detect_tip_in_roi's docstring on why this matters) -- they must
    agree almost everywhere, checked directly rather than assumed."""
    cam = _straight_down_camera(height_mm=1000.0)
    calib = _calib_from_synthetic(cam)
    roi = board_roi_polygon_px(calib, radius_mm=255.5, n_samples=144)
    assert roi.ok
    mask = board_roi_mask(roi.polygon_px, IMAGE_WIDTH, IMAGE_HEIGHT)
    assert mask.shape == (IMAGE_HEIGHT, IMAGE_WIDTH)
    assert mask.dtype == bool

    rng = np.random.default_rng(0)
    disagreements = 0
    n_checked = 200
    for _ in range(n_checked):
        x = rng.uniform(0, IMAGE_WIDTH - 1)
        y = rng.uniform(0, IMAGE_HEIGHT - 1)
        poly_says_inside = point_in_board_roi((x, y), roi.polygon_px)
        mask_says_inside = bool(mask[int(round(y)), int(round(x))])
        if poly_says_inside != mask_says_inside:
            disagreements += 1
    # Disagreement should only ever happen within ~1px of the true
    # boundary (rasterization rounding) -- essentially never for 200
    # uniformly random points against a boundary curve of near-zero area
    # measure. Not asserting exactly 0 to avoid a flaky test on a
    # boundary-adjacent random draw, but this must stay a rare event.
    assert disagreements <= 2, f"{disagreements}/{n_checked} mask/polygon disagreements -- too many"


def test_default_roi_radius_is_board_face_plus_margin():
    assert ROI_RADIUS_MM == pytest.approx(BOARD_FACE_RADIUS_MM + ROI_MARGIN_MM)


# --------------------------------------------------------------------------
# reject_outside_roi() -- the recommended integration approach.
# --------------------------------------------------------------------------

def test_reject_outside_roi_keeps_inside_result_unchanged():
    cam = _straight_down_camera(height_mm=1000.0)
    calib = _calib_from_synthetic(cam)
    result = TipDetectionResult(ok=True, tip_px=(IMAGE_WIDTH / 2.0, IMAGE_HEIGHT / 2.0), reason="ok")
    filtered = reject_outside_roi(result, calib, radius_mm=255.5)
    assert filtered.ok is True
    assert filtered.tip_px == result.tip_px
    assert filtered.diagnostics["board_roi_rejected"] is False


def test_reject_outside_roi_rejects_outside_result_with_clear_reason():
    cam = _straight_down_camera(height_mm=1000.0)
    calib = _calib_from_synthetic(cam)
    far_px = (-5000.0, -5000.0)
    result = TipDetectionResult(ok=True, tip_px=far_px, reason="ok")
    filtered = reject_outside_roi(result, calib, radius_mm=255.5)
    assert filtered.ok is False
    assert filtered.tip_px == far_px  # preserved for diagnostics, even though rejected
    assert "outside the calibrated board ROI" in filtered.reason
    assert filtered.diagnostics["board_roi_rejected"] is True


def test_reject_outside_roi_passes_through_already_failed_result():
    """A result that already failed (ok=False, no tip_px) must pass
    through unchanged -- there is nothing to ROI-check."""
    cam = _straight_down_camera(height_mm=1000.0)
    calib = _calib_from_synthetic(cam)
    result = TipDetectionResult(ok=False, tip_px=None, reason="no diff components found")
    filtered = reject_outside_roi(result, calib, radius_mm=255.5)
    assert filtered is result or (filtered.ok is False and filtered.tip_px is None and filtered.reason == result.reason)


def test_reject_outside_roi_fails_open_on_degenerate_calibration():
    """If the ROI itself can't be computed, the original result must be
    returned UNCHANGED (fail-open) rather than spuriously rejected for a
    reason unrelated to the tip pixel itself -- see reject_outside_roi's
    docstring."""
    import cv2

    camera_matrix = make_camera_matrix(image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT, fov_deg=90.0)
    R = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    tvec = -R @ np.array([0.0, 0.0, 0.0])
    rvec, _ = cv2.Rodrigues(R)
    degenerate_calib = CameraCalibration(
        camera_matrix=camera_matrix,
        dist_coeffs=np.zeros(5),
        rvec=rvec.reshape(3, 1),
        tvec=tvec.reshape(3, 1),
        landmark_spread_ok=True,
    )
    result = TipDetectionResult(ok=True, tip_px=(640.0, 360.0), reason="ok")
    filtered = reject_outside_roi(result, degenerate_calib, radius_mm=255.5)
    assert filtered.ok is True
    assert filtered.tip_px == result.tip_px
    assert filtered.diagnostics["board_roi_checked"] is False


def test_reject_outside_roi_preserves_alt_tip_px_when_inside():
    """Real bug found and fixed 2026-08-12 (see this module's own
    docstring): every TipDetectionResult reconstruction
    in reject_outside_roi()/detect_tip_in_roi() used to omit `alt_tip_px`,
    silently dropping it back to None even on the accepted (inside-ROI)
    path -- which would have quietly broken the cross-camera alt-
    candidate mechanism (opendarts.engines.apollo.scoring.score_dart's alt_tip_pixels) the
    moment this module got wired into a live call site. Direct regression
    test: a result with a real alt_tip_px set must still carry it after
    passing through the accept path."""
    cam = _straight_down_camera(height_mm=1000.0)
    calib = _calib_from_synthetic(cam)
    result = TipDetectionResult(
        ok=True,
        tip_px=(IMAGE_WIDTH / 2.0, IMAGE_HEIGHT / 2.0),
        reason="ok",
        alt_tip_px=(IMAGE_WIDTH / 2.0 + 3.0, IMAGE_HEIGHT / 2.0 - 4.0),
    )
    filtered = reject_outside_roi(result, calib, radius_mm=255.5)
    assert filtered.ok is True
    assert filtered.alt_tip_px == result.alt_tip_px


def test_reject_outside_roi_preserves_alt_tip_px_when_rejected():
    """Same regression as above, on the rejected (outside-ROI) path --
    alt_tip_px is not functionally used once ok=False (score_dart never
    sees it, see opendarts.live.capture_daemon.handle_ready_to_capture()),
    but must still round-trip faithfully rather than be silently
    discarded, for any future caller that inspects a rejected result's
    diagnostics."""
    cam = _straight_down_camera(height_mm=1000.0)
    calib = _calib_from_synthetic(cam)
    far_px = (-5000.0, -5000.0)
    result = TipDetectionResult(
        ok=True, tip_px=far_px, reason="ok", alt_tip_px=(-4990.0, -4995.0)
    )
    filtered = reject_outside_roi(result, calib, radius_mm=255.5)
    assert filtered.ok is False
    assert filtered.alt_tip_px == result.alt_tip_px


def test_reject_outside_roi_preserves_alt_tip_px_on_degenerate_calibration_failopen():
    """Same regression, on the fail-open (ROI itself uncomputable) path."""
    import cv2

    camera_matrix = make_camera_matrix(image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT, fov_deg=90.0)
    R = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    tvec = -R @ np.array([0.0, 0.0, 0.0])
    rvec, _ = cv2.Rodrigues(R)
    degenerate_calib = CameraCalibration(
        camera_matrix=camera_matrix,
        dist_coeffs=np.zeros(5),
        rvec=rvec.reshape(3, 1),
        tvec=tvec.reshape(3, 1),
        landmark_spread_ok=True,
    )
    result = TipDetectionResult(
        ok=True, tip_px=(640.0, 360.0), reason="ok", alt_tip_px=(645.0, 358.0)
    )
    filtered = reject_outside_roi(result, degenerate_calib, radius_mm=255.5)
    assert filtered.diagnostics["board_roi_checked"] is False
    assert filtered.alt_tip_px == result.alt_tip_px


# --------------------------------------------------------------------------
# detect_tip_in_roi() -- alternative approach, synthetic behavior checks.
# --------------------------------------------------------------------------

def _make_dart_blob_images(tip_xy, shaft_len_px=60, angle_deg=20.0):
    """A minimal synthetic "dart-shaped" diff blob: an elongated,
    tapered shape drawn into an otherwise-identical bg/frame pair so
    only this shape shows up in the diff -- same technique this
    project's other detection tests use to build controlled synthetic
    diff scenes."""
    import cv2

    bg = np.full((IMAGE_HEIGHT, IMAGE_WIDTH, 3), 60, dtype=np.uint8)
    frame = bg.copy()
    tip = np.array(tip_xy, dtype=np.float64)
    direction = np.array([math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))])
    tail = tip + direction * shaft_len_px
    cv2.line(frame, tuple(tip.astype(int)), tuple(tail.astype(int)), (200, 200, 200), thickness=3)
    cv2.circle(frame, tuple(tail.astype(int)), 10, (200, 200, 200), thickness=-1)  # fletching-ish blob
    return bg, frame


def test_detect_tip_in_roi_passes_through_a_clean_in_roi_detection():
    cam = _straight_down_camera(height_mm=1000.0)
    calib = _calib_from_synthetic(cam)
    tip_xy = (IMAGE_WIDTH / 2.0, IMAGE_HEIGHT / 2.0)  # board center -- always inside ROI
    bg, frame = _make_dart_blob_images(tip_xy)

    baseline = detect_tip(bg, frame)
    assert baseline.ok

    filtered = detect_tip_in_roi(bg, frame, calib, radius_mm=255.5)
    assert filtered.ok is True
    assert filtered.diagnostics["board_roi_masking_applied"] is True
    # Masking a fully-inside blob should not move the result at all.
    assert filtered.tip_px == pytest.approx(baseline.tip_px, abs=1e-6)


def test_detect_tip_in_roi_masks_out_a_decoy_blob_entirely_outside_roi():
    """A synthetic 'reflection' blob placed far outside the ROI (e.g.
    near the top-left corner, well beyond any board-mm radius at this
    camera height) should be invisible after masking -- detect_tip_in_roi
    on an image containing ONLY that decoy (no real in-ROI blob) must
    report failure (no diff components survive masking), not silently
    return the decoy's location."""
    cam = _straight_down_camera(height_mm=1000.0)
    calib = _calib_from_synthetic(cam)
    decoy_xy = (5.0, 5.0)  # corner of the frame -- far outside a 255.5mm-radius ROI at 1000mm height
    bg, frame = _make_dart_blob_images(decoy_xy, angle_deg=200.0)

    baseline = detect_tip(bg, frame)
    assert baseline.ok  # unmodified detect_tip happily finds the decoy

    filtered = detect_tip_in_roi(bg, frame, calib, radius_mm=255.5)
    assert filtered.ok is False
    assert filtered.diagnostics["board_roi_masking_applied"] is True


def test_detect_tip_in_roi_fails_open_on_degenerate_calibration():
    import cv2

    camera_matrix = make_camera_matrix(image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT, fov_deg=90.0)
    R = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    tvec = -R @ np.array([0.0, 0.0, 0.0])
    rvec, _ = cv2.Rodrigues(R)
    degenerate_calib = CameraCalibration(
        camera_matrix=camera_matrix,
        dist_coeffs=np.zeros(5),
        rvec=rvec.reshape(3, 1),
        tvec=tvec.reshape(3, 1),
        landmark_spread_ok=True,
    )
    tip_xy = (IMAGE_WIDTH / 2.0, IMAGE_HEIGHT / 2.0)
    bg, frame = _make_dart_blob_images(tip_xy)
    baseline = detect_tip(bg, frame)
    filtered = detect_tip_in_roi(bg, frame, degenerate_calib, radius_mm=255.5)
    assert filtered.ok == baseline.ok
    assert filtered.tip_px == pytest.approx(baseline.tip_px, abs=1e-6)
    assert filtered.diagnostics["board_roi_masking_applied"] is False


# --------------------------------------------------------------------------
# far_end_px threading + the board_roi_far_end_inside diagnostic
# (2026-08-14). The gate REPORTS whether the component's opposite end is
# on-board; it deliberately does not act on it, because whether promoting
# it is safe depends on how many other cameras survived -- knowledge only
# ApolloEngine.score() has. See that method for the measured evidence.
# --------------------------------------------------------------------------

def test_reject_outside_roi_preserves_far_end_px_on_every_path():
    """far_end_px must round-trip through all three paths (accept,
    reject, fail-open) -- the exact regression class that already bit
    alt_tip_px once (see the alt_tip_px tests above)."""
    import cv2

    cam = _straight_down_camera(height_mm=1000.0)
    calib = _calib_from_synthetic(cam)
    centre = (IMAGE_WIDTH / 2.0, IMAGE_HEIGHT / 2.0)
    far_end = (IMAGE_WIDTH / 2.0 + 20.0, IMAGE_HEIGHT / 2.0 + 10.0)

    accepted = reject_outside_roi(
        TipDetectionResult(ok=True, tip_px=centre, reason="ok", far_end_px=far_end),
        calib, radius_mm=255.5,
    )
    assert accepted.ok is True and accepted.far_end_px == far_end

    rejected = reject_outside_roi(
        TipDetectionResult(ok=True, tip_px=(-5000.0, -5000.0), reason="ok", far_end_px=far_end),
        calib, radius_mm=255.5,
    )
    assert rejected.ok is False and rejected.far_end_px == far_end

    # Fail-open: a degenerate calibration whose ROI cannot be computed.
    degenerate = CameraCalibration(
        camera_matrix=cam.camera_matrix,
        dist_coeffs=cam.dist_coeffs,
        rvec=cv2.Rodrigues(np.eye(3))[0].reshape(3, 1),
        tvec=np.array([[0.0], [0.0], [0.0]]),
        pnp_result=None,
        landmark_spread_ok=True,
    )
    failopen = reject_outside_roi(
        TipDetectionResult(ok=True, tip_px=centre, reason="ok", far_end_px=far_end),
        degenerate, radius_mm=255.5,
    )
    assert failopen.far_end_px == far_end


def test_rejected_result_reports_far_end_inside_when_the_other_end_is_on_board():
    """The real failure this exists for: a dart high on the board with
    its flight angled up out of the board face. The detector confidently
    takes the flight end (off-board, so the gate rejects the camera), but
    the tip end is squarely on the board -- and no alt_tip_px was ever
    offered, because the width comparison was not ambiguous."""
    cam = _straight_down_camera(height_mm=1000.0)
    calib = _calib_from_synthetic(cam)
    rejected = reject_outside_roi(
        TipDetectionResult(
            ok=True,
            tip_px=(-5000.0, -5000.0),
            reason="ok",
            alt_tip_px=None,
            far_end_px=(IMAGE_WIDTH / 2.0, IMAGE_HEIGHT / 2.0),
        ),
        calib, radius_mm=255.5,
    )
    assert rejected.ok is False
    assert rejected.diagnostics["board_roi_far_end_inside"] is True
    assert rejected.diagnostics["board_roi_alt_inside"] is None


def test_rejected_result_reports_far_end_inside_false_when_both_ends_are_off_board():
    cam = _straight_down_camera(height_mm=1000.0)
    calib = _calib_from_synthetic(cam)
    rejected = reject_outside_roi(
        TipDetectionResult(
            ok=True, tip_px=(-5000.0, -5000.0), reason="ok", far_end_px=(-4900.0, -4900.0)
        ),
        calib, radius_mm=255.5,
    )
    assert rejected.ok is False
    assert rejected.diagnostics["board_roi_far_end_inside"] is False


def test_rejected_result_reports_far_end_inside_none_when_no_far_end_available():
    cam = _straight_down_camera(height_mm=1000.0)
    calib = _calib_from_synthetic(cam)
    rejected = reject_outside_roi(
        TipDetectionResult(ok=True, tip_px=(-5000.0, -5000.0), reason="ok", far_end_px=None),
        calib, radius_mm=255.5,
    )
    assert rejected.diagnostics["board_roi_far_end_inside"] is None


def test_gate_does_not_itself_promote_the_far_end():
    """Explicit non-behavior pin: the per-camera gate stays context-free.
    Promotion is the engine's call (it depends on how many OTHER cameras
    survived) -- if that ever moves in here, this test should fail and
    make whoever moved it justify the ungated version, which measured
    +1/-2 on the real corpus versus +1/-0 gated."""
    cam = _straight_down_camera(height_mm=1000.0)
    calib = _calib_from_synthetic(cam)
    rejected = reject_outside_roi(
        TipDetectionResult(
            ok=True,
            tip_px=(-5000.0, -5000.0),
            reason="ok",
            far_end_px=(IMAGE_WIDTH / 2.0, IMAGE_HEIGHT / 2.0),
        ),
        calib, radius_mm=255.5,
    )
    assert rejected.ok is False, "the gate must not accept on the strength of a far end alone"
    assert rejected.tip_px == (-5000.0, -5000.0), "the gate must not rewrite tip_px"
