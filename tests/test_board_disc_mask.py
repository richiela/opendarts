"""Board-disc detection-region mask (`opendarts.capture.board_disc`): the
geometry (`board_disc_polygon_px()`/`board_disc_mask()`) and the live
per-camera registry the lifecycle reads its masks from.
"""
from __future__ import annotations


import cv2
import numpy as np
import pytest

from tests.support.synthetic import SyntheticCamera, make_camera_matrix
from opendarts.pipeline import CameraCalibration
from opendarts.capture.board_disc import (
    BOARD_DISC_MASK_MARGIN_MM,
    BOARD_DISC_MASK_RADIUS_MM,
    BOARD_FACE_RADIUS_MM,
    DART_HEIGHT_OFFSET_MM,
    board_disc_mask,
    board_disc_polygon_px,
)
import opendarts.capture.board_disc as tt

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
    """Same fixture as tests/test_engine_apollo_board_roi.py's own
    helper of the same name -- a camera directly above the board center
    looking straight down, no rotation about its own optical axis."""
    camera_matrix = make_camera_matrix(image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT, fov_deg=fov_deg)
    R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    tvec = -R @ np.array([0.0, 0.0, height_mm])
    rvec, _ = cv2.Rodrigues(R)
    return SyntheticCamera(
        camera_matrix=camera_matrix,
        dist_coeffs=np.zeros(5),
        rvec=rvec.reshape(3, 1),
        tvec=tvec.reshape(3, 1),
    )


@pytest.fixture(autouse=True)
def _reset_board_disc_state():
    """Every test starts from a clean registry -- these are module
    globals, and other test files (or a prior test) must never leak a
    mask/warned-set into this file's own assertions."""
    tt.set_calibrated_board_disc_masks(None)
    yield
    tt.set_calibrated_board_disc_masks(None)


# --------------------------------------------------------------------------
# 1. Synthetic, provable geometry (board_disc.py).
# --------------------------------------------------------------------------


def test_board_disc_radius_reuses_established_board_face_constant():
    """BOARD_DISC_MASK_RADIUS_MM must be BOARD_FACE_RADIUS_MM (225.5mm,
    the same constant `derive_top_crop_fraction()`/apollo/board_roi.py
    both already use) plus the margin -- not a fresh, disconnected
    guess."""
    assert BOARD_FACE_RADIUS_MM == pytest.approx(225.5)
    assert BOARD_DISC_MASK_RADIUS_MM == pytest.approx(
        BOARD_FACE_RADIUS_MM + BOARD_DISC_MASK_MARGIN_MM
    )


def test_straight_down_camera_disc_polygon_matches_hand_computed_outer_radius():
    """A straight-down camera's projected board-disc polygon (the convex
    hull of the circle projected at z=0 AND z=+DART_HEIGHT_OFFSET_MM)
    must, for THIS specific pose, equal the z=+DART_HEIGHT_OFFSET_MM
    circle's own hand-derivable pixel radius exactly -- a raised point is
    strictly CLOSER to a camera directly overhead, so it projects to a
    LARGER pixel radius than the z=0 circle at the same board-mm radius;
    the hull of the two must therefore equal the outer (raised) one."""
    height_mm = 1000.0
    fov_deg = 90.0
    cam = _straight_down_camera(height_mm=height_mm, fov_deg=fov_deg)
    calib = _calib_from_synthetic(cam)

    polygon = board_disc_polygon_px(calib, n_samples=360)
    assert polygon is not None
    center = np.array([IMAGE_WIDTH / 2.0, IMAGE_HEIGHT / 2.0])
    px_radii = np.linalg.norm(polygon - center, axis=1)

    f = calib.camera_matrix[0, 0]
    expected_outer_radius_px = f * (BOARD_DISC_MASK_RADIUS_MM / (height_mm - DART_HEIGHT_OFFSET_MM))
    # The hull's own vertices are a SUBSET of the two source point clouds
    # (only the outer circle's own points should survive, for this pose)
    # -- every vertex should sit at (or extremely near) the hand-derived
    # outer radius, not some blend of the two.
    assert np.allclose(px_radii, expected_outer_radius_px, rtol=1e-3)


def test_disc_polygon_returns_none_when_every_z_level_degenerate():
    """If every z-offset's own projection is degenerate (behind the
    camera), the polygon must be None -- never a fabricated/garbage
    shape. A camera looking AWAY from the board (180deg rotated) puts
    every world point behind it."""
    camera_matrix = make_camera_matrix(image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT, fov_deg=90.0)
    R = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])  # looking straight UP, away from board
    tvec = -R @ np.array([0.0, 0.0, 1000.0])
    rvec, _ = cv2.Rodrigues(R)
    cam = SyntheticCamera(camera_matrix=camera_matrix, dist_coeffs=np.zeros(5),
                           rvec=rvec.reshape(3, 1), tvec=tvec.reshape(3, 1))
    calib = _calib_from_synthetic(cam)
    assert board_disc_polygon_px(calib) is None
    assert board_disc_mask(calib, IMAGE_WIDTH, IMAGE_HEIGHT) is None


def test_board_disc_mask_shape_and_roughly_circular_coverage():
    """board_disc_mask() returns a boolean (H, W) array; its own True
    fraction is a real, non-trivial fraction of the frame for a
    plausible board-viewing pose (neither empty nor the whole frame)."""
    cam = _straight_down_camera(height_mm=1000.0, fov_deg=90.0)
    calib = _calib_from_synthetic(cam)
    mask = board_disc_mask(calib, IMAGE_WIDTH, IMAGE_HEIGHT)
    assert mask is not None
    assert mask.shape == (IMAGE_HEIGHT, IMAGE_WIDTH)
    assert mask.dtype == bool
    frac = mask.mean()
    assert 0.01 < frac < 0.99
    # Center of frame (board center) must be inside.
    assert mask[IMAGE_HEIGHT // 2, IMAGE_WIDTH // 2]
    # A far corner must be outside (comfortably off-board at this pose).
    assert not mask[0, 0]


def test_board_disc_mask_accepts_float_dimensions_real_live_bug_20260906():
    """Real live bug, 2026-09-06: `opendarts.live.capture_daemon.
    _negotiated_resolution_for()` deliberately returns
    `tuple[float, float]` (see its own docstring), and the real
    `_bootstrap_calibrations_unlocked()` call site passed that straight
    into `board_disc_mask()` with no cast -- `np.zeros((image_height,
    image_width), ...)` requires ints and raised `TypeError: 'float'
    object cannot be interpreted as an integer` on a live rig during a
    real calibration event, for every camera in turn, breaking this
    function's own "never raises" docstring contract. Must accept
    float dimensions (and non-integer floats specifically, not just
    whole-number floats) and produce the identical mask an int call
    would."""
    cam = _straight_down_camera(height_mm=1000.0, fov_deg=90.0)
    calib = _calib_from_synthetic(cam)

    int_mask = board_disc_mask(calib, IMAGE_WIDTH, IMAGE_HEIGHT)
    float_mask = board_disc_mask(calib, float(IMAGE_WIDTH), float(IMAGE_HEIGHT))
    # A genuinely non-integer float, matching a real negotiated-resolution
    # read more closely than a suspiciously-round x.0 value would.
    non_integer_float_mask = board_disc_mask(calib, IMAGE_WIDTH + 0.4, IMAGE_HEIGHT - 0.4)

    assert int_mask is not None
    assert float_mask is not None
    assert non_integer_float_mask is not None
    assert float_mask.shape == int_mask.shape == (IMAGE_HEIGHT, IMAGE_WIDTH)
    assert non_integer_float_mask.shape == (IMAGE_HEIGHT, IMAGE_WIDTH)
    np.testing.assert_array_equal(float_mask, int_mask)
    np.testing.assert_array_equal(non_integer_float_mask, int_mask)


def test_disc_hull_at_least_as_large_as_flat_z0_circle_alone():
    """The z-offset correction can only ever GROW the region relative to
    a flat z=0-only disc, never shrink it -- real, checkable structural
    property (a convex hull of a superset of points has area >= the hull
    of any subset)."""
    cam = _straight_down_camera(height_mm=1000.0, fov_deg=90.0)
    calib = _calib_from_synthetic(cam)
    mask_with_height = board_disc_mask(calib, IMAGE_WIDTH, IMAGE_HEIGHT)
    mask_flat_only = board_disc_mask(calib, IMAGE_WIDTH, IMAGE_HEIGHT, z_offsets_mm=(0.0,))
    assert mask_with_height is not None and mask_flat_only is not None
    # Every flat-only pixel must also be covered by the height-corrected mask.
    assert np.all(mask_with_height[mask_flat_only])
    assert mask_with_height.sum() >= mask_flat_only.sum()


# --------------------------------------------------------------------------
# 2. The live board-disc mask registry (opendarts.capture.board_disc).
# --------------------------------------------------------------------------


def test_set_get_calibrated_board_disc_masks_round_trip():
    mask0 = np.ones((100, 100), dtype=bool)
    tt.set_calibrated_board_disc_masks({0: mask0})
    got = tt.get_calibrated_board_disc_masks()
    assert set(got) == {0}
    np.testing.assert_array_equal(got[0], mask0)
    # Returned dict is a copy, not the live one.
    got[0] = np.zeros((100, 100), dtype=bool)
    assert tt.get_calibrated_board_disc_masks()[0].all()

    tt.set_calibrated_board_disc_masks(None)
    assert tt.get_calibrated_board_disc_masks() == {}
