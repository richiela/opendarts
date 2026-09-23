"""Unit tests for opendarts.engines.athena.board_gate -- verifies the
projected board-ROI polygon actually contains points near the board
center and excludes points far outside it, using the same synthetic
camera infrastructure tests/test_triangulation.py and
tests/test_athena_plane_intersect.py already use, rather than only
being covered indirectly via the real-corpus accuracy test.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.support.synthetic import make_camera_matrix, make_ring_camera
from opendarts.engines.athena.board_gate import point_in_board_roi, project_board_roi_polygon
from opendarts.pipeline import CameraCalibration


def _calibration_for_synthetic_camera(cam) -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=cam.camera_matrix,
        dist_coeffs=cam.dist_coeffs,
        rvec=cam.rvec,
        tvec=cam.tvec,
        pnp_result=None,
        landmark_spread_ok=True,
    )


def _project_true_point(cam, point_xyz):
    import cv2

    pt = np.asarray(point_xyz, dtype=np.float64).reshape(1, 1, 3)
    px, _ = cv2.projectPoints(pt, cam.rvec, cam.tvec, cam.camera_matrix, cam.dist_coeffs)
    return tuple(px.reshape(2))


def test_board_center_pixel_is_inside_the_roi():
    camera_matrix = make_camera_matrix(fov_deg=90.0)
    cam = make_ring_camera(0, n_cameras=3, camera_matrix=camera_matrix)
    calib = _calibration_for_synthetic_camera(cam)

    polygon = project_board_roi_polygon(calib)
    bull_pixel = _project_true_point(cam, (0.0, 0.0, 0.0))
    assert point_in_board_roi(bull_pixel, polygon)


@pytest.mark.parametrize(
    "board_point",
    [
        (50.0, 30.0, 0.0),
        (-100.0, 60.0, 0.0),
        (150.0, -80.0, 0.0),  # within the double ring
    ],
)
def test_on_board_pixels_are_inside_the_roi(board_point):
    camera_matrix = make_camera_matrix(fov_deg=90.0)
    cam = make_ring_camera(0, n_cameras=3, camera_matrix=camera_matrix)
    calib = _calibration_for_synthetic_camera(cam)

    polygon = project_board_roi_polygon(calib)
    pixel = _project_true_point(cam, board_point)
    assert point_in_board_roi(pixel, polygon)


def test_a_pixel_far_from_the_projected_board_is_excluded():
    """A pixel nowhere near the board's own projected boundary (e.g. a
    corner of a large image, far from the board's real on-screen
    location) must be rejected -- the actual thing this gate exists to
    catch (a wrong-blob-entirely false positive, see board_gate.py's own
    module docstring)."""
    camera_matrix = make_camera_matrix(fov_deg=90.0, image_width=1280, image_height=720)
    cam = make_ring_camera(0, n_cameras=3, camera_matrix=camera_matrix)
    calib = _calibration_for_synthetic_camera(cam)

    polygon = project_board_roi_polygon(calib)
    # A far corner of a much larger image than the board could plausibly
    # occupy -- guaranteed outside any reasonable padded ROI.
    far_pixel = (-5000.0, -5000.0)
    assert not point_in_board_roi(far_pixel, polygon)
