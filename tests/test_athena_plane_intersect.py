"""Unit tests for opendarts.engines.athena.plane_intersect -- validates
the new ray/board-plane intersection helper against KNOWN synthetic
ground truth, the same pattern tests/test_triangulation.py already uses
for opendarts.triangulation.rays (back-project a known Z=0 point through a
real camera pose, recover it exactly with no noise), plus the two
explicit "not a valid detection" guards (parallel ray, behind-camera
solution) that mirror opendarts.triangulation.rays.triangulate()'s own
depth guards (see plane_intersect.py's own module docstring).
"""
from __future__ import annotations

import numpy as np
import pytest

from tests.support.synthetic import make_camera_matrix, make_ring_camera
from opendarts.engines.athena.plane_intersect import ray_plane_intersect
from opendarts.triangulation.rays import Ray, back_project_ray


def _project_true_point(camera, point_xyz):
    import cv2

    pt = np.asarray(point_xyz, dtype=np.float64).reshape(1, 1, 3)
    px, _ = cv2.projectPoints(pt, camera.rvec, camera.tvec, camera.camera_matrix, camera.dist_coeffs)
    return tuple(px.reshape(2))


@pytest.mark.parametrize(
    "true_point",
    [
        (0.0, 0.0, 0.0),  # bull
        (50.0, 30.0, 0.0),  # somewhere on the board
        (-100.0, 60.0, 0.0),  # near double ring
        (162.0, -45.0, 0.0),  # near the double ring's outer edge
    ],
)
def test_ray_plane_intersect_recovers_exact_point_no_noise(true_point):
    """A camera looking at a known Z=0 board point: back-project the
    exact pixel that point projects to, intersect with Z=0, and recover
    the original point -- the single-ray analog of
    test_triangulation.py's test_triangulate_recovers_exact_point_no_noise_3_cams,
    but for Athena's own single-ray helper (no multi-ray triangulation
    involved at all)."""
    camera_matrix = make_camera_matrix(fov_deg=90.0)
    cam = make_ring_camera(0, n_cameras=3, camera_matrix=camera_matrix)

    pixel = _project_true_point(cam, true_point)
    ray = back_project_ray(pixel, cam.camera_matrix, cam.dist_coeffs, cam.rvec, cam.tvec, cam=0)

    xy = ray_plane_intersect(ray, z_mm=0.0)
    assert xy is not None
    assert xy[0] == pytest.approx(true_point[0], abs=1e-6)
    assert xy[1] == pytest.approx(true_point[1], abs=1e-6)


def test_ray_plane_intersect_returns_none_for_ray_parallel_to_plane():
    """A ray with zero Z-component in its direction never crosses Z=0
    (unless it's already exactly on the plane, a degenerate case this
    function doesn't need to special-case) -- must return None, not
    divide-by-zero or a garbage point."""
    ray = Ray(origin=np.array([0.0, 0.0, 100.0]), direction=np.array([1.0, 0.0, 0.0]), cam=0)
    assert ray_plane_intersect(ray, z_mm=0.0) is None


def test_ray_plane_intersect_returns_none_when_solution_is_behind_camera():
    """A ray pointing AWAY from the board plane (its forward direction
    only ever increases |Z|, moving further from Z=0, not toward it) --
    the algebraic solution exists but sits at negative t (behind the
    camera). Must return None, not a physically-nonsensical point --
    mirrors opendarts.triangulation.rays.TriangulationResult's own
    per_ray_depth_mm/all_positive_depth guard for the identical class of
    bug."""
    # Camera at Z=100, board at Z=0, but the ray points further AWAY
    # (increasing Z) instead of toward the board -- never reaches Z=0
    # going forward.
    ray = Ray(origin=np.array([0.0, 0.0, 100.0]), direction=np.array([0.0, 0.0, 1.0]), cam=0)
    assert ray_plane_intersect(ray, z_mm=0.0) is None


def test_ray_plane_intersect_respects_custom_z():
    """z_mm is a real parameter, not hardcoded to 0.0 internally --
    verify intersecting a non-default plane works too."""
    ray = Ray(origin=np.array([0.0, 0.0, 100.0]), direction=np.array([0.0, 0.0, -1.0]), cam=0)
    xy = ray_plane_intersect(ray, z_mm=25.0)
    assert xy is not None
    assert xy == pytest.approx((0.0, 0.0), abs=1e-9)
