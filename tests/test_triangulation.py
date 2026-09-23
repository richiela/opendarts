"""Validates ray back-projection + triangulation against KNOWN synthetic
ground truth -- the other half (with PnP calibration) of docs/DESIGN.md
Phase 3-4's "prove it against synthetic data" requirement."""
from __future__ import annotations

import numpy as np
import pytest

from tests.support.synthetic import make_camera_matrix, make_ring_camera
from opendarts.triangulation.rays import (
    CONDITION_NUMBER_WARNING_THRESHOLD,
    Ray,
    back_project_ray,
    triangulate,
)


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
    ],
)
def test_triangulate_recovers_exact_point_no_noise_3_cams(true_point):
    camera_matrix = make_camera_matrix(fov_deg=90.0)
    cams = [make_ring_camera(i, n_cameras=3, camera_matrix=camera_matrix) for i in range(3)]

    rays = []
    for i, cam in enumerate(cams):
        pixel = _project_true_point(cam, true_point)
        ray = back_project_ray(pixel, cam.camera_matrix, cam.dist_coeffs, cam.rvec, cam.tvec, cam=i)
        rays.append(ray)

    result = triangulate(rays)
    assert result.ok, result.reason
    assert result.n_rays == 3
    recovered = result.point_xyz
    true = np.array(true_point)
    assert np.linalg.norm(recovered - true) < 1e-6, (recovered, true)
    assert result.plane_discrepancy_mm < 1e-6
    assert result.board_plane_xy == pytest.approx((true_point[0], true_point[1]), abs=1e-6)


def test_triangulate_2_of_3_cameras_still_works_noiseless():
    """The routine occlusion case -- one
    camera dropped, triangulation must still work with the other 2.
    Noiseless exact-recovery check only -- see the noisy variant below
    for why zero noise alone can't validate this case (any 2 non-
    parallel lines through a shared point intersect exactly regardless
    of whether 2-ray triangulation has some real noise-sensitivity bug)."""
    camera_matrix = make_camera_matrix(fov_deg=90.0)
    cams = [make_ring_camera(i, n_cameras=3, camera_matrix=camera_matrix) for i in range(3)]
    true_point = (20.0, -15.0, 0.0)

    rays = []
    for i in (0, 2):
        cam = cams[i]
        pixel = _project_true_point(cam, true_point)
        ray = back_project_ray(pixel, cam.camera_matrix, cam.dist_coeffs, cam.rvec, cam.tvec, cam=i)
        rays.append(ray)

    result = triangulate(rays)
    assert result.ok, result.reason
    assert result.n_rays == 2
    recovered = result.point_xyz
    assert np.linalg.norm(recovered - np.array(true_point)) < 1e-6


def test_triangulate_2_of_3_has_measurably_worse_accuracy_than_3_of_3():
    """2-of-3 must be handled "gracefully," but graceful != equally
    accurate, and the
    noiseless test above cannot distinguish the two (2 rays recover a
    point exactly with zero noise regardless of any real 2-ray-specific
    degradation). This test measures the real, expected accuracy gap
    under identical noise, so a future regression specific to the 2-ray
    case has something to actually fail against."""
    camera_matrix = make_camera_matrix(fov_deg=90.0)
    cams = [make_ring_camera(i, n_cameras=3, camera_matrix=camera_matrix) for i in range(3)]
    true_point = np.array([20.0, -15.0, 0.0])

    def _errors(cam_indices, seed):
        rng = np.random.default_rng(seed=seed)
        errs = []
        for trial in range(100):
            rays = []
            for i in cam_indices:
                cam = cams[i]
                pixel = np.array(_project_true_point(cam, true_point))
                noisy = pixel + rng.normal(0.0, 0.5, 2)
                rays.append(back_project_ray(
                    tuple(noisy), cam.camera_matrix, cam.dist_coeffs, cam.rvec, cam.tvec, cam=i
                ))
            result = triangulate(rays)
            assert result.ok, result.reason
            errs.append(float(np.linalg.norm(result.point_xyz - true_point)))
        return np.array(errs)

    errs_3of3 = _errors((0, 1, 2), seed=101)
    errs_2of3 = _errors((0, 2), seed=101)

    # 2-of-3 should be noticeably worse (loses the redundant constraint)
    # but still bounded/sane -- both halves matter, same pattern as the
    # clustered-quad PnP test (real risk, not total breakage).
    assert errs_2of3.mean() > errs_3of3.mean(), (
        f"expected 2-of-3 to be measurably worse than 3-of-3, got "
        f"2of3={errs_2of3.mean():.3f}mm 3of3={errs_3of3.mean():.3f}mm"
    )
    assert errs_2of3.max() < 5.0, f"2-of-3 error blew up unexpectedly: {errs_2of3.max()}mm"


def test_triangulate_catches_backward_ray():
    """Regression test for Verifier pass 3's critical finding: (I-dd^T)
    is invariant under d -> -d, so the math alone can't distinguish a
    correctly-forward ray from its backward mirror image. Reproduces the
    verifier's exact repro: flip one ray's direction, confirm the result
    is now correctly flagged (previously: ok=True, near-zero per-ray
    distances, while the "solution" was actually behind that camera)."""
    camera_matrix = make_camera_matrix(fov_deg=90.0)
    cams = [make_ring_camera(i, n_cameras=3, camera_matrix=camera_matrix) for i in range(3)]
    true_point = (40.0, -20.0, 0.0)

    rays = []
    for i, cam in enumerate(cams):
        pixel = _project_true_point(cam, true_point)
        rays.append(back_project_ray(pixel, cam.camera_matrix, cam.dist_coeffs, cam.rvec, cam.tvec, cam=i))

    # Flip camera 2's ray direction -- simulates e.g. a sign-convention
    # bug in a future ray producer, or a detector returning a physically
    # impossible backward direction.
    rays[2] = Ray(origin=rays[2].origin, direction=-rays[2].direction, cam=2)

    result = triangulate(rays)
    # The linear solve itself still "succeeds" numerically (that part of
    # the old behavior is unchanged and correct) -- but ok must now be
    # False because not every ray sees the point in front of it.
    assert not result.ok
    assert result.all_positive_depth is False
    assert result.per_ray_depth_mm is not None
    assert result.per_ray_depth_mm[2] < 0, (
        f"expected camera 2's depth to be negative (point behind it), "
        f"got {result.per_ray_depth_mm[2]}"
    )
    assert "behind" in result.reason


def test_triangulate_flags_near_parallel_rays_as_poorly_conditioned():
    """Regression test for Verifier pass 3's other critical finding:
    np.linalg.solve only raises on EXACT singularity -- near-parallel
    rays are technically solvable but numerically garbage. Reproduces a
    variant of the verifier's repro (rays ~1e-6 rad apart) and confirms
    it's now flagged rather than silently returned as ok=True."""
    origin = np.array([0.0, 0.0, 300.0])
    d1 = np.array([0.0, 0.0, -1.0])
    d1 = d1 / np.linalg.norm(d1)
    tiny_angle = 1e-6
    d2 = np.array([np.sin(tiny_angle), 0.0, -np.cos(tiny_angle)])
    rays = [
        Ray(origin=origin, direction=d1, cam=0),
        Ray(origin=origin + np.array([1.0, 0.0, 0.0]), direction=d2, cam=1),
    ]

    result = triangulate(rays)
    assert result.condition_number is not None
    assert result.condition_number > CONDITION_NUMBER_WARNING_THRESHOLD
    assert result.well_conditioned is False
    assert not result.ok
    assert "conditioned" in result.reason


def test_triangulate_refuses_single_ray_per_design():
    """The <=1-camera case is NOT this
    module's job -- callers handle the old-architecture fallback
    explicitly. triangulate() must fail honestly, not silently do
    something with 1 ray."""
    camera_matrix = make_camera_matrix(fov_deg=90.0)
    cam = make_ring_camera(0, n_cameras=3, camera_matrix=camera_matrix)
    pixel = _project_true_point(cam, (10.0, 10.0, 0.0))
    ray = back_project_ray(pixel, cam.camera_matrix, cam.dist_coeffs, cam.rvec, cam.tvec, cam=0)

    result = triangulate([ray])
    assert not result.ok
    assert "2" in result.reason

    result_empty = triangulate([])
    assert not result_empty.ok


def test_triangulate_degrades_gracefully_with_pixel_noise():
    """Small pixel noise on the tip detection -> small but nonzero
    triangulation error -- and the
    Z-discrepancy should correlate with how bad the observations were
    (not asserted exactly here, just that noise produces bounded, not
    wild, error)."""
    rng = np.random.default_rng(seed=11)
    camera_matrix = make_camera_matrix(fov_deg=90.0)
    cams = [make_ring_camera(i, n_cameras=3, camera_matrix=camera_matrix) for i in range(3)]
    true_point = np.array([40.0, -20.0, 0.0])

    errs = []
    discrepancies = []
    for trial in range(50):
        rays = []
        for i, cam in enumerate(cams):
            pixel = np.array(_project_true_point(cam, true_point))
            noisy_pixel = pixel + rng.normal(0.0, 0.5, 2)
            ray = back_project_ray(
                tuple(noisy_pixel), cam.camera_matrix, cam.dist_coeffs, cam.rvec, cam.tvec, cam=i
            )
            rays.append(ray)
        result = triangulate(rays)
        assert result.ok, result.reason
        errs.append(float(np.linalg.norm(result.point_xyz - true_point)))
        discrepancies.append(result.plane_discrepancy_mm)

    errs = np.array(errs)
    discrepancies = np.array(discrepancies)
    # Bounds set from actually measured data (mean 0.46mm, max 0.92mm
    # over these 50 trials), not a round-number guess -- an earlier draft
    # of this test used <20mm, ~20x looser than reality, the same
    # too-loose-tolerance mistake Verifier pass 2 already caught once in
    # the PnP tests (docs/DESIGN.md). Caught and fixed here proactively
    # this time instead of needing a third verifier pass to catch the
    # same class of issue again.
    assert errs.max() < 3.0, f"max triangulation error too large: {errs.max()} mm"
    assert discrepancies.max() < 3.0


def test_ray_direction_points_away_from_camera_toward_target():
    """Independent sanity check (not just round-trip): the back-projected
    ray's direction, from the camera's known position, should point
    roughly toward the true target, not away from it -- catches a sign
    error that a pure project-then-triangulate round-trip test could
    mask (the same concern that applies to calibration).

    Target FIXED per Verifier pass 3 (2026-08-12): the original version
    used the bull (0,0,0) -- which sits exactly on every synthetic
    camera's boresight (make_ring_camera always looks directly at the
    origin), so undistorted normalized coords are always (0,0)
    regardless of any x/y sign error. Verifier proved a real injected
    x-sign-flip bug PASSED this test unchanged (cos_angle=1.0 exactly)
    while failing against off-axis points -- i.e. this test provided
    ~zero protection against the exact bug class its docstring claimed
    to catch. Now uses an off-axis target instead.
    """
    camera_matrix = make_camera_matrix(fov_deg=90.0)
    cam = make_ring_camera(0, n_cameras=3, camera_matrix=camera_matrix)
    target = (-100.0, 60.0, 0.0)  # off-axis -- NOT the boresight-aligned bull
    pixel = _project_true_point(cam, target)
    ray = back_project_ray(pixel, cam.camera_matrix, cam.dist_coeffs, cam.rvec, cam.tvec)

    to_target = np.array(target) - ray.origin
    to_target /= np.linalg.norm(to_target)
    cos_angle = np.dot(ray.direction, to_target)
    assert cos_angle > 0.99, f"ray direction not pointing at target, cos={cos_angle}"
