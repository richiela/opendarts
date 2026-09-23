"""Multi-view ray back-projection + triangulation. The ≤1-camera
degenerate case is NOT handled here -- callers must check ray count >= 2
before calling triangulate(); single-ray fallback is old-architecture
behavior, kept
separate/out of this module on purpose so this module can stay "this is
the genuinely new part" without a silent single-ray branch inside it).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Ray:
    """A 3D ray in world coordinates: origin (camera center) + unit
    direction, pointing away from the camera into the scene."""

    origin: np.ndarray  # (3,)
    direction: np.ndarray  # (3,), unit length
    cam: int | None = None  # which camera this came from, for diagnostics


def back_project_ray(
    pixel_xy: tuple[float, float],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    cam: int | None = None,
) -> Ray:
    """A single camera's 2D pixel detection -> a 3D ray in world space.

    rvec/tvec are the world-to-camera pose (same convention as
    opendarts.calibration.pnp/synthetic -- X_c = R @ X_w + t).

    NOTE (2026-08-12): uses the pinhole/Brown-Conrady
    cv2.undistortPoints API unconditionally, same deferred-fisheye gap
    documented in tests/support/synthetic.py -- fisheye/equidistant
    (cv2.fisheye.*, a different API family)
    is the current default assumption for the real rig pending an actual
    FOV measurement. Not implemented here either; flagged explicitly
    (previously this module had the same gap with zero acknowledgment,
    inconsistent with synthetic.py's prominent disclosure of it).
    """
    import cv2

    pixel = np.asarray(pixel_xy, dtype=np.float64).reshape(1, 1, 2)
    undistorted = cv2.undistortPoints(pixel, camera_matrix, dist_coeffs)
    x, y = undistorted.reshape(2)
    dir_cam = np.array([x, y, 1.0], dtype=np.float64)

    R, _ = cv2.Rodrigues(rvec)
    camera_center_world = -R.T @ tvec.reshape(3)
    direction_world = R.T @ dir_cam
    direction_world = direction_world / np.linalg.norm(direction_world)

    return Ray(origin=camera_center_world, direction=direction_world, cam=cam)


@dataclass
class TriangulationResult:
    ok: bool
    point_xyz: np.ndarray | None  # raw least-squares triangulated point
    board_plane_xy: tuple[float, float] | None  # Z=0 projection
    plane_discrepancy_mm: float | None  # |point.z|
    per_ray_distance_mm: list[float] | None  # perpendicular distance, point to each ray
    # Added 2026-08-12: the math (I - dd^T) is
    # invariant under d -> -d, so it cannot by itself tell a correctly-
    # forward ray from its backward mirror image through the same
    # origin -- verified concretely: a flipped ray produced ok=True and
    # near-zero per_ray_distance_mm while its "solution" was actually
    # 587mm BEHIND that camera. per_ray_depth_mm (signed: positive =
    # point is in front of that camera, negative = behind) and
    # all_positive_depth close that blind spot -- ok alone no longer
    # means "the math didn't crash," it also requires every ray to
    # genuinely see the point in front of it.
    per_ray_depth_mm: list[float] | None
    all_positive_depth: bool | None
    # np.linalg.solve only raises on EXACT
    # singularity -- near-parallel/clustered-camera geometry (poorly
    # conditioned but technically solvable) silently returns numerically
    # garbage otherwise. Verified: 1e-6-radian-apart rays recovered a
    # point ~10 million mm from truth with ok=True and no other signal
    # of a problem. condition_number/well_conditioned expose this --
    # same underlying risk category as the PnP point-spread conditioning
    # finding from the point-spread work.
    condition_number: float | None
    well_conditioned: bool | None
    n_rays: int = 0
    reason: str = ""


# Above this condition number, treat the linear solve as untrustworthy
# even though it "succeeded" -- chosen conservatively (well below the
# ~10 million mm garbage case that was found, comfortably above
# the well-spread 3-camera-120-degree baseline which sits near ~2-3 for
# this rig's geometry) rather than precisely tuned; revisit once there
# is real occlusion-pattern data.
CONDITION_NUMBER_WARNING_THRESHOLD = 1000.0


def triangulate(rays: list[Ray]) -> TriangulationResult:
    """Least-squares closest point to N>=2 rays (the general multi-view
    case -- used uniformly for 2-of-3 or 3-of-3, not special-cased per
    ray count).

    Standard closest-point-to-multiple-lines solution: minimize
    sum_i || (I - d_i d_i^T)(P - o_i) ||^2 over P. Setting the gradient
    to zero gives a linear system A @ P = b with
    A = sum_i (I - d_i d_i^T), b = sum_i (I - d_i d_i^T) @ o_i.
    (Hand-derived and cross-validated against cv2.triangulatePoints to
    1.6e-13 agreement -- the math itself is correct.)

    ok=True means: the linear solve succeeded, AND every ray sees the
    recovered point in front of it (all_positive_depth), AND the system
    was well-conditioned (well_conditioned). A caller that only checks
    the old, narrower meaning of ok should re-check against this
    docstring after the depth/conditioning fixes.
    """

    def _empty_result(n: int, reason: str) -> TriangulationResult:
        return TriangulationResult(
            ok=False,
            point_xyz=None,
            board_plane_xy=None,
            plane_discrepancy_mm=None,
            per_ray_distance_mm=None,
            per_ray_depth_mm=None,
            all_positive_depth=None,
            condition_number=None,
            well_conditioned=None,
            n_rays=n,
            reason=reason,
        )

    if len(rays) < 2:
        return _empty_result(
            len(rays),
            f"need >=2 rays to triangulate, got {len(rays)} -- the "
            "<=1-camera case is a "
            "deliberate old-architecture fallback handled by the "
            "CALLER, not this function",
        )

    A = np.zeros((3, 3))
    b = np.zeros(3)
    for ray in rays:
        d = ray.direction
        # Defensive normalization: the (I - dd^T)
        # projection math is only valid for unit-length d. Holds today
        # only because back_project_ray always normalizes -- any future
        # direct Ray(...) construction with a non-unit direction would
        # otherwise silently corrupt every downstream result.
        d = d / np.linalg.norm(d)
        proj = np.eye(3) - np.outer(d, d)
        A += proj
        b += proj @ ray.origin

    condition_number = float(np.linalg.cond(A))
    well_conditioned = condition_number < CONDITION_NUMBER_WARNING_THRESHOLD

    try:
        point = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return _empty_result(
            len(rays), "singular system (rays parallel or otherwise degenerate)"
        )

    per_ray_dist = []
    per_ray_depth = []
    for ray in rays:
        v = point - ray.origin
        d = ray.direction / np.linalg.norm(ray.direction)
        t = float(np.dot(v, d))  # signed depth along the ray
        perp = v - t * d
        per_ray_dist.append(float(np.linalg.norm(perp)))
        per_ray_depth.append(t)

    all_positive_depth = all(t > 0 for t in per_ray_depth)
    board_xy = (float(point[0]), float(point[1]))
    discrepancy = float(abs(point[2]))

    return TriangulationResult(
        ok=all_positive_depth and well_conditioned,
        point_xyz=point,
        board_plane_xy=board_xy,
        plane_discrepancy_mm=discrepancy,
        per_ray_distance_mm=per_ray_dist,
        per_ray_depth_mm=per_ray_depth,
        all_positive_depth=all_positive_depth,
        condition_number=condition_number,
        well_conditioned=well_conditioned,
        n_rays=len(rays),
        reason=(
            ""
            if (all_positive_depth and well_conditioned)
            else (
                ("point behind at least one camera; " if not all_positive_depth else "")
                + (f"poorly conditioned (cond={condition_number:.1f}); " if not well_conditioned else "")
            ).strip()
        ),
    )
