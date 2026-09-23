"""Small geometric helpers for Ares: single-ray/board-plane
intersection, per-camera board-plane shaft lines, and the camera's own
ground-plane position (used by the geometric tip-end rule).

All board-plane work happens at Z=0 exactly -- the plane the calibration
landmarks were solved on. Deliberately NO fitted plane offset and NO
radial correction anywhere in this engine (the task's own success bar:
"prefer methods that do not need a per-epoch millimetre radial/Z fudge").
The fusion design makes that affordable: the engine's primary evidence is
each camera's shaft LINE, whose perpendicular error at z=0 was measured
essentially unbiased (+0.28mm median over 944 corpus reads), unlike the
tip-point reads' along-shaft component (-2.8mm median bias, the quantity
Athena's BOARD_PLANE_Z_MM/RADIAL_CORRECTION_MM constants exist to patch).
"""
from __future__ import annotations

import numpy as np

from opendarts.pipeline import CameraCalibration
from opendarts.triangulation.rays import Ray, back_project_ray

# "t <= 0" would put the intersection behind the camera -- geometrically
# solvable, physically meaningless. Same guard every ray/plane helper in
# this project carries.
_MIN_FORWARD_DEPTH_MM = 1e-6


def ray_board_xy(ray: Ray) -> tuple[float, float] | None:
    """Where a single ray crosses the board plane Z=0, or None when the
    ray runs parallel to the plane or the crossing sits behind the
    camera."""
    origin = np.asarray(ray.origin, dtype=np.float64)
    direction = np.asarray(ray.direction, dtype=np.float64)
    dz = direction[2]
    if abs(dz) < 1e-9:
        return None
    t = -origin[2] / dz
    if t <= _MIN_FORWARD_DEPTH_MM:
        return None
    point = origin + t * direction
    return (float(point[0]), float(point[1]))


def pixel_board_xy(
    pixel_xy: tuple[float, float], calib: CameraCalibration, cam: int
) -> tuple[float, float] | None:
    """Back-project one pixel and intersect with the board plane."""
    ray = back_project_ray(
        pixel_xy, calib.camera_matrix, calib.dist_coeffs,
        calib.rvec, calib.tvec, cam=cam,
    )
    return ray_board_xy(ray)


def board_plane_line(
    end_a_px: tuple[float, float],
    end_b_px: tuple[float, float],
    calib: CameraCalibration,
    cam: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Project the image line through two pixels of the detected shaft
    onto the board plane. Returns (point, unit_direction) in board mm, or
    None when either ray fails to cross the plane or the two projections
    coincide.

    Why this line is the engine's primary evidence: the plane through
    the camera centre and the dart's 3D centerline axis intersects the
    board plane in exactly this line, and the dart's true entry point
    lies ON that 3D axis -- so every camera's board-plane shaft line
    passes through the true entry point regardless of where ALONG the
    shaft that camera's tip pixel was localized. Line evidence is
    structurally immune to the along-shaft error family that dominates
    per-camera tip-point reads (measured on this corpus: 3.6mm median
    along-shaft vs 0.74mm median perpendicular)."""
    a = pixel_board_xy(end_a_px, calib, cam)
    b = pixel_board_xy(end_b_px, calib, cam)
    if a is None or b is None:
        return None
    p = np.array(a, dtype=np.float64)
    d = np.array(b, dtype=np.float64) - p
    n = np.linalg.norm(d)
    if n < 1e-9:
        return None
    return p, d / n


def camera_ground_xy(calib: CameraCalibration) -> tuple[float, float]:
    """The camera centre projected straight down onto the board plane
    (its X, Y in the board frame)."""
    import cv2

    rotation, _ = cv2.Rodrigues(np.asarray(calib.rvec, dtype=np.float64).reshape(3))
    translation = np.asarray(calib.tvec, dtype=np.float64).reshape(3, 1)
    centre = (-rotation.T @ translation).reshape(3)
    return float(centre[0]), float(centre[1])


def perp_distance_mm(
    point_xy: tuple[float, float], line: tuple[np.ndarray, np.ndarray]
) -> float:
    """Perpendicular distance from a board-plane point to a board-plane
    line given as (point, unit_direction)."""
    p, d = line
    e = np.array(point_xy, dtype=np.float64) - p
    return float(abs(e[0] * (-d[1]) + e[1] * d[0]))
