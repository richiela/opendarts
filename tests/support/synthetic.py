"""Synthetic camera + scene generation for validating calibration math
against a KNOWN-correct answer. Real images have no independent ground
truth, so only synthetic data can actually *validate* correctness --
real data can only ever be a plausibility check.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticCamera:
    """A simulated camera with known ground-truth intrinsics + extrinsics.

    Extrinsics stored as rvec/tvec in OpenCV's convention: a world point
    X_w maps to camera coordinates via X_c = R @ X_w + t, where R =
    Rodrigues(rvec). tvec/rvec here describe the WORLD-TO-CAMERA
    transform (matching cv2.solvePnP's return convention), not the
    camera's own pose in world space directly (camera center in world
    space is C = -R^T @ t -- see camera_center_world()).
    """

    # NOTE (2026-08-12): fisheye support is DEFERRED,
    # not implemented. Every function in this module and pnp.py uses the
    # plain pinhole/Brown-Conrady cv2 API (cv2.projectPoints,
    # cv2.solvePnP/solvePnPRansac) regardless of dist_coeffs' shape --
    # despite fisheye/equidistant being the
    # current default *assumption* for the real rig (needs cv2.fisheye.*,
    # a genuinely different API family, not just more distortion terms).
    # An earlier version of this dataclass had an `is_fisheye` flag that
    # was never actually read anywhere -- removed rather than left as a
    # signal nobody consumes. Wiring real cv2.fisheye.* support through is untracked
    # future work, not assumed done.
    camera_matrix: np.ndarray # 3x3
    dist_coeffs: np.ndarray # (5,) pinhole/Brown-Conrady only, for now
    rvec: np.ndarray # (3,1) world-to-camera rotation
    tvec: np.ndarray # (3,1) world-to-camera translation

    def camera_center_world(self) -> np.ndarray:
        import cv2

        R, _ = cv2.Rodrigues(self.rvec)
        C = -R.T @ self.tvec
        return C.reshape(3)


def make_camera_matrix(
    image_width: int = 1280,
    image_height: int = 720,
    fov_deg: float = 90.0,
) -> np.ndarray:
    """A plausible camera_matrix for a given FOV, centered principal point.

    fov_deg is the horizontal field of view -- 90 deg is a reasonable
    stand-in for "wide but not extreme fisheye" until a real FOV
    measurement lands; this function
    exists so tests aren't hardcoding magic intrinsic numbers inline.
    """
    fx = (image_width / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    fy = fx # square pixels assumed
    cx = image_width / 2.0
    cy = image_height / 2.0
    return np.array(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64
    )


def make_ring_camera(
    index: int,
    n_cameras: int,
    camera_matrix: np.ndarray,
    ring_radius_mm: float = 500.0,
    height_mm: float = 300.0,
    dist_coeffs: np.ndarray | None = None,
) -> SyntheticCamera:
    """A camera at index `index` of `n_cameras` evenly spaced around a
    ring centered on the board (board-centered world frame, board face =
    Z=0), looking at the board center.

    Default n_cameras=3 with ~120 deg spacing matches the confirmed real
    rig -- this function is general so
    tests can also probe degenerate/edge configurations (e.g. 2 cameras,
    or cameras NOT evenly spaced) without duplicating the pose math.
    """
    import cv2

    angle = 2.0 * math.pi * index / n_cameras
    cam_pos_world = np.array(
        [ring_radius_mm * math.cos(angle), ring_radius_mm * math.sin(angle), height_mm],
        dtype=np.float64,
    )
    target = np.array([0.0, 0.0, 0.0], dtype=np.float64)

    forward = target - cam_pos_world
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-9:
        raise ValueError("camera position coincides with target — no valid orientation")
    forward /= forward_norm

    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-9:
        # 2026-08-12: forward is parallel to world_up
        # (camera directly overhead/underneath the board on the world
        # Z-axis, e.g. ring_radius_mm=0.0) -- previously silently
        # produced tvec=[nan, nan, ...] via 0/0 division. Fail loudly
        # instead; this is a genuinely undefined camera orientation
        # (which way is "right" when looking straight down?), not
        # something to paper over with an arbitrary default.
        raise ValueError(
            "camera forward direction is parallel to world_up — "
            "orientation is undefined (camera directly overhead/underneath "
            "the target); pick a non-degenerate ring_radius_mm/height_mm"
        )
    right /= right_norm
    true_up = np.cross(right, forward)

    # Camera-space axes expressed in world coords: camera looks down +Z_cam
    # (OpenCV convention), X_cam = right, Y_cam = -true_up (image Y grows
    # downward).
    R_world_to_cam_rows = np.stack([right, -true_up, forward], axis=0)
    tvec = -R_world_to_cam_rows @ cam_pos_world

    rvec, _ = cv2.Rodrigues(R_world_to_cam_rows)

    if dist_coeffs is None:
        dist_coeffs = np.zeros(5, dtype=np.float64)

    return SyntheticCamera(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvec=rvec.reshape(3, 1),
        tvec=tvec.reshape(3, 1),
    )


def project_points(
    camera: SyntheticCamera, object_points_mm: np.ndarray
) -> np.ndarray:
    """Project Nx3 known-world-mm points through a synthetic camera to
    Nx2 pixel coordinates -- the forward direction (known pose -> pixels),
    used to generate test input for the calibration solver, which then
    has to recover the pose in the REVERSE direction."""
    import cv2

    object_points_mm = np.asarray(object_points_mm, dtype=np.float64).reshape(-1, 1, 3)
    image_points, _ = cv2.projectPoints(
        object_points_mm,
        camera.rvec,
        camera.tvec,
        camera.camera_matrix,
        camera.dist_coeffs,
    )
    return image_points.reshape(-1, 2)
