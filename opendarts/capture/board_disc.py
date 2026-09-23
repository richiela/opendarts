"""Per-camera board-disc detection region: geometry and the live registry.

``board_disc_mask()`` rasterises the board rim (plus a margin, plus the
same circle lifted by a dart's height so leaning flights stay inside)
into a boolean mask at the camera's capture resolution, from this rig's
own calibration -- pure board-face geometry, no corpus tuning. ``opendarts.lifecycle.signals`` splits every frame's changed pixels
into board/outside with it.

The registry (``set_calibrated_board_disc_masks`` /
``get_calibrated_board_disc_masks``) is filled once per successful
calibration by ``opendarts.live.capture_daemon`` and read by the lifecycle
driver's ``masks_provider``; clearing it (``None``/``{}``) stops throw
detection until the next calibration, which is the honest behaviour --
without a board region nothing can be judged.
"""
from __future__ import annotations

import numpy as np

from opendarts.pipeline import CameraCalibration

# Scoring edge (double ring outer) is 170mm; the physical board face runs
# out to ~225.5mm. The detection region is the face plus a margin.
BOARD_FACE_RADIUS_MM = 225.5
# Margin outside the physical face. Wide enough that a dart landing on the
# outer face (a miss that still hangs in the board) is inside the region;
# narrow enough that a player standing at the oche never is.
BOARD_DISC_MASK_MARGIN_MM = 30.0
BOARD_DISC_MASK_RADIUS_MM = BOARD_FACE_RADIUS_MM + BOARD_DISC_MASK_MARGIN_MM
# A dart standing in the board reaches this far out of the face; the rim
# circle is projected at Z=0 and at this height and the region is the
# convex hull of both, so flights leaning toward the camera stay inside.
DART_HEIGHT_OFFSET_MM = 70.0
_DISC_MASK_Z_OFFSETS_MM = (0.0, DART_HEIGHT_OFFSET_MM)
# Points sampled around each circle before the hull (sub-pixel deviation
# from the true conic at this rig's geometry).
_DISC_MASK_N_BOUNDARY_SAMPLES = 144


def _project_circle_px(
    calib: CameraCalibration, radius_mm: float, z_mm: float, n_samples: int
) -> np.ndarray | None:
    """Project one board-centred circle at height ``z_mm`` into pixels.
    ``None`` if any point lands behind the camera or non-finite."""
    import cv2

    ang = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    pts_w = np.stack(
        [radius_mm * np.cos(ang), radius_mm * np.sin(ang), np.full_like(ang, z_mm)],
        axis=1,
    )
    rvec = np.asarray(calib.rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(calib.tvec, dtype=np.float64).reshape(3, 1)
    R, _ = cv2.Rodrigues(rvec)
    cam_frame_pts = (R @ pts_w.T + tvec).T
    if not np.all(cam_frame_pts[:, 2] > 0):
        return None
    proj, _ = cv2.projectPoints(
        pts_w.reshape(-1, 1, 3), rvec, tvec, calib.camera_matrix, calib.dist_coeffs
    )
    proj = proj.reshape(-1, 2)
    if not np.all(np.isfinite(proj)):
        return None
    return proj


def board_disc_polygon_px(
    calib: CameraCalibration,
    *,
    radius_mm: float = BOARD_DISC_MASK_RADIUS_MM,
    z_offsets_mm: tuple[float, ...] = _DISC_MASK_Z_OFFSETS_MM,
    n_samples: int = _DISC_MASK_N_BOUNDARY_SAMPLES,
) -> np.ndarray | None:
    """Convex hull of the rim circle projected at every ``z_offsets_mm``.
    ``None`` only if every level degenerates."""
    import cv2

    projected = []
    for z_mm in z_offsets_mm:
        proj = _project_circle_px(calib, radius_mm, z_mm, n_samples)
        if proj is not None:
            projected.append(proj)
    if not projected:
        return None
    combined = np.concatenate(projected, axis=0).astype(np.float32)
    hull = cv2.convexHull(combined)
    return hull.reshape(-1, 2).astype(np.float64)


def board_disc_mask(
    calib: CameraCalibration,
    image_width: int,
    image_height: int,
    *,
    radius_mm: float = BOARD_DISC_MASK_RADIUS_MM,
    z_offsets_mm: tuple[float, ...] = _DISC_MASK_Z_OFFSETS_MM,
    n_samples: int = _DISC_MASK_N_BOUNDARY_SAMPLES,
) -> np.ndarray | None:
    """Boolean ``(image_height, image_width)`` mask, True inside the
    detection region; ``None`` (never raises) when the polygon can't be
    computed. Dimensions are rounded to int: the negotiated resolution
    lookup hands back floats."""
    import cv2

    image_width = int(round(image_width))
    image_height = int(round(image_height))
    polygon = board_disc_polygon_px(
        calib, radius_mm=radius_mm, z_offsets_mm=z_offsets_mm, n_samples=n_samples
    )
    if polygon is None:
        return None
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    poly_int = np.round(polygon).astype(np.int32)
    cv2.fillPoly(mask, [poly_int.reshape(-1, 1, 2)], 255)
    return mask.astype(bool)


# ----------------------------------------------------------------------
# live registry
_PER_CAMERA_BOARD_DISC_MASK: dict[int, np.ndarray] = {}


def set_calibrated_board_disc_masks(masks: dict[int, np.ndarray] | None) -> None:
    """(Re-)set the per-camera masks from a fresh calibration. ``None`` or
    ``{}`` clears them."""
    global _PER_CAMERA_BOARD_DISC_MASK
    _PER_CAMERA_BOARD_DISC_MASK = dict(masks) if masks else {}


def get_calibrated_board_disc_masks() -> dict[int, np.ndarray]:
    """A copy of whatever was last set, never the live dict."""
    return dict(_PER_CAMERA_BOARD_DISC_MASK)
