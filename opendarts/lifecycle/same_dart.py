"""Did another camera see THE SAME dart that one camera saw?

The single-camera rule (state.py) refuses a would-be dart that only one
camera sees on the board: a hand, a light or a person at the edge of one
view, never a dart, which every camera watches. But a real dart can reach
a second camera only off the board in its image -- with low cameras a
flight standing 100-150 mm out of the face lands over the wall above the
board -- and its shaft, a 1 px line at detection scale, is erased by the
change detector's 3x3 open. That is how a dart grouped in the 18 on a
low-camera rig went unscored (2026-10-02 10:13:05): camera 0 saw it
whole, cameras 1 and 2 only its flight. Change in those places is also
where hands and lights show up, in the same amounts, so no region or
size can tell the two apart.

Where it is can. Every changed pixel in the camera that saw the dart is
a line of sight, perhaps through some point of the dart. Cut to the space
a dart can occupy -- out of the face up to ``dart_len_mm``, within the
face plus ``lean_deg`` of lean -- each line becomes a short segment in
every other camera (a line that cannot pass through a dart -- the wall,
a person -- gives none). A real dart's change in another camera is the same
object, so it lies on those segments (measured on 1,242 single-camera
trials over 477 recorded darts: 1,241 at 50% or more, the lowest 46%,
median 100%); a hand or a glow elsewhere does not (16 refused ghosts: at
most 43%). The other camera's change is measured without the open, so the
shaft and tip it erased count too.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from opendarts.lifecycle.signals import SignalConfig, change_mask

#: The board face (the scoring area and the number ring); a tip is on it.
FACE_RADIUS_MM = 225.5


def _rt(cal: Any) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(cal.rvec, float).reshape(3, 1), np.asarray(cal.tvec, float).reshape(3, 1)


def band(
    cal_seen: Any,
    seen_px_small: np.ndarray,
    cal_other: Any,
    small_shape: tuple[int, int],
    scale: int,
    *,
    dart_len_mm: float,
    lean_deg: float,
    max_rays: int,
    band_px: int,
    n_heights: int = 16,
) -> np.ndarray:
    """Boolean mask at ``small_shape``: where, in the other camera, the
    seeing camera's changed pixels (``seen_px_small``, (N, 2) x/y at
    detection scale) can lie if they are points of a dart."""
    m = np.zeros(small_shape, np.uint8)
    if len(seen_px_small) == 0:
        return m.astype(bool)
    px = seen_px_small
    if len(px) > max_rays:
        # Fixed seed: the same frames always give the same band.
        px = px[np.random.default_rng(0).choice(len(px), max_rays, replace=False)]
    full = px.astype(np.float64) * scale + scale / 2.0
    und = cv2.undistortPoints(full.reshape(-1, 1, 2), cal_seen.camera_matrix,
                              cal_seen.dist_coeffs).reshape(-1, 2)
    rvec, tvec = _rt(cal_seen)
    R, _ = cv2.Rodrigues(rvec)
    centre = -R.T @ tvec.reshape(3)
    rays = (R.T @ np.column_stack([und, np.ones(len(und))]).T).T
    rays = rays[np.abs(rays[:, 2]) > 1e-9]
    if len(rays) == 0:
        return m.astype(bool)
    heights = np.linspace(0.0, dart_len_mm, n_heights)
    s = (heights[None, :] - centre[2]) / rays[:, 2:3]
    pts = centre[None, None, :] + s[:, :, None] * rays[:, None, :]
    inside = (s > 0) & (np.hypot(pts[..., 0], pts[..., 1])
                        <= FACE_RADIUS_MM + heights[None, :] * np.tan(np.radians(lean_deg)))
    rvec_o, tvec_o = _rt(cal_other)
    uv, _ = cv2.projectPoints(pts.reshape(-1, 3), rvec_o, tvec_o,
                              cal_other.camera_matrix, cal_other.dist_coeffs)
    uv = uv.reshape(len(rays), n_heights, 2) / scale
    for i in range(len(rays)):
        seg = uv[i, inside[i]]
        if len(seg) and np.isfinite(seg).all():
            cv2.polylines(m, [np.round(seg).astype(np.int32).reshape(-1, 1, 2)], False, 1, 1)
    if band_px:
        m = cv2.dilate(m, np.ones((2 * band_px + 1, 2 * band_px + 1), np.uint8))
    return m.astype(bool)


def support(
    cals: dict[int, Any],
    seen_cam: int,
    seen_change: np.ndarray,
    others: dict[int, tuple[np.ndarray, np.ndarray]],
    signal_cfg: SignalConfig,
    *,
    dart_len_mm: float,
    lean_deg: float,
    max_rays: int,
    band_px: int,
) -> dict[int, tuple[int, int]]:
    """Per other camera: (its changed pixels on the band, its changed pixels
    in all). ``others`` maps a camera to its (current, reference) small
    grey images; ``seen_change`` is the seeing camera's live change mask,
    all of it -- its view of a flight is often off its board. Cameras
    without a calibration are left out."""
    if seen_cam not in cals:
        return {}
    ys, xs = np.nonzero(seen_change)
    seen_px = np.column_stack([xs, ys])
    unfiltered = SignalConfig(
        scale=signal_cfg.scale, diff_threshold=signal_cfg.diff_threshold,
        open_kernel=0, min_blob_px=2, delta_min_blob_px=signal_cfg.delta_min_blob_px,
    )
    out: dict[int, tuple[int, int]] = {}
    for cam, (cur, ref) in others.items():
        if cam == seen_cam or cam not in cals:
            continue
        changed = change_mask(cur, ref, unfiltered)
        on = band(cals[seen_cam], seen_px, cals[cam], cur.shape[:2], signal_cfg.scale,
                  dart_len_mm=dart_len_mm, lean_deg=lean_deg, max_rays=max_rays, band_px=band_px)
        out[cam] = (int(np.count_nonzero(changed & on)), int(np.count_nonzero(changed)))
    return out
