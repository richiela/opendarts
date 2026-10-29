"""Argus's per-camera feature extraction: raw frames in, the 17 numbers
`opendarts.engines.argus.decision` reads out, one dict per camera.

The decision (`decision.decide`: `visual_decision`, `fuse`) is pure
arithmetic over exactly these per-camera fields and nothing else: `ok`, `board_xy`, `token`, `paint_token`, `margin_mm`,
`mm_per_px`, `width_ratio`, `elongation_ratio`, `texture_tiebreak_used`,
`contam_suspected`, `entry_occluded`, `clarity_z`, `paint_margin_ratio`,
`agree_mm`, `needle_w_mm`, `jac_u`, `jac_v`. Every computation below
exists to produce exactly those fields, as cheaply as the arithmetic
allows, and nothing else.

Per camera:
  1. `opendarts.engines.apollo.tip_detection.detect_tip` finds the tip
     (and the losing end) from that one camera's own (bg, dart) pair --
     no calibration, no other camera.
  2. Across cameras, `parallax_choose_ends` decides which end of each
     camera's blob is really the tip, by back-projecting both ends of
     every camera through that camera's own calibration onto the board
     plane (Z=0) and picking the assignment whose board points agree
     best; with fewer than two usable cameras it just keeps each
     camera's own call.
  3. The chosen tip projects to a board point, a (number, ring) label
     and a wire/ring margin; `mm_per_px` is the local scale along the
     needle axis, for turning pixel distances into millimetres.
  4. One full-frame per-pixel channel-max |bg - dart| diff feeds three
     independent reads: `clarity_z` (how far the tip patch stands out of
     the frame's own diff noise, via an exact histogram median/MAD),
     entry occlusion (whether the tip sits on a dart already lying on
     the board from earlier in this visit), and the needle's own width
     measurement below.
  5. The bed paint colour (Lab) under the tip, read only within the
     candidate beds the board map places near the tip, can override the
     map's own label when a candidate paint clearly reads closer than
     the second-best one.
  6. `needle_w_mm`: the needle's own measured width a few pixels back
     from the tip, in millimetres, reusing the tip/far points and
     calibration already settled above.
  7. `jac_u`/`jac_v`: the local Jacobian of the pixel-to-board projection
     at the tip, along and across the needle -- how far the board point
     moves per pixel in each direction, for `decision.fuse`'s per-camera
     uncertainty ellipse.
  8. `agree_mm`: how far this camera's board point sits from the mean of
     the other cameras' -- a cheap cross-camera sanity read.

Inputs (all in memory already -- no package loading, no caching):
  `bg_images`/`frame_images`: {camera: BGR frame} background and dart
    frame, same camera, same shape.
  `calibration`: {camera: `opendarts.pipeline.CameraCalibration`} this
    throw's own calibration, one projective model built fresh per call
    (never cached -- the caller's calibration is authoritative).
  `prior_dart_line_px`: optional {camera: (tip_px, far_px)}, the SAME
    camera's own prior-throw-in-visit line (see
    `opendarts.engines.apollo.prior_dart_context`), passed straight
    through to `detect_tip`.
  `visit_bg_images`: optional {camera: BGR frame}, the visit's first
    dart's background frame, for the entry-occlusion check below. Given
    only from the second dart of a visit onward; `None` (or a camera
    missing from it, or a shape mismatch against this throw's own dart
    frame) means "nothing to check against" for that camera, not "no
    occlusion found".

Returns `{"cams": {str(camera): {...}}, "parallax": parallax_diag}`.
A camera missing calibration, or whose calibration fails to build a
projective model, is left out of `cams` entirely -- exactly as if it had
not been given at all.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from opendarts.engines.apollo.tip_detection import (
    DIFF_THRESHOLD,
    OPEN_KERNEL_PX,
    _blurred_diff,
    detect_tip,
    ellipse_kernel,
    threshold_mask,
)
from opendarts.geometry.board import (
    BULL_RADIUS_MM,
    DOUBLE_INNER_RADIUS_MM,
    DOUBLE_OUTER_RADIUS_MM,
    OUTER_BULL_RADIUS_MM,
    SECTOR_NUMBERS_CLOCKWISE,
    TREBLE_INNER_RADIUS_MM,
    TREBLE_OUTER_RADIUS_MM,
)
from opendarts.pipeline import CameraCalibration

# --------------------------------------------------------------------------- #
# Board map: calibration-projected pixel <-> board mm (Z=0 plane).
# --------------------------------------------------------------------------- #

# A cross-view pick within this many px of detect_tip's own ends is that
# same detection (Apollo's cross_view.SAME_DETECTION_PX).
_SAME_DETECTION_PX = 3.0

# OD's board geometry. The REGULATION radii throughout this module: the
# per-camera map token and paint candidates are regulation-map reads, and
# `decision` moves the two inner boundaries to OD's scoring radii itself.
CANON_NUMBERS = SECTOR_NUMBERS_CLOCKWISE

RING_MM = dict(db=BULL_RADIUS_MM, ob=OUTER_BULL_RADIUS_MM,
               treble_in=TREBLE_INNER_RADIUS_MM, treble_out=TREBLE_OUTER_RADIUS_MM,
               double_in=DOUBLE_INNER_RADIUS_MM, double_out=DOUBLE_OUTER_RADIUS_MM)
# Ascending ring-boundary radii, for the nearest-boundary (wire/ring
# margin) search below.
_RING_RADII_MM = (RING_MM["db"], RING_MM["ob"], RING_MM["treble_in"],
                   RING_MM["treble_out"], RING_MM["double_in"], RING_MM["double_out"])

WHITE_LAB = np.array([255.0, 128.0, 128.0])  # OpenCV 8-bit Lab: bright, neutral


@dataclass
class CalibModel:
    ok: bool
    camera_matrix: np.ndarray = None
    dist_coeffs: np.ndarray = None
    rvec: np.ndarray = None
    tvec: np.ndarray = None
    R: np.ndarray = None
    error: str | None = None


def build_calib_model(calib: CameraCalibration) -> CalibModel:
    try:
        R, _ = cv2.Rodrigues(np.asarray(calib.rvec, dtype=np.float64))
    except Exception as e:
        return CalibModel(False, error=f"Rodrigues failed: {e}")
    return CalibModel(
        True,
        camera_matrix=np.asarray(calib.camera_matrix, dtype=np.float64),
        dist_coeffs=np.asarray(calib.dist_coeffs, dtype=np.float64),
        rvec=np.asarray(calib.rvec, dtype=np.float64),
        tvec=np.asarray(calib.tvec, dtype=np.float64).reshape(3),
        R=R,
    )


def pixel_to_board_xy(model: CalibModel, pts_px: np.ndarray) -> np.ndarray:
    """pts_px: (N,2) pixel coords -> (N,2) board XY in mm (Z=0 plane)."""
    pts = pts_px.reshape(-1, 1, 2).astype(np.float64)
    und = cv2.undistortPoints(pts, model.camera_matrix, model.dist_coeffs)
    xp = und[:, 0, 0]
    yp = und[:, 0, 1]
    r0, r1, t = model.R[:, 0], model.R[:, 1], model.tvec
    a11 = r0[0] - xp * r0[2]
    a12 = r1[0] - xp * r1[2]
    b1 = xp * t[2] - t[0]
    a21 = r0[1] - yp * r0[2]
    a22 = r1[1] - yp * r1[2]
    b2 = yp * t[2] - t[1]
    det = a11 * a22 - a12 * a21
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)
    X = (b1 * a22 - b2 * a12) / det
    Y = (a11 * b2 - a21 * b1) / det
    return np.stack([X, Y], axis=1)


def label_at_board_xy(X: float, Y: float):
    r = math.hypot(X, Y)
    theta_deg = math.degrees(math.atan2(X, Y)) % 360
    k = int(round(theta_deg / 18.0)) % 20
    number = CANON_NUMBERS[k]
    if r <= RING_MM["db"]:
        ring = "DB"
    elif r <= RING_MM["ob"]:
        ring = "OB"
    elif r <= RING_MM["treble_in"]:
        ring = "S"
    elif r <= RING_MM["treble_out"]:
        ring = "T"
    elif r <= RING_MM["double_in"]:
        ring = "S"
    elif r <= RING_MM["double_out"]:
        ring = "D"
    else:
        ring = "OUT"
    return number, ring, r, theta_deg, k


def token_of(number, ring) -> str:
    if ring in ("DB", "OB", "OUT"):
        return ring
    return f"{ring}{number}"


def _nearest_boundary_mm(r_mm: float, theta_deg: float, k: int) -> float:
    """Distance (mm) from a board point already labelled (r_mm, theta_deg,
    sector k) to the nearest ring boundary or sector wire -- whichever is
    closer, the wire's angular distance converted to mm at this radius."""
    ring_mm_dist = min(abs(r_mm - rb) for rb in _RING_RADII_MM)
    wire_lo = (18 * k - 9) % 360
    wire_hi = (18 * k + 9) % 360

    def ang_diff(a, b):
        d = (a - b) % 360
        return min(d, 360 - d)

    wire_deg_dist = min(ang_diff(theta_deg, wire_lo), ang_diff(theta_deg, wire_hi))
    wire_mm_dist = math.radians(wire_deg_dist) * max(r_mm, 20.0)
    return min(ring_mm_dist, wire_mm_dist)


# --------------------------------------------------------------------------- #
# Which end of each camera's blob is the tip: back-project both ends of
# every camera through its own calibration onto the board plane and pick
# the assignment whose board points agree best (small spread). Needs >=2
# usable cameras; otherwise each camera's own detect_tip call stands.
# --------------------------------------------------------------------------- #

PARALLAX_GATE_RATIO = 1.5  # below this, the two board-plane spreads (best vs.
                           # opposite assignment) are not separated enough to
                           # overrule detect_tip's own per-camera call.


def _board_xy1(model: CalibModel, px) -> np.ndarray:
    return pixel_to_board_xy(model, np.asarray(px, float).reshape(1, 2))[0]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _local_mm_per_px(model: CalibModel, px, u) -> float:
    eps = 1.0
    q0 = _board_xy1(model, px)
    q1 = _board_xy1(model, np.asarray(px, float) + eps * u)
    return float(np.linalg.norm(q1 - q0)) / eps


def parallax_choose_ends(models: dict, raw_tips: dict) -> tuple[dict, dict]:
    """`raw_tips[cam] = dict(ok, tip_px, far_px)` (detect_tip's own call).
    Returns `(chosen, diag)`; `chosen[cam] = (tip_px, far_px)` with ends
    possibly swapped from detect_tip's own guess. `diag['method']` is
    `"parallax"`, `"detect_tip"` (gated off -- ambiguous), or `"n/a"`
    (<2 usable cameras)."""
    cams = [c for c, t in raw_tips.items() if t.get("ok")]
    chosen = {c: (np.asarray(raw_tips[c]["tip_px"], float), np.asarray(raw_tips[c]["far_px"], float))
              for c in cams}
    if len(cams) < 2:
        return chosen, dict(method="n/a", cams=cams)

    board_pts = {}
    for c in cams:
        e0, e1 = chosen[c]
        try:
            board_pts[c] = (_board_xy1(models[c], e0), _board_xy1(models[c], e1))
        except Exception:
            return chosen, dict(method="n/a", cams=cams, error="projection failed")

    n = len(cams)
    results = []
    for bits in range(2 ** n):
        pts = [board_pts[cams[i]][(bits >> i) & 1] for i in range(n)]
        spread = max(float(np.linalg.norm(a - b)) for a in pts for b in pts) if len(pts) > 1 else 0.0
        results.append((spread, bits))
    results.sort(key=lambda r: r[0])
    spread_tip, best_bits = results[0]
    comp_bits = best_bits ^ (2 ** n - 1)
    spread_far = next(s for s, b in results if b == comp_bits)
    far_ratio = None if spread_tip < 1e-9 else spread_far / max(spread_tip, 1e-9)

    if far_ratio is not None and far_ratio < PARALLAX_GATE_RATIO:
        return chosen, dict(method="detect_tip", cams=cams, spread_tip=spread_tip,
                             spread_far=spread_far, far_ratio=far_ratio)

    out = {}
    for i, c in enumerate(cams):
        bit = (best_bits >> i) & 1
        e0, e1 = chosen[c]
        out[c] = (e0, e1) if bit == 0 else (e1, e0)
    return out, dict(method="parallax", cams=cams, spread_tip=spread_tip,
                      spread_far=spread_far, far_ratio=far_ratio)


# --------------------------------------------------------------------------- #
# One full-frame per-pixel channel-max |bg-dart| per camera, reused three
# ways below (clarity, occlusion, the needle-width diff crop) instead of
# recomputing it per use. `cv2.max` twice beats numpy's generic
# `.max(axis=2)` reduce for this shape and is bit-identical (plain
# elementwise uint8 max, nothing to round).
# --------------------------------------------------------------------------- #

def _abs_diff_max(bg_bgr: np.ndarray, dart_bgr: np.ndarray) -> np.ndarray:
    d = cv2.absdiff(bg_bgr, dart_bgr)
    return cv2.max(cv2.max(d[:, :, 0], d[:, :, 1]), d[:, :, 2])


# --------------------------------------------------------------------------- #
# clarity_z: how far the tip patch's mean diff value stands out of the
# frame's own diff noise (median/MAD), computed via O(256)-bin histograms
# instead of sorting the whole frame twice. Exact: every value here (the
# diff itself, the first median, the deviations, the second median) is a
# multiple of 0.5 or 0.25, all exactly representable in float, so the
# histogram order-statistic agrees with a direct sort bit for bit.
# --------------------------------------------------------------------------- #

def _hist_order_stat(csum: np.ndarray, k: int) -> int:
    """Smallest value v such that csum[v] >= k (1-indexed k-th smallest)."""
    return int(np.searchsorted(csum, k, side="left"))


def _clarity_stats(diff_u8: np.ndarray):
    """(noise_med, mad) -- exact equivalent of
    noise_med = float(np.median(diff_gray))
    mad = float(np.median(np.abs(diff_gray - noise_med))) * 1.4826
    for diff_gray = diff_u8.astype(float32)."""
    n = diff_u8.size
    hist = cv2.calcHist([diff_u8], [0], None, [256], [0, 256]).ravel().astype(np.int64)
    csum = np.cumsum(hist)
    if n % 2 == 1:
        v = _hist_order_stat(csum, (n + 1) // 2)
        med2 = 2 * v
    else:
        v1 = _hist_order_stat(csum, n // 2)
        v2 = _hist_order_stat(csum, n // 2 + 1)
        med2 = v1 + v2
    noise_med = med2 / 2.0

    dvals = np.arange(256)
    aa = np.abs(2 * dvals - med2)  # doubled |diff - noise_med| per diff-value bucket
    maxaa = int(aa.max())
    hist2 = np.bincount(aa, weights=hist.astype(np.float64), minlength=maxaa + 1)
    csum2 = np.cumsum(hist2)
    if n % 2 == 1:
        v = _hist_order_stat(csum2, (n + 1) // 2)
        mad_doubled2 = 2 * v
    else:
        v1 = _hist_order_stat(csum2, n // 2)
        v2 = _hist_order_stat(csum2, n // 2 + 1)
        mad_doubled2 = v1 + v2
    mad_raw = (mad_doubled2 / 2.0) / 2.0  # undo the doubling
    mad = mad_raw * 1.4826
    return noise_med, mad


# --------------------------------------------------------------------------- #
# Entry occlusion: is the tip sitting on a dart already lying on the board
# from earlier in this visit? Computed only on a crop around the (now-
# known) tip -- core(S0,Sn) -> threshold -> open -> dilate -- restricted to
# a crop of half-size WIDE_WIN + WIDE_MARGIN, clipped to the image so a tip
# near the edge gets the same border treatment (cv2's reflect-101 for the
# blur/open, replicate-by-construction for the dilate) as a full-frame
# computation would give it; the margin is well over the blur + open +
# dilate radii, so every pixel actually read (the tip pixel itself, and the
# paint check's own <=25px window) is unaffected by the crop boundary.
# --------------------------------------------------------------------------- #

SAME_LINE_PX = 10
WIDE_WIN = 26      # >= the paint check's own 25px window, +1 slack
WIDE_MARGIN = 48   # >> blur radius(2) + open radius(1) + dilate radius(10) = 13


def _gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2GRAY)


class WideWindow:
    """wide_crop (bool array) + its origin (x0,y0) in full-image pixel
    coords, covering at least tip +/- WIDE_WIN (clipped to the image).
    `arr is None` means "no occlusion" -- cheap, no cv2 call."""

    __slots__ = ("arr", "x0", "y0", "w", "h")

    def __init__(self, arr, x0, y0, w, h):
        self.arr, self.x0, self.y0, self.w, self.h = arr, x0, y0, w, h

    def end_on_old(self, pt) -> bool:
        """Strict (unclipped) bounds check; False when out of image or
        there is no occlusion at all."""
        if self.arr is None:
            return False
        x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
        if not (0 <= x < self.w and 0 <= y < self.h):
            return False
        return bool(self.arr[y - self.y0, x - self.x0])

    def at_clipped(self, xi: np.ndarray, yi: np.ndarray) -> np.ndarray:
        """xi,yi already clipped to the image (the paint check's own
        convention). False everywhere when there is no occlusion."""
        if self.arr is None:
            return np.zeros(xi.shape, dtype=bool)
        return self.arr[yi - self.y0, xi - self.x0]


def _wide_window(s0, sn, tip_px, shape) -> WideWindow:
    """`s0`/`sn` are two bg frames of the SAME camera, same shape as
    `shape`: `s0` from before the occluding dart(s) landed, `sn` from
    just before this throw. `s0 is None` means "nothing to check" for
    this camera."""
    h, w = shape
    if s0 is None:
        return WideWindow(None, 0, 0, w, h)

    xi, yi = int(round(float(tip_px[0]))), int(round(float(tip_px[1])))
    half = WIDE_WIN + WIDE_MARGIN
    x0, x1 = max(0, xi - half), min(w, xi + half + 1)
    y0, y1 = max(0, yi - half), min(h, yi + half + 1)

    g0 = _gray(s0[y0:y1, x0:x1])
    g1 = _gray(sn[y0:y1, x0:x1])
    db = _blurred_diff(g0, g1)
    m = threshold_mask(db, DIFF_THRESHOLD)
    core_crop = cv2.morphologyEx(m, cv2.MORPH_OPEN, ellipse_kernel(OPEN_KERNEL_PX)) > 0
    k = 2 * SAME_LINE_PX + 1
    wide_crop = cv2.dilate(core_crop.astype(np.uint8),
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    return WideWindow(wide_crop, x0, y0, w, h)


# --------------------------------------------------------------------------- #
# Bed paint colour (Lab) under the tip: among the beds the board map places
# within 4px of the tip, does the bg paint colour under the tip clearly
# read closer to one of them than to the runner-up? If so that candidate
# overrides the map's own label; otherwise the map label stands.
# `token_of()` only ever takes 6 rings x 20 numbers of distinct pairs, so
# the token for a (ring, number) pair is looked up from a small table
# built once at import time instead of called per pixel.
# --------------------------------------------------------------------------- #

PAINT_WIN = 25  # the paint check's own window half-width, px

_RING_NAMES = ["DB", "OB", "S", "T", "D", "OUT"]
_TOKEN_LUT = np.empty((len(_RING_NAMES), 21), dtype=object)
for _ridx, _rname in enumerate(_RING_NAMES):
    for _num in range(21):
        _TOKEN_LUT[_ridx, _num] = token_of(_num, _rname)


def _label_grid(model: CalibModel, cx: float, cy: float, win: int):
    """Per-pixel (token, number, ring, distance-from-centre) over a
    (2*win+1)^2 pixel window centred near (cx, cy)."""
    xs = np.arange(int(round(cx)) - win, int(round(cx)) + win + 1)
    ys = np.arange(int(round(cy)) - win, int(round(cy)) + win + 1)
    gx, gy = np.meshgrid(xs, ys)
    pix = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)
    xy = pixel_to_board_xy(model, pix)
    r = np.hypot(xy[:, 0], xy[:, 1])
    theta = np.degrees(np.arctan2(xy[:, 0], xy[:, 1])) % 360
    kk = np.round(theta / 18.0).astype(int) % 20
    numbers = np.asarray(CANON_NUMBERS)[kk]

    ring_code = np.empty(len(r), dtype=np.int64)
    ring = np.empty(len(r), dtype=object)
    m = r <= RING_MM["db"]
    ring_code[m], ring[m] = 0, "DB"
    m = (r > RING_MM["db"]) & (r <= RING_MM["ob"])
    ring_code[m], ring[m] = 1, "OB"
    m = (r > RING_MM["ob"]) & (r <= RING_MM["treble_in"])
    ring_code[m], ring[m] = 2, "S"
    m = (r > RING_MM["treble_in"]) & (r <= RING_MM["treble_out"])
    ring_code[m], ring[m] = 3, "T"
    m = (r > RING_MM["treble_out"]) & (r <= RING_MM["double_in"])
    ring_code[m], ring[m] = 2, "S"
    m = (r > RING_MM["double_in"]) & (r <= RING_MM["double_out"])
    ring_code[m], ring[m] = 4, "D"
    m = r > RING_MM["double_out"]
    ring_code[m], ring[m] = 5, "OUT"

    tokens = _TOKEN_LUT[ring_code, numbers]
    dpix = np.hypot(pix[:, 0] - cx, pix[:, 1] - cy)
    return pix, tokens, numbers, ring, dpix, (len(ys), len(xs))


def _paint_variant(bg_bgr: np.ndarray, model: CalibModel, tip_px: np.ndarray,
                    map_token: str, map_number, map_ring,
                    wide: WideWindow, this_occ_full: np.ndarray) -> dict:
    win = PAINT_WIN
    pix, tokens, numbers, rings, dpix, shape = _label_grid(model, tip_px[0], tip_px[1], win)

    near4 = dpix <= 4
    cand_tokens = list(dict.fromkeys(tokens[near4].tolist()))
    if map_token not in cand_tokens:
        cand_tokens.append(map_token)
    cand_fields = {}
    for ct in cand_tokens:
        idx = np.flatnonzero(tokens == ct)
        if len(idx):
            cand_fields[ct] = (numbers[idx[0]], rings[idx[0]])
    cand_fields.setdefault(map_token, (map_number, map_ring))

    if len(cand_tokens) <= 1:
        return dict(paint_token=map_token, paint_number=map_number, paint_ring=map_ring,
                    paint_candidates=cand_tokens, paint_distances={})

    h, w = this_occ_full.shape[:2]
    px0, px1 = max(0, int(round(tip_px[0])) - win), min(w, int(round(tip_px[0])) + win + 1)
    py0, py1 = max(0, int(round(tip_px[1])) - win), min(h, int(round(tip_px[1])) + win + 1)
    lab_crop = cv2.cvtColor(bg_bgr[py0:py1, px0:px1], cv2.COLOR_BGR2LAB).astype(np.float64)

    def lab_patch(cx, cy, half):
        x0, x1 = max(0, int(round(cx)) - half), min(w, int(round(cx)) + half + 1)
        y0, y1 = max(0, int(round(cy)) - half), min(h, int(round(cy)) + half + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        return lab_crop[y0 - py0:y1 - py0, x0 - px0:x1 - px0].reshape(-1, 3)

    tip_patch = lab_patch(tip_px[0], tip_px[1], 1)
    tip_color = np.median(tip_patch, axis=0) if tip_patch is not None and len(tip_patch) else WHITE_LAB.copy()

    distances = {}
    for ct in cand_tokens:
        mask = (tokens == ct).reshape(shape)
        eroded = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8))
        sel = eroded.ravel() > 0
        sel &= (dpix <= win)
        if not sel.any():
            continue
        pts = pix[sel]
        xi = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
        occ_val = this_occ_full[yi, xi] | wide.at_clipped(xi, yi)
        keep = ~occ_val
        xi, yi = xi[keep], yi[keep]
        if len(xi) == 0:
            continue
        ref = np.median(lab_crop[yi - py0, xi - px0], axis=0)
        distances[ct] = float(np.linalg.norm(tip_color - ref))

    if not distances:
        return dict(paint_token=map_token, paint_number=map_number, paint_ring=map_ring,
                    paint_candidates=cand_tokens, paint_distances={})

    ranked = sorted(distances.items(), key=lambda kv: kv[1])
    best_token, best_d = ranked[0]
    second_d = ranked[1][1] if len(ranked) > 1 else best_d * 1.3 + 1.0
    white_d = float(np.linalg.norm(tip_color - WHITE_LAB))
    ambiguous = ((second_d - best_d) / max(second_d, 1e-6) < 0.15) or (white_d < best_d)
    paint_token = map_token if ambiguous else best_token
    paint_number, paint_ring = cand_fields.get(paint_token, (map_number, map_ring))

    return dict(paint_token=paint_token, paint_number=paint_number, paint_ring=paint_ring,
                paint_candidates=cand_tokens, paint_distances=distances)


# --------------------------------------------------------------------------- #
# needle_w_mm: the needle's own measured width, a few pixels back from the
# tip (NEAR_D only -- the decision never reads the wider, farther-back
# measurements a full shaft-width profile would also produce), reusing the
# tip/far points and calibration model already settled above.
# --------------------------------------------------------------------------- #

NEAR_D = (2, 4, 6)
PROFILE_HALF_PX = 15.0
PROFILE_STEP_PX = 0.5
OUTER_BG_PX = 4.0
CLOSE_HALF_W_SAMPLES = 3
PEAK_SEARCH_HALF_PX = 6.0
MIN_PEAK_CONTRAST = 20.0

# Needle crop: worst-case sample offset from the tip is
# sqrt(PROFILE_HALF_PX^2 + max(NEAR_D)^2) = sqrt(15^2+6^2) ~= 16.16 px;
# + GaussianBlur(3x3) radius (1) + slack.
NEEDLE_CROP_HALF = 24


def _grid(start_px, u, v, steps_px, t_vals):
    pts = (start_px[None, None, :] + steps_px[:, None, None] * u[None, None, :]
           + t_vals[None, :, None] * v[None, None, :])
    return pts[..., 0].astype(np.float32), pts[..., 1].astype(np.float32)


def _max_filter_1d(arr, half_w):
    out = arr.copy()
    for k in range(1, half_w + 1):
        out[:-k] = np.maximum(out[:-k], arr[k:])
        out[k:] = np.maximum(out[k:], arr[:-k])
    return out


def _min_filter_1d(arr, half_w):
    out = arr.copy()
    for k in range(1, half_w + 1):
        out[:-k] = np.minimum(out[:-k], arr[k:])
        out[k:] = np.minimum(out[k:], arr[:-k])
    return out


def _closing_1d(arr, half_w):
    if half_w <= 0:
        return arr
    return _min_filter_1d(_max_filter_1d(arr, half_w), half_w)


def _crossing(prof, t_vals, peak_idx, half, direction):
    n = len(prof)
    i = peak_idx
    j = i + direction
    while 0 <= j < n and prof[j] >= half:
        i = j
        j = i + direction
    if j < 0 or j >= n:
        return None
    if prof[i] == prof[j]:
        return float(t_vals[i])
    frac = (prof[i] - half) / (prof[i] - prof[j])
    frac = min(max(float(frac), 0.0), 1.0)
    return float(t_vals[i] + frac * (t_vals[j] - t_vals[i]))


def _sample_crop(img_f32_crop, cx0, cy0, map_x, map_y, full_w, full_h):
    """Bilinear sample from a crop of the full image, but `in_bounds` is
    checked against the FULL image size, not the crop -- a sample is
    "in bounds" exactly when it would have been inside the uncropped
    image, regardless of where the crop happens to end."""
    in_bounds = (map_x >= 0) & (map_x <= full_w - 1) & (map_y >= 0) & (map_y <= full_h - 1)
    sampled = cv2.remap(img_f32_crop, map_x - cx0, map_y - cy0, interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    return sampled, in_bounds


def _measure_widths_near(diff_blur_crop, cx0, cy0, full_w, full_h, tip_px, u, v_perp,
                          distances=NEAR_D):
    tip_px = np.asarray(tip_px, dtype=np.float64)
    t_vals = np.arange(-PROFILE_HALF_PX, PROFILE_HALF_PX + 1e-6, PROFILE_STEP_PX)
    steps = -np.asarray(distances, dtype=np.float64)
    map_x, map_y = _grid(tip_px, u, v_perp, steps, t_vals)
    sampled, in_bounds = _sample_crop(diff_blur_crop, cx0, cy0, map_x, map_y, full_w, full_h)

    outer_mask = np.abs(t_vals) >= (PROFILE_HALF_PX - OUTER_BG_PX)
    widths = {}
    track_center = 0.0
    dist_list = list(distances)
    for d in sorted(distances):
        i = dist_list.index(d)
        prof = sampled[i]
        ib = in_bounds[i]
        if ib.sum() < 0.8 * len(t_vals):
            widths[d] = None
            continue
        env = _closing_1d(prof, CLOSE_HALF_W_SAMPLES)
        bg_sel = outer_mask & ib
        bg_val = float(np.median(env[bg_sel])) if bg_sel.any() else float(np.median(env[ib]))
        search_sel = (np.abs(t_vals - track_center) <= PEAK_SEARCH_HALF_PX) & ib
        if not search_sel.any():
            widths[d] = None
            continue
        peak_idx = int(np.argmax(np.where(search_sel, env, -np.inf)))
        peak_val = float(env[peak_idx])
        contrast = peak_val - bg_val
        if contrast < MIN_PEAK_CONTRAST:
            widths[d] = None
            continue
        track_center = float(t_vals[peak_idx])
        half = bg_val + contrast / 2.0
        left = _crossing(env, t_vals, peak_idx, half, -1)
        right = _crossing(env, t_vals, peak_idx, half, +1)
        widths[d] = float(right - left) if (left is not None and right is not None) else None
    return widths


def _needle_w_mm(diff_u8_full: np.ndarray, tip_px, far_px, model: CalibModel) -> float | None:
    tip_px = np.asarray(tip_px, dtype=np.float64)
    far_px = np.asarray(far_px, dtype=np.float64)
    u = _unit(tip_px - far_px)
    if np.linalg.norm(u) < 1e-6:
        return None
    v_perp = np.array([-u[1], u[0]], dtype=np.float64)
    mm_per_px_perp = _local_mm_per_px(model, tip_px, v_perp)
    if not np.isfinite(mm_per_px_perp):
        mm_per_px_perp = None

    h, w = diff_u8_full.shape[:2]
    half = NEEDLE_CROP_HALF
    xi, yi = int(round(float(tip_px[0]))), int(round(float(tip_px[1])))
    x0, x1 = max(0, xi - half), min(w, xi + half + 1)
    y0, y1 = max(0, yi - half), min(h, yi + half + 1)
    diff_crop_f32 = diff_u8_full[y0:y1, x0:x1].astype(np.float32)
    diff_blur_crop = cv2.GaussianBlur(diff_crop_f32, (3, 3), 0)

    widths = _measure_widths_near(diff_blur_crop, x0, y0, w, h, tip_px, u, v_perp, distances=NEAR_D)
    near_vals = [widths.get(d) for d in NEAR_D]
    good = [val for val in near_vals if val is not None]
    if not good or mm_per_px_perp is None:
        return None
    near_med_px = float(np.median(good))
    return near_med_px * mm_per_px_perp


# --------------------------------------------------------------------------- #
# extract_features -- the public entry point.
# --------------------------------------------------------------------------- #

def extract_features(
    bg_images: dict[int, np.ndarray],
    frame_images: dict[int, np.ndarray],
    calibration: dict[int, CameraCalibration],
    *,
    prior_dart_line_px: dict | None = None,
    visit_bg_images: dict[int, np.ndarray] | None = None,
    tip_overrides: dict | None = None,
) -> dict:
    """Per-camera features for one throw -- see the module docstring.

    `tip_overrides`: {camera: (tip_px, far_px) or None} from a cross-view
    blob pick. A camera listed there uses that blob and end instead of its
    own detect_tip call (None = the pick dropped it), and the parallax end
    choice is skipped -- the pick already settled the ends in 3D. Shape
    readings (taper, elongation, texture tie-break, prior-dart suspicion)
    come from detect_tip and are kept only when the pick IS that same
    detection, same ends."""
    cameras = sorted(set(bg_images) & set(frame_images) & set(calibration))
    models: dict[int, CalibModel] = {}
    for cam in cameras:
        model = build_calib_model(calibration[cam])
        if model.ok:
            models[cam] = model
        # A calibration that fails to build a model is treated exactly like
        # having no calibration for that camera at all -- it is simply
        # left out of everything below.

    # ---- tip detection + cross-camera end choice (the floor) ----
    raw_tips, raw_diag = {}, {}
    for cam in sorted(models):
        bg, dart = bg_images[cam], frame_images[cam]
        pline = prior_dart_line_px.get(cam) if prior_dart_line_px else None
        try:
            r = detect_tip(bg, dart, prior_dart_line_px=pline)
            if r.ok and r.tip_px is not None and r.far_end_px is not None:
                raw_tips[cam] = dict(ok=True, tip_px=r.tip_px, far_px=r.far_end_px)
                d = r.diagnostics or {}
                raw_diag[cam] = dict(
                    width_ratio=d.get("width_ratio"),
                    elongation_ratio=d.get("elongation_ratio"),
                    texture_tiebreak_used=bool(d.get("texture_tiebreak_used")),
                    contam_suspected=bool(d.get("prior_dart_contamination_suspected", False)),
                )
            else:
                raw_tips[cam] = dict(ok=False)
        except Exception as e:
            raw_tips[cam] = dict(ok=False, error=repr(e))
    if tip_overrides is not None:
        chosen, parallax_diag = {}, dict(method="cross_view")
        for cam in sorted(models):
            if cam not in tip_overrides:
                continue
            ov = tip_overrides[cam]
            if ov is None:
                raw_diag.pop(cam, None)
                continue
            tip, far = np.asarray(ov[0], float), np.asarray(ov[1], float)
            own = raw_tips.get(cam) or {}
            same = (own.get("ok")
                    and np.linalg.norm(tip - np.asarray(own["tip_px"], float)) < _SAME_DETECTION_PX
                    and np.linalg.norm(far - np.asarray(own["far_px"], float)) < _SAME_DETECTION_PX)
            if not same:
                raw_diag[cam] = dict(width_ratio=None, elongation_ratio=None,
                                     texture_tiebreak_used=False, contam_suspected=False)
            chosen[cam] = (tip, far)
    else:
        chosen, parallax_diag = parallax_choose_ends(models, raw_tips)

    cams_out: dict[str, dict] = {}
    board_xy: dict[str, tuple[float, float]] = {}

    for cam in sorted(models):
        if cam not in chosen:
            cams_out[str(cam)] = dict(ok=False, error="no tip detected")
            continue
        tip_px, far_px = chosen[cam]
        model = models[cam]
        diag = raw_diag.get(cam, {})

        X, Y = _board_xy1(model, tip_px)
        number, ring, r_mm, theta_deg, k = label_at_board_xy(X, Y)
        token = token_of(number, ring)
        margin_mm = _nearest_boundary_mm(r_mm, theta_deg, k)

        u = _unit(np.asarray(tip_px, float) - np.asarray(far_px, float))
        mm_per_px = (_local_mm_per_px(model, np.asarray(tip_px, float), u)
                     if np.linalg.norm(u) > 1e-6 else None)
        if mm_per_px is not None and not np.isfinite(mm_per_px):
            mm_per_px = None

        bg_bgr, dart_bgr = bg_images[cam], frame_images[cam]
        # One full-frame channel-max |bg-dart| diff, reused for clarity,
        # entry occlusion and the needle-width crop below.
        diff_u8 = _abs_diff_max(bg_bgr, dart_bgr)

        noise_med, mad = _clarity_stats(diff_u8)
        h, w = diff_u8.shape
        xi, yi = int(round(tip_px[0])), int(round(tip_px[1]))
        x0p, x1p = max(0, xi - 2), min(w, xi + 3)
        y0p, y1p = max(0, yi - 2), min(h, yi + 3)
        # float32 to match the diff's own dtype exactly where it is cast
        # before being sliced/averaged.
        patch = diff_u8[y0p:y1p, x0p:x1p].astype(np.float32)
        patch_mean = float(patch.mean()) if patch.size else 0.0
        clarity_z = (patch_mean - noise_med) / max(mad, 1.0)

        # Entry occlusion: an earlier dart of this visit already lying
        # where this tip lands.
        s0 = None
        if visit_bg_images is not None and cam in visit_bg_images:
            cand = visit_bg_images[cam]
            if cand.shape == frame_images[cam].shape:
                s0 = cand
        sn = bg_images[cam]
        wide = _wide_window(s0, sn, tip_px, (h, w))
        this_occlusion = diff_u8 > 30
        entry_occluded = wide.end_on_old(tip_px)

        paint = _paint_variant(bg_bgr, model, np.asarray(tip_px, float), token, number, ring,
                                wide, this_occlusion)
        dists = paint.get("paint_distances") or {}
        if len(dists) >= 2:
            ranked = sorted(dists.values())
            paint_margin_ratio = (ranked[1] - ranked[0]) / max(ranked[1], 1e-6)
        else:
            paint_margin_ratio = None

        needle_w_mm = _needle_w_mm(diff_u8, tip_px, far_px, model)

        # Local Jacobian of the pixel->board projection at the tip, along
        # and across the needle axis, reusing the model and board point
        # already settled above.
        d_axis = np.asarray(tip_px, float) - np.asarray(far_px, float)
        n_axis = float(np.linalg.norm(d_axis))
        if n_axis > 1e-9:
            u_j = d_axis / n_axis
            v_j = np.array([-u_j[1], u_j[0]], dtype=np.float64)
            tp = np.asarray(tip_px, float)
            pts4 = np.stack([tp + u_j, tp - u_j, tp + v_j, tp - v_j], axis=0)
            B = pixel_to_board_xy(model, pts4)
            b_pu, b_mu, b_pv, b_mv = B
            jac_u = [float((b_pu[0] - b_mu[0]) / 2.0), float((b_pu[1] - b_mu[1]) / 2.0)]
            jac_v = [float((b_pv[0] - b_mv[0]) / 2.0), float((b_pv[1] - b_mv[1]) / 2.0)]
        else:
            jac_u = None
            jac_v = None

        cams_out[str(cam)] = dict(
            ok=True,
            tip_px=[float(tip_px[0]), float(tip_px[1])],
            number=number, ring=ring, token=token,
            margin_mm=float(margin_mm),
            mm_per_px=(float(mm_per_px) if mm_per_px is not None else None),
            width_ratio=(float(diag["width_ratio"]) if diag.get("width_ratio") is not None else None),
            elongation_ratio=(float(diag["elongation_ratio"]) if diag.get("elongation_ratio") is not None else None),
            texture_tiebreak_used=bool(diag.get("texture_tiebreak_used", False)),
            contam_suspected=bool(diag.get("contam_suspected", False)),
            entry_occluded=bool(entry_occluded),
            clarity_z=float(clarity_z),
            paint_token=paint["paint_token"], paint_number=paint["paint_number"], paint_ring=paint["paint_ring"],
            paint_margin_ratio=(float(paint_margin_ratio) if paint_margin_ratio is not None else None),
            board_xy=[float(X), float(Y)],
            needle_w_mm=(float(needle_w_mm) if needle_w_mm is not None else None),
            jac_u=jac_u, jac_v=jac_v,
        )
        board_xy[str(cam)] = (float(X), float(Y))

    # Cross-camera agreement: how far this camera's board point sits from
    # the mean of the others.
    ok_cams = [c for c in cams_out if cams_out[c].get("ok")]
    for c in ok_cams:
        others = [board_xy[o] for o in ok_cams if o != c]
        if others:
            cx = sum(o[0] for o in others) / len(others)
            cy = sum(o[1] for o in others) / len(others)
            d = math.hypot(board_xy[c][0] - cx, board_xy[c][1] - cy)
        else:
            d = None
        cams_out[c]["agree_mm"] = (float(d) if d is not None else None)

    for cam, t in raw_tips.items():
        if not t.get("ok") and str(cam) not in cams_out:
            cams_out[str(cam)] = dict(ok=False, error="detect_tip failed")

    return dict(cams=cams_out, parallax=parallax_diag)
