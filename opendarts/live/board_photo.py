"""A real photo of the board for the scoring page, and what it takes to
draw each dart on it the way a player at the oche sees it.

THE PHOTO. Every camera's empty-board frame is warped onto the board
plane and the three are combined into one straight-on picture. Each spot
comes from the camera that sees it most squarely; a soft blend of all
three would double anything that is not flat. The number ring is the
case that forced this: its digits are raised wire, so two cameras put a
digit in two slightly different places, and the 11 -- which sits exactly
on a tie between the two left-hand cameras -- came out as four bars. In
the number ring the camera is therefore chosen once per NUMBER, at that
number's centre, so a join between cameras falls in the gap between two
numbers and never through one.

THE JOINS ARE ALIGNED, MEASURED. Where the per-number choice hands the
ring from one camera to the next, the two cameras rarely agree exactly
about where the outer wire and the digits are: calibration is fitted to
the scoring wires and is least certain at the rim, and the ring stands
proud of the board. On a real rig that showed as a 5-7 mm step in the
outer wire at every hand-over. So once per calibration, from the first
photo's own empty-board frames, each join is measured -- the shift that
best lines up the two cameras' edges there -- and each camera's ring is
moved half of it toward the other, the correction fading to nothing over
the next few numbers (`BoardPhotoMaps.align_joins`). Nothing here is
specific to one board or one camera layout: it is whatever these frames
say, and a join the frames cannot settle is left as it was. The scoring
area is never moved -- the correction starts past the spider wires' tips.

The expensive part (where every output pixel lands in every camera) only
changes on a recalibration, so it is computed once per calibration and
cached; a photo is then three remaps and a blend, on a background thread.

WHEN. One at Start, from the session's first frames, and one after each
takeout (and manual Reset), from the lifecycle's fresh empty-board
reference -- never inside a dart's burst, where the Pi has no CPU to
spare. A takeout photo is skipped when the board has not visibly changed
since the last one (see `board_unchanged()`): most turns end on the same
board they started on.

THE DARTS are drawn by the dashboard, not baked into the photo, so the
photo only changes when the board is empty. This module supplies the two
per-dart facts the drawing needs, both computed from what scoring
already produced:

  * `dart_axis()` -- the shaft direction in 3D, from Talos' per-camera
    shaft lines (each line and its camera centre span a plane; the shaft
    lies in all of them). Zeus runs Talos on every throw, so this costs
    nothing but the arithmetic.
  * `flight_color()` -- the flights' colour, sampled from the scored
    frames where they changed against the empty board.

Board frame throughout: mm, bull at the origin, x right, y up (20 at the
top), +z out of the board toward the thrower.
"""
from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from opendarts.capture.lazy_frame import as_frames, handles_of
from opendarts.geometry.board import DOUBLE_OUTER_RADIUS_MM, SECTOR_ANGLE_DEG

log = logging.getLogger(__name__)

#: The photo covers +/- this many mm: the whole number ring and a little
#: of the surround.
HALF_MM = 235.0
#: Output size in px. 1000 px over 470 mm is ~2.1 px/mm -- sharper than a
#: large dashboard board needs, and ~200 KB as a JPEG.
SIZE_PX = 1000
PX_PER_MM = SIZE_PX / (2 * HALF_MM)
JPEG_QUALITY = 88

#: Outside the double wire is the number ring, chosen per number (see the
#: module docstring). A little past the wire so the wire itself is not split.
_NUMBER_RING_FROM_MM = DOUBLE_OUTER_RADIUS_MM + 2.0
#: Where the number ring's digits sit, for choosing its camera.
_NUMBER_RADIUS_MM = 200.0
#: The in-board blend is this power of the view cosine: high enough that
#: the squarest camera all but wins, low enough that joins are not hard.
_BLEND_POWER = 40
#: Softening of the per-number joins, in output px (~1.5 mm).
_JOIN_SOFTEN_PX = 3.0
#: Join alignment (see the module docstring). Measured over the number
#: ring's band, within this many degrees either side of a join ...
_ALIGN_BAND_MM = (188.0, 228.0)
_ALIGN_HALF_DEG = 9.0
#: ... searching shifts up to this many output px (~7.5 mm) each way.
_ALIGN_MAX_SHIFT_PX = 16
#: Accepted only when the edges then disagree at most this fraction of
#: what they did -- anything less is a join the frames cannot settle.
_ALIGN_MIN_GAIN = 0.85
#: The correction fades to nothing this many degrees from its join ...
_ALIGN_TAPER_DEG = 40.0
#: ... and fades in radially across the black band, from past the spider
#: wires' tips to where the digits begin, so the scoring area never moves.
_ALIGN_RAMP_MM = (178.0, 188.0)

# Where Talos' flights sit, for colour sampling: mm out from the tip,
# along the shaft (a steel-tip dart is ~150 mm; the flights are its last
# ~40), and how far to either side of the shaft to look.
_FLIGHT_FROM_MM = 95.0
_FLIGHT_TO_MM = 125.0
_FLIGHT_HALF_SPAN_MM = 14.0
#: A pixel counts as dart when it moved at least this much against the
#: empty board (max over channels, 0-255).
_CHANGED_MIN = 40
#: Fewer changed pixels than this across all cameras and the colour is
#: not trusted.
_MIN_FLIGHT_PIXELS = 60

# SKIPPING AN UNCHANGED BOARD. Each photo keeps a summary of the frames
# it came from: every camera's frame in grey at 1/4 scale (the
# lifecycle's own detection scale), inside the double ring. A takeout's
# board counts as unchanged when, in EVERY camera, both
#   * the mean absolute difference is at most UNCHANGED_MAX_MEAN_DIFF
#     grey levels -- exposure/lighting has not moved, and
#   * at most UNCHANGED_MAX_CHANGED_FRAC of the disc changed by
#     _CHANGED_PX_DIFF or more -- nothing new sits in the board (a dart
#     left in barely moves the mean but is a patch of large changes).
# Measured 2026-09-26 on the 6 sessions in data/packages, each visit's
# first-dart bg (an empty board) against the previous visit's: noise
# alone is ~1.0-1.4 grey levels mean, and 194 of 214 pairs (91%) fall
# under both limits; compared as live, with the last photo actually
# rendered, 193 of 214 takeouts (90%) skip. Against the same package's
# one-dart frame the rule called all 210 darts on the board changed;
# the 10 it did not were OUT packages, with no dart on the board to see.
# The check costs ~0.9 ms of CPU against ~73 ms for the render (both on
# an M-series Mac, 3 x 1280x720), and runs on the render thread.
_SUMMARY_SCALE = 4
UNCHANGED_MAX_MEAN_DIFF = 2.0
_CHANGED_PX_DIFF = 25
UNCHANGED_MAX_CHANGED_FRAC = 0.002


def _rotation_and_centre(cal: Any) -> tuple[np.ndarray, np.ndarray]:
    R, _ = cv2.Rodrigues(np.asarray(cal.rvec, dtype=np.float64))
    centre = -R.T @ np.asarray(cal.tvec, dtype=np.float64).reshape(3)
    return R, centre


def calibration_signature(calibrations: Mapping[int, Any]) -> str:
    """Changes whenever any camera's solve does -- the cache key for the
    warp maps."""
    h = hashlib.sha1()
    for cam in sorted(calibrations):
        cal = calibrations[cam]
        h.update(str(cam).encode())
        for arr in (cal.camera_matrix, cal.dist_coeffs, cal.rvec, cal.tvec):
            h.update(np.ascontiguousarray(np.asarray(arr, dtype=np.float64)).tobytes())
    return h.hexdigest()


@dataclass
class _CameraMap:
    map1: np.ndarray       # cv2.convertMaps fixed-point maps
    map2: np.ndarray
    weight: np.ndarray     # float32, SIZE x SIZE, already normalised across cameras
    overlap: np.ndarray    # bool, where every camera sees the board (for brightness matching)
    overlap_idx: np.ndarray  # np.flatnonzero(overlap), so a render gathers without re-deriving it


class BoardPhotoMaps:
    """Everything about the warp that depends only on the calibration."""

    def __init__(self, calibrations: Mapping[int, Any], frame_sizes: Mapping[int, tuple[int, int]]):
        self.signature = calibration_signature(calibrations)
        ys, xs = np.mgrid[0:SIZE_PX, 0:SIZE_PX].astype(np.float32)
        X = (xs - SIZE_PX / 2) / PX_PER_MM
        Y = (SIZE_PX / 2 - ys) / PX_PER_MM
        radius = np.hypot(X, Y)
        pts = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size, np.float32)], 1).astype(np.float64)
        # Which number each spot of the ring belongs to (0..19, clockwise
        # from +x in board terms -- only the grouping matters).
        step = np.radians(SECTOR_ANGLE_DEG)
        sector = np.round(np.arctan2(Y, X) / step).astype(np.int32) % 20
        centres = np.array([[
            _NUMBER_RADIUS_MM * np.cos(k * step), _NUMBER_RADIUS_MM * np.sin(k * step), 0.0,
        ] for k in range(20)])
        in_ring = radius > _NUMBER_RING_FROM_MM

        cams = sorted(calibrations)
        self._grid = (xs, ys, radius, np.degrees(np.arctan2(Y, X)) % 360.0)
        self._float_maps: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.join_shifts: list[dict[str, Any]] = []
        maps, inside, cos_here, cos_number = {}, [], [], []
        for cam in cams:
            cal = calibrations[cam]
            _, centre = _rotation_and_centre(cal)
            uv, _ = cv2.projectPoints(pts, cal.rvec, cal.tvec, cal.camera_matrix, cal.dist_coeffs)
            mx = uv[:, 0, 0].reshape(SIZE_PX, SIZE_PX).astype(np.float32)
            my = uv[:, 0, 1].reshape(SIZE_PX, SIZE_PX).astype(np.float32)
            w, h = frame_sizes[cam]
            inside.append((mx >= 0) & (mx < w - 1) & (my >= 0) & (my < h - 1))
            maps[cam] = cv2.convertMaps(mx, my, cv2.CV_16SC2)
            self._float_maps[cam] = (mx, my)
            ray = centre[None, :] - pts
            cos_here.append(np.clip(ray[:, 2] / np.linalg.norm(ray, axis=1), 0, 1).reshape(SIZE_PX, SIZE_PX))
            ray_n = centre[None, :] - centres
            cos_number.append(np.clip(ray_n[:, 2] / np.linalg.norm(ray_n, axis=1), 0, 1))
        inside_a = np.array(inside)
        # One camera per number: the squarest one, a tie to the first.
        best = np.argmax(np.array(cos_number), axis=0)          # per sector
        # The hand-overs: (angle of the join, camera before it, camera after
        # it), going counter-clockwise in board terms.
        self._joins = [((k + 0.5) * SECTOR_ANGLE_DEG % 360.0, cams[best[k]], cams[best[(k + 1) % 20]])
                       for k in range(20) if best[k] != best[(k + 1) % 20]]
        pick = np.array([best[sector] == i for i in range(len(cams))], dtype=np.float32)
        pick = np.array([cv2.GaussianBlur(m, (0, 0), _JOIN_SOFTEN_PX) for m in pick]) * inside_a
        board = np.where(inside_a, np.array(cos_here), 0.0) ** _BLEND_POWER
        weight = np.where(in_ring[None], pick, board).astype(np.float32)
        weight /= np.maximum(weight.sum(axis=0), 1e-6)
        overlap = inside_a.all(axis=0) & (radius < HALF_MM - 10)
        overlap_idx = np.flatnonzero(overlap)
        self.cameras: dict[int, _CameraMap] = {
            cam: _CameraMap(maps[cam][0], maps[cam][1], weight[i], overlap, overlap_idx)
            for i, cam in enumerate(cams)
        }
        # Outside the board and its surround: fade to the dashboard's own
        # dark, so the photo's square corners never show.
        self.fade = np.clip((HALF_MM - 3.0 - radius) / 6.0, 0, 1).astype(np.float32)[..., None]
        # Where the double ring's disc falls in each camera at summary
        # scale -- what board_unchanged() compares.
        t = np.linspace(0, 2 * np.pi, 180, endpoint=False)
        rim = np.stack([DOUBLE_OUTER_RADIUS_MM * np.cos(t), DOUBLE_OUTER_RADIUS_MM * np.sin(t),
                        np.zeros_like(t)], 1)
        self.summary_masks: dict[int, np.ndarray] = {}
        for cam in cams:
            cal = calibrations[cam]
            uv, _ = cv2.projectPoints(rim, cal.rvec, cal.tvec, cal.camera_matrix, cal.dist_coeffs)
            w, h = frame_sizes[cam]
            mask = np.zeros((h // _SUMMARY_SCALE, w // _SUMMARY_SCALE), np.uint8)
            cv2.fillPoly(mask, [np.round(uv.reshape(-1, 2) / _SUMMARY_SCALE).astype(np.int32)], 1)
            self.summary_masks[cam] = mask.astype(bool)

    def summary(self, frames: Mapping[int, np.ndarray]) -> dict[int, np.ndarray]:
        """Each camera's board disc in grey at 1/4 scale, flattened --
        what the next photo's frames are compared with."""
        out = {}
        for cam, frame in frames.items():
            mask = self.summary_masks.get(cam)
            if mask is None:
                continue
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            small = cv2.resize(grey, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
            out[cam] = small[mask].astype(np.int16)
        return out

    def align_joins(self, frames: Mapping[int, np.ndarray]) -> list[dict[str, Any]]:
        """Measure each camera-to-camera join in the number ring from these
        (empty-board) frames and move each camera's ring half the measured
        shift toward the other, fading out away from the join. Once per
        calibration; returns what it found, for the log. A join whose
        frames cannot settle it (no clear improvement) is left alone."""
        xs, ys, radius, theta = self._grid
        cams = [c for c in self.cameras if c in frames]
        if len(cams) < 2 or not self._joins:
            return []

        def edges(cam: int) -> np.ndarray:
            m = self.cameras[cam]
            g = cv2.cvtColor(cv2.remap(frames[cam], m.map1, m.map2, cv2.INTER_LINEAR), cv2.COLOR_BGR2GRAY)
            g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.5)
            return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))

        E = {c: edges(c) for c in cams}
        band = (radius >= _ALIGN_BAND_MM[0]) & (radius <= _ALIGN_BAND_MM[1])
        rng = _ALIGN_MAX_SHIFT_PX
        found: list[dict[str, Any]] = []
        disp = {c: [np.zeros(radius.shape, np.float32), np.zeros(radius.shape, np.float32)] for c in cams}
        for angle, a, b in self._joins:
            if a not in E or b not in E:
                continue
            dth = (theta - angle + 180.0) % 360.0 - 180.0
            mask = band & (np.abs(dth) <= _ALIGN_HALF_DEG)
            rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
            if not len(rows) or not len(cols):
                continue
            y0, y1 = max(rows[0], rng), min(rows[-1], SIZE_PX - 1 - rng)
            x0, x1 = max(cols[0], rng), min(cols[-1], SIZE_PX - 1 - rng)
            if y1 <= y0 or x1 <= x0:
                continue
            ea, mk = E[a][y0:y1 + 1, x0:x1 + 1], mask[y0:y1 + 1, x0:x1 + 1]
            base = float(np.abs(ea - E[b][y0:y1 + 1, x0:x1 + 1])[mk].mean())
            best = (base, 0, 0)
            for dy in range(-rng, rng + 1):
                for dx in range(-rng, rng + 1):
                    s = float(np.abs(ea - E[b][y0 + dy:y1 + 1 + dy, x0 + dx:x1 + 1 + dx])[mk].mean())
                    if s < best[0]:
                        best = (s, dx, dy)
            score, dx, dy = best
            ok = base > 0 and score <= base * _ALIGN_MIN_GAIN and (dx or dy)
            found.append({"angle_deg": round(angle, 1), "cameras": [a, b], "shift_px": [dx, dy],
                          "shift_mm": round(float(np.hypot(dx, dy)) / PX_PER_MM, 1),
                          "edge_diff": [round(base, 1), round(score, 1)], "applied": bool(ok)})
            if not ok:
                continue
            # b's content at p + (dx, dy) is what a shows at p: each moves
            # half-way, a on its side of the join and b on its own.
            fade = np.clip(1.0 - np.abs(dth) / _ALIGN_TAPER_DEG, 0.0, 1.0)
            wa, wb = fade * (dth <= 0), fade * (dth > 0)
            disp[b][0] += wb * dx / 2.0; disp[b][1] += wb * dy / 2.0
            disp[a][0] -= wa * dx / 2.0; disp[a][1] -= wa * dy / 2.0
        if not any(j["applied"] for j in found):
            self.join_shifts = found
            return found
        ramp = np.clip((radius - _ALIGN_RAMP_MM[0]) / (_ALIGN_RAMP_MM[1] - _ALIGN_RAMP_MM[0]), 0.0, 1.0).astype(np.float32)
        for c in cams:
            mx, my = self._float_maps[c]
            sx = (xs + disp[c][0] * ramp).astype(np.float32)
            sy = (ys + disp[c][1] * ramp).astype(np.float32)
            m = self.cameras[c]
            m.map1, m.map2 = cv2.convertMaps(cv2.remap(mx, sx, sy, cv2.INTER_LINEAR),
                                             cv2.remap(my, sx, sy, cv2.INTER_LINEAR), cv2.CV_16SC2)
        self.join_shifts = found
        return found

    def render(self, frames: Mapping[int, np.ndarray]) -> np.ndarray:
        """BGR photo from each camera's frame. Cameras missing from
        `frames` are left out and the others' weights stretch to cover."""
        cams = [c for c in self.cameras if c in frames]
        if not cams:
            raise ValueError("board photo: no frame for any calibrated camera")
        warped = [
            cv2.remap(frames[c], self.cameras[c].map1, self.cameras[c].map2, cv2.INTER_LINEAR).astype(np.float32)
            for c in cams
        ]
        # Match each camera's brightness to the others' where all overlap,
        # or a join shows as a step in exposure.
        overlap = self.cameras[cams[0]].overlap
        if overlap.any():
            # The same pixels, in the same order, as w[overlap] -- so the
            # same means to the bit -- gathered by the index list cached
            # with the maps instead of re-deriving it from the mask for
            # every camera of every photo: 113 -> 43 ms of a ~270 ms
            # render on the Pi 5.
            idx = self.cameras[cams[0]].overlap_idx
            means = [
                np.take(w.reshape(-1, w.shape[2]), idx, axis=0).mean(axis=0)
                if w.ndim == 3 else w[overlap].mean(axis=0)
                for w in warped
            ]
            target = np.mean(means, axis=0)
            warped = [w * (target / np.maximum(m, 1e-3)) for w, m in zip(warped, means)]
        weights = [self.cameras[c].weight for c in cams]
        total = np.maximum(np.sum(weights, axis=0), 1e-6)
        out = sum(w[..., None] * img for w, img in zip(weights, warped)) / total[..., None]
        out = out * self.fade + np.float32(12.0) * (1 - self.fade)
        return np.clip(out, 0, 255).astype(np.uint8)


def board_unchanged(before: Mapping[int, np.ndarray], after: Mapping[int, np.ndarray]) -> bool:
    """True when two summaries (BoardPhotoMaps.summary) show the same
    board, by the limits above. Different cameras is always a change."""
    if not before or set(before) != set(after):
        return False
    for cam, a in before.items():
        b = after[cam]
        if a.shape != b.shape or not a.size:
            return False
        diff = np.abs(a - b)
        if diff.mean() > UNCHANGED_MAX_MEAN_DIFF:
            return False
        if np.count_nonzero(diff >= _CHANGED_PX_DIFF) > UNCHANGED_MAX_CHANGED_FRAC * diff.size:
            return False
    return True


def dart_axis(engine_diagnostics: Mapping[str, Any] | None, calibrations: Mapping[int, Any]) -> list[float] | None:
    """Unit shaft direction (tip -> flights) in the board frame, or None.

    Reads Talos' per-camera shaft lines from a Zeus result's
    `sub_results["Talos"]`, or from a Talos result directly. Needs two
    cameras; None when Talos did not run or found fewer lines."""
    if not engine_diagnostics:
        return None
    diag = engine_diagnostics
    sub = (diag.get("sub_results") or {}).get("Talos")
    if sub is not None:
        diag = sub.get("diagnostics") or {}
    lines = diag.get("line_px") or {}
    normals = []
    for cam, line in lines.items():
        try:
            cal = calibrations[int(cam)]
        except (KeyError, ValueError):
            continue
        if not isinstance(line, Mapping) or line.get("dropped") or "p1_px" not in line or "p2_px" not in line:
            continue
        R, _ = _rotation_and_centre(cal)
        px = np.array([line["p1_px"], line["p2_px"]], dtype=np.float64).reshape(-1, 1, 2)
        und = cv2.undistortPoints(px, np.asarray(cal.camera_matrix, float), np.asarray(cal.dist_coeffs, float)).reshape(-1, 2)
        n = np.cross(R.T @ np.array([*und[0], 1.0]), R.T @ np.array([*und[1], 1.0]))
        norm = np.linalg.norm(n)
        if norm > 0:
            normals.append(n / norm)
    if len(normals) < 2:
        return None
    _, _, vt = np.linalg.svd(np.array(normals))
    axis = vt[-1]
    if axis[2] < 0:
        axis = -axis
    # A shaft lying almost flat on the board is a failed solve, not a dart.
    if axis[2] < 0.5:
        return None
    return [float(v) for v in axis]


def flight_color(
    frames: Mapping[int, np.ndarray],
    bg_frames: Mapping[int, np.ndarray],
    calibrations: Mapping[int, Any],
    tip_xy_mm: tuple[float, float],
    axis: list[float],
) -> str | None:
    """The flights' colour as '#rrggbb', or None when too little of them
    could be told apart from the board.

    Samples, in every camera, the pixels around where the flights must be
    (from the tip and the shaft direction) that changed against the empty
    board, and takes their median. Lighting shifts it, so it is close,
    not exact."""
    a = np.asarray(axis, dtype=np.float64)
    tip = np.array([tip_xy_mm[0], tip_xy_mm[1], 0.0])
    ref = np.array([0.0, 1.0, 0.0]) if abs(a[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(a, ref); e1 /= np.linalg.norm(e1)
    e2 = np.cross(a, e1)
    samples = []
    ss = np.linspace(_FLIGHT_FROM_MM, _FLIGHT_TO_MM, 7)
    offs = np.linspace(-_FLIGHT_HALF_SPAN_MM, _FLIGHT_HALF_SPAN_MM, 7)
    pts = np.array([tip + a * s + e1 * u + e2 * v for s in ss for u in offs for v in offs])
    for cam, frame in frames.items():
        bg = bg_frames.get(cam)
        cal = calibrations.get(cam)
        if bg is None or cal is None or bg.shape != frame.shape:
            continue
        uv, _ = cv2.projectPoints(pts, cal.rvec, cal.tvec, cal.camera_matrix, cal.dist_coeffs)
        uv = uv.reshape(-1, 2)
        h, w = frame.shape[:2]
        uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)]
        if not len(uv):
            continue
        # Only the flights' own patch is compared, never the whole frame:
        # this runs on the scoring path.
        x0, y0 = np.maximum(np.floor(uv.min(axis=0)).astype(int) - 4, 0)
        x1, y1 = np.minimum(np.ceil(uv.max(axis=0)).astype(int) + 5, [w, h])
        mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        for u, v in uv:
            cv2.circle(mask, (int(u) - x0, int(v) - y0), 3, 255, -1)
        patch = frame[y0:y1, x0:x1]
        diff = cv2.absdiff(patch, bg[y0:y1, x0:x1]).max(axis=2)
        sel = (mask > 0) & (diff >= _CHANGED_MIN)
        if sel.any():
            samples.append(patch[sel])
    if not samples:
        return None
    px = np.concatenate(samples)
    if len(px) < _MIN_FLIGHT_PIXELS:
        return None
    b, g, r = (int(round(v)) for v in np.median(px, axis=0))
    return f"#{r:02x}{g:02x}{b:02x}"


class BoardPhotoRenderer:
    """Renders board photos off the capture thread and hands each finished
    JPEG to a callback. Keeps the warp maps for the current calibration;
    a new calibration rebuilds them (once, on the render thread). Also
    keeps the summary of the last photo's frames, for skipping a board
    that has not changed."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._maps: BoardPhotoMaps | None = None
        self._busy = False
        self._pending: tuple | None = None
        # (calibration signature, summary) of the last photo rendered.
        # Only the render thread touches it -- one render at a time.
        self._last: tuple[str, dict[int, np.ndarray]] | None = None

    def submit(
        self,
        frames: Mapping[int, np.ndarray],
        calibrations: Mapping[int, Any],
        on_done: Callable[[bytes], None],
        *,
        skip_if_unchanged: bool = False,
    ) -> None:
        """Queue a render. Never blocks and never raises: if one is already
        running, this one replaces any queued one (only the newest empty
        board matters). With `skip_if_unchanged`, a board that matches the
        last photo's (same calibration, board_unchanged()) is not rendered
        and `on_done` is not called; the check runs on the render thread
        too, so the caller pays nothing either way."""
        # Copied without decoding: a lazy frame (detect_from_small_decode,
        # opendarts/capture/lazy_frame.py) is decoded to full resolution on
        # the render thread when the render reads it, not here on the
        # capture loop's.
        job = (as_frames(dict(handles_of(frames))), dict(calibrations), on_done,
               skip_if_unchanged)
        with self._lock:
            if self._busy:
                self._pending = job
                return
            self._busy = True
        threading.Thread(target=self._run, args=(job,), name="board-photo", daemon=True).start()

    def _run(self, job: tuple) -> None:
        while job is not None:
            frames, calibrations, on_done, skip_if_unchanged = job
            try:
                jpeg = self.render_jpeg(frames, calibrations, skip_if_unchanged=skip_if_unchanged)
                if jpeg is not None:
                    on_done(jpeg)
            except Exception:  # noqa: BLE001 -- a photo is decoration; scoring must never feel it
                log.warning("board photo: render failed", exc_info=True)
            with self._lock:
                job, self._pending = self._pending, None
                if job is None:
                    self._busy = False

    def render_jpeg(
        self,
        frames: Mapping[int, np.ndarray],
        calibrations: Mapping[int, Any],
        *,
        skip_if_unchanged: bool = False,
    ) -> bytes | None:
        """The photo as JPEG bytes -- or None when `skip_if_unchanged` and
        the board is the last photo's (never None without it)."""
        cams = {c: calibrations[c] for c in frames if c in calibrations}
        if not cams:
            raise ValueError("board photo: no calibrated camera among the frames")
        sig = calibration_signature(cams)
        maps = self._maps
        if maps is None or maps.signature != sig:
            sizes = {c: (frames[c].shape[1], frames[c].shape[0]) for c in cams}
            maps = BoardPhotoMaps(cams, sizes)
            # Once per calibration, from these empty-board frames: line the
            # cameras up at their joins. Decoration -- never costs the photo.
            try:
                joins = maps.align_joins(frames)
                log.info("board photo: joins %s", [
                    f"{j['angle_deg']:.0f}deg cam{j['cameras'][0]}|cam{j['cameras'][1]} "
                    f"{j['shift_mm']}mm {'aligned' if j['applied'] else 'left'}" for j in joins])
            except Exception:  # noqa: BLE001
                log.warning("board photo: join alignment failed -- joins left as measured", exc_info=True)
            self._maps = maps
        summary = maps.summary({c: frames[c] for c in cams})
        last = self._last
        if skip_if_unchanged and last is not None and last[0] == sig and board_unchanged(last[1], summary):
            log.info("board photo: board unchanged since the last photo, not re-rendered")
            return None
        photo = maps.render(frames)
        ok, buf = cv2.imencode(".jpg", photo, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise RuntimeError("board photo: JPEG encode failed")
        self._last = (sig, summary)
        return buf.tobytes()


#: One per process: the capture loop is one per process too.
RENDERER = BoardPhotoRenderer()
