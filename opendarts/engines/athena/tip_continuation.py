"""Continue a camera's tip down the shaft when the dart shows up in pieces
(2026-09-28).

**The failure this fixes**: Athena's tip is the far end of its diff blob,
cut at `crossing_detection.DIFF_THRESHOLD` (31) and joined by a 19px
close. On about 9% of used camera views (2026-09 corpus, 1,618 judged
darts) the dart does not come out as one blob: the dark lower barrel and
the needle sit under that threshold against a dark bed, or past a gap the
close cannot bridge, so the tip lands at the end of the bright upper part,
30-50px short. Athena reads each camera's tip straight onto the board
plane, so a short tip is a several-mm board error on its own, not one ray
among three to be outvoted.

**What this does**: from the tip, walk AWAY from the flight along the
shaft's own image axis, inside a +-`CORRIDOR_HALF_PX` corridor, on the
same blurred diff Athena already thresholds -- but cut at the lower
`CONTINUATION_THRESHOLD`. Covered stretches chain outward: one that starts
right where the chain ends extends it; one across a gap must itself be
at least `MIN_STRETCH_PX` long and at least `GAP_STRETCH_FRACTION` of the
gap it jumps (a long jump to a few stray pixels is a speck, not the lower
shaft -- 018-S17 jumped 30px to one). The lower threshold reaches into
the blur halo past the real end, so the end then steps back to the
half-maximum between the local noise floor and the level just inside it.

**Evidence** (offline replay of every package, then an independent blind
audit of the images): Athena 1588 -> 1607 right on 1,618 darts; the
triangulated tip height moved from a median +2.8mm above the board to on
it (-0.08mm, Athena's own board plane is -0.75mm) on the darts it
changes; blind before/after image pairs favoured the continued tip about
10:1. Sweeps of every constant here were flat (every variant 8-20 darts
above the old engine), and the gain held in each half of the corpus.

**The one place it is held back** (`engine._continue_candidate()`): when
exactly one camera sees the dart inside the board, a continuation may not
move that camera's read from off the board onto it -- 130708-045-OUT's
lone camera walked from outside onto a 4, the one call the continuation
broke. Broader limits were tried and judged blind in the images: a 60px
cap, no continuation for a lone camera off the board, none when no camera
sees the dart inside the board. Each blocked more right continuations than
wrong ones, and none changed a call for the better.
"""
from __future__ import annotations

import math

import numpy as np

# Cut for the continuation's own mask on the blurred diff. Deliberately
# lower than Athena's detection threshold (`crossing_detection.
# DIFF_THRESHOLD`, 31): the missing part of the dart is exactly the part
# under 31. Swept 15/20/25/31 -- 15 overshoots onto the board (+22 -6),
# 25 and 31 recover less; 20 kept.
CONTINUATION_THRESHOLD = 20.0

# Longest walk from the detected tip, px. A 60px cap was tried
# (2026-09-28) against the audit's worry that the longest walks overshoot:
# it changed 163 darts and no call, and judged blind in the images the
# 90px walk was the right one 17 times to 1 -- a dart can really be
# broken up over that length.
MAX_CONTINUATION_PX = 90

# Longest empty stretch of corridor the chain may cross, px.
MAX_GAP_PX = 45

# A covered stretch across a gap must be at least this long, px, and at
# least this share of the gap it jumps.
MIN_STRETCH_PX = 4
GAP_STRETCH_FRACTION = 0.25

# Half-width of the corridor across the axis, px.
CORRIDOR_HALF_PX = 3

# Where between the noise floor and the level just inside the end the
# tip settles.
HALF_MAX_FRACTION = 0.5

# Smaller moves are blur halo, not missed shaft.
MIN_MOVE_PX = 3.0

# Sampling reach around the tip -- the window below must contain every
# pixel the walk and the half-max step look at.
_NOISE_OFFSETS_PX = (-12, -8, 8, 12)
_WINDOW_MARGIN_PX = 6


def _window(bg_bgr: np.ndarray, frame_bgr: np.ndarray, tip, u, v):
    """The blurred diff on a small window around the walk -- identical,
    pixel for pixel, to the same crop of Athena's full-frame blurred diff
    (a per-pixel grey conversion and |diff|, then a 5x5 Gaussian whose
    border effects stay inside `_WINDOW_MARGIN_PX`; at a frame edge the
    window's edge IS the frame's, so the reflected border is the same).
    A few thousand pixels instead of the whole frame."""
    import cv2

    from opendarts.engines.athena.crossing_detection import BLUR_KSIZE
    from opendarts.imageops import blurred_abs_diff

    reach = MAX_CONTINUATION_PX + 6
    side = max(abs(o) for o in _NOISE_OFFSETS_PX)
    corners = np.array([tip + s * u + o * v for s in (0, reach) for o in (-side, side)])
    h, w = frame_bgr.shape[:2]
    x0 = max(int(math.floor(corners[:, 0].min())) - _WINDOW_MARGIN_PX, 0)
    y0 = max(int(math.floor(corners[:, 1].min())) - _WINDOW_MARGIN_PX, 0)
    x1 = min(int(math.ceil(corners[:, 0].max())) + _WINDOW_MARGIN_PX + 1, w)
    y1 = min(int(math.ceil(corners[:, 1].max())) + _WINDOW_MARGIN_PX + 1, h)
    if x1 <= x0 or y1 <= y0:
        return None, (0, 0)
    g0 = cv2.cvtColor(np.ascontiguousarray(bg_bgr[y0:y1, x0:x1]), cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(np.ascontiguousarray(frame_bgr[y0:y1, x0:x1]), cv2.COLOR_BGR2GRAY)
    return blurred_abs_diff(g0, g1, BLUR_KSIZE), (x0, y0)


def continue_tip(
    bg_bgr: np.ndarray,
    frame_bgr: np.ndarray,
    tip_px: tuple[float, float],
    axis_unit: tuple[float, float],
) -> tuple[tuple[float, float], float]:
    """`(new_tip_px, move_px)` for a tip whose `axis_unit` points from the
    tip toward the flight (`crossing_detection`'s convention); `move_px`
    is 0.0 and the tip unchanged when there is nothing to continue onto."""
    import cv2

    u = -np.asarray(axis_unit, dtype=np.float64)
    u /= np.linalg.norm(u) + 1e-12
    v = np.array([-u[1], u[0]])
    t = np.asarray(tip_px, dtype=np.float64)
    h, w = frame_bgr.shape[:2]
    diff, (x0, y0) = _window(bg_bgr, frame_bgr, t, u, v)
    if diff is None:
        return (float(t[0]), float(t[1])), 0.0

    def sample(p) -> float | None:
        if not (0 <= p[0] < w - 1 and 0 <= p[1] < h - 1):
            return None
        return float(cv2.getRectSubPix(diff, (1, 1), (float(p[0] - x0), float(p[1] - y0)))[0, 0])

    # Which steps along the shaft have any covered pixel across the
    # corridor -- every step at once, nearest pixel of each sample point.
    steps = np.arange(1, MAX_CONTINUATION_PX + 1)
    across = np.arange(-CORRIDOR_HALF_PX, CORRIDOR_HALF_PX + 1)
    points = t + steps[:, None, None] * u + across[None, :, None] * v
    xi, yi = np.rint(points[..., 0]).astype(int), np.rint(points[..., 1]).astype(int)
    in_frame = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    covered = np.zeros(xi.shape, dtype=bool)
    covered[in_frame] = diff[yi[in_frame] - y0, xi[in_frame] - x0] > CONTINUATION_THRESHOLD
    hits = covered.any(axis=1)

    best, run, start = 0, 0, 0
    for s, hit in zip(steps.tolist(), hits.tolist()):
        if hit:
            if run == 0:
                start = s
            run += 1
            gap = start - best - 1
            if gap <= 0 or run >= max(MIN_STRETCH_PX, GAP_STRETCH_FRACTION * gap):
                best = s
        else:
            run = 0
            if s - best > MAX_GAP_PX:
                break

    if best > 0:
        prof = {}
        for s in range(max(1, best - 12), best + 1):
            vals = [q for q in (sample(t + s * u + o * v) for o in (-1, 0, 1)) if q is not None]
            if vals:
                prof[s] = max(vals)
        noise = [
            q for s in range(max(1, best - 12), best + 5, 2) for o in _NOISE_OFFSETS_PX
            for q in (sample(t + s * u + o * v),) if q is not None
        ]
        inside = [prof[k] for k in range(best - 6, best - 1) if k in prof]
        if noise and inside:
            floor, level = float(np.median(noise)), float(np.median(inside))
            if level > floor:
                half = floor + HALF_MAX_FRACTION * (level - floor)
                while best > 1 and best in prof and prof[best] < half:
                    best -= 1

    if best < MIN_MOVE_PX:
        return (float(t[0]), float(t[1])), 0.0
    new = t + best * u
    return (float(new[0]), float(new[1])), float(best)
