"""Bull-anchored, projectively phase-locked double-outer landmark finder.

This is opendarts's third-generation landmark detector, and the
root-cause fix for a bias every prior calibration task on this project
kept running into from a different direction:

  * `landmark_detection.py` fits an ellipse to the double ring's colour
    mask -- good to a measured median ~3px, but with a per-landmark
    SYSTEMATIC (not random) component that N-frame averaging structurally
    cannot remove.
  * `sector_correspondence.py` then picks the 4 landmark pixels by
    looking up a FIXED per-camera reference angle on that ellipse. That
    is wrong in principle: the board's 20 sector wires are evenly spaced
    at 18 degrees ON THE BOARD, but a perspective projection does NOT
    preserve equal angular spacing, so a fixed-angle lookup is
    structurally mis-registered by an amount that grows with the
    camera's obliquity.
  * `active_landmarks.py` blended that model point with real per-image
    boundary evidence and recovered a good chunk of the loss (measured
    88.8% BOTH-match vs 81.7%), but it still starts from the same
    fixed-reference-angle prior and inherits the same ellipse.

This module drops the fixed-angle scheme entirely and solves for the
board's real rotational phase in the image, per image, using the actual
projective geometry.

Provenance and scope
--------------------
The staged shape of this pipeline (rough ellipse -> bull -> bull-anchored
ellipse re-seat -> projective phase lock against a dense angular
edge-energy profile -> orientation lock -> bounded refine -> reject
gates) follows a board-finding design owns and independently
developed. What is NOT carried over, deliberately:

  * **Every fixed-for-one-rig pixel constant.** The reference hardcodes
    the expected ellipse axes (341 x 669 px) and the bull-to-ellipse
    offset (55 px) for one specific camera-to-board distance and camera
    elevation. Nothing like that appears here: the ellipse prior comes
    from the seed ellipse detected in THAT image, the bull is found by a
    board-relative colour/shape signature, and every guardrail below is
    expressed as a fraction of the detected geometry. A rig with
    different camera distances or angles changes those numbers and this
    module does not care.
  * **Number-ring template matching against captured PNGs of one
    board.** A template cropped from one physical board's "20" cannot
    match a different board. See `lock_orientation()` for what is used
    instead, and for an honest statement of the one piece that is still
    not fully rig-independent.
  * **The approximate Moebius/disk map.** The reference's
    `_mobius_c_to_0` is not exactly circle-preserving (expanding
    `M^T diag(1,1,-1) M` for that matrix leaves a residual
    `|c|^2 - (c.p)^2` term). The exact object is derived from scratch
    below -- see `disk_boost()`.

The core idea, derived independently
------------------------------------
Let `E` be the ellipse the board's double-outer circle projects to, and
let `bull` be the pixel the board CENTRE projects to. Both are directly
measurable. Let `A` be the affine map taking the unit circle onto `E`.
Then `A^-1 . H` maps the unit circle onto itself, i.e. it is a projective
automorphism of the unit disk. Those form O(2,1). Because `A^-1 . H`
sends the board origin to `u := A^-1(bull)`, the hyperbolic part is
pinned by `u`, leaving only a rotation:

        H = A . B(-u) . R(phi)

where `B(u)` is the 2+1D Lorentz boost taking `u` to the origin (see
`disk_boost()`), and `R(phi)` is an ordinary rotation. So **the entire
board-to-image homography is determined, up to ONE scalar `phi`, by the
ellipse and the bull.** `phi` is then found by matching the 20 predicted
wire directions against a real angular edge-energy profile built from
the image -- a genuine search over perspective-correct wire angles, not
an assumption that they are 18 degrees apart in the image.

Validated against three real camera homographies before any of this was
built: the family reproduces a real
camera's own 20 wire pixels to a maximum error of 0.016-0.073 px.

Why rays from the bull are the right sampling geometry: a homography
maps straight lines to straight lines, and all 20 sector wires are
straight lines through the board centre, so their images are straight
lines through the bull's image, exactly. An angular profile taken about
the bull therefore has 20 genuinely sharp peaks with no radial smearing
-- which is not true of a profile taken about the ellipse's centre, and
is the reason `landmark_detection.py`'s centroid-anchored boundary trace
carries a systematic radial bias (measured: its ellipse sits +2.9 to
+10.4 px OUTSIDE the true ring).

This module is deliberately additive. It does not modify, and is not
imported by, `sector_correspondence.py`, `active_landmarks.py`, or any
engine.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from opendarts.calibration.landmark_detection import (
    Ellipse,
    _color_mask,
    _outer_ring_component_mask,
    _robust_fit_ellipse,
    detect_double_ring_quad,
)
from opendarts.geometry.board import (
    DOUBLE_INNER_RADIUS_MM,
    DOUBLE_OUTER_RADIUS_MM,
    SECTOR_ANGLE_DEG,
    SECTOR_NUMBERS_CLOCKWISE,
    polar_to_xy_mm,
)

N_SECTORS = len(SECTOR_NUMBERS_CLOCKWISE) # 20
FIRST_WIRE_ANGLE_DEG = SECTOR_ANGLE_DEG / 2.0 # 9.0 -- the 20/1 wire

# The 4-point cardinal-quad landmark indices, as ring indices. Ring index k
# is the wire at board angle 9 + 18k; quad index i is at board angle 9 + 90i (see
# sector_correspondence.ad_quad_board_angle_deg), so i -> k = 5i.
AD_QUAD_RING_INDICES = (0, 5, 10, 15)

# --- Board-standard facts (regulation geometry, not rig measurements) ---
#
# On a regulation dartboard the double beds strictly alternate red/green
# and sector 20's double is RED. Verified directly on this project's own
# archived frames rather than assumed (25
# folders x 3 cameras x 20 beds): every even index of
# SECTOR_NUMBERS_CLOCKWISE voted red 74-75 times out of 75 and every odd
# index voted green 74-75 out of 75 -- a clean, unambiguous alternation
# with no mixed sector. NOTE this disagrees with the reference finder's
# own comment ("D20 reads green"); that comment describes a sampling
# offset half a sector clockwise, not a different physical board.
RED_DOUBLE_SECTOR_INDICES = frozenset(range(0, N_SECTORS, 2))

# --- Illuminant normalisation -------------------------------------------
#
# Every colour stage below (the seed ellipse's mask, `detect_bull`, the
# bull-anchored `reseat_ellipse` trace, `double_colour_score`) ultimately
# thresholds HSV against ABSOLUTE cut points -- `landmark_detection`'s
# `GREEN_HSV_LOW = (35, 60, 40)` and friends. An absolute saturation cut
# is only meaningful relative to the image's own grey point, and this
# rig's cameras do not hold their grey point fixed.
#
# REAL FAILURE THIS FIXES (2026-08-13 session, cam1, throws 0-59 of 180).
# Camera 1's white balance sat green-shifted for the first ~60 throws and
# then returned to normal; nothing physical moved. Measured whole-image channel means:
#
# frame R/G B/G mean BGR of dark (V<90) board px
# today i=30 0.914 0.933 (48.6, 52.3, 46.9) <- G highest
# today i=57 0.904 0.920 (47.6, 52.7, 46.2)
# today i=60 0.942 0.940 (49.5, 51.0, 48.4)
# today i=120 0.948 0.973 (51.5, 50.1, 48.3)
# hist 162346 i=0 0.963 0.997 (50.5, 47.2, 47.7)
#
# Under that cast, dark sisal bristle crosses `S >= 60` with a green hue
# and gets masked as ring paint: the fraction of dark board pixels with
# S >= 60 rose from ~22-27% (healthy/historical) to 37-44%. The mask then
# speckles across the black beds, `_outer_ring_component_mask`'s
# largest-component pick swallows them, and the seed ellipse -- and the
# re-seat built on it -- come out ~20% too big (minor axis 806-909 px vs
# a true ~671). An ellipse that size pushes `angular_edge_profile`'s
# 0.30-0.88 radial band clean off the sector wires, so the spoke peaks
# smear and everything downstream collapses together: spoke_score
# 603-1459 vs ~3100, phase_confidence 1.37-2.02 vs 2.43, colour_margin
# 0.099-0.149 vs 0.83, and 15-50% of frames rejected outright for an
# ambiguous orientation lock.
#
# That is why the phase-confidence gate was firing: it was RIGHT. The
# frames really were bad. The bug is upstream, in a colour threshold that
# is not illuminant-invariant -- so the fix belongs there, not in
# `MIN_PHASE_CONFIDENCE`.
#
# The correction is a plain grey-world estimate: scale each channel so
# the image's channel means agree. That is the exact inverse of the thing
# measured above (a per-channel gain applied by the camera), it needs no
# threshold of its own, and it is scene-independent in the sense that
# matters here -- a fixed camera looking at a fixed scene has a stable
# channel-mean ratio, so on a HEALTHY frame the gains land within a
# percent or two of unity and change nothing.
#
# Measured effect on the ring component's pixel count (the quantity that
# actually breaks), before -> after:
# today cam1 i=30 (broken) 76303 -> 33523
# today cam1 i=57 (broken) 128085 -> 36346
# today cam1 i=60 (healthy) 35030 -> 32888
# hist 162346 i=0 (healthy) 33008 -> 33673
# i.e. it pulls the broken frames back onto the healthy value (~33k) and
# leaves healthy frames essentially where they were.
#
# `MAX_ILLUMINANT_GAIN` is a guard, not a tuning knob: it stops a frame
# whose scene composition has genuinely changed (something large and
# strongly coloured entering view) from being violently re-tinted on the
# strength of a grey-world assumption that no longer holds. The real
# gains this corpus produces are tiny -- the worst measured frame above
# needs (1.024, 0.941, 1.041) -- so the clamp never binds on real data
# and exists purely to bound the damage in a case none of the 1047
# archived frames exhibit.
MAX_ILLUMINANT_GAIN = 1.6


def normalise_illuminant(image_bgr: np.ndarray) -> np.ndarray:
    """Grey-world white balance: per-channel gain making the channel
    means equal, so an absolute HSV threshold downstream means the same
    thing under a camera whose white balance has drifted.

    Returns a new uint8 BGR image; the input is never modified. A
    degenerate image (an all-black channel, or a channel mean of zero)
    is returned unchanged rather than divided by ~zero.
    """
    f = np.asarray(image_bgr, dtype=np.float32)
    if f.ndim != 3 or f.shape[2] != 3:
        return np.asarray(image_bgr)
    means = f.reshape(-1, 3).mean(axis=0)
    if not np.isfinite(means).all() or float(means.min()) <= 1e-6:
        return np.asarray(image_bgr)
    gains = float(means.mean()) / means
    gains = np.clip(gains, 1.0 / MAX_ILLUMINANT_GAIN, MAX_ILLUMINANT_GAIN)
    return np.clip(f * gains[None, None, :], 0.0, 255.0).astype(np.uint8)


# --- Tunable stage parameters (all relative, no absolute pixel sizes) ---

# Bull search: a red blob's area as a fraction of the fitted ellipse's
# area. The inner bull is (6.35/170)^2 = 0.14% of the board face; the
# window is wide enough to survive perspective foreshortening and
# threshold slop without admitting a whole double/treble bed (0.46% /
# 0.29% respectively -- which is why area alone cannot decide, and the
# green-halo term below does the real work).
BULL_AREA_FRACTION_MIN = 0.0002
BULL_AREA_FRACTION_MAX = 0.0150
BULL_MAX_NORMALISED_RADIUS = 0.80

# Re-seat guardrails, all relative to the seed ellipse.
RESEAT_MAX_CENTRE_SHIFT_FRACTION = 0.15
RESEAT_MAX_AXIS_CHANGE_FRACTION = 0.25
RESEAT_MAX_ANGLE_CHANGE_DEG = 12.0

# Angular edge-energy profile.
N_ANGLE_BINS = 1440 # 0.25 deg per bin
PROFILE_RADIAL_LO = 0.30 # fraction of the bull->ring distance
PROFILE_RADIAL_HI = 0.88
PROFILE_RADIAL_SAMPLES = 24
PROFILE_SMOOTH_BINS = 7

# Phase search.
PHASE_STEP_DEG = 0.05
PHASE_POLISH_SPAN_DEG = 0.30
PHASE_POLISH_STEP_DEG = 0.005
SPOKE_NEIGHBOUR_LO_DEG = 2.0 # background band around each spoke
SPOKE_NEIGHBOUR_HI_DEG = 6.0

# Per-wire angular refine.
WIRE_REFINE_WINDOW_DEG = 1.0
WIRE_REFINE_BLEND = 1.0 # 1.0 = take the refined peak outright

# --- Quality gates, set from measured distributions (114 real frames
# from two sessions, plus deliberately degraded inputs) ---
#
# COLOUR MARGIN is the gate that actually works. Measured on real frames:
# min 0.110, p05 0.698, median 0.988. On a heavily blurred board (51px
# Gaussian, orientation genuinely destroyed): 0.009. 0.05 sits 2.2x below
# the worst real frame and 5.5x above the blurred one -- a real
# separation, not a round number.
MIN_COLOUR_MARGIN = 0.05
#
# PHASE CONFIDENCE is a QUALITY gate, not a validity gate, and the
# distinction was measured rather than assumed. It does NOT separate a
# real board from a destroyed one -- a heavily blurred board still scored
# 2.02 sigma, inside the 1.46-2.78 range real frames occupy -- so it is
# useless for "is this even a dartboard" (the colour margin above does
# that job). But it is a strong predictor of how good a frame's landmarks
# actually are, which matters because frames are averaged per session:
# admitting the low-confidence tail measurably drags the average down.
#
# Real end-to-end sweep (169 throws, the metric that actually matters
# rather than a proxy):
# thresh 0.00 -> homography 85.2% PnP 92.3% (505/505 frames kept)
# thresh 1.80 -> homography 88.8% PnP 95.9%
# thresh 2.30 -> homography 89.3% PnP 96.4% (475 frames kept)
# thresh 2.50 -> homography 88.8% PnP 96.4% (342 frames kept)
# A broad plateau from ~1.8 to ~2.45, not a knife edge.
#
# The value below is the one LEAVE-ONE-SESSION-OUT cross-validation picks
# 6 of the 7 folds chose 2.10 and the
# seventh chose 1.80, and the pooled HELD-OUT result was 95.9% PnP /
# 88.8% homography versus 96.4% / 89.3% for the in-sample optimum -- a
# 0.5pp gap, i.e. essentially no overfitting. Deliberately NOT set to the
# in-sample best (2.30); this closes the "threshold not cross-validated
# leave-one-session-out" gap that was flagged and left open.
#
# **This constant is the DEFAULT, never a hard-wired one.**
# `find_oriented_landmarks(..., min_phase_confidence=X)` overrides it, and
# `0.0` disables the gate entirely so a caller sees the raw
# `phase_confidence` on EVERY frame including ones the shipped gate would
# reject. That override is not a convenience -- it is what makes the
# tuning above reproducible at all. When the gate was baked in with no
# way past it, the very sweep/LOSO scripts that produced these numbers
# became circular: they only looked at frames where `result.ok` was True,
# i.e. frames that had already passed 2.1, so sweeping candidate
# thresholds against that pre-filtered pool could no longer see what a
# looser gate would have admitted (every value <= 2.1 measured
# identically). Regression-tested by
# tests/test_oriented_landmarks_gate_override.py -- do not reintroduce an
# un-overridable gate here or anywhere downstream of it.
MIN_PHASE_CONFIDENCE = 2.1
MAX_NORMALISED_BULL_RADIUS = 0.95 # |u| -- beyond this the family degenerates


# ---------------------------------------------------------------------
# Projective disk geometry
# ---------------------------------------------------------------------


def affine_unit_circle_to_ellipse(ellipse: Ellipse) -> np.ndarray:
    """The affine map (3x3 homogeneous) taking the unit circle onto
    `ellipse`, in exactly the parameterisation
    `Ellipse.boundary_samples()` already uses: semi-axis
    `major_axis_px/2` along the direction `angle_deg` from +x, and
    `minor_axis_px/2` perpendicular to it.

    (`Ellipse.major_axis_px` holds cv2.fitEllipse's FIRST size element,
    which for this rig's frames is actually the shorter one -- the name
    is historical. It does not matter here: this function only has to
    agree with `boundary_samples()`, and it does.)
    """
    a = ellipse.major_axis_px / 2.0
    b = ellipse.minor_axis_px / 2.0
    th = math.radians(ellipse.angle_deg)
    ct, st = math.cos(th), math.sin(th)
    return np.array(
        [[a * ct, -b * st, ellipse.cx],
         [a * st, b * ct, ellipse.cy],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def disk_boost(u) -> np.ndarray:
    """The projective automorphism of the unit disk that sends `u` to the
    origin -- the 2+1 dimensional Lorentz boost of rapidity
    `artanh(|u|)` along `u`.

    Derived here rather than adapted: the projective maps preserving the
    conic `x^2 + y^2 - w^2 = 0` are exactly those `M` with
    `M^T J M ~ J` for `J = diag(1, 1, -1)`, i.e. O(2,1). The boost below
    satisfies that identity exactly (it is a Lorentz transform by
    construction), and `B(u) . (ux, uy, 1) = (0, 0, 1/gamma)`, so it maps
    the unit circle exactly onto itself and `u` exactly onto the origin.

    This matters: the natural "obvious" matrix
    `[[1,0,-cx],[0,1,-cy],[-cx,-cy,1]]` also sends `u` to the origin but
    is NOT circle-preserving -- expanding its quadratic form leaves a
    residual `|u|^2 - (u.p)^2`, which is zero only for `p` parallel to
    `u`. Using it would put a small, systematically angle-dependent error
    into every predicted wire direction, which is precisely the class of
    bug this module exists to remove.
    """
    ux, uy = float(u[0]), float(u[1])
    n2 = ux * ux + uy * uy
    if n2 < 1e-18:
        return np.eye(3, dtype=np.float64)
    if n2 >= 1.0:
        raise ValueError(f"disk_boost needs |u| < 1, got |u|={math.sqrt(n2):.4f}")
    g = 1.0 / math.sqrt(1.0 - n2)
    k = (g - 1.0) / n2
    return np.array(
        [[1.0 + k * ux * ux, k * ux * uy, -g * ux],
         [k * ux * uy, 1.0 + k * uy * uy, -g * uy],
         [-g * ux, -g * uy, g]],
        dtype=np.float64,
    )


def _rotation(phi_deg: float) -> np.ndarray:
    p = math.radians(phi_deg)
    c, s = math.cos(p), math.sin(p)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def apply_homography(H: np.ndarray, points) -> np.ndarray:
    """Apply a 3x3 homogeneous map to (N, 2) points, returning (N, 2)."""
    P = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    with np.errstate(all="ignore"):
        q = (H @ np.hstack([P, np.ones((len(P), 1))]).T).T
        w = q[:, 2:3]
        w = np.where(np.abs(w) < 1e-12, np.nan, w)
        return q[:, :2] / w


def normalised_board_point(board_angle_deg: float, radius_fraction: float = 1.0):
    """A board point in the unit-disk frame this module's `H` consumes.

    The board frame is this repo's own (`polar_to_xy_mm`: x = R sin a,
    y = R cos a, angle clockwise from 12 o'clock). The y flip folded in
    here is the fixed handedness of any camera looking at the board face
    with image y pointing down -- confirmed on all three real cameras
    (found the same sign for every one) and
    physically forced: the opposite sign would require viewing the board
    from behind.
    """
    a = math.radians(board_angle_deg)
    return (radius_fraction * math.sin(a), -radius_fraction * math.cos(a))


def board_to_image_homography(ellipse: Ellipse, bull_px, phase_deg: float) -> np.ndarray:
    """`H = A . B(-u) . R(phi)` -- the unit-disk board frame to pixels.

    Every `H` this returns maps the unit circle exactly onto `ellipse`
    and the board origin exactly onto `bull_px`, for any `phase_deg`.
    """
    A = affine_unit_circle_to_ellipse(ellipse)
    u = apply_homography(np.linalg.inv(A), [bull_px])[0]
    return A @ disk_boost(-u) @ _rotation(phase_deg)


def normalised_bull(ellipse: Ellipse, bull_px) -> np.ndarray:
    A = affine_unit_circle_to_ellipse(ellipse)
    return apply_homography(np.linalg.inv(A), [bull_px])[0]


def ellipse_ray_intersection(ellipse: Ellipse, origin, angle_deg: float):
    """Where the ray from `origin` at image angle `angle_deg` (atan2
    degrees, y down) leaves `ellipse`. None if it never does."""
    a = ellipse.major_axis_px / 2.0
    b = ellipse.minor_axis_px / 2.0
    th = math.radians(ellipse.angle_deg)
    ct, st = math.cos(th), math.sin(th)
    ox, oy = float(origin[0]), float(origin[1])
    ux = math.cos(math.radians(angle_deg))
    uy = math.sin(math.radians(angle_deg))
    dx, dy = ox - ellipse.cx, oy - ellipse.cy
    q0x, q0y = ct * dx + st * dy, -st * dx + ct * dy
    q1x, q1y = ct * ux + st * uy, -st * ux + ct * uy
    A = (q1x / a) ** 2 + (q1y / b) ** 2
    B = 2.0 * (q0x * q1x / a**2 + q0y * q1y / b**2)
    C = (q0x / a) ** 2 + (q0y / b) ** 2 - 1.0
    disc = B * B - 4.0 * A * C
    if disc < 0.0 or abs(A) < 1e-12:
        return None
    root = math.sqrt(disc)
    ts = [t for t in ((-B + root) / (2 * A), (-B - root) / (2 * A)) if t > 0.0]
    if not ts:
        return None
    t = max(ts)
    return (ox + t * ux, oy + t * uy)


def ellipse_ray_intersections(ellipse: Ellipse, origin, angles_deg) -> np.ndarray:
    """Vectorised `ellipse_ray_intersection` over an array of angles.

    Returns (N, 2) with NaN rows where the ray never leaves the ellipse.
    Same maths as the scalar version -- kept as a separate function
    rather than replacing it because the scalar one is what the
    readable/tested path uses, and this one exists purely so the 1440-bin
    profile and the phase sweep are not Python loops.
    """
    a = ellipse.major_axis_px / 2.0
    b = ellipse.minor_axis_px / 2.0
    th = math.radians(ellipse.angle_deg)
    ct, st = math.cos(th), math.sin(th)
    ox, oy = float(origin[0]), float(origin[1])
    ang = np.radians(np.asarray(angles_deg, dtype=np.float64))
    ux, uy = np.cos(ang), np.sin(ang)
    dx, dy = ox - ellipse.cx, oy - ellipse.cy
    q0x, q0y = ct * dx + st * dy, -st * dx + ct * dy
    q1x, q1y = ct * ux + st * uy, -st * ux + ct * uy
    A = (q1x / a) ** 2 + (q1y / b) ** 2
    B = 2.0 * (q0x * q1x / a**2 + q0y * q1y / b**2)
    C = (q0x / a) ** 2 + (q0y / b) ** 2 - 1.0
    disc = B * B - 4.0 * A * C
    with np.errstate(all="ignore"):
        root = np.sqrt(np.where(disc < 0.0, np.nan, disc))
        t = np.maximum((-B + root) / (2.0 * A), (-B - root) / (2.0 * A))
        t = np.where((t > 0.0) & np.isfinite(t), t, np.nan)
        return np.stack([ox + t * ux, oy + t * uy], axis=1)


def _image_angle(point, origin) -> float:
    return math.degrees(math.atan2(point[1] - origin[1], point[0] - origin[0])) % 360.0


def _wire_unit_points() -> np.ndarray:
    """The 20 sector wires as unit-disk board points, ring index order.

    Cached at module level (2026-09-11 calibration-speed pass): the
    values are pure board-geometry constants, recomputed identically on
    every call before -- `lock_phase()` alone called this twice per
    frame. Callers only ever read from the returned array (verified:
    `_wire_angles_for_phases()` broadcasts it, never writes), so
    returning the shared cached array is behaviour-identical.
    """
    global _WIRE_UNIT_POINTS_CACHE
    if _WIRE_UNIT_POINTS_CACHE is None:
        _WIRE_UNIT_POINTS_CACHE = np.asarray(
            [normalised_board_point(FIRST_WIRE_ANGLE_DEG + SECTOR_ANGLE_DEG * k)
             for k in range(N_SECTORS)],
            dtype=np.float64,
        )
        # Read-only, so the "callers only ever read" claim above is
        # ENFORCED rather than merely true today. A shared cached array a
        # caller can write to is a silent corruption waiting for the
        # second call; this turns that into an immediate error at the
        # write instead of wrong numbers much later.
        _WIRE_UNIT_POINTS_CACHE.flags.writeable = False
    return _WIRE_UNIT_POINTS_CACHE


_WIRE_UNIT_POINTS_CACHE: np.ndarray | None = None


def _wire_angles_for_phases(M: np.ndarray, bull_px, phases_deg: np.ndarray) -> np.ndarray:
    """Image angles of all 20 wires for every phase at once.

    `M = A . B(-u)` is the phase-independent part of the homography, so a
    phase sweep is just a 2-D rotation of the 20 board points before
    applying `M`. Returns (n_phases, 20) degrees.
    """
    m = _wire_unit_points() # (20, 2)
    p = np.radians(np.asarray(phases_deg, dtype=np.float64)) # (P,)
    c, s = np.cos(p)[:, None], np.sin(p)[:, None] # (P, 1)
    rx = c * m[None, :, 0] - s * m[None, :, 1] # (P, 20)
    ry = s * m[None, :, 0] + c * m[None, :, 1]
    with np.errstate(all="ignore"):
        w = M[2, 0] * rx + M[2, 1] * ry + M[2, 2]
        x = (M[0, 0] * rx + M[0, 1] * ry + M[0, 2]) / w
        y = (M[1, 0] * rx + M[1, 1] * ry + M[1, 2]) / w
    return np.degrees(np.arctan2(y - bull_px[1], x - bull_px[0])) % 360.0


# ---------------------------------------------------------------------
# Stage 1 -- bull detection
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class BullDetection:
    ok: bool
    xy: tuple[float, float] | None
    green_halo: float = 0.0
    compactness: float = 0.0
    score: float = 0.0
    reason: str = ""


@dataclass(frozen=True, eq=False)
class _BullFrameEvidence:
    """Everything `detect_bull()` computes that depends ONLY on the
    image, not on the candidate ellipse -- HSV conversion, the red/green
    threshold masks, the opened red mask, and its connected-component
    labelling. Factored out (2026-09-11 calibration-speed pass) because
    `locate_pre_orientation_landmarks()` calls `detect_bull()` TWICE per
    frame (once on the seed ellipse, once after the re-seat) on the SAME
    image -- before this, every one of these full-frame operations ran
    twice per frame for identical results. The ellipse only enters
    `detect_bull()` through the area window (`lo`/`hi`) and the
    centrality term, both applied AFTER these image-level products, so
    sharing them across the two calls is exactly equivalent.

    `eq=False` for the same reason `PreOrientationLandmarks` uses it --
    frozen dataclasses holding ndarrays must not auto-generate
    `__eq__`/`__hash__` (see that class's own docstring).
    """

    green: np.ndarray  # bool, full frame
    labels: np.ndarray  # int32 component labels of the opened red mask
    stats: np.ndarray
    centroids: np.ndarray
    n: int
    shape: tuple[int, int]  # (h, w)


def _bull_frame_evidence(image_bgr: np.ndarray) -> _BullFrameEvidence:
    """Compute `_BullFrameEvidence` for one frame -- the exact same
    operations (in the exact same order, producing bit-identical masks)
    the un-factored `detect_bull()` performed inline. The only textual
    change is comparing the uint8 HSV channels directly instead of
    first casting each full channel to int16 -- pure comparisons against
    small positive constants, no arithmetic, so the resulting booleans
    are identical while three full-frame int16 allocations per call
    disappear."""
    import cv2

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    hh = hsv[:, :, 0]
    ss = hsv[:, :, 1]
    vv = hsv[:, :, 2]
    red = (((hh < 12) | (hh > 168)) & (ss > 70) & (vv > 40)).astype(np.uint8)
    green = ((hh > 35) & (hh < 95) & (ss > 40) & (vv > 40))
    red_clean = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(red_clean, connectivity=8)
    return _BullFrameEvidence(
        green=green, labels=labels, stats=stats, centroids=centroids, n=int(n), shape=(h, w),
    )


def detect_bull(
    image_bgr: np.ndarray,
    ellipse: Ellipse,
    *,
    evidence: _BullFrameEvidence | None = None,
) -> BullDetection:
    """Find the red inner bull, using only board-relative evidence.

    The discriminating signal is the bull's GREEN HALO: the inner bull
    (red, 6.35mm) is completely ringed by the outer bull (green, to
    15.9mm), and nothing else on a dartboard is a small red region
    surrounded by green. Area alone cannot do this job -- a red treble
    bed is 0.29% of the board face and a red double bed 0.46%, both
    LARGER than the inner bull's 0.14%, so a size window admits them and
    the halo term is what actually rejects them.

    Everything here is expressed relative to `ellipse`, so no
    rig-specific pixel offset or size is involved (the reference
    implementation's equivalent step leans on a fixed "bull sits ~55px
    above the ellipse centre" prior, which encodes one particular camera
    elevation).

    `evidence`, if given, must be `_bull_frame_evidence(image_bgr)` for
    the SAME image -- lets a caller that runs this twice per frame
    (`locate_pre_orientation_landmarks()`'s seed + post-reseat calls)
    pay the image-level cost once. `None` (the default, and every
    external caller) computes it here, exactly as before.
    """
    import cv2

    if evidence is None:
        evidence = _bull_frame_evidence(image_bgr)
    h, w = evidence.shape
    green = evidence.green

    ellipse_area = math.pi * (ellipse.major_axis_px / 2.0) * (ellipse.minor_axis_px / 2.0)
    lo = BULL_AREA_FRACTION_MIN * ellipse_area
    hi = BULL_AREA_FRACTION_MAX * ellipse_area

    n, labels, stats, centroids = (
        evidence.n, evidence.labels, evidence.stats, evidence.centroids,
    )
    if n <= 1:
        return BullDetection(False, None, reason="no red components")

    A_inv = np.linalg.inv(affine_unit_circle_to_ellipse(ellipse))
    best = None
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if not (lo <= area <= hi):
            continue
        cx, cy = float(centroids[i][0]), float(centroids[i][1])
        nb = apply_homography(A_inv, [(cx, cy)])[0]
        r_norm = float(math.hypot(nb[0], nb[1]))
        if not np.isfinite(r_norm) or r_norm > BULL_MAX_NORMALISED_RADIUS:
            continue

        x0 = max(0, stats[i, cv2.CC_STAT_LEFT] - 12)
        y0 = max(0, stats[i, cv2.CC_STAT_TOP] - 12)
        x1 = min(w, x0 + stats[i, cv2.CC_STAT_WIDTH] + 24)
        y1 = min(h, y0 + stats[i, cv2.CC_STAT_HEIGHT] + 24)
        comp = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
        if comp.sum() == 0:
            continue
        halo = cv2.dilate(comp, np.ones((9, 9), np.uint8)) - comp
        halo_n = int(halo.sum())
        green_halo = float(green[y0:y1, x0:x1][halo > 0].mean()) if halo_n else 0.0

        bw = float(stats[i, cv2.CC_STAT_WIDTH])
        bh = float(stats[i, cv2.CC_STAT_HEIGHT])
        compactness = area / max(bw * bh, 1.0)
        centrality = math.exp(-0.5 * (r_norm / 0.55) ** 2)
        score = (0.05 + 0.95 * green_halo) * (0.3 + 0.7 * compactness) * (0.3 + 0.7 * centrality)
        if best is None or score > best[0]:
            best = (score, i, green_halo, compactness, (x0, y0, x1, y1))

    if best is None:
        return BullDetection(False, None, reason="no red component matched the bull signature")

    score, idx, green_halo, compactness, (x0, y0, x1, y1) = best
    comp = (labels[y0:y1, x0:x1] == idx).astype(np.uint8)
    mom = cv2.moments(comp)
    if mom["m00"] <= 1e-6:
        return BullDetection(False, None, reason="degenerate bull component")
    return BullDetection(
        ok=True,
        xy=(x0 + mom["m10"] / mom["m00"], y0 + mom["m01"] / mom["m00"]),
        green_halo=green_halo,
        compactness=compactness,
        score=score,
    )


# ---------------------------------------------------------------------
# Stage 2 -- bull-anchored ellipse re-seat
# ---------------------------------------------------------------------


def _debias_band_outer_radius(
    ellipse: Ellipse, bull_px, dirs: np.ndarray, r_inner: np.ndarray, r_outer: np.ndarray
) -> np.ndarray:
    """Where the double-OUTER wire really is, per ray, given both edges of
    the colour band the mask actually produced.

    THE BIAS THIS REMOVES, measured (2026-08-13). The colour mask does not
    stop at the paint: sensor and lens blur, plus the dark-red and
    dark-green transition pixels either side of a bed, spill it outward
    past the double-outer wire AND inward past the double-inner wire. So
    an ellipse fitted to the mask's OUTER edge -- which is what this
    module did, and what `landmark_detection` does -- sits systematically
    outside the true 170mm ring. Every landmark then gets labelled 170mm
    while living further out, PnP solves a board that is too big in
    pixels, and every real tip reads SHORT in board radius, proportionally.

    The bias is directly measurable without any external reference,
    because the double bed is bounded by TWO real wires (162.0mm and
    170.0mm). Trace both mask edges on the same bull ray, map the inner
    one through the very same homography the outer one defines, and it
    must come back at 162.0mm if the outer edge is really 170.0mm. Pooled
    over 8 real sessions x 3 cameras (12 frames each) it came back at
    **158.47mm**, i.e. 3.53mm short, with a per-camera spread of only
    -2.41 to -3.82mm -- a strikingly stable, systematic offset, not noise.
    Solving `(162 - d) * 170/(170 + d) = 158.47` gives a symmetric mask
    dilation of **d = 1.83mm** (3.60 px on a 335 px semi-axis), i.e. a
    **1.075% radial scale error**.

    That number was then checked against something the derivation never
    saw. The predicted signed radial residual of a scored tip is
    -1.075% of its radius; measured against the oracle's tip positions over
    the whole corpus, the real residual was -0.93mm mean / -0.99mm median
    (-0.81% to -1.17% of radius, growing with radius exactly as a scale
    error must). Applying the correction moved it to +0.18mm mean /
    -0.00mm median, and cut the median 2-D board error from 2.59mm to
    2.36mm. A correction derived purely from the board's own wire spacing
    landing an INDEPENDENT reference's residual on zero is the evidence
    that this is a real geometric bias and not a fitted fudge.

    The correction is computed per image, from that image's own two
    edges, so nothing about the mask's spill width is hardcoded -- a
    different camera, exposure, or board finish that spills more or less
    is measured, not assumed. What IS assumed, and is the honest
    limitation: the spill is symmetric, i.e. the mask grows by the same
    amount at both edges. That follows from it being a blur/threshold
    effect at two similar paint-to-dark transitions, and the check above
    is consistent with it, but it is not separately proven -- an
    asymmetric spill would leave a residual this cannot see.

    Given the measured band `[r_inner, r_outer]` in pixels along a ray,
    the unbiased estimate of the band's CENTRE is its midpoint, and the
    true band centre is board radius 166.0mm. So the true outer wire sits
    half a TRUE band-width outboard of that midpoint, where the true
    width is predicted for this ray from the current ellipse. Returns the
    corrected outer radius per ray.
    """
    A = affine_unit_circle_to_ellipse(ellipse)
    u = apply_homography(np.linalg.inv(A), [bull_px])[0]
    try:
        M = A @ disk_boost(-u)
    except ValueError:
        # disk_boost() is CORRECTLY refusing here -- see its own
        # docstring -- because THIS CANDIDATE `ellipse`'s own homography
        # genuinely maps `bull_px` outside the unit disk (|u| >= 1). That
        # is real, confirmed-live-on-a-sibling-rig behaviour
        # (`reseat_ellipse()`'s own 2-pass refit loop calls this function
        # again on a freshly REFITTED candidate every iteration, and that
        # intermediate candidate is not validated against
        # `MAX_NORMALISED_BULL_RADIUS` the way the FINAL ellipse
        # `reseat_ellipse()` returns is, in
        # `locate_pre_orientation_landmarks()` -- so a degenerate
        # second-pass refit can reach here even though the seed ellipse
        # and bull were both already sane). The bug this fix closes is
        # NOT `disk_boost()`'s check itself (do not widen or remove it --
        # the input genuinely is invalid here) -- it is that nothing
        # upstream used to catch the resulting `ValueError`, so it
        # propagated out of `reseat_ellipse()`, out of
        # `locate_pre_orientation_landmarks()`, out of the per-camera
        # detect loop's `ThreadPoolExecutor` worker, and aborted the
        # WHOLE calibration event (all cameras, discarding whatever
        # progress the others had made) over what should be a single bad
        # frame. This project's own established convention for exactly
        # this shape of problem -- one item in a burst of frames failing
        # geometry recovery is an ordinary per-item decline, never a
        # reason to fail the whole burst -- is documented on
        # `ring_correlation_orientation.solve_frame_orientation()`'s own
        # docstring; the same convention already used THROUGHOUT this
        # module (`BullDetection.ok`, `PreOrientationLandmarks.ok`,
        # `OrientedLandmarkResult.ok`, and this exact function's own
        # sibling early-return guards a few lines below --
        # `usable.sum() < 40`, `k` out of `[0.94, 1.02]` -- all decline
        # the correction and return `r_outer` UNCHANGED rather than
        # raising). Declining the correction here (instead of e.g.
        # aborting `reseat_ellipse()` outright) is deliberately the
        # LEAST destructive response available: `reseat_ellipse()`'s own
        # post-loop guardrails (centre shift / axis change / tilt change
        # vs the seed) and `locate_pre_orientation_landmarks()`'s own
        # post-reseat bull-radius re-check still run against whatever
        # ellipse this ultimately produces, so a genuinely bad candidate
        # is still caught -- this only stops ONE bad ray-batch's debias
        # correction from crashing detection outright.
        return r_outer
    Minv = np.linalg.inv(M)

    bx, by = float(bull_px[0]), float(bull_px[1])
    # Disk-frame preimage of each ray's measured OUTER point. Its
    # direction is the ray's board direction; H maps board radii through
    # the origin to straight lines through the bull, exactly, so the same
    # direction serves for every radius on this ray.
    p_out = np.stack([bx + dirs[:, 0] * r_outer, by + dirs[:, 1] * r_outer], axis=1)
    q = apply_homography(Minv, p_out)
    n = np.hypot(q[:, 0], q[:, 1])
    with np.errstate(all="ignore"):
        qhat = q / np.where(n[:, None] > 1e-12, n[:, None], np.nan)

    def pixel_radius_at(board_r_mm: float) -> np.ndarray:
        pts = qhat * (board_r_mm / DOUBLE_OUTER_RADIUS_MM)
        img = apply_homography(M, pts)
        return np.hypot(img[:, 0] - bx, img[:, 1] - by)

    w_true = pixel_radius_at(DOUBLE_OUTER_RADIUS_MM) - pixel_radius_at(DOUBLE_INNER_RADIUS_MM)
    mid = 0.5 * (r_inner + r_outer)
    per_ray = mid + 0.5 * w_true

    # A single ROBUST global correction, not a per-ray one. Two reasons,
    # both real. First, the inner-edge walk is the fragile half of the
    # measurement -- it can over-run through a wire crossing and land in
    # the treble ring, which on that ray produces an absurd band and an
    # absurd correction; a median is immune where a per-ray application
    # is not. Second, the quantity being corrected is a radial SCALE
    # error, which is one number for the whole ring by construction, so
    # estimating it 700 times independently only adds noise.
    #
    # The statistic is the RATIO, not the pixel offset: the bull-to-ring
    # distance varies strongly with angle under perspective (that is why
    # every other radial band in this module is expressed as a fraction),
    # so a median pixel offset would be dominated by whichever side of
    # the board happens to be nearer the camera.
    with np.errstate(all="ignore"):
        ratio = per_ray / r_outer
    w_meas = r_outer - r_inner
    usable = (
        np.isfinite(ratio) & np.isfinite(w_true) & np.isfinite(w_meas)
        & (r_outer > 1.0) & (w_true > 0.0)
        # The measured band must be recognisably the double bed: at least
        # as wide as the true bed and no more than three times it. A ray
        # whose walk fell short, or ran on into the next ring, is dropped.
        & (w_meas >= 0.5 * w_true) & (w_meas <= 3.0 * w_true)
    )
    if int(usable.sum()) < 40:
        return r_outer
    k = float(np.median(ratio[usable]))
    # A damage limiter, not a tuning knob, and deliberately NOT one-sided:
    # a mask that UNDER-covers the paint would need k > 1, and silently
    # refusing to correct that direction would be an assumption dressed
    # up as a guard. Measured over 144 real frames (8 sessions x 3
    # cameras) k sits in [0.98190, 0.99384] with median 0.98974 -- a 1.03%
    # median correction, matching the 1.075% the two-wire derivation
    # above predicts, and never within 5x of either bound. Anything that
    # does reach a bound means the band measurement has gone wrong, and
    # leaving the ellipse where the outer edge put it is the safe answer.
    if not math.isfinite(k) or not (0.94 <= k <= 1.02):
        return r_outer
    return r_outer * k


def reseat_ellipse(image_bgr: np.ndarray, seed: Ellipse, bull_px) -> tuple[Ellipse, str]:
    """Re-trace the double ring's outer boundary along rays FROM THE BULL
    and refit.

    `landmark_detection._trace_outer_boundary` walks rays from the colour
    mask's own centroid. Under perspective that centroid is not the image
    of the board centre, so those rays are not board radii and the
    "furthest mask pixel per angular bin" trace skews systematically --
    measured at +2.9 to +10.4 px mean radial error against a real
    reference ring. Rays from the bull ARE board radii exactly.

    Returns `(ellipse, note)`; the seed is returned unchanged, with a
    note, whenever the re-seat fails or violates a guardrail. Guardrails
    are all relative to the seed, so nothing here assumes a particular
    board size in pixels.
    """

    mask = _color_mask(image_bgr)
    component = _outer_ring_component_mask(mask)
    if component is None:
        return seed, "reseat skipped: no ring component"
    region = mask.astype(bool) & component
    h, w = region.shape
    bx, by = float(bull_px[0]), float(bull_px[1])

    n_rays = 720
    ray_angles = np.arange(n_rays, dtype=np.float64) * (360.0 / n_rays)
    seed_hits = ellipse_ray_intersections(seed, (bx, by), ray_angles)
    seed_radii = np.hypot(seed_hits[:, 0] - bx, seed_hits[:, 1] - by)

    # ------------------------------------------------------------------
    # Vectorised radial band trace (2026-09-11 calibration-speed pass).
    # This used to be a 720-iteration Python loop with an inner
    # sample-by-sample walk -- per-frame cost that held the GIL and
    # starved the live capture pump during calibration. Every decision
    # below reproduces the old loop's semantics element-for-element:
    #
    #  * per-ray `rs` grids are still built with np.arange (same start/
    #    stop/step expressions), so every sample radius is bit-identical
    #    -- rays only get PADDED to a shared width, and pad slots are
    #    excluded from the hit mask so they can never trace;
    #  * ux/uy still come from scalar math.cos/math.sin (a 720-element
    #    list comp costs microseconds) rather than np.cos/np.sin --
    #    numpy's SIMD float64 trig is not guaranteed bit-identical to
    #    libm's on every platform, and these values feed the fit;
    #  * "last" is the last hit index; the run-purity check computes the
    #    same boolean-mean over [max(0, last-10), last] via a cumulative
    #    sum (integer count / window length -- the exact division
    #    np.mean performed);
    #  * the inward walk `while first > 0 and (hit[first-1] or (first>1
    #    and hit[first-2]))` decrements through consecutive indices from
    #    `last`, so it stops at the LARGEST f <= last where that
    #    continue-condition is False. C[f] below IS that condition, so
    #    `first = max f <= last with C[f] False` -- and C[0] is always
    #    False (the `first > 0` term), so a maximum always exists.
    # ------------------------------------------------------------------
    valid_ray = np.isfinite(seed_radii) & (seed_radii > 1.0)
    ray_ids = np.nonzero(valid_ray)[0]
    dirs_a = np.zeros((0, 2), dtype=np.float64)
    r_outer_a = np.zeros(0, dtype=np.float64)
    r_inner_a = np.zeros(0, dtype=np.float64)
    if len(ray_ids):
        uxs = np.array(
            [math.cos(math.radians(float(ray_angles[i]))) for i in ray_ids],
            dtype=np.float64,
        )
        uys = np.array(
            [math.sin(math.radians(float(ray_angles[i]))) for i in ray_ids],
            dtype=np.float64,
        )
        rs_rows = [
            np.arange(0.70 * float(seed_radii[i]), 1.20 * float(seed_radii[i]), 0.5)
            for i in ray_ids
        ]
        klen = np.array([len(r) for r in rs_rows], dtype=np.int64)
        kmax = int(klen.max()) if len(klen) else 0
        if kmax > 0:
            m = len(ray_ids)
            rs_mat = np.zeros((m, kmax), dtype=np.float64)
            for row, rrow in enumerate(rs_rows):
                rs_mat[row, : len(rrow)] = rrow
            cols = np.arange(kmax)
            in_len = cols[None, :] < klen[:, None]
            xs = np.rint(bx + uxs[:, None] * rs_mat).astype(int)
            ys = np.rint(by + uys[:, None] * rs_mat).astype(int)
            ok = in_len & (xs >= 0) & (ys >= 0) & (xs < w) & (ys < h)
            hit = np.zeros((m, kmax), dtype=bool)
            hit[ok] = region[ys[ok], xs[ok]]

            any_hit = hit.any(axis=1)
            last = kmax - 1 - np.argmax(hit[:, ::-1], axis=1)  # valid where any_hit
            # Run-purity: same boolean mean over [run_lo, last] as before.
            cs = np.cumsum(hit, axis=1)
            run_lo = np.maximum(0, last - 10)
            rows = np.arange(m)
            count = cs[rows, last] - np.where(run_lo > 0, cs[rows, np.maximum(run_lo - 1, 0)], 0)
            window = last - run_lo + 1
            with np.errstate(invalid="ignore"):
                run_mean = count / window
            pure = any_hit & (run_mean >= 0.7)

            # C[f] = (f > 0) and (hit[f-1] or (f > 1 and hit[f-2]))
            C = np.zeros_like(hit)
            C[:, 1:] = hit[:, :-1]
            C[:, 2:] |= hit[:, :-2]
            cand = np.where(~C & (cols[None, :] <= last[:, None]), cols[None, :], -1)
            first = cand.max(axis=1)

            keep = pure
            if keep.any():
                kr = np.nonzero(keep)[0]
                dirs_a = np.stack([uxs[kr], uys[kr]], axis=1)
                r_outer_a = rs_mat[kr, last[kr]] + 0.25
                r_inner_a = rs_mat[kr, first[kr]] - 0.25

    if len(dirs_a) < 40:
        return seed, f"reseat skipped: only {len(dirs_a)} boundary points"

    cand = seed
    for _ in range(2):
        r_corrected = _debias_band_outer_radius(cand, (bx, by), dirs_a, r_inner_a, r_outer_a)
        pts = np.stack([bx + dirs_a[:, 0] * r_corrected,
                        by + dirs_a[:, 1] * r_corrected], axis=1)
        fitted, _kept = _robust_fit_ellipse(pts.astype(np.float32))
        if fitted is None:
            return seed, "reseat skipped: ellipse fit failed"
        cand = Ellipse.from_cv2(fitted)

    seed_scale = min(seed.major_axis_px, seed.minor_axis_px)
    if math.hypot(cand.cx - seed.cx, cand.cy - seed.cy) > RESEAT_MAX_CENTRE_SHIFT_FRACTION * seed_scale:
        return seed, "reseat rejected: centre moved too far"
    for new, old in ((cand.major_axis_px, seed.major_axis_px),
                     (cand.minor_axis_px, seed.minor_axis_px)):
        if old <= 1e-6 or abs(new - old) / old > RESEAT_MAX_AXIS_CHANGE_FRACTION:
            return seed, "reseat rejected: axis changed too much"
    dang = abs(cand.angle_deg - seed.angle_deg) % 180.0
    if min(dang, 180.0 - dang) > RESEAT_MAX_ANGLE_CHANGE_DEG:
        return seed, "reseat rejected: tilt changed too much"
    return cand, "reseat ok"


# ---------------------------------------------------------------------
# Stage 3 -- angular edge-energy profile about the bull
# ---------------------------------------------------------------------


def edge_magnitude(image_bgr: np.ndarray) -> np.ndarray:
    import cv2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def angular_edge_profile(
    edges: np.ndarray, ellipse: Ellipse, bull_px, n_bins: int = N_ANGLE_BINS
) -> np.ndarray:
    """Mean edge energy along each ray from the bull, over a band of the
    board's interior -- 20 sharp peaks, one per sector wire.

    The radial band is a FRACTION of that ray's own bull-to-ring
    distance, so it adapts to any board size in pixels and to the strong
    per-angle variation in that distance the bull's offset creates.
    """
    h, w = edges.shape[:2]
    bx, by = float(bull_px[0]), float(bull_px[1])
    fracs = np.linspace(PROFILE_RADIAL_LO, PROFILE_RADIAL_HI, PROFILE_RADIAL_SAMPLES)
    angles = np.arange(n_bins, dtype=np.float64) * (360.0 / n_bins)
    hits = ellipse_ray_intersections(ellipse, (bx, by), angles) # (n_bins, 2)
    r = np.hypot(hits[:, 0] - bx, hits[:, 1] - by) # (n_bins,)
    rad = np.radians(angles)
    # (n_bins, n_samples) sample coordinates
    xs = bx + np.cos(rad)[:, None] * r[:, None] * fracs[None, :]
    ys = by + np.sin(rad)[:, None] * r[:, None] * fracs[None, :]
    valid = np.isfinite(xs) & np.isfinite(ys)
    xi = np.rint(np.where(valid, xs, 0.0)).astype(np.int64)
    yi = np.rint(np.where(valid, ys, 0.0)).astype(np.int64)
    inside = valid & (xi >= 1) & (yi >= 1) & (xi < w - 1) & (yi < h - 1)
    vals = np.where(inside, edges[np.clip(yi, 0, h - 1), np.clip(xi, 0, w - 1)], 0.0)
    counts = inside.sum(axis=1)
    acc = np.where(counts > 0, vals.sum(axis=1) / np.maximum(counts, 1), 0.0)
    k = np.ones(PROFILE_SMOOTH_BINS) / PROFILE_SMOOTH_BINS
    pad = PROFILE_SMOOTH_BINS
    wrapped = np.concatenate([acc[-pad:], acc, acc[:pad]])
    return np.convolve(wrapped, k, mode="same")[pad:-pad]


def _spoke_offsets(n: int) -> np.ndarray:
    lo = max(1, int(round(SPOKE_NEIGHBOUR_LO_DEG / 360.0 * n)))
    hi = max(lo + 1, int(round(SPOKE_NEIGHBOUR_HI_DEG / 360.0 * n)))
    return np.concatenate([np.arange(-hi, -lo + 1), np.arange(lo, hi + 1)])


def spoke_score(profile: np.ndarray, angles_deg) -> float:
    """Edge energy on the predicted wires, minus the local background."""
    return float(spoke_scores(profile, np.asarray(angles_deg, dtype=np.float64)[None, :])[0])


def spoke_scores(profile: np.ndarray, angles_deg: np.ndarray) -> np.ndarray:
    """Vectorised `spoke_score` over a (n_candidates, 20) angle array."""
    n = len(profile)
    offsets = _spoke_offsets(n)
    ang = np.asarray(angles_deg, dtype=np.float64)
    idx = np.rint(np.nan_to_num(ang, nan=0.0) / 360.0 * n).astype(np.int64) % n
    on = profile[idx] # (P, 20)
    bg = profile[(idx[..., None] + offsets) % n].mean(axis=-1) # (P, 20)
    good = np.isfinite(ang)
    return np.where(good, on - bg, 0.0).sum(axis=-1)


# ---------------------------------------------------------------------
# Stage 4 -- projective phase lock
# ---------------------------------------------------------------------


def predicted_wire_angles(ellipse: Ellipse, bull_px, phase_deg: float) -> list[float]:
    """Image angles, about the bull, of the 20 sector wires under
    `H(phase)`. These are NOT 18 degrees apart -- that is the whole
    point."""
    H = board_to_image_homography(ellipse, bull_px, phase_deg)
    pts = apply_homography(
        H,
        [normalised_board_point(FIRST_WIRE_ANGLE_DEG + SECTOR_ANGLE_DEG * k)
         for k in range(N_SECTORS)],
    )
    return [_image_angle(p, bull_px) for p in pts]


def lock_phase(profile: np.ndarray, ellipse: Ellipse, bull_px) -> tuple[float, float, float]:
    """Search the one free parameter of the projective family.

    The 20-wire set is invariant under `phi -> phi + 18deg` (with a
    relabelling), so the continuous search only needs one sector's worth
    of phase; the remaining 20-fold choice is a separate, discrete
    problem handled by `lock_orientation()`.

    Returns `(phase_deg, score, phase_confidence)`, where the confidence
    is how many standard deviations the winning phase stands above the
    sweep's own score distribution. A ratio against the MEAN would be
    useless here: the spoke score is background-subtracted, so its mean
    over a full sweep sits near zero and the ratio explodes to
    meaningless four-figure numbers regardless of lock quality (measured:
    1557-13478 on genuinely good locks). A z-score is scale-free and
    actually discriminates.
    """
    A = affine_unit_circle_to_ellipse(ellipse)
    u = apply_homography(np.linalg.inv(A), [bull_px])[0]
    M = A @ disk_boost(-u)

    coarse = np.arange(0.0, SECTOR_ANGLE_DEG, PHASE_STEP_DEG)
    scores = spoke_scores(profile, _wire_angles_for_phases(M, bull_px, coarse))
    i = int(np.argmax(scores))
    best_phase, best_score = float(coarse[i]), float(scores[i])
    sd = float(scores.std())
    confidence = (best_score - float(scores.mean())) / sd if sd > 1e-9 else 0.0

    fine = np.arange(best_phase - PHASE_POLISH_SPAN_DEG,
                     best_phase + PHASE_POLISH_SPAN_DEG + 1e-9,
                     PHASE_POLISH_STEP_DEG)
    fine_scores = spoke_scores(profile, _wire_angles_for_phases(M, bull_px, fine))
    j = int(np.argmax(fine_scores))
    if float(fine_scores[j]) > best_score:
        best_phase, best_score = float(fine[j]), float(fine_scores[j])
    return best_phase % SECTOR_ANGLE_DEG, best_score, confidence


def refine_wire_angles(profile: np.ndarray, angles_deg) -> list[float]:
    """Pull each predicted wire onto the real local edge-energy peak,
    with parabolic sub-bin interpolation, inside a window far narrower
    than the inter-wire spacing so a wire can never capture its
    neighbour."""
    n = len(profile)
    half = max(1, int(round(WIRE_REFINE_WINDOW_DEG / 360.0 * n)))
    out = []
    for a in angles_deg:
        i0 = int(round(a / 360.0 * n)) % n
        best_i, best_v = i0, -math.inf
        for d in range(-half, half + 1):
            j = (i0 + d) % n
            if profile[j] > best_v:
                best_v, best_i = float(profile[j]), j
        ym = float(profile[(best_i - 1) % n])
        y0 = float(profile[best_i])
        yp = float(profile[(best_i + 1) % n])
        denom = ym - 2.0 * y0 + yp
        delta = 0.0
        if abs(denom) > 1e-9 and y0 >= ym and y0 >= yp:
            delta = max(-0.5, min(0.5, 0.5 * (ym - yp) / denom))
        refined = ((best_i + delta) * 360.0 / n) % 360.0
        if WIRE_REFINE_BLEND >= 1.0:
            out.append(refined)
        else:
            d = ((refined - a + 180.0) % 360.0) - 180.0
            out.append((a + WIRE_REFINE_BLEND * d) % 360.0)
    return out


# ---------------------------------------------------------------------
# Stage 5 -- absolute orientation
# ---------------------------------------------------------------------


# Board-constant sample geometry for double_colour_score(), built once
# (2026-09-11 calibration-speed pass). These 300 unit-disk points (20
# beds x 3 radial x 5 angular offsets) and the per-point bed index are
# pure regulation-geometry constants -- the pre-cached values are
# byte-identical to what the old per-call loop rebuilt every time, and
# `lock_orientation()` calls the scorer 20 times PER FRAME, so the
# rebuild (plus a full-frame cv2.cvtColor per call -- see
# `_double_colour_score_hsv()` below) was 20x redundant per frame.
def _double_bed_sample_tables() -> tuple[np.ndarray, np.ndarray]:
    mid = 0.5 * (DOUBLE_INNER_RADIUS_MM + DOUBLE_OUTER_RADIUS_MM) / DOUBLE_OUTER_RADIUS_MM
    span = 0.35 * (DOUBLE_OUTER_RADIUS_MM - DOUBLE_INNER_RADIUS_MM) / DOUBLE_OUTER_RADIUS_MM
    all_pts: list[tuple[float, float]] = []
    bed_idx: list[int] = []
    for j in range(N_SECTORS):
        centre = SECTOR_ANGLE_DEG * j
        for dr in (-span, 0.0, span):
            for da in (-5.0, -2.5, 0.0, 2.5, 5.0):
                all_pts.append(normalised_board_point(centre + da, mid + dr))
                bed_idx.append(j)
    return (
        np.asarray(all_pts, dtype=np.float64),
        np.asarray(bed_idx, dtype=np.int64),
    )


_DOUBLE_BED_SAMPLE_POINTS, _DOUBLE_BED_INDEX = _double_bed_sample_tables()
# 3x3 patch offsets, matching the old `hsv[yi-1:yi+2, xi-1:xi+2]` slice.
_PATCH_DY = np.arange(-1, 2).reshape(1, 3, 1)
_PATCH_DX = np.arange(-1, 2).reshape(1, 1, 3)


def double_colour_score(image_bgr: np.ndarray, H: np.ndarray) -> float:
    """How well the image's double beds match a regulation board's
    red/green alternation under the candidate map `H`.

    Rig-independent by construction: it tests a property of the BOARD
    (alternating doubles, sector 20 red), not of this installation.
    Range is roughly [-1, +1]; a correct lock measures near +1.

    Implementation note (2026-09-11 calibration-speed pass): now a thin
    wrapper over `_double_colour_score_hsv()` so `lock_orientation()`
    (which scores 20 candidate rotations of the SAME image) converts to
    HSV once instead of 20 times. Same numbers: the conversion is of the
    same image either way, and the scoring below is a pure vectorisation
    of the old per-point Python loop (see that function's own
    equivalence notes).
    """
    import cv2

    return _double_colour_score_hsv(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV), H)


def _double_colour_score_hsv(hsv: np.ndarray, H: np.ndarray) -> float:
    """`double_colour_score()` on an already-converted HSV image.

    Vectorised drop-in for the old per-point loop, equivalence argued
    point by point (2026-09-11 calibration-speed pass):

      * point placement -- `int(round(x))` on a float64 rounds half to
        even, exactly like `np.rint`; both produce the same integer for
        every finite input, and non-finite points are masked out first
        exactly like the old `np.isfinite(p).all()` continue.
      * the in-bounds gate is the same `1 <= xi < w-1` / `1 <= yi < h-1`
        window, so every gathered patch is a full 3x3, matching the old
        slice.
      * the red/green tests compare the same uint8 channel values
        against the same constants -- the old per-patch `.astype(int16)`
        casts changed no comparison outcome (pure comparisons, no
        arithmetic), so the per-pixel booleans are identical.
      * per-bed red/green tallies are integer sums -- order-independent,
        so accumulating them via np.add.at instead of the old
        point-by-point `+=` is exact.
      * the final `total` is accumulated bed-by-bed in the same j order
        with the same float64 expressions, so the sum is bit-identical.
    """
    h, w = hsv.shape[:2]
    mapped = apply_homography(H, _DOUBLE_BED_SAMPLE_POINTS)

    finite = np.isfinite(mapped).all(axis=1)
    xi = np.rint(np.where(finite, mapped[:, 0], 0.0)).astype(np.int64)
    yi = np.rint(np.where(finite, mapped[:, 1], 0.0)).astype(np.int64)
    ok = finite & (xi >= 1) & (xi < w - 1) & (yi >= 1) & (yi < h - 1)

    red_by_bed = np.zeros(N_SECTORS, dtype=np.int64)
    green_by_bed = np.zeros(N_SECTORS, dtype=np.int64)
    if ok.any():
        py = yi[ok][:, None, None] + _PATCH_DY  # (m, 3, 1) -> broadcast
        px = xi[ok][:, None, None] + _PATCH_DX  # (m, 1, 3)
        patch = hsv[py, px]  # (m, 3, 3, 3) uint8
        hh = patch[..., 0]
        ss = patch[..., 1]
        vv = patch[..., 2]
        red_hits = (((hh < 12) | (hh > 168)) & (ss > 70) & (vv > 50)).sum(axis=(1, 2))
        green_hits = ((hh > 35) & (hh < 95) & (ss > 40) & (vv > 40)).sum(axis=(1, 2))
        beds = _DOUBLE_BED_INDEX[ok]
        np.add.at(red_by_bed, beds, red_hits)
        np.add.at(green_by_bed, beds, green_hits)

    total, n_used = 0.0, 0
    for j in range(N_SECTORS):
        red = int(red_by_bed[j])
        green = int(green_by_bed[j])
        if red + green == 0:
            continue
        frac = (red - green) / float(red + green)
        total += frac if j in RED_DOUBLE_SECTOR_INDICES else -frac
        n_used += 1
    return total / n_used if n_used else 0.0


@dataclass(frozen=True)
class OrientationLock:
    roll: int
    colour_score: float
    colour_margin: float
    candidates: tuple[int, ...]
    used_hint: bool
    ambiguous: bool
    reason: str = ""


def lock_orientation(
    image_bgr: np.ndarray,
    ellipse: Ellipse,
    bull_px,
    phase_deg: float,
    *,
    orientation_hint_deg: float | None = None,
) -> OrientationLock:
    """Pick which of the 20 rotations of the locked wire set is the real
    one.

    Two signals, in order:

    1. **Double-ring colour alternation** (`double_colour_score`) -- pure
       image evidence, no rig knowledge, no template. This is a real
       property of any regulation board. Because the alternation has a
       period of two sectors it narrows 20 candidates to 10, and cannot
       by itself do better than that.

    2. **A per-camera orientation hint** -- the image angle, measured
       about the bull, toward board angle 0 (sector 20's centre). The 10
       surviving candidates are ~36 degrees apart, so this only has to be
       good to well within +-18 degrees.

    **Honest statement of the one remaining rig dependency.** Step 2 is
    the piece that is not derivable from the board alone, and two
    attempts to remove it were made and both failed on real data:

      * "A board is hung with 20 at the top and cameras are upright, so
        board-up projects to image-up." Refuted:
        a measurement of board angle 0 put it at
        image angles 343.7 / 88.2 / 195.3 degrees on cams
        0/1/2 -- about 115 degrees apart, i.e. tracking each camera's own
        mounting azimuth, not any board property.
      * A template-free number-ring lock, correlating a per-sector ink
        measurement against the regulation digit-count signature
        (2,1,2,1,2,1,2,2,1,2,1,2,1,2,1,2,2,1,2,1, which is unique under
        every non-trivial rotation). The number band rectifies legibly
        (a rectified strip literally reads 20 1 18 4 13 ...), but
        every measurement tried -- ink width, ink area, ink
        mass, with and without per-patch contrast normalisation and a
        circular high-pass, over three radial bands -- was dominated by
        the illumination gradient around the board rather than by digit
        count. Best result was 23/25 on one camera and one band, with
        0/25 on another camera at the same band. Not usable.

    So the hint stays, but deliberately NOT as a hardcoded correctness
    dependency: it is a caller argument, it has a documented one-time
    per-camera bootstrap (`derive_orientation_hint_deg()`), and when it
    is absent this returns the best colour candidate with
    `ambiguous=True` so a caller can reject rather than silently trust a
    1-in-10 guess.
    """
    import cv2

    # One HSV conversion for all 20 candidate rotations (2026-09-11
    # calibration-speed pass) -- double_colour_score() used to reconvert
    # the identical full frame on every one of these 20 calls; the
    # scorer itself is unchanged (see _double_colour_score_hsv()).
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    scores = []
    for roll in range(N_SECTORS):
        H = board_to_image_homography(
            ellipse, bull_px, phase_deg + SECTOR_ANGLE_DEG * roll
        )
        scores.append(_double_colour_score_hsv(hsv, H))
    arr = np.asarray(scores)
    best_colour = float(arr.max())
    margin = best_colour - float(arr.mean())
    # Every candidate within a whisker of the best is colour-equivalent;
    # on a clean board that is exactly the 10 same-parity rotations.
    keep = tuple(int(i) for i in np.nonzero(arr >= best_colour - 0.25 * max(margin, 1e-6))[0])
    if not keep:
        keep = (int(np.argmax(arr)),)

    if orientation_hint_deg is None:
        return OrientationLock(
            roll=int(np.argmax(arr)),
            colour_score=best_colour,
            colour_margin=margin,
            candidates=keep,
            used_hint=False,
            ambiguous=True,
            reason="no orientation hint: colour alternation alone leaves 10 candidates",
        )

    best_roll, best_dev = keep[0], math.inf
    for roll in keep:
        H = board_to_image_homography(
            ellipse, bull_px, phase_deg + SECTOR_ANGLE_DEG * roll
        )
        p = apply_homography(H, [normalised_board_point(0.0)])[0]
        if not np.isfinite(p).all():
            continue
        dev = abs(((_image_angle(p, bull_px) - orientation_hint_deg + 180.0) % 360.0) - 180.0)
        if dev < best_dev:
            best_dev, best_roll = dev, roll
    return OrientationLock(
        roll=best_roll,
        colour_score=float(arr[best_roll]),
        colour_margin=margin,
        candidates=keep,
        used_hint=True,
        ambiguous=best_dev > SECTOR_ANGLE_DEG,
        reason=f"hint deviation {best_dev:.1f} deg",
    )


# ---------------------------------------------------------------------
# Result / top-level driver
# ---------------------------------------------------------------------


@dataclass
class OrientedLandmarkResult:
    ok: bool
    reason: str = ""
    ellipse: Ellipse | None = None
    seed_ellipse: Ellipse | None = None
    bull_px: tuple[float, float] | None = None
    ring20_px: np.ndarray | None = None # (20, 2), ring index k = wire at 9+18k
    quad_px: np.ndarray | None = None # (4, 2), quad index order
    object_points_mm: np.ndarray | None = None # (4, 3)
    phase_deg: float = 0.0
    roll: int = 0
    normalised_bull_radius: float = 0.0
    spoke_score: float = 0.0
    phase_confidence: float = 0.0
    colour_score: float = 0.0
    colour_margin: float = 0.0
    orientation_ambiguous: bool = False
    notes: list[str] = field(default_factory=list)
    # Per-ring-index outcome of the local wire-junction refinement stage
    # (opendarts.calibration.wire_junction): "full" / "radial_only" /
    # "rejected" per landmark, or None when the stage was disabled or
    # never reached. Diagnostics only -- `ring20_px`/`quad_px` already
    # contain the refined coordinates wherever refinement passed its
    # gates and the untouched ellipse seed everywhere else.
    ring20_refine_modes: list[str] | None = None


# ---------------------------------------------------------------------
# Two-stage split of find_oriented_landmarks(), added 2026-08-20 for
# opendarts.live.capture_daemon.bootstrap_calibrations()'s live-derived
# orientation-hint wiring (opendarts.calibration.
# ring_correlation_orientation).
#
# WHY THIS SPLIT EXISTS -- read before changing either half.
# `find_oriented_landmarks()` needs `orientation_hint_deg` as an INPUT to
# its `lock_orientation()` call partway through. Every caller so far has
# had that hint available up front (a fixed per-rig constant). A caller
# that wants to DERIVE its own hint from the very frames it is about to
# process cannot know the hint before running detection once -- but it
# also should not have to run the (measured, real-cost -- see
# opendarts.live.capture_daemon's own ~110ms/frame timing note) full
# detection pipeline TWICE per frame just to get there. `lock_orientation()`
# is a small, cheap step relative to the seed-ellipse/bull/reseat/
# phase-lock work before it and the wire-refinement work after it, so the
# real fix is exposing the BEFORE-`lock_orientation()` stage as its own
# function: run it once per frame (this is also exactly the per-frame
# geometry the session-wide orientation solve needs), derive a
# session-wide hint from every frame's own geometry, then
# run the (cheap) AFTER-`lock_orientation()` stage once more per frame with
# that now-known hint.
#
# `find_oriented_landmarks()` itself is rewritten in terms of these two
# pieces below (`locate_pre_orientation_landmarks()` then
# `_finish_oriented_landmarks()`) but its own behaviour, signature, and
# docstring contract are UNCHANGED -- this is a pure refactor, verified by
# the existing `tests/test_oriented_landmarks*.py` suite (already covering
# `find_oriented_landmarks()` end to end) passing unmodified.
# ---------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PreOrientationLandmarks:
    """Everything `find_oriented_landmarks()` computes BEFORE its call to
    `lock_orientation()` -- seed ellipse, bull, the (optional) bull-
    anchored re-seat, and the projective phase lock. None of this depends
    on `orientation_hint_deg` at all, which is what makes it safe to
    compute once per frame and reuse for BOTH a session-level hint
    derivation pass (`opendarts.calibration.ring_correlation_orientation`,
    which reuses exactly this function's `ellipse`/`bull_px`/`phase_deg`
    fields) and, once that hint is known,
    the real correspondence pass (`correspond_landmarks_from_pre_orientation()`
    below).

    `ok=False` means detection failed at or before the phase lock (seed
    ellipse, bull, or a degenerate edge profile) -- `reason` names which
    stage, matching exactly the reasons `find_oriented_landmarks()` itself
    would have returned at this point. `ellipse`/`bull_px` may still be
    partially populated even on failure (whatever stage got that far),
    same as `OrientedLandmarkResult`'s own partial-failure fields.

    `eq=False`, fixing a verifier-found bug (2026-08-20): this dataclass
    is `frozen=True` with an `np.ndarray` field (`profile`) and a `list`
    field (`notes`). Plain `@dataclass(frozen=True)` (the default `eq=True`)
    auto-generates BOTH `__eq__` (comparing every field, including
    `profile` with `==`, which returns an array and raises
    "truth value ambiguous" the moment it's used in a boolean context)
    AND `__hash__` (frozen + eq=True hashes the field tuple, and both an
    ndarray and a list are themselves unhashable) -- either one raises the
    instant it is actually invoked (e.g. a caller ever puts one of these
    in a set/dict key, or a test framework's own assertion helper tries
    an equality compare). `eq=False` disables both generated methods,
    falling back to identity comparison (`is`), which is what every real
    usage of this dataclass in this module actually needs.
    """

    ok: bool
    reason: str
    ellipse: Ellipse | None
    seed_ellipse: Ellipse | None
    bull_px: tuple[float, float] | None
    normalised_bull_radius: float
    profile: np.ndarray | None
    phase_deg: float
    spoke_score: float
    phase_confidence: float
    notes: list[str]


def locate_pre_orientation_landmarks(
    image_bgr: np.ndarray,
    *,
    reseat: bool = True,
    white_balance: bool = True,
) -> tuple[np.ndarray, PreOrientationLandmarks]:
    """The seed-ellipse -> bull -> reseat -> angular-profile -> phase-lock
    stages of `find_oriented_landmarks()`, factored out (see this
    section's header comment for why). Returns `(image_bgr_used,
    PreOrientationLandmarks)` -- `image_bgr_used` is the possibly
    illuminant-normalised image every downstream stage (this function's
    own reseat/bull-relock, and a caller's later
    `correspond_landmarks_from_pre_orientation()` / number-ring hint
    derivation) must keep using, exactly as `find_oriented_landmarks()`
    itself reassigns its own local `image_bgr` after white-balancing.

    `reseat`/`white_balance` mean exactly what they mean on
    `find_oriented_landmarks()` -- forwarded unchanged.
    """
    notes: list[str] = []

    if white_balance:
        image_bgr = normalise_illuminant(image_bgr)

    seed_det = detect_double_ring_quad(image_bgr)
    if not seed_det.ok or seed_det.ellipse is None:
        return image_bgr, PreOrientationLandmarks(
            ok=False, reason=f"seed ellipse failed: {seed_det.reason}",
            ellipse=None, seed_ellipse=None, bull_px=None,
            normalised_bull_radius=0.0, profile=None, phase_deg=0.0,
            spoke_score=0.0, phase_confidence=0.0, notes=notes,
        )
    seed = seed_det.ellipse

    # Image-level bull evidence computed ONCE per frame and shared by
    # both detect_bull() calls below (seed + post-reseat) -- see
    # _BullFrameEvidence's own docstring; the two calls previously
    # recomputed identical full-frame HSV/threshold/component products.
    bull_evidence = _bull_frame_evidence(image_bgr)
    bull = detect_bull(image_bgr, seed, evidence=bull_evidence)
    if not bull.ok or bull.xy is None:
        return image_bgr, PreOrientationLandmarks(
            ok=False, reason=f"bull detection failed: {bull.reason}",
            ellipse=None, seed_ellipse=seed, bull_px=None,
            normalised_bull_radius=0.0, profile=None, phase_deg=0.0,
            spoke_score=0.0, phase_confidence=0.0, notes=notes,
        )

    ellipse = seed
    if reseat:
        ellipse, note = reseat_ellipse(image_bgr, seed, bull.xy)
        notes.append(note)
        # The bull is worth re-locking to the re-seated ellipse: the area
        # and centrality terms in detect_bull() both key off it.
        bull2 = detect_bull(image_bgr, ellipse, evidence=bull_evidence)
        if bull2.ok and bull2.xy is not None:
            bull = bull2

    u = normalised_bull(ellipse, bull.xy)
    r_u = float(math.hypot(u[0], u[1]))
    if not np.isfinite(r_u) or r_u >= MAX_NORMALISED_BULL_RADIUS:
        return image_bgr, PreOrientationLandmarks(
            ok=False,
            reason=f"bull too far off the ellipse centre in normalised units (|u|={r_u:.3f})",
            ellipse=ellipse, seed_ellipse=seed, bull_px=tuple(bull.xy),
            normalised_bull_radius=r_u, profile=None, phase_deg=0.0,
            spoke_score=0.0, phase_confidence=0.0, notes=notes,
        )

    edges = edge_magnitude(image_bgr)
    profile = angular_edge_profile(edges, ellipse, bull.xy)
    if not np.isfinite(profile).all() or float(profile.max()) <= 0.0:
        # normalised_bull_radius=0.0, NOT r_u -- fixing a verifier-found
        # purity gap in this split (2026-08-20): the pre-split
        # `find_oriented_landmarks()` never passed `normalised_bull_radius`
        # on this specific early-return branch (only the "bull too far off
        # centre" branch above did), so it defaulted to
        # `OrientedLandmarkResult`'s own `normalised_bull_radius: float =
        # 0.0` field default here, even though `r_u` was already computed
        # by this point. This branch now reproduces that exact behaviour
        # rather than "improving" it, matching this split's own claimed-
        # pure-refactor contract (see this section's header comment) --
        # r_u is a real, already-computed value that WOULD be more
        # informative here, but changing it is a real behaviour change
        # outside this task's scope, not a bug fix.
        return image_bgr, PreOrientationLandmarks(
            ok=False, reason="degenerate angular edge profile",
            ellipse=ellipse, seed_ellipse=seed, bull_px=tuple(bull.xy),
            normalised_bull_radius=0.0, profile=None, phase_deg=0.0,
            spoke_score=0.0, phase_confidence=0.0, notes=notes,
        )

    phase, score, confidence = lock_phase(profile, ellipse, bull.xy)

    return image_bgr, PreOrientationLandmarks(
        ok=True, reason="ok",
        ellipse=ellipse, seed_ellipse=seed, bull_px=tuple(bull.xy),
        normalised_bull_radius=r_u, profile=profile, phase_deg=phase,
        spoke_score=score, phase_confidence=confidence, notes=notes,
    )


def _finish_oriented_landmarks(
    image_bgr: np.ndarray,
    pre: PreOrientationLandmarks,
    *,
    orientation_hint_deg: float | None,
    refine_wires: bool,
    local_refine: bool,
    min_phase_confidence: float | None,
) -> OrientedLandmarkResult:
    """The `lock_orientation()` -> wire-refine -> quality-gate stages of
    `find_oriented_landmarks()`, given an already-computed
    `PreOrientationLandmarks` (`pre.ok` must be True -- callers with a
    failed `pre` should not call this; `find_oriented_landmarks()` itself
    returns early in that case, matching its pre-split behaviour exactly).
    `image_bgr` must be the SAME (possibly white-balanced) image
    `locate_pre_orientation_landmarks()` returned alongside `pre` -- every
    stage here (`lock_orientation()`, `refine_ring_junctions()`) re-reads
    real pixels, not just `pre`'s already-derived geometry.
    """
    gate = MIN_PHASE_CONFIDENCE if min_phase_confidence is None else float(min_phase_confidence)
    ellipse = pre.ellipse
    bull_px = pre.bull_px
    profile = pre.profile
    phase = pre.phase_deg
    score = pre.spoke_score
    confidence = pre.phase_confidence
    notes = list(pre.notes)
    seed = pre.seed_ellipse

    lock = lock_orientation(
        image_bgr, ellipse, bull_px, phase, orientation_hint_deg=orientation_hint_deg
    )
    full_phase = phase + SECTOR_ANGLE_DEG * lock.roll

    angles = predicted_wire_angles(ellipse, bull_px, full_phase)
    if refine_wires:
        angles = refine_wire_angles(profile, angles)

    ring = []
    for a in angles:
        hit = ellipse_ray_intersection(ellipse, bull_px, a)
        if hit is None:
            return OrientedLandmarkResult(
                ok=False, reason="a wire ray missed the ring ellipse",
                ellipse=ellipse, seed_ellipse=seed, bull_px=bull_px,
            )
        ring.append(hit)
    ring20 = np.asarray(ring, dtype=np.float64)

    refine_modes: list[str] | None = None
    if local_refine:
        # Local per-junction refinement OFF the shared ellipse -- see
        # opendarts.calibration.wire_junction. Runs on the same
        # (white-balanced, if enabled) image every other stage used.
        from opendarts.calibration.wire_junction import refine_ring_junctions

        H = board_to_image_homography(ellipse, bull_px, full_phase)
        refinements = refine_ring_junctions(
            image_bgr, ellipse, bull_px, angles, ring20, H,
            DOUBLE_INNER_RADIUS_MM / DOUBLE_OUTER_RADIUS_MM,
        )
        refine_modes = [r.mode for r in refinements]
        for i, r in enumerate(refinements):
            if r.ok:
                ring20[i] = r.xy
        n_full = sum(1 for r in refinements if r.mode == "full")
        n_rad = sum(1 for r in refinements if r.mode == "radial_only")
        notes.append(
            f"local refine: {n_full + n_rad}/20 refined "
            f"({n_full} full, {n_rad} radial-only)"
        )

    quad = ring20[list(AD_QUAD_RING_INDICES)]

    result = OrientedLandmarkResult(
        ok=True,
        reason="ok",
        ellipse=ellipse,
        seed_ellipse=seed,
        bull_px=(float(bull_px[0]), float(bull_px[1])),
        ring20_px=ring20,
        quad_px=quad,
        object_points_mm=ad_quad_object_points_mm(),
        phase_deg=full_phase % 360.0,
        roll=lock.roll,
        normalised_bull_radius=pre.normalised_bull_radius,
        spoke_score=score,
        phase_confidence=confidence,
        colour_score=lock.colour_score,
        colour_margin=lock.colour_margin,
        orientation_ambiguous=lock.ambiguous,
        notes=(notes + [lock.reason]) if lock.reason else notes,
        ring20_refine_modes=refine_modes,
    )

    if confidence < gate:
        result.ok = False
        result.reason = f"phase lock untrustworthy (confidence {confidence:.2f} sigma)"
    elif lock.colour_margin < MIN_COLOUR_MARGIN:
        result.ok = False
        result.reason = f"orientation lock untrustworthy (colour margin {lock.colour_margin:.3f})"
    return result


def ad_quad_object_points_mm() -> np.ndarray:
    """The 4 cardinal-quad landmarks as board-mm points on the Z=0 plane,
    in quad index order (board angles 9 / 99 / 189 / 279).

    Independently constructed from `opendarts.geometry.board` here rather
    than imported from `sector_correspondence`, which this module is
    deliberately parallel to and does not depend on. The two agree by
    construction: quad index i is ring index 5i, i.e. board angle
    9 + 90i.
    """
    out = np.zeros((4, 3), dtype=np.float64)
    for i, k in enumerate(AD_QUAD_RING_INDICES):
        angle = FIRST_WIRE_ANGLE_DEG + SECTOR_ANGLE_DEG * k
        x, y = polar_to_xy_mm(DOUBLE_OUTER_RADIUS_MM, angle)
        out[i] = (x, y, 0.0)
    return out


def find_oriented_landmarks(
    image_bgr: np.ndarray,
    *,
    orientation_hint_deg: float | None = None,
    refine_wires: bool = True,
    reseat: bool = True,
    min_phase_confidence: float | None = None,
    white_balance: bool = True,
    local_refine: bool = True,
) -> OrientedLandmarkResult:
    """Locate the board's 20 double-outer wire points, and the 4-point
    cardinal quad among them, in one real camera frame.

    Multi-stage with a real reject at every stage, so a bad step fails
    loudly instead of cascading: seed ellipse -> bull -> bull-anchored
    re-seat -> angular profile -> projective phase lock -> orientation
    lock -> bounded per-wire refine -> quality gates.

    `min_phase_confidence` overrides the module's `MIN_PHASE_CONFIDENCE`
    quality gate for this call only; `None` (the default, and what every
    production caller uses) means the module constant. Pass `0.0` to
    disable the gate entirely -- the result then reports its real
    `phase_confidence` with `ok=True` regardless, which is exactly what a
    threshold sweep or a leave-one-session-out cross-validation needs in
    order to see the frames a stricter gate WOULD have rejected. See
    `MIN_PHASE_CONFIDENCE`'s own comment for the circularity this exists
    to prevent. Note the gate is the only thing this changes: every stage
    above it, and the colour-margin gate below it, run identically.

    `white_balance` (default True) runs `normalise_illuminant()` over the
    frame first and uses the corrected image for every stage. See that
    function's own comment for the real measured failure it fixes and
    why the correction belongs here rather than in the phase gate. Pass
    False to see the raw, uncorrected behaviour -- which is what a
    before/after measurement of the correction itself needs, and is the
    only thing this flag is for.

    `local_refine` (default True) runs the local wire-junction
    refinement stage (`opendarts.calibration.wire_junction`) over all 20
    ring landmarks after they are placed on the fitted ellipse: each
    landmark's ellipse position becomes a SEED, and the true junction of
    the sector wire with the double-outer ring wire is re-located from
    local ridge evidence, bounded and quality-gated per point, with a
    per-point fallback to the ellipse seed. This is the fix for the
    measured systematic radial/tangential landmark bias a shared global
    ellipse structurally cannot represent (see that module's docstring
    for the numbers). Pass False to get the pure ellipse-based
    landmarks -- which is what a before/after measurement of the
    refinement itself needs, and is the only thing this flag is for.

    **Implementation note (2026-08-20)**: this is now a thin composition
    of `locate_pre_orientation_landmarks()` (seed ellipse -> bull ->
    reseat -> phase lock) and `_finish_oriented_landmarks()`
    (`lock_orientation()` -> wire refine -> quality gates) -- see the
    section header comment above `PreOrientationLandmarks` for why. Pure
    refactor: this function's own behaviour is unchanged.
    """
    processed_bgr, pre = locate_pre_orientation_landmarks(
        image_bgr, reseat=reseat, white_balance=white_balance
    )
    if not pre.ok:
        return OrientedLandmarkResult(
            ok=False, reason=pre.reason,
            ellipse=pre.ellipse, seed_ellipse=pre.seed_ellipse, bull_px=pre.bull_px,
            normalised_bull_radius=pre.normalised_bull_radius,
        )
    return _finish_oriented_landmarks(
        processed_bgr, pre,
        orientation_hint_deg=orientation_hint_deg,
        refine_wires=refine_wires,
        local_refine=local_refine,
        min_phase_confidence=min_phase_confidence,
    )


def derive_orientation_hint_deg(
    image_bgr: np.ndarray, board_to_image: np.ndarray
) -> float | None:
    """One-time per-camera bootstrap for `orientation_hint_deg`.

    Given ANY already-trusted board-mm -> pixel mapping for a camera (a
    known-good homography, a PnP projection, or an operator-confirmed
    calibration), returns the image angle about the detected bull toward
    board angle 0 -- which is the only thing `lock_orientation()` needs.
    This is what makes the orientation hint a per-camera bootstrap step
    rather than a constant baked into this module: a new rig runs this
    once per camera and is done, with no code change.

    Illuminant-normalises first, for the same reason and by the same
    call `find_oriented_landmarks()` does -- the hint is measured about
    the DETECTED bull, and a bull found under a drifted white balance
    moves the angle (measured on this rig: cam2 deviations up to 50
    degrees traced to exactly that).
    """
    image_bgr = normalise_illuminant(image_bgr)
    seed_det = detect_double_ring_quad(image_bgr)
    if not seed_det.ok or seed_det.ellipse is None:
        return None
    bull = detect_bull(image_bgr, seed_det.ellipse)
    if not bull.ok or bull.xy is None:
        return None
    x, y = polar_to_xy_mm(DOUBLE_OUTER_RADIUS_MM, 0.0)
    with np.errstate(all="ignore"):
        q = np.asarray(board_to_image, dtype=np.float64) @ np.array([x, y, 1.0])
    if not np.isfinite(q).all() or abs(q[2]) < 1e-12:
        return None
    return _image_angle((q[0] / q[2], q[1] / q[2]), bull.xy)


def correspond_landmarks_oriented(
    image_bgr: np.ndarray,
    camera_index: int,
    *,
    orientation_hints_deg: dict[int, float] | None = None,
    min_phase_confidence: float | None = None,
    results_out: list[OrientedLandmarkResult] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Drop-in shape-compatible parallel to
    `sector_correspondence.detect_and_correspond()`: returns
    `(object_points_mm (4,3), image_points_px (4,2))` or None.

    Index i is the same quad index that module uses, so the output
    feeds `opendarts.pipeline.calibrate_camera()` unchanged, and can
    be averaged across frames by
    `sector_correspondence.average_correspondences()` (index i means the
    same physical board point on every call, by construction).

    `min_phase_confidence` is forwarded to `find_oriented_landmarks()`
    unchanged (see its docstring) -- `None`, the default, means the
    shipped `MIN_PHASE_CONFIDENCE`.

    **An ambiguous orientation lock is a REJECT here, not a warning.**
    `find_oriented_landmarks()` deliberately leaves `ok=True` with
    `orientation_ambiguous=True` when the colour lock and the caller's
    hint disagree by more than a whole sector (or when no hint was given
    at all, in which case the colour alternation alone leaves 10
    equally-good candidates) -- it reports, and lets the caller decide.
    This function IS that decision for the production calibration path:
    an ambiguous lock means the 4 returned pixels may belong to a
    rotation of the board 90/180/270 degrees away from the truth, which
    would not degrade a calibration but INVERT it, silently and
    confidently. No calibration is strictly better than that, so this
    returns None.

    `results_out`, if given, is appended with the full
    `OrientedLandmarkResult` for this call whether it succeeded or not --
    a diagnostics sink so a caller (e.g.
    `opendarts.live.capture_daemon.bootstrap_calibrations()`) can say WHY a
    frame was rejected, and specifically distinguish an ambiguous
    orientation lock from an ordinary detection failure, without paying
    for a second detection pass over the same frame.
    """
    hint = (orientation_hints_deg or {}).get(camera_index)
    result = find_oriented_landmarks(
        image_bgr,
        orientation_hint_deg=hint,
        min_phase_confidence=min_phase_confidence,
    )
    if results_out is not None:
        results_out.append(result)
    if not result.ok or result.quad_px is None or result.object_points_mm is None:
        return None
    if result.orientation_ambiguous:
        return None
    return result.object_points_mm, result.quad_px


def correspond_landmarks_from_pre_orientation(
    image_bgr: np.ndarray,
    pre: PreOrientationLandmarks,
    *,
    orientation_hint_deg: float | None = None,
    min_phase_confidence: float | None = None,
    results_out: list[OrientedLandmarkResult] | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Same contract and same reject rules as `correspond_landmarks_oriented()`
    (an ambiguous orientation lock is a reject here too, for the identical
    reason -- see that function's own docstring), for a caller that
    ALREADY has this frame's `PreOrientationLandmarks` from
    `locate_pre_orientation_landmarks()`.

    This is the second pass of a two-pass bootstrap that derives its own
    `orientation_hint_deg` from the very frames it is processing (see
    `opendarts.calibration.ring_correlation_orientation`
    and `opendarts.live.capture_daemon.bootstrap_calibrations()`'s own
    "LIVE-DERIVED ORIENTATION HINT" docstring section) rather than a
    pre-existing constant, and so cannot know the hint before the
    pre-orientation stage has already run once for every frame in the
    batch. Skips re-running seed-ellipse/bull/reseat/phase-lock detection
    entirely -- the real, measured-costly part of
    `correspond_landmarks_oriented()` for a bootstrap capturing up to
    `opendarts.live.capture_daemon.CALIBRATION_MAX_N_FRAMES` frames per
    camera -- a plain two-call composition of
    `correspond_landmarks_oriented()` (once to get geometry for the hint,
    once more with the hint known) would double that real per-frame cost
    for no benefit, since the pre-orientation geometry does not depend on
    the hint at all. `image_bgr` must be the SAME (possibly white-
    balanced) image `locate_pre_orientation_landmarks()` returned
    alongside `pre` -- same requirement `_finish_oriented_landmarks()`
    has.

    `pre.ok=False` is handled the same way `find_oriented_landmarks()`'s
    own early-return would be: this returns `None`, and (if `results_out`
    is given) appends an `OrientedLandmarkResult(ok=False, reason=pre.reason,
    ...)` carrying `pre`'s own partial fields -- so a caller's per-reason
    failure breakdown (e.g. `bootstrap_calibrations()`'s own rejection-
    reason log line) sees the SAME reason string a single-pass
    `find_oriented_landmarks()` call would have produced for this frame,
    not a generic "skipped" placeholder.
    """
    if not pre.ok:
        if results_out is not None:
            results_out.append(OrientedLandmarkResult(
                ok=False, reason=pre.reason,
                ellipse=pre.ellipse, seed_ellipse=pre.seed_ellipse, bull_px=pre.bull_px,
                normalised_bull_radius=pre.normalised_bull_radius,
            ))
        return None
    result = _finish_oriented_landmarks(
        image_bgr, pre,
        orientation_hint_deg=orientation_hint_deg,
        refine_wires=True,
        local_refine=True,
        min_phase_confidence=min_phase_confidence,
    )
    if results_out is not None:
        results_out.append(result)
    if not result.ok or result.quad_px is None or result.object_points_mm is None:
        return None
    if result.orientation_ambiguous:
        return None
    return result.object_points_mm, result.quad_px
