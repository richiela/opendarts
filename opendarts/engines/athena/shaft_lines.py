"""Board-plane shaft LINES and their pairwise intersections -- a second,
structurally different kind of evidence for Athena's consensus.

**The concept, and why it is real new evidence rather than a re-blend of
what the engine already has**: each camera's own detection localizes the
dart two different ways at once -- a POINT (where along the shaft the tip
sits, `crossing_px`) and a LINE (the shaft's own image axis, `axis_unit`).
The point is the noisy part: this engine's whole weight design already
documents that ray-plane intersection amplifies pixel-level TIP
localization noise into large board-plane displacement, and the detector's
biggest observed errors are exactly along-shaft (a diff mask that ends
early on a dark shaft slides the tip along the axis). The LINE is much
more stable -- it comes from the whole blob's PCA axis, hundreds of
pixels, not the few most-extreme ones. Projecting that image line through
the camera onto the board plane gives a board-plane line the dart's true
tip must sit on REGARDLESS of where along the shaft the tip pixel was
picked; intersecting two cameras' such lines therefore yields a tip
estimate immune to both cameras' along-shaft localization error -- the
exact error family the per-camera point reads cannot escape. Each
intersection feeds Athena's own continuous confidence-weighted consensus
as one more WEIGHTED READ in the existing Weiszfeld blend (see
`engine._combine_reads`) rather than casting a discrete label vote --
label votes were measured twice on real corpora and rejected both times
(engine.py's `_combine_reads` docstring records the numbers and the
correlated-bias trap).

**Measured before designing anything** (full real 420-throw
data/archive/clean/ corpus, fresh
wire-junction-refined per-session calibration, vs the oracle's tip_xy_mm):
per-camera point reads median error 2.81mm (n=1029); pairwise line
intersections median 2.24mm (n=840) -- genuinely better evidence, not
just different. And the intersections' one real failure mode is cleanly
identifiable from geometry alone: crossings shallower than ~15 degrees
are unconditioned (median 9.8mm, p90 487mm) while every bucket at or
above 15 degrees is well-behaved (median 2.1-2.7mm). Hence the two gates
below: a hard minimum crossing angle, and a sin(angle) factor in the
weight so a barely-passing crossing still counts less than an orthogonal
one.
"""
from __future__ import annotations

import math

import numpy as np

from opendarts.engines.athena.plane_intersect import ray_plane_intersect
from opendarts.pipeline import CameraCalibration
from opendarts.triangulation.rays import back_project_ray

# Pixel step along the detected image axis used to pick the SECOND point
# defining the shaft's image line. Any two distinct points on the same
# image line define the same board-plane line, so this is a numerical-
# conditioning choice, not a tuned parameter -- 40px is comfortably above
# sub-pixel noise and comfortably inside every real frame.
AXIS_STEP_PX = 40.0

# Hard floor on the crossing angle between two cameras' board-plane shaft
# lines before their intersection is admitted as a read at all. Two
# separate real measurements behind the value:
# 1. Error-vs-angle census (module docstring): the <15-degree bucket is
#    the only pathological one (median 9.8mm, p90 487mm); every bucket at
#    or above 15 degrees is well-behaved.
# 2. Full-engine sweep, 0-30 degrees x INTERSECTION_WEIGHT_SCALE
#    0.0-3.0, real 420-throw data/archive/clean/ corpus, fresh
#    per-session wire-junction calibration, corroboration override
#    active, at the final (RADIAL_CORRECTION_MM=0.25,
#    BOARD_PLANE_Z_MM=-0.75) operating point: at scale=1.0 angles
#    0/10/15 all measure an
#    identical 411/420 and 20-30 measure 410 -- i.e. on this corpus the
#    sin(angle) weight + MAX_INTERSECTION_RADIUS_MM cap already neutralise
#    the sub-15-degree pathology on their own (those intersections never
#    flip a label), and LOSO across all four sessions picks the 0-15
#    shelf in every fold (held-out total 410/420, equal to in-sample).
#    15 shipped rather than 0: zero measured cost on the corpus, and it
#    keeps the hard guarantee that measurement 1's genuinely pathological
#    bucket can never reach the blend at all -- protection by measured
#    evidence, not by hope that the soft weights always suffice.
MIN_CROSSING_ANGLE_DEG = 15.0

# Global scale on every intersection read's weight, relative to the
# geometric mean of the two contributing cameras' own point-read weights.
# The sin(angle) factor below already handles per-pair conditioning; this
# constant sets how much the blend trusts intersection evidence as a CLASS
# against point evidence. 1.0 -- equal class trust, the natural default --
# is also the measured optimum of the joint sweep above (at angle=15,
# final rz operating point: 0.75 -> 410, 1.0 -> 411, 1.5 -> 409,
# 2.0 -> 406, 3.0 -> 400): intersection evidence neither needs boosting
# nor damping relative to point evidence, so this ships as the no-op
# value rather than a fitted knob.
INTERSECTION_WEIGHT_SCALE = 1.0

# An intersection landing implausibly far off the board is a broken pair
# (bad line on at least one side), not a real "outside" read -- the double
# wire is at 170mm and every genuinely-thrown dart in every real corpus
# session lands well inside 250mm. Generous on purpose: this is a sanity
# cap, not a fitted constant, and it must never eat a real near-the-board
# "outside" answer.
MAX_INTERSECTION_RADIUS_MM = 250.0


def board_plane_line(
    crossing_px: tuple[float, float],
    axis_unit: tuple[float, float],
    calib: CameraCalibration,
    cam: int,
    z_mm: float,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Project one camera's detected shaft image line onto the board
    plane: back-project two points of the image line and intersect each
    ray with the plane. Returns two board-plane points defining the line,
    or None when either ray fails to cross the plane validly."""
    p0 = crossing_px
    p1 = (p0[0] + AXIS_STEP_PX * axis_unit[0], p0[1] + AXIS_STEP_PX * axis_unit[1])
    pts = []
    for px in (p0, p1):
        ray = back_project_ray(
            px, calib.camera_matrix, calib.dist_coeffs, calib.rvec, calib.tvec, cam=cam
        )
        xy = ray_plane_intersect(ray, z_mm=z_mm)
        if xy is None:
            return None
        pts.append(xy)
    a, b = pts
    if math.hypot(b[0] - a[0], b[1] - a[1]) < 1e-9:
        return None
    return (a, b)


def intersect_board_lines(
    line_a: tuple[tuple[float, float], tuple[float, float]],
    line_b: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], float] | None:
    """Intersect two board-plane lines. Returns ((x_mm, y_mm),
    crossing_angle_deg) or None for (near-)parallel lines. The angle is
    the acute angle between the two lines -- the conditioning signal the
    weight formula and the hard floor both use."""
    (x1, y1), (x2, y2) = line_a
    (x3, y3), (x4, y4) = line_b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den

    va = np.array([x2 - x1, y2 - y1], dtype=np.float64)
    vb = np.array([x4 - x3, y4 - y3], dtype=np.float64)
    va /= np.linalg.norm(va) + 1e-12
    vb /= np.linalg.norm(vb) + 1e-12
    cosang = min(1.0, abs(float(va @ vb)))
    angle_deg = math.degrees(math.acos(cosang))
    return ((float(px), float(py)), angle_deg)


def intersection_weight(weight_a: float, weight_b: float, angle_deg: float) -> float:
    """One intersection read's consensus weight: geometric mean of the two
    contributing cameras' own point-read weights (so a pair of trusted
    cameras yields a trusted intersection, and one distrusted camera drags
    the pair down), times sin(crossing angle) for conditioning (an
    orthogonal crossing pins the point; a shallow one barely does), times
    the class-level scale."""
    base = math.sqrt(max(0.0, weight_a) * max(0.0, weight_b))
    return INTERSECTION_WEIGHT_SCALE * base * math.sin(math.radians(angle_deg))
