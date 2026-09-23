"""Image-only, per-camera radial-distortion (k1) derivation AT
CALIBRATION TIME -- fits ALONGSIDE `opendarts.calibration.focal_length`'s
homography-based focal-length derivation (same ring20 correspondence
pool, same calibration event), replacing `dist_coeffs=np.zeros(5)`'s
hardcoded zero-lens-distortion assumption with a real, per-camera,
live-derived value when the data safely supports it.

REAL, CONFIRMED PROBLEM this task set out to investigate: one specific
camera (currently cam0, though WHICH physical camera sits at that index
can change -- see focal_length.py's own module docstring for why) shows
a real, systematic, camera-specific reprojection-error floor around
2.9-3.1px that neither the best-available focal length nor pose can
close -- two independent methods (a manual focal sweep, and
`dev/calibration/intrinsics_estimation.py`'s joint optimizer) land on
essentially the identical number (2.944px / 2.941px) from structurally
different approaches, and 10 fresh live calibrations the same day show
the same floor (3.025-3.104px), repeatably. That signature -- a
consistent, camera-specific, non-zero residual after the best possible
focal+pose fit -- is exactly what un-modeled lens distortion looks like.
(A DIFFERENT camera, cam2, shows real run-to-run VARIANCE with no
consistent floor instead -- noise, not bias; this module's own real-data
validation confirms distortion-fitting correctly does NOT help that
camera, see this module's own tests.)

WHY THIS WAS CORRECTLY REJECTED BEFORE, AND WHAT CHANGED
----------------------------------------------------------------------
`opendarts.live.capture_daemon._try_solve()`'s own dated comment
(2026-08-12) already investigated exactly this
question and correctly said no: the live pose solve's correspondence
budget was only the 4-point cardinal quad -- solving (f, rvec, tvec, k1) jointly
from 4 points is 8 unknowns from 8 equations, zero DOF of slack, "not
safely identifiable." This module does NOT reopen that specific
4-point solve. `opendarts.calibration.focal_length` (2026-08-26, same day)
established that the SAME per-frame detection pass already produces ALL
20 of the double-ring's wire-junction landmarks
(`OrientedLandmarkResult.ring20_px`), not just the 4-point quad
subset -- 40 equations for the same 8 unknowns (f, rvec, tvec, k1), a
completely different, much better-conditioned regime. Whether 20 points
is actually ENOUGH to safely add distortion on top is exactly what this
module's own synthetic conditioning analysis (below) answers -- measured,
not assumed.

CONDITIONING VALIDATED SYNTHETICALLY FIRST. Swept at this rig's real geometry
(W=1280,H=720, ring_radius_mm=500, height_mm=300, n_cams=3, real focal
lengths 800.0/830.0/835.0px, board ring radius 170mm):

  - NOISELESS: jointly recovering (f, rvec, tvec, k1) from the 20-point
    ring, for every injected k1 in [-0.5, -0.1] at all 3 real cameras,
    is EXACT to the numerical floor (~1e-11px RMS) -- and, notably, all
    7 multi-init FOV seeds (50-130deg) converge to the IDENTICAL global
    optimum every single time (spread-across-inits std == 0.0 in every
    noisy trial too, see below) -- a completely different, much better-
    conditioned regime than the 4-point case, which this project's own
    prior history shows CAN land in different local optima from
    different seeds. (This also means "spread across multi-init seeds,"
    the confidence signal `MultiInitEstimationResult` uses for the
    4-point case, is NOT an informative per-event confidence signal
    here -- this module does not rely on it; it keeps multi-init purely
    for robustness against a real-world geometry this synthetic sweep
    did not cover, and reports n_converged/n_attempted for diagnostics
    only.)
  - k2 is explicitly NOT solved for -- confirmed NOT separately
    identifiable from this same 20-point ring: adding k2 as a free 9th
    unknown, at 0.3-0.5px per-point noise, inflates k1's OWN std by
    >10x (0.10-0.18 vs 0.006-0.03 for k1 alone) and k2's own std is
    0.77-1.42 -- classic radial-distortion aliasing for a single-plane,
    limited-radius-range target. Fixing k2=p1=p2=k3=0 and solving for k1
    only is a measured decision, not a guess (`_project_with_jacobian`
    below only ever varies dist index 0).
  - REALISTIC noise (0.3-0.5px per point, this project's own documented
    landmark-detection precision), with per-index MEDIAN averaging
    across N=30-50 frames (`average_ring20_image_points`, matching
    `CALIBRATION_N_FRAMES_DETECT`): k1 recovers with ~0 bias, std
    0.004-0.007, max |error| ~0.01-0.02 across 40 independent synthetic
    sessions -- small relative to the ~0.3-0.6 magnitude needed to
    explain the real observed 2.9-3.1px floor (see next point).
  - A synthetic zero-distortion-assumed solve (mirrors
    `capture_daemon._try_solve()`'s existing behavior) against ring20
    data generated WITH a realistic injected k1 reproduces a
    reprojection floor of the SAME MAGNITUDE as the real observed one:
    k1 around -0.55 to -0.6 is needed to reach ~2.9-3.1px at cam0's real
    geometry/focal length -- strong barrel distortion for a webcam-class
    lens, but well within plausible range for a wide-FOV UVC board
    camera, not an extreme/fisheye value. This is consistent with, not
    proof of, the real floor being explained by un-modeled distortion --
    stated as such, not overclaimed.

FITS ALONGSIDE, NOT INSTEAD OF, THE EXISTING FOCAL-LENGTH DERIVATION.
`opendarts.calibration.focal_length.derive_focal_length_from_ring20()`'s
pose-decoupled Zhang homography method remains the PRIMARY focal-length
source and is UNCHANGED by this module. This module additionally
attempts a joint (f, pose, k1) refit on the SAME ring20 correspondence
pool -- when it converges and passes the plausibility check below, its
own SELF-CONSISTENT (f, k1) pair together supersede the homography-only
f for that calibration event (using a biased f fixed from the
homography-only estimate while only k1 floats would let k1 silently
compensate for that bias rather than recovering a self-consistent
answer -- confirmed synthetically: the homography method's own f
estimate IS measurably biased in the presence of un-modeled distortion,
see that sweep's Step A). When this module's
own derivation does NOT converge or fails its plausibility check, this
event's calibration falls back to exactly today's existing behavior
(homography-only f, `dist_coeffs=zeros(5)`) -- purely additive, no
existing coverage removed.

STORAGE -- same JSON-not-code discipline as focal_length.py. `DISTORTION_FALLBACK_FILENAME` mirrors
`FOCAL_LENGTH_FALLBACK_FILENAME` exactly: a small, self-correcting
per-camera-INDEX JSON file living next to a rig's own calibration-event
packages, written only after a SUCCESSFUL live joint derivation, read as
a last-resort fallback (paired (focal_length_px, k1) together, never
mixed with a different-tier value for the other) when THIS event's own
live derivation doesn't converge. No third, hardcoded-constant tier --
same "no live path back to a stale hardcoded number" posture
`_resolve_focal_length_px()` already established; a camera this module
cannot resolve this event, with no persisted fallback either, simply
keeps `dist_coeffs=zeros(5)` (today's existing, safe default), never a
hardcoded distortion guess.

WIRING (see `opendarts.live.capture_daemon._try_solve()`'s own updated
call): `opendarts.pipeline.calibrate_camera()`'s own `dist_coeffs`
parameter already flows generically through `CameraCalibration` into
`opendarts.capture.throw_package.calibration_to_dict()`/
`calibration_from_dict()` and hence into
`current_calibration.json`/`derived_calibration.json` -- no NEW
persistence convention needed, this module only needs to stop always
handing that parameter a zero. `opendarts.triangulation.rays.py` already
calls `cv2.undistortPoints(pixel, camera_matrix, dist_coeffs)` --
already correctly consumes a non-zero `dist_coeffs`, unchanged by this
module (see this task's own final report for the explicit downstream
check).

PRINCIPAL POINT (cx only), added 2026-08-26.
Investigated principal point offset (cx, cy) and tangential distortion
(p1, p2) as further candidate unmodeled terms, same synthetic-first
discipline as k1/k2 above, same rig geometry, same real 4-package
validation:

  - NOISELESS: every one of (cx,cy) alone, (p1,p2) alone, and all four
    together recovers exactly (~1e-7-1e-11px/unit error) -- as expected,
    noiseless identifiability alone does not settle anything (k2 was
    also noiseless-exact and still correctly rejected -- see above).
  - REALISTIC noise (0.4px/point), MULTI-FRAME (N=40 median-averaged,
    matching CALIBRATION_N_FRAMES_DETECT), 40 independent synthetic
    sessions per camera: adding (cx,cy) JOINTLY inflates f's own std
    ~2.4-2.7x (0.6-0.68px -> 1.4-1.85px) versus the shipped k1-only
    model, even at zero true offset -- a real cost. Isolating each
    axis tells a DIFFERENT story per axis: cx ALONE leaves f's std
    UNCHANGED (0.595-0.678px, statistically identical to the k1-only
    baseline) and its own noise floor is small (std 1.06-1.29px). cy
    ALONE is measurably worse-conditioned on its own (f std still
    inflates ~2.4-2.7x, cy's own std 3.5-4.8px, ~3-4x worse than cx's) --
    a real, repeatable geometric ASYMMETRY for this rig (3 cameras
    spaced in azimuth around the ring gives rich cross-camera-style
    horizontal parallax information to a SINGLE view's own correspondence
    too, azimuth/yaw-adjacent; elevation/pitch information from one
    view is comparatively starved). p1,p2 land in between: f std
    inflates ~1.5-2x, their own noise floor (p1 std ~0.001, p2 std
    ~0.0002) is small in absolute terms but ambiguous relative to a
    genuinely plausible small true value.
  - The (cx,cy,p1,p2)-ALL-FREE model is the clearest reject: convergence
    rate collapses under realistic single-frame noise (31-39/60 trials,
    vs 60/60 for every simpler model) and, among converged trials, f
    std explodes 15-20x (47-63px) -- classic severe aliasing, the same
    signature that killed k2, just from a different combination.
  - REAL DATA (the decisive test, not the synthetic proxy): the SAME
    cx/cy asymmetry the multi-frame synthetic sweep predicted shows up
    even more starkly on the 4 real local calibration packages (all 3
    physical cameras, joint (f,pose,k1,cx,cy) fit on real ring20
    correspondence, N=30 detected frames each). Recovered cx is STABLE
    and REPEATABLE across independent real events for a given physical
    camera slot -- floor camera: -44.29/-47.01/-43.53/-43.49px (std
    1.44px, i.e. a real signal ~30x its own noise floor); the other two
    camera slots: +2.93/+4.18/+2.31/+1.06px (std ~1.1px) and
    -22.45/-9.57px-ish range (std ~13px, one event's own combined-model
    convergence failure narrowed this group to 3 clean points) --
    exactly the "stable + physically plausible across independent
    events" signature that validated k1 for the floor camera above.
    Recovered cy on the SAME real events is the opposite: wildly
    unstable for every camera slot (54.18 / 91.74 / 271.92 / 314.39 /
    362.85 / 365.49 / 374.28 / 384.27 / 391.60 / 419.65 / 432.81 /
    576.06px against a true center of 360px -- std 9.9-170px depending
    on grouping), an order of magnitude worse than even the pessimistic
    multi-frame synthetic prediction (3.5-4.8px) -- real per-frame
    landmark-detection noise along the vertical image axis evidently
    isn't well-modeled by this module's own i.i.d.-Gaussian synthetic
    noise assumption, or there is a further real effect aliasing into cy
    specifically that this investigation did not identify. Either way:
    not safely identifiable, on real data, full stop. p1/p2 on the same
    real events showed a partial, inconsistent signal (p2 fairly stable
    for 2 of 3 camera slots, p1 sign-inconsistent throughout, closer to
    cam2's own "no real fittable component" signature from the k1
    investigation above) -- genuinely inconclusive with only 4 real
    events, NOT shipped, and not chased further given the much cleaner
    cx result already in hand (same "generalization now, don't chase
    every anomaly" posture this project has used throughout).

  **Decision: ship cx alone** (`estimate_focal_pose_k1_cx()`,
  `derive_focal_k1_cx_from_oriented_results()` below) -- a joint
  (f, pose, k1, cx) solve, cy FIXED at the image geometric center
  (this project's pre-existing assumption, unchanged, per the evidence
  above). Real, measured reprojection-RMS improvement on top of the
  already-shipped k1-only model, same 4 packages: floor camera
  2.20/2.28/2.20/2.49px -> 1.98/1.96/1.97/2.35px; the other two camera
  slots also improve, by smaller but consistent margins (see this task's
  own final report for the complete before/after table, including the
  specific dart-43-session replay). `MAX_ABS_CX_OFFSET_FRACTION_OF_WIDTH`
  below is this addition's own plausibility gate, mirroring MIN_K1/MAX_K1's
  role exactly -- a fraction of image width (not a fixed px constant),
  so it scales sanely if this rig's resolution ever changes.
  `p1 = p2 = cy_offset = 0` remain fixed, exactly today's existing
  assumption -- this is a strict, evidence-narrowed SUBSET of the task's
  original "principal point (cx, cy) as one candidate, tangential (p1,
  p2) as another" framing, not the full 2-DOF-each extension, because the
  real data cleanly separates cx's identifiability from cy's -- forcing
  cy or p1/p2 in alongside it would be exactly the kind of "ship a term
  the numbers say isn't safely recoverable" move this module's own k2
  precedent already correctly declined to make.

  STORAGE: a NEW, separate fallback file,
  `PRINCIPAL_POINT_FALLBACK_FILENAME` (`principal_point_fallback.json`,
  schema `principal-point-fallback-v1`) -- deliberately NOT merged into
  the existing `distortion_fallback.json`/(f,k1) tier, to avoid ANY risk
  of pairing a fresh (f,k1) from one fit with a stale cx from a
  different one, or of this addition's own bugs regressing the
  already-shipped, already-tested (f,k1) persistence path. Same
  "self-correcting, written only after a successful live derivation,
  never a third hardcoded-constant tier" posture as every other fallback
  file in this project -- see WIRING section below and this module's own
  tests.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from opendarts.calibration.focal_length import (
    N_RING_LANDMARKS,
    average_ring20_image_points,
    ring20_image_points_from_result,
    ring20_object_points_mm,
)

log = logging.getLogger(__name__)

_RING20_OBJECT_POINTS_MM = ring20_object_points_mm()

# Plausibility bounds on the derived k1 -- a real physical sanity range,
# not a rig-specific tuning knob. This project's own established
# assumption (see tests/support/synthetic.py's own note) is a
# pinhole+Brown-Conrady model, "wide but
# not extreme fisheye" (~90deg HFOV) -- a genuine fisheye lens needs a
# structurally different cv2.fisheye.* API, out of scope here (see that
# module's own docstring). This rig's own real observed reprojection
# floor (2.9-3.1px on cam0) is consistent with k1 around -0.55 to -0.6
# at this rig's real geometry (this module's own synthetic conditioning
# analysis, module docstring above) -- MIN_K1=-1.0 sits with real
# measured margin below that without admitting an implausible
# fisheye-strength value. MIN_K1 is NOT tightened by the MAX_K1 fix
# below -- this rig shows real barrel distortion (negative k1) only;
# every genuinely healthy negative k1 measured to date (see below) is
# well inside -0.23, nowhere near -1.0.
MIN_K1 = -1.0

# MAX_K1 tightened 1.0 -> 0.10, 2026-09-03, asymmetrically (MIN_K1 left
# untouched -- see its own comment above). Real incident: calibration
# event (one recorded calibration package), cam2's joint (focal,
# pose, k1) solve landed on k1=+0.20487 -- a genuinely degenerate solve
# (cam2's own focal length that event, 705.1px, was 15.6% off the
# 3-camera median vs <2% on every other real event measured; an
# independent focal-consistency check on the SAME event agreed) that
# poisoned real live scoring that night (2 trebles pushed across the
# 107mm wire in a 120-throw session). The old MAX_K1=1.0 admitted it
# without complaint --
# comfortably wide enough that a real, live-scoring-affecting defect
# passed this plausibility gate silently.
#
# Re-derived from the real corpus, not copied from a prior estimate
# (this project's own "measure the real number" discipline) --
# an exploratory scan (not shipped) walked
# every real calibration event's
# `calibrations/derived_calibration.json` with distortion_source=="live"
# (k1-only tier) or principal_point_source=="live" (+cx tier -- BOTH
# tiers share this same MAX_K1 gate, see `derive_focal_and_k1_from_
# oriented_results()`/`derive_focal_k1_cx_from_oriented_results()`
# below -- confirmed the +cx tier's own real k1 distribution before
# reusing one shared bound for both, not assumed): 116 real live
# camera-events per tier, both rigs. Every genuinely healthy k1 stays
# negative or barely positive: k1-only tier's real positive ceiling is
# +0.03258 (cam1, one recorded calibration event); the +cx tier's own
# real positive ceiling is HIGHER, +0.05639 (cam0, calib_20260902-
# 234045-9a27e9cb) -- the +cx tier's own conditioning is genuinely
# different (see the "PRINCIPAL POINT (cx only)" module-docstring
# section above), so the higher of the two ceilings is what a SHARED
# bound must clear, not the k1-only tier's alone. The one broken value
# (+0.20487, reproduced near-identically in both tiers on the same
# event: cx tier's own cam2 that event is +0.20482) is the single
# largest positive k1 ever observed across all 232 real camera-events
# scanned (116 x 2 tiers) -- not a marginal outlier close to the real
# ceiling, a real order-of-magnitude jump.
#
# MAX_K1 = 0.10: 1.77x margin above the real +cx-tier ceiling
# (0.05639), 2.05x margin below the real broken value (0.20487) --
# comparable to this project's own other real-measured-ceiling
# thresholds (e.g. MIN_ILLUMINATION_MEAN_LUMINANCE_DELTA_GRAY's 1.96x
# margin above its own measured no-fault ceiling,
# `opendarts/capture/throw_trigger.py`). Deliberately NOT split into two
# tier-specific constants -- one shared, asymmetric MAX_K1 already
# covers both tiers' real ceilings with real margin, matching the project's
# own "3) ok 3" (simplest fix) choice over a more elaborate per-tier or
# cross-camera-consistency gate.
MAX_K1 = 0.10

# Plausibility bound on the derived principal-point-X offset (cx minus
# image_width/2) -- see module docstring's "PRINCIPAL POINT (cx only)"
# section. Real per-camera cx offsets measured on this rig range from
# roughly -47px to +4px (image_width=1280) -- a fraction of image width,
# not a fixed px constant, so this scales sanely if this rig's
# resolution ever changes. 0.15 (192px at 1280 width) sits with real
# measured margin on both sides of the observed range without admitting
# a degenerate/runaway solution.
MAX_ABS_CX_OFFSET_FRACTION_OF_WIDTH = 0.15

# Same FOV-derived multi-init sweep dev/calibration/intrinsics_estimation.py
# uses for the 4-point case -- kept here for robustness against a real-world
# geometry this module's own synthetic conditioning sweep did not cover, even
# though that sweep found every seed converges to the identical answer for
# this rig's real geometry (module docstring above) -- NOT relied on as a
# per-event confidence signal.
DEFAULT_FOV_DEG_CANDIDATES: tuple[float, ...] = (50.0, 65.0, 80.0, 90.0, 100.0, 115.0, 130.0)

# Same competitive-fit ratio dev/calibration/intrinsics_estimation.py
# uses (WORSE_LOCAL_OPTIMUM_RMS_RATIO/_ABS_FLOOR_PX) -- kept for
# consistency, not independently re-measured (this module's own
# conditioning sweep never observed a competing local optimum to filter
# in the first place, see module docstring).
WORSE_LOCAL_OPTIMUM_RMS_RATIO = 5.0
WORSE_LOCAL_OPTIMUM_RMS_ABS_FLOOR_PX = 0.05


@dataclass
class DistortionEstimationResult:
    ok: bool
    focal_px: float | None
    k1: float | None
    rvec: np.ndarray | None
    tvec: np.ndarray | None
    reprojection_rms_px: float | None
    initial_focal_px: float | None
    n_iterations: int
    reason: str = ""


def _project_with_jacobian(
    object_points_mm: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    f: float,
    cx: float,
    cy: float,
    k1: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project points with the restricted (fx=fy=f, fixed cx/cy, radial
    k1 only -- k2=p1=p2=k3=0, see module docstring's k2-not-identifiable
    finding) model and return (image_points (N,2), d(image_points)/d[
    rvec(3), tvec(3), f(1), k1(1)] as an (2N, 8) matrix), built from
    cv2's own analytic Jacobian -- same technique
    `dev/calibration/intrinsics_estimation.py`'s `_project_with_jacobian`
    uses for the zero-distortion 4-point case, extended with the k1
    column (cv2's own dist-coeff Jacobian column order is k1, k2, p1, p2,
    k3 -- column index 10 of the full (rvec,tvec,fx,fy,cx,cy,dist)
    Jacobian cv2.projectPoints returns)."""
    import cv2

    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    proj, J = cv2.projectPoints(
        object_points_mm.reshape(-1, 1, 3), rvec.reshape(3, 1), tvec.reshape(3, 1), K, dist,
    )
    proj = proj.reshape(-1, 2)
    J_rvec = J[:, 0:3]
    J_tvec = J[:, 3:6]
    J_f = (J[:, 6] + J[:, 7]).reshape(-1, 1)
    J_k1 = J[:, 10].reshape(-1, 1)
    J_params = np.hstack([J_rvec, J_tvec, J_f, J_k1])
    return proj, J_params


def _levenberg_marquardt(
    object_points_mm: np.ndarray,
    image_points_px: np.ndarray,
    params0: np.ndarray, # [rvec(3), tvec(3), f(1), k1(1)]
    cx: float,
    cy: float,
    max_iter: int = 200,
    tol: float = 1e-12,
) -> tuple[np.ndarray, int, bool, str]:
    """Standard damped Levenberg-Marquardt on reprojection-error
    residuals -- identical structure to
    `dev/calibration/intrinsics_estimation.py`'s `_levenberg_marquardt`
    (same COST_FLOOR/damping-saturation/stuck-but-tiny-residual handling,
    all independently re-validated for THIS 8-unknown/20-point problem
    by this module's own synthetic conditioning sweep, not just copied on
    faith), extended to carry k1 as an 8th free parameter."""
    params = params0.copy()
    lam = 1e-3

    def residuals(p: np.ndarray) -> np.ndarray:
        proj, _ = _project_with_jacobian(object_points_mm, p[0:3], p[3:6], p[6], cx, cy, p[7])
        return (proj - image_points_px).reshape(-1)

    r = residuals(params)
    cost = 0.5 * float(np.dot(r, r))
    if not np.isfinite(cost):
        return params, 0, False, "non-finite cost at initial guess"

    COST_FLOOR = 1e-16 # see dev/calibration/intrinsics_estimation.py's own comment for why

    for it in range(max_iter):
        if cost < COST_FLOOR:
            return params, it, True, "converged (residual at numerical floor)"
        if params[6] <= 1.0:
            return params, it, False, "focal length collapsed to <=1px during optimization"
        _, J = _project_with_jacobian(
            object_points_mm, params[0:3], params[3:6], params[6], cx, cy, params[7]
        )
        if not np.all(np.isfinite(J)):
            return params, it, False, "non-finite Jacobian (diverging step)"
        # An extreme FOV seed (this module's own DEFAULT_FOV_DEG_CANDIDATES
        # spans 50-130deg) can occasionally overshoot into a huge/ill-scaled
        # J before the params[6]<=1.0 guard catches it on the NEXT
        # iteration -- suppress the resulting (harmless, already handled by
        # the isfinite checks immediately below/above) numpy overflow
        # warnings rather than letting them spam real production logs on
        # every calibration event.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            JTJ = J.T @ J
            JTr = J.T @ r
        diag = np.diag(JTJ)
        if not np.all(np.isfinite(diag)):
            return params, it, False, "non-finite Jacobian"

        improved = False
        for _ in range(30):
            A = JTJ + lam * np.diag(diag + 1e-12)
            try:
                delta = np.linalg.solve(A, -JTr)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            new_params = params + delta
            if new_params[6] <= 1.0 or not np.all(np.isfinite(new_params)):
                lam *= 10.0
                continue
            new_r = residuals(new_params)
            new_cost = 0.5 * float(np.dot(new_r, new_r))
            if np.isfinite(new_cost) and new_cost < cost:
                params = new_params
                r = new_r
                if cost - new_cost < tol * max(cost, 1e-12):
                    cost = new_cost
                    return params, it + 1, True, "converged"
                cost = new_cost
                lam = max(lam * 0.5, 1e-12)
                improved = True
                break
            lam *= 10.0
        if not improved:
            if cost < 1e-8:
                return params, it, True, "converged (no further improving step, residual already tiny)"
            return params, it, False, "step rejected repeatedly (lambda saturated)"
        if lam > 1e12:
            return params, it + 1, False, "damping saturated"

    return params, max_iter, False, "max iterations reached without convergence"


def estimate_focal_pose_k1(
    object_points_mm: np.ndarray,
    image_points_px: np.ndarray,
    image_width: float,
    image_height: float,
    initial_focal_px: float,
    initial_k1: float = 0.0,
) -> DistortionEstimationResult:
    """Jointly estimate focal length `f`, radial distortion `k1`
    (k2=p1=p2=k3=0, see module docstring), and pose (rvec, tvec) from
    known 2D<->3D correspondences -- intended for the 20-point ring
    (`opendarts.calibration.focal_length.ring20_object_points_mm()`), NOT
    the 4-point cardinal quad (see module docstring for why that distinction
    matters -- this is exactly the previously-rejected joint solve,
    finally safe because of the richer correspondence, not because the
    math itself changed).

    Seeds pose via `cv2.solvePnPGeneric(..., SOLVEPNP_IPPE)` when exactly
    4 points are given (mirrors
    `dev/calibration/intrinsics_estimation.py`'s own coplanar-ambiguity
    handling) or plain `cv2.solvePnP` otherwise (the expected real case:
    20 points, no coplanar ambiguity to speak of)."""
    import cv2

    object_points_mm = np.asarray(object_points_mm, dtype=np.float64).reshape(-1, 3)
    image_points_px = np.asarray(image_points_px, dtype=np.float64).reshape(-1, 2)
    n = len(object_points_mm)
    if n < 4:
        return DistortionEstimationResult(
            ok=False, focal_px=None, k1=None, rvec=None, tvec=None,
            reprojection_rms_px=None, initial_focal_px=initial_focal_px,
            n_iterations=0, reason=f"need >=4 point correspondences, got {n}",
        )

    cx = image_width / 2.0
    cy = image_height / 2.0

    K0 = np.array([[initial_focal_px, 0, cx], [0, initial_focal_px, cy], [0, 0, 1]], dtype=np.float64)
    obj = object_points_mm.reshape(-1, 1, 3)
    img = image_points_px.reshape(-1, 1, 2)
    dist0 = np.zeros(5)

    pose_seeds: list[tuple[np.ndarray, np.ndarray]] = []
    if n == 4:
        try:
            n_sol, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, K0, dist0, flags=cv2.SOLVEPNP_IPPE)
            if n_sol > 0:
                pose_seeds = [
                    (np.asarray(rvecs[i]).reshape(3, 1), np.asarray(tvecs[i]).reshape(3, 1))
                    for i in range(n_sol)
                ]
        except cv2.error:
            pass
    if not pose_seeds:
        ok, rvec0, tvec0 = cv2.solvePnP(obj, img, K0, dist0)
        if not ok:
            return DistortionEstimationResult(
                ok=False, focal_px=None, k1=None, rvec=None, tvec=None,
                reprojection_rms_px=None, initial_focal_px=initial_focal_px,
                n_iterations=0, reason="initial PnP seed failed to converge",
            )
        pose_seeds = [(rvec0.reshape(3, 1), tvec0.reshape(3, 1))]

    best: DistortionEstimationResult | None = None
    for rvec0, tvec0 in pose_seeds:
        params0 = np.concatenate(
            [rvec0.reshape(3), tvec0.reshape(3), [float(initial_focal_px), float(initial_k1)]]
        )
        params, n_iter, converged, reason = _levenberg_marquardt(
            object_points_mm, image_points_px, params0, cx, cy
        )
        if not converged:
            candidate = DistortionEstimationResult(
                ok=False, focal_px=None, k1=None, rvec=None, tvec=None,
                reprojection_rms_px=None, initial_focal_px=initial_focal_px,
                n_iterations=n_iter, reason=reason,
            )
        else:
            proj, _ = _project_with_jacobian(
                object_points_mm, params[0:3], params[3:6], params[6], cx, cy, params[7]
            )
            rms = float(np.sqrt(np.mean(np.sum((proj - image_points_px) ** 2, axis=1))))
            candidate = DistortionEstimationResult(
                ok=True, focal_px=float(params[6]), k1=float(params[7]),
                rvec=params[0:3].reshape(3, 1), tvec=params[3:6].reshape(3, 1),
                reprojection_rms_px=rms, initial_focal_px=initial_focal_px,
                n_iterations=n_iter, reason=reason,
            )
        if best is None:
            best = candidate
        elif candidate.ok and (not best.ok or candidate.reprojection_rms_px < best.reprojection_rms_px):
            best = candidate

    assert best is not None
    return best


@dataclass(frozen=True)
class DistortionDerivationResult:
    ok: bool
    focal_length_px: float | None
    k1: float | None
    reason: str
    n_points_used: int
    n_frames_used: int
    reprojection_rms_px: float | None = None


def derive_focal_and_k1_from_oriented_results(
    results: list[Any],
    principal_point: tuple[float, float],
    image_width: float,
    image_height: float,
    *,
    min_frames: int = 1,
    initial_focal_px: float | None = None,
) -> DistortionDerivationResult:
    """Full pipeline: filter `results` (a camera's accumulated
    `OrientedLandmarkResult` list, same input
    `opendarts.calibration.focal_length.derive_focal_length_from_oriented_
    results()` already takes) down to usable ring20 correspondences,
    average them across frames (per-index median, same
    `average_ring20_image_points()` focal_length.py uses -- averaging the
    noisy INPUT once, not solving once per frame then averaging, same
    "average then solve" discipline established across this project's
    calibration derivations), then run the joint (f, pose, k1) multi-init
    solve ONCE on the averaged correspondence.

    `initial_focal_px`: the caller's own already-resolved focal-length
    guess (typically `opendarts.calibration.focal_length`'s own homography-
    derived value for this same event) -- used as ONE extra multi-init
    seed alongside the fixed FOV-derived candidates, since it is usually
    already close to the true answer. If None, only the fixed FOV
    candidates are tried.

    Plausibility gate: a converged fit with `k1` outside
    `[MIN_K1, MAX_K1]` is rejected as a non-physical/degenerate solution,
    never clamped or silently accepted -- same posture
    `derive_focal_length_from_ring20()` already uses for its own
    focal-length plausibility bounds.
    """
    per_frame = [
        pts for pts in (ring20_image_points_from_result(r) for r in results) if pts is not None
    ]
    n_frames = len(per_frame)
    if n_frames < max(1, min_frames):
        return DistortionDerivationResult(
            ok=False, focal_length_px=None, k1=None,
            reason=f"only {n_frames} usable frame(s) for ring20 correspondence, need >= {min_frames}",
            n_points_used=0, n_frames_used=n_frames,
        )
    averaged = average_ring20_image_points(per_frame)

    candidates = list(DEFAULT_FOV_DEG_CANDIDATES)
    seed_results = []
    for fov_deg in candidates:
        f0 = (image_width / 2.0) / math.tan(math.radians(fov_deg / 2.0))
        seed_results.append(
            estimate_focal_pose_k1(_RING20_OBJECT_POINTS_MM, averaged, image_width, image_height, f0)
        )
    if initial_focal_px is not None and initial_focal_px > 0:
        seed_results.append(
            estimate_focal_pose_k1(_RING20_OBJECT_POINTS_MM, averaged, image_width, image_height, float(initial_focal_px))
        )

    converged = [r for r in seed_results if r.ok]
    if not converged:
        reasons = {r.reason for r in seed_results if not r.ok}
        return DistortionDerivationResult(
            ok=False, focal_length_px=None, k1=None,
            reason=f"no multi-init seed converged: {sorted(reasons)}",
            n_points_used=N_RING_LANDMARKS, n_frames_used=n_frames,
        )
    best = min(converged, key=lambda r: r.reprojection_rms_px)

    if not (MIN_K1 <= best.k1 <= MAX_K1):
        return DistortionDerivationResult(
            ok=False, focal_length_px=None, k1=None,
            reason=f"derived k1={best.k1!r} outside plausible [{MIN_K1}, {MAX_K1}] range",
            n_points_used=N_RING_LANDMARKS, n_frames_used=n_frames,
        )
    if best.focal_px is None or not (1.0 < best.focal_px < 1e6):
        return DistortionDerivationResult(
            ok=False, focal_length_px=None, k1=None,
            reason=f"derived focal length {best.focal_px!r} non-physical",
            n_points_used=N_RING_LANDMARKS, n_frames_used=n_frames,
        )

    return DistortionDerivationResult(
        ok=True, focal_length_px=best.focal_px, k1=best.k1, reason="ok",
        n_points_used=N_RING_LANDMARKS, n_frames_used=n_frames,
        reprojection_rms_px=best.reprojection_rms_px,
    )


# ---------------------------------------------------------------------
# JSON persistence -- mirrors opendarts.calibration.focal_length's
# FOCAL_LENGTH_FALLBACK_FILENAME exactly (schema, merge-update-preserve-
# other-cameras, read-degrades-safely-to-empty). See module docstring's
# "STORAGE" section.
# ---------------------------------------------------------------------

DISTORTION_FALLBACK_FILENAME = "distortion_fallback.json"
SCHEMA = "distortion-fallback-v1"


def load_distortion_fallback(calibration_package_root: Path) -> dict[int, dict]:
    """Load the persisted last-known-good per-camera-INDEX (focal_length_px,
    k1) pair from `<calibration_package_root>/distortion_fallback.json`.
    Returns `{}` on missing/corrupt/wrong-schema, same safe-degrade
    posture `load_focal_length_fallback()` already established."""
    path = Path(calibration_package_root) / DISTORTION_FALLBACK_FILENAME
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        log.exception("%s: failed to read/parse -- treating as absent", path)
        return {}
    if payload.get("schema") != SCHEMA:
        log.warning("%s: schema %r does not match current %r -- treating as absent",
                    path, payload.get("schema"), SCHEMA)
        return {}
    cameras = payload.get("cameras")
    if not isinstance(cameras, dict):
        return {}
    out: dict[int, dict] = {}
    for key, record in cameras.items():
        try:
            cam = int(key)
        except (TypeError, ValueError):
            continue
        if (
            isinstance(record, dict)
            and isinstance(record.get("focal_length_px"), (int, float))
            and isinstance(record.get("k1"), (int, float))
        ):
            out[cam] = record
    return out


def write_distortion_fallback_entry(
    calibration_package_root: Path,
    cam: int,
    focal_length_px: float,
    k1: float,
    *,
    n_frames_used: int,
    n_points_used: int,
    derived_at_utc: str,
    package_id: str | None = None,
) -> None:
    """Merge-update ONE camera's (focal_length_px, k1) entry -- see
    `opendarts.calibration.focal_length.write_focal_length_fallback_entry()`
    for the identical read-modify-write/preserve-other-cameras design
    this mirrors. Called only after a SUCCESSFUL live joint derivation."""
    calibration_package_root = Path(calibration_package_root)
    calibration_package_root.mkdir(parents=True, exist_ok=True)
    path = calibration_package_root / DISTORTION_FALLBACK_FILENAME
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text())
            if isinstance(loaded, dict) and loaded.get("schema") == SCHEMA:
                existing = loaded
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            log.exception("%s: failed to read existing fallback table before updating -- "
                          "starting a fresh one", path)
    cameras = existing.get("cameras")
    if not isinstance(cameras, dict):
        cameras = {}
    cameras[str(cam)] = {
        "focal_length_px": float(focal_length_px),
        "k1": float(k1),
        "derived_at_utc": derived_at_utc,
        "n_frames_used": int(n_frames_used),
        "n_points_used": int(n_points_used),
        "package_id": package_id,
    }
    payload = {"schema": SCHEMA, "updated_at_utc": derived_at_utc, "cameras": cameras}
    path.write_text(json.dumps(payload, indent=2))


# =======================================================================
# PRINCIPAL POINT (cx only) -- see module docstring's own "PRINCIPAL
# POINT (cx only)" section for the full synthetic + real-data
# conditioning analysis that justifies solving for cx alone (cy, p1, p2
# all stay fixed at their existing values). Deliberately a SEPARATE set
# of functions from the (f, pose, k1) ones above, not a parameterized
# generalization of them -- mirrors this module's own relationship to
# `opendarts.calibration.focal_length` (a new, independently-tested layer
# fit ALONGSIDE a proven one, never edited in place) so this addition
# carries zero risk of regressing the already-shipped k1-only path.
# =======================================================================


def _project_with_jacobian_cx(
    object_points_mm: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    f: float,
    cx: float,
    cy: float,
    k1: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Same restricted model as `_project_with_jacobian()` (fx=fy=f, k2=
    p1=p2=k3=0), extended with a `cx` Jacobian column (cv2's own dist/
    intrinsics Jacobian column order is rvec(3),tvec(3),fx,fy,cx,cy,dist
    (5) -- cx is column index 8) so `cx` can be solved for as a 9th free
    unknown -- `cy` stays a FIXED input, never a free parameter (see
    module docstring for why)."""
    import cv2

    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    proj, J = cv2.projectPoints(
        object_points_mm.reshape(-1, 1, 3), rvec.reshape(3, 1), tvec.reshape(3, 1), K, dist,
    )
    proj = proj.reshape(-1, 2)
    J_rvec = J[:, 0:3]
    J_tvec = J[:, 3:6]
    J_f = (J[:, 6] + J[:, 7]).reshape(-1, 1)
    J_cx = J[:, 8].reshape(-1, 1)
    J_k1 = J[:, 10].reshape(-1, 1)
    J_params = np.hstack([J_rvec, J_tvec, J_f, J_k1, J_cx])
    return proj, J_params


def _levenberg_marquardt_cx(
    object_points_mm: np.ndarray,
    image_points_px: np.ndarray,
    params0: np.ndarray, # [rvec(3), tvec(3), f(1), k1(1), cx(1)]
    cy: float,
    max_iter: int = 200,
    tol: float = 1e-12,
) -> tuple[np.ndarray, int, bool, str]:
    """Same damped Levenberg-Marquardt structure as `_levenberg_marquardt()`
    above, extended to carry `cx` as a 9th free parameter (`cy` fixed)."""
    params = params0.copy()
    lam = 1e-3

    def residuals(p: np.ndarray) -> np.ndarray:
        proj, _ = _project_with_jacobian_cx(
            object_points_mm, p[0:3], p[3:6], p[6], p[8], cy, p[7]
        )
        return (proj - image_points_px).reshape(-1)

    r = residuals(params)
    cost = 0.5 * float(np.dot(r, r))
    if not np.isfinite(cost):
        return params, 0, False, "non-finite cost at initial guess"

    COST_FLOOR = 1e-16

    for it in range(max_iter):
        if cost < COST_FLOOR:
            return params, it, True, "converged (residual at numerical floor)"
        if params[6] <= 1.0:
            return params, it, False, "focal length collapsed to <=1px during optimization"
        _, J = _project_with_jacobian_cx(
            object_points_mm, params[0:3], params[3:6], params[6], params[8], cy, params[7]
        )
        if not np.all(np.isfinite(J)):
            return params, it, False, "non-finite Jacobian (diverging step)"
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            JTJ = J.T @ J
            JTr = J.T @ r
        diag = np.diag(JTJ)
        if not np.all(np.isfinite(diag)):
            return params, it, False, "non-finite Jacobian"

        improved = False
        for _ in range(30):
            A = JTJ + lam * np.diag(diag + 1e-12)
            try:
                delta = np.linalg.solve(A, -JTr)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            new_params = params + delta
            if new_params[6] <= 1.0 or not np.all(np.isfinite(new_params)):
                lam *= 10.0
                continue
            new_r = residuals(new_params)
            new_cost = 0.5 * float(np.dot(new_r, new_r))
            if np.isfinite(new_cost) and new_cost < cost:
                params = new_params
                r = new_r
                if cost - new_cost < tol * max(cost, 1e-12):
                    cost = new_cost
                    return params, it + 1, True, "converged"
                cost = new_cost
                lam = max(lam * 0.5, 1e-12)
                improved = True
                break
            lam *= 10.0
        if not improved:
            if cost < 1e-8:
                return params, it, True, "converged (no further improving step, residual already tiny)"
            return params, it, False, "step rejected repeatedly (lambda saturated)"
        if lam > 1e12:
            return params, it + 1, False, "damping saturated"

    return params, max_iter, False, "max iterations reached without convergence"


@dataclass
class DistortionCxEstimationResult:
    ok: bool
    focal_px: float | None
    k1: float | None
    cx: float | None
    rvec: np.ndarray | None
    tvec: np.ndarray | None
    reprojection_rms_px: float | None
    initial_focal_px: float | None
    n_iterations: int
    reason: str = ""


def estimate_focal_pose_k1_cx(
    object_points_mm: np.ndarray,
    image_points_px: np.ndarray,
    image_width: float,
    image_height: float,
    initial_focal_px: float,
    initial_k1: float = 0.0,
    initial_cx: float | None = None,
) -> DistortionCxEstimationResult:
    """Jointly estimate focal length `f`, radial distortion `k1`, pose
    (rvec, tvec), AND the principal-point-X offset `cx` (`cy` FIXED at
    `image_height/2` -- see module docstring's "PRINCIPAL POINT (cx
    only)" section for why cx alone, not cy/p1/p2 too) from known 2D<->3D
    correspondences -- same 20-point-ring intent as `estimate_focal_pose_k1()`,
    9 unknowns instead of 8."""
    import cv2

    object_points_mm = np.asarray(object_points_mm, dtype=np.float64).reshape(-1, 3)
    image_points_px = np.asarray(image_points_px, dtype=np.float64).reshape(-1, 2)
    n = len(object_points_mm)
    if n < 4:
        return DistortionCxEstimationResult(
            ok=False, focal_px=None, k1=None, cx=None, rvec=None, tvec=None,
            reprojection_rms_px=None, initial_focal_px=initial_focal_px,
            n_iterations=0, reason=f"need >=4 point correspondences, got {n}",
        )

    cx0 = image_width / 2.0 if initial_cx is None else float(initial_cx)
    cy = image_height / 2.0

    K0 = np.array([[initial_focal_px, 0, cx0], [0, initial_focal_px, cy], [0, 0, 1]], dtype=np.float64)
    obj = object_points_mm.reshape(-1, 1, 3)
    img = image_points_px.reshape(-1, 1, 2)
    dist0 = np.zeros(5)

    pose_seeds: list[tuple[np.ndarray, np.ndarray]] = []
    if n == 4:
        try:
            n_sol, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, K0, dist0, flags=cv2.SOLVEPNP_IPPE)
            if n_sol > 0:
                pose_seeds = [
                    (np.asarray(rvecs[i]).reshape(3, 1), np.asarray(tvecs[i]).reshape(3, 1))
                    for i in range(n_sol)
                ]
        except cv2.error:
            pass
    if not pose_seeds:
        ok, rvec0, tvec0 = cv2.solvePnP(obj, img, K0, dist0)
        if not ok:
            return DistortionCxEstimationResult(
                ok=False, focal_px=None, k1=None, cx=None, rvec=None, tvec=None,
                reprojection_rms_px=None, initial_focal_px=initial_focal_px,
                n_iterations=0, reason="initial PnP seed failed to converge",
            )
        pose_seeds = [(rvec0.reshape(3, 1), tvec0.reshape(3, 1))]

    best: DistortionCxEstimationResult | None = None
    for rvec0, tvec0 in pose_seeds:
        params0 = np.concatenate(
            [rvec0.reshape(3), tvec0.reshape(3), [float(initial_focal_px), float(initial_k1), cx0]]
        )
        params, n_iter, converged, reason = _levenberg_marquardt_cx(
            object_points_mm, image_points_px, params0, cy
        )
        if not converged:
            candidate = DistortionCxEstimationResult(
                ok=False, focal_px=None, k1=None, cx=None, rvec=None, tvec=None,
                reprojection_rms_px=None, initial_focal_px=initial_focal_px,
                n_iterations=n_iter, reason=reason,
            )
        else:
            proj, _ = _project_with_jacobian_cx(
                object_points_mm, params[0:3], params[3:6], params[6], params[8], cy, params[7]
            )
            rms = float(np.sqrt(np.mean(np.sum((proj - image_points_px) ** 2, axis=1))))
            candidate = DistortionCxEstimationResult(
                ok=True, focal_px=float(params[6]), k1=float(params[7]), cx=float(params[8]),
                rvec=params[0:3].reshape(3, 1), tvec=params[3:6].reshape(3, 1),
                reprojection_rms_px=rms, initial_focal_px=initial_focal_px,
                n_iterations=n_iter, reason=reason,
            )
        if best is None:
            best = candidate
        elif candidate.ok and (not best.ok or candidate.reprojection_rms_px < best.reprojection_rms_px):
            best = candidate

    assert best is not None
    return best


@dataclass(frozen=True)
class DistortionCxDerivationResult:
    ok: bool
    focal_length_px: float | None
    k1: float | None
    cx: float | None
    reason: str
    n_points_used: int
    n_frames_used: int
    reprojection_rms_px: float | None = None


def derive_focal_k1_cx_from_oriented_results(
    results: list[Any],
    principal_point: tuple[float, float],
    image_width: float,
    image_height: float,
    *,
    min_frames: int = 1,
    initial_focal_px: float | None = None,
) -> DistortionCxDerivationResult:
    """Full pipeline, mirrors `derive_focal_and_k1_from_oriented_results()`
    exactly (filter -> per-index median average -> multi-init solve once
    on the averaged correspondence), but for the (f, pose, k1, cx) 9-
    unknown joint solve -- see module docstring's "PRINCIPAL POINT (cx
    only)" section for the conditioning analysis behind this.

    Plausibility gates: `k1` outside `[MIN_K1, MAX_K1]` (same as the
    k1-only path) OR the cx OFFSET (`cx - image_width/2`) outside
    `+-MAX_ABS_CX_OFFSET_FRACTION_OF_WIDTH * image_width` rejects the fit
    as non-physical/degenerate, never clamped or silently accepted.
    """
    per_frame = [
        pts for pts in (ring20_image_points_from_result(r) for r in results) if pts is not None
    ]
    n_frames = len(per_frame)
    if n_frames < max(1, min_frames):
        return DistortionCxDerivationResult(
            ok=False, focal_length_px=None, k1=None, cx=None,
            reason=f"only {n_frames} usable frame(s) for ring20 correspondence, need >= {min_frames}",
            n_points_used=0, n_frames_used=n_frames,
        )
    averaged = average_ring20_image_points(per_frame)
    cx_nominal = image_width / 2.0

    candidates = list(DEFAULT_FOV_DEG_CANDIDATES)
    seed_results = []
    for fov_deg in candidates:
        f0 = (image_width / 2.0) / math.tan(math.radians(fov_deg / 2.0))
        seed_results.append(
            estimate_focal_pose_k1_cx(_RING20_OBJECT_POINTS_MM, averaged, image_width, image_height, f0)
        )
    if initial_focal_px is not None and initial_focal_px > 0:
        seed_results.append(
            estimate_focal_pose_k1_cx(
                _RING20_OBJECT_POINTS_MM, averaged, image_width, image_height, float(initial_focal_px)
            )
        )

    converged = [r for r in seed_results if r.ok]
    if not converged:
        reasons = {r.reason for r in seed_results if not r.ok}
        return DistortionCxDerivationResult(
            ok=False, focal_length_px=None, k1=None, cx=None,
            reason=f"no multi-init seed converged: {sorted(reasons)}",
            n_points_used=N_RING_LANDMARKS, n_frames_used=n_frames,
        )
    best = min(converged, key=lambda r: r.reprojection_rms_px)

    if not (MIN_K1 <= best.k1 <= MAX_K1):
        return DistortionCxDerivationResult(
            ok=False, focal_length_px=None, k1=None, cx=None,
            reason=f"derived k1={best.k1!r} outside plausible [{MIN_K1}, {MAX_K1}] range",
            n_points_used=N_RING_LANDMARKS, n_frames_used=n_frames,
        )
    if best.focal_px is None or not (1.0 < best.focal_px < 1e6):
        return DistortionCxDerivationResult(
            ok=False, focal_length_px=None, k1=None, cx=None,
            reason=f"derived focal length {best.focal_px!r} non-physical",
            n_points_used=N_RING_LANDMARKS, n_frames_used=n_frames,
        )
    max_offset = MAX_ABS_CX_OFFSET_FRACTION_OF_WIDTH * image_width
    cx_offset = best.cx - cx_nominal
    if abs(cx_offset) > max_offset:
        return DistortionCxDerivationResult(
            ok=False, focal_length_px=None, k1=None, cx=None,
            reason=f"derived cx offset {cx_offset:+.1f}px outside plausible "
                   f"+-{max_offset:.1f}px range",
            n_points_used=N_RING_LANDMARKS, n_frames_used=n_frames,
        )

    return DistortionCxDerivationResult(
        ok=True, focal_length_px=best.focal_px, k1=best.k1, cx=best.cx, reason="ok",
        n_points_used=N_RING_LANDMARKS, n_frames_used=n_frames,
        reprojection_rms_px=best.reprojection_rms_px,
    )


# ---------------------------------------------------------------------
# JSON persistence for the cx addition -- a SEPARATE fallback file from
# DISTORTION_FALLBACK_FILENAME (see module docstring's "STORAGE"
# paragraph in the "PRINCIPAL POINT (cx only)" section for why: zero
# risk of pairing a fresh (f,k1) with a stale cx or of regressing the
# already-shipped (f,k1) persistence path). Same schema/merge-update/
# safe-degrade shape as DISTORTION_FALLBACK_FILENAME itself.
# ---------------------------------------------------------------------

PRINCIPAL_POINT_FALLBACK_FILENAME = "principal_point_fallback.json"
PRINCIPAL_POINT_SCHEMA = "principal-point-fallback-v1"


def load_principal_point_fallback(calibration_package_root: Path) -> dict[int, dict]:
    """Load the persisted last-known-good per-camera-INDEX (focal_length_px,
    k1, cx) triple from `<calibration_package_root>/principal_point_fallback.json`.
    Returns `{}` on missing/corrupt/wrong-schema, same safe-degrade
    posture `load_distortion_fallback()` already established."""
    path = Path(calibration_package_root) / PRINCIPAL_POINT_FALLBACK_FILENAME
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        log.exception("%s: failed to read/parse -- treating as absent", path)
        return {}
    if payload.get("schema") != PRINCIPAL_POINT_SCHEMA:
        log.warning("%s: schema %r does not match current %r -- treating as absent",
                    path, payload.get("schema"), PRINCIPAL_POINT_SCHEMA)
        return {}
    cameras = payload.get("cameras")
    if not isinstance(cameras, dict):
        return {}
    out: dict[int, dict] = {}
    for key, record in cameras.items():
        try:
            cam = int(key)
        except (TypeError, ValueError):
            continue
        if (
            isinstance(record, dict)
            and isinstance(record.get("focal_length_px"), (int, float))
            and isinstance(record.get("k1"), (int, float))
            and isinstance(record.get("cx"), (int, float))
        ):
            out[cam] = record
    return out


def write_principal_point_fallback_entry(
    calibration_package_root: Path,
    cam: int,
    focal_length_px: float,
    k1: float,
    cx: float,
    *,
    n_frames_used: int,
    n_points_used: int,
    derived_at_utc: str,
    package_id: str | None = None,
) -> None:
    """Merge-update ONE camera's (focal_length_px, k1, cx) entry -- see
    `write_distortion_fallback_entry()` for the identical read-modify-
    write/preserve-other-cameras design this mirrors. Called only after a
    SUCCESSFUL live joint (f, pose, k1, cx) derivation."""
    calibration_package_root = Path(calibration_package_root)
    calibration_package_root.mkdir(parents=True, exist_ok=True)
    path = calibration_package_root / PRINCIPAL_POINT_FALLBACK_FILENAME
    existing: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text())
            if isinstance(loaded, dict) and loaded.get("schema") == PRINCIPAL_POINT_SCHEMA:
                existing = loaded
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            log.exception("%s: failed to read existing fallback table before updating -- "
                          "starting a fresh one", path)
    cameras = existing.get("cameras")
    if not isinstance(cameras, dict):
        cameras = {}
    cameras[str(cam)] = {
        "focal_length_px": float(focal_length_px),
        "k1": float(k1),
        "cx": float(cx),
        "derived_at_utc": derived_at_utc,
        "n_frames_used": int(n_frames_used),
        "n_points_used": int(n_points_used),
        "package_id": package_id,
    }
    payload = {"schema": PRINCIPAL_POINT_SCHEMA, "updated_at_utc": derived_at_utc, "cameras": cameras}
    path.write_text(json.dumps(payload, indent=2))
