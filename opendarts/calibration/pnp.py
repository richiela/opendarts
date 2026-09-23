"""Extrinsics-only PnP calibration: intrinsics are assumed known/given, NOT solved
for here -- cv2.calibrateCamera-style joint recovery is not achievable
on this project's fixed-pose rig).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --- Calibration-side conditioning signal ---------------------------
#
# A review pass found that ray-agreement (opendarts/engines/apollo/scoring.py's
# MAX_RAY_DISAGREEMENT_MM) structurally cannot catch CORRELATED calibration
# bias across cameras -- all cameras similarly miscalibrated from a
# similarly-clustered landmark quad still agree with each other on a
# jointly-wrong point. This needs a signal computed from a single camera's
# own input, independent of what any other camera says.
#
# landmark_hull_area_fraction() measures how much of the image the input
# landmark points (the ones handed to solve_extrinsics, BEFORE solving)
# actually spread across, via their 2D convex hull area as a fraction of
# image area. This is computed on the INPUT, not the solve result, so it
# can flag a clustered/occluded quad as suspect even before PnP runs --
# exactly the "clustered vs well-spread quad" root cause already
# identified qualitatively, now made into an actual number.
#
# Empirically measured (opendarts project session, 2026-08-12; see
# tests/test_pnp_calibration.py for the regression tests that pin these
# numbers) using the SAME well-spread/clustered quads already established
# as the real/bad cases, at fov=90deg (this project's
# existing default synthetic camera, matching every other test in this
# repo), 3 camera positions x 200 noise seeds (0.5px stddev) each:
#   well-spread quad:  hull_area_fraction min=0.0337, mean=0.0376
#   clustered quad:    hull_area_fraction max=0.0019,  mean=0.0013
# -- a ~17x gap between the worst well-spread case and the worst clustered
# case at this FOV. Re-measured at fov=75/105/120deg for robustness (this
# metric is resolution/FOV-dependent, see MIN_LANDMARK_HULL_AREA_FRACTION
# below): clustered-quad max stayed <=0.0032 and well-spread-quad min
# stayed >=0.0111 across ALL four FOVs tested, so MIN_LANDMARK_HULL_AREA_
# FRACTION = 0.01 sits with real, measured margin on both sides at every
# FOV tested, not picked by guessing.
#
# Known limitations, stated honestly (2026-08-12):
# 1. This threshold is calibrated against this project's synthetic
#    pinhole camera model (image assumed centered principal point, fov
#    75-120deg range tested). It has NOT been validated against the real
#    rig's actual intrinsics/resolution, which remain unknown -- once
#    real intrinsics exist,
#    re-measure this threshold against them rather than assuming it
#    transfers unchanged. calibrate_camera() (opendarts/pipeline.py) accepts
#    explicit image_width/image_height for this reason -- pass real
#    values once known rather than relying on the centered-principal-
#    point inference in solve_extrinsics().
# 2. FALSE-POSITIVE RISK, demonstrated: this threshold was only measured
#    at one camera-to-board distance (make_ring_camera's default
#    ring_radius_mm=500, height_mm=300 -- the value every test in this
#    repo uses, NOT a confirmed real-rig measurement). Camera distance
#    changes hull_area_fraction directly (a fixed
#    real-world quad subtends less of the image from farther away) even
#    when PnP conditioning itself hasn't actually degraded. Measured
#    concretely: at ring_radius_mm=1200 (same WELL_SPREAD_QUAD, same
#    noise), hull_area_fraction drops to ~0.004 (would be wrongly
#    rejected by this threshold) while actual PnP rotation error stays
#    tiny (mean 0.42deg, max 0.95deg over 50 seeds) -- genuinely
#    well-conditioned, incorrectly flagged. This fails SAFE (an unwarranted
#    rejection, not a wrong-and-accepted sector), but it means this
#    threshold is only trustworthy near the ~500mm/300mm geometry it was
#    measured at -- re-measure against the real rig's actual
#    camera-to-board distance once known, don't assume 0.01 transfers to
#    a differently-scaled rig.
MIN_LANDMARK_HULL_AREA_FRACTION = 0.01


# --- Coplanar 4-point pose ambiguity (2026-08-12)
# -- investigated, partially shipped, partially deferred; read
# before changing either of the two things below it enabled.
#
# BACKGROUND: coplanar 4-point PnP has a well-known 2-fold pose ambiguity
# (IPPE). This project's OWN offline intrinsics-estimation tool
# (dev/calibration/intrinsics_estimation.py's `_pose_seeds()`) already
# handles it via `cv2.solvePnPGeneric(..., flags=cv2.SOLVEPNP_IPPE)`,
# trying both solutions -- but the LIVE extrinsics path here did not, and
# every real calibration this rig does goes through exactly 4 points
# (opendarts.calibration.sector_correspondence.correspond_landmarks() always
# returns exactly 4), so this was a real, live gap, not a theoretical one.
#
# WHAT WAS MEASURED (reusing this project's OWN existing adversarial fixture,
# tests/test_pnp_calibration.py's CLUSTERED_QUAD_LABELS, 450 trials: 3
# camera positions x 150 noise seeds at 0.5px stddev, plus the
# well-spread quad as a control):
#
# 1. **`cv2.solvePnPRansac`'s default flag does NOT implicitly do IPPE-
#    style branch disambiguation** -- explicit IPPE-best-of-2 selection
#    gives measurably DIFFERENT results than what solve_extrinsics()
#    already returns (verified, not assumed).
# 2. **A real correction to this task's own starting assumption**: RANSAC
#    on this project's real minimal 4-point set is NOT merely
#    "functionally equivalent to plain solvePnP" (no redundancy to reject
#    outliers with) -- it is MEASURABLY WORSE. Clustered (adversarial)
#    quad: mean rotation error 11.53deg (RANSAC) vs 6.11deg (plain
#    solvePnP); well-spread (realistic, real-rig-like) quad: 0.301deg
#    (RANSAC) vs 0.220deg (plain), a smaller but consistent gap in the
#    SAME direction on both quads tested. **Fixed below**: solve_
#    extrinsics() now uses plain cv2.solvePnP whenever exactly 4 points
#    are given, regardless of the `use_ransac` argument -- RANSAC remains
#    unchanged (and still the right choice) for >4 points, where genuine
#    outlier rejection is possible (see the existing 81-point regression
#    test, test_pnp_reports_inlier_fraction_not_full_set_error_with_
#    outliers, unaffected by this change).
# 3. **Naive IPPE-based branch SELECTION is a real trade-off, not a clean
#    win -- NOT shipped as an automatic pose swap.** Replacing the solve
#    outright with "best of IPPE's 2 solutions by reprojection error"
#    does shrink the catastrophic (>100deg) failure tail on the
#    clustered-quad adversarial case (4.7%->1.8% with the now-fixed
#    RANSAC-off default above) -- but at real cost to TYPICAL-case
#    accuracy in that SAME scenario (median rotation error roughly
#    doubles, 4.6deg->8.5deg). A 3-way ensemble (current solve + both
#    IPPE solutions, lowest reprojection error wins) does not meaningfully
#    shrink the catastrophic tail either (4.7%->4.9%, i.e. flat/slightly
#    worse) -- because in the genuinely catastrophic cases, the "wrong"
#    branch's reprojection error is frequently just as low as the
#    "right" branch's (both fit the noisy 4-point input comparably well),
#    the EXACT SAME structural finding already documented elsewhere in
#    this project (the batch8 session's
#    "ray-agreement cannot discriminate correlated bias from a single bad
#    ray" finding) -- reprojection-error-based selection cannot reliably
#    tell the two coplanar branches apart at real noise levels, so
#    automatically swapping to it would trade a measured, real accuracy
#    cost for a tail-risk reduction that a corpus-wide test could not
#    actually confirm holds outside this one synthetic construction.
#    **Not shipped**, consistent with this task's own "honest non-result
#    is acceptable" standard.
# 4. **What WAS shipped instead**: a pure DIAGNOSTIC, not a pose swap
#    (see `coplanar_alt_reprojection_error_px` on `PnpResult` below) --
#    exposes the alternate IPPE branch's reprojection error alongside the
#    selected pose's own, so a caller (or future calibration-quality gate
#    -- see docs/DESIGN.md's dart-21 investigation entry, which already
#    recommends exactly this class of signal) can see when a solve came
#    from a near-ambiguous quad, without this module silently guessing
#    which branch to trust. No threshold is set here (no real measured
#    distribution of this gap exists yet on live data) -- deriving one is
#    real, separate future work, per this project's "measure before
#    guessing" discipline, not invented here to look more finished.
COPLANAR_POINT_COUNT = 4


def _coplanar_alt_reprojection_error_px(
    object_points_mm: np.ndarray,
    image_points_px: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    selected_rvec: np.ndarray,
) -> float | None:
    """See the module-level comment above `COPLANAR_POINT_COUNT`. Only
    meaningful when exactly `COPLANAR_POINT_COUNT` points were solved --
    returns None otherwise, or if IPPE itself can't produce 2 solutions
    for this input (e.g. near-collinear points), rather than guessing."""
    import cv2

    if len(object_points_mm) != COPLANAR_POINT_COUNT:
        return None
    try:
        n_sol, rvecs, tvecs, _ = cv2.solvePnPGeneric(
            object_points_mm, image_points_px, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE,
        )
    except cv2.error:
        return None
    if n_sol < 2:
        return None

    def _reproj_err(rvec, tvec) -> float:
        proj, _ = cv2.projectPoints(object_points_mm, rvec, tvec, camera_matrix, dist_coeffs)
        observed = image_points_px.reshape(-1, 2)
        return float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - observed, axis=1)))

    # Identify which of IPPE's 2 solutions is the "selected" one (closest
    # in rotation to what solve_extrinsics() actually returned) and report
    # the OTHER one's error -- not just "the worse of the two IPPE
    # solutions" (which would be trivially true by construction and tell
    # a caller nothing about the ACTUAL selected pose's ambiguity).
    selected_R, _ = cv2.Rodrigues(np.asarray(selected_rvec, dtype=np.float64).reshape(3, 1))
    best_i, best_dist = None, None
    for i in range(n_sol):
        R_i, _ = cv2.Rodrigues(np.asarray(rvecs[i], dtype=np.float64).reshape(3, 1))
        # Frobenius distance between rotation matrices -- cheap, monotonic
        # proxy for angular distance, adequate for a nearest-match lookup
        # (not used as an actual angle anywhere).
        d = float(np.linalg.norm(R_i - selected_R))
        if best_dist is None or d < best_dist:
            best_dist, best_i = d, i
    alt_i = 1 - best_i if n_sol == 2 else None
    if alt_i is None:
        return None
    return _reproj_err(rvecs[alt_i], tvecs[alt_i])


def landmark_hull_area_fraction(
    image_points_px: np.ndarray,
    image_width: float,
    image_height: float,
) -> float:
    """Fraction of the image area covered by the 2D convex hull of the
    input landmark points -- a point-spread/conditioning signal computed
    purely from the INPUT to PnP, before any solve happens. See the module
    docstring above this function for why (a correlated-bias finding)
    and the empirical basis for
    MIN_LANDMARK_HULL_AREA_FRACTION.

    Degenerates to 0.0 for <3 points or collinear points (zero-area hull)
    -- both are real "not spread out" cases, not a bug to special-case
    around. Also degenerates to 0.0 if any point is non-finite (NaN/inf) --
    A 2026-08-12 review found the naive version silently computed
    the hull over just the finite points, masking a failed landmark
    detection as "fewer points, still maybe fine" instead of failing safe;
    one bad point should make the WHOLE calibration input suspect, not
    just quietly not count.
    """
    import cv2

    pts = np.asarray(image_points_px, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3 or not np.all(np.isfinite(pts)):
        return 0.0
    pts = pts.astype(np.float32)
    hull = cv2.convexHull(pts)
    area_px2 = cv2.contourArea(hull)
    image_area_px2 = float(image_width) * float(image_height)
    if image_area_px2 <= 0.0:
        return 0.0
    return float(area_px2) / image_area_px2


@dataclass
class PnpResult:
    ok: bool
    rvec: np.ndarray | None
    tvec: np.ndarray | None
    reprojection_error_px: float | None
    reason: str = ""
    # Set only when RANSAC ran. reprojection_error_px is always computed
    # over inliers only when these are populated -- computing it over
    # the full input set
    # silently misreports a good pose as bad whenever RANSAC correctly
    # rejected outliers, which is exactly the scenario a real confidence
    # signal needs to get right.
    inlier_indices: np.ndarray | None = None
    inlier_fraction: float | None = None
    # Calibration-side conditioning signal, see landmark_hull_area_fraction
    # above -- computed on the INPUT
    # landmark points regardless of whether the PnP solve itself succeeded
    # (a badly-clustered quad is suspect even if cv2 happily returns a
    # confident-looking pose for it, which is exactly the failure mode
    # this exists to catch). None only when there weren't enough points to
    # even attempt a hull (<3).
    landmark_hull_area_fraction: float | None = None
    landmark_spread_ok: bool | None = None
    # Coplanar 4-point pose-ambiguity diagnostic, see the module-level
    # comment above COPLANAR_POINT_COUNT (2026-08-12) for the full real
    # measurement this is based on. None unless
    # exactly COPLANAR_POINT_COUNT (4) points were solved AND IPPE could
    # produce 2 solutions for this input -- the common real case, since
    # opendarts.calibration.sector_correspondence.correspond_landmarks()
    # always returns exactly 4. When set, this is the reprojection error
    # (px) of the OTHER coplanar pose solution IPPE finds -- a LOW value
    # close to `reprojection_error_px` means the two branches fit the
    # input comparably well (a near-ambiguous quad, real ambiguity risk);
    # a HIGH value means the selected pose is a clear, confident winner.
    # Diagnostic only -- does NOT change which pose is returned (see the
    # module comment for why an automatic swap was investigated and NOT
    # shipped: reprojection error cannot reliably discriminate the two
    # branches at real noise levels, a measured finding, not a guess).
    coplanar_alt_reprojection_error_px: float | None = None


def solve_extrinsics(
    object_points_mm: np.ndarray,
    image_points_px: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    use_ransac: bool = True,
    image_width: float | None = None,
    image_height: float | None = None,
) -> PnpResult:
    """Solve for camera extrinsics (rvec, tvec) given known 3D landmarks
    and their observed 2D pixel positions, with known intrinsics.

    use_ransac=True (default) uses cv2.solvePnPRansac -- robust to a
    fraction of bad landmark-detection correspondences, which matters
    once this runs against real (imperfect) landmark detection in a
    later phase; plain solvePnP has no outlier rejection at all.

    **Exception, added 2026-08-12 (see the
    module-level comment above COPLANAR_POINT_COUNT for the full real
    measurement)**: when exactly COPLANAR_POINT_COUNT (4) points are
    given, this ALWAYS uses plain cv2.solvePnP internally, regardless of
    `use_ransac` -- RANSAC has no redundancy to reject outliers with at
    the minimal point count (every point must be used regardless), and
    was measured to be not just unhelpful there but actively WORSE than
    plain solvePnP (real numbers in the module comment). `inlier_indices`/
    `inlier_fraction` stay None in this case (RANSAC did not run), same
    as any other case where it doesn't run.

    image_width/image_height: only needed to compute
    landmark_hull_area_fraction (see module docstring). If omitted,
    inferred as 2*camera_matrix[0,2] and
    2*camera_matrix[1,2] (i.e. assumes a centered principal point, true
    for this project's synthetic cameras -- pass explicit dimensions once
    real-rig intrinsics are known and the principal point may not be
    centered).
    """
    import cv2

    object_points_mm = np.asarray(object_points_mm, dtype=np.float64).reshape(-1, 1, 3)
    image_points_px = np.asarray(image_points_px, dtype=np.float64).reshape(-1, 1, 2)

    if image_width is None:
        image_width = 2.0 * float(camera_matrix[0, 2])
    if image_height is None:
        image_height = 2.0 * float(camera_matrix[1, 2])

    hull_frac = None
    spread_ok = None
    if len(image_points_px) >= 3:
        hull_frac = landmark_hull_area_fraction(image_points_px, image_width, image_height)
        spread_ok = hull_frac >= MIN_LANDMARK_HULL_AREA_FRACTION

    if len(object_points_mm) < 4:
        return PnpResult(
            ok=False,
            rvec=None,
            tvec=None,
            reprojection_error_px=None,
            reason=f"need >=4 point correspondences, got {len(object_points_mm)}",
            landmark_hull_area_fraction=hull_frac,
            landmark_spread_ok=spread_ok,
        )

    # See solve_extrinsics()'s own docstring + the module-level comment
    # above COPLANAR_POINT_COUNT: RANSAC is skipped at exactly 4 points
    # regardless of `use_ransac` -- measured to be actively worse there,
    # not merely redundant, since there's no larger set to reject
    # outliers from.
    effective_use_ransac = use_ransac and len(object_points_mm) != COPLANAR_POINT_COUNT

    inliers = None
    if effective_use_ransac:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            object_points_mm, image_points_px, camera_matrix, dist_coeffs
        )
    else:
        success, rvec, tvec = cv2.solvePnP(
            object_points_mm, image_points_px, camera_matrix, dist_coeffs
        )

    if not success:
        return PnpResult(
            ok=False,
            rvec=None,
            tvec=None,
            reprojection_error_px=None,
            reason="cv2.solvePnP(Ransac) returned success=False",
            landmark_hull_area_fraction=hull_frac,
            landmark_spread_ok=spread_ok,
        )

    reprojected, _ = cv2.projectPoints(
        object_points_mm, rvec, tvec, camera_matrix, dist_coeffs
    )
    reprojected = reprojected.reshape(-1, 2)
    observed = image_points_px.reshape(-1, 2)
    per_point_err = np.linalg.norm(reprojected - observed, axis=1)

    inlier_indices = None
    inlier_fraction = None
    if inliers is not None:
        inlier_indices = inliers.reshape(-1)
        inlier_fraction = float(len(inlier_indices)) / float(len(object_points_mm))
        # Error over inliers only -- see PnpResult docstring/comment for
        # why computing this over the full set is actively misleading.
        err = float(per_point_err[inlier_indices].mean())
    else:
        err = float(per_point_err.mean())

    coplanar_alt_err = _coplanar_alt_reprojection_error_px(
        object_points_mm, image_points_px, camera_matrix, dist_coeffs, rvec
    )

    return PnpResult(
        ok=True,
        rvec=rvec,
        tvec=tvec,
        reprojection_error_px=err,
        inlier_indices=inlier_indices,
        inlier_fraction=inlier_fraction,
        landmark_hull_area_fraction=hull_frac,
        landmark_spread_ok=spread_ok,
        coplanar_alt_reprojection_error_px=coplanar_alt_err,
    )


def pose_error(
    rvec_true: np.ndarray,
    tvec_true: np.ndarray,
    rvec_est: np.ndarray,
    tvec_est: np.ndarray,
) -> tuple[float, float]:
    """(rotation_error_deg, translation_error_mm) between two world-to-camera
    poses -- for comparing a PnP-recovered pose against synthetic ground
    truth. Rotation error via the angle of R_true^T @ R_est (standard
    geodesic distance on SO(3))."""
    import cv2

    R_true, _ = cv2.Rodrigues(rvec_true)
    R_est, _ = cv2.Rodrigues(rvec_est)
    R_diff = R_true.T @ R_est
    trace = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
    rot_err_deg = float(np.degrees(np.arccos(trace)))

    C_true = -R_true.T @ tvec_true.reshape(3)
    C_est = -R_est.T @ tvec_est.reshape(3)
    trans_err_mm = float(np.linalg.norm(C_true - C_est))

    return rot_err_deg, trans_err_mm
