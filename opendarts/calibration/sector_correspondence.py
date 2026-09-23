"""4-point cardinal-quad geometric indexing convention + robust N-frame
averaging for landmark correspondences.

**The quad convention** (`AD_QUAD_DST_BASE_DEG`,
`ad_quad_board_angle_deg()`, `ad_quad_object_point_mm()`): the four
double-outer wire junctions at board angles 9 / 99 / 189 / 279 deg, in a
fixed index order. A pure coordinate-labeling CONVENTION matching
universal WDF/PDC board geometry (cross-checked against
`opendarts.geometry.board`'s own independent wire-boundary-angle
formula), not a fitted/measured value specific to this rig.
`opendarts.calibration.oriented_landmarks`'s own `AD_QUAD_RING_INDICES`
convention and this project's test suite (`tests/test_oriented_landmarks*.py`)
use it as an independent cross-check of that module's own
landmark-indexing geometry, even though no live call site imports it
directly today. (An older per-camera orientation-constant mechanism that
also lived here was removed on 2026-08-20; the live path uses
`opendarts.calibration.oriented_landmarks.correspond_landmarks_oriented()`.)

Also here, and very much live: the robust N-frame averaging utility
below, used by `opendarts.live.capture_daemon.bootstrap_calibrations()`
and by the offline session refit to combine several independent
(object_points_mm, image_points_px) correspondences from the SAME
camera into one, trimming per-(frame, landmark-index) outliers before
averaging. Nothing about this utility is camera-specific in its own
right -- it operates on whatever correspondences its caller supplies
(today, `oriented_landmarks.correspond_landmarks_oriented()`'s output),
and its trimming thresholds (see the TRIM_* constants below) were
measured from this rig's own real per-frame noise.
"""
from __future__ import annotations

import math

import numpy as np

# A coordinate-labeling convention, not fitted/measured data (see module
# docstring). Order matters: index i here is "quad index i", the same
# convention
# `opendarts.calibration.oriented_landmarks.AD_QUAD_RING_INDICES` follows.
AD_QUAD_DST_BASE_DEG: tuple[float, float, float, float] = (279.0, 9.0, 99.0, 189.0)


def ad_quad_board_angle_deg(index: int, orient_k: int = 0) -> float:
    """Board angle (this repo's convention: clockwise from +Y/12 o'clock,
    see opendarts.geometry.board) of quad landmark `index` (0..3), for a
    board mounted with rotation `orient_k` (0 for every real
    calibration.json seen so far in this project's data). Derivation:
    board_angle = virtual_angle + 90 (mod 360) -- see this module's
    git history (pre-2026-08-20) for the full algebraic derivation this
    docstring used to carry in full; not reproduced here since it's no
    longer load-bearing for anything beyond this pure geometry helper."""
    if not 0 <= index < 4:
        raise ValueError(f"quad index must be 0..3, got {index}")
    od_virtual_angle = AD_QUAD_DST_BASE_DEG[index] + 18.0 * orient_k
    return (od_virtual_angle + 90.0) % 360.0


def ad_quad_object_point_mm(index: int, orient_k: int = 0) -> tuple[float, float]:
    """(x_mm, y_mm) of quad landmark `index` on the double-outer ring
    (Z=0 board plane), using this repo's board-angle convention (see
    opendarts.geometry.board.polar_to_xy_mm: x = R*sin(angle), y = R*cos(angle))."""
    from opendarts.geometry.board import DOUBLE_OUTER_RADIUS_MM

    angle = math.radians(ad_quad_board_angle_deg(index, orient_k))
    return (
        DOUBLE_OUTER_RADIUS_MM * math.sin(angle),
        DOUBLE_OUTER_RADIUS_MM * math.cos(angle),
    )

# ROBUST/TRIMMED AVERAGING, added 2026-08-14. Replaces the plain np.mean() below with a trim-then-mean: reject
# individual (frame, landmark-index) samples that sit too far from that
# index's own median BEFORE averaging the survivors, so a handful of
# plausible-but-wrong detections (occlusion, glare, motion blur) can no
# longer get blindly averaged in with the good ones.
#
# **MEASURED, not guessed** (not
# shipped, throwaway harness per this project's "measure the real number"
# discipline) -- real per-landmark-index pixel deviation from the group
# median, across up to 30 independent real bg-image detections per
# (session, camera) drawn from data/archive/clean/'s 3 real sessions (the
# same "board doesn't move within a session, so many throws' bg images
# ARE many independent repeat measurements" logic tools/geometry/
# measure_board_boundary_paint.py already relies on). Pooled across all
# 9 real (session, camera) pairs (n=1064 real samples): median=0.454px,
# p90=1.540px, p99=6.568px, MAD=0.255px -- i.e. real per-frame noise is
# normally SUB-PIXEL, with an occasional few-px tail. But the SAME real
# data also caught one genuine real outlier already sitting in the
# corpus: one session's camera 0 has one frame whose landmark
# lands 237.893px from that index's own median (vs a 0.705px median for
# the rest of that same group) -- exactly the "plausible-but-wrong
# detection" this trimming exists to catch, found in real production
# data, not synthesized.
#
# **Why the threshold is computed FRESH from each call's own batch, not
# a fixed constant from the measurement above**: the "normal" noise floor
# genuinely differs by camera/session in that same real data (e.g.
# that session's cam0 has a non-outlier median of 0.705px vs its
# cam1's 0.273px -- already >2x apart) -- a
# single global fixed px threshold would be too tight for a legitimately
# noisier real burst or too loose for
# a very precise one. Each call instead computes its own median and MAD
# of per-point deviations (pooled across all 4 landmark indices for a
# more stable robust-scale estimate at the realistically small per-round
# N this project actually uses -- N as low as 3) and rejects a point only
# if it sits more than TRIM_MAD_MULTIPLIER robust-sigmas past that
# BATCH's own median deviation, floored at TRIM_ABS_FLOOR_PX so an
# unusually tight/consistent batch can't make the gate fire on genuinely
# tiny, real noise.
#
# TRIM_MAD_MULTIPLIER=6.0 is deliberately looser than the standard
# Iglewicz & Hoaglin outlier recommendation (3.5) -- checked directly
# against this project's own existing
# test_average_correspondences_closer_to_ground_truth_than_any_single_
# noisy_frame fixture (6 synthetic detections, isotropic Gaussian jitter,
# std=8px/axis, seed 20260812): at MAD_MULTIPLIER=6.0 every one of the 6
# frames survives trimming (the synthetic batch's own largest deviation,
# 23.46px, is real 8px-std Gaussian tail, not an outlier, and must not be
# discarded) -- a tighter 3.5 multiplier trims 1 of 24 points on that
# exact fixture, which is the over-aggressive failure mode being guarded
# against here. TRIM_ABS_FLOOR_PX=4.0 sits comfortably above the real
# pooled p90 (1.540px) and below the real p99.5 (8.939px) -- generous to
# normal noise, still well under the real 237.893px outlier case above.
TRIM_MAD_MULTIPLIER = 6.0
TRIM_ABS_FLOOR_PX = 4.0
# Below this many successful detections, a per-batch median/MAD estimate
# is too unstable to trust (e.g. with 2 points, "the median" is just
# whichever point happens to be picked, and MAD is either 0 or the full
# spread) -- fall back to the plain mean, exactly today's old behavior,
# rather than let a tiny sample size make trimming decisions look
# confident when they aren't.
MIN_DETECTIONS_FOR_TRIMMING = 4


def _trimmed_mean_image_points(
    stack: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """stack: (n, 4, 2) image_points_px from n independent detections of
    the SAME camera. Returns (trimmed_mean (4, 2), diagnostics dict).

    Trims per (frame, landmark-index) -- not per whole frame -- since a
    real bad detection (e.g. glare on ONE of the 4 physical wire
    crossings) can plausibly shift just one landmark index while leaving
    the other 3 fine; whole-frame rejection would need to throw away 3
    good points to discard 1 bad one. See this module's own TRIM_*
    constants above for the real measured thresholds and reasoning.
    """
    n = stack.shape[0]
    median = np.median(stack, axis=0) # (4, 2)
    dev = np.linalg.norm(stack - median[None, :, :], axis=2) # (n, 4)

    if n < MIN_DETECTIONS_FOR_TRIMMING:
        return stack.mean(axis=0), {
            "trimmed": False,
            "reason": f"only {n} detections, below MIN_DETECTIONS_FOR_TRIMMING={MIN_DETECTIONS_FOR_TRIMMING}",
            "n_trimmed_per_index": [0, 0, 0, 0],
        }

    pooled = dev.flatten()
    med_dev = float(np.median(pooled))
    mad_dev = float(np.median(np.abs(pooled - med_dev)))
    threshold_px = max(TRIM_ABS_FLOOR_PX, med_dev + TRIM_MAD_MULTIPLIER * 1.4826 * mad_dev)
    keep_mask = dev <= threshold_px # (n, 4)

    out = np.zeros((4, 2), dtype=np.float64)
    n_trimmed_per_index = [0, 0, 0, 0]
    for idx in range(4):
        kept = stack[keep_mask[:, idx], idx, :]
        if kept.shape[0] == 0:
            # Every single frame flagged as an outlier for this index
            # simultaneously -- astronomically unlikely with a real
            # median-relative threshold (would need >half the batch to
            # agree on a wrong answer while disagreeing with each other
            # enough to all exceed the threshold), but handled explicitly
            # rather than assumed impossible: fall back to the plain mean
            # for this index alone rather than produce a NaN.
            kept = stack[:, idx, :]
        else:
            n_trimmed_per_index[idx] = int(n - kept.shape[0])
        out[idx] = kept.mean(axis=0)

    return out, {
        "trimmed": True,
        "median_dev_px": med_dev,
        "mad_dev_px": mad_dev,
        "threshold_px": threshold_px,
        "n_trimmed_per_index": n_trimmed_per_index,
        "n_total": n,
    }


def average_correspondences(
    detections: list[tuple[np.ndarray, np.ndarray] | None],
    *,
    min_required: int = 1,
    trim: bool = True,
    trim_diagnostics_out: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Combine N independent (object_points_mm, image_points_px)
    correspondences from the SAME camera into ONE pair, ready for a
    single opendarts.pipeline.calibrate_camera() PnP solve -- the core of
    the N-frame-averaged calibration fix, added 2026-08-12 (see
    opendarts.live.capture_daemon.bootstrap_calibrations() for the real,
    measured motivation and the call sites that use this). Today's
    caller supplies correspondences via
    `opendarts.calibration.oriented_landmarks.correspond_landmarks_oriented()`
    (see that module and `bootstrap_calibrations()`), not this module's
    own former (now-removed) orientation-constant mechanism.

    `trim` (default True, 2026-08-14): reject per-(frame, landmark-index)
    outliers before averaging -- see the TRIM_* constants and
    `_trimmed_mean_image_points()` above for the real measured basis.
    Pass False to get the old, un-trimmed plain mean (kept only for
    callers that explicitly want to compare against the pre-2026-08-14
    behavior, e.g. a before/after measurement script -- no production
    call site should need this). `trim_diagnostics_out`, if given, is
    filled with the trimming diagnostics dict (threshold used, how many
    points got trimmed per index) -- a caller like
    opendarts.live.capture_daemon.bootstrap_calibrations() can log this so a
    trimmed outlier is visible, not silent.

    WHY AVERAGING image_points ACROSS FRAMES IS VALID HERE (checked, not
    assumed -- this was an explicit blocker to verify before relying on
    it): averaging index-i across frames only makes sense if landmark
    index i means the SAME physical board point in every frame being
    averaged. That is the CALLER's contract to uphold (today,
    `correspond_landmarks_oriented()`'s own fixed quad-ring-index
    convention, independent of any per-image content) -- this function
    still asserts object_points_mm is identical across every non-None
    input it receives (a real runtime check, not just a comment)
    precisely because "verify this rather than assume it" was the
    explicit instruction this function was built under -- if that
    assertion ever fires, it means the correspondence contract broke
    somewhere upstream and averaging must NOT silently proceed.

    detections: one entry per captured frame, in any order -- None for a
    frame where correspondence itself failed (landmark detection or
    orientation lock). Only non-None entries contribute; this is the
    partial-failure handling bootstrap_calibrations()'s own docstring
    describes ("average whatever succeeded, don't fail the whole
    calibration for one bad frame among many").

    min_required: the minimum COUNT of non-None entries needed to trust
    the resulting average -- returns None (not a best-effort average)
    below this floor, so a caller can tell "too few real samples to
    trust this pass" apart from "a valid, if imperfect, average". The
    actual number and its reasoning live in
    opendarts.live.capture_daemon._min_calibration_frames_required(), not
    here -- this function just enforces whatever floor it's given.

    Returns (object_points_mm, image_points_px, n_used) on success --
    n_used is how many of `detections` actually contributed (always >=
    min_required). Returns None if fewer than min_required succeeded.

    NOISE THIS DOES / DOES NOT FIX -- see bootstrap_calibrations()'s own
    docstring for the full, honest statement: this reduces RANDOM
    per-frame noise (sensor noise, sub-pixel ellipse-fit jitter) via
    averaging, which is genuinely measured to shrink pose/reprojection-
    error spread roughly per a 1/sqrt(N) trend on real data. It does
    NOT fix a SYSTEMATIC bias shared by every frame in one quick capture
    burst (e.g. a lighting-condition-dependent detector bias affecting
    every frame the same way) -- averaging N correlated-biased samples
    just reproduces the same bias N times, not cancels it. That's the
    same "correlated calibration bias" gap already tracked in
    docs/DESIGN.md (opendarts/engines/apollo/scoring.py's
    MAX_RAY_DISAGREEMENT_MM docstring), unresolved by this change.
    """
    successes = [d for d in detections if d is not None]
    if len(successes) < min_required:
        return None

    object_points_ref = successes[0][0]
    for obj, _img in successes[1:]:
        assert np.allclose(obj, object_points_ref), (
            "object_points_mm differed across independent correspondence "
            "calls for the same camera -- the caller's fixed-index "
            "correspondence contract has been violated somewhere upstream; "
            "averaging image_points under a mismatched object_points_mm would "
            "silently produce a wrong calibration, so this is a hard failure, "
            "not a warning."
        )

    stack = np.stack([s[1] for s in successes], axis=0)
    if trim:
        image_points_avg, diagnostics = _trimmed_mean_image_points(stack)
    else:
        image_points_avg = stack.mean(axis=0)
        diagnostics = {"trimmed": False, "reason": "trim=False", "n_trimmed_per_index": [0, 0, 0, 0]}
    if trim_diagnostics_out is not None:
        trim_diagnostics_out.update(diagnostics)
    return object_points_ref.copy(), image_points_avg, len(successes)
