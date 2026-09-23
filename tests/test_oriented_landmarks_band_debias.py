"""Tests for `oriented_landmarks._debias_band_outer_radius()` -- the
radial de-biasing of the double ring's outer edge.

WHY THIS EXISTS (real finding, 2026-08-13). The colour mask does not stop
at the paint: blur plus the dark transition pixels either side of a bed
spill it outward past the double-OUTER wire (170.0mm) and inward past the
double-INNER wire (162.0mm). An ellipse fitted to the mask's outer edge
therefore sits outside the true 170mm ring, every landmark gets labelled
170mm while living further out, PnP solves a board too big in pixels, and
every scored tip reads SHORT in board radius in proportion to its radius.

Measured non-circularly on 8 real sessions x 3 cameras, by mapping the
mask's INNER edge through the very same homography its OUTER edge
defines: it came back at 158.52mm instead of 162.0mm, implying a
symmetric mask dilation of 1.80mm (3.6 px on a 335 px semi-axis) and a
1.075% radial scale error. Cross-checked against the oracle's tip positions,
which the derivation never consulted: the real signed radial residual was
-0.93mm mean / -0.99mm median before the correction and +0.01mm mean /
-0.15mm median after it, with median 2-D board error 2.59mm -> 2.32mm.

The real frames are gitignored, so the tests below work on EXACT
synthetic geometry instead, which is the stronger check anyway: the true
170mm and 162mm pixel radii are known in closed form, so the correction
can be asserted to recover the true ring rather than merely to improve.

Every tolerance was measured first and is quoted with what was observed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import opendarts.calibration.oriented_landmarks as oriented_landmarks
from opendarts.calibration.landmark_detection import Ellipse
from opendarts.calibration.oriented_landmarks import (
    _debias_band_outer_radius,
    affine_unit_circle_to_ellipse,
    apply_homography,
    disk_boost,
    ellipse_ray_intersections,
    locate_pre_orientation_landmarks,
    reseat_ellipse,
)
from opendarts.geometry.board import DOUBLE_INNER_RADIUS_MM, DOUBLE_OUTER_RADIUS_MM

from tests.test_oriented_landmarks import (
    board_mm_to_unit,
    ellipse_and_bull_from,
    render_synthetic_board,
    synthetic_board_homography,
)


def _exact_band(cam_index, n_rays=720):
    """A real projective camera's EXACT pixel radii for the two wires
    bounding the double bed, on a dense set of rays from the bull.

    These are closed-form, not detected, so a test can assert that the
    correction lands on the true ring rather than merely nearer it.
    """
    H_mm, _ = synthetic_board_homography(cam_index)
    ellipse, bull = ellipse_and_bull_from(board_mm_to_unit(H_mm))

    angles = np.arange(n_rays, dtype=np.float64) * (360.0 / n_rays)
    hits = ellipse_ray_intersections(ellipse, bull, angles)
    keep = np.isfinite(hits).all(axis=1)
    hits, angles = hits[keep], angles[keep]
    dirs = np.stack([np.cos(np.radians(angles)), np.sin(np.radians(angles))], axis=1)

    r_outer_true = np.hypot(hits[:, 0] - bull[0], hits[:, 1] - bull[1])

    A = affine_unit_circle_to_ellipse(ellipse)
    u = apply_homography(np.linalg.inv(A), [bull])[0]
    M = A @ disk_boost(-u)
    q = apply_homography(np.linalg.inv(M), hits)
    qhat = q / np.hypot(q[:, 0], q[:, 1])[:, None]
    inner = apply_homography(M, qhat * (DOUBLE_INNER_RADIUS_MM / DOUBLE_OUTER_RADIUS_MM))
    r_inner_true = np.hypot(inner[:, 0] - bull[0], inner[:, 1] - bull[1])

    return ellipse, bull, dirs, r_inner_true, r_outer_true


@pytest.mark.parametrize("cam_index", [0, 1, 2])
def test_an_undilated_band_is_left_exactly_alone(cam_index):
    """The correction is a no-op when there is nothing to correct.

    This is the property that makes it safe to apply unconditionally: a
    camera or board finish whose mask happens to land on the paint edge
    exactly is not nudged off it. Measured residual on exact geometry:
    0.000 px on all three cameras.
    """
    ellipse, bull, dirs, r_in, r_out = _exact_band(cam_index)
    corrected = _debias_band_outer_radius(ellipse, bull, dirs, r_in, r_out)
    assert float(np.abs(corrected - r_out).max()) < 0.05


@pytest.mark.parametrize("cam_index", [0, 1, 2])
@pytest.mark.parametrize("delta_px", [2.0, 3.6, 6.0])
def test_recovers_the_true_ring_from_a_symmetrically_dilated_band(cam_index, delta_px):
    """The fix itself: given a band spilled outward AND inward by the
    same amount, recover the true 170mm ring.

    3.6 px is the real measured dilation on this rig; 2.0 and 6.0 bracket
    it. Measured mean residual after correction, against an uncorrected
    error of exactly `delta_px`:
        delta 2.0 px -> -0.005 px
        delta 3.6 px -> +0.274 px
        delta 6.0 px -> +0.887 px
    The residual grows with delta because the band's PIXEL midpoint is
    not exactly its BOARD midpoint under perspective -- a second-order
    term, negligible at the real 3.6 px and still a 7x improvement at an
    exaggerated 6.0 px.
    """
    ellipse, bull, dirs, r_in, r_out = _exact_band(cam_index)
    corrected = _debias_band_outer_radius(
        ellipse, bull, dirs, r_in - delta_px, r_out + delta_px
    )
    residual = float(np.mean(corrected - r_out))
    assert abs(residual) < 0.25 * delta_px, residual


@pytest.mark.parametrize("cam_index", [0, 1, 2])
def test_an_asymmetric_spill_leaves_exactly_half_the_asymmetry(cam_index):
    """The honest limitation, asserted rather than only documented.

    The correction assumes the mask spills equally at both edges. If it
    does not, the band midpoint moves by half the difference and the
    correction inherits exactly that error -- it cannot see it, because
    the two edges are the only evidence there is. Measured on a band
    spilled 4.0 px out and 1.0 px in: residual +1.494 px, i.e. (4-1)/2 to
    within 0.01 px, and the mirror case gives -1.507 px.

    This test exists so that if someone later claims the correction is
    robust to asymmetric spill, the claim fails here.
    """
    ellipse, bull, dirs, r_in, r_out = _exact_band(cam_index)
    for out_px, in_px in ((4.0, 1.0), (1.0, 4.0)):
        corrected = _debias_band_outer_radius(
            ellipse, bull, dirs, r_in - in_px, r_out + out_px
        )
        residual = float(np.mean(corrected - r_out))
        assert abs(residual - 0.5 * (out_px - in_px)) < 0.1, (out_px, in_px, residual)


@pytest.mark.parametrize("cam_index", [0, 1, 2])
def test_a_nonsense_band_falls_back_to_the_raw_outer_edge(cam_index):
    """A band measurement that has gone wrong must not move the ellipse.

    Two real ways it goes wrong on this rig's frames: the inner-edge walk
    over-runs the gap and lands in the TREBLE ring (an enormous band), or
    it stops immediately (a band of nothing). Both are rejected per ray
    by the width window, and if too few rays survive the whole correction
    is abandoned rather than computed from the remnant.
    """
    ellipse, bull, dirs, r_in, r_out = _exact_band(cam_index)
    for bad_inner in (r_in * 0.1, r_out - 0.01):
        corrected = _debias_band_outer_radius(ellipse, bull, dirs, bad_inner, r_out)
        assert np.array_equal(corrected, r_out)


@pytest.mark.parametrize("cam_index", [0, 1, 2])
@pytest.mark.parametrize("spill_fraction", [0.2, 0.25, 0.5])
def test_a_grossly_wrong_band_moves_the_ellipse_nowhere(cam_index, spill_fraction):
    """The guards are damage limiters, and they actually limit damage.

    A band far wider than the real double bed cannot be the double bed,
    and must leave the ellipse exactly where the outer edge put it rather
    than dragging it somewhere arbitrary. Two guards can do that -- the
    per-ray width window and the clamp on the resulting factor -- and on
    this geometry the WIDTH WINDOW is the one that fires first: measured
    on the synthetic camera (true band 7.93 px, median outer radius
    134.9 px), an inner edge spilled 20 px gives a measured band 3.52x
    the true width and still yields a factor of 0.9468, just inside the
    0.94 clamp, while 26 px is rejected outright. The clamp is the
    backstop behind it, not the front line. On 144 real frames the factor
    sat in [0.98190, 0.99384], so neither guard has ever fired on real
    data -- which is what a damage limiter should look like.
    """
    ellipse, bull, dirs, r_in, r_out = _exact_band(cam_index)
    spill = spill_fraction * float(np.median(r_out))
    corrected = _debias_band_outer_radius(ellipse, bull, dirs, r_in - spill, r_out)
    assert np.array_equal(corrected, r_out)


@pytest.mark.parametrize("cam_index", [0, 1, 2])
def test_the_returned_radii_are_never_larger_than_a_real_correction(cam_index):
    """Whatever the input, the function returns finite radii on the same
    order as the ring it was given -- it can shift the ellipse by a
    percent or two, never rescale it.

    Guards against the failure mode this whole module exists to avoid: a
    silently wrong ellipse that still looks like an ellipse.
    """
    ellipse, bull, dirs, r_in, r_out = _exact_band(cam_index)
    for delta in (0.0, 1.0, 3.6, 8.0):
        corrected = _debias_band_outer_radius(
            ellipse, bull, dirs, r_in - delta, r_out + delta
        )
        assert np.isfinite(corrected).all()
        ratio = corrected / (r_out + delta)
        assert float(ratio.min()) >= 0.94 and float(ratio.max()) <= 1.02


# ---------------------------------------------------------------------
# A degenerate CANDIDATE ellipse -- disk_boost()'s own domain guard, and
# the real crash it used to cause (confirmed live on OpenDarts's rig,
# 2026-08-31; byte-identical logic here, just not yet rolled by opendarts).
#
#   File "oriented_landmarks.py", line 1040, in locate_pre_orientation_landmarks
#       ellipse, note = reseat_ellipse(image_bgr, seed, bull.xy)
#   File "oriented_landmarks.py", line 591, in reseat_ellipse
#       r_corrected = _debias_band_outer_radius(cand, (bx, by), dirs_a, r_inner_a, r_outer_a)
#   File "oriented_landmarks.py", line 481, in _debias_band_outer_radius
#       M = A @ disk_boost(-u)
#   File "oriented_landmarks.py", line 205, in disk_boost
#       raise ValueError(f"disk_boost needs |u| < 1, got |u|={math.sqrt(n2):.4f}")
#   ValueError: disk_boost needs |u| < 1, got |u|=1.0329
#
# `reseat_ellipse()`'s own 2-pass refit loop re-invokes
# `_debias_band_outer_radius()` on the freshly REFITTED candidate every
# iteration, and that intermediate candidate is never checked against
# `MAX_NORMALISED_BULL_RADIUS` the way the FINAL ellipse
# `reseat_ellipse()` returns is (that check lives one level up, in
# `locate_pre_orientation_landmarks()`) -- so a genuinely degenerate
# second-pass refit (a real, if rare, `_robust_fit_ellipse()` outcome)
# can put the bull outside that candidate's own unit disk even though
# the seed ellipse and bull were both perfectly sane. `disk_boost()`'s
# own check (below) is correct to refuse that input -- these tests are
# NOT about weakening it; they are about the fact that, before the fix
# in `_debias_band_outer_radius()`, nothing caught the resulting
# `ValueError`, so it propagated out of `reseat_ellipse()`, out of
# `locate_pre_orientation_landmarks()`, out of the per-camera detect
# batch's list comprehension, and aborted the WHOLE calibration event.
# ---------------------------------------------------------------------


def _degenerate_candidate(cam_index: int) -> tuple[Ellipse, tuple, np.ndarray]:
    """A real candidate ellipse, and the same bull point used elsewhere in
    this file, arranged so `bull` genuinely sits OUTSIDE the candidate's
    own unit disk (|u| >= 1.0) -- the exact geometric condition
    `disk_boost()` refuses. Not an arbitrary/blind construction: shrunk
    directly from the real synthetic ellipse `_exact_band()` uses, kept
    centred on the SAME point, so the only thing that changed is scale --
    the same shape of change a bad `_robust_fit_ellipse()` refit can
    produce in practice (a plausible-looking ellipse that no longer
    contains the bull the way the real one does).
    """
    ellipse, bull, _dirs, _r_in, _r_out = _exact_band(cam_index)
    degenerate = Ellipse(
        cx=ellipse.cx,
        cy=ellipse.cy,
        major_axis_px=ellipse.major_axis_px * 0.05,
        minor_axis_px=ellipse.minor_axis_px * 0.05,
        angle_deg=ellipse.angle_deg,
    )
    A = affine_unit_circle_to_ellipse(degenerate)
    u = apply_homography(np.linalg.inv(A), [bull])[0]
    r_u = math.hypot(u[0], u[1])
    assert r_u >= 1.0, (
        f"test construction bug: candidate is not actually degenerate (|u|={r_u:.4f})"
    )
    return degenerate, bull, u


@pytest.mark.parametrize("cam_index", [0, 1, 2])
def test_disk_boost_itself_still_correctly_refuses_the_degenerate_candidate(cam_index):
    """Sanity check that this test's own construction really does trigger
    `disk_boost()`'s real guard, and that the guard is UNCHANGED --
    the fix under test is a catch one level up, never a relaxation of
    this check (see the module note above)."""
    _degenerate, _bull, u = _degenerate_candidate(cam_index)
    with pytest.raises(ValueError, match=r"disk_boost needs \|u\| < 1"):
        disk_boost(-u)


@pytest.mark.parametrize("cam_index", [0, 1, 2])
def test_debias_declines_gracefully_instead_of_raising_on_a_degenerate_candidate(cam_index):
    """The actual fix, isolated to the one function that changed:
    `_debias_band_outer_radius()` must decline the correction (return
    `r_outer` unmodified -- the SAME "can't trust this correction"
    contract every other early-return in this function already uses,
    e.g. `test_a_nonsense_band_falls_back_to_the_raw_outer_edge` above)
    rather than let `disk_boost()`'s `ValueError` propagate."""
    degenerate, bull, _u = _degenerate_candidate(cam_index)
    _ellipse, _bull2, dirs, r_in, r_out = _exact_band(cam_index)

    corrected = _debias_band_outer_radius(degenerate, bull, dirs, r_in, r_out)

    assert np.array_equal(corrected, r_out)


@pytest.mark.parametrize("cam_index", [0, 1, 2])
def test_reseat_ellipse_survives_a_degenerate_second_pass_refit(cam_index, monkeypatch):
    """Reproduces the REAL traceback's own call chain end to end:
    `reseat_ellipse()` -> `_debias_band_outer_radius()` -> `disk_boost()`
    -- via the real mechanism that produces it (a bad `_robust_fit_ellipse()`
    refit on the loop's first pass, fed back in as `cand` on the second),
    not a hand-wired shortcut.

    BEFORE the fix in `_debias_band_outer_radius()`, this raised
    `ValueError` straight out of `reseat_ellipse()` -- confirmed directly
    by re-running this exact test against the pre-fix source (`git
    stash`); see this task's own report for the transcript. AFTER the
    fix, `reseat_ellipse()` must return its own normal `(ellipse, note)`
    contract, never a raised exception.

    The REAL (measured, not predicted) recovery is more graceful than
    "fall all the way back to the seed": `_debias_band_outer_radius()`
    declines the bias CORRECTION only, for that one ray-batch, and the
    loop's own second `_robust_fit_ellipse()` call (real fitting, not
    mocked) then refits from the RAW, uncorrected boundary points --
    which are close to the seed's own ray geometry regardless of the
    degenerate intermediate candidate, since neither `seed` nor the
    original boundary walk was ever touched by the poison. The result is
    a real, sane, close-to-seed ellipse that clears every one of
    `reseat_ellipse()`'s own post-loop guardrails on its own merits, not
    a forced identity-equal fallback.
    """
    H_mm, _ = synthetic_board_homography(cam_index)
    img = render_synthetic_board(H_mm)
    degenerate, bull_px, _u = _degenerate_candidate(cam_index)

    from opendarts.calibration.landmark_detection import detect_double_ring_quad

    seed_det = detect_double_ring_quad(img)
    assert seed_det.ok and seed_det.ellipse is not None
    seed = seed_det.ellipse

    real_fit = oriented_landmarks._robust_fit_ellipse
    calls = {"n": 0}

    def poisoned_fit(pts):
        calls["n"] += 1
        if calls["n"] == 1:
            # The loop's first refit lands on a genuinely bad candidate
            # (same mechanism `_robust_fit_ellipse()` uses for a real
            # fit -- cv2.fitEllipse's own tuple shape -- just forced bad
            # here instead of relying on rolling the dice on real
            # degenerate data).
            return degenerate.to_cv2(), None
        return real_fit(pts)

    monkeypatch.setattr(oriented_landmarks, "_robust_fit_ellipse", poisoned_fit)

    ellipse, note = reseat_ellipse(img, seed, bull_px)

    # Both loop iterations ran to completion -- in particular, iteration
    # 2's own `_debias_band_outer_radius(cand=<degenerate>, ...)` call
    # did NOT raise, and iteration 2's own REAL `_robust_fit_ellipse()`
    # call (the `real_fit` branch above) still ran afterwards.
    assert calls["n"] == 2, calls
    assert ellipse is not None
    assert "reseat rejected" not in note and "reseat skipped" not in note, note
    # The recovered ellipse is a real, sane fit close to the true/seed
    # geometry -- nowhere near the 95%-smaller degenerate candidate that
    # broke the loop's second iteration.
    assert abs(ellipse.major_axis_px - seed.major_axis_px) / seed.major_axis_px < 0.05
    assert abs(ellipse.minor_axis_px - seed.minor_axis_px) / seed.minor_axis_px < 0.05
    assert math.hypot(ellipse.cx - seed.cx, ellipse.cy - seed.cy) < 5.0


def test_locate_pre_orientation_landmarks_batch_survives_one_poisoned_frame(monkeypatch):
    """The exact real-world shape this bug broke: a per-camera batch of
    frames run through `locate_pre_orientation_landmarks()` in a plain
    list comprehension (this is a literal quote of the real call site,
    `opendarts.live.capture_daemon._detect_batch()`):

        pre_stage = [locate_pre_orientation_landmarks(frame) for frame in frames]

    BEFORE the fix, one poisoned frame's `ValueError` aborted this ENTIRE
    list comprehension -- none of the other, perfectly good frames in the
    batch ever got a result, and the exception propagated out of the
    whole per-camera detect round (and, in the real incident, out of
    `bootstrap_calibrations()` entirely, discarding every other camera's
    progress too). AFTER the fix, every frame in the batch produces a
    real `PreOrientationLandmarks`, including the poisoned one (which
    gracefully falls back to its own seed ellipse via `reseat_ellipse()`'s
    pre-existing axis-change guardrail and still locates successfully) --
    and the two good frames on either side of it are completely
    unaffected.
    """
    cam_index = 0
    H_mm, _ = synthetic_board_homography(cam_index)
    good_frame_a = render_synthetic_board(H_mm)
    good_frame_b = render_synthetic_board(H_mm)
    poisoned_frame = render_synthetic_board(H_mm)
    degenerate, _bull, _u = _degenerate_candidate(cam_index)

    real_fit = oriented_landmarks._robust_fit_ellipse
    poison_fired = {"n": 0}

    def poisoned_fit_once(pts):
        # Un-patches itself back to the real fitter on its very first
        # call, so it forces exactly ONE degenerate refit -- the
        # poisoned frame's own first reseat-loop iteration -- and every
        # other call, on every other frame (including this same
        # poisoned frame's own SECOND reseat-loop iteration, if
        # reached), uses the real fitting code.
        poison_fired["n"] += 1
        monkeypatch.setattr(oriented_landmarks, "_robust_fit_ellipse", real_fit)
        return degenerate.to_cv2(), None

    # Processed exactly the way the real call site does -- a plain
    # per-frame loop -- poisoning ONLY the middle frame.
    results = []
    for frame in (good_frame_a, poisoned_frame, good_frame_b):
        if frame is poisoned_frame:
            monkeypatch.setattr(oriented_landmarks, "_robust_fit_ellipse", poisoned_fit_once)
        results.append(locate_pre_orientation_landmarks(frame))

    assert len(results) == 3
    _img_a, pre_a = results[0]
    _img_p, pre_p = results[1]
    _img_b, pre_b = results[2]

    assert poison_fired["n"] == 1, "test construction bug: the poison did not fire exactly once"
    assert pre_a.ok, pre_a.reason
    assert pre_b.ok, pre_b.reason
    # The poisoned frame itself still locates -- reseat declined (fell
    # back to seed), but detection as a whole did not crash or abstain.
    assert pre_p.ok, pre_p.reason
