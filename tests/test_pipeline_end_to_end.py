"""End-to-end test: calibrate cameras via PnP (not using ground-truth
pose directly, the way the individual-module tests do) -> triangulate a
dart tip using those PnP-SOLVED calibrations -> score it. This is the
one test that actually proves Phases 2-4 compose correctly as a whole
pipeline, not just that each piece works in isolation against its own
ground truth.

Deliberately composes across two modules that stayed separate in the
2026-08-12 engines refactor: `calibrate_camera` (calibration-SOLVING,
genuinely shared, `opendarts.pipeline`) and `score_dart` (Apollo's own
scoring STRATEGY, `opendarts.engines.apollo.scoring`, moved out of
`opendarts.pipeline` that same day) -- this file's whole point is proving
the two compose correctly, so it imports from both real locations rather
than moving alongside either one."""
from __future__ import annotations

import numpy as np
import pytest

from tests.support.synthetic import make_camera_matrix, make_ring_camera, project_points
from opendarts.engines.apollo.scoring import (
    MAX_RAY_DISAGREEMENT_MM,
    MAX_RAY_DISAGREEMENT_MM_FALLBACK_PAIR,
    score_dart,
)
from opendarts.geometry.board import (
    DOUBLE_OUTER_RADIUS_MM,
    TREBLE_INNER_RADIUS_MM,
    TREBLE_OUTER_RADIUS_MM,
    polar_to_xy_mm,
    sector_center_angle_deg,
    wire_intersection_landmarks,
)
from opendarts.pipeline import CameraCalibration, calibrate_camera


WELL_SPREAD_QUAD = (
    "double_outer_20", "double_outer_6", "double_outer_3", "double_outer_9",
)
# Same clustered/occluded quad Verifier pass 2 already proved is a real
# PnP-conditioning risk (tests/test_pnp_calibration.py) -- reused here at
# the pipeline-composition level per Verifier pass 4's finding #2: the
# original pipeline tests only ever used the well-spread quad, so the
# real, demonstrated wrong-sector risk (finding #1, see
# MAX_RAY_DISAGREEMENT_MM in opendarts/engines/apollo/scoring.py) had
# zero coverage at this layer despite dedicated coverage existing one
# layer down.
CLUSTERED_QUAD = (
    "double_outer_20", "double_outer_1", "double_outer_18", "double_outer_4",
)


def _calibration_landmarks_for(true_cam, labels=WELL_SPREAD_QUAD):
    """The real trap: calibration.json only ever
    has ~4 genuinely independent points per camera, not the idealized 81
    -- this builds that realistic point budget from a chosen quad."""
    landmarks = wire_intersection_landmarks()
    four = [p for p in landmarks if p.label in labels]
    object_points = np.array([p.xyz for p in four], dtype=np.float64)
    image_points = project_points(true_cam, object_points)
    return object_points, image_points


@pytest.mark.parametrize(
    "sector_number,ring_name,expected_ring",
    [
        (20, "treble", "treble"),
        (5, "double_outer", "double"),
        (11, "single_outer_mid", "single_outer"),
    ],
)
def test_full_pipeline_calibrate_then_triangulate_then_score(
    sector_number, ring_name, expected_ring
):
    camera_matrix = make_camera_matrix(fov_deg=90.0)
    true_cams = [make_ring_camera(i, n_cameras=3, camera_matrix=camera_matrix) for i in range(3)]

    # Step 1: calibrate each camera via PnP using only the realistic
    # ~4-point quad -- NOT the ground-truth pose directly. This is the
    # part pure-triangulation-module tests skip (they hand triangulate()
    # the true pose); here the pose itself has to be recovered first,
    # same as the real system would have to do.
    calibrations: dict[int, CameraCalibration] = {}
    for i, true_cam in enumerate(true_cams):
        obj_pts, img_pts = _calibration_landmarks_for(true_cam)
        attempt = calibrate_camera(obj_pts, img_pts, camera_matrix, true_cam.dist_coeffs)
        assert attempt.ok, f"camera {i} failed to calibrate: {attempt.reason}"
        calibrations[i] = attempt.calibration

    # Step 2: place a "dart tip" at a known sector/ring, project it
    # through each (real, ground-truth) camera to get the pixel a real
    # tip detector would have found.
    angle = sector_center_angle_deg(sector_number)
    if ring_name == "treble":
        radius = (TREBLE_INNER_RADIUS_MM + TREBLE_OUTER_RADIUS_MM) / 2
    elif ring_name == "double_outer":
        radius = DOUBLE_OUTER_RADIUS_MM - 1.0
    else:
        radius = (TREBLE_OUTER_RADIUS_MM + 162.0) / 2  # single_outer band
    x, y = polar_to_xy_mm(radius, angle)
    true_point = (x, y, 0.0)

    tip_pixels: dict[int, tuple[float, float]] = {}
    for i, true_cam in enumerate(true_cams):
        import cv2

        pt = np.asarray(true_point, dtype=np.float64).reshape(1, 1, 3)
        px, _ = cv2.projectPoints(
            pt, true_cam.rvec, true_cam.tvec, true_cam.camera_matrix, true_cam.dist_coeffs
        )
        tip_pixels[i] = tuple(px.reshape(2))

    # Step 3: the actual thing this whole project is for -- score using
    # PnP-recovered calibrations (not ground truth), real triangulation
    # (not per-camera election).
    result = score_dart(tip_pixels, calibrations)
    assert result.ok, result.reason
    assert result.sector == str(sector_number), (
        f"expected sector {sector_number}, got {result.sector} "
        f"(board_xy={result.board_xy_mm}, true={true_point})"
    )
    assert result.ring == expected_ring
    assert result.n_cameras_used == 3
    recovered = np.array(result.board_xy_mm)
    true_xy = np.array(true_point[:2])
    err = float(np.linalg.norm(recovered - true_xy))
    # Bound set from actually measured data (this exact noiseless
    # scenario: ~1.19e-5mm) per Verifier pass 4 -- the original <1.0mm
    # bound was ~30,000-100,000x looser than reality, the same
    # too-loose-tolerance mistake caught (and supposedly fixed) twice
    # already tonight, sneaking back in a third time in this exact test.
    assert err < 0.01, f"end-to-end board-plane error too large: {err} mm"


def test_full_pipeline_with_realistic_noise_at_both_stages():
    """Noise in BOTH calibration landmark detection AND tip detection --
    the realistic combined scenario, not just one or the other."""
    rng = np.random.default_rng(seed=99)
    camera_matrix = make_camera_matrix(fov_deg=90.0)
    true_cams = [make_ring_camera(i, n_cameras=3, camera_matrix=camera_matrix) for i in range(3)]

    calibrations: dict[int, CameraCalibration] = {}
    for i, true_cam in enumerate(true_cams):
        obj_pts, img_pts = _calibration_landmarks_for(true_cam)
        noisy_img_pts = img_pts + rng.normal(0.0, 0.5, img_pts.shape)
        attempt = calibrate_camera(obj_pts, noisy_img_pts, camera_matrix, true_cam.dist_coeffs)
        assert attempt.ok, attempt.reason
        calibrations[i] = attempt.calibration

    true_point = (30.0, -40.0, 0.0)
    tip_pixels: dict[int, tuple[float, float]] = {}
    for i, true_cam in enumerate(true_cams):
        import cv2

        pt = np.asarray(true_point, dtype=np.float64).reshape(1, 1, 3)
        px, _ = cv2.projectPoints(
            pt, true_cam.rvec, true_cam.tvec, true_cam.camera_matrix, true_cam.dist_coeffs
        )
        noisy_px = px.reshape(2) + rng.normal(0.0, 0.5, 2)
        tip_pixels[i] = tuple(noisy_px)

    result = score_dart(tip_pixels, calibrations)
    assert result.ok, result.reason
    recovered = np.array(result.board_xy_mm)
    true_xy = np.array(true_point[:2])
    err = float(np.linalg.norm(recovered - true_xy))
    # Bound set from actually measured data (30-seed sweep of this exact
    # scenario: mean 0.56mm, max 1.42mm), not a round-number guess --
    # applying the same discipline Verifier passes 2/3 established after
    # catching too-loose tolerances twice already tonight. Independently
    # re-verified reproducible by Verifier pass 4.
    assert err < 3.0, f"combined-noise end-to-end error too large: {err} mm"
    assert result.max_ray_disagreement_mm is not None
    assert result.max_ray_disagreement_mm < 10.0  # well under the reject threshold


def test_pipeline_refuses_single_camera_explicitly():
    camera_matrix = make_camera_matrix(fov_deg=90.0)
    true_cam = make_ring_camera(0, n_cameras=3, camera_matrix=camera_matrix)
    obj_pts, img_pts = _calibration_landmarks_for(true_cam)
    attempt = calibrate_camera(obj_pts, img_pts, camera_matrix, true_cam.dist_coeffs)
    assert attempt.ok

    result = score_dart({0: (640.0, 360.0)}, {0: attempt.calibration})
    assert not result.ok
    assert "2" in result.reason


def test_calibrate_camera_preserves_failure_reason():
    """Regression test for Verifier pass 4: calibrate_camera() used to
    return a bare None on failure, silently discarding PnpResult.reason
    -- an asymmetry with score_dart(), which always preserves the
    wrapped TriangulationResult (and its .reason) even on failure."""
    camera_matrix = make_camera_matrix(fov_deg=90.0)
    true_cam = make_ring_camera(0, n_cameras=3, camera_matrix=camera_matrix)
    obj_pts, img_pts = _calibration_landmarks_for(true_cam)
    # Only 3 points -- solve_extrinsics requires >=4, see test_pnp_calibration.py.
    attempt = calibrate_camera(
        obj_pts[:3], img_pts[:3], camera_matrix, true_cam.dist_coeffs
    )
    assert not attempt.ok
    assert attempt.calibration is None
    assert "4" in attempt.reason  # the underlying PnpResult.reason, preserved
    assert attempt.pnp_result is not None
    assert not attempt.pnp_result.ok


def _run_clustered_quad_trials(disable_landmark_spread_gate: bool, n_seeds: int = 300):
    """Shared trial loop for the two tests below -- runs the exact same
    300-trial CLUSTERED_QUAD scenario Verifier pass 4 used to measure the
    original 15% residual risk, with the calibration-side landmark-spread
    gate (opendarts/pipeline.py's use of CameraCalibration.landmark_spread_ok,
    the correlated-bias gate) either active (default, real behavior)
    or forced off (disable_landmark_spread_gate=True, simulating the OLD
    ray-agreement-only behavior for a same-seeds, same-trials causal
    comparison) -- returns (n_trials, n_rejected, n_accepted_right,
    n_accepted_wrong).
    """
    import cv2

    camera_matrix = make_camera_matrix(fov_deg=90.0)
    true_point = np.array([20.0, -15.0, 0.0])

    n_trials = 0
    n_rejected = 0
    n_accepted_right = 0
    n_accepted_wrong = 0

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed=seed * 13)
        true_cams = [
            make_ring_camera(i, n_cameras=3, camera_matrix=camera_matrix) for i in range(3)
        ]
        calibrations: dict[int, CameraCalibration] = {}
        all_calibrated = True
        for i, true_cam in enumerate(true_cams):
            obj_pts, img_pts = _calibration_landmarks_for(true_cam, labels=CLUSTERED_QUAD)
            noisy_img_pts = img_pts + rng.normal(0.0, 0.5, img_pts.shape)
            attempt = calibrate_camera(obj_pts, noisy_img_pts, camera_matrix, true_cam.dist_coeffs)
            if not attempt.ok:
                all_calibrated = False
                break
            calib = attempt.calibration
            if disable_landmark_spread_gate:
                # Simulates the pre-fix world: score_dart() only rejects on
                # landmark_spread_ok is False, so forcing it to None
                # (rather than monkeypatching the module constant)
                # reproduces the OLD ray-agreement-only decision path
                # exactly, on the same seeds, for a real causal comparison.
                calib.landmark_spread_ok = None
            calibrations[i] = calib
        if not all_calibrated:
            continue
        n_trials += 1

        tip_pixels: dict[int, tuple[float, float]] = {}
        for i, true_cam in enumerate(true_cams):
            pt = true_point.reshape(1, 1, 3)
            px, _ = cv2.projectPoints(
                pt, true_cam.rvec, true_cam.tvec, true_cam.camera_matrix, true_cam.dist_coeffs
            )
            noisy_px = px.reshape(2) + rng.normal(0.0, 0.5, 2)
            tip_pixels[i] = tuple(noisy_px)

        result = score_dart(tip_pixels, calibrations)
        if not result.ok:
            n_rejected += 1
            continue
        recovered = np.array(result.board_xy_mm)
        if np.linalg.norm(recovered - true_point[:2]) < 5.0:
            n_accepted_right += 1
        else:
            n_accepted_wrong += 1

    return n_trials, n_rejected, n_accepted_right, n_accepted_wrong


def test_ray_agreement_alone_reproduces_the_known_residual_risk():
    """Reproduces Verifier pass 4's original finding (same 300 seeds,
    same CLUSTERED_QUAD scenario) with the calibration-side landmark-
    spread gate forced off -- i.e. simulates the pipeline as it would
    behave WITHOUT that gate. This exists so the next test's improvement
    claim is a real, same-seeds causal comparison, not two numbers
    measured at different times that might differ for unrelated reasons.

    **Baseline updated 2026-08-12** (real-throw investigation): this
    used to assert a ~15% residual
    wrong-and-accepted rate (Verifier pass 4's original number, ray
    agreement on the FULL 3-ray set only). Adding the 2-of-3 RANSAC
    fallback to opendarts.engines.apollo.scoring's score_dart() (real-throw fix, same
    session) changed what "ray agreement alone" actually produces here:
    with the gate off, this scenario's measured wrong-and-accepted rate
    became ~67% (200/300), not ~15% -- because the fallback finds a
    spuriously-agreeing 2-of-3 pair far more often than the full 3-ray
    set spuriously agreed. **This is not a production regression** --
    opendarts.engines.apollo.scoring's real score_dart() ALWAYS runs the landmark-
    spread gate before the RANSAC fallback ever executes (see
    test_landmark_spread_gate_closes_the_correlated_bias_gap_ray_agreement_missed
    below, which uses the gate ON -- the actual code path -- and still
    measures 0% wrong-and-accepted).

    **Baseline updated AGAIN, same day** (the coplanar 4-point PnP
    investigation): `opendarts/calibration/pnp.py`'s
    `solve_extrinsics()` now skips RANSAC at exactly 4 points (measured
    to be actively worse than plain solvePnP there, see that module's own
    dated comment for the real numbers) -- this scenario's real 4-point
    CLUSTERED_QUAD calibration now produces measurably MORE ACCURATE,
    more mutually-consistent per-camera poses, which moves BOTH numbers
    here again: reject_rate ~23% -> ~1.7% (5/300, fewer trials disagree
    enough to hit the outright reject threshold at all now) and
    residual_wrong_rate ~67% -> ~46.3% (139/300, a real, additional
    improvement from the same fix, not just noise). Re-measured directly
    (not guessed) before updating these bounds, per this project's own
    "measure the real number" discipline. This test's own point still
    holds either way: WITHOUT the landmark-spread gate, a real, large
    residual wrong-and-accepted rate remains in this adversarial
    scenario -- see the next test for why the gate (the actual production
    code path) closes it.
    """
    n_trials, n_rejected, n_right, n_wrong = _run_clustered_quad_trials(
        disable_landmark_spread_gate=True
    )
    assert n_trials > 100, f"too few valid trials to measure reliably: {n_trials}"
    reject_rate = n_rejected / n_trials
    residual_wrong_rate = n_wrong / n_trials
    # Real numbers measured 2026-08-12 with the 2-of-3 RANSAC fallback
    # active, solve_extrinsics()'s RANSAC-at-4-points fix active, and the
    # landmark-spread gate forced off (see docstring): reject_rate ~1.7%,
    # wrong_rate ~46.3%. Wide-ish tolerance bands (not exact equality) to
    # absorb reasonable RNG/library-version churn without silently hiding
    # a real regression in either direction.
    assert 0.0 < reject_rate < 0.10, f"reject rate drifted from the measured ~1.7%: {reject_rate:.1%}"
    assert 0.30 < residual_wrong_rate < 0.65, (
        f"expected to reproduce the measured ~46.3% residual "
        f"wrong-and-accepted rate (WITH the RANSAC fallback and the "
        f"RANSAC-at-4-points fix, gate OFF), got {residual_wrong_rate:.1%} "
        f"-- if ray-agreement+fallback behavior changed, re-verify this "
        f"baseline rather than loosening the bound to hide a real change"
    )


def test_landmark_spread_gate_closes_the_correlated_bias_gap_ray_agreement_missed():
    """Correlated calibration bias: ray
    agreement (MAX_RAY_DISAGREEMENT_MM) structurally cannot catch
    CORRELATED calibration bias across cameras -- a similarly-clustered
    landmark quad on every camera produces similarly-biased poses whose
    rays genuinely agree with each other on a point that's still wrong.
    Verifier pass 4 originally measured this leaves a real ~15% residual
    wrong-and-confidently-accepted rate with ray agreement alone
    (reproduced, same seeds, by
    test_ray_agreement_alone_reproduces_the_known_residual_risk above --
    **that test's own baseline was updated 2026-08-12** when the 2-of-3
    RANSAC fallback was added to score_dart(): with the gate off, the
    fallback makes the "ray agreement alone" world measurably WORSE
    (~67% wrong-and-accepted, not ~15%) since it now finds spuriously-
    agreeing 2-of-3 pairs far more readily than the full 3-ray set
    spuriously agreed. See that test's docstring for the full
    explanation -- the short version: this makes the gate THIS test
    covers even more clearly load-bearing, not less).

    The fix: opendarts/calibration/pnp.py's landmark_hull_area_fraction
    computes a point-spread/conditioning signal on the INPUT landmark
    points themselves, before PnP even solves -- independent of what any
    other camera reports, so it isn't fooled by cameras agreeing with each
    other on a jointly-wrong answer. score_dart() now rejects outright if
    any contributing camera's calibration had a poorly-spread quad
    (CameraCalibration.landmark_spread_ok is False), BEFORE triangulation
    (including the RANSAC 2-of-3 fallback) even runs.

    Measured result on the exact same 300-trial CLUSTERED_QUAD scenario:
    every trial's landmark quad is (by construction) the same clustered
    quad on every camera, so this gate rejects essentially all of them --
    residual wrong-and-accepted rate is 0% (0/300) with the gate on,
    regardless of whether the RANSAC fallback exists (it never gets a
    chance to run). This is a real, same-seeds causal result, not a
    coincidence: the ONLY difference between this test and the one above
    is whether the gate is active.

    Honest limitation, not glossed over: this closes the SPECIFIC
    correlated-bias mechanism this project has concrete evidence for
    (coplanar/clustered-quad pose ambiguity). It is a point-spread
    check, not a general-purpose
    "is this calibration accurate" check -- a well-spread quad that is
    biased for some OTHER shared reason (e.g. unmodeled shared lens
    distortion, wrong assumed intrinsics) would look fine to this signal.
    A probe of one such alternative mechanism (shared systematic pixel
    offset on an otherwise well-spread quad, biasing sizes up to 25px)
    found no case where it evaded BOTH ray-agreement and this gate --
    below ~12px the bias was too small to move the pose meaningfully
    (accepted-and-right), above ~12px ray-agreement alone already caught
    it -- but this is one probed mechanism, not a proof that no
    correlated-bias mechanism can ever evade both signals.
    """
    n_trials, n_rejected, n_right, n_wrong = _run_clustered_quad_trials(
        disable_landmark_spread_gate=False
    )
    assert n_trials > 100, f"too few valid trials to measure reliably: {n_trials}"
    reject_rate = n_rejected / n_trials
    residual_wrong_rate = n_wrong / n_trials

    # With the gate active, this specific scenario (every camera sees the
    # SAME clustered quad) should be caught almost universally -- measured
    # 300/300 when this test was built. A small tolerance (not exactly
    # asserting ==0) guards against reasonable RNG/library-version churn
    # without silently accepting a real regression.
    assert reject_rate > 0.95, (
        f"reject rate dropped well below the measured ~100% -- the "
        f"landmark-spread gate may have regressed: {reject_rate:.1%}"
    )
    assert residual_wrong_rate < 0.03, (
        f"residual wrong-and-accepted rate should be ~0 with the "
        f"landmark-spread gate active (measured 0/300 when built), got "
        f"{residual_wrong_rate:.1%} -- if this rises meaningfully above "
        f"noise, the gate has a real gap and correlated bias is NOT "
        f"handled, don't loosen this assertion to hide that"
    )


def test_landmark_spread_gate_does_not_claim_to_catch_every_bias_mechanism():
    """Honesty check, not a proof of completeness: landmark_spread_ok is a
    point-SPREAD check, not a general calibration-accuracy check -- a
    well-spread quad biased for some OTHER shared reason (e.g. unmodeled
    lens distortion) would look fine to it. Probed one such alternative
    mechanism -- a shared systematic pixel offset (same direction/magnitude
    on every landmark, every camera) applied to the WELL_SPREAD_QUAD -- to
    see whether it evades BOTH ray-agreement and the new gate. It doesn't,
    across the range tested: small shared offsets (<=8px) don't move the
    pose enough to matter (accepted-and-right); larger ones (>=12px)
    already trip MAX_RAY_DISAGREEMENT_MM on their own. No accepted-and-
    wrong case was found in this sweep. This is evidence about ONE probed
    mechanism, explicitly not a claim that no correlated-bias mechanism
    can ever evade both signals.
    """
    import cv2

    camera_matrix = make_camera_matrix(fov_deg=90.0)
    true_point = np.array([20.0, -15.0, 0.0])

    for bias_px in (4.0, 8.0, 12.0, 16.0, 20.0, 25.0):
        n_trials = n_rejected = n_wrong = 0
        for seed in range(60):
            rng = np.random.default_rng(seed=seed * 13)
            true_cams = [
                make_ring_camera(i, n_cameras=3, camera_matrix=camera_matrix) for i in range(3)
            ]
            calibrations: dict[int, CameraCalibration] = {}
            ok = True
            for i, true_cam in enumerate(true_cams):
                obj_pts, img_pts = _calibration_landmarks_for(true_cam, labels=WELL_SPREAD_QUAD)
                bias = np.array([bias_px, -0.8 * bias_px])
                noisy_img_pts = img_pts + bias + rng.normal(0.0, 0.5, img_pts.shape)
                attempt = calibrate_camera(
                    obj_pts, noisy_img_pts, camera_matrix, true_cam.dist_coeffs
                )
                if not attempt.ok:
                    ok = False
                    break
                calibrations[i] = attempt.calibration
            if not ok:
                continue
            n_trials += 1
            tip_pixels: dict[int, tuple[float, float]] = {}
            for i, true_cam in enumerate(true_cams):
                pt = true_point.reshape(1, 1, 3)
                px, _ = cv2.projectPoints(
                    pt, true_cam.rvec, true_cam.tvec, true_cam.camera_matrix, true_cam.dist_coeffs
                )
                noisy_px = px.reshape(2) + rng.normal(0.0, 0.5, 2)
                tip_pixels[i] = tuple(noisy_px)
            result = score_dart(tip_pixels, calibrations)
            if not result.ok:
                n_rejected += 1
                continue
            recovered = np.array(result.board_xy_mm)
            if np.linalg.norm(recovered - true_point[:2]) >= 5.0:
                n_wrong += 1
        assert n_trials > 30, f"too few valid trials at bias={bias_px}px: {n_trials}"
        # The claim under test: no accepted-and-wrong cases in this probe,
        # at any bias magnitude tried.
        assert n_wrong == 0, (
            f"found {n_wrong}/{n_trials} accepted-and-wrong cases at "
            f"bias={bias_px}px -- this WOULD be a real gap in both "
            f"signals, do not suppress this failure, investigate it"
        )


def test_landmark_spread_gate_rejects_when_only_one_of_three_cameras_is_clustered():
    """Verifier pass 5 (2026-08-12) finding #4: every existing test of the
    landmark-spread gate used the SAME quad (all-well-spread or
    all-clustered) on every camera in a trial -- no coverage of the
    realistic MIXED case (e.g. one camera partially occluded, the other
    two clear), which is exactly the "only tests the uniform/easy case"
    mistake already found once at this same layer.
    The gate's `any(...)` logic was correct
    even before this test existed (verified by the pass-5 reviewer's own
    probe: 300/300 rejected, 0 wrong-and-accepted with a 1-of-3-clustered
    mix) -- this closes the missing coverage, not a bug fix.
    """
    import cv2

    camera_matrix = make_camera_matrix(fov_deg=90.0)
    true_point = np.array([20.0, -15.0, 0.0])

    n_trials = 0
    n_rejected = 0
    n_wrong = 0
    for seed in range(150):
        rng = np.random.default_rng(seed=seed * 13)
        true_cams = [
            make_ring_camera(i, n_cameras=3, camera_matrix=camera_matrix) for i in range(3)
        ]
        # Camera 0 sees only the clustered/occluded quad; cameras 1 and 2
        # see the well-spread quad -- one partially-occluded camera among
        # three, the routine case.
        quads = [CLUSTERED_QUAD, WELL_SPREAD_QUAD, WELL_SPREAD_QUAD]
        calibrations: dict[int, CameraCalibration] = {}
        all_calibrated = True
        for i, true_cam in enumerate(true_cams):
            obj_pts, img_pts = _calibration_landmarks_for(true_cam, labels=quads[i])
            noisy_img_pts = img_pts + rng.normal(0.0, 0.5, img_pts.shape)
            attempt = calibrate_camera(obj_pts, noisy_img_pts, camera_matrix, true_cam.dist_coeffs)
            if not attempt.ok:
                all_calibrated = False
                break
            calibrations[i] = attempt.calibration
        if not all_calibrated:
            continue
        n_trials += 1

        tip_pixels: dict[int, tuple[float, float]] = {}
        for i, true_cam in enumerate(true_cams):
            pt = true_point.reshape(1, 1, 3)
            px, _ = cv2.projectPoints(
                pt, true_cam.rvec, true_cam.tvec, true_cam.camera_matrix, true_cam.dist_coeffs
            )
            noisy_px = px.reshape(2) + rng.normal(0.0, 0.5, 2)
            tip_pixels[i] = tuple(noisy_px)

        result = score_dart(tip_pixels, calibrations)
        if not result.ok:
            n_rejected += 1
            # Confirm it was actually the landmark-spread gate that fired,
            # not some other rejection path -- camera 0 is the only one
            # with a poorly-spread quad in this scenario.
            assert "camera(s) [0]" in result.reason, result.reason
            continue
        recovered = np.array(result.board_xy_mm)
        if np.linalg.norm(recovered - true_point[:2]) >= 5.0:
            n_wrong += 1

    assert n_trials > 100, f"too few valid trials: {n_trials}"
    reject_rate = n_rejected / n_trials
    assert reject_rate > 0.95, (
        f"expected the single poorly-spread camera to reliably trigger "
        f"the gate: {reject_rate:.1%}"
    )
    assert n_wrong == 0, f"found {n_wrong} wrong-and-accepted mixed-camera cases"


def _clean_well_spread_calibrations(camera_matrix, true_cams, rng):
    """Clean calibration on all 3 cameras -- NOT the failure mode under
    test in the RANSAC tests below (those are about a bad TIP PIXEL, not
    a bad calibration)."""
    calibrations: dict[int, CameraCalibration] = {}
    for i, true_cam in enumerate(true_cams):
        obj_pts, img_pts = _calibration_landmarks_for(true_cam, labels=WELL_SPREAD_QUAD)
        noisy_img_pts = img_pts + rng.normal(0.0, 0.5, img_pts.shape)
        attempt = calibrate_camera(obj_pts, noisy_img_pts, camera_matrix, true_cam.dist_coeffs)
        assert attempt.ok, attempt.reason
        calibrations[i] = attempt.calibration
    return calibrations


def _project_true_point(true_cams, true_point):
    import cv2

    tip_pixels: dict[int, tuple[float, float]] = {}
    for i, true_cam in enumerate(true_cams):
        pt = true_point.reshape(1, 1, 3)
        px, _ = cv2.projectPoints(
            pt, true_cam.rvec, true_cam.tvec, true_cam.camera_matrix, true_cam.dist_coeffs
        )
        tip_pixels[i] = tuple(px.reshape(2))
    return tip_pixels


def test_ransac_2of3_recovers_a_good_score_when_one_camera_ray_is_bad():
    """The actual real-throw fix: a single bad-camera
    tip detection used to reject the WHOLE throw even when the other two
    cameras' rays agreed closely (measured on that real throw: cam0
    disagreed by 27-31mm while cam1+cam2 alone agreed to 2.5mm).
    Reproduces the mechanism synthetically -- clean, well-spread
    calibration on all 3 cameras (this is deliberately NOT about the
    landmark-spread gate, see the CLUSTERED_QUAD tests above for that),
    one camera's TIP PIXEL corrupted by a large single-camera offset
    (simulating a real detect_tip() failure, not a calibration problem)
    -- and checks score_dart() now recovers a correct, accepted result
    via the 2-of-3 fallback instead of rejecting outright."""
    camera_matrix = make_camera_matrix(fov_deg=90.0)
    true_cams = [make_ring_camera(i, n_cameras=3, camera_matrix=camera_matrix) for i in range(3)]
    rng = np.random.default_rng(seed=2026)
    calibrations = _clean_well_spread_calibrations(camera_matrix, true_cams, rng)

    true_point = np.array([20.0, -15.0, 0.0])
    tip_pixels = _project_true_point(true_cams, true_point)
    # Corrupt camera 0's tip pixel by a large offset -- simulates
    # detect_tip() picking the wrong end of a split shaft component (the
    # real failure mode this fallback exists for).
    bad_x, bad_y = tip_pixels[0]
    tip_pixels[0] = (bad_x + 80.0, bad_y - 60.0)

    result = score_dart(tip_pixels, calibrations)
    assert result.ok, result.reason
    assert result.n_cameras_used == 2
    assert result.cameras_used == (1, 2)
    assert result.outlier_camera == 0
    assert result.max_ray_disagreement_mm is not None
    assert result.max_ray_disagreement_mm <= MAX_RAY_DISAGREEMENT_MM_FALLBACK_PAIR
    assert "RANSAC fallback" in result.reason

    recovered = np.array(result.board_xy_mm)
    err = float(np.linalg.norm(recovered - true_point[:2]))
    assert err < 5.0, f"recovered point too far from truth via fallback: {err}mm"


def test_ransac_fallback_does_not_engage_when_full_set_already_agrees():
    """The full 3-ray set is tried FIRST and preferred when it agrees --
    the fallback must not change behavior in the common, healthy case.
    Explicit regression coverage for cameras_used/outlier_camera on top
    of the existing n_cameras_used==3 assertion elsewhere in this file."""
    camera_matrix = make_camera_matrix(fov_deg=90.0)
    true_cams = [make_ring_camera(i, n_cameras=3, camera_matrix=camera_matrix) for i in range(3)]
    rng = np.random.default_rng(seed=2028)
    calibrations = _clean_well_spread_calibrations(camera_matrix, true_cams, rng)

    true_point = np.array([20.0, -15.0, 0.0])
    tip_pixels = _project_true_point(true_cams, true_point)

    result = score_dart(tip_pixels, calibrations)
    assert result.ok, result.reason
    assert result.n_cameras_used == 3
    assert result.cameras_used == (0, 1, 2)
    assert result.outlier_camera is None
    assert result.reason == ""


def test_ransac_fallback_still_rejects_when_no_camera_pair_agrees():
    """Not an unconditional accept-more-often change: when NO 2-camera
    pair agrees within the (stricter) fallback threshold either,
    score_dart() must still reject outright. Offsets below were checked
    directly to produce all-three-pairs-
    disagree-by-more-than-the-fallback-threshold on this exact seed/
    scenario -- not assumed to generalize to arbitrary large offsets
    (some large-offset combinations coincidentally leave one pair
    agreeing, by the same real geometry this fallback is designed to
    exploit when it's actually one bad ray)."""
    camera_matrix = make_camera_matrix(fov_deg=90.0)
    true_cams = [make_ring_camera(i, n_cameras=3, camera_matrix=camera_matrix) for i in range(3)]
    rng = np.random.default_rng(seed=2026)
    calibrations = _clean_well_spread_calibrations(camera_matrix, true_cams, rng)

    true_point = np.array([20.0, -15.0, 0.0])
    tip_pixels = _project_true_point(true_cams, true_point)
    offsets = {0: (200.0, -150.0), 1: (-180.0, 50.0), 2: (150.0, 200.0)}
    for i, (dx, dy) in offsets.items():
        x, y = tip_pixels[i]
        tip_pixels[i] = (x + dx, y + dy)

    result = score_dart(tip_pixels, calibrations)
    assert not result.ok
    assert "no 2-of-3 camera pair agreed" in result.reason, result.reason


# --------------------------------------------------------------------------
# alt_tip_pixels / the primary-vs-alt combination search (2026-08-12, real
# incident: throw throw_1786580447119 -- see
# score_dart()'s own docstring for the full write-up). Real numbers from
# actually re-running the fixed pipeline against that exact package:
# max_ray_disagreement_mm went from 32.4mm (rejected) to 1.79mm (accepted,
# cameras 1 and 2's alt candidates both used), recovering board_xy within
# 2.48mm of AD's own ground-truth tip for that throw.
# --------------------------------------------------------------------------


def test_alt_candidate_search_prefers_full_set_over_fallback_pair_when_alt_agrees():
    """The real incident's simpler shape: ONE camera's primary tip pixel
    is corrupted (simulating a wrong monocular tip-end call) but its
    alt_tip_pixels entry holds the correct pixel. WITHOUT alt_tip_pixels,
    score_dart() already recovers via the existing 2-of-3 RANSAC fallback
    (n_cameras_used=2, camera 0 excluded as the outlier) -- this is NOT
    new behavior on its own. WITH alt_tip_pixels, the combination search
    finds that substituting camera 0's alt candidate makes the FULL
    3-camera set agree, which score_dart() prefers over any 2-camera
    fallback (more rays is more information, the same preference the
    single-combo code already had) -- a real, measurable improvement over
    the fallback-only behavior, not just "also works"."""
    camera_matrix = make_camera_matrix(fov_deg=90.0)
    true_cams = [make_ring_camera(i, n_cameras=3, camera_matrix=camera_matrix) for i in range(3)]
    rng = np.random.default_rng(seed=2026)
    calibrations = _clean_well_spread_calibrations(camera_matrix, true_cams, rng)

    true_point = np.array([20.0, -15.0, 0.0])
    tip_pixels = _project_true_point(true_cams, true_point)
    correct_cam0_pixel = tip_pixels[0]
    x0, y0 = correct_cam0_pixel
    tip_pixels[0] = (x0 + 80.0, y0 - 60.0)  # the wrong end, monocularly
    alt_tip_pixels = {0: correct_cam0_pixel}  # the correct end, as the alt

    without_alt = score_dart(tip_pixels, calibrations)
    assert without_alt.ok, without_alt.reason
    assert without_alt.n_cameras_used == 2
    assert without_alt.cameras_used == (1, 2)
    assert without_alt.outlier_camera == 0
    assert without_alt.alt_candidates_used is None

    with_alt = score_dart(tip_pixels, calibrations, alt_tip_pixels=alt_tip_pixels)
    assert with_alt.ok, with_alt.reason
    assert with_alt.n_cameras_used == 3
    assert with_alt.cameras_used == (0, 1, 2)
    assert with_alt.outlier_camera is None
    assert with_alt.alt_candidates_used == (0,)
    assert "camera(s) [0]" in with_alt.reason and "alternate ambiguous tip candidate" in with_alt.reason

    recovered = np.array(with_alt.board_xy_mm)
    err = float(np.linalg.norm(recovered - true_point[:2]))
    # Measured on this exact scenario: ~0.077mm (all 3 cameras agree
    # almost perfectly once camera 0's correct pixel is substituted in --
    # this is noiseless synthetic geometry, so near-zero is expected, not
    # a round-number guess).
    assert err < 1.0, f"recovered point too far from truth via alt search: {err}mm"


def test_alt_candidate_search_rescues_a_throw_the_primary_guess_alone_rejects():
    """The real incident's actual shape (throw
    throw_1786580447119 had TWO cameras -- 1 and 2 -- with an alt
    candidate, not just one): TWO cameras' primary tip pixels are
    corrupted, each with its alt_tip_pixels entry holding the correct
    pixel. Without alt candidates, NEITHER the full 3-ray set NOR any
    2-of-3 fallback pair agrees well enough (two bad rays, at most one
    good one) -- score_dart() rejects outright, matching what really
    happened live. With alt candidates, the combination search finds the
    combo using both cameras' alt (correct) pixels, which agrees closely
    across the full set -- a genuine reject-to-accept recovery, not just
    a better version of an already-accepted result."""
    camera_matrix = make_camera_matrix(fov_deg=90.0)
    true_cams = [make_ring_camera(i, n_cameras=3, camera_matrix=camera_matrix) for i in range(3)]
    rng = np.random.default_rng(seed=2026)
    calibrations = _clean_well_spread_calibrations(camera_matrix, true_cams, rng)

    true_point = np.array([20.0, -15.0, 0.0])
    tip_pixels = _project_true_point(true_cams, true_point)
    alt_tip_pixels = {}
    for cam, (dx, dy) in {0: (80.0, -60.0), 1: (-80.0, 48.0)}.items():
        alt_tip_pixels[cam] = tip_pixels[cam]  # correct pixel, as the alt
        x, y = tip_pixels[cam]
        tip_pixels[cam] = (x + dx, y + dy)  # wrong end, as the primary

    without_alt = score_dart(tip_pixels, calibrations)
    assert not without_alt.ok, (
        "expected the two-bad-camera scenario to reject outright without "
        "alt candidates -- if this fails, the synthetic offsets no longer "
        "demonstrate the reject-without/accept-with case this test exists "
        "to check"
    )

    with_alt = score_dart(tip_pixels, calibrations, alt_tip_pixels=alt_tip_pixels)
    assert with_alt.ok, with_alt.reason
    assert with_alt.n_cameras_used == 3
    assert with_alt.cameras_used == (0, 1, 2)
    assert with_alt.alt_candidates_used == (0, 1)
    assert with_alt.max_ray_disagreement_mm is not None
    assert with_alt.max_ray_disagreement_mm < MAX_RAY_DISAGREEMENT_MM

    recovered = np.array(with_alt.board_xy_mm)
    err = float(np.linalg.norm(recovered - true_point[:2]))
    assert err < 1.0, f"recovered point too far from truth via alt search: {err}mm"


def test_alt_candidate_search_still_rejects_when_no_combination_agrees_well_enough():
    """Not an unconditional accept-more-often change (mirrors
    test_ransac_fallback_still_rejects_when_no_camera_pair_agreed's own
    honesty check, at the alt-candidate-search layer): when a camera's
    ALT candidate is itself just another wrong guess (not the real tip --
    the realistic case for an ambiguous detection that truly can't be
    resolved from a single image), score_dart() must still reject, with
    the SAME reason format as if alt_tip_pixels had never been passed at
    all (this is the "degrade exactly as today" requirement) -- proving
    the combination search cannot manufacture a false accept out of two
    wrong candidates."""
    camera_matrix = make_camera_matrix(fov_deg=90.0)
    true_cams = [make_ring_camera(i, n_cameras=3, camera_matrix=camera_matrix) for i in range(3)]
    rng = np.random.default_rng(seed=2026)
    calibrations = _clean_well_spread_calibrations(camera_matrix, true_cams, rng)

    true_point = np.array([20.0, -15.0, 0.0])
    tip_pixels = _project_true_point(true_cams, true_point)
    alt_tip_pixels = {}
    for cam, (dx, dy) in {0: (80.0, -60.0), 1: (-80.0, 48.0)}.items():
        x, y = tip_pixels[cam]
        tip_pixels[cam] = (x + dx, y + dy)  # wrong end, as the primary
        # alt is ALSO wrong -- a different bad guess, not the real tip.
        alt_tip_pixels[cam] = (x - 55.0, y + 65.0)

    without_alt = score_dart(tip_pixels, calibrations)
    with_alt = score_dart(tip_pixels, calibrations, alt_tip_pixels=alt_tip_pixels)
    assert not without_alt.ok
    assert not with_alt.ok
    assert with_alt.alt_candidates_used is None
    # Bit-for-bit the same rejection score_dart() would have produced
    # with no alt_tip_pixels at all -- the actual "degrade exactly as
    # today" guarantee, not just "also rejects".
    assert with_alt.reason == without_alt.reason
    assert with_alt.max_ray_disagreement_mm == without_alt.max_ray_disagreement_mm
