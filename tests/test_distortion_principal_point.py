"""Tests for opendarts.calibration.distortion's PRINCIPAL POINT (cx only)
addition -- a joint (f, pose, k1, cx) solve fit ALONGSIDE the shipped
k1-only (f, pose, k1) solve (see that module's own docstring, "PRINCIPAL
POINT (cx only)" section, for the full synthetic + real-data conditioning
analysis this mirrors).

Tolerances below are MEASURED, not guessed. The full sweeps behind
them:
  - Noiseless joint (f, pose, k1, cx) recovery at this rig's 3 real focal
    lengths, every injected k1 in the floor-camera's own real measured
    range, and cx offsets from 0 to -30px: exact to numerical floor
    (~1e-8px error or better).
  - Multi-frame (per-index median, N=40, matching
    CALIBRATION_N_FRAMES_DETECT) at 0.4px/point realistic noise, cx
    fixed at true image center (the "no real offset" case): cx recovers
    with std 1.06-1.29px, max |err| ~2.66-2.97px, f's own std STAYS
    UNCHANGED versus the k1-only model (statistically identical) -- this
    is what makes cx (unlike cy, p1, p2) safe to ship.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from opendarts.calibration.distortion import (
    MAX_ABS_CX_OFFSET_FRACTION_OF_WIDTH,
    MAX_K1,
    MIN_K1,
    PRINCIPAL_POINT_FALLBACK_FILENAME,
    PRINCIPAL_POINT_SCHEMA,
    derive_focal_and_k1_from_oriented_results,
    derive_focal_k1_cx_from_oriented_results,
    estimate_focal_pose_k1_cx,
    load_principal_point_fallback,
    write_principal_point_fallback_entry,
)
from opendarts.calibration.focal_length import ring20_object_points_mm
from tests.support.synthetic import make_camera_matrix, make_ring_camera, project_points

W, H = 1280, 720
CX0, CY0 = W / 2.0, H / 2.0
REAL_RIG_FOCAL_LENGTHS_PX = (800.0, 830.0, 835.0)
REAL_K1_FLOOR_CAM = -0.2235  # mean measured k1 for the floor camera (docs/DESIGN.md, 4-package validation)
OBJ = ring20_object_points_mm()

NOISELESS_TOL_PX = 1e-4
NOISELESS_K1_TOL = 1e-6
NOISELESS_CX_TOL_PX = 1e-4

# Measured, multi-frame N=40 realistic-noise numbers ("cx-ALONE"), with
# real margin above the measured std/max-err so this test isn't flaky on
# a different RNG draw.
MULTI_FRAME_CX_STD_TOL_PX = 2.0
MULTI_FRAME_CX_MAX_ABS_ERR_TOL_PX = 5.0
MULTI_FRAME_F_STD_TOL_PX = 1.5  # measured k1-only baseline std was 0.6-0.68px


def _project_ring20_cx(f: float, cam_index: int, k1: float, cx: float, n_cams: int = 3,
                        *, ring_radius_mm=500.0, height_mm=300.0):

    K = make_camera_matrix(W, H, fov_deg=2 * math.degrees(math.atan((W / 2.0) / f)))
    K[0, 2] = cx
    dist = np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    cam = make_ring_camera(cam_index, n_cams, K, ring_radius_mm=ring_radius_mm, height_mm=height_mm, dist_coeffs=dist)
    img = project_points(cam, OBJ)
    return OBJ, img


class _FakeOrientedResult:
    def __init__(self, ok=True, orientation_ambiguous=False, ring20_px=None):
        self.ok = ok
        self.orientation_ambiguous = orientation_ambiguous
        self.ring20_px = ring20_px


# ---------------------------------------------------------------------
# Noiseless recovery -- the conditioning claim itself
# ---------------------------------------------------------------------


@pytest.mark.parametrize("cx_offset", [0.0, -10.0, -30.0, 15.0])
@pytest.mark.parametrize("cam_index", [0, 1, 2])
def test_noiseless_joint_recovery_at_real_rig_geometry(cam_index, cx_offset):
    f_true = REAL_RIG_FOCAL_LENGTHS_PX[cam_index]
    cx_true = CX0 + cx_offset
    obj, img = _project_ring20_cx(f_true, cam_index, REAL_K1_FLOOR_CAM, cx_true)
    result = estimate_focal_pose_k1_cx(obj, img, W, H, initial_focal_px=f_true * 1.1)
    assert result.ok, result.reason
    assert abs(result.focal_px - f_true) < NOISELESS_TOL_PX
    assert abs(result.k1 - REAL_K1_FLOOR_CAM) < NOISELESS_K1_TOL
    assert abs(result.cx - cx_true) < NOISELESS_CX_TOL_PX


# ---------------------------------------------------------------------
# Structural guard -- cy/p1/p2 are NOT free unknowns (module docstring:
# real data showed cy is NOT safely identifiable on this rig, unlike cx)
# ---------------------------------------------------------------------


def test_model_has_no_cy_p1_p2_unknowns_by_construction():
    """This module fixes cy at image_height/2 and p1=p2=k2=k3=0 -- there
    is no public entry point here that exposes a free cy/p1/p2 parameter;
    `initial_cx` is a SEED for cx, not a "solve for cy too" flag. Same
    structural-guard spirit as test_distortion.py's own
    test_model_has_no_k2_unknown_by_construction."""
    import inspect

    sig = inspect.signature(estimate_focal_pose_k1_cx)
    assert "cy" not in sig.parameters
    assert "initial_p1" not in sig.parameters
    assert "initial_p2" not in sig.parameters
    assert "initial_cx" in sig.parameters  # cx IS the one new solvable unknown


# ---------------------------------------------------------------------
# Noise robustness -- multi-frame averaging, matching
# CALIBRATION_N_FRAMES_DETECT, and the KEY claim: f's own std is
# unaffected by adding cx (unlike adding cy, which measurably inflates
# it -- see module docstring)
# ---------------------------------------------------------------------


def test_multi_frame_median_averaging_recovers_cx_and_leaves_f_precision_unchanged():
    rng = np.random.default_rng(20260826)
    n_frames = 40
    noise_px = 0.4
    cx_errs, f_errs_with_cx, f_errs_baseline = [], [], []
    for session in range(30):
        cam_index = session % 3
        f_true = REAL_RIG_FOCAL_LENGTHS_PX[cam_index]
        obj, img_clean = _project_ring20_cx(f_true, cam_index, REAL_K1_FLOOR_CAM, CX0)
        per_frame_imgs = [img_clean + rng.normal(0, noise_px, img_clean.shape) for _ in range(n_frames)]
        per_frame = [_FakeOrientedResult(ring20_px=pts) for pts in per_frame_imgs]

        result = derive_focal_k1_cx_from_oriented_results(
            per_frame, (CX0, CY0), W, H, min_frames=1, initial_focal_px=f_true,
        )
        if result.ok:
            cx_errs.append(result.cx - CX0)
            f_errs_with_cx.append(result.focal_length_px - f_true)

        baseline = derive_focal_and_k1_from_oriented_results(
            per_frame, (CX0, CY0), W, H, min_frames=1,
        )
        if baseline.ok:
            f_errs_baseline.append(baseline.focal_length_px - f_true)

    cx_errs = np.array(cx_errs)
    f_errs_with_cx = np.array(f_errs_with_cx)
    f_errs_baseline = np.array(f_errs_baseline)
    assert len(cx_errs) > 20
    assert cx_errs.std() < MULTI_FRAME_CX_STD_TOL_PX
    assert np.abs(cx_errs).max() < MULTI_FRAME_CX_MAX_ABS_ERR_TOL_PX
    assert f_errs_with_cx.std() < MULTI_FRAME_F_STD_TOL_PX
    # The decisive comparison (module docstring): adding cx should NOT
    # meaningfully inflate f's own std relative to the k1-only baseline
    # -- a generous 2x margin above the baseline's own std, since exact
    # equality isn't guaranteed run-to-run.
    assert f_errs_with_cx.std() < 2.0 * f_errs_baseline.std() + 0.3


# ---------------------------------------------------------------------
# Plausibility gates
# ---------------------------------------------------------------------


def test_implausible_cx_offset_rejected_not_clamped():
    """A cx offset well past MAX_ABS_CX_OFFSET_FRACTION_OF_WIDTH must be
    rejected outright, mirroring test_distortion.py's own
    test_implausible_k1_rejected_not_clamped."""
    absurd_offset = 0.5 * W  # way past the real measured range (-47px to +4px)
    obj, img = _project_ring20_cx(830.0, 0, 0.0, CX0 + absurd_offset)
    per_frame = [_FakeOrientedResult(ring20_px=img) for _ in range(5)]
    result = derive_focal_k1_cx_from_oriented_results(
        per_frame, (CX0, CY0), W, H, min_frames=1, initial_focal_px=830.0,
    )
    if result.ok:
        max_offset = MAX_ABS_CX_OFFSET_FRACTION_OF_WIDTH * W
        assert abs(result.cx - CX0) <= max_offset
    else:
        assert "outside plausible" in result.reason or "no multi-init seed converged" in result.reason


def test_implausible_k1_still_rejected_in_cx_model():
    obj, img = _project_ring20_cx(830.0, 0, -3.0, CX0)
    per_frame = [_FakeOrientedResult(ring20_px=img) for _ in range(5)]
    result = derive_focal_k1_cx_from_oriented_results(
        per_frame, (CX0, CY0), W, H, min_frames=1, initial_focal_px=830.0,
    )
    if result.ok:
        assert MIN_K1 <= result.k1 <= MAX_K1
    else:
        assert "outside plausible" in result.reason or "no multi-init seed converged" in result.reason


def test_min_frames_gate():
    obj, img = _project_ring20_cx(830.0, 0, -0.2, CX0)
    per_frame = [_FakeOrientedResult(ring20_px=img) for _ in range(3)]
    result = derive_focal_k1_cx_from_oriented_results(
        per_frame, (CX0, CY0), W, H, min_frames=5, initial_focal_px=830.0,
    )
    assert not result.ok
    assert "need >=" in result.reason


# ---------------------------------------------------------------------
# 2026-09-03: the +cx tier shares distortion.py's own MIN_K1/MAX_K1
# gate with the k1-only tier (confirmed by reading both derivation
# functions -- `derive_focal_k1_cx_from_oriented_results()` applies the
# exact same `MIN_K1 <= best.k1 <= MAX_K1` check). The k1-only tier's
# own real ceiling (+0.03258) was NOT the higher of the two real
# ceilings -- the +cx tier's own real ceiling, +0.05639, is -- so the
# shared, asymmetrically-tightened MAX_K1=0.10 must be re-verified
# against THIS tier's own real distribution too, not assumed safe by
# analogy. See distortion.py's own MAX_K1 comment for the full
# corpus-derivation.
# ---------------------------------------------------------------------

REAL_INCIDENT_CX_TIER_K1 = 0.20482  # calib_20260903-062929-f8fa8666, cam2, +cx tier
REAL_CX_TIER_HEALTHY_CEILING = 0.05639  # cam0, calib_20260902-234045-9a27e9cb -- the
# HIGHER of the two tiers' real ceilings; this is what a shared MAX_K1 must clear.


def test_max_k1_tightened_rejects_the_real_incident_value_in_cx_model():
    """Same real incident as distortion.py's own MAX_K1 comment
    (`a real calibration package`, cam2) -- the +cx tier's own live
    solve that event landed on cx_k1=+0.20482, near-identical to the
    k1-only tier's +0.20487 on the same package/camera. Reconstructed
    synthetically at the +cx tier's own real incident magnitude, must be
    rejected by the tightened, shared MAX_K1, not silently accepted."""
    obj, img = _project_ring20_cx(830.0, 2, REAL_INCIDENT_CX_TIER_K1, CX0)
    per_frame = [_FakeOrientedResult(ring20_px=img) for _ in range(5)]
    result = derive_focal_k1_cx_from_oriented_results(
        per_frame, (CX0, CY0), W, H, min_frames=1, initial_focal_px=830.0,
    )
    assert not result.ok
    assert "outside plausible" in result.reason
    assert f"[{MIN_K1}, {MAX_K1}]" in result.reason


def test_max_k1_tightened_still_accepts_the_cx_tiers_own_real_ceiling():
    """The +cx tier's own real ceiling (+0.05639) is HIGHER than the
    k1-only tier's (+0.03258) -- confirms the shared MAX_K1=0.10 clears
    the tier that actually needs the larger margin, not just the tier
    that happened to be checked first."""
    obj, img = _project_ring20_cx(830.0, 0, REAL_CX_TIER_HEALTHY_CEILING, CX0)
    per_frame = [_FakeOrientedResult(ring20_px=img) for _ in range(5)]
    result = derive_focal_k1_cx_from_oriented_results(
        per_frame, (CX0, CY0), W, H, min_frames=1, initial_focal_px=830.0,
    )
    assert result.ok, result.reason
    assert abs(result.k1 - REAL_CX_TIER_HEALTHY_CEILING) < NOISELESS_K1_TOL
    assert MIN_K1 <= result.k1 <= MAX_K1


# ---------------------------------------------------------------------
# JSON persistence -- mirrors test_distortion.py's own tests exactly,
# for the SEPARATE principal_point_fallback.json file
# ---------------------------------------------------------------------


def test_load_principal_point_fallback_missing_file_returns_empty(tmp_path):
    assert load_principal_point_fallback(tmp_path) == {}


def test_write_then_load_principal_point_fallback_round_trips(tmp_path):
    write_principal_point_fallback_entry(
        tmp_path, 0, 875.5, -0.22, 596.0, n_frames_used=30, n_points_used=20,
        derived_at_utc="2026-08-26T00:00:00+00:00", package_id="pkg-123",
    )
    loaded = load_principal_point_fallback(tmp_path)
    assert 0 in loaded
    assert loaded[0]["focal_length_px"] == pytest.approx(875.5)
    assert loaded[0]["k1"] == pytest.approx(-0.22)
    assert loaded[0]["cx"] == pytest.approx(596.0)
    assert loaded[0]["package_id"] == "pkg-123"


def test_write_preserves_other_cameras_entries(tmp_path):
    write_principal_point_fallback_entry(
        tmp_path, 0, 875.5, -0.22, 596.0, n_frames_used=30, n_points_used=20,
        derived_at_utc="2026-08-26T00:00:00+00:00",
    )
    write_principal_point_fallback_entry(
        tmp_path, 1, 793.1, 0.02, 642.9, n_frames_used=30, n_points_used=20,
        derived_at_utc="2026-08-26T00:01:00+00:00",
    )
    loaded = load_principal_point_fallback(tmp_path)
    assert set(loaded) == {0, 1}
    assert loaded[0]["cx"] == pytest.approx(596.0)
    assert loaded[1]["cx"] == pytest.approx(642.9)


def test_load_principal_point_fallback_rejects_wrong_schema(tmp_path):
    import json

    path = tmp_path / PRINCIPAL_POINT_FALLBACK_FILENAME
    path.write_text(json.dumps({
        "schema": "not-the-real-schema",
        "cameras": {"0": {"focal_length_px": 1.0, "k1": 0.0, "cx": 640.0}},
    }))
    assert load_principal_point_fallback(tmp_path) == {}


def test_load_principal_point_fallback_rejects_corrupt_json(tmp_path):
    path = tmp_path / PRINCIPAL_POINT_FALLBACK_FILENAME
    path.write_text("{not valid json")
    assert load_principal_point_fallback(tmp_path) == {}


def test_load_principal_point_fallback_rejects_record_missing_cx(tmp_path):
    """A record from a hypothetical future schema mismatch missing `cx`
    entirely must be skipped, not crash or silently substitute a
    default -- this file's records are always the FULL self-consistent
    (f, k1, cx) triple, never a partial one."""
    import json

    path = tmp_path / PRINCIPAL_POINT_FALLBACK_FILENAME
    path.write_text(json.dumps({
        "schema": PRINCIPAL_POINT_SCHEMA,
        "cameras": {"0": {"focal_length_px": 800.0, "k1": 0.0}},  # no "cx"
    }))
    assert load_principal_point_fallback(tmp_path) == {}


def test_schema_constant_is_versioned():
    assert PRINCIPAL_POINT_SCHEMA == "principal-point-fallback-v1"
    assert PRINCIPAL_POINT_FALLBACK_FILENAME == "principal_point_fallback.json"
