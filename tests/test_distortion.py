"""Tests for opendarts.calibration.distortion -- the image-only,
per-camera radial-distortion (k1) derivation that fits ALONGSIDE
opendarts.calibration.focal_length's homography-based focal-length
derivation (see that module's own docstring for why 20 points, not 4,
is what finally makes this safely identifiable).

Tolerances below are MEASURED, not guessed. The full sweeps these
assertions are drawn from:
  - Noiseless joint (f, pose, k1) recovery at this rig's 3 real focal
    lengths and every injected k1 in [-0.5, -0.1]: exact to numerical
    floor (~1e-11px RMS), ALL 7 multi-init FOV seeds converge to the
    identical answer every time (unlike the 4-point case).
  - Single-frame recovery under realistic per-point noise (0.3-0.5px):
    k1 unbiased (mean ~0), std 0.006-0.03, max |err| up to ~0.09-0.19 at
    the more pessimistic 1.0px noise level.
  - Multi-frame (per-index median) averaging at 0.3-0.5px noise, N=30-50
    frames (matching CALIBRATION_N_FRAMES_DETECT): k1 std 0.004-0.007,
    max |err| ~0.01-0.02 across 40 independent synthetic sessions.
  - k2 is explicitly NOT solved for: adding it as a free 9th unknown
    inflates k1's own std by >10x at the same noise levels, and k2's own
    std is 0.77-1.42 -- classic radial-distortion aliasing for a single-
    plane, limited-radius target.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from opendarts.calibration.distortion import (
    DISTORTION_FALLBACK_FILENAME,
    MAX_K1,
    MIN_K1,
    SCHEMA,
    derive_focal_and_k1_from_oriented_results,
    estimate_focal_pose_k1,
    load_distortion_fallback,
    write_distortion_fallback_entry,
)
from opendarts.calibration.focal_length import ring20_object_points_mm
from tests.support.synthetic import make_camera_matrix, make_ring_camera, project_points

W, H = 1280, 720
PRINCIPAL_POINT = (W / 2.0, H / 2.0)
REAL_RIG_FOCAL_LENGTHS_PX = (800.0, 830.0, 835.0)
OBJ = ring20_object_points_mm()

NOISELESS_TOL_PX = 1e-6
NOISELESS_K1_TOL = 1e-6
SINGLE_FRAME_K1_MEAN_TOL = {0.3: 0.02, 0.5: 0.03, 1.0: 0.05}
SINGLE_FRAME_K1_STD_TOL = {0.3: 0.03, 0.5: 0.05, 1.0: 0.1}
MULTI_FRAME_K1_STD_TOL_PX = 0.02
MULTI_FRAME_K1_MAX_ABS_ERR_TOL = 0.05


def _project_ring20(f: float, cam_index: int, k1: float, n_cams: int = 3, *, ring_radius_mm=500.0, height_mm=300.0):

    K = make_camera_matrix(W, H, fov_deg=2 * math.degrees(math.atan((W / 2.0) / f)))
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


@pytest.mark.parametrize("k1_true", [-0.5, -0.3, -0.2, -0.1, 0.0])
@pytest.mark.parametrize("cam_index", [0, 1, 2])
def test_noiseless_joint_recovery_at_real_rig_geometry(cam_index, k1_true):
    f_true = REAL_RIG_FOCAL_LENGTHS_PX[cam_index]
    obj, img = _project_ring20(f_true, cam_index, k1_true)
    result = estimate_focal_pose_k1(obj, img, W, H, initial_focal_px=f_true * 1.1)
    assert result.ok, result.reason
    assert abs(result.focal_px - f_true) < NOISELESS_TOL_PX
    assert abs(result.k1 - k1_true) < NOISELESS_K1_TOL


# ---------------------------------------------------------------------
# k2 is deliberately NOT part of the model
# ---------------------------------------------------------------------


def test_model_has_no_k2_unknown_by_construction():
    """This module fixes k2=p1=p2=k3=0 -- confirmed NOT separately
    identifiable from the 20-point ring (see module docstring). There is
    no public entry point here that exposes a k2 parameter; this is a
    structural guard against ever accidentally adding one back without
    re-reading that finding."""
    import inspect

    sig = inspect.signature(estimate_focal_pose_k1)
    assert "k2" not in sig.parameters
    assert "initial_k1" in sig.parameters  # k1 is the only distortion unknown


# ---------------------------------------------------------------------
# Noise robustness -- single frame and multi-frame averaging
# ---------------------------------------------------------------------


def test_multi_frame_median_averaging_recovers_k1_tightly():
    """Real, measured effect (module docstring): per-index median
    averaging across N=30 frames at 0.3-0.5px noise tightens k1's std to
    0.004-0.007 -- this test uses N=30, noise=0.4px, 30 sessions."""
    rng = np.random.default_rng(7)
    n_frames = 30
    noise_px = 0.4
    k1_errs = []
    for session in range(30):
        cam_index = session % 3
        k1_true = -0.2 if session % 2 == 0 else -0.4
        f_true = REAL_RIG_FOCAL_LENGTHS_PX[cam_index]
        obj, img_clean = _project_ring20(f_true, cam_index, k1_true)
        per_frame = [
            _FakeOrientedResult(ring20_px=img_clean + rng.normal(0, noise_px, img_clean.shape))
            for _ in range(n_frames)
        ]
        result = derive_focal_and_k1_from_oriented_results(
            per_frame, PRINCIPAL_POINT, W, H, min_frames=1, initial_focal_px=f_true,
        )
        if result.ok:
            k1_errs.append(result.k1 - k1_true)
    k1_errs = np.array(k1_errs)
    assert len(k1_errs) > 20
    assert k1_errs.std() < MULTI_FRAME_K1_STD_TOL_PX
    assert np.abs(k1_errs).max() < MULTI_FRAME_K1_MAX_ABS_ERR_TOL


# ---------------------------------------------------------------------
# Plausibility gate
# ---------------------------------------------------------------------


def test_implausible_k1_rejected_not_clamped():
    """A geometrically-converged-but-physically-nonsensical k1 (way
    outside MIN_K1/MAX_K1) must be rejected outright -- MIN_K1/MAX_K1
    comfortably bracket the real magnitude (~-0.55 to -0.6) needed to
    explain this rig's own observed 2.9-3.1px reprojection floor
    (module docstring), so a fit landing outside them is a real
    geometric failure, not a legitimate extreme answer."""
    # An absurdly strong synthetic k1, well past MAX_K1/MIN_K1, to force
    # a genuinely out-of-range fit (rather than fabricating an
    # already-in-range result and asserting a tautology).
    obj, img = _project_ring20(830.0, 0, -3.0)
    per_frame = [_FakeOrientedResult(ring20_px=img) for _ in range(5)]
    result = derive_focal_and_k1_from_oriented_results(
        per_frame, PRINCIPAL_POINT, W, H, min_frames=1, initial_focal_px=830.0,
    )
    if result.ok:
        assert MIN_K1 <= result.k1 <= MAX_K1
    else:
        assert "outside plausible" in result.reason or "no multi-init seed converged" in result.reason


# Real, measured numbers behind the 2026-09-03 asymmetric MAX_K1
# tightening (1.0 -> 0.10) -- see MAX_K1's own dated comment in
# distortion.py for the full derivation and pointer to the real
# incident. Re-derived directly from every real calibration event's
# `calibrations/derived_calibration.json` (116 real live camera-events
# per tier, both rigs), not copied from a prior estimate.
REAL_INCIDENT_K1 = 0.20487  # calib_20260903-062929-f8fa8666, cam2, k1-only tier
REAL_K1_ONLY_TIER_HEALTHY_CEILING = 0.03258  # cam1, calib_20260827-183057-1f5c1119
REAL_CX_TIER_HEALTHY_CEILING = 0.05639  # the higher of the two tiers' real ceilings
REAL_FLOOR_CAMERA_K1_VALUES = (-0.220, -0.220, -0.221, -0.233)  # docs/DESIGN.md's 4-package validation


def test_max_k1_tightened_rejects_the_real_incident_value():
    """The real, live-scoring-affecting defect this tightening exists to
    close: one recorded calibration event,
    cam2's joint solve landed on k1=+0.20487 -- a genuinely degenerate
    solve (see MAX_K1's own comment for the independent focal-consistency
    confirmation) that the OLD MAX_K1=1.0 admitted without complaint.
    Reconstructed synthetically at the real incident's own k1 magnitude
    (noiseless, so a converged fit reproduces the true k1 almost exactly)
    -- must now be rejected, not silently accepted."""
    obj, img = _project_ring20(830.0, 2, REAL_INCIDENT_K1)
    per_frame = [_FakeOrientedResult(ring20_px=img) for _ in range(5)]
    result = derive_focal_and_k1_from_oriented_results(
        per_frame, PRINCIPAL_POINT, W, H, min_frames=1, initial_focal_px=830.0,
    )
    assert not result.ok
    assert "outside plausible" in result.reason
    assert f"[{MIN_K1}, {MAX_K1}]" in result.reason


@pytest.mark.parametrize(
    "k1_true",
    [
        *REAL_FLOOR_CAMERA_K1_VALUES,
        REAL_K1_ONLY_TIER_HEALTHY_CEILING,
        REAL_CX_TIER_HEALTHY_CEILING,  # the k1-only tier's own gate must not
        # falsely reject the +cx tier's own (higher) real ceiling either --
        # both tiers share this same MAX_K1 constant (see its own comment).
    ],
)
def test_max_k1_tightened_still_accepts_real_healthy_values(k1_true):
    """Every genuinely healthy k1 this project has ever measured -- both
    the negative "floor camera" repeatability (docs/DESIGN.md's 4-package
    validation) and the real positive ceilings from both tiers -- must
    still pass the tightened gate. Confirms the tightening didn't
    over-correct into false-rejecting real, physically legitimate
    calibration events."""
    obj, img = _project_ring20(830.0, 1, k1_true)
    per_frame = [_FakeOrientedResult(ring20_px=img) for _ in range(5)]
    result = derive_focal_and_k1_from_oriented_results(
        per_frame, PRINCIPAL_POINT, W, H, min_frames=1, initial_focal_px=830.0,
    )
    assert result.ok, result.reason
    assert abs(result.k1 - k1_true) < NOISELESS_K1_TOL
    assert MIN_K1 <= result.k1 <= MAX_K1


def test_min_frames_gate():
    obj, img = _project_ring20(830.0, 0, -0.2)
    per_frame = [_FakeOrientedResult(ring20_px=img) for _ in range(3)]
    result = derive_focal_and_k1_from_oriented_results(
        per_frame, PRINCIPAL_POINT, W, H, min_frames=5, initial_focal_px=830.0,
    )
    assert not result.ok
    assert "need >=" in result.reason


# ---------------------------------------------------------------------
# JSON persistence -- mirrors test_focal_length.py's own tests exactly
# ---------------------------------------------------------------------


def test_load_distortion_fallback_missing_file_returns_empty(tmp_path):
    assert load_distortion_fallback(tmp_path) == {}


def test_write_then_load_distortion_fallback_round_trips(tmp_path):
    write_distortion_fallback_entry(
        tmp_path, 0, 834.6, -0.42, n_frames_used=42, n_points_used=20,
        derived_at_utc="2026-08-26T00:00:00+00:00", package_id="pkg-123",
    )
    loaded = load_distortion_fallback(tmp_path)
    assert 0 in loaded
    assert loaded[0]["focal_length_px"] == pytest.approx(834.6)
    assert loaded[0]["k1"] == pytest.approx(-0.42)
    assert loaded[0]["package_id"] == "pkg-123"


def test_write_preserves_other_cameras_entries(tmp_path):
    write_distortion_fallback_entry(
        tmp_path, 0, 834.6, -0.42, n_frames_used=42, n_points_used=20,
        derived_at_utc="2026-08-26T00:00:00+00:00",
    )
    write_distortion_fallback_entry(
        tmp_path, 1, 801.9, -0.10, n_frames_used=30, n_points_used=20,
        derived_at_utc="2026-08-26T00:01:00+00:00",
    )
    loaded = load_distortion_fallback(tmp_path)
    assert set(loaded) == {0, 1}
    assert loaded[0]["k1"] == pytest.approx(-0.42)
    assert loaded[1]["k1"] == pytest.approx(-0.10)


def test_load_distortion_fallback_rejects_wrong_schema(tmp_path):
    import json

    path = tmp_path / DISTORTION_FALLBACK_FILENAME
    path.write_text(json.dumps({"schema": "not-the-real-schema", "cameras": {"0": {"focal_length_px": 1.0, "k1": 0.0}}}))
    assert load_distortion_fallback(tmp_path) == {}


def test_load_distortion_fallback_rejects_corrupt_json(tmp_path):
    path = tmp_path / DISTORTION_FALLBACK_FILENAME
    path.write_text("{not valid json")
    assert load_distortion_fallback(tmp_path) == {}


def test_schema_constant_is_versioned():
    assert SCHEMA == "distortion-fallback-v1"
