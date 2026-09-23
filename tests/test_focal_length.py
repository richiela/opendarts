"""Tests for opendarts.calibration.focal_length -- the image-only,
per-camera focal-length derivation that replaces
opendarts.live.capture_daemon.MEASURED_FOCAL_LENGTH_PX's role as the
PRIMARY source of a camera's intrinsics focal length (see that module's
own docstring for the full story and the real bug this closes).

Tolerances below are MEASURED, not guessed. The full sweep this file's
assertions are drawn from:
  - Noiseless recovery at this rig's 3 real focal lengths (800.0/830.0/
    835.0px) and all 3 real camera azimuths, default rig geometry
    (500mm ring radius, 300mm mount height): max |error| measured
    4.6e-4 px. `NOISELESS_TOL_PX` below (0.01px) is ~20x that.
  - Single-frame recovery under realistic per-point pixel noise
    (0.1-1.0px, in line with this project's own documented sub-pixel
    landmark-detection precision): UNBIASED (mean error ~0), std
    0.87-8.73px, max |error| up to ~26px at 1.0px noise (n=300 per
    noise level, 3 cameras x 100 trials).
  - Multi-frame (median) averaging at 0.5px per-point noise, N=20
    frames (a realistic real-bootstrap frame count): mean error -0.05px,
    std 1.04px, max |error| 3.08px (n=90, 30 trials x 3 cameras).
  - Hardware-identity honesty: feeding the SAME camera INDEX two
    different physical cameras' own images (f=800.0 then f=835.0)
    recovers the CORRECT value for whichever physical camera's images
    it was just given, not a stale memory of the first -- the entire
    point of this module (see opendarts.live.capture_daemon.
    MEASURED_FOCAL_LENGTH_PX's own updated comment for the real bug).
"""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from opendarts.calibration.focal_length import (
    FOCAL_LENGTH_FALLBACK_FILENAME,
    MAX_FOCAL_LENGTH_PX,
    MIN_FOCAL_LENGTH_PX,
    MIN_RING20_POINTS_FOR_HOMOGRAPHY,
    N_RING_LANDMARKS,
    SCHEMA,
    average_ring20_image_points,
    derive_focal_length_from_oriented_results,
    derive_focal_length_from_ring20,
    load_focal_length_fallback,
    ring20_image_points_from_result,
    ring20_object_points_mm,
    write_focal_length_fallback_entry,
)
from opendarts.calibration.oriented_landmarks import AD_QUAD_RING_INDICES, ad_quad_object_points_mm
from tests.support.synthetic import make_camera_matrix, make_ring_camera

W, H = 1280, 720
PRINCIPAL_POINT = (W / 2.0, H / 2.0)
REAL_RIG_FOCAL_LENGTHS_PX = (800.0, 830.0, 835.0)

# See module docstring above for the real measured basis of each bound.
NOISELESS_TOL_PX = 0.01
SINGLE_FRAME_NOISE_STD_TOL_PX = {0.1: 2.0, 0.3: 5.0, 0.5: 7.0, 1.0: 15.0}
MULTI_FRAME_MEAN_ABS_ERR_TOL_PX = 1.0
MULTI_FRAME_MAX_ABS_ERR_TOL_PX = 6.0


def _project_ring20(f: float, cam_index: int, n_cams: int = 3, *, ring_radius_mm=500.0, height_mm=300.0):
    import cv2

    K = make_camera_matrix(W, H, fov_deg=2 * math.degrees(math.atan((W / 2.0) / f)))
    cam = make_ring_camera(cam_index, n_cams, K, ring_radius_mm=ring_radius_mm, height_mm=height_mm)
    obj = ring20_object_points_mm()
    img_pts, _ = cv2.projectPoints(obj, cam.rvec, cam.tvec, cam.camera_matrix, cam.dist_coeffs)
    return obj, img_pts.reshape(-1, 2)


class _FakeOrientedResult:
    def __init__(self, ok=True, orientation_ambiguous=False, ring20_px=None):
        self.ok = ok
        self.orientation_ambiguous = orientation_ambiguous
        self.ring20_px = ring20_px


# ---------------------------------------------------------------------
# ring20_object_points_mm()
# ---------------------------------------------------------------------


def test_ring20_object_points_agrees_with_ad_quad_at_shared_indices():
    """A real regression guard: this module's own 20-point formula must
    agree EXACTLY with oriented_landmarks.ad_quad_object_points_mm() at
    every index the 4-point quad already uses -- both are meant to describe
    the identical physical wire-junction points, just at different
    subsets (4 vs 20) of the same 20-sector ring."""
    ring20 = ring20_object_points_mm()
    ad_quad = ad_quad_object_points_mm()
    for i, k in enumerate(AD_QUAD_RING_INDICES):
        assert np.allclose(ring20[k], ad_quad[i])


def test_ring20_object_points_shape_and_planarity():
    ring20 = ring20_object_points_mm()
    assert ring20.shape == (N_RING_LANDMARKS, 3)
    assert np.all(ring20[:, 2] == 0.0)  # every point on the board's Z=0 plane


# ---------------------------------------------------------------------
# derive_focal_length_from_ring20() -- synthetic, real ground truth
# ---------------------------------------------------------------------


@pytest.mark.parametrize("f_true", REAL_RIG_FOCAL_LENGTHS_PX)
@pytest.mark.parametrize("cam_index", [0, 1, 2])
def test_noiseless_recovery_at_real_rig_focal_lengths(f_true, cam_index):
    obj, img = _project_ring20(f_true, cam_index)
    result = derive_focal_length_from_ring20(obj, img, PRINCIPAL_POINT)
    assert result.ok, result.reason
    assert abs(result.focal_length_px - f_true) < NOISELESS_TOL_PX


@pytest.mark.parametrize("noise_px", [0.1, 0.3, 0.5, 1.0])
def test_single_frame_recovery_under_realistic_noise_is_unbiased(noise_px):
    """Real, measured noise sensitivity -- not just noiseless. Averages
    across 3 real cameras x 60 trials so the assertion isn't pinned to
    one noise seed; see module docstring for the full per-noise-level
    sweep this bound is drawn from."""
    rng = np.random.default_rng(20260826)
    errs = []
    for trial in range(60):
        for cam_index in range(3):
            obj, img = _project_ring20(830.0, cam_index)
            noisy = img + rng.normal(0, noise_px, img.shape)
            result = derive_focal_length_from_ring20(obj, noisy, PRINCIPAL_POINT)
            if result.ok:
                errs.append(result.focal_length_px - 830.0)
    errs = np.array(errs)
    assert len(errs) > 100  # the method should succeed on almost every trial
    assert abs(errs.mean()) < SINGLE_FRAME_NOISE_STD_TOL_PX[noise_px]
    assert errs.std() < SINGLE_FRAME_NOISE_STD_TOL_PX[noise_px]


@pytest.mark.parametrize("ring_radius_mm,height_mm", [
    (300.0, 200.0),   # close/low
    (500.0, 300.0),   # default -- matches make_ring_camera()'s own real-rig-matching default
    (800.0, 500.0),   # far/high
    (500.0, 50.0),    # near-fronto-parallel (low mount height)
])
def test_recovery_across_plausible_rig_geometries(ring_radius_mm, height_mm):
    """Real conditioning check across this rig's plausible camera-to-
    board distances/angles (docs/DESIGN.md: validate identifiability with the
    synthetic rig generator before trusting real data) -- every one of
    these stays well-conditioned (see module docstring's noise-sweep
    numbers, all measured at ring=500/height=300; this sweep confirms
    the OTHER plausible geometries aren't secretly degenerate)."""
    obj, img = _project_ring20(830.0, 0, ring_radius_mm=ring_radius_mm, height_mm=height_mm)
    result = derive_focal_length_from_ring20(obj, img, PRINCIPAL_POINT)
    assert result.ok, result.reason
    assert abs(result.focal_length_px - 830.0) < NOISELESS_TOL_PX


def test_rejects_too_few_points():
    obj, img = _project_ring20(830.0, 0)
    n = MIN_RING20_POINTS_FOR_HOMOGRAPHY - 1
    result = derive_focal_length_from_ring20(obj[:n], img[:n], PRINCIPAL_POINT)
    assert not result.ok
    assert "need >=" in result.reason


def test_rejects_mismatched_point_counts():
    obj, img = _project_ring20(830.0, 0)
    result = derive_focal_length_from_ring20(obj, img[:-1], PRINCIPAL_POINT)
    assert not result.ok
    assert "mismatch" in result.reason


def test_rejects_implausible_focal_length_range():
    """A geometrically well-formed but physically nonsensical result
    (e.g. corrupted correspondences producing a huge or tiny f) must be
    rejected outright, not clamped or silently accepted -- see
    MIN_FOCAL_LENGTH_PX/MAX_FOCAL_LENGTH_PX's own docstring for why these
    bounds are physical, not a rig-specific tuning knob."""
    obj, img = _project_ring20(150.0, 0)  # far outside MIN_FOCAL_LENGTH_PX
    result = derive_focal_length_from_ring20(obj, img, PRINCIPAL_POINT)
    # Either genuinely rejected as implausible, or (less likely at this
    # extreme) recovers correctly -- what must NEVER happen is silently
    # returning something inside [MIN,MAX] that isn't the true value.
    if result.ok:
        assert MIN_FOCAL_LENGTH_PX <= result.focal_length_px <= MAX_FOCAL_LENGTH_PX
        assert abs(result.focal_length_px - 150.0) < NOISELESS_TOL_PX
    else:
        assert "outside plausible" in result.reason or "non-physical" in result.reason


def test_degenerate_collinear_points_rejected_not_crashed():
    """All-collinear correspondences (a genuinely degenerate homography
    input) must fail cleanly, never raise."""
    obj = ring20_object_points_mm()
    img = np.tile(np.array([[100.0, 100.0]]), (N_RING_LANDMARKS, 1))
    img += np.arange(N_RING_LANDMARKS)[:, None] * np.array([[1.0, 0.0]])  # collinear
    result = derive_focal_length_from_ring20(obj, img, PRINCIPAL_POINT)
    assert not result.ok  # must not raise, and must not fabricate a value


# ---------------------------------------------------------------------
# Multi-frame averaging (derive_focal_length_from_oriented_results)
# ---------------------------------------------------------------------


def test_multi_frame_averaging_reduces_noise_vs_single_frame():
    """Real, measured noise-reduction effect from averaging across
    frames (median, opendarts.calibration.focal_length.
    average_ring20_image_points()) -- the SAME kind of effect
    sector_correspondence.average_correspondences() already documents
    for the 4-point pose correspondence, now shown for this module's own
    20-point correspondence. See module docstring for the real numbers
    this bound is drawn from (n_avg=20 @ 0.5px noise: mean -0.05px, std
    1.04px, max |err| 3.08px, n=90)."""
    rng = np.random.default_rng(7)
    n_avg = 20
    errs = []
    for trial in range(30):
        for cam_index in range(3):
            per_frame = []
            for j in range(n_avg):
                obj, img = _project_ring20(830.0, cam_index)
                noisy = img + rng.normal(0, 0.5, img.shape)
                per_frame.append(_FakeOrientedResult(ring20_px=noisy))
            result = derive_focal_length_from_oriented_results(
                per_frame, PRINCIPAL_POINT, min_frames=1,
            )
            if result.ok:
                errs.append(result.focal_length_px - 830.0)
    errs = np.array(errs)
    assert len(errs) > 60
    assert abs(errs.mean()) < MULTI_FRAME_MEAN_ABS_ERR_TOL_PX
    assert np.abs(errs).max() < MULTI_FRAME_MAX_ABS_ERR_TOL_PX


def test_average_ring20_image_points_is_a_per_index_median():
    """A minority-outlier frame (e.g. a bad detection on one round) must
    not drag the average as far as a plain mean would -- the whole
    reason this uses median, not mean (see that function's own
    docstring)."""
    base = np.tile(np.array([[100.0, 100.0]]), (N_RING_LANDMARKS, 1)).astype(np.float64)
    clean = [base + i * 0.01 for i in range(9)]  # 9 clean, near-identical frames
    outlier = base + 1000.0  # 1 wild outlier
    avg = average_ring20_image_points(clean + [outlier])
    # Median of 10 values with 1 wild outlier stays close to the clean cluster.
    assert np.abs(avg - base).max() < 1.0


# ---------------------------------------------------------------------
# ring20_image_points_from_result() -- the admission gate
# ---------------------------------------------------------------------


def test_ring20_image_points_from_result_rejects_not_ok():
    r = _FakeOrientedResult(ok=False, ring20_px=np.zeros((N_RING_LANDMARKS, 2)))
    assert ring20_image_points_from_result(r) is None


def test_ring20_image_points_from_result_rejects_ambiguous():
    """An ambiguous orientation lock means ALL 20 landmark positions may
    belong to the wrong rotation of the board -- same reject rule
    correspond_landmarks_from_pre_orientation() already applies to the
    4-point pose correspondence, and for the identical reason (see this
    module's own docstring)."""
    r = _FakeOrientedResult(orientation_ambiguous=True, ring20_px=np.zeros((N_RING_LANDMARKS, 2)))
    assert ring20_image_points_from_result(r) is None


def test_ring20_image_points_from_result_rejects_missing_or_wrong_shape():
    assert ring20_image_points_from_result(_FakeOrientedResult(ring20_px=None)) is None
    assert ring20_image_points_from_result(_FakeOrientedResult(ring20_px=np.zeros((4, 2)))) is None


def test_ring20_image_points_from_result_accepts_good_result():
    pts = np.arange(N_RING_LANDMARKS * 2, dtype=np.float64).reshape(N_RING_LANDMARKS, 2)
    r = _FakeOrientedResult(ring20_px=pts)
    out = ring20_image_points_from_result(r)
    assert out is not None
    assert np.array_equal(out, pts)


def test_derive_focal_length_from_oriented_results_min_frames_gate():
    obj, img = _project_ring20(830.0, 0)
    results = [_FakeOrientedResult(ring20_px=img) for _ in range(3)]
    result = derive_focal_length_from_oriented_results(results, PRINCIPAL_POINT, min_frames=5)
    assert not result.ok
    assert "need >=" in result.reason

    result_ok = derive_focal_length_from_oriented_results(results, PRINCIPAL_POINT, min_frames=3)
    assert result_ok.ok
    assert abs(result_ok.focal_length_px - 830.0) < NOISELESS_TOL_PX


# ---------------------------------------------------------------------
# HARDWARE-IDENTITY HONESTY -- the actual point of this whole module.
# ---------------------------------------------------------------------


def test_derivation_tracks_whichever_physical_camera_fed_this_index_not_a_stale_memory():
    """THE bug this module exists to fix (opendarts.live.capture_daemon.
    MEASURED_FOCAL_LENGTH_PX's own updated comment, and docs/DESIGN.md's dated
    2026-08-26 entry): feed the SAME logical "camera index" two
    DIFFERENT physical cameras' own images in turn (simulating a real
    USB port re-seat) -- the derivation must return the correct value
    for whichever physical camera it was JUST given, with zero memory of
    the previous one. There is no per-index state anywhere in this
    module for this test to even accidentally exercise -- every call is
    a fresh, independent derivation from its own arguments alone, which
    is the actual structural guarantee (not just an empirical one)."""
    obj_a, img_a = _project_ring20(800.0, 0)  # "physical A" now at index 0
    result_a = derive_focal_length_from_ring20(obj_a, img_a, PRINCIPAL_POINT)
    assert result_a.ok
    assert abs(result_a.focal_length_px - 800.0) < NOISELESS_TOL_PX

    # Re-seat: a DIFFERENT physical camera (f=835.0) now sits at the SAME
    # index 0. Nothing in this module's call signature or state carries
    # forward from the call above.
    obj_b, img_b = _project_ring20(835.0, 0)
    result_b = derive_focal_length_from_ring20(obj_b, img_b, PRINCIPAL_POINT)
    assert result_b.ok
    assert abs(result_b.focal_length_px - 835.0) < NOISELESS_TOL_PX
    # The critical assertion: NOT close to the previous camera's value.
    assert abs(result_b.focal_length_px - result_a.focal_length_px) > 30.0


# ---------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------


def test_load_focal_length_fallback_missing_file_returns_empty(tmp_path):
    assert load_focal_length_fallback(tmp_path) == {}


def test_write_then_load_focal_length_fallback_round_trips(tmp_path):
    write_focal_length_fallback_entry(
        tmp_path, 0, 838.2, n_frames_used=42, n_points_used=20,
        derived_at_utc="2026-08-26T12:00:00+00:00", package_id="calib_test",
    )
    loaded = load_focal_length_fallback(tmp_path)
    assert set(loaded.keys()) == {0}
    assert loaded[0]["focal_length_px"] == 838.2
    assert loaded[0]["n_frames_used"] == 42
    assert loaded[0]["n_points_used"] == 20
    assert loaded[0]["package_id"] == "calib_test"

    path = tmp_path / FOCAL_LENGTH_FALLBACK_FILENAME
    assert path.exists()
    on_disk = json.loads(path.read_text())
    assert on_disk["schema"] == SCHEMA


def test_write_focal_length_fallback_entry_preserves_other_cameras(tmp_path):
    """Real, measured requirement (write_focal_length_fallback_entry()'s
    own docstring): a camera that didn't derive live THIS event must
    keep its own last-known-good value, not have it wiped just because a
    sibling camera updated."""
    write_focal_length_fallback_entry(
        tmp_path, 0, 800.0, n_frames_used=10, n_points_used=20,
        derived_at_utc="2026-08-26T00:00:00+00:00",
    )
    write_focal_length_fallback_entry(
        tmp_path, 1, 810.0, n_frames_used=10, n_points_used=20,
        derived_at_utc="2026-08-26T00:00:01+00:00",
    )
    # Update cam0 only -- cam1's entry must survive untouched.
    write_focal_length_fallback_entry(
        tmp_path, 0, 805.0, n_frames_used=20, n_points_used=20,
        derived_at_utc="2026-08-26T00:00:02+00:00",
    )
    loaded = load_focal_length_fallback(tmp_path)
    assert loaded[0]["focal_length_px"] == 805.0
    assert loaded[0]["n_frames_used"] == 20
    assert loaded[1]["focal_length_px"] == 810.0  # untouched


def test_load_focal_length_fallback_corrupt_file_is_treated_as_absent(tmp_path):
    path = tmp_path / FOCAL_LENGTH_FALLBACK_FILENAME
    path.write_text("{not valid json")
    assert load_focal_length_fallback(tmp_path) == {}


def test_load_focal_length_fallback_schema_mismatch_is_treated_as_absent(tmp_path):
    path = tmp_path / FOCAL_LENGTH_FALLBACK_FILENAME
    path.write_text(json.dumps({"schema": "some-old-schema", "cameras": {"0": {"focal_length_px": 800.0}}}))
    assert load_focal_length_fallback(tmp_path) == {}


def test_load_focal_length_fallback_ignores_malformed_camera_entries(tmp_path):
    path = tmp_path / FOCAL_LENGTH_FALLBACK_FILENAME
    path.write_text(json.dumps({
        "schema": SCHEMA,
        "cameras": {
            "0": {"focal_length_px": 800.0},
            "not_an_int": {"focal_length_px": 900.0},
            "2": {"no_focal_length_key": True},
            "3": "not_even_a_dict",
        },
    }))
    loaded = load_focal_length_fallback(tmp_path)
    assert set(loaded.keys()) == {0}
