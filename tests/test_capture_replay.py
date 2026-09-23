"""Tests for opendarts/capture/ (throw_package.py, replay.py).

Per docs/DESIGN.md's "Replay is the source of truth": the whole point of
this package is that a saved package stores exact raw inputs (not just a
result), so replaying it through the CURRENTLY-installed pipeline code
can produce a genuinely DIFFERENT answer than what was originally
stored, if the pipeline has changed since capture. That property (not
just "save/load doesn't crash") is what this file is actually trying to
prove, most rigorously in
test_replay_through_changed_detection_produces_different_result below.

Two data sources are used, deliberately:
  - Synthetic cameras/images (tests.support.synthetic, same helpers
    tests/test_pipeline_end_to_end.py uses) for anything that needs a
    KNOWN-correct answer or fully controlled inputs (round-trip byte-
    exactness, partial-write refusal, the changed-vs-unchanged-detection
    comparison). Synthetic data is the right tool here because these
    tests are about the CAPTURE/REPLAY PLUMBING, not about detection
    accuracy -- see docs/DESIGN.md's "measure the real number" discipline
    for why *detection accuracy* tests specifically need real images
    instead.
  - Real reference case images, for one test that exercised the real
    detect_tip() algorithm end-to-end through a save/replay round trip.
    Those images no longer exist anywhere, so that test went with them
    (2026-09-17); everything here is synthetic now.

Key implementation fact this file was written around, not assumed:
**2026-08-12 -- `replay.py` no longer calls `detect_tip()` directly at
all** (it now routes through `opendarts.engines.apollo.ApolloEngine`,
see that module's own docstring for the direct-call-bypass removal).
`opendarts.engines.apollo.engine` does `from opendarts.engines.apollo.
tip_detection import detect_tip`, binding the NAME `detect_tip` into
THAT module's own namespace. Patching
`opendarts.engines.apollo.tip_detection.detect_tip` would NOT affect the
engine's already-bound reference -- the correct monkeypatch seam is
`opendarts.engines.apollo.engine.detect_tip` itself (same principle this
file's tests always used, just relocated to the new call site -- verified
by reading `opendarts/engines/apollo/engine.py`'s imports, not assumed).
"""
from __future__ import annotations


import numpy as np
import pytest

import opendarts.engines.apollo.engine as apollo_engine_module
from tests.support.synthetic import make_camera_matrix, make_ring_camera
from opendarts.capture.throw_package import load_throw_package, save_throw_package
from opendarts.capture.replay import replay_and_compare, replay_throw_with_engine
from opendarts.engines.apollo import engine_result_to_score_result as apollo_engine_result_to_score_result
from opendarts.engines.apollo.board_roi import reject_outside_roi
from opendarts.engines.apollo.scoring import score_dart
from opendarts.engines.apollo.tip_detection import TipDetectionResult, detect_tip
from opendarts.geometry.board import (
    DOUBLE_INNER_RADIUS_MM,
    DOUBLE_OUTER_RADIUS_MM,
    TREBLE_INNER_RADIUS_MM,
    TREBLE_OUTER_RADIUS_MM,
    polar_to_xy_mm,
    sector_center_angle_deg,
)
from opendarts.pipeline import CameraCalibration, ScoreResult


# --------------------------------------------------------------------------
# Scratch directory discipline: docs/DESIGN.md is explicit that this repo's
# `<repo>/tmp/` (gitignored) must be used for scratch files, never `/tmp`
# or the harness scratchpad -- real, repeated approval friction before
# this was fixed. Each test gets its own subdirectory, removed afterward.
# --------------------------------------------------------------------------
@pytest.fixture()
def pkg_dir(tmp_path):
    return tmp_path / "pkg"


# --------------------------------------------------------------------------
# Synthetic-scene helpers (mirrors tests/test_pipeline_end_to_end.py's
# pattern: ground-truth cameras + project_points, so tests have a KNOWN
# answer to check against rather than just "did it run").
# --------------------------------------------------------------------------

def _synthetic_rig(n_cameras: int = 3, fov_deg: float = 90.0):
    """Ground-truth cameras + matching CameraCalibration dicts (calibration
    built directly from the TRUE pose, not recovered via PnP -- these
    tests are about capture/replay plumbing, not calibration accuracy,
    which tests/test_pipeline_end_to_end.py already covers separately)."""
    camera_matrix = make_camera_matrix(fov_deg=fov_deg)
    true_cams = [
        make_ring_camera(i, n_cameras=n_cameras, camera_matrix=camera_matrix)
        for i in range(n_cameras)
    ]
    calibrations = {
        i: CameraCalibration(
            camera_matrix=cam.camera_matrix,
            dist_coeffs=cam.dist_coeffs,
            rvec=cam.rvec,
            tvec=cam.tvec,
            pnp_result=None,
            landmark_spread_ok=True,
        )
        for i, cam in enumerate(true_cams)
    }
    return true_cams, calibrations


def _board_point_mm(sector_number: int, ring: str) -> tuple[float, float, float]:
    angle = sector_center_angle_deg(sector_number)
    if ring == "treble":
        radius = (TREBLE_INNER_RADIUS_MM + TREBLE_OUTER_RADIUS_MM) / 2
    elif ring == "double_outer":
        # Mid-band, not edge-of-ring -- see tests/test_rescore_all.py's
        # identical helper. Keeps a >=4mm margin from either ring
        # boundary regardless of engine-path noise, robust without
        # depending on the exact edge radius.
        radius = (DOUBLE_INNER_RADIUS_MM + DOUBLE_OUTER_RADIUS_MM) / 2
    else:
        raise ValueError(ring)
    x, y = polar_to_xy_mm(radius, angle)
    return (x, y, 0.0)


def _project_to_pixels(true_cams, point_xyz) -> dict[int, tuple[float, float]]:
    import cv2

    pt = np.asarray(point_xyz, dtype=np.float64).reshape(1, 1, 3)
    out = {}
    for i, cam in enumerate(true_cams):
        px, _ = cv2.projectPoints(pt, cam.rvec, cam.tvec, cam.camera_matrix, cam.dist_coeffs)
        out[i] = tuple(px.reshape(2))
    return out


def _marker_image(cam_idx: int, seed: int, h: int = 24, w: int = 32) -> np.ndarray:
    """A small deterministic BGR image whose (0,0) pixel encodes `cam_idx`
    -- used so a monkeypatched detect_tip can identify which camera it
    was called for without needing a camera-index parameter (detect_tip's
    real signature, per opendarts/engines/apollo/tip_detection.py, takes only
    (bg_bgr, frame_bgr) -- no camera index)."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    img[0, 0, 0] = cam_idx
    return img


def _replay_throw_apollo(package_or_dir):
    """Test-local stand-in for the old `replay_throw()` (Apollo-only,
    `ScoreResult`-shaped), deleted 2026-09-05 because it silently
    defaulted to Apollo -- see `opendarts/capture/replay.py`'s own
    docstring for the real incident that closed. Every real caller
    (including `replay_and_compare()` itself) must now say which engine
    it means; this helper says it once, explicitly, for the tests below
    that are specifically about Apollo's own replay behavior."""
    return apollo_engine_result_to_score_result(
        replay_throw_with_engine(package_or_dir, "Apollo")
    )


# --------------------------------------------------------------------------
# 1. Save -> load round trip is byte-exact / lossless.
# --------------------------------------------------------------------------

def test_save_load_round_trip_is_lossless(pkg_dir):
    true_cams, calibrations = _synthetic_rig(n_cameras=3)
    # Give the calibrations some real, non-trivial (non-zero, non-round)
    # numbers, e.g. a non-trivial dist_coeffs, so the round-trip check is
    # actually exercising real values, not coincidentally-round defaults.
    calibrations[1].dist_coeffs = np.array([0.1234567, -0.0765432, 0.001, -0.002, 0.3], dtype=np.float64)

    bg_frames = {i: _marker_image(i, seed=100 + i) for i in range(3)}
    dart_frames = {i: _marker_image(i, seed=200 + i) for i in range(3)}

    tip_pixels = _project_to_pixels(true_cams, _board_point_mm(20, "treble"))
    result = score_dart(tip_pixels, calibrations)
    assert result.ok, result.reason

    # Flip landmark_spread_ok to False on one camera AFTER scoring (score_dart
    # itself correctly rejects any result built from a poorly-spread-quad
    # camera -- tested separately in test_pipeline_end_to_end.py; this test's
    # only concern here is whether the field round-trips faithfully through
    # save/load, not whether score_dart's own gate still works).
    calibrations[2].landmark_spread_ok = False

    save_throw_package(pkg_dir, "session-abc", bg_frames, dart_frames, calibrations, result)
    pkg = load_throw_package(pkg_dir)

    assert pkg.session == "session-abc"
    assert pkg.cameras == [0, 1, 2]

    for cam in range(3):
        assert np.array_equal(pkg.bg_frames[cam], bg_frames[cam]), f"cam{cam} bg frame not byte-exact"
        assert np.array_equal(pkg.dart_frames[cam], dart_frames[cam]), f"cam{cam} dart frame not byte-exact"

        loaded_calib = pkg.calibrations[cam]
        orig_calib = calibrations[cam]
        assert np.array_equal(loaded_calib.camera_matrix, orig_calib.camera_matrix)
        assert np.array_equal(loaded_calib.dist_coeffs, orig_calib.dist_coeffs)
        assert np.array_equal(loaded_calib.rvec, orig_calib.rvec)
        assert np.array_equal(loaded_calib.tvec, orig_calib.tvec)
        assert loaded_calib.landmark_spread_ok == orig_calib.landmark_spread_ok
        # Documented, intentional NON-round-trip: pnp_result is dropped
        # on save (throw_package.py: "not needed for replay, not
        # round-tripped") -- assert that's still true, not an oversight.
        assert loaded_calib.pnp_result is None

    # Original result metadata also survives intact.
    assert pkg.original_result["sector"] == result.sector
    assert pkg.original_result["ring"] == result.ring
    assert pkg.original_result["n_cameras_used"] == result.n_cameras_used
    assert pkg.original_result["board_xy_mm"] == list(result.board_xy_mm)
    assert pkg.original_result["max_ray_disagreement_mm"] == pytest.approx(
        result.max_ray_disagreement_mm
    )


def test_save_load_round_trip_preserves_failed_result_and_null_triangulation(pkg_dir):
    """The result-metadata round trip must also survive an ok=False
    result with triangulation=None (e.g. the <2-camera rejection path) --
    not just the happy-path shape."""
    true_cams, calibrations = _synthetic_rig(n_cameras=3)
    bg_frames = {0: _marker_image(0, seed=1), 1: _marker_image(1, seed=2)}
    dart_frames = {0: _marker_image(0, seed=3), 1: _marker_image(1, seed=4)}

    single_cam_calib = {0: calibrations[0]}
    result = score_dart({0: (640.0, 360.0)}, single_cam_calib)
    assert not result.ok
    assert result.triangulation is None

    save_throw_package(pkg_dir, "session-fail", {0: bg_frames[0]}, {0: dart_frames[0]}, single_cam_calib, result)
    pkg = load_throw_package(pkg_dir)

    assert pkg.original_result["ok"] is False
    assert pkg.original_result["triangulation"] is None
    assert pkg.original_result["reason"] == result.reason


def test_save_throw_package_raises_when_no_camera_has_all_three_inputs(pkg_dir):
    """Disjoint camera sets across bg/dart/calibration -- the {bg, frame,
    calibration} intersection is empty, so there is nothing complete to
    write at all. Must raise, not write an empty-but-real-looking dir."""
    true_cams, calibrations = _synthetic_rig(n_cameras=3)
    bg_frames = {0: _marker_image(0, seed=1)}
    dart_frames = {1: _marker_image(1, seed=2)}
    calib_only_cam2 = {2: calibrations[2]}

    result = ScoreResult(
        ok=False, sector=None, ring=None, board_xy_mm=None, triangulation=None,
        n_cameras_used=0, reason="n/a",
    )

    with pytest.raises(ValueError, match="partial package"):
        save_throw_package(pkg_dir, "session-partial", bg_frames, dart_frames, calib_only_cam2, result)

    # A partial-looking directory must not exist: no meta.json means
    # load_throw_package() can never mistake this for a complete package.
    assert not (pkg_dir / "meta.json").exists()
    assert not (pkg_dir / "calibration.json").exists()


def test_save_throw_package_raises_ioerror_and_leaves_no_metadata_on_partial_image_write(pkg_dir, monkeypatch):
    """Even when SOME cameras have complete {bg, frame, calibration}
    triples, a failed frame write for any one of them must abort the whole
    save before any JSON metadata is written -- otherwise a directory
    with some clips but a stale/missing meta.json could later be
    (mis)treated as complete. (The frames are per-camera clips since
    2026-09-22; before that this failed a PNG write the same way.)"""
    from opendarts.capture import clip as clip_mod

    true_cams, calibrations = _synthetic_rig(n_cameras=3)
    bg_frames = {i: _marker_image(i, seed=10 + i) for i in range(3)}
    dart_frames = {i: _marker_image(i, seed=20 + i) for i in range(3)}
    tip_pixels = _project_to_pixels(true_cams, _board_point_mm(20, "treble"))
    result = score_dart(tip_pixels, calibrations)
    assert result.ok

    real_write = clip_mod.write_clip_ffv1
    calls = {"n": 0}

    def flaky_write(path, frames):
        calls["n"] += 1
        # Fail on the 2nd call (cam1's clip, in ascending camera order) --
        # deep enough that cam0 fully wrote its clip first, proving this
        # isn't a trivial "fails immediately" case.
        if calls["n"] == 2:
            raise OSError("simulated disk full")
        return real_write(path, frames)

    monkeypatch.setattr(clip_mod, "write_clip_ffv1", flaky_write)

    with pytest.raises(IOError):
        save_throw_package(pkg_dir, "session-flaky", bg_frames, dart_frames, calibrations, result)

    assert not (pkg_dir / "meta.json").exists()
    assert not (pkg_dir / "calibration.json").exists()
    assert not (pkg_dir / "result.json").exists()


# --------------------------------------------------------------------------
# 3. THE most important property: replaying through a CHANGED detection
#    function produces a genuinely different result than what was
#    originally stored -- proving replay reruns current code against
#    stored raw inputs, not a cached answer.
# --------------------------------------------------------------------------

def test_replay_through_changed_detection_produces_different_result(pkg_dir, monkeypatch):
    true_cams, calibrations = _synthetic_rig(n_cameras=3)

    point_original = _board_point_mm(20, "treble")       # sector 20
    point_changed = _board_point_mm(3, "double_outer")   # a different sector, far away

    pixels_original = _project_to_pixels(true_cams, point_original)
    pixels_changed = _project_to_pixels(true_cams, point_changed)

    # Simulate "what the live system originally produced": the ORIGINAL
    # detection function found `point_original`. Real packages always get
    # their original_result via ApolloEngine (the direct-call bypass was
    # removed 2026-08-12); ApolloEngine.score() calls score_dart()
    # directly with no further correction (the 2026-08-13 empirical
    # registration-offset correction was removed 2026-08-13 -- see
    # opendarts/engines/apollo/engine.py's own docstring), so plain
    # score_dart() output already matches what a real saved package's
    # result.json contains.
    original_result = score_dart(pixels_original, calibrations)
    assert original_result.ok, original_result.reason
    assert original_result.sector == "20"

    bg_frames = {i: _marker_image(i, seed=300 + i) for i in range(3)}
    dart_frames = {i: _marker_image(i, seed=400 + i) for i in range(3)}
    save_throw_package(pkg_dir, "session-replay", bg_frames, dart_frames, calibrations, original_result)

    # Now install a DIFFERENT "current" detection function -- simulating
    # a genuine pipeline change since capture -- that reports
    # `point_changed`'s pixels instead. The correct monkeypatch seam is
    # opendarts.engines.apollo.engine.detect_tip (the name the engine
    # module imported into its own namespace -- replay.py itself no
    # longer imports detect_tip at all since 2026-08-12's bypass
    # removal), NOT opendarts.engines.apollo.tip_detection.detect_tip.
    def changed_detect_tip(bg_bgr, frame_bgr, prior_dart_line_px=None):
        cam = int(frame_bgr[0, 0, 0])
        return TipDetectionResult(ok=True, tip_px=pixels_changed[cam], reason="changed-for-test")

    monkeypatch.setattr(apollo_engine_module, "detect_tip", changed_detect_tip)

    fresh_result = _replay_throw_apollo(pkg_dir)

    assert fresh_result.ok, fresh_result.reason
    assert fresh_result.sector == "3"
    assert fresh_result.sector != original_result.sector, (
        "replay through a changed detection function must NOT reproduce "
        "the originally-stored result -- this is the whole point of "
        "opendarts/capture/ (docs/DESIGN.md's 'Replay is the source of truth')"
    )

    # Also exercise replay_and_compare(), the actual drift-detection tool
    # this whole mechanism exists to support.
    comparison = replay_and_compare(pkg_dir, "Apollo")
    assert comparison.sector_changed is True
    assert comparison.board_xy_changed_mm is not None
    expected_distance_mm = float(
        np.linalg.norm(np.array(point_changed[:2]) - np.array(point_original[:2]))
    )
    # Noiseless synthetic projection: tests/test_pipeline_end_to_end.py's
    # equivalent noiseless case measures ~1e-5mm board-plane error, so a
    # 0.01mm tolerance here is consistent with that established, measured
    # (not guessed) precision floor, not a fresh round-number guess.
    assert comparison.board_xy_changed_mm == pytest.approx(expected_distance_mm, abs=0.01)


def test_replay_throw_excludes_a_camera_whose_tip_is_outside_the_board_roi(pkg_dir, monkeypatch):
    """Same real-geometry behavioral proof as tests/test_capture_daemon.py's
    test_handle_ready_to_capture_excludes_a_camera_whose_tip_is_outside_
    the_board_roi, for the OTHER live call site this fix wired the board-
    ROI gate into (2026-08-12) -- per
    docs/DESIGN.md's "Replay is the source of truth", replay must apply the identical
    current-pipeline gate the live capture path does, not a weaker subset
    of it, or replay would systematically under-report what live scoring
    actually does today."""
    true_cams, calibrations = _synthetic_rig(n_cameras=2)
    point_on_board = _board_point_mm(20, "treble")
    pixels_on_board = _project_to_pixels(true_cams, point_on_board)

    bg_frames = {i: _marker_image(i, seed=500 + i) for i in range(2)}
    dart_frames = {i: _marker_image(i, seed=600 + i) for i in range(2)}

    original_result = score_dart(pixels_on_board, calibrations)
    save_throw_package(
        pkg_dir, "session-roi-replay", bg_frames, dart_frames, calibrations, original_result
    )

    # Fresh replay detection: cam0 reports a real on-board tip, cam1
    # reports a wildly implausible off-board tip -- the correct
    # monkeypatch seam is opendarts.engines.apollo.engine.detect_tip (the
    # bound name in the engine module's own namespace, see this file's
    # module docstring), not opendarts.engines.apollo.tip_detection.detect_tip.
    tip_choices = {0: pixels_on_board[0], 1: (-100000.0, -100000.0)}

    def fake_detect_tip(bg_bgr, frame_bgr, prior_dart_line_px=None):
        cam = int(frame_bgr[0, 0, 0])
        return TipDetectionResult(ok=True, tip_px=tip_choices[cam], reason="fake")

    monkeypatch.setattr(apollo_engine_module, "detect_tip", fake_detect_tip)

    fresh_result = _replay_throw_apollo(pkg_dir)
    assert fresh_result.n_cameras_used <= 1, (
        f"expected cam1's wildly-off-board tip to be excluded by the ROI "
        f"gate on replay, got n_cameras_used={fresh_result.n_cameras_used}"
    )


def test_replay_and_compare_reports_no_change_when_original_result_missing(pkg_dir, monkeypatch):
    """result.json is explicitly optional (load_throw_package: `if
    result_path.exists()`) -- replay_and_compare must degrade cleanly
    (None comparisons, not a crash) rather than assume it's always
    present."""
    true_cams, calibrations = _synthetic_rig(n_cameras=2)
    pixels = _project_to_pixels(true_cams, _board_point_mm(20, "treble"))
    result = score_dart(pixels, {0: calibrations[0], 1: calibrations[1]})
    assert result.ok

    bg_frames = {i: _marker_image(i, seed=500 + i) for i in range(2)}
    dart_frames = {i: _marker_image(i, seed=600 + i) for i in range(2)}
    save_throw_package(pkg_dir, "session-noresult", bg_frames, dart_frames,
                        {0: calibrations[0], 1: calibrations[1]}, result)
    (pkg_dir / "result.json").unlink()

    def stub_detect_tip(bg_bgr, frame_bgr, prior_dart_line_px=None):
        cam = int(frame_bgr[0, 0, 0])
        return TipDetectionResult(ok=True, tip_px=pixels[cam], reason="stub")

    monkeypatch.setattr(apollo_engine_module, "detect_tip", stub_detect_tip)

    comparison = replay_and_compare(pkg_dir, "Apollo")
    assert comparison.original_result is None
    assert comparison.sector_changed is None
    assert comparison.board_xy_changed_mm is None
    assert comparison.fresh_result.ok

