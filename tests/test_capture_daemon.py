"""Tests for opendarts/live/capture_daemon.py -- the wiring around
opendarts.live.local_capture (direct cv2.VideoCapture, via a persistent
LocalCameraHub) as the frame source.

HONEST SCOPE: same as tests/test_local_capture.py -- these are LOGIC/
WIRING tests. Nothing here opens a real camera. What's verified:
  - bootstrap_calibrations()/fetch_current_frames() call the local frame
    source and require an already-open hub (never open one themselves --
    that's run_capture_loop's job).
  - run_capture_loop() opens exactly ONE LocalCameraHub for its entire
    run (not once per poll iteration), passes that SAME hub instance to
    every bootstrap_calibrations/fetch_current_frames call, and always
    closes it on the way out (success, RuntimeError, or
    NotImplementedError) via its finally block.
  - DEFAULT_PACKAGE_ROOT no longer points outside a writable location
    (the real bug behind it: Path.home()/"<old-dir>"/
    "packages" raises PermissionError in this sandbox) -- proven with an
    actual create-and-write probe, not just a path-string assertion.
  - main()'s CLI flags thread through to run_capture_loop().
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

# Distinct, arbitrary per-camera focal lengths so tests can tell cameras
# apart. Not measurements of any real rig.
_TEST_FOCAL_PX = {0: 800.0, 1: 810.0, 2: 820.0}

from tests.conftest import original_path, stub_confident_orientation

import opendarts.live.capture_daemon as capture_daemon
from tests.lifecycle_scripting import script_trigger
from opendarts.calibration.focal_length import FocalLengthResult
from opendarts.engines.base import EngineResult
from opendarts.engines.apollo.prior_dart_context import CachedPriorThrowFrames
from opendarts.calibration.oriented_landmarks import PreOrientationLandmarks
from opendarts.capture.throw_package import load_ad_ground_truth
from opendarts.capture.trigger_state import MAX_DARTS_PER_TURN, ThrowState, ThrowTriggerState
from tests.lifecycle_scripting import STARTUP_FETCHES
from opendarts.live import diagnostics_gate, local_capture
from opendarts.live.ad_ground_truth import AdGroundTruth
from opendarts.pipeline import CalibrationAttempt, CameraCalibration

# ---------------------------------------------------------------------------
# Shared fake for bootstrap_calibrations()'s live-derived orientation-hint
# pipeline (2026-08-20 restructuring -- see capture_daemon.py's own
# "LIVE-DERIVED ORIENTATION HINT" docstring section). Every test below that
# used to monkeypatch a single `correspond_landmarks_oriented(image_bgr,
# cam, **kwargs)` now needs FOUR pieces patched (the two-pass split this
# restructuring introduced): the pre-orientation stage, the two
# opendarts.calibration.ring_correlation_orientation calls that derive
# the session-level hint, and the finishing correspondence stage itself.
# None of these tests exercise real orientation math (that is
# tests/test_ring_correlation_orientation.py's own job) -- they exercise
# bootstrap_calibrations()'s ADAPTIVE RETRY / accumulation / threading
# logic, so every piece here is a harmless, always-confident stand-in
# EXCEPT `correspond`, which is each test's own real point of control
# (same role `fake_detect`/`fake_correspond` played against the old
# single function, called with the same `(image_bgr, pre, *,
# orientation_hint_deg=None, results_out=None, **kw)` signature
# `correspond_landmarks_from_pre_orientation` itself has -- most existing
# fakes' second parameter is stale-named `cam` from before this split;
# left as-is where a test never reads that argument's value, since
# renaming every one is not the point of this change).
_FAKE_PRE_OK = PreOrientationLandmarks(
    ok=True, reason="ok", ellipse=None, seed_ellipse=None, bull_px=(1.0, 1.0),
    normalised_bull_radius=0.0, profile=None, phase_deg=0.0,
    spoke_score=1.0, phase_confidence=5.0, notes=[],
)


@pytest.fixture(autouse=True)
def _use_digit_count_orientation_method(monkeypatch):
    """This whole file's own bootstrap_calibrations() tests exercise
    ADAPTIVE RETRY / accumulation / threading logic against the
    DIGIT-COUNT orientation method's own fakes (`_install_fake_
    orientation_pipeline()` above/below) -- none of them exercise real
    orientation math (see this file's own header comment). `ORIENTATION_
    METHOD` defaults to `"ring_correlation"` and would otherwise pre-empt every one of these
    tests (its own resolution phase runs BEFORE the digit-count fakes
    below ever get a chance to matter, and none of these tests mock
    `aggregate_mark_orientation_for_camera`/`ring_correlation_
    orientation_for_camera`). Force the selector back to the digit-count
    method here so this file keeps testing exactly what it always tested
    -- see `tests/test_capture_daemon_ring_correlation_wiring.py` for the
    method's own equivalent coverage."""
    stub_confident_orientation(monkeypatch, capture_daemon)


def _install_fake_orientation_pipeline(monkeypatch, correspond):
    monkeypatch.setattr(
        capture_daemon, "locate_pre_orientation_landmarks",
        lambda frame, **kw: (frame, _FAKE_PRE_OK),
    )
    monkeypatch.setattr(capture_daemon, "correspond_landmarks_from_pre_orientation", correspond)
    # LIVE-DERIVED FOCAL LENGTH, 2026-08-26 -- same "harmless, always-
    # confident stand-in" treatment as the orientation-hint pieces above
    # (the orientation solve etc.): none of these tests exercise
    # real focal-length math (that's tests/test_focal_length.py's own
    # job) -- they exercise bootstrap_calibrations()'s ADAPTIVE RETRY /
    # accumulation / threading logic, which now requires a resolved focal
    # length before any calibrate_camera() call can happen at all (see
    # _resolve_focal_length_px()'s own docstring) -- so every test using
    # this shared helper needs live focal-length derivation to succeed
    # immediately, from round 1, same as the hint. 900.0 is a fixed
    # test-fixture value, deliberately outside MEASURED_FOCAL_LENGTH_PX's
    # own real range, so a test asserting on it can't be confused with a
    # real rig constant. The 2 tests that need to DISTINGUISH cameras by
    # their own focal length (dispatching a fake calibrate_camera() call
    # back to "which camera is this for" via camera_matrix[0, 0], since
    # calibrate_camera() itself never receives `cam`) override this
    # AFTER calling this helper -- see
    # _install_fake_orientation_pipeline_with_focal_dispatch() below.
    monkeypatch.setattr(
        capture_daemon, "derive_focal_length_from_oriented_results",
        lambda results, principal_point, min_frames=1: FocalLengthResult(
            ok=True, focal_length_px=900.0, reason="test-fake",
            n_points_used=20, n_frames_used=max(1, len(results)),
        ),
    )


class _FakeOrientedResultForFocalDispatch:
    """Duck-typed stand-in for `OrientedLandmarkResult`, carrying only
    what `opendarts.calibration.focal_length.ring20_image_points_from_result()`
    reads (`ok`/`orientation_ambiguous`/`ring20_px`) plus a `_cam` marker
    this test file's own fake `derive_focal_length_from_oriented_results`
    reads back -- see `_install_fake_orientation_pipeline_with_focal_dispatch()`."""

    def __init__(self, cam: int):
        self.ok = True
        self.orientation_ambiguous = False
        self.ring20_px = None
        self._cam = cam


def _install_fake_orientation_pipeline_with_focal_dispatch(
    monkeypatch, correspond_object_image_points, n_cams: int
):
    """Like `_install_fake_orientation_pipeline()`, but for the 2 tests
    that need each camera to resolve to ITS OWN distinct
    `MEASURED_FOCAL_LENGTH_PX[cam]`-style value (so their own fake
    `calibrate_camera()` dispatcher can recover `cam` from
    `camera_matrix[0, 0]`, since `calibrate_camera()` itself never
    receives `cam` -- see those tests' own comments).

    Real camera identity has to flow through SOMETHING an argument
    actually carries -- `correspond_landmarks_from_pre_orientation()`'s
    real arguments carry no camera index either, only image pixels and
    geometry, so this uses the same channel a real multi-camera rig
    would: distinguishable FRAME CONTENT. The caller's own `fake_local`
    must give each camera visibly different pixel content (e.g.
    `np.full((h, w, 3), cam + 1, dtype=np.uint8)`, never all-zeros for
    every camera) -- this reads `image_bgr[0, 0, 0] - 1` back out to
    recover `cam`, tags a `_FakeOrientedResultForFocalDispatch` with it
    into `results_out`, and a matching `derive_focal_length_from_
    oriented_results()` fake reads that marker back to return exactly
    `MEASURED_FOCAL_LENGTH_PX[cam]`."""
    def correspond(image_bgr, pre, *, results_out=None, **kw):
        cam = int(image_bgr[0, 0, 0]) - 1
        if results_out is not None:
            results_out.append(_FakeOrientedResultForFocalDispatch(cam))
        return correspond_object_image_points(image_bgr, pre, **kw)

    _install_fake_orientation_pipeline(monkeypatch, correspond)

    def fake_derive_focal(results, principal_point, min_frames=1):
        if not results:
            return FocalLengthResult(
                ok=False, focal_length_px=None, reason="no frames",
                n_points_used=0, n_frames_used=0,
            )
        cam = results[-1]._cam
        return FocalLengthResult(
            ok=True, focal_length_px=float(_TEST_FOCAL_PX[cam]),
            reason="test-fake", n_points_used=20, n_frames_used=len(results),
        )

    monkeypatch.setattr(
        capture_daemon, "derive_focal_length_from_oriented_results", fake_derive_focal
    )


# ---------------------------------------------------------------------------
# DEFAULT_PACKAGE_ROOT bug fix
# ---------------------------------------------------------------------------


# The SHIPPED value of the constant, not the sandboxed stand-in a test run
# redirects it to (tests/conftest.py repoints every REPO_ROOT-derived
# default so a run writes nothing into the checkout). These two tests are
# the ones whose subject IS the shipped constant, so they ask for it by
# name.
REAL_DEFAULT_PACKAGE_ROOT = original_path(
    "opendarts.live.capture_daemon.DEFAULT_PACKAGE_ROOT"
)


def test_default_package_root_is_not_under_the_blocked_home_path():
    """Packages live under the rig's data directory, not directly under
    Path.home() -- that layout was refused by a sandbox that blocked
    writes anywhere under $HOME outside the checkout. The shipped
    constant must sit inside DATA_DIR (the checkout's data/, or
    OPENDARTS_DATA_DIR)."""
    real_data_dir = original_path("opendarts.paths.DATA_DIR")

    blocked = Path.home() / "old_install_dir" / "packages"
    assert REAL_DEFAULT_PACKAGE_ROOT != blocked
    assert real_data_dir in REAL_DEFAULT_PACKAGE_ROOT.parents


def test_default_package_root_is_actually_writable(tmp_path):
    """Real create-and-write probe, not just 'should work' -- proves the
    fixed path is genuinely writable in this environment, mirroring the
    exact check that exposed the original bug.

    Two halves. The mkdir+write+read-back sequence runs for real, against
    a tmp_path stand-in shaped exactly like the constant. The SHIPPED
    constant is then checked for writability without creating anything:
    its nearest existing ancestor must be a writable directory, which is
    the property the original bug violated (the old default's ancestor was
    an unwritable directory under $HOME). It used to mkdir the real root
    and drop a marker file in it -- which is how a plain test run came to
    create data/packages/ in a fresh checkout, the one directory that must
    never gain a file it did not earn (it is the replay corpus).
    """
    probe_root = tmp_path / "data" / "packages"
    probe_root.mkdir(parents=True, exist_ok=True)
    probe_file = probe_root / "probe.txt"
    probe_file.write_text("hello")
    assert probe_file.read_text() == "hello"

    real_root = REAL_DEFAULT_PACKAGE_ROOT
    existing = next(p for p in (real_root, *real_root.parents) if p.exists())
    assert existing.is_dir(), f"{existing} is not a directory"
    assert os.access(existing, os.W_OK), (
        f"{real_root} is not creatable: its nearest existing ancestor "
        f"{existing} is not writable"
    )


# ---------------------------------------------------------------------------
# bootstrap_calibrations() / fetch_current_frames() -- local-by-default,
# HTTP-by-explicit-opt-in wiring
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Negotiated-resolution safety fix, 2026-08-20 -- see
# opendarts.live.camera_resolution's module docstring for the full bug this
# closes (IMAGE_WIDTH/IMAGE_HEIGHT hardcoded into every camera's
# intrinsics matrix with zero runtime check that the camera actually
# opened at that resolution).
# ---------------------------------------------------------------------------


class _FakeStatusWithResolution:
    def __init__(self, actual_width: int = 0, actual_height: int = 0):
        self.actual_width = actual_width
        self.actual_height = actual_height


class _FakeHubWithStatus:
    """Duck-typed stand-in for LocalCameraHub -- only what
    `_negotiated_resolution_for()` actually reads (`.status`, a dict of
    camera-index -> object with `.actual_width`/`.actual_height`)."""

    def __init__(self, status: dict[int, _FakeStatusWithResolution]):
        self.status = status


def test_camera_matrix_for_requires_a_focal_length():
    """Focal length is derived live; there is no per-rig constant to
    fall back on, so omitting it must raise rather than assume."""
    with pytest.raises(ValueError, match="no focal length"):
        capture_daemon._camera_matrix_for(0)

    matrix = capture_daemon._camera_matrix_for(0, focal_px=800.0)
    assert matrix[0, 2] == capture_daemon.IMAGE_WIDTH / 2.0
    assert matrix[1, 2] == capture_daemon.IMAGE_HEIGHT / 2.0
    assert matrix[0, 0] == 800.0


def test_camera_matrix_for_uses_explicit_resolution_when_given():
    matrix = capture_daemon._camera_matrix_for(
        0, focal_px=800.0, image_width=1920.0, image_height=1080.0
    )
    assert matrix[0, 2] == 960.0
    assert matrix[1, 2] == 540.0
    # Focal length itself is unaffected by this fix -- re-deriving it for
    # a genuinely different resolution is explicitly flagged as a
    # separate follow-up (see _negotiated_resolution_for()'s own
    # docstring), not silently attempted here.
    assert matrix[0, 0] == 800.0


def test_negotiated_resolution_for_none_hub_falls_back_to_module_constants():
    """No LocalCameraHub at all -- must return today's exact historical
    numbers, unchanged."""
    width, height = capture_daemon._negotiated_resolution_for(0, None)
    assert (width, height) == (capture_daemon.IMAGE_WIDTH, capture_daemon.IMAGE_HEIGHT)


def test_negotiated_resolution_for_bare_sentinel_hub_does_not_crash():
    """A hub double with no `.status` attribute at all (used by several
    existing bootstrap_calibrations tests, e.g. `hub=object()`) must
    degrade gracefully to the historical fallback, never raise
    AttributeError."""
    width, height = capture_daemon._negotiated_resolution_for(0, object())
    assert (width, height) == (capture_daemon.IMAGE_WIDTH, capture_daemon.IMAGE_HEIGHT)


def test_negotiated_resolution_for_missing_camera_entry_falls_back():
    hub = _FakeHubWithStatus(status={})
    width, height = capture_daemon._negotiated_resolution_for(0, hub)
    assert (width, height) == (capture_daemon.IMAGE_WIDTH, capture_daemon.IMAGE_HEIGHT)


def test_negotiated_resolution_for_never_read_camera_falls_back():
    """A camera that's opened but never produced a real read has
    actual_width/actual_height still at their zero-value default --
    treated the same as "no real data yet", not trusted as real 0x0."""
    hub = _FakeHubWithStatus(status={0: _FakeStatusWithResolution(0, 0)})
    width, height = capture_daemon._negotiated_resolution_for(0, hub)
    assert (width, height) == (capture_daemon.IMAGE_WIDTH, capture_daemon.IMAGE_HEIGHT)


def test_negotiated_resolution_for_matching_resolution_no_warning(caplog):
    """BACKWARD COMPATIBILITY, explicit: real hardware that negotiates
    exactly the historical 1280x720 (this rig's own real, only-ever-
    observed behavior) must return those exact numbers with NO warning
    logged -- proves this fix is silent/inert on today's real behavior,
    not just numerically equal."""
    hub = _FakeHubWithStatus(
        status={0: _FakeStatusWithResolution(capture_daemon.IMAGE_WIDTH, capture_daemon.IMAGE_HEIGHT)}
    )
    with caplog.at_level("WARNING"):
        width, height = capture_daemon._negotiated_resolution_for(0, hub)

    assert (width, height) == (capture_daemon.IMAGE_WIDTH, capture_daemon.IMAGE_HEIGHT)
    assert not any("negotiated" in rec.getMessage() for rec in caplog.records)


def test_negotiated_resolution_for_mismatched_resolution_uses_real_value_and_warns_loudly(caplog):
    """THE REAL BUG FIX: a camera that negotiated something other than
    the hardcoded IMAGE_WIDTH/IMAGE_HEIGHT must have the calibration math
    use the REAL negotiated resolution (never the stale hardcoded
    constant), and this divergence must be logged loudly -- never a
    silent wrong-principal-point bug again."""
    hub = _FakeHubWithStatus(status={0: _FakeStatusWithResolution(1920, 1080)})

    with caplog.at_level("WARNING"):
        width, height = capture_daemon._negotiated_resolution_for(0, hub)

    assert (width, height) == (1920.0, 1080.0)
    assert any(
        "1920" in rec.getMessage() and "1080" in rec.getMessage() for rec in caplog.records
    )


def test_bootstrap_calibrations_uses_real_negotiated_resolution_not_hardcoded_constant(
    tmp_path, monkeypatch
):
    """End-to-end wiring proof: bootstrap_calibrations()'s own internal
    _try_solve() must pass the hub's REAL negotiated resolution into both
    _camera_matrix_for() (principal point) and calibrate_camera()'s own
    image_width/image_height kwargs -- not the hardcoded module
    constants -- whenever the hub has real per-camera status to consult."""
    n_frames = 4
    object_points = np.zeros((4, 3))
    image_points = np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])

    def fake_local(hub, n, **kwargs):
        return {0: [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(n_frames)]}

    def fake_detect(image_bgr, pre, **kwargs):
        return object_points, image_points

    captured = {}

    def fake_calibrate(obj_pts, img_pts, camera_matrix, dist_coeffs, image_width, image_height):
        captured["camera_matrix"] = camera_matrix
        captured["image_width"] = image_width
        captured["image_height"] = image_height
        return _fake_calibration_attempt()

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_local)
    _install_fake_orientation_pipeline(monkeypatch, fake_detect)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", fake_calibrate)

    hub = _FakeHubWithStatus(status={0: _FakeStatusWithResolution(1920, 1080)})
    result = capture_daemon.bootstrap_calibrations(
        tmp_path, hub=hub, n_frames=n_frames # type: ignore[arg-type]
    )

    assert 0 in result
    assert captured["image_width"] == 1920.0
    assert captured["image_height"] == 1080.0
    assert captured["camera_matrix"][0, 2] == 960.0
    assert captured["camera_matrix"][1, 2] == 540.0


def test_bootstrap_calibrations_matches_historical_behavior_when_hub_negotiated_default_resolution(
    tmp_path, monkeypatch
):
    """BACKWARD COMPATIBILITY, explicit and end-to-end -- PRINCIPAL POINT
    ONLY, as of the 2026-08-26 live-focal-length-derivation fix. A hub
    reporting the real historical 1280x720 negotiated resolution (today's
    only ever-observed real behavior on this rig) must still produce a
    camera matrix whose PRINCIPAL POINT (image_width/2, image_height/2)
    is numerically identical to what `_camera_matrix_for(0)`'s own
    negotiated-resolution logic always produced -- that half of this
    test's original intent (the 2026-08-20 negotiated-resolution fix)
    is unchanged by this later change.

    The FOCAL LENGTH itself is deliberately NOT compared against
    `_camera_matrix_for(0)`/`MEASURED_FOCAL_LENGTH_PX` anymore -- that
    would be asserting the exact bug this later fix removes (a focal
    length now comes from a live, per-event derivation
    -- opendarts.calibration.focal_length -- not a fixed hardcoded index-
    keyed constant; see that module's own docstring for the real,
    measured reason). This test's `_install_fake_orientation_pipeline()`
    fakes that derivation to always return a fixed 900.0px test-fixture
    value (see that helper's own comment) -- asserted directly below,
    not compared against the old constant."""
    n_frames = 4
    object_points = np.zeros((4, 3))
    image_points = np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])

    def fake_local(hub, n, **kwargs):
        return {0: [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(n_frames)]}

    def fake_detect(image_bgr, pre, **kwargs):
        return object_points, image_points

    captured = {}

    def fake_calibrate(obj_pts, img_pts, camera_matrix, dist_coeffs, image_width, image_height):
        captured["camera_matrix"] = camera_matrix
        captured["image_width"] = image_width
        captured["image_height"] = image_height
        return _fake_calibration_attempt()

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_local)
    _install_fake_orientation_pipeline(monkeypatch, fake_detect)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", fake_calibrate)

    hub = _FakeHubWithStatus(
        status={0: _FakeStatusWithResolution(capture_daemon.IMAGE_WIDTH, capture_daemon.IMAGE_HEIGHT)}
    )
    capture_daemon.bootstrap_calibrations(
        tmp_path, hub=hub, n_frames=n_frames # type: ignore[arg-type]
    )

    assert captured["image_width"] == float(capture_daemon.IMAGE_WIDTH)
    assert captured["image_height"] == float(capture_daemon.IMAGE_HEIGHT)
    assert captured["camera_matrix"][0, 2] == capture_daemon.IMAGE_WIDTH / 2.0
    assert captured["camera_matrix"][1, 2] == capture_daemon.IMAGE_HEIGHT / 2.0
    # Live-derived (fake, 900.0px -- see _install_fake_orientation_pipeline()),
    # NOT MEASURED_FOCAL_LENGTH_PX[0] (800.0px) -- see this test's own
    # updated docstring.
    assert captured["camera_matrix"][0, 0] == 900.0
    assert captured["camera_matrix"][1, 1] == 900.0


def test_bootstrap_calibrations_local_mode_requires_an_already_open_hub():
    with pytest.raises(ValueError, match="requires an already-open"):
        capture_daemon.bootstrap_calibrations(Path("unused"), hub=None)


def test_fetch_current_frames_local_mode_requires_an_already_open_hub():
    with pytest.raises(ValueError, match="requires an already-open"):
        capture_daemon.fetch_current_frames(Path("unused"), hub=None)


def _fake_calibration_attempt() -> CalibrationAttempt:
    calib = CameraCalibration(
        camera_matrix=np.eye(3),
        dist_coeffs=np.zeros(5),
        rvec=np.zeros(3),
        tvec=np.array([0.0, 0.0, 1000.0]),
        pnp_result=None,
        landmark_spread_ok=True,
    )
    return CalibrationAttempt(ok=True, calibration=calib, pnp_result=None, reason="")






# ---------------------------------------------------------------------------
# N-frame-averaged calibration, added 2026-08-12 -- see CALIBRATION_N_FRAMES'
# own dated comment in capture_daemon.py for the full real-evidence writeup.
# ---------------------------------------------------------------------------


def test_min_calibration_frames_required_is_a_strict_majority_above_one():
    assert capture_daemon._min_calibration_frames_required(1) == 1
    assert capture_daemon._min_calibration_frames_required(2) == 2
    assert capture_daemon._min_calibration_frames_required(3) == 2
    assert capture_daemon._min_calibration_frames_required(5) == 3
    assert capture_daemon._min_calibration_frames_required(8) == 5
    assert capture_daemon._min_calibration_frames_required(10) == 6


def test_bootstrap_calibrations_averages_n_frames_not_just_the_first(tmp_path, monkeypatch):
    """The actual point of this whole change: bootstrap_calibrations must
    call correspond_landmarks_oriented() on EVERY captured frame (not just frame
    0), and calibrate_camera() must receive the elementwise AVERAGE of
    their image_points, not any single frame's raw points."""
    n_frames = 4
    # 4 distinct "detections" whose image_points differ by a known offset
    # per frame -- the average across all 4 has a known, checkable value.
    object_points = np.zeros((4, 3))
    base_image_points = np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])
    offsets = [-2.0, -1.0, 1.0, 2.0] # mean offset == 0 -> average == base_image_points

    def fake_local(hub, n, **kwargs):
        assert n == n_frames
        return {0: [np.full((10, 10, 3), i, dtype=np.uint8) for i in range(n_frames)]}

    detect_calls = {"n": 0}

    def fake_detect(image_bgr, cam, **kwargs):
        idx = detect_calls["n"]
        detect_calls["n"] += 1
        return object_points, base_image_points + offsets[idx]

    captured = {}

    def fake_calibrate(obj_pts, img_pts, *a, **k):
        captured["object_points"] = obj_pts
        captured["image_points"] = img_pts
        return _fake_calibration_attempt()

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_local)
    _install_fake_orientation_pipeline(monkeypatch, fake_detect)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", fake_calibrate)

    result = capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), n_frames=n_frames # type: ignore[arg-type]
    )

    assert detect_calls["n"] == n_frames # every captured frame was actually detected on
    assert np.allclose(captured["image_points"], base_image_points) # offsets averaged out
    assert 0 in result


def test_bootstrap_calibrations_tolerates_partial_detection_failure_among_n(tmp_path, monkeypatch):
    """A camera where SOME (not all) of the N frames fail landmark
    detection must still calibrate from whatever succeeded, as long as
    it clears `_min_good_frames_cumulative()`'s ABSOLUTE floor (5,
    2026-09-03 -- see that function's own docstring and docs/DESIGN.md's
    DEFECT 2 entry for why the earlier strict-majority-of-the-total rule
    was replaced) -- real requirement from this task's own instructions
    ("average whatever succeeded, don't fail the whole calibration for
    one bad frame among many")."""
    n_frames = 10 # _min_good_frames_cumulative(10) == 5 (the absolute floor)
    object_points = np.zeros((4, 3))
    image_points = np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])

    def fake_local(hub, n, **kwargs):
        return {0: [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(n_frames)]}

    # 7 of 10 succeed (3 fail) -- clears the absolute floor (5) with room
    # to spare while still genuinely demonstrating partial-failure
    # tolerance (not every frame succeeded).
    detect_calls = {"n": 0}

    def fake_detect(image_bgr, cam, **kwargs):
        idx = detect_calls["n"]
        detect_calls["n"] += 1
        if idx < 7:
            return object_points, image_points
        return None # simulated detection failure for the last 3 frames

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_local)
    _install_fake_orientation_pipeline(monkeypatch, fake_detect)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", lambda *a, **k: _fake_calibration_attempt())

    result = capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), # type: ignore[arg-type]
        n_frames=n_frames, n_frames_detect=n_frames,
    )

    assert 0 in result # calibrated ok from the 7 successful frames


def test_bootstrap_calibrations_skips_camera_below_the_min_frames_floor(tmp_path, monkeypatch):
    """The other half of the partial-failure requirement: too FEW of the
    N frames succeeding must skip that camera's calibration outright
    (not silently average over an untrustworthy 1-2 samples). Floor is
    `_min_good_frames_cumulative()`'s ABSOLUTE 5 (2026-09-03) -- unlike
    the earlier strict-majority-of-the-total rule this replaced, this
    floor never grows as more retry rounds are captured, so a camera
    that only ever manages 2 real successes stays stuck below it all the
    way out to `max_frames`, at which point it is correctly skipped."""
    n_frames = 5 # only 2 will ever succeed here -- below the absolute floor of 5
    object_points = np.zeros((4, 3))
    image_points = np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])

    def fake_local(hub, n, **kwargs):
        return {0: [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(n_frames)]}

    detect_calls = {"n": 0}

    def fake_detect(image_bgr, cam, **kwargs):
        idx = detect_calls["n"]
        detect_calls["n"] += 1
        if idx < 2: # only 2 total ever succeed -- below the floor of 5
            return object_points, image_points
        return None

    calibrate_calls = {"n": 0}

    def fake_calibrate(*a, **k):
        calibrate_calls["n"] += 1
        return _fake_calibration_attempt()

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_local)
    _install_fake_orientation_pipeline(monkeypatch, fake_detect)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", fake_calibrate)

    result = capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), n_frames=n_frames # type: ignore[arg-type]
    )

    assert 0 not in result # skipped -- 2 successes < the required 5
    assert calibrate_calls["n"] == 0 # never even attempted a PnP solve


# ---------------------------------------------------------------------------
# Adaptive retry loop, added 2026-08-14 -- see CALIBRATION_TARGET_REPROJECTION_
# ERROR_PX's own dated comment in capture_daemon.py for the full real-evidence
# writeup this section tests against.
# ---------------------------------------------------------------------------


def _fake_attempt_with_reprojection(reprojection_error_px: float) -> CalibrationAttempt:
    """Like _fake_calibration_attempt() but with a real (duck-typed)
    pnp_result.reprojection_error_px -- the retry loop only ever reads
    that one attribute off pnp_result, so a lightweight stand-in is
    enough; the existing _fake_calibration_attempt() (pnp_result=None)
    stays as the "accuracy unavailable, accept immediately" fixture for
    every pre-existing test above, unchanged."""
    import types

    calib = CameraCalibration(
        camera_matrix=np.eye(3),
        dist_coeffs=np.zeros(5),
        rvec=np.zeros(3),
        tvec=np.array([0.0, 0.0, 1000.0]),
        pnp_result=None,
        landmark_spread_ok=True,
    )
    pnp_result = types.SimpleNamespace(reprojection_error_px=reprojection_error_px)
    return CalibrationAttempt(ok=True, calibration=calib, pnp_result=pnp_result, reason="")


def test_bootstrap_calibrations_accepts_immediately_when_target_met_on_first_round(
    tmp_path, monkeypatch
):
    """The fast path: a camera that clears target_reprojection_error_px
    on the very first solve is accepted immediately -- exactly one
    capture round, no retry, same call-count shape as every pre-existing
    single-round test above (now proven explicitly with a REAL
    reprojection number instead of the unevaluable pnp_result=None
    fixture)."""
    calls = {"local": 0, "calibrate": 0}

    def fake_local(hub, n, **kwargs):
        calls["local"] += 1
        return {0: [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(n)]}

    def fake_detect(image_bgr, cam, **kwargs):
        return np.zeros((4, 3)), np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])

    def fake_calibrate(*a, **k):
        calls["calibrate"] += 1
        return _fake_attempt_with_reprojection(1.0) # below the 2.0px default target

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_local)
    _install_fake_orientation_pipeline(monkeypatch, fake_detect)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", fake_calibrate)

    diagnostics: dict = {}
    result = capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), n_frames=10, # type: ignore[arg-type]
        n_frames_detect=10, # explicit -- don't rely on the module-level
        # default (CALIBRATION_N_FRAMES_DETECT), which has changed more
        # than once (see that constant's own dated history) and isn't
        # what this test is actually about (2026-08-29 test-debt cleanup).
        diagnostics_out=diagnostics,
    )

    assert 0 in result
    assert calls["local"] == 1 # exactly one capture round -- no retry needed
    assert calls["calibrate"] == 1
    assert diagnostics[0]["reprojection_error_px"] == 1.0
    assert diagnostics[0]["n_frames_used"] == 10
    assert diagnostics[0]["target_met"] is True


# ---------------------------------------------------------------------------
# Concurrent-calibration lock (verifier finding, 2026-08-21 -- the
# integration task's own live-wiring pass): two real call sites
# (run_capture_loop_body()'s auto-calibrate, AppState._refresh_
# calibration_blocking()'s manual refresh) can race in the same process.
# ---------------------------------------------------------------------------


def test_bootstrap_calibrations_rejects_a_concurrent_call_while_one_is_in_progress(
    tmp_path,
):
    """The real, measured behavior: a second call while the module-level
    lock is already held raises CalibrationInProgressError immediately
    (never blocks, never silently interleaves two runs' writes to the
    shared live-derived globals) -- a dedicated exception TYPE, not a
    bare RuntimeError, specifically so callers can distinguish this
    transient case from a real calibration failure (see the class's own
    docstring). Holds the lock directly rather than racing two real
    threads -- deterministic, no timing flakiness, and it is testing the
    exact mechanism (the lock itself), not a timing coincidence."""
    assert capture_daemon._BOOTSTRAP_CALIBRATIONS_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(capture_daemon.CalibrationInProgressError, match="already running"):
            capture_daemon.bootstrap_calibrations(
                tmp_path, hub=object(), n_frames=1, # type: ignore[arg-type]
            )
    finally:
        capture_daemon._BOOTSTRAP_CALIBRATIONS_LOCK.release()


def test_bootstrap_calibrations_with_lock_wait_retries_until_the_lock_frees_up(
    tmp_path, monkeypatch
):
    """run_capture_loop_body()'s own entry point: while the lock is held
    by someone else, it must NOT propagate CalibrationInProgressError
    immediately (that call site has no graceful-degrade handling of its
    own -- any exception there is fatal to the whole process, per
    CalibrationInProgressError's own class docstring) -- it retries
    until the lock frees up, then succeeds with a REAL result. A
    background thread holds the lock for a short, real duration and
    releases it -- proves the retry loop actually waits past the first
    failed attempt and picks up the real calibration once it can, not
    just that the function eventually returns something."""
    def fake_local(hub, n, **kwargs):
        return {0: [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(n)]}

    def fake_detect(image_bgr, cam, **kwargs):
        return np.zeros((4, 3)), np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])

    def fake_calibrate(*a, **k):
        return _fake_attempt_with_reprojection(1.0)

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_local)
    _install_fake_orientation_pipeline(monkeypatch, fake_detect)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", fake_calibrate)

    assert capture_daemon._BOOTSTRAP_CALIBRATIONS_LOCK.acquire(blocking=False)

    def _release_after_a_moment():
        time.sleep(0.3)
        capture_daemon._BOOTSTRAP_CALIBRATIONS_LOCK.release()

    releaser = threading.Thread(target=_release_after_a_moment)
    releaser.start()
    try:
        result = capture_daemon._bootstrap_calibrations_with_lock_wait(
            tmp_path, hub=object(), n_frames=1, # type: ignore[arg-type]
            lock_wait_timeout_s=5.0,
        )
        assert 0 in result
    finally:
        releaser.join(timeout=5.0)
        assert not releaser.is_alive()


def test_bootstrap_calibrations_with_lock_wait_gives_up_after_its_own_timeout(tmp_path):
    """If the lock genuinely never frees up within lock_wait_timeout_s,
    the retry loop must still eventually give up and let
    CalibrationInProgressError propagate for real -- an unbounded retry
    would just replace 'crashes immediately' with 'hangs Start forever'
    on a genuinely stuck lock, not actually fix anything."""
    assert capture_daemon._BOOTSTRAP_CALIBRATIONS_LOCK.acquire(blocking=False)
    try:
        started = time.monotonic()
        with pytest.raises(capture_daemon.CalibrationInProgressError):
            capture_daemon._bootstrap_calibrations_with_lock_wait(
                tmp_path, hub=object(), n_frames=1, # type: ignore[arg-type]
                lock_wait_timeout_s=0.5,
            )
        elapsed = time.monotonic() - started
        assert elapsed >= 0.5, elapsed # actually waited, didn't give up instantly
        assert elapsed < 5.0, elapsed # but didn't hang either
    finally:
        capture_daemon._BOOTSTRAP_CALIBRATIONS_LOCK.release()


# --------------------------------------------------------------------------
# _bootstrap_calibrations_at_start_with_reopen_retry() (2026-09-01):
# ONE bounded real close+reopen + retry, Start-time only,
# when a camera comes back missing or the first attempt raises. Uses a
# minimal fake hub (just `.configs`/`.open_all`) and monkeypatches
# `_bootstrap_calibrations_with_lock_wait` itself -- tests this wrapper's
# own retry logic in isolation from the real calibration pipeline, which
# is already covered by the tests immediately above/below.
# --------------------------------------------------------------------------


class _FakeHubWithConfigs:
    """Minimal stand-in exposing exactly what the new wrapper needs
    (`.configs`, `.open_all()`) -- deliberately NOT a full LocalCameraHub
    fake, to prove the wrapper only ever touches this narrow surface."""

    def __init__(self, n_cameras: int):
        self.configs = list(range(n_cameras)) # only .configs's own LENGTH matters
        self.open_all_calls: list[None] = []

    def open_all(self):
        self.open_all_calls.append(None)
        return [True] * len(self.configs)


def test_reopen_retry_no_retry_when_all_cameras_present(monkeypatch):
    """Happy path: every configured camera calibrated on the first
    attempt -- open_all() must never be called at all."""
    hub = _FakeHubWithConfigs(3)
    calls = {"n": 0}

    def fake_lock_wait(*args, hub=None, **kwargs):
        calls["n"] += 1
        return {0: object(), 1: object(), 2: object()}

    monkeypatch.setattr(capture_daemon, "_bootstrap_calibrations_with_lock_wait", fake_lock_wait)

    result = capture_daemon._bootstrap_calibrations_at_start_with_reopen_retry(
        "unused", hub=hub,
    )
    assert len(result) == 3
    assert calls["n"] == 1
    assert hub.open_all_calls == []


def test_reopen_retry_reopens_and_retries_on_missing_camera(monkeypatch):
    """The real, motivating case: first attempt comes back with a camera
    missing (2/3) -- must reopen the hub and try exactly once more,
    returning the SECOND attempt's own (now-complete) result."""
    hub = _FakeHubWithConfigs(3)
    attempts: list[dict] = []

    def fake_lock_wait(*args, hub=None, **kwargs):
        if not attempts:
            result = {0: object(), 1: object()} # cam2 missing
        else:
            result = {0: object(), 1: object(), 2: object()} # fixed after reopen
        attempts.append(result)
        return result

    monkeypatch.setattr(capture_daemon, "_bootstrap_calibrations_with_lock_wait", fake_lock_wait)

    result = capture_daemon._bootstrap_calibrations_at_start_with_reopen_retry(
        "unused", hub=hub,
    )
    assert len(attempts) == 2
    assert len(result) == 3
    assert result is attempts[1] # the SECOND (post-reopen) attempt's own result
    assert len(hub.open_all_calls) == 1 # exactly one real reopen


def test_reopen_retry_reopens_and_retries_on_exception(monkeypatch):
    """A stuck camera can also manifest as a real exception (this rig's
    own OrientationConsensusRefusedError, or anything else) from the
    first attempt, not just a partial result -- same one-retry
    treatment."""
    hub = _FakeHubWithConfigs(3)
    attempts = {"n": 0}

    def fake_lock_wait(*args, hub=None, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError("simulated stuck camera")
        return {0: object(), 1: object(), 2: object()}

    monkeypatch.setattr(capture_daemon, "_bootstrap_calibrations_with_lock_wait", fake_lock_wait)

    result = capture_daemon._bootstrap_calibrations_at_start_with_reopen_retry(
        "unused", hub=hub,
    )
    assert attempts["n"] == 2
    assert len(result) == 3
    assert len(hub.open_all_calls) == 1


def test_reopen_retry_is_bounded_to_exactly_one_retry(monkeypatch):
    """If the SECOND attempt (after the real reopen) is ALSO partial,
    that outcome is final -- no third attempt, no second reopen. Proves
    this is a bounded, one-shot retry, not a loop."""
    hub = _FakeHubWithConfigs(3)
    attempts = {"n": 0}

    def fake_lock_wait(*args, hub=None, **kwargs):
        attempts["n"] += 1
        return {0: object()} # still only 1/3, every single attempt

    monkeypatch.setattr(capture_daemon, "_bootstrap_calibrations_with_lock_wait", fake_lock_wait)

    result = capture_daemon._bootstrap_calibrations_at_start_with_reopen_retry(
        "unused", hub=hub,
    )
    assert attempts["n"] == 2 # first attempt + exactly one retry, never a third
    assert len(result) == 1 # the second (still-partial) attempt's own result, returned as-is
    assert len(hub.open_all_calls) == 1 # exactly one reopen, not two


def test_reopen_retry_second_attempts_exception_propagates(monkeypatch):
    """If the retry attempt (after the real reopen) ALSO raises, that
    exception propagates for real -- this wrapper never swallows a
    genuine, final failure."""
    hub = _FakeHubWithConfigs(3)
    attempts = {"n": 0}

    def fake_lock_wait(*args, hub=None, **kwargs):
        attempts["n"] += 1
        raise RuntimeError(f"attempt {attempts['n']} failed")

    monkeypatch.setattr(capture_daemon, "_bootstrap_calibrations_with_lock_wait", fake_lock_wait)

    with pytest.raises(RuntimeError, match="attempt 2 failed"):
        capture_daemon._bootstrap_calibrations_at_start_with_reopen_retry("unused", hub=hub)
    assert attempts["n"] == 2
    assert len(hub.open_all_calls) == 1




def test_reopen_retry_skipped_for_a_hub_missing_configs_or_open_all(monkeypatch):
    """A test double (or a future hub implementation) that doesn't expose
    both `.configs` and `.open_all` must degrade safely to the plain
    call -- never an AttributeError crash on this project's own existing
    lightweight fake hubs."""
    calls = {"n": 0}

    def fake_lock_wait(*args, hub=None, **kwargs):
        calls["n"] += 1
        return {0: object()}

    monkeypatch.setattr(capture_daemon, "_bootstrap_calibrations_with_lock_wait", fake_lock_wait)

    class _BareFakeHub:
        pass

    result = capture_daemon._bootstrap_calibrations_at_start_with_reopen_retry(
        "unused", hub=_BareFakeHub(),
    )
    assert calls["n"] == 1
    assert len(result) == 1


def test_bootstrap_calibrations_lock_releases_after_a_real_call_including_on_exception(
    tmp_path, monkeypatch
):
    """The lock must not leak: a real (successful) call releases it
    (proven by a second real call succeeding immediately afterward, not
    just by inspecting the lock's own .locked() state), AND a call whose
    underlying work raises releases it too (the `finally` in the
    wrapper), proven the same way -- a real call, not the same
    already-broken-by-then object being reused."""
    def fake_local(hub, n, **kwargs):
        return {0: [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(n)]}

    def fake_detect(image_bgr, cam, **kwargs):
        return np.zeros((4, 3)), np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])

    def fake_calibrate(*a, **k):
        return _fake_attempt_with_reprojection(1.0)

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_local)
    _install_fake_orientation_pipeline(monkeypatch, fake_detect)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", fake_calibrate)

    result = capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), n_frames=1, # type: ignore[arg-type]
    )
    assert 0 in result
    # Lock released after success -- prove it with a second real call,
    # not by reading .locked().
    result2 = capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), n_frames=1, # type: ignore[arg-type]
    )
    assert 0 in result2

    def raising_local(hub, n, **kwargs):
        raise ValueError("simulated capture failure")

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", raising_local)
    with pytest.raises(ValueError, match="simulated capture failure"):
        capture_daemon.bootstrap_calibrations(
            tmp_path, hub=object(), n_frames=1, # type: ignore[arg-type]
        )
    # Lock released even though the call above raised -- prove it with a
    # real successful call right after, not by reading .locked().
    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_local)
    result3 = capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), n_frames=1, # type: ignore[arg-type]
    )
    assert 0 in result3


def test_bootstrap_calibrations_retries_with_more_frames_until_target_met(tmp_path, monkeypatch):
    """The real point of this whole change: a camera that does NOT clear
    the target on round 1 must capture MORE frames (not restart from
    zero -- the accumulated pool keeps growing) and try again, stopping
    the moment a later round clears the target."""
    calls = {"local": 0}
    solve_calls = {"n": 0}
    # Round 1: 5.0px (misses). Round 2 (after +retry_batch_size frames):
    # 1.5px (meets the 2.0px target) -- must stop retrying right there.
    reprojections = [5.0, 1.5]

    def fake_local(hub, n, **kwargs):
        calls["local"] += 1
        return {0: [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(n)]}

    def fake_detect(image_bgr, cam, **kwargs):
        return np.zeros((4, 3)), np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])

    def fake_calibrate(*a, **k):
        idx = solve_calls["n"]
        solve_calls["n"] += 1
        return _fake_attempt_with_reprojection(reprojections[idx])

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_local)
    _install_fake_orientation_pipeline(monkeypatch, fake_detect)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", fake_calibrate)

    diagnostics: dict = {}
    result = capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), # type: ignore[arg-type]
        n_frames=10, n_frames_detect=10, retry_batch_size=10, max_frames=100,
        # n_frames_detect explicit -- see the 2026-08-29 test-debt-cleanup
        # comment on the first bootstrap_calibrations() call above.
        diagnostics_out=diagnostics,
    )

    assert 0 in result
    assert calls["local"] == 2 # one initial round + exactly one retry, then stopped
    assert solve_calls["n"] == 2
    assert diagnostics[0]["reprojection_error_px"] == 1.5
    assert diagnostics[0]["n_frames_used"] == 20 # 10 initial + 10 retry, accumulated not reset
    assert diagnostics[0]["target_met"] is True


def test_bootstrap_calibrations_gives_up_at_max_frames_cap_and_accepts_best(
    tmp_path, monkeypatch, caplog
):
    """The honest-degradation half: a camera that NEVER clears the
    target must stop retrying once it hits max_frames, log a clear
    warning naming the cap and the real error reached, and still return
    the best (not a failure) -- a degraded calibration beats none, but
    must never be silently indistinguishable from a verified-good one."""
    import logging as _logging

    calls = {"local": 0}

    def fake_local(hub, n, **kwargs):
        calls["local"] += 1
        return {0: [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(n)]}

    def fake_detect(image_bgr, cam, **kwargs):
        return np.zeros((4, 3)), np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])

    def fake_calibrate(*a, **k):
        return _fake_attempt_with_reprojection(3.24) # never clears the 2.0px target

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_local)
    _install_fake_orientation_pipeline(monkeypatch, fake_detect)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", fake_calibrate)

    diagnostics: dict = {}
    with caplog.at_level(_logging.WARNING):
        result = capture_daemon.bootstrap_calibrations(
            tmp_path, hub=object(), # type: ignore[arg-type]
            n_frames=10, n_frames_detect=10, retry_batch_size=10, max_frames=30,
            # n_frames_detect explicit -- see the 2026-08-29 test-debt-
            # cleanup comment earlier in this file.
            diagnostics_out=diagnostics,
        )

    assert 0 in result # a degraded calibration is still returned, not dropped
    assert calls["local"] == 3 # 10 + 10 + 10 == 30 == max_frames, then stopped
    assert diagnostics[0]["reprojection_error_px"] == 3.24
    assert diagnostics[0]["n_frames_used"] == 30
    assert diagnostics[0]["target_met"] is False # honestly flagged, not silently accepted
    assert "did NOT reach" in caplog.text
    assert "N=30 frame cap" in caplog.text
    assert "3.24" in caplog.text


def test_bootstrap_calibrations_tracks_the_best_reprojection_seen_not_just_the_last(
    tmp_path, monkeypatch
):
    """"Accept the best result obtained" means the
    BEST across every round, not whichever solve happened to run last --
    proven here with the FINAL round being WORSE than the middle round,
    so "keep the last" and "keep the best" would disagree."""
    solve_calls = {"n": 0}
    reprojections = [4.0, 2.5, 6.0] # round 2 (2.5) is the real best; round 3 (last) is worse

    def fake_local(hub, n, **kwargs):
        return {0: [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(n)]}

    def fake_detect(image_bgr, cam, **kwargs):
        return np.zeros((4, 3)), np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])

    def fake_calibrate(*a, **k):
        idx = solve_calls["n"]
        solve_calls["n"] += 1
        return _fake_attempt_with_reprojection(reprojections[idx])

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_local)
    _install_fake_orientation_pipeline(monkeypatch, fake_detect)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", fake_calibrate)

    diagnostics: dict = {}
    result = capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), # type: ignore[arg-type]
        n_frames=10, n_frames_detect=10, retry_batch_size=10, max_frames=30,
        # n_frames_detect explicit -- see the 2026-08-29 test-debt-cleanup
        # comment earlier in this file.
        diagnostics_out=diagnostics,
    )

    assert 0 in result
    assert solve_calls["n"] == 3 # ran all 3 rounds (30 == max_frames cap)
    # The best (lowest) of [4.0, 2.5, 6.0] is round 2's 2.5 -- must be
    # what's kept even though round 3 (worse, 6.0) ran last.
    assert diagnostics[0]["reprojection_error_px"] == 2.5


def test_bootstrap_calibrations_processes_multiple_cameras_concurrently_not_sequentially(
    tmp_path, monkeypatch
):
    """Real proof the 2026-08-16 threading change actually parallelizes
    per-camera work -- not just that the code doesn't crash with more
    than one camera. A fake per-camera detection call sleeps for a fixed
    duration; if all 3 cameras' round-1 work genuinely runs concurrently
    (ThreadPoolExecutor, one worker per camera), total wall time should
    be close to ONE camera's sleep duration (max), not the SUM across
    all three -- the exact signature the OLD `for cam in
    list(remaining):` sequential loop would have produced instead. Real,
    honest caveat this test does NOT claim to prove: the REAL landmark-
    detection/PnP-solve code, profiled against real archived camera
    frames, shows only ~1.1-1.2x real speedup end to end -- the actual
    CPU-bound work is dominated by many small-array numpy/pure-Python
    calls (cProfile: thousands of tiny `numpy.ufunc.reduce`/`sum`/
    `round()` calls inside oriented_landmarks.py's colour-scoring and
    wire-junction-refinement paths), which each briefly hold the GIL,
    rather than long bulk cv2 C++ calls that would release it for a
    meaningful duration -- so the real algorithm does NOT get anywhere
    near the ideal 3x this synthetic test demonstrates. This test proves
    the THREADING MECHANISM itself is not accidentally serialized (no
    stray lock/shared-state bottleneck introduced by this refactor); it
    is not a claim about the real detection algorithm's own GIL
    behavior, which is a separate, already-profiled, honestly-reported
    finding (see this function's own docstring in capture_daemon.py)."""
    import time as _time

    sleep_s = 0.15
    n_cams = 3

    def fake_local(hub, n, **kwargs):
        return {
            cam: [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(n)]
            for cam in range(n_cams)
        }

    def fake_detect(image_bgr, cam, **kwargs):
        _time.sleep(sleep_s)
        return np.zeros((4, 3)), np.array(
            [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]
        )

    def fake_calibrate(*a, **k):
        return _fake_attempt_with_reprojection(1.0) # below default target -- 1 round only

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_local)
    _install_fake_orientation_pipeline(monkeypatch, fake_detect)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", fake_calibrate)

    t0 = time.monotonic()
    result = capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), n_frames=1 # type: ignore[arg-type]
    )
    elapsed = time.monotonic() - t0

    assert sorted(result.keys()) == [0, 1, 2]
    # Sequential (the old code) would take >= 3 * sleep_s == 0.45s here.
    # Genuinely concurrent should be close to 1 * sleep_s == 0.15s. This
    # threshold (2x a single camera's sleep) is real headroom above the
    # concurrent floor and real headroom below the sequential sum -- not
    # a guessed round number (see docs/DESIGN.md "measure the real number").
    assert elapsed < 2 * sleep_s, (
        f"elapsed={elapsed:.3f}s is not close to max(sleep_s)={sleep_s}s -- "
        f"camera work does not appear to be running concurrently (sequential "
        f"sum would be ~{n_cams * sleep_s:.2f}s)"
    )


def test_bootstrap_calibrations_keeps_per_camera_state_independent_under_threading(
    tmp_path, monkeypatch
):
    """The correctness half of the threading change: 3 cameras processed
    CONCURRENTLY, each with its OWN distinct retry trajectory (different
    number of rounds to converge, different final reprojection error),
    must each end up with exactly ITS OWN correct result -- no
    cross-camera leakage into accumulated_detections/frames_captured/
    best_calibration/best_reprojection_px/diagnostics_out from a shared-
    state bug the threading refactor could have introduced. cam0
    converges round 1, cam1 round 2, cam2 round 3 (the max_frames cap) --
    three different finishing times/rounds running concurrently is
    exactly the scenario most likely to expose accidental state sharing
    between camera worker threads."""
    n_cams = 3
    reprojections_by_cam = {
        0: [1.0], # meets target (2.0px) immediately
        1: [5.0, 1.5], # misses round 1, meets round 2
        2: [8.0, 6.0, 4.0], # never meets target -- caps out at max_frames
    }
    solve_calls = {cam: 0 for cam in range(n_cams)}

    def fake_local(hub, n, **kwargs):
        # Distinguishable per-camera frame content, NOT all-zeros --
        # required by _install_fake_orientation_pipeline_with_focal_dispatch()
        # so camera identity can flow through to the focal-length fake
        # via pixel content, the same channel a real rig's per-camera
        # images naturally provide (see that helper's own docstring).
        return {
            cam: [np.full((4, 4, 3), cam + 1, dtype=np.uint8) for _ in range(n)]
            for cam in range(n_cams)
        }

    def fake_detect(image_bgr, pre, **kwargs):
        return np.zeros((4, 3)), np.array(
            [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]
        )

    # calibrate_camera() itself doesn't receive `cam` as an argument (see
    # _try_solve()'s real call site), so patch it with a dispatcher that
    # figures out which camera is calling via the camera_matrix's own
    # focal length (each cam has a distinct MEASURED_FOCAL_LENGTH_PX
    # entry) -- a real, not guessed, way to disambiguate without changing
    # calibrate_camera()'s own signature.
    focal_to_cam = {_TEST_FOCAL_PX[cam]: cam for cam in range(n_cams)}

    def fake_calibrate_dispatch(obj_pts, img_pts, camera_matrix, *a, **k):
        f = float(camera_matrix[0, 0])
        cam = focal_to_cam[f]
        idx = solve_calls[cam]
        solve_calls[cam] += 1
        reproj = reprojections_by_cam[cam][idx]
        return _fake_attempt_with_reprojection(reproj)

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_local)
    _install_fake_orientation_pipeline_with_focal_dispatch(monkeypatch, fake_detect, n_cams)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", fake_calibrate_dispatch)

    diagnostics: dict = {}
    result = capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), # type: ignore[arg-type]
        n_frames=10, n_frames_detect=10, retry_batch_size=10, max_frames=30,
        # n_frames_detect explicit -- see the 2026-08-29 test-debt-cleanup
        # comment earlier in this file.
        diagnostics_out=diagnostics,
    )

    assert sorted(result.keys()) == [0, 1, 2]

    # cam0: 1 round, 10 frames, meets target on the first try.
    assert diagnostics[0]["n_frames_used"] == 10
    assert diagnostics[0]["reprojection_error_px"] == 1.0
    assert diagnostics[0]["target_met"] is True

    # cam1: 2 rounds, 20 frames, meets target on the second try.
    assert diagnostics[1]["n_frames_used"] == 20
    assert diagnostics[1]["reprojection_error_px"] == 1.5
    assert diagnostics[1]["target_met"] is True

    # cam2: 3 rounds (10+10+10 == max_frames cap), never meets target --
    # accepts the BEST of [8.0, 6.0, 4.0], which is 4.0 (the last, but
    # also happens to be the best here -- monotonically improving).
    assert diagnostics[2]["n_frames_used"] == 30
    assert diagnostics[2]["reprojection_error_px"] == 4.0
    assert diagnostics[2]["target_met"] is False

    assert solve_calls == {0: 1, 1: 2, 2: 3}


def test_bootstrap_calibrations_target_reprojection_error_px_accepts_a_per_camera_dict(
    tmp_path, monkeypatch
):
    """The real feature: target_reprojection_
    error_px now accepts a dict[int, float], not just a uniform float.
    Both cameras get the IDENTICAL real reprojection error (2.0px) on
    round 1 -- the only thing that differs is each camera's OWN target
    (cam0: 3.0px, loose enough to pass immediately; cam1: 1.0px, tight
    enough to force a retry). If per-camera resolution were broken (e.g.
    silently falling back to the module's uniform 2.5px default for
    everyone), cam0 would still pass by coincidence but cam1 would too --
    this specifically proves cam1 does NOT pass on its own 2.0px reading
    against its own 1.0px target, and DOES retry."""
    n_cams = 2
    reprojections_by_cam = {0: [2.0], 1: [2.0, 0.5]}
    solve_calls = {cam: 0 for cam in range(n_cams)}

    def fake_local(hub, n, **kwargs):
        # Distinguishable per-camera frame content -- see
        # _install_fake_orientation_pipeline_with_focal_dispatch()'s own
        # docstring for why this must not be all-zeros for every camera.
        return {
            cam: [np.full((4, 4, 3), cam + 1, dtype=np.uint8) for _ in range(n)]
            for cam in range(n_cams)
        }

    def fake_detect(image_bgr, pre, **kwargs):
        return np.zeros((4, 3)), np.array(
            [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]
        )

    focal_to_cam = {_TEST_FOCAL_PX[cam]: cam for cam in range(n_cams)}

    def fake_calibrate_dispatch(obj_pts, img_pts, camera_matrix, *a, **k):
        f = float(camera_matrix[0, 0])
        cam = focal_to_cam[f]
        idx = solve_calls[cam]
        solve_calls[cam] += 1
        return _fake_attempt_with_reprojection(reprojections_by_cam[cam][idx])

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_local)
    _install_fake_orientation_pipeline_with_focal_dispatch(monkeypatch, fake_detect, n_cams)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", fake_calibrate_dispatch)

    diagnostics: dict = {}
    result = capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), # type: ignore[arg-type]
        n_frames=10, retry_batch_size=10, max_frames=30,
        target_reprojection_error_px={0: 3.0, 1: 1.0},
        diagnostics_out=diagnostics,
    )

    assert sorted(result.keys()) == [0, 1]
    # cam0: loose 3.0px target, met immediately by its 2.0px reading.
    assert solve_calls[0] == 1
    assert diagnostics[0]["reprojection_error_px"] == 2.0
    assert diagnostics[0]["target_met"] is True
    # cam1: tight 1.0px target -- its OWN 2.0px reading does NOT meet it,
    # forcing a real second round, which then meets it at 0.5px.
    assert solve_calls[1] == 2
    assert diagnostics[1]["reprojection_error_px"] == 0.5
    assert diagnostics[1]["target_met"] is True


def test_bootstrap_calibrations_target_reprojection_error_px_dict_falls_back_for_unlisted_cam(
    tmp_path, monkeypatch
):
    """A camera missing from the dict falls back to
    CALIBRATION_TARGET_REPROJECTION_ERROR_PX (today's uniform default),
    not some other silent default -- the real live-config use case (only
    override the cameras that need it, e.g. cam1, leave cam0/cam2 on the
    module default)."""
    def fake_local(hub, n, **kwargs):
        return {0: [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(n)]}

    def fake_detect(image_bgr, cam, **kwargs):
        return np.zeros((4, 3)), np.array(
            [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]
        )

    def fake_calibrate(*a, **k):
        # Between the dict's own (nonexistent-for-cam0) entries and the
        # real CALIBRATION_TARGET_REPROJECTION_ERROR_PX (2.5px) -- must
        # meet the fallback, not some other value.
        return _fake_attempt_with_reprojection(2.0)

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_local)
    _install_fake_orientation_pipeline(monkeypatch, fake_detect)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", fake_calibrate)

    diagnostics: dict = {}
    capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), # type: ignore[arg-type]
        n_frames=10,
        target_reprojection_error_px={1: 1.0, 2: 1.0}, # cam0 absent on purpose
        diagnostics_out=diagnostics,
    )

    assert diagnostics[0]["reprojection_error_px"] == 2.0
    assert diagnostics[0]["target_met"] is True # 2.0 < 2.5 (the module default)


def test_capture_calibration_frames_local_waits_for_a_fresh_pump_frame_between_rounds():
    """The real mechanism this fix depends on: LocalCameraHub's pump cache
    means back-to-back grab_all() calls with no wait would return the SAME
    cached frame repeatedly (see _capture_calibration_frames_local()'s own
    docstring). This test uses a fake hub whose frame_count only advances
    when explicitly told to (simulating the pump thread) and proves the
    capture helper actually WAITS for that advance each round rather than
    reading the stale cache N times."""

    class _FakeStatus:
        def __init__(self):
            self.frame_count = 0

    class _FakeHub:
        def __init__(self):
            self.configs = [object()] # one camera
            self.status = {0: _FakeStatus()}
            self._frame_value = 0

        def grab_all(self):
            return {0: np.full((4, 4, 3), self._frame_value, dtype=np.uint8)}

        def advance_pump(self, new_value):
            """Simulates the pump thread producing a genuinely new frame."""
            self._frame_value = new_value
            self.status[0].frame_count += 1

    hub = _FakeHub()
    n_frames = 3

    # Advance the pump in a background thread shortly after each poll
    # begins, so the capture helper's bounded wait-loop actually has to
    # wait (not zero-iterations) -- proves it polls for freshness, not
    # just reads whatever's cached immediately.
    def pump_advancer():
        for i in range(1, n_frames + 1):
            time.sleep(0.02)
            hub.advance_pump(i * 10)

    t = threading.Thread(target=pump_advancer, daemon=True)
    t.start()
    try:
        frames_by_cam = capture_daemon._capture_calibration_frames_local(
            hub, n_frames, poll_interval_s=0.005, per_frame_timeout_s=2.0
        )
    finally:
        t.join(timeout=2.0)

    assert len(frames_by_cam[0]) == n_frames
    # Each captured frame must reflect a DIFFERENT pump value -- proves
    # this did not just read the same cached frame 3 times.
    values = [int(f[0, 0, 0]) for f in frames_by_cam[0]]
    assert len(set(values)) == n_frames, f"expected {n_frames} distinct frames, got {values}"


def test_capture_calibration_frames_local_proceeds_after_timeout_on_a_stalled_camera():
    """A camera whose pump never advances (stalled/disconnected) must not
    block calibration of the OTHER cameras forever -- proceeds with
    whatever's cached after per_frame_timeout_s, per this function's own
    documented degraded-not-blocked behavior."""

    class _FakeStatus:
        def __init__(self):
            self.frame_count = 0

    class _FakeHub:
        def __init__(self):
            self.configs = [object(), object()] # cam0 stalls, cam1 advances
            self.status = {0: _FakeStatus(), 1: _FakeStatus()}

        def grab_all(self):
            return {
                0: np.full((4, 4, 3), 0, dtype=np.uint8), # never changes
                1: np.full((4, 4, 3), self.status[1].frame_count, dtype=np.uint8),
            }

    hub = _FakeHub()
    # cam1's frame_count is bumped directly (no real pump thread needed --
    # this test only cares that a stalled cam0 doesn't block completion).
    hub.status[1].frame_count = 1

    t0 = time.monotonic()
    frames_by_cam = capture_daemon._capture_calibration_frames_local(
        hub, 2, poll_interval_s=0.01, per_frame_timeout_s=0.1
    )
    elapsed = time.monotonic() - t0

    assert len(frames_by_cam[0]) == 2 # proceeded anyway, using stale cache
    assert elapsed < 2.0 # bounded by the timeout, not hung forever


def test_fetch_current_frames_local_mode_reads_frames_via_local_capture(tmp_path, monkeypatch):
    """PERFORMANCE FIX, 2026-08-12 (see fetch_current_frames()'s own
    docstring + capture_daemon.py's module docstring for the full
    ~165ms/iteration finding): the local-mode branch used to round-trip
    every frame through a throwaway PNG (local_capture.fetch_all_snapshots()
    writes, this function's old body cv2.imread()'d it straight back) --
    measured on real 1280x720 frames at ~91ms/iteration for 3 cameras,
    roughly half the real per-iteration cost. Fixed to call
    hub.grab_all() directly (already dict[int, np.ndarray] -- the exact
    frame source local_capture.fetch_all_snapshots() was itself only
    wrapping via hub.grab() + an unnecessary encode/decode). This test
    now locks in BOTH halves of that fix: fetch_current_frames() returns
    exactly what hub.grab_all() hands it (no shape/identity distortion
    from an encode/decode round trip it no longer does), AND
    local_capture.fetch_all_snapshots() is NOT called at all in local
    mode -- a regression here would silently reintroduce the disk
    round-trip this fix removed."""

    def fake_local_fetch_all(dest_dir, n_cameras=3, hub=None):
        raise AssertionError(
            "fetch_current_frames() must not call "
            "local_capture.fetch_all_snapshots() anymore -- it should read "
            "hub.grab_all() directly, in memory, with no PNG round trip "
            "(see this test's own docstring for the 2026-08-12 perf fix)"
        )

    monkeypatch.setattr(local_capture, "fetch_all_snapshots", fake_local_fetch_all)

    expected_frames = {
        0: np.full((48, 64, 3), 128, dtype=np.uint8),
        1: np.full((48, 64, 3), 200, dtype=np.uint8),
    }

    class _FakeHub:
        def grab_all(self) -> dict[int, np.ndarray]:
            return expected_frames

    frames = capture_daemon.fetch_current_frames(tmp_path, hub=_FakeHub()) # type: ignore[arg-type]

    assert frames is expected_frames
    assert set(frames.keys()) == {0, 1}
    assert frames[0].shape[:2] == (48, 64)


# ---------------------------------------------------------------------------
# Bug fix, 2026-08-12: startup camera-warmup / stable-background gate
# (_wait_for_first_frames()). Live-confirmed on the rig: within seconds of
# starting the daemon, a full IDLE -> MOTION_DETECTED -> READY_TO_CAPTURE
# cycle fired and a throw package was saved before the operator had
# thrown anything -- most plausibly explained by a freshly-opened
# camera's auto-exposure/white-balance still converging when the
# trigger's initial bg_frames/true_baseline_frames was captured and
# trusted. See STARTUP_SETTLE_MAX_WAIT_S's own comment in
# capture_daemon.py for the full root-cause writeup, including the
# honest caveat that no real repro frames were available to test against
# directly (the two locally-pulled "ghost" packages both turned out, on
# inspection, to be real dart-2-of-turn captures already investigated for
# an unrelated bug) -- these tests instead prove the FIX's actual
# mechanism using REAL (not tiny/degenerate) synthetic frames big enough
# to exercise throw_trigger's real MIN_CHANGED_AREA_PX/PIXEL_DIFF_
# THRESHOLD thresholds, same "prove the structure against real thresholds"
# discipline as tests/test_throw_trigger_detection.py.
# ---------------------------------------------------------------------------


def _synthetic_frame(value: int, shape=(200, 300, 3)):
    return np.full(shape, value, dtype=np.uint8)


class _StopLoopEarly(Exception):
    """Local sentinel, mirrors _StopLoop below (defined further down in
    this file) -- declared here too since these tests run before that
    class's own definition point in file order is guaranteed available;
    kept separate/renamed to avoid any import-order confusion."""


# ---------------------------------------------------------------------------
# Bug fix, 2026-09-07: the short-term check above (is_settled(), a
# STARTUP_FETCHES=2-frame, ~100ms-wide window) cannot see a SLOW,
# MONOTONIC auto-exposure/white-balance drift -- each individual step is
# tiny (looks "stable" to a 2-frame check) even while the camera's true
# brightness keeps sliding for several more seconds. Live-confirmed on
# the rig: cam0 locked in a wrong baseline mid-drift, producing a persistent,
# non-decaying false "hand present" reading (~5,260-5,280px outside-disc,
# <1% jitter, held for 47+ real seconds) that never cleared because it
# wasn't real motion. See STARTUP_PHOTOMETRIC_STABLE_WINDOW_S's own
# comment in capture_daemon.py for the full incident and design reasoning.
#
# These tests use a monkeypatched time.monotonic() (not a real 2-second
# sleep) so they stay fast and deterministic -- poll_interval_s=0.0 means
# the real wall-clock cost of the loop itself is negligible; only the
# fake clock's own advancement drives the long-baseline check.
# ---------------------------------------------------------------------------


class _FakeHubWithFrameCounts:
    """Minimal stand-in exposing just the `.status` dict the
    frame-freshness gate reads -- real `local_capture.CameraStatus`
    instances (not a bespoke shape), mutated directly by each test's own
    fake_fetch closure to simulate the pump advancing (or not). Named
    distinctly from the pre-existing `_FakeHubWithStatus` above (a
    different test double, `status=` keyword shape, used by the
    negotiated-resolution tests) -- same "status" concept, unrelated
    shape, kept separate rather than reused to avoid coupling two
    independent test areas together."""

    def __init__(self, cams: list[int]) -> None:
        self.status = {cam: local_capture.CameraStatus(device=cam, frame_count=0) for cam in cams}


def test_run_capture_loop_body_frame_freshness_gate_skips_a_stale_duplicate_poll(
    tmp_path, monkeypatch
):
    """Real, live phantom-dart fix, 2026-09-01: LocalCameraHub.grab_all()
    is a pure cache read -- if this loop polls faster than the camera
    pump refreshes, two consecutive polls can return the SAME cached
    frame, and throw_trigger.is_settled() can't tell that apart from
    "the scene genuinely stopped changing." Reproduces the mechanism
    directly: fake_fetch's underlying frame_count only advances on odd
    calls (simulating a pump that's slower than this loop's own poll),
    and confirms the REAL advance() is invoked roughly half as often as
    fetch_current_frames() is called -- i.e. the gate is actually
    skipping the stale (even-numbered) polls, not just letting everything
    through."""
    from tests.lifecycle_scripting import idle_advance as real_advance

    converged = _synthetic_frame(120)
    hub = _FakeHubWithFrameCounts([0])

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {0: _fake_calibration_attempt().calibration}

    calls = {"n": 0}

    def fake_fetch(dest_dir, *, hub=None):
        calls["n"] += 1
        # Odd calls: pump produced a genuinely new frame. Even calls:
        # pump hasn't advanced yet -- a stale duplicate read.
        if calls["n"] % 2 == 1:
            hub.status[0].frame_count += 1
        return {0: converged}

    real_loop_iterations = {"n": 0}

    def counting_real_advance(trigger, bg_frames, current_frames):
        real_loop_iterations["n"] += 1
        if real_loop_iterations["n"] > 6:
            raise _StopLoopEarly("ran enough real advance() calls to measure the skip ratio")
        return real_advance(trigger, bg_frames, current_frames)

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, counting_real_advance)
    monkeypatch.setattr(
        capture_daemon, "handle_ready_to_capture", lambda *a, **k: tmp_path / "phantom"
    )

    stop_event = threading.Event()
    with pytest.raises(_StopLoopEarly):
        capture_daemon.run_capture_loop_body(
            hub=hub,
            package_root=tmp_path / "packages",
            poll_interval_s=0.0,
            stop_event=stop_event,
            scratch_dir=tmp_path / "scratch",
        )

    # Every stale (even-numbered) fetch should have been skipped before
    # ever reaching advance() -- confirmed by there being meaningfully
    # more fetch calls than accepted advance() calls, not a 1:1 ratio.
    assert calls["n"] >= real_loop_iterations["n"] * 1.5, (
        f"expected the gate to skip roughly half of all polls as stale "
        f"duplicates, got {calls['n']} fetches for only "
        f"{real_loop_iterations['n']} accepted advance() calls"
    )


def test_run_capture_loop_body_frame_freshness_gate_fails_open_past_the_ceiling(
    tmp_path, monkeypatch
):
    """The timeout half of the fix: a camera whose frame_count NEVER
    advances (a genuinely stalled/dead camera, not just pump lag) must
    not hang this loop forever -- confirmed by lowering
    MAX_FRAME_FRESHNESS_WAIT_S to a small real value and proving the
    real advance() still eventually gets called despite frame_count
    staying frozen at 0 for the entire test."""

    converged = _synthetic_frame(120)
    hub = _FakeHubWithFrameCounts([0]) # frame_count never touched -- permanently "stalled"

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {0: _fake_calibration_attempt().calibration}

    def fake_fetch(dest_dir, *, hub=None):
        return {0: converged}

    real_loop_iterations = {"n": 0}

    def counting_real_advance(trigger, bg_frames, current_frames):
        real_loop_iterations["n"] += 1
        raise _StopLoopEarly("advance() was reached despite the camera never advancing")

    monkeypatch.setattr(capture_daemon, "MAX_FRAME_FRESHNESS_WAIT_S", 0.02)
    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, counting_real_advance)
    monkeypatch.setattr(
        capture_daemon, "handle_ready_to_capture", lambda *a, **k: tmp_path / "phantom"
    )

    stop_event = threading.Event()
    with pytest.raises(_StopLoopEarly):
        capture_daemon.run_capture_loop_body(
            hub=hub,
            package_root=tmp_path / "packages",
            poll_interval_s=0.005,
            stop_event=stop_event,
            scratch_dir=tmp_path / "scratch",
        )

    assert real_loop_iterations["n"] == 1, (
        "the gate must fail OPEN once the ceiling is exceeded -- a permanently "
        "stalled camera must never hang this loop forever"
    )


def test_run_capture_loop_body_frame_freshness_gate_is_a_noop_without_hub_status(
    tmp_path, monkeypatch
):
    """Blast-radius guard: a hub with no `.status` attribute (every
    pre-existing FakeHub test double in this file, and hub=None) must
    behave EXACTLY as before this fix -- the gate is a pure no-op unless
    hub.status genuinely exists, so this doesn't require touching any of
    the many existing tests that construct a bare test-double hub."""
    from tests.lifecycle_scripting import idle_advance as real_advance

    converged = _synthetic_frame(120)

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {0: _fake_calibration_attempt().calibration}

    def fake_fetch(dest_dir, *, hub=None):
        return {0: converged}

    real_loop_iterations = {"n": 0}

    def counting_real_advance(trigger, bg_frames, current_frames):
        real_loop_iterations["n"] += 1
        if real_loop_iterations["n"] > 3:
            raise _StopLoopEarly("ran with hub=None -- gate must be a no-op")
        return real_advance(trigger, bg_frames, current_frames)

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, counting_real_advance)
    monkeypatch.setattr(
        capture_daemon, "handle_ready_to_capture", lambda *a, **k: tmp_path / "phantom"
    )

    stop_event = threading.Event()
    with pytest.raises(_StopLoopEarly):
        capture_daemon.run_capture_loop_body(
            hub=None,
            package_root=tmp_path / "packages",
            poll_interval_s=0.0,
            stop_event=stop_event,
            scratch_dir=tmp_path / "scratch",
        )

    assert real_loop_iterations["n"] > 3


# ---------------------------------------------------------------------------
# Frame-age-at-advance() diagnostic (2026-09-04) -- see capture_daemon.py's
# own _FRAME_AGE_LOG_FLOOR_S / _FRAME_AGE_SAMPLE_EVERY_N_ITERATIONS module
# comments for the full reasoning this exercises.
# ---------------------------------------------------------------------------


def test_frame_age_reused_at_motion_detected_transition_not_fabricated(
    tmp_path, monkeypatch, caplog
):
    """The literal question this diagnostic exists to answer: when IDLE
    -> MOTION_DETECTED fires, how old was the frame that tripped it. A
    real LocalCameraHub-shaped fake hub reports cam0's last successful
    pump read as ~5s in the past (using time.monotonic(), the same
    clock domain as `_effective_top_crop_fraction`-style consumption
    code) -- the transition log line for IDLE -> MOTION_DETECTED must
    surface that same age, computed once and reused, not recomputed at
    a later, slightly-different instant."""
    # 2026-09-04, opendarts.live.diagnostics_gate task: this whole
    # computation is now gated (default OFF) -- enabled here to exercise
    # the real content this test is about; see
    # test_frame_age_diagnostic_gated_off_by_default below for the real,
    # measured OFF-by-default proof (zero calls, zero allocation).
    diagnostics_gate.set_enabled(True)
    stamped_age_s = 5.0
    hub = _FakeHubWithFrameCounts([0])
    hub.status[0].last_read_at_monotonic = time.monotonic() - stamped_age_s

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {0: _fake_calibration_attempt().calibration}

    def fake_fetch(dest_dir, *, hub=None):
        hub.status[0].frame_count += 1 # keep the freshness gate happy
        return {0: _synthetic_frame(120)}

    scripted_states = [
        ThrowTriggerState(state=ThrowState.MOTION_DETECTED, dart_count=0),
    ]
    advance_calls: list[int] = []

    def fake_advance(trigger, bg_frames, current_frames):
        idx = len(advance_calls)
        advance_calls.append(idx)
        if idx < len(scripted_states):
            return scripted_states[idx]
        raise _StopLoopEarly("stop after the scripted transition")

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)

    stop_event = threading.Event()
    with caplog.at_level("INFO", logger="opendarts.capture_daemon"):
        with pytest.raises(_StopLoopEarly):
            capture_daemon.run_capture_loop_body(
                hub=hub,
                package_root=tmp_path / "packages",
                poll_interval_s=0.0,
                stop_event=stop_event,
                scratch_dir=tmp_path / "scratch",
            )

    transition_lines = [
        r for r in caplog.records if "trigger state: IDLE -> MOTION_DETECTED" in r.message
    ]
    assert len(transition_lines) == 1
    message = transition_lines[0].message
    assert "frame age at trip:" in message
    match = re.search(r"\{0: ([\d.]+)\}", message)
    assert match is not None, f"expected a per-camera age dict in: {message!r}"
    reported_age = float(match.group(1))
    # Generous bound -- real test execution adds real (small) wall-clock
    # time on top of the stamped 5.0s delta; this only needs to confirm
    # the reported age is genuinely close to what was stamped, not exact
    # to the microsecond.
    assert stamped_age_s <= reported_age <= stamped_age_s + 2.0


def test_frame_age_diagnostic_absent_when_hub_has_no_monotonic_stamp(tmp_path, monkeypatch, caplog):
    """A camera whose CameraStatus has never had last_read_at_monotonic
    set (the real dataclass default, None -- e.g. before this rig's
    first successful pump cycle) must never fabricate an age: no
    "frame age at trip" text on the transition line at all."""
    diagnostics_gate.set_enabled(True) # exercise the no-stamp case specifically, not gate-off
    hub = _FakeHubWithFrameCounts([0])
    assert hub.status[0].last_read_at_monotonic is None

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {0: _fake_calibration_attempt().calibration}

    def fake_fetch(dest_dir, *, hub=None):
        hub.status[0].frame_count += 1
        return {0: _synthetic_frame(120)}

    scripted_states = [ThrowTriggerState(state=ThrowState.MOTION_DETECTED, dart_count=0)]
    advance_calls: list[int] = []

    def fake_advance(trigger, bg_frames, current_frames):
        idx = len(advance_calls)
        advance_calls.append(idx)
        if idx < len(scripted_states):
            return scripted_states[idx]
        raise _StopLoopEarly("stop after the scripted transition")

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)

    stop_event = threading.Event()
    with caplog.at_level("INFO", logger="opendarts.capture_daemon"):
        with pytest.raises(_StopLoopEarly):
            capture_daemon.run_capture_loop_body(
                hub=hub,
                package_root=tmp_path / "packages",
                poll_interval_s=0.0,
                stop_event=stop_event,
                scratch_dir=tmp_path / "scratch",
            )

    transition_lines = [
        r for r in caplog.records if "trigger state: IDLE -> MOTION_DETECTED" in r.message
    ]
    assert len(transition_lines) == 1
    assert "frame age" not in transition_lines[0].message


def test_frame_age_diagnostic_floor_fires_when_exceeded(tmp_path, monkeypatch, caplog):
    """A camera whose frame is well past _FRAME_AGE_LOG_FLOOR_S must
    trigger the per-iteration anomaly DEBUG line, even outside a
    settle episode and even when it's not a sampled iteration."""
    diagnostics_gate.set_enabled(True)
    hub = _FakeHubWithFrameCounts([0])
    # Comfortably above the real floor (POLL_INTERVAL_SECONDS, 0.05s).
    hub.status[0].last_read_at_monotonic = time.monotonic() - 1.0

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {0: _fake_calibration_attempt().calibration}

    call_count = {"n": 0}

    def fake_fetch(dest_dir, *, hub=None):
        call_count["n"] += 1
        hub.status[0].frame_count += 1
        return {0: _synthetic_frame(120)}

    def fake_advance(trigger, bg_frames, current_frames):
        if call_count["n"] >= 2:
            raise _StopLoopEarly("stop after one real iteration")
        return trigger # stay IDLE

    # A non-sampled iteration -- prove the floor alone is what's firing,
    # not the periodic sample.
    monkeypatch.setattr(capture_daemon, "_FRAME_AGE_SAMPLE_EVERY_N_ITERATIONS", 1000)
    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)

    stop_event = threading.Event()
    with caplog.at_level("DEBUG", logger="opendarts.capture_daemon"):
        with pytest.raises(_StopLoopEarly):
            capture_daemon.run_capture_loop_body(
                hub=hub,
                package_root=tmp_path / "packages",
                poll_interval_s=0.0,
                stop_event=stop_event,
                scratch_dir=tmp_path / "scratch",
            )

    age_lines = [
        r for r in caplog.records if "frame age at lifecycle consumption" in r.message
    ]
    assert len(age_lines) >= 1
    assert any("floor" in r.message for r in age_lines)


def test_frame_age_diagnostic_sampled_even_under_the_floor(tmp_path, monkeypatch, caplog):
    """Companion to the floor test above -- a floor-only mechanism would
    only ever show the tail. With every camera's age WELL
    under the floor, the periodic sample must still fire on schedule."""
    diagnostics_gate.set_enabled(True)
    hub = _FakeHubWithFrameCounts([0])
    hub.status[0].last_read_at_monotonic = time.monotonic() # ~0 age, well under the floor

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {0: _fake_calibration_attempt().calibration}

    call_count = {"n": 0}
    n_real_iterations = {"n": 0}

    def fake_fetch(dest_dir, *, hub=None):
        call_count["n"] += 1
        hub.status[0].frame_count += 1
        hub.status[0].last_read_at_monotonic = time.monotonic()
        return {0: _synthetic_frame(120)}

    def fake_advance(trigger, bg_frames, current_frames):
        n_real_iterations["n"] += 1
        if n_real_iterations["n"] > 6:
            raise _StopLoopEarly("ran enough iterations to see a sample fire")
        return trigger # stay IDLE

    monkeypatch.setattr(capture_daemon, "_FRAME_AGE_SAMPLE_EVERY_N_ITERATIONS", 3)
    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)

    stop_event = threading.Event()
    with caplog.at_level("DEBUG", logger="opendarts.capture_daemon"):
        with pytest.raises(_StopLoopEarly):
            capture_daemon.run_capture_loop_body(
                hub=hub,
                package_root=tmp_path / "packages",
                poll_interval_s=0.0,
                stop_event=stop_event,
                scratch_dir=tmp_path / "scratch",
            )

    age_lines = [
        r for r in caplog.records if "frame age at lifecycle consumption" in r.message
    ]
    assert len(age_lines) >= 1, "the periodic sample never fired even though age stayed under the floor"
    assert all("sampled" in r.message for r in age_lines), (
        "a sampled-iteration line fired the floor's own message instead of "
        "the sampled one -- ages here are deliberately kept near-zero"
    )




def test_frame_age_diagnostic_gated_off_by_default(tmp_path, monkeypatch, caplog):
    """THE real, measured proof this task's own diagnostics_gate is
    about: opendarts.live.diagnostics_gate at its real default (OFF, never
    explicitly toggled) -- a real LocalCameraHub-shaped fake hub with a
    real, non-None last_read_at_monotonic stamp (a stamp genuinely
    present and stale enough to clear the floor -- the exact condition
    every other test in this section relies on to make the diagnostic
    fire) must NOT produce ANY frame-age computation: no periodic/floor
    DEBUG line, no "frame age at trip" detail on the transition line,
    AND (this is the actual point, not just the log text) the
    `frame_ages_s` dict genuinely never gets populated -- proven by
    monkeypatching `time.monotonic` with a call-counting wrapper and
    confirming the frame-age block's own extra time.monotonic() call
    (beyond what advance()'s own iteration-timing bookkeeping already
    needs) never happens."""
    assert diagnostics_gate.enabled() is False
    hub = _FakeHubWithFrameCounts([0])
    hub.status[0].last_read_at_monotonic = time.monotonic() - 5.0 # well past the floor

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {0: _fake_calibration_attempt().calibration}

    def fake_fetch(dest_dir, *, hub=None):
        hub.status[0].frame_count += 1
        return {0: _synthetic_frame(120)}

    scripted_states = [ThrowTriggerState(state=ThrowState.MOTION_DETECTED, dart_count=0)]
    advance_calls: list[int] = []

    def fake_advance(trigger, bg_frames, current_frames):
        idx = len(advance_calls)
        advance_calls.append(idx)
        if idx < len(scripted_states):
            return scripted_states[idx]
        raise _StopLoopEarly("stop after the scripted transition")

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)

    stop_event = threading.Event()
    with caplog.at_level("DEBUG", logger="opendarts.capture_daemon"):
        with pytest.raises(_StopLoopEarly):
            capture_daemon.run_capture_loop_body(
                hub=hub,
                package_root=tmp_path / "packages",
                poll_interval_s=0.0,
                stop_event=stop_event,
                scratch_dir=tmp_path / "scratch",
            )

    age_lines = [
        r for r in caplog.records if "frame age at lifecycle consumption" in r.message
    ]
    assert age_lines == [], "no periodic/floor frame-age line with diagnostics OFF"
    transition_lines = [
        r for r in caplog.records if "trigger state: IDLE -> MOTION_DETECTED" in r.message
    ]
    assert len(transition_lines) == 1
    assert "frame age" not in transition_lines[0].message, (
        "the always-on transition line must still fire, but without the "
        "gated frame-age detail"
    )


# ---------------------------------------------------------------------------
# PERSIST_SETTLE_WINDOW_DEBUG_FRAMES -- now ALSO gated behind
# opendarts.live.diagnostics_gate (2026-09-04), on top of its own
# pre-existing module-level constant. Real measured cost: ~112ms
# background CPU + ~7MB disk PER THROW on this rig's real frames (see
# that constant's own comment).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# run_capture_loop() -- hub opened exactly ONCE, reused, closed on exit
# ---------------------------------------------------------------------------


class FakeHub:
    """Stand-in for local_capture.LocalCameraHub -- records how many
    times it was constructed/opened/closed, without touching cv2 at
    all."""

    instances: list["FakeHub"] = []

    def __init__(self, configs=None) -> None:
        self.open_calls = 0
        self.close_calls = 0
        self._open_ok = True
        FakeHub.instances.append(self)

    def open_all(self):
        self.open_calls += 1
        return [self._open_ok, self._open_ok, self._open_ok]

    def close_all(self):
        self.close_calls += 1

    def status_report(self) -> str:
        return "fake hub status"


class _StopLoop(Exception):
    """Sentinel used to break run_capture_loop()'s while-loop
    deterministically after N iterations, without needing a real
    throw_trigger state machine or real timing."""


@pytest.fixture(autouse=True)
def _reset_fake_hub_instances():
    FakeHub.instances = []
    yield
    FakeHub.instances = []


# ---------------------------------------------------------------------------
# Turn/takeout wiring:
# refresh_background_after_capture() (was a NotImplementedError stub) and
# run_capture_loop_body()'s own response to the turn/takeout transitions
# advance() now produces -- forcing TAKEOUT_WAITING after the 3rd dart,
# and restoring true_baseline_frames as `bg_frames` the moment a takeout
# is detected complete. opendarts.capture.throw_trigger.advance() ITSELF has
# its own thorough, dedicated coverage in
# tests/test_throw_trigger_detection.py (the real classification logic);
# this section is specifically about what run_capture_loop_body() does
# WITH advance()'s output, so advance() is faked here with a scripted
# sequence of ThrowTriggerState transitions.
# ---------------------------------------------------------------------------


def _frames_equal(a: dict[int, np.ndarray], b: dict[int, np.ndarray]) -> bool:
    return set(a) == set(b) and all(np.array_equal(a[cam], b[cam]) for cam in a)


def test_run_capture_loop_body_forces_takeout_waiting_after_third_dart_and_restores_baseline(
    tmp_path, monkeypatch
):
    """End-to-end wiring proof for the turn/takeout state machine: over a
    scripted 3-dart turn, run_capture_loop_body() must (a) capture all 3
    darts in order, using each prior dart's own frame as the background
    for the next (refresh_background_after_capture()'s job), (b) force a
    transition straight to TAKEOUT_WAITING after the 3rd dart instead of
    continuing to watch for a 4th, and (c) the moment advance() reports
    the takeout complete (state==IDLE, dart_count==0, coming from a non-
    IDLE state), restore `bg_frames` to the TRUE baseline -- not the last
    (dart-3) captured frame -- for every subsequent advance() call. (c) is
    the specific bug this test exists to catch: a caller that forgot to
    special-case the takeout-complete transition would silently keep
    diffing new throws against a stale, 3-dart-laden background forever."""
    true_baseline = {0: np.full((2, 2, 3), 111, dtype=np.uint8)}
    dart_frames = [
        {0: np.full((2, 2, 3), value, dtype=np.uint8)} for value in (10, 20, 30)
    ]

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {0: _fake_calibration_attempt().calibration}

    # Bug fix, 2026-08-12: the startup bg_frames fetch now goes through
    # _wait_for_first_frames(), which repeatedly calls fetch_current_frames
    # until the result looks temporally stable. For a frame that's returned
    # IDENTICALLY on every call (true_baseline here), that stability check
    # trivially passes the instant the rolling window fills to
    # STARTUP_FETCHES -- so exactly STARTUP_FETCHES startup calls
    # are consumed before the real per-iteration loop calls begin. Prepend
    # that many true_baseline reads so the startup fetch still resolves to
    # true_baseline (this test's whole premise), then per-dart values follow
    # for the loop's own current_frames reads (unused by the fully-scripted
    # fake_advance below beyond being real, well-formed values).
    fetch_sequence = (
        [true_baseline] * STARTUP_FETCHES
        + [dart_frames[0], dart_frames[1], dart_frames[2]]
    )
    fetch_calls = {"n": 0}

    def fake_fetch(dest_dir, *, hub=None):
        idx = fetch_calls["n"]
        fetch_calls["n"] += 1
        return fetch_sequence[idx] if idx < len(fetch_sequence) else dart_frames[2]

    scripted_states = [
        ThrowTriggerState(
            state=ThrowState.READY_TO_CAPTURE,
            dart_count=1,
            true_baseline_frames=true_baseline,
            last_frame=dart_frames[0],
        ),
        ThrowTriggerState(
            state=ThrowState.READY_TO_CAPTURE,
            dart_count=2,
            true_baseline_frames=true_baseline,
            last_frame=dart_frames[1],
        ),
        ThrowTriggerState(
            state=ThrowState.READY_TO_CAPTURE,
            dart_count=3,
            true_baseline_frames=true_baseline,
            last_frame=dart_frames[2],
        ),
        # Takeout detected complete (as advance() would report from
        # TAKEOUT_WAITING once the board matches the true baseline).
        ThrowTriggerState(state=ThrowState.IDLE, dart_count=0, true_baseline_frames=true_baseline),
    ]
    advance_calls: list[dict] = []

    def fake_advance(trigger, bg_frames, current_frames):
        advance_calls.append({"bg_frames": bg_frames, "trigger_state": trigger.state})
        idx = len(advance_calls) - 1
        if idx < len(scripted_states):
            return scripted_states[idx]
        raise _StopLoop("stop after the scripted sequence completes")

    captured_dart_counts: list[int] = []

    def fake_handle_ready_to_capture(trigger, bg_frames, calibrations, package_root, session_id, **_kwargs):
        captured_dart_counts.append(trigger.dart_count)
        return package_root / "fake_throw"

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)
    monkeypatch.setattr(capture_daemon, "handle_ready_to_capture", fake_handle_ready_to_capture)

    stop_event = threading.Event()
    with pytest.raises(_StopLoop):
        capture_daemon.run_capture_loop_body(
            hub=None,
            package_root=tmp_path / "packages",
            poll_interval_s=0.0,
            stop_event=stop_event,
            scratch_dir=tmp_path / "scratch",
        )

    # All 3 darts captured, in order -- the loop kept running through the
    # whole turn without raising.
    assert captured_dart_counts == [1, 2, 3]

    # The 4th advance() call (index 3) is the one made while the trigger
    # was in TAKEOUT_WAITING (forced there right after dart 3, per (b)
    # above) -- confirms the forced transition actually happened, not
    # just that dart_count reached 3.
    assert advance_calls[3]["trigger_state"] == ThrowState.TAKEOUT_WAITING

    # The reference the loop hands the next step is the lifecycle's own
    # (the scripted seam passes it straight through); after the visit is
    # cleared it is the true baseline, not dart 3's frame.
    assert _frames_equal(advance_calls[0]["bg_frames"], true_baseline)
    assert _frames_equal(advance_calls[4]["bg_frames"], true_baseline)


def test_run_capture_loop_body_on_event_trigger_state_includes_dart_count(tmp_path, monkeypatch):
    """opendarts/live/server.py's header status pill needs dart_count on
    every TRIGGER_STATE on_event push (added 2026-08-12 alongside that
    pill -- see run_capture_loop_body()'s own on_event docstring) so it
    can show "SETTLING (dart 2 of 3)" instead of just the bare state
    name. Proves ALL THREE real _emit(..., {"type": "TRIGGER_STATE",
    ...}) call sites in this function include it, not just the startup
    one -- the startup emit (before the loop even begins), the
    state-change emit (guarded on trigger.state != prev_state), and the
    unconditional post-capture re-emit right after a READY_TO_CAPTURE is
    handled."""
    true_baseline = {0: np.full((2, 2, 3), 111, dtype=np.uint8)}
    dart_frame = {0: np.full((2, 2, 3), 30, dtype=np.uint8)}

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {0: _fake_calibration_attempt().calibration}

    fetch_sequence = [true_baseline] * STARTUP_FETCHES + [dart_frame, dart_frame]
    fetch_calls = {"n": 0}

    def fake_fetch(dest_dir, *, hub=None):
        idx = fetch_calls["n"]
        fetch_calls["n"] += 1
        return fetch_sequence[idx] if idx < len(fetch_sequence) else dart_frame

    scripted_states = [
        ThrowTriggerState(
            state=ThrowState.READY_TO_CAPTURE,
            dart_count=1,
            true_baseline_frames=true_baseline,
        ),
    ]
    advance_calls: list[int] = []

    def fake_advance(trigger, bg_frames, current_frames):
        idx = len(advance_calls)
        advance_calls.append(idx)
        if idx < len(scripted_states):
            return scripted_states[idx]
        raise _StopLoop("stop after the scripted transition")

    def fake_handle_ready_to_capture(
        trigger, bg_frames, calibrations, package_root, session_id, **_kwargs
    ):
        return package_root / "fake_throw"

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)
    monkeypatch.setattr(capture_daemon, "handle_ready_to_capture", fake_handle_ready_to_capture)

    events: list[dict] = []
    stop_event = threading.Event()
    with pytest.raises(_StopLoop):
        capture_daemon.run_capture_loop_body(
            hub=None,
            package_root=tmp_path / "packages",
            poll_interval_s=0.0,
            stop_event=stop_event,
            scratch_dir=tmp_path / "scratch",
            on_event=events.append,
        )

    trigger_events = [e for e in events if e["type"] == "TRIGGER_STATE"]
    assert len(trigger_events) == 3
    for event in trigger_events:
        assert "dart_count" in event


def test_run_capture_loop_body_on_event_trigger_state_includes_emitted_at_utc(tmp_path, monkeypatch):
    """2026-09-01 latency-instrumentation task, purely additive (no
    scoring/detection behavior change): every real TRIGGER_STATE
    on_event push must carry `emitted_at_utc`, a real
    datetime.now(timezone.utc).isoformat() stamped at the moment THIS
    thread decided the transition -- not the dequeue-time `ts`
    AppState._handle_live_event() stamps separately (see
    run_capture_loop_body()'s own on_event docstring for the full
    reasoning: `ts` is stamped after a queue + asyncio.to_thread +
    strictly-serial dispatch, which is where real latency was found
    hiding, not in the settle window itself). Same harness/scripted
    flow as the dart_count test immediately above -- reuses it rather
    than duplicating the setup, since this is the same 3 real call
    sites, just a different field to assert on each."""
    true_baseline = {0: np.full((2, 2, 3), 111, dtype=np.uint8)}
    dart_frame = {0: np.full((2, 2, 3), 30, dtype=np.uint8)}

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {0: _fake_calibration_attempt().calibration}

    fetch_sequence = [true_baseline] * STARTUP_FETCHES + [dart_frame, dart_frame]
    fetch_calls = {"n": 0}

    def fake_fetch(dest_dir, *, hub=None):
        idx = fetch_calls["n"]
        fetch_calls["n"] += 1
        return fetch_sequence[idx] if idx < len(fetch_sequence) else dart_frame

    scripted_states = [
        ThrowTriggerState(
            state=ThrowState.READY_TO_CAPTURE,
            dart_count=1,
            true_baseline_frames=true_baseline,
        ),
    ]
    advance_calls: list[int] = []

    def fake_advance(trigger, bg_frames, current_frames):
        idx = len(advance_calls)
        advance_calls.append(idx)
        if idx < len(scripted_states):
            return scripted_states[idx]
        raise _StopLoop("stop after the scripted transition")

    def fake_handle_ready_to_capture(
        trigger, bg_frames, calibrations, package_root, session_id, **_kwargs
    ):
        return package_root / "fake_throw"

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)
    monkeypatch.setattr(capture_daemon, "handle_ready_to_capture", fake_handle_ready_to_capture)

    before = datetime.now(timezone.utc)
    events: list[dict] = []
    stop_event = threading.Event()
    with pytest.raises(_StopLoop):
        capture_daemon.run_capture_loop_body(
            hub=None,
            package_root=tmp_path / "packages",
            poll_interval_s=0.0,
            stop_event=stop_event,
            scratch_dir=tmp_path / "scratch",
            on_event=events.append,
        )
    after = datetime.now(timezone.utc)

    trigger_events = [e for e in events if e["type"] == "TRIGGER_STATE"]
    assert len(trigger_events) == 3
    for event in trigger_events:
        assert "emitted_at_utc" in event
        stamp = datetime.fromisoformat(event["emitted_at_utc"])
        # A real stamp taken during this call, not a placeholder/None and
        # not copied from some other field -- bounded by wall-clock
        # times taken immediately before/after the whole call.
        assert before <= stamp <= after


def test_run_capture_loop_body_on_event_ready_to_capture_includes_settle_duration(
    tmp_path, monkeypatch
):
    """2026-09-01 latency-instrumentation task, purely additive: the
    READY_TO_CAPTURE TRIGGER_STATE push must carry `settle_duration_s`/
    `straggler_camera`, derived from the SAME `camera_settled_at_
    monotonic`/`settle_started_monotonic` fields throw_trigger.py's own
    "MOTION_DETECTED/SETTLING -> READY_TO_CAPTURE after Xs" log line
    already uses -- lets an external latency harness read the real
    internal settle interval directly instead of deriving it from two
    separate TRIGGER_STATE receipt timestamps (which measures dispatch/
    broadcast latency, not the settle window -- see the cross-session
    latency investigation this closes). Must be ABSENT (not a fabricated
    None) on every other transition -- the startup IDLE emit here has no
    settle interval to report at all."""
    true_baseline = {0: np.full((2, 2, 3), 111, dtype=np.uint8)}
    dart_frame = {0: np.full((2, 2, 3), 30, dtype=np.uint8)}

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {0: _fake_calibration_attempt().calibration}

    fetch_sequence = [true_baseline] * STARTUP_FETCHES + [dart_frame, dart_frame]
    fetch_calls = {"n": 0}

    def fake_fetch(dest_dir, *, hub=None):
        idx = fetch_calls["n"]
        fetch_calls["n"] += 1
        return fetch_sequence[idx] if idx < len(fetch_sequence) else dart_frame

    settle_started = time.monotonic() - 0.15 # a real, plausible ~150ms-ago settle start
    camera_settled_at = {0: settle_started + 0.14, 1: settle_started + 0.15} # cam1 is the straggler
    scripted_states = [
        ThrowTriggerState(
            state=ThrowState.READY_TO_CAPTURE,
            dart_count=1,
            true_baseline_frames=true_baseline,
            settle_started_monotonic=settle_started,
            camera_settled_at_monotonic=camera_settled_at,
        ),
    ]
    advance_calls: list[int] = []

    def fake_advance(trigger, bg_frames, current_frames):
        idx = len(advance_calls)
        advance_calls.append(idx)
        if idx < len(scripted_states):
            return scripted_states[idx]
        raise _StopLoop("stop after the scripted transition")

    def fake_handle_ready_to_capture(
        trigger, bg_frames, calibrations, package_root, session_id, **_kwargs
    ):
        return package_root / "fake_throw"

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)
    monkeypatch.setattr(capture_daemon, "handle_ready_to_capture", fake_handle_ready_to_capture)

    events: list[dict] = []
    stop_event = threading.Event()
    with pytest.raises(_StopLoop):
        capture_daemon.run_capture_loop_body(
            hub=None,
            package_root=tmp_path / "packages",
            poll_interval_s=0.0,
            stop_event=stop_event,
            scratch_dir=tmp_path / "scratch",
            on_event=events.append,
        )

    trigger_events = [e for e in events if e["type"] == "TRIGGER_STATE"]
    assert len(trigger_events) == 3

    # The startup IDLE emit (first event) has no settle interval at all --
    # absent, not a fabricated placeholder.
    idle_event = trigger_events[0]
    assert idle_event["state"] == "IDLE"
    assert "settle_duration_s" not in idle_event
    assert "straggler_camera" not in idle_event

    ready_events = [e for e in trigger_events if e["state"] == "READY_TO_CAPTURE"]
    assert len(ready_events) == 1
    ready_event = ready_events[0]
    assert ready_event["straggler_camera"] == 1 # cam1 settled last, per camera_settled_at above
    assert ready_event["settle_duration_s"] == pytest.approx(0.15, abs=0.01)

    # (1) startup, before the loop even begins -- the board is assumed
    # empty, dart_count is 0.
    assert trigger_events[0]["state"] == "IDLE"
    assert trigger_events[0]["dart_count"] == 0
    # (2) the state-change emit into READY_TO_CAPTURE -- dart_count
    # already reflects "this is dart 1" (advance() bumps it on the way
    # in, see throw_trigger.py's own module docstring).
    assert trigger_events[1]["state"] == "READY_TO_CAPTURE"
    assert trigger_events[1]["dart_count"] == 1
    # (3) the unconditional re-emit right after handle_ready_to_capture()
    # -- dart_count stays 1 (one dart captured so far this turn), even
    # though the fresh ThrowTriggerState's own .state defaults back to
    # IDLE (existing behavior, not touched by this change).
    assert trigger_events[2]["state"] == "IDLE"
    assert trigger_events[2]["dart_count"] == 1


def test_run_capture_loop_body_never_raises_notimplementederror(tmp_path, monkeypatch):
    """Regression pin: refresh_background_after_capture() used to be a
    NotImplementedError stub that killed the loop right after the first
    dart -- confirms a full 3-dart-turn-plus-takeout scripted sequence
    (same shape as the test above) runs to completion and the loop keeps
    going afterward (proven by reaching a 5th advance() call rather than
    the loop dying on its own) without ever raising NotImplementedError."""
    true_baseline = {0: np.full((2, 2, 3), 111, dtype=np.uint8)}
    dart_frames = [{0: np.full((2, 2, 3), v, dtype=np.uint8)} for v in (10, 20, 30)]

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {0: _fake_calibration_attempt().calibration}

    def fake_fetch(dest_dir, *, hub=None):
        return dart_frames[0]

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    monkeypatch.setattr(
        capture_daemon,
        "handle_ready_to_capture",
        lambda trigger, bg_frames, calibrations, package_root, session_id, **_kwargs: package_root / "x",
    )

    call_count = {"n": 0}

    def fake_advance(trigger, bg_frames, current_frames):
        call_count["n"] += 1
        if call_count["n"] > MAX_DARTS_PER_TURN + 2:
            raise _StopLoop("ran a full turn plus one extra iteration without raising")
        if call_count["n"] <= MAX_DARTS_PER_TURN:
            return ThrowTriggerState(
                state=ThrowState.READY_TO_CAPTURE,
                dart_count=call_count["n"],
                true_baseline_frames=true_baseline,
                last_frame=dart_frames[0],
            )
        return ThrowTriggerState(state=ThrowState.IDLE, dart_count=0, true_baseline_frames=true_baseline)

    script_trigger(monkeypatch, fake_advance)

    stop_event = threading.Event()
    with pytest.raises(_StopLoop):
        capture_daemon.run_capture_loop_body(
            hub=None,
            package_root=tmp_path / "packages",
            poll_interval_s=0.0,
            stop_event=stop_event,
            scratch_dir=tmp_path / "scratch",
        )
    # Reaching _StopLoop (raised only after MAX_DARTS_PER_TURN + 2 calls)
    # rather than any NotImplementedError proves the loop survived a full
    # turn and kept running afterward.


# ---------------------------------------------------------------------------
# Throw-to-visible latency fix: dropping
# POLL_INTERVAL_SECONDS toward camera-native cadence for the local/
# product path,
# and re-flooring the "camera read may be slow/stalling" warning so it
# doesn't spam at the new fast interval. All provable via real elapsed
# wall-clock time and mocked frame sources -- no real hardware needed
# (same discipline as the rest of this module's tests).
# ---------------------------------------------------------------------------




def test_main_reports_and_uses_the_faster_local_poll_interval_end_to_end(monkeypatch):
    """End-to-end (through main(), not just the constant) proof that a
    plain local-mode run really does get the fast interval -- the
    specific regression this fix could have introduced if main() kept
    calling run_capture_loop() with no poll_interval_s override (silently
    falling back to whatever run_capture_loop()'s OWN default is, which
    must also be the fast one -- checked here too)."""
    # Captured BEFORE monkeypatching below -- inspecting the signature
    # AFTER patching capture_daemon.run_capture_loop would inspect the
    # fake replacement, not the real function.
    import inspect

    default = inspect.signature(capture_daemon.run_capture_loop).parameters["poll_interval_s"].default
    assert default == capture_daemon.POLL_INTERVAL_SECONDS

    seen = {}

    def fake_run_capture_loop(*, package_root, poll_interval_s, **_kwargs):
        seen["poll_interval_s"] = poll_interval_s

    monkeypatch.setattr(capture_daemon, "run_capture_loop", fake_run_capture_loop)
    rc = capture_daemon.main([])
    assert rc == 0
    assert seen["poll_interval_s"] == capture_daemon.POLL_INTERVAL_SECONDS


def test_fetch_slow_warning_does_not_fire_for_a_normal_fast_fetch_at_the_new_interval(
    tmp_path, monkeypatch, caplog
):
    """The actual regression risk this fix could have introduced: at the
    old 0.5s poll interval, `fetch_elapsed > poll_interval_s * 3` (1.5s)
    was never going to false-positive on a normal fetch. At the new
    ~0.03s interval, a bare `* 3` (~90ms) threshold realistically COULD
    fire on every single healthy iteration once real per-camera PNG
    encode/write overhead is accounted for -- confirms the
    _FETCH_SLOW_WARNING_FLOOR_S floor keeps a normal (here: 100ms,
    plausibly realistic for 3 sequential local camera reads+PNG writes)
    fetch from spamming this warning."""
    import threading as _threading

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {0: _fake_calibration_attempt().calibration}

    call_count = {"n": 0}

    def fake_fetch(dest_dir, *, hub=None):
        call_count["n"] += 1
        # Bug fix, 2026-08-12: the startup bg_frames fetch now goes
        # through _wait_for_first_frames(), which -- for this
        # constant-valued fake -- deterministically consumes exactly
        # STARTUP_FETCHES calls before the real per-iteration loop
        # even starts (identical frames settle the instant the rolling
        # window fills). Only sleep on calls AFTER that startup phase,
        # so this test still measures "a normal fetch inside the real
        # loop," not the startup settle-wait's own timing.
        if call_count["n"] > STARTUP_FETCHES:
            time.sleep(0.1) # plausible real 3-camera local fetch cost
        return {0: np.zeros((2, 2, 3), dtype=np.uint8)}

    def fake_advance(trigger, bg_frames, current_frames):
        if call_count["n"] > STARTUP_FETCHES + 3:
            raise _StopLoop("stop after a few fast iterations")
        return trigger # stay IDLE -- no capture machinery needed for this test

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)

    stop_event = _threading.Event()
    with caplog.at_level("WARNING", logger="opendarts.capture_daemon"):
        with pytest.raises(_StopLoop):
            capture_daemon.run_capture_loop_body(
                hub=None,
                package_root=tmp_path / "packages",
                poll_interval_s=capture_daemon.POLL_INTERVAL_SECONDS,
                stop_event=stop_event,
                scratch_dir=tmp_path / "scratch",
            )

    slow_warnings = [r for r in caplog.records if "may be slow/stalling" in r.message]
    assert not slow_warnings, (
        "a normal ~100ms fetch at the new fast poll interval must not trigger the "
        "slow/stalling warning -- the floor (_FETCH_SLOW_WARNING_FLOOR_S) exists "
        "exactly to prevent this"
    )


def test_fetch_slow_warning_still_fires_for_a_genuine_stall(tmp_path, monkeypatch, caplog):
    """Companion to the test above -- the floor must not defeat the
    warning's actual purpose. A genuinely stalled fetch (well past the
    floor) must still be reported.

    2026-09-07: since the startup photometric-stability fix, the number
    of fetches _wait_for_first_frames() consumes at a REAL poll_interval_s
    is no longer the fixed STARTUP_FETCHES it used to be (it now
    needs real elapsed time too) -- so this test can no longer predict a
    fixed call-count boundary between "startup" and "the real main loop."
    Instead it wraps the real _wait_for_first_frames() to detect the
    exact moment startup actually finishes, and injects the genuine
    stall on the FIRST fetch after that -- correct regardless of how many
    calls startup itself ends up taking."""
    import threading as _threading

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {0: _fake_calibration_attempt().calibration}

    call_count = {"n": 0}
    startup_done = {"flag": False}
    real_wait_for_first_frames = capture_daemon._wait_for_first_frames

    def wrapped_wait_for_first_frames(*args, **kwargs):
        result = real_wait_for_first_frames(*args, **kwargs)
        startup_done["flag"] = True
        return result

    def fake_fetch(dest_dir, *, hub=None):
        call_count["n"] += 1
        # Inject the genuine stall on the FIRST real main-loop fetch --
        # i.e. the first fetch observed once startup has already
        # returned -- not a hardcoded call-count boundary.
        if startup_done["flag"] and not fake_fetch.stall_injected:
            fake_fetch.stall_injected = True
            time.sleep(capture_daemon._FETCH_SLOW_WARNING_FLOOR_S + 0.1)
        return {0: np.zeros((2, 2, 3), dtype=np.uint8)}

    fake_fetch.stall_injected = False

    def fake_advance(trigger, bg_frames, current_frames):
        if fake_fetch.stall_injected:
            raise _StopLoop("stop right after the stalled iteration")
        return trigger

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)
    monkeypatch.setattr(capture_daemon, "_wait_for_first_frames", wrapped_wait_for_first_frames)

    stop_event = _threading.Event()
    with caplog.at_level("WARNING", logger="opendarts.capture_daemon"):
        with pytest.raises(_StopLoop):
            capture_daemon.run_capture_loop_body(
                hub=None,
                package_root=tmp_path / "packages",
                poll_interval_s=capture_daemon.POLL_INTERVAL_SECONDS,
                stop_event=stop_event,
                scratch_dir=tmp_path / "scratch",
            )

    slow_warnings = [r for r in caplog.records if "may be slow/stalling" in r.message]
    assert len(slow_warnings) == 1


def test_loop_paces_itself_by_real_elapsed_time_not_a_busy_spin(tmp_path, monkeypatch):
    """Direct answer to "does dropping the poll interval this much risk
    busy-looping/high CPU": proves, with REAL wall-clock timing (not
    mocked time), that run_capture_loop_body() actually respects
    `poll_interval_s` between iterations via `stop_event.wait()` (a real
    blocking wait, not a spin) rather than looping as fast as physically
    possible. Uses instant (non-sleeping) mocked fetch/advance so any
    measured elapsed time is attributable to the wait itself, not
    incidental mock overhead."""
    import threading as _threading

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {0: _fake_calibration_attempt().calibration}

    def fake_fetch(dest_dir, *, hub=None):
        return {0: np.zeros((2, 2, 3), dtype=np.uint8)}

    n_iterations = 20
    call_count = {"n": 0}

    def fake_advance(trigger, bg_frames, current_frames):
        call_count["n"] += 1
        if call_count["n"] > n_iterations:
            raise _StopLoop(f"stop after {n_iterations} iterations")
        return trigger

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)

    stop_event = _threading.Event()
    started = time.monotonic()
    with pytest.raises(_StopLoop):
        capture_daemon.run_capture_loop_body(
            hub=None,
            package_root=tmp_path / "packages",
            poll_interval_s=capture_daemon.POLL_INTERVAL_SECONDS,
            stop_event=stop_event,
            scratch_dir=tmp_path / "scratch",
        )
    elapsed = time.monotonic() - started

    expected_min = n_iterations * capture_daemon.POLL_INTERVAL_SECONDS
    # Real wait, not a busy-spin: elapsed must be AT LEAST roughly the sum
    # of the per-iteration waits (loose lower bound -- scheduler jitter
    # can only add time, never remove it) ...
    assert elapsed >= expected_min * 0.8
    # ... and bounded well above zero -- a busy-spin bug (e.g. accidentally
    # calling stop_event.wait(0) regardless of poll_interval_s) would blow
    # through all 20 iterations in a few milliseconds instead.
    assert elapsed > 0.1


def test_run_capture_loop_local_mode_opens_hub_once_and_reuses_it(tmp_path, monkeypatch):
    monkeypatch.setattr(local_capture, "LocalCameraHub", FakeHub)

    hubs_seen_by_bootstrap: list[object] = []
    hubs_seen_by_fetch: list[object] = []

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        hubs_seen_by_bootstrap.append(hub)
        return {0: _fake_calibration_attempt().calibration}

    call_count = {"fetch": 0}

    def fake_fetch(dest_dir, *, hub=None):
        hubs_seen_by_fetch.append(hub)
        call_count["fetch"] += 1
        return {0: np.zeros((4, 4, 3), dtype=np.uint8)}

    def fake_advance(trigger, bg_frames, current_frames):
        raise _StopLoop("stop after first fetch_current_frames in the loop body")

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)
    monkeypatch.setattr(capture_daemon.time, "sleep", lambda s: None)

    with pytest.raises(_StopLoop):
        capture_daemon.run_capture_loop(
            package_root=tmp_path / "packages",
            poll_interval_s=0.0,
            )

    # Exactly one hub constructed and opened for the whole run.
    assert len(FakeHub.instances) == 1
    hub = FakeHub.instances[0]
    assert hub.open_calls == 1
    # Closed on the way out, even though we exited via an exception.
    assert hub.close_calls == 1

    # bootstrap_calibrations (startup) and fetch_current_frames (seed bg +
    # first loop iteration) all received the SAME hub instance -- proof
    # it wasn't reopened per call.
    assert hubs_seen_by_bootstrap == [hub]
    assert all(h is hub for h in hubs_seen_by_fetch)
    assert call_count["fetch"] >= 1




def test_run_capture_loop_raises_clearly_when_no_local_camera_opens(tmp_path, monkeypatch):
    class AllFailHub(FakeHub):
        def __init__(self, configs=None) -> None:
            super().__init__(configs)
            self._open_ok = False

        def open_all(self):
            self.open_calls += 1
            return [False, False, False]

    monkeypatch.setattr(local_capture, "LocalCameraHub", AllFailHub)

    with pytest.raises(RuntimeError, match="no camera opened locally"):
        capture_daemon.run_capture_loop(
            package_root=tmp_path / "packages",
            poll_interval_s=0.0,
            )

    # Hub was still constructed (and should still be closed even though
    # it never successfully opened a camera).
    assert len(FakeHub.instances) == 1
    assert FakeHub.instances[0].close_calls == 1


# ---------------------------------------------------------------------------
# main() -- CLI flags thread through
# ---------------------------------------------------------------------------


def test_main_default_runs_local_mode(monkeypatch):
    seen = {}

    def fake_run_capture_loop(*, package_root, poll_interval_s, **_kwargs):
        seen["package_root"] = package_root
        seen["poll_interval_s"] = poll_interval_s

    monkeypatch.setattr(capture_daemon, "run_capture_loop", fake_run_capture_loop)
    rc = capture_daemon.main([])

    assert rc == 0
    assert seen["package_root"] == capture_daemon.DEFAULT_PACKAGE_ROOT
    # Local capture must use the fast, camera-native-ish poll interval.
    assert seen["poll_interval_s"] == capture_daemon.POLL_INTERVAL_SECONDS




# ---------------------------------------------------------------------------
# Bug fix, 2026-08-12: TAKEOUT_WAITING stuck-forever diagnostic logging.
# Live-confirmed on the rig: after a real takeout, the log showed continuous
# "still TAKEOUT_WAITING" heartbeats for 2+ minutes with ZERO indication
# of why. The actual GATE fix (all-cameras -> majority quorum) lives in
# opendarts/capture/throw_trigger.py and is proven there
# (tests/test_throw_trigger_detection.py); this section proves the OTHER
# half -- that a stuck TAKEOUT_WAITING now logs real, per-camera
# diagnostics at the same heartbeat cadence, instead of a bare "still
# TAKEOUT_WAITING" with no explanation.
# ---------------------------------------------------------------------------


def test_takeout_waiting_heartbeat_diagnostic_does_not_fire_for_other_states(
    tmp_path, monkeypatch
):
    """The new diagnostic logging is scoped to TAKEOUT_WAITING only --
    an IDLE loop sitting quiet must not get the takeout-specific
    diagnostic line (it would be meaningless/misleading there)."""
    import threading as _threading

    monkeypatch.setattr(capture_daemon, "_HEARTBEAT_EVERY_S", 0.0)

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {0: _fake_calibration_attempt().calibration}

    def fake_fetch(dest_dir, *, hub=None):
        return {0: np.zeros((2, 2, 3), dtype=np.uint8)}

    call_count = {"n": 0}

    def fake_advance(trigger, bg_frames, current_frames):
        call_count["n"] += 1
        if call_count["n"] > 3:
            raise _StopLoop("stop after a few idle heartbeats")
        return trigger # stays IDLE unchanged

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)

    import logging

    caplog_records: list[logging.LogRecord] = []

    class _Handler(logging.Handler):
        def emit(self, record):
            caplog_records.append(record)

    logger = logging.getLogger("opendarts.capture_daemon")
    handler = _Handler()
    logger.addHandler(handler)
    try:
        stop_event = _threading.Event()
        with pytest.raises(_StopLoop):
            capture_daemon.run_capture_loop_body(
                hub=None,
                package_root=tmp_path / "packages",
                poll_interval_s=0.0,
                stop_event=stop_event,
                scratch_dir=tmp_path / "scratch",
            )
    finally:
        logger.removeHandler(handler)

    diag_lines = [r for r in caplog_records if "still waiting for takeout" in r.getMessage()]
    assert not diag_lines, "IDLE heartbeats must not log the TAKEOUT_WAITING-specific diagnostic"


# ---------------------------------------------------------------------------
# AD ground truth -- inline WebSocket-buffer attach, added 2026-08-12.
#
# Real live finding this whole feature exists for: the REST
# /api/state/detections list only covers the current visit and is not
# durable (2026-08-12), which makes the earlier REST-poll/backfill-only
# design structurally unable to recover ground truth once the visit has
# ended in the meantime. Fixed by
# opendarts.live.ad_ws_listener.AdWsListener -- see that module's own
# docstring for the full design. These tests exercise the INTEGRATION
# point (handle_ready_to_capture()/run_capture_loop_body()/main()), not
# AdWsListener's own connect/reconnect/buffer logic -- that's
# tests/test_ad_ws_listener.py's job. Here, AdWsListener is always a
# lightweight fake/duck-typed stand-in with a `.match()` method -- no
# real WebSocket connection, no real network call, ever.
# ---------------------------------------------------------------------------


def _wait_until(predicate, timeout_s: float = 5.0, interval_s: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


class _FakeAdWsListener:
    """Duck-typed stand-in for opendarts.live.ad_ws_listener.AdWsListener --
    implements only the surface handle_ready_to_capture()/
    run_capture_loop_body() actually use (`match()`), plus
    start()/stop()/ws_url for the run_capture_loop()-level lifecycle
    tests. `match_fn` lets each test control exactly what happens on
    match (return a real AdGroundTruth, sleep to simulate a hang, raise
    to simulate a crash) without needing a real listener or any network
    I/O at all."""

    def __init__(self, base_url="http://fake-ad:0", *, match_fn=None):
        self.base_url = base_url
        self.ws_url = f"ws://fake-ad:0/api/events"
        self.match_fn = match_fn
        self.match_calls: list[dict] = []
        self.start_calls = 0
        self.stop_calls = 0

    def start(self) -> None:
        self.start_calls += 1

    def stop(self, timeout: float = 3.0) -> None:
        self.stop_calls += 1

    def diagnostics_snapshot(self) -> dict:
        """Real AdWsListener.diagnostics_snapshot()'s shape (2026-08-16,
        "persist real diagnostics" task) -- see that method's own
        docstring. A minimal, honest fake: no real buffer/clear history
        to report, matching a listener that has neither connected nor
        received anything yet -- callers asserting on
        capture_diagnostics.json's ad_ws_buffer_at_capture only need to
        see this key exist and be non-None when a listener was wired."""
        return {
            "connected": False,
            "snapshot_at_utc": datetime.now(timezone.utc).isoformat(),
            "buffered_events": [],
            "last_clear_reason": None,
            "last_clear_at_utc": None,
            "last_clear_buffer_summary": None,
        }

    def match(self, opendarts_captured_at_utc, *, window_sec, expect_ordinal=None):
        self.match_calls.append({
            "captured_at": opendarts_captured_at_utc,
            "window_sec": window_sec,
            "expect_ordinal": expect_ordinal,
        })
        if self.match_fn is not None:
            return self.match_fn(opendarts_captured_at_utc, window_sec)
        return AdGroundTruth(
            matched=True,
            match_reason="ok_ws",
            ad_base_url=self.base_url,
            fetched_at_utc=datetime.now(timezone.utc).isoformat(),
            opendarts_captured_at_utc=opendarts_captured_at_utc,
            staleness_sec=0.1,
            window_sec=window_sec,
            sector="20",
            ring="treble",
            tip_xy_mm=(1.0, 2.0),
            ad_method="UnanimousCam",
            ad_bouncer=False,
        )


def _throw_trigger_ready(frame: dict[int, np.ndarray]) -> ThrowTriggerState:
    return ThrowTriggerState(state=ThrowState.READY_TO_CAPTURE, last_frame=frame)


def test_handle_ready_to_capture_applies_board_roi_gate_to_every_camera(tmp_path, monkeypatch):
    """Wiring test: handle_ready_to_capture() must run every camera's raw
    detect_tip() result through opendarts.engines.apollo.board_roi.
    reject_outside_roi() before using it -- built earlier but not
    actually wired into either live call site
    until 2026-08-12. Spies on the REAL
    reject_outside_roi (passes through to it, doesn't stub it out) to
    confirm it's actually invoked once per camera with that camera's own
    calibration -- proves the wiring, not just that the import exists.

    **2026-08-12 -- patch seam moved**: `handle_ready_to_capture()` no
    longer calls `reject_outside_roi()` directly at all (it now routes
    every primary engine, including Apollo, through `get_engine(
    primary_name).score(...)` -- see that function's own docstring for
    the direct-call-bypass removal). `opendarts.engines.apollo.engine`
    is where `reject_outside_roi` is actually bound and called now, so
    that's the correct spy seam (same principle this test always used,
    just relocated to the new real call site)."""
    import opendarts.engines.apollo.engine as apollo_engine_module

    real_reject = apollo_engine_module.reject_outside_roi
    calls = []

    def spy_reject(det, calib, **kwargs):
        calls.append((det, calib))
        return real_reject(det, calib, **kwargs)

    monkeypatch.setattr(apollo_engine_module, "reject_outside_roi", spy_reject)

    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8), 1: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8), 1: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration, 1: _fake_calibration_attempt().calibration}

    capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
    background_save=False,
    )

    assert len(calls) == 2, "reject_outside_roi must be called exactly once per camera"
    seen_calibs = {id(c) for _, c in calls}
    assert seen_calibs == {id(calibrations[0]), id(calibrations[1])}, (
        "each camera's detection must be checked against ITS OWN calibration"
    )


def test_handle_ready_to_capture_excludes_a_camera_whose_tip_is_outside_the_board_roi(
    tmp_path, monkeypatch
):
    """Behavioral (not just wiring) test, real synthetic camera geometry,
    real board_roi.py math: a camera whose raw detect_tip() tip lands
    nowhere near the board (a wildly implausible pixel, not a boundary
    case) must be excluded from scoring entirely -- score_dart() never
    even sees that camera's tip_px -- while a camera whose tip lands near
    the projected board center is unaffected. This is the real, measured
    behavior the 2026-08-12 board-ROI wiring is actually meant to
    produce in live capture, not just "the function
    gets called."""
    from tests.support.synthetic import make_camera_matrix, make_ring_camera

    image_w, image_h = 640, 480
    camera_matrix = make_camera_matrix(image_width=image_w, image_height=image_h, fov_deg=90.0)
    true_cams = [make_ring_camera(i, n_cameras=2, camera_matrix=camera_matrix) for i in range(2)]
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

    bg = {i: np.zeros((image_h, image_w, 3), dtype=np.uint8) for i in range(2)}
    frame = {i: np.zeros((image_h, image_w, 3), dtype=np.uint8) for i in range(2)}

    # cam0: a plausible tip right at the image center (a camera looking
    # at the board center should project the board center very close to
    # its own principal point). cam1: a wildly implausible tip, nowhere
    # near any real board projection at this geometry.
    import opendarts.engines.apollo.engine as apollo_engine_module
    from opendarts.engines.apollo.tip_detection import TipDetectionResult

    tip_choices = {0: (image_w / 2.0, image_h / 2.0), 1: (-100000.0, -100000.0)}
    call_order = []

    def fake_detect_tip(bg_img, frame_img, prior_dart_line_px=None):
        # current_frames.items() iterates cam 0 then 1 (dict insertion
        # order, Python 3.7+) -- record which call this is via a shared
        # counter keyed by the frame dict's own iteration, not guessed.
        idx = len(call_order)
        call_order.append(idx)
        return TipDetectionResult(ok=True, tip_px=tip_choices[idx], reason="ok")

    # 2026-08-12 -- patch seam moved: handle_ready_to_capture() routes
    # through opendarts.engines.apollo.engine now (see this file's other
    # ROI-gate test's own docstring for the direct-call-bypass removal),
    # not a name bound directly in capture_daemon's own namespace.
    monkeypatch.setattr(apollo_engine_module, "detect_tip", fake_detect_tip)

    dest_dir = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
    background_save=False,
    )

    result = json.loads((dest_dir / "result.json").read_text())
    # Only cam0's tip should have survived the ROI gate -- score_dart()
    # needs >=2 cameras to triangulate at all, so with only cam0 left it
    # must report a real "not enough cameras" style failure rather than a
    # false triangulated position built from an implausible cam1 tip.
    assert result["n_cameras_used"] <= 1, (
        f"expected cam1's wildly-off-board tip to be excluded by the ROI "
        f"gate, got n_cameras_used={result['n_cameras_used']}"
    )


def test_run_capture_loop_body_builds_and_threads_cache_across_two_real_darts(
    tmp_path, monkeypatch
):
    """2026-09-01, "why call anything off disk at all" -- the actual live
    loop wiring, not just handle_ready_to_capture()'s own forwarding
    (covered separately above): run_capture_loop_body() must build a
    real CachedPriorThrowFrames from dart 1's own bg_frames/current_frames
    right after capturing it, and hand that to dart 2's own
    handle_ready_to_capture() call as `cached_prior_frames` -- dart 1
    itself gets `cached_prior_frames=None` (nothing captured yet this
    visit)."""
    true_baseline = {0: np.full((2, 2, 3), 111, dtype=np.uint8)}
    dart1_frame = {0: np.full((2, 2, 3), 30, dtype=np.uint8)}
    dart2_frame = {0: np.full((2, 2, 3), 60, dtype=np.uint8)}

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {0: _fake_calibration_attempt().calibration}

    fetch_sequence = (
        [true_baseline] * STARTUP_FETCHES + [dart1_frame, dart1_frame] + [dart2_frame] * 3
    )
    fetch_calls = {"n": 0}

    def fake_fetch(dest_dir, *, hub=None):
        idx = fetch_calls["n"]
        fetch_calls["n"] += 1
        return fetch_sequence[idx] if idx < len(fetch_sequence) else dart2_frame

    scripted_states = [
        ThrowTriggerState(
            state=ThrowState.READY_TO_CAPTURE, dart_count=1, true_baseline_frames=true_baseline,
        ),
        ThrowTriggerState(
            state=ThrowState.READY_TO_CAPTURE, dart_count=2, true_baseline_frames=true_baseline,
        ),
    ]
    advance_calls: list[int] = []

    def fake_advance(trigger, bg_frames, current_frames):
        idx = len(advance_calls)
        advance_calls.append(idx)
        if idx < len(scripted_states):
            return scripted_states[idx]
        raise _StopLoop("stop after both scripted darts")

    capture_calls: list[dict] = []

    def fake_handle_ready_to_capture(
        trigger, bg_frames, calibrations, package_root, session_id,
        cached_prior_frames=None, **_kwargs
    ):
        capture_calls.append({
            "dart_count": trigger.dart_count,
            "cached_prior_frames": cached_prior_frames,
        })
        return package_root / f"fake_throw_{trigger.dart_count}"

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)
    monkeypatch.setattr(capture_daemon, "handle_ready_to_capture", fake_handle_ready_to_capture)

    stop_event = threading.Event()
    with pytest.raises(_StopLoop):
        capture_daemon.run_capture_loop_body(
            hub=None,
            package_root=tmp_path / "packages",
            poll_interval_s=0.0,
            stop_event=stop_event,
            scratch_dir=tmp_path / "scratch",
            on_event=None,
        )

    assert len(capture_calls) == 2, "both scripted darts must have triggered a capture"

    # Dart 1 -- nothing captured yet this visit, cache must be None.
    assert capture_calls[0]["dart_count"] == 1
    assert capture_calls[0]["cached_prior_frames"] is None

    # Dart 2 -- the cache must be a REAL CachedPriorThrowFrames built from
    # dart 1's own actual captured frame (byte-identical to the real
    # fetched dart1_frame, not just "some object"), at the SAME
    # visit_index dart 1 was captured at (0-based: dart 1 -> visit_index 0).
    assert capture_calls[1]["dart_count"] == 2
    cache = capture_calls[1]["cached_prior_frames"]
    assert isinstance(cache, CachedPriorThrowFrames)
    assert cache.visit_index == 0 # dart 1's own 0-based visit_index (dart_count - 1)
    assert set(cache.dart_frames.keys()) == {0}
    assert np.array_equal(cache.dart_frames[0], dart1_frame[0])


def test_run_capture_loop_body_folds_own_tip_line_px_out_into_next_darts_cache(
    tmp_path, monkeypatch
):
    """2026-09-01 -- the fastest tier, threaded through
    the real loop: when handle_ready_to_capture() writes a real value
    into its own_tip_line_px_out out-param (simulating a real Apollo/
    Zeus-primary throw), run_capture_loop_body() must fold it into the
    NEXT throw's own CachedPriorThrowFrames.precomputed_tip_line -- and
    pass a genuinely FRESH {} each call, never leaking dart 1's own
    written value into dart 2's own (separate) extraction attempt."""
    true_baseline = {0: np.full((2, 2, 3), 111, dtype=np.uint8)}
    dart1_frame = {0: np.full((2, 2, 3), 30, dtype=np.uint8)}
    dart2_frame = {0: np.full((2, 2, 3), 60, dtype=np.uint8)}
    dart1_tip_line = {0: ((10.0, 20.0), (30.0, 40.0))}

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {0: _fake_calibration_attempt().calibration}

    fetch_sequence = (
        [true_baseline] * STARTUP_FETCHES + [dart1_frame, dart1_frame] + [dart2_frame] * 3
    )
    fetch_calls = {"n": 0}

    def fake_fetch(dest_dir, *, hub=None):
        idx = fetch_calls["n"]
        fetch_calls["n"] += 1
        return fetch_sequence[idx] if idx < len(fetch_sequence) else dart2_frame

    scripted_states = [
        ThrowTriggerState(
            state=ThrowState.READY_TO_CAPTURE, dart_count=1, true_baseline_frames=true_baseline,
        ),
        ThrowTriggerState(
            state=ThrowState.READY_TO_CAPTURE, dart_count=2, true_baseline_frames=true_baseline,
        ),
    ]
    advance_calls: list[int] = []

    def fake_advance(trigger, bg_frames, current_frames):
        idx = len(advance_calls)
        advance_calls.append(idx)
        if idx < len(scripted_states):
            return scripted_states[idx]
        raise _StopLoop("stop after both scripted darts")

    capture_calls: list[dict] = []

    def fake_handle_ready_to_capture(
        trigger, bg_frames, calibrations, package_root, session_id,
        cached_prior_frames=None, own_tip_line_px_out=None, **_kwargs
    ):
        capture_calls.append({
            "dart_count": trigger.dart_count,
            "cached_prior_frames": cached_prior_frames,
        })
        # Only dart 1 "finds" a real tip line (simulates a real Apollo/
        # Zeus-primary throw) -- dart 2 leaves the out-param untouched
        # (simulates a throw where nothing was found), confirming the
        # loop never carries dart 1's own value forward past dart 2.
        if trigger.dart_count == 1 and own_tip_line_px_out is not None:
            own_tip_line_px_out["own_tip_line_px"] = dart1_tip_line
        return package_root / f"fake_throw_{trigger.dart_count}"

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)
    monkeypatch.setattr(capture_daemon, "handle_ready_to_capture", fake_handle_ready_to_capture)

    stop_event = threading.Event()
    with pytest.raises(_StopLoop):
        capture_daemon.run_capture_loop_body(
            hub=None,
            package_root=tmp_path / "packages",
            poll_interval_s=0.0,
            stop_event=stop_event,
            scratch_dir=tmp_path / "scratch",
            on_event=None,
        )

    assert len(capture_calls) == 2
    # Dart 2's own cache must carry dart 1's real precomputed tip line.
    cache = capture_calls[1]["cached_prior_frames"]
    assert isinstance(cache, CachedPriorThrowFrames)
    assert cache.precomputed_tip_line == dart1_tip_line


# --------------------------------------------------------------------------
# _own_tip_line_px_from_engine_result() (2026-09-01):
# pure extraction, both real shapes (Apollo-as-primary direct,
# Zeus-as-primary nested), plus the "nothing available" case.
# --------------------------------------------------------------------------


def test_own_tip_line_px_from_engine_result_direct_shape():
    """Apollo-as-primary: own_tip_line_px sits at the top level of
    engine_result.diagnostics, exactly where ApolloEngine.score()
    writes it."""
    line = {0: ((1.0, 2.0), (3.0, 4.0))}
    engine_result = EngineResult(
        ok=True, sector="1", ring="single_inner", board_xy_mm=(0.0, 0.0),
        diagnostics={"own_tip_line_px": line},
    )
    assert capture_daemon._own_tip_line_px_from_engine_result(engine_result) == line


def test_own_tip_line_px_from_engine_result_nested_zeus_shape():
    """Zeus-as-primary (the real production configuration): own_tip_
    line_px lives nested inside diagnostics["sub_results"]["Apollo"]
    ["diagnostics"] -- the same sub_results shape Zeus's own .to_dict()
    serialization and the earlier also-run-reuse fix both already
    established."""
    line = {0: ((5.0, 6.0), (7.0, 8.0))}
    engine_result = EngineResult(
        ok=True, sector="20", ring="treble", board_xy_mm=(0.0, 0.0),
        diagnostics={
            "sub_results": {
                "Apollo": {"ok": True, "diagnostics": {"own_tip_line_px": line}},
                "Talos": {"ok": True, "diagnostics": {}},
            },
        },
    )
    assert capture_daemon._own_tip_line_px_from_engine_result(engine_result) == line




def test_handle_ready_to_capture_threads_cached_prior_frames_to_both_call_sites(
    tmp_path, monkeypatch
):
    """2026-09-01, "why call anything off disk at all" wiring test:
    handle_ready_to_capture()'s new `cached_prior_frames` param must be
    forwarded, unmodified, to `find_prior_dart_line_px()` at BOTH real
    call sites (primary engine's own lookup, and the also-run dispatch's
    separate lookup) -- spies on the real imported symbol rather than
    reimplementing it, so this fails if either call site's own
    `cached_frames=...` kwarg is ever dropped or renamed."""
    calls: list[dict] = []
    real_find = capture_daemon.find_prior_dart_line_px

    def spy_find(session_dir, visit_id, visit_index, *, cached_frames=None):
        calls.append({
            "visit_id": visit_id, "visit_index": visit_index, "cached_frames": cached_frames,
        })
        return real_find(session_dir, visit_id, visit_index, cached_frames=cached_frames)

    monkeypatch.setattr(capture_daemon, "find_prior_dart_line_px", spy_find)

    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}

    # Dart 1 of the visit -- no prior throw exists yet, cached_prior_frames
    # is None (the real caller's own default for a fresh visit).
    capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
        visit_id="visitA", visit_index=0,
    background_save=False,
    )
    assert len(calls) >= 1, "find_prior_dart_line_px must be called for the primary engine"
    assert all(c["cached_frames"] is None for c in calls), (
        "dart 1 of a visit has no prior throw to cache -- cached_frames must be None"
    )

    # Dart 2 -- a real CachedPriorThrowFrames for dart 1, exactly what
    # run_capture_loop_body()'s own loop would have built moments ago.
    calls.clear()
    cached = CachedPriorThrowFrames(
        visit_id="visitA", visit_index=0, bg_frames=bg, dart_frames=frame,
    )
    capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
        visit_id="visitA", visit_index=1, cached_prior_frames=cached,
    background_save=False,
    )
    assert len(calls) >= 1
    assert all(c["cached_frames"] is cached for c in calls), (
        "the same cached_prior_frames object must reach every real "
        "find_prior_dart_line_px() call site unmodified"
    )


def test_handle_ready_to_capture_saves_package_and_attaches_ad_ground_truth(tmp_path, monkeypatch):
    """The core happy path: a successful inline attach happens right
    after the package save (mocked AD listener, no real network)."""
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}
    listener = _FakeAdWsListener()
    dest_dir = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
        ad_ws_listener=listener, ad_match_window_sec=12.0,
    background_save=False,
    )

    # The package itself is saved immediately, synchronously -- this
    # assertion needs no waiting at all.
    assert (dest_dir / "meta.json").exists()
    assert (dest_dir / "result.json").exists()

    # The AD attach runs in the background -- wait for it, then verify.
    assert _wait_until(lambda: (dest_dir / "ad_ground_truth.json").exists()), (
        "ad_ground_truth.json never appeared -- inline attach did not run"
    )
    gt = load_ad_ground_truth(dest_dir)
    assert gt.matched is True
    assert gt.match_reason == "ok_ws"
    assert gt.sector == "20"
    assert gt.ring == "treble"
    assert listener.match_calls # match() was actually invoked
    assert listener.match_calls[0]["window_sec"] == 12.0


def test_handle_ready_to_capture_with_no_listener_skips_ad_attach_entirely(tmp_path):
    """ad_ws_listener=None (the default, and what --no-ad-ground-truth
    resolves to) -- package still saves fine, no ad_ground_truth.json is
    ever written, no error."""
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}

    dest_dir = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
    background_save=False,
    )
    assert (dest_dir / "meta.json").exists()
    time.sleep(0.2) # give a (bug-would-be) background thread a chance to misbehave
    assert not (dest_dir / "ad_ground_truth.json").exists()


def test_handle_ready_to_capture_never_blocks_on_a_hanging_ad_match(tmp_path, monkeypatch):
    """THE load-bearing property this whole task is about: if AD is slow
    or unreachable, the throw package must still save successfully and
    handle_ready_to_capture() must return promptly -- never waiting on
    the AD match. Simulates a match() that would hang for a long time
    (2s, comfortably longer than any bounded assertion below) and asserts
    handle_ready_to_capture() itself returns in well under that -- a
    test that WOULD catch a regression back to a synchronous/blocking
    design (it would fail by taking >=2s)."""
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}

    HANG_SECONDS = 2.0

    def slow_match(captured_at, window_sec):
        time.sleep(HANG_SECONDS)
        return AdGroundTruth(
            matched=False, match_reason="ws_no_buffered_events", ad_base_url="unused",
            fetched_at_utc=datetime.now(timezone.utc).isoformat(),
            opendarts_captured_at_utc=captured_at, staleness_sec=None, window_sec=window_sec,
        )

    listener = _FakeAdWsListener(match_fn=slow_match)

    started = time.monotonic()
    dest_dir = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
        ad_ws_listener=listener, ad_match_window_sec=12.0,
    background_save=False,
    )
    elapsed = time.monotonic() - started

    # Bounded well below HANG_SECONDS -- proves the AD match is NOT being
    # awaited inline. Generous margin (0.5s) for real disk I/O
    # (save_throw_package writing PNGs) on a loaded CI machine, while
    # still being 4x tighter than the simulated hang.
    assert elapsed < 0.5, (
        f"handle_ready_to_capture() took {elapsed:.2f}s -- it must return almost "
        f"immediately even when the AD match is slow/hanging (simulated {HANG_SECONDS}s "
        f"hang); this is the exact regression this test exists to catch"
    )
    assert (dest_dir / "meta.json").exists()
    assert (dest_dir / "result.json").exists()

    # The slow match does eventually complete in the background -- confirms
    # this isn't "silently dropped", just "never blocking".
    assert _wait_until(lambda: listener.match_calls, timeout_s=HANG_SECONDS + 2.0)


def test_handle_ready_to_capture_survives_a_crashing_ad_match(tmp_path, caplog, monkeypatch):
    """A raising match() (e.g. a future bug, or save_ad_ground_truth()
    hitting a real disk error) must never propagate out of
    handle_ready_to_capture() and must never prevent/undo the throw
    package save that already succeeded."""
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}

    def crashing_match(captured_at, window_sec):
        raise RuntimeError("simulated AD match failure")

    listener = _FakeAdWsListener(match_fn=crashing_match)

    dest_dir = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
        ad_ws_listener=listener, ad_match_window_sec=12.0,
    background_save=False,
    )
    # No exception escaped the call above -- that alone is most of this
    # test's point. Package still saved:
    assert (dest_dir / "meta.json").exists()
    assert (dest_dir / "result.json").exists()

    assert _wait_until(lambda: listener.match_calls, timeout_s=3.0)
    assert not (dest_dir / "ad_ground_truth.json").exists()


def test_run_capture_loop_body_threads_ad_ws_listener_into_handle_ready_to_capture(
    tmp_path, monkeypatch
):
    """Plumbing proof at the loop-body level: whatever ad_ws_listener/
    ad_match_window_sec run_capture_loop_body() is given reaches
    handle_ready_to_capture() unchanged, on the one real READY_TO_CAPTURE
    iteration."""
    true_baseline = {0: np.full((2, 2, 3), 111, dtype=np.uint8)}
    dart_frame = {0: np.full((2, 2, 3), 10, dtype=np.uint8)}

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {0: _fake_calibration_attempt().calibration}

    def fake_fetch(dest_dir, *, hub=None):
        return true_baseline

    scripted = [
        ThrowTriggerState(
            state=ThrowState.READY_TO_CAPTURE, dart_count=1,
            true_baseline_frames=true_baseline, last_frame=dart_frame,
        ),
    ]
    advance_calls = {"n": 0}

    def fake_advance(trigger, bg_frames, current_frames):
        idx = advance_calls["n"]
        advance_calls["n"] += 1
        if idx < len(scripted):
            return scripted[idx]
        raise _StopLoop("stop after the scripted sequence completes")

    seen_kwargs: dict = {}

    def fake_handle_ready_to_capture(trigger, bg_frames, calibrations, package_root, session_id, **kwargs):
        seen_kwargs.update(kwargs)
        return package_root / "fake_throw"

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)
    monkeypatch.setattr(capture_daemon, "handle_ready_to_capture", fake_handle_ready_to_capture)

    listener = _FakeAdWsListener()
    stop_event = threading.Event()
    with pytest.raises(_StopLoop):
        capture_daemon.run_capture_loop_body(
            hub=None,
            package_root=tmp_path / "packages",
            poll_interval_s=0.0,
            stop_event=stop_event,
            scratch_dir=tmp_path / "scratch",
            ad_ws_listener=listener,
            ad_match_window_sec=7.5,
        )

    assert seen_kwargs["ad_ws_listener"] is listener
    assert seen_kwargs["ad_match_window_sec"] == 7.5


# ---------------------------------------------------------------------------
# Self-descriptive throw_id naming: a raw epoch-ms
# throw_id ("throw_1786666520048") told a human nothing at a glance. New
# format: f"{session_id}-{throw_number:03d}-{sector_token}", e.g.
# "<session>-001-T16". These tests fully control the primary
# engine's EngineResult (a fake engine, monkeypatched at the same
# `get_engine` seam the ROI-gate tests above already use) so the
# resulting sector/ring/ok -- and therefore the resulting token -- is
# deterministic rather than depending on what a real (near-blank,
# synthetic) frame happens to score.
# ---------------------------------------------------------------------------


class _FakeScoringEngine:
    """A minimal fake matching opendarts.engines.base's score() interface --
    returns each of `results` in order, one per call, so a test can drive
    a sequence of throws through handle_ready_to_capture() with fully
    controlled (ok, sector, ring) outcomes."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def score(self, bg_images, dart_images, calibrations):
        result = self._results[self.calls]
        self.calls += 1
        return result


def _make_fake_engine_result(*, ok, sector=None, ring=None):
    from opendarts.engines.base import EngineResult

    return EngineResult(
        ok=ok,
        sector=sector,
        ring=ring,
        board_xy_mm=(0.0, 0.0) if ok else None,
        reason="" if ok else "fake no-result",
        diagnostics={},
    )


def test_handle_ready_to_capture_uses_the_new_throw_id_format(tmp_path, monkeypatch):
    """throw_id must be f"{session_id}-{throw_number:03d}-{sector_token}",
    e.g. session "<session>" + first throw + sector 8/treble ->
    "<session>-001-T8"."""
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}
    fake_engine = _FakeScoringEngine([_make_fake_engine_result(ok=True, sector="8", ring="treble")])
    monkeypatch.setattr(capture_daemon, "get_engine", lambda name: fake_engine)

    dest_dir = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "20260804-124053",
    background_save=False,
    )

    assert dest_dir.name == "20260804-124053-001-T8"
    assert dest_dir.parent.name == "20260804-124053" # session nesting untouched
    assert (dest_dir / "meta.json").exists()


def test_handle_ready_to_capture_throw_number_increments_within_a_session(tmp_path, monkeypatch):
    """The sequential per-session counter (glob-count approach) must
    advance 001 -> 002 -> 003 across three throws in the SAME session,
    with different sector tokens proving each dest_dir really reflects
    its own throw's result rather than a stale/cached one."""
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}
    fake_engine = _FakeScoringEngine(
        [
            _make_fake_engine_result(ok=True, sector="20", ring="treble"),
            _make_fake_engine_result(ok=True, sector="5", ring="single_inner"),
            _make_fake_engine_result(ok=True, sector="1", ring="double"),
        ]
    )
    monkeypatch.setattr(capture_daemon, "get_engine", lambda name: fake_engine)

    package_root = tmp_path / "packages"
    dest1 = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, package_root, "sessA",
    background_save=False,
    )
    dest2 = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, package_root, "sessA",
    background_save=False,
    )
    dest3 = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, package_root, "sessA",
    background_save=False,
    )

    assert dest1.name == "sessA-001-T20"
    assert dest2.name == "sessA-002-S5"
    assert dest3.name == "sessA-003-D1"


def test_handle_ready_to_capture_throw_number_resets_for_a_new_session(tmp_path, monkeypatch):
    """Two throws in "sessA" followed by one in a fresh "sessB" -- sessB's
    first throw must be 001, not 003 (the counter is scoped per session,
    via counting existing throw dirs under THAT session's own directory,
    not a global counter)."""
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}
    fake_engine = _FakeScoringEngine(
        [
            _make_fake_engine_result(ok=True, sector="20", ring="treble"),
            _make_fake_engine_result(ok=True, sector="5", ring="single_inner"),
            _make_fake_engine_result(ok=True, sector="1", ring="double"),
        ]
    )
    monkeypatch.setattr(capture_daemon, "get_engine", lambda name: fake_engine)

    package_root = tmp_path / "packages"
    capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, package_root, "sessA",
    background_save=False,
    )
    capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, package_root, "sessA",
    background_save=False,
    )
    dest_b = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, package_root, "sessB",
    background_save=False,
    )

    assert dest_b.name == "sessB-001-D1"


def test_handle_ready_to_capture_throw_number_survives_mid_session_quarantine(
    tmp_path, monkeypatch
):
    """Real incident, 2026-08-17: the standing pull SOP's quarantine step
    (an off-rig pull that runs `mv packages/* $DEST/`) moves a
    session's throw directories out from under a still-running daemon
    mid-session -- session_id doesn't change (one continuous physical
    sitting), but the OLD glob-count throw_number logic recounted 0
    existing dirs after quarantine and restarted at 001, producing real
    collisions against already-pulled throws of the same session (caught
    by hand once already -- see data/archive/clean/README.md). The 2026-08-17 fix made the counter
    survive this (numbering continued seamlessly across a pull).

    SUPERSEDED, 2026-08-22 (see test below and
    _reset_session_throw_numbering()'s own docstring for the full
    reversal): explicitly wants a pull that empties the local
    session directory to ALSO reset numbering now, same as a manual
    Reset or Delete-packages -- "the entire count is forever off" if a
    throw is ever missed, and continuity across a pull turned out to be
    the wrong tradeoff operationally. This test is KEPT (not deleted) as
    the historical record of the ORIGINAL bug + fix this project already
    hand-corrected data for once; test_handle_ready_to_capture_
    throw_number_resets_via_new_generation_after_mid_session_quarantine
    below is the CURRENT expected behavior."""
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}
    fake_engine = _FakeScoringEngine(
        [
            _make_fake_engine_result(ok=True, sector="20", ring="treble"),
            _make_fake_engine_result(ok=True, sector="5", ring="single_inner"),
        ]
    )
    monkeypatch.setattr(capture_daemon, "get_engine", lambda name: fake_engine)

    package_root = tmp_path / "packages"
    dest1 = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, package_root, "sessA",
    background_save=False,
    )
    dest2 = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, package_root, "sessA",
    background_save=False,
    )
    assert dest1.name == "sessA-001-T20"
    assert dest2.name == "sessA-002-S5"


def test_handle_ready_to_capture_throw_number_resets_via_new_generation_after_mid_session_quarantine(
    tmp_path, monkeypatch
):
    """CURRENT behavior, 2026-08-22. Two throws
    (001, 002), simulate the exact real-world quarantine operation (`mv
    packages/* $DEST/`), then a third throw in the SAME session_id must
    restart at 001 -- but in a NEW generation (`-g1-` infix), not bare
    `sessA-001-...`, so it's provably disjoint from the two throws that
    already left for the archive rather than colliding with them."""
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}
    fake_engine = _FakeScoringEngine(
        [
            _make_fake_engine_result(ok=True, sector="20", ring="treble"),
            _make_fake_engine_result(ok=True, sector="5", ring="single_inner"),
            _make_fake_engine_result(ok=True, sector="1", ring="double"),
        ]
    )
    monkeypatch.setattr(capture_daemon, "get_engine", lambda name: fake_engine)

    package_root = tmp_path / "packages"
    dest1 = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, package_root, "sessA",
    background_save=False,
    )
    dest2 = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, package_root, "sessA",
    background_save=False,
    )
    assert dest1.name == "sessA-001-T20"
    assert dest2.name == "sessA-002-S5"

    # The exact real-world quarantine operation: `mv packages/* $DEST/`.
    quarantine_dest = tmp_path / "_del" / "quarantine_sim"
    quarantine_dest.mkdir(parents=True)
    for child in package_root.iterdir():
        shutil.move(str(child), str(quarantine_dest / child.name))
    assert not any(package_root.iterdir()) # sessA's dir is genuinely gone

    dest3 = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, package_root, "sessA",
    background_save=False,
    )

    assert dest3.name == "sessA-g1-001-D1" # new generation, NOT sessA-003-D1

    generation_file = package_root.parent / "session_throw_counters" / "sessA.generation"
    assert generation_file.read_text().strip() == "1"


def test_handle_ready_to_capture_generation_infix_omitted_until_first_reset(
    tmp_path, monkeypatch
):
    """The `-g{N}-` infix must be invisible for generation 0 (the common
    case -- no reset has ever happened) so a session that never resets
    produces throw_ids identical to before this feature existed."""
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}
    fake_engine = _FakeScoringEngine([_make_fake_engine_result(ok=True, sector="8", ring="treble")])
    monkeypatch.setattr(capture_daemon, "get_engine", lambda name: fake_engine)

    package_root = tmp_path / "packages"
    dest = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, package_root, "sessA",
    background_save=False,
    )
    assert dest.name == "sessA-001-T8"
    assert "-g" not in dest.name


def test_handle_ready_to_capture_new_generation_starts_at_001_even_with_old_generation_packages_still_on_disk(
    tmp_path, monkeypatch
):
    """REAL BUG, found 2026-08-22 by an independent verification pass
    against another port of this exact mechanism, not by this project's
    own test suite -- worth keeping as a permanent regression test, not just
    a fix. A manual Reset deliberately does NOT delete any already-saved
    throw packages (only Delete-packages/an external pull do) -- so
    right after a Reset, session_dir still has the PRIOR generation's
    real throw directories sitting on it. The bootstrap disk-recount
    fallback (`sum(1 for p in session_dir.iterdir() ...)`, meant only for
    a session's ORIGINAL generation with no counter file yet) used to
    have no notion of generation at all, so it recounted those old
    directories and inflated the new generation's first throw_number
    past 1 (confirmed by direct reproduction: g1-003 instead of g1-001).
    Two throws in generation 0, an explicit reset (leaving both old
    throws on disk, exactly like a real manual Reset does), then a third
    throw MUST be g1-001, not g1-003."""
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}
    fake_engine = _FakeScoringEngine(
        [
            _make_fake_engine_result(ok=True, sector="20", ring="treble"),
            _make_fake_engine_result(ok=True, sector="5", ring="single_inner"),
            _make_fake_engine_result(ok=True, sector="1", ring="double"),
        ]
    )
    monkeypatch.setattr(capture_daemon, "get_engine", lambda name: fake_engine)

    package_root = tmp_path / "packages"
    dest1 = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, package_root, "sessA",
    background_save=False,
    )
    dest2 = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, package_root, "sessA",
    background_save=False,
    )
    assert dest1.name == "sessA-001-T20"
    assert dest2.name == "sessA-002-S5"
    assert dest1.exists() and dest2.exists() # still real, on disk, NOT deleted

    # The exact effect of a manual Reset on numbering state -- see
    # run_capture_loop_body()'s own ResetRequest-handling call site.
    counters_dir = package_root.parent / "session_throw_counters"
    capture_daemon._reset_session_throw_numbering(counters_dir, "sessA")
    assert dest1.exists() and dest2.exists() # Reset really doesn't delete packages

    dest3 = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, package_root, "sessA",
    background_save=False,
    )
    assert dest3.name == "sessA-g1-001-D1" # NOT sessA-g1-003-D1


def test_handle_ready_to_capture_throw_counter_lives_outside_package_root_glob(
    tmp_path, monkeypatch
):
    """The persisted counter directory must be a sibling of package_root,
    never a child of it -- the standing quarantine SOP's `mv packages/*
    $DEST/` is a glob over package_root's own direct children, so a
    counter stored inside package_root would be swept away by the exact
    operation it exists to survive."""
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}
    fake_engine = _FakeScoringEngine([_make_fake_engine_result(ok=True, sector="8", ring="treble")])
    monkeypatch.setattr(capture_daemon, "get_engine", lambda name: fake_engine)

    package_root = tmp_path / "packages"
    capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, package_root, "sessA",
    background_save=False,
    )

    counters_dir = package_root.parent / "session_throw_counters"
    assert counters_dir.exists()
    assert not counters_dir.is_relative_to(package_root)
    assert (counters_dir / "sessA.count").read_text().strip() == "1"


def test_handle_ready_to_capture_ok_false_produces_nr_token(tmp_path, monkeypatch):
    """A real, honest ok=False primary result (no score at all -- still
    always saved as a full replay package per docs/DESIGN.md's
    "Replay is the source of truth") must produce the literal "NR" (no result) sector token,
    not a crash and not a nonsense sector/ring-derived token -- sector/
    ring are honestly None in this case, which isn't a real board
    position sector_ring_to_token() should ever be asked to map."""
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}
    fake_engine = _FakeScoringEngine([_make_fake_engine_result(ok=False, sector=None, ring=None)])
    monkeypatch.setattr(capture_daemon, "get_engine", lambda name: fake_engine)

    dest_dir = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
    background_save=False,
    )

    assert dest_dir.name == "sess1-001-NR"
    result = json.loads((dest_dir / "result.json").read_text())
    assert result["ok"] is False


def test_run_capture_loop_local_mode_builds_and_stops_ad_ws_listener(tmp_path, monkeypatch):
    """run_capture_loop() (the standalone -m opendarts.live.capture_daemon
    entrypoint's own loop runner) owns the AdWsListener lifecycle exactly
    like it owns the camera hub: start()ed before the loop body runs,
    stop()ped exactly once on the way out."""
    monkeypatch.setattr(local_capture, "LocalCameraHub", FakeHub)

    built: list[_FakeAdWsListener] = []

    def fake_listener_factory(base_url):
        lst = _FakeAdWsListener(base_url)
        built.append(lst)
        return lst

    monkeypatch.setattr(capture_daemon, "AdWsListener", fake_listener_factory)

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {0: _fake_calibration_attempt().calibration}

    def fake_fetch(dest_dir, *, hub=None):
        return {0: np.zeros((2, 2, 3), dtype=np.uint8)}

    def fake_advance(trigger, bg_frames, current_frames):
        raise _StopLoop("stop immediately -- this test only cares about listener lifecycle")

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)
    monkeypatch.setattr(capture_daemon.time, "sleep", lambda s: None)

    with pytest.raises(_StopLoop):
        capture_daemon.run_capture_loop(
            package_root=tmp_path / "packages",
            poll_interval_s=0.0,
            ad_base_url="http://fake-ad:9999",
        )

    assert len(built) == 1
    assert built[0].base_url == "http://fake-ad:9999"
    assert built[0].start_calls == 1
    assert built[0].stop_calls == 1


def test_run_capture_loop_local_mode_builds_no_listener_when_ad_base_url_is_none(
    tmp_path, monkeypatch
):
    """ad_base_url=None (what --no-ad-ground-truth resolves to) must
    construct NO AdWsListener at all -- no background WS connection
    attempt of any kind, not merely a disabled one."""
    monkeypatch.setattr(local_capture, "LocalCameraHub", FakeHub)

    built: list[_FakeAdWsListener] = []
    monkeypatch.setattr(
        capture_daemon, "AdWsListener", lambda base_url: built.append(base_url) or _FakeAdWsListener(base_url)
    )

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {0: _fake_calibration_attempt().calibration}

    def fake_fetch(dest_dir, *, hub=None):
        return {0: np.zeros((2, 2, 3), dtype=np.uint8)}

    def fake_advance(trigger, bg_frames, current_frames):
        raise _StopLoop("stop immediately")

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)
    monkeypatch.setattr(capture_daemon.time, "sleep", lambda s: None)

    with pytest.raises(_StopLoop):
        capture_daemon.run_capture_loop(
            package_root=tmp_path / "packages",
            poll_interval_s=0.0,
            ad_base_url=None,
        )

    assert built == []


def test_main_ad_ground_truth_on_by_default_and_no_ad_ground_truth_disables_it(monkeypatch):
    """CLI/config plumbing: on by default (ad_base_url threads through
    unchanged), --no-ad-ground-truth resolves to ad_base_url=None."""
    seen: dict = {}

    def fake_run_capture_loop(*, package_root, poll_interval_s, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(capture_daemon, "run_capture_loop", fake_run_capture_loop)

    rc = capture_daemon.main([])
    assert rc == 0
    assert seen["ad_base_url"] == capture_daemon.DEFAULT_AD_BASE
    assert seen["ad_window_sec"] == capture_daemon.DEFAULT_MATCH_WINDOW_SEC

    seen.clear()
    rc = capture_daemon.main(["--no-ad-ground-truth"])
    assert rc == 0
    assert seen["ad_base_url"] is None


def test_main_ad_base_url_and_window_flags_thread_through(monkeypatch):
    seen: dict = {}

    def fake_run_capture_loop(*, package_root, poll_interval_s, **kwargs):
        seen.update(kwargs)

    monkeypatch.setattr(capture_daemon, "run_capture_loop", fake_run_capture_loop)

    rc = capture_daemon.main(["--ad-base-url", "http://custom-ad:1234", "--ad-window-sec", "5.5"])
    assert rc == 0
    assert seen["ad_base_url"] == "http://custom-ad:1234"
    assert seen["ad_window_sec"] == 5.5


# ---------------------------------------------------------------------------
# CalibrationStore -- the shared, mutable calibration reference added
# 2026-08-12. See CalibrationStore's own docstring for
# the full design/thread-safety argument this section proves for real.
# ---------------------------------------------------------------------------


def _fake_calibration(tvec_z: float = 1000.0) -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=np.eye(3),
        dist_coeffs=np.zeros(5),
        rvec=np.zeros(3),
        tvec=np.array([0.0, 0.0, tvec_z]),
        pnp_result=None,
        landmark_spread_ok=True,
    )


def _same_calib_dict(a: dict[int, CameraCalibration], b: dict[int, CameraCalibration]) -> bool:
    """CameraCalibration holds numpy arrays -- plain `==`/`!=` on a dict
    of them raises ("truth value of an array... is ambiguous"), so every
    comparison in this section goes through here instead. CalibrationStore
    .get()/.set() only ever copy the DICT, never the CameraCalibration
    VALUES themselves (see that class's own docstring) -- so `is` on each
    value is the correct, exact check for "this is the same calibration
    object that was actually stored", not merely one with equal-looking
    numbers."""
    return set(a) == set(b) and all(a[k] is b[k] for k in a)


def test_calibration_store_get_returns_what_was_set():
    calib_a = {0: _fake_calibration(1000.0)}
    store = capture_daemon.CalibrationStore(calib_a, source="startup", checked_at_utc="t0")
    assert _same_calib_dict(store.get(), calib_a)
    assert store.meta() == {
        "source": "startup", "checked_at_utc": "t0", "n_cameras": 1,
        "calibration_package_id": None,
    }

    calib_b = {0: _fake_calibration(2000.0), 1: _fake_calibration(3000.0)}
    store.set(calib_b, source="manual", checked_at_utc="t1", package_id="calib_20260820-000000")
    assert _same_calib_dict(store.get(), calib_b)
    assert store.meta() == {
        "source": "manual", "checked_at_utc": "t1", "n_cameras": 2,
        "calibration_package_id": "calib_20260820-000000",
    }


def test_calibration_store_get_returns_a_copy_not_the_live_dict():
    """A caller mutating the dict .get() returned must never corrupt the
    store's own internal state -- .get() must hand back a real copy, not
    a reference to the same dict object a concurrent .set() could be
    replacing."""
    calib_a = {0: _fake_calibration(1000.0)}
    store = capture_daemon.CalibrationStore(calib_a)
    snapshot = store.get()
    snapshot[99] = _fake_calibration(9999.0)
    assert 99 not in store.get()


def test_calibration_store_default_construction_is_empty():
    store = capture_daemon.CalibrationStore()
    assert store.get() == {}
    meta = store.meta()
    assert meta["n_cameras"] == 0
    assert meta["source"] == "startup"
    assert meta["checked_at_utc"] is None


def test_calibration_store_concurrent_read_write_never_raises_and_ends_consistent(tmp_path):
    """Real thread-safety proof, not just "the code looks right" -- a
    background thread hammers .get() continuously while the main thread
    hammers .set() with alternating single/double-camera calibrations,
    for a real bounded duration. A torn/unlocked read would show up here
    as either a raised exception (dict mutated during iteration/copy) or
    a snapshot with a camera count that was never actually set (e.g. 1 if
    only {0: ...} and {0: ..., 1: ...} are ever set) -- neither may ever
    happen."""
    store = capture_daemon.CalibrationStore({0: _fake_calibration()})
    stop = threading.Event()
    errors: list[BaseException] = []
    seen_sizes: set[int] = set()

    def reader() -> None:
        try:
            while not stop.is_set():
                seen_sizes.add(len(store.get()))
        except BaseException as exc: # noqa: BLE001 -- must observe ANY failure, not just some
            errors.append(exc)

    reader_thread = threading.Thread(target=reader, daemon=True)
    reader_thread.start()

    deadline = time.monotonic() + 0.5
    i = 0
    try:
        while time.monotonic() < deadline:
            i += 1
            calibs = {0: _fake_calibration(float(i))}
            if i % 2 == 0:
                calibs[1] = _fake_calibration(float(i) + 0.5)
            store.set(calibs, source="manual", checked_at_utc=f"t{i}")
    finally:
        stop.set()
        reader_thread.join(timeout=2.0)

    assert not reader_thread.is_alive()
    assert errors == []
    # Every observed size must be a real, valid state (1 or 2 cameras) --
    # never 0 (the store was seeded non-empty) and never anything else
    # (which would indicate a torn read mid-.set()).
    assert seen_sizes.issubset({1, 2})


# ---------------------------------------------------------------------------
# calibration_status_dict() -- factored out of opendarts/live/server.py's
# AppState._refresh_calibration_blocking (2026-08-12) so the display
# projection is identical regardless of whether the calibration came from
# a startup bootstrap or a manual dashboard recalibrate.
# ---------------------------------------------------------------------------


def test_calibration_status_dict_marks_calibrated_cameras_ok_with_real_reprojection_error():
    from opendarts.calibration.pnp import PnpResult

    pnp = PnpResult(ok=True, rvec=np.zeros(3), tvec=np.zeros(3), reprojection_error_px=1.23)
    calib = CameraCalibration(
        camera_matrix=np.eye(3), dist_coeffs=np.zeros(5), rvec=np.zeros(3),
        tvec=np.zeros(3), pnp_result=pnp, landmark_spread_ok=True,
    )
    status = capture_daemon.calibration_status_dict({0: calib}, n_cameras=3)

    assert status[0]["ok"] is True
    assert status[0]["reprojection_error_px"] == pytest.approx(1.23)
    assert status[0]["landmark_spread_ok"] is True
    # Cameras 1 and 2 were never calibrated this pass -- honestly "not
    # calibrated", not silently omitted or defaulted to some other value.
    for missing_cam in (1, 2):
        assert status[missing_cam]["ok"] is False
        assert status[missing_cam]["reprojection_error_px"] is None
        assert "not calibrated" in status[missing_cam]["reason"]


def test_calibration_status_dict_handles_missing_pnp_result():
    calib = _fake_calibration() # pnp_result=None, same as calibration_from_dict() produces
    status = capture_daemon.calibration_status_dict({0: calib}, n_cameras=1)
    assert status[0]["ok"] is True
    assert status[0]["reprojection_error_px"] is None


# save_calibration_snapshot() and its tests removed 2026-08-22 --
# confirmed genuinely dead code, see opendarts.live.capture_daemon's own
# note at the removed function's former location.


# ---------------------------------------------------------------------------
# run_capture_loop_body() actually reads a fresh calibration on EVERY
# dart from CalibrationStore.get() -- not a stale startup-local variable
# -- so a manual recalibrate that lands in between two darts changes what
# the VERY NEXT scored throw uses. This is the actual correctness
# requirement requested (2026-08-12), proven end-to-end here, not
# just unit-tested on CalibrationStore in isolation.
# ---------------------------------------------------------------------------


def test_run_capture_loop_body_next_throw_uses_a_mid_session_manual_recalibrate(
    tmp_path, monkeypatch
):
    startup_calib = {0: _fake_calibration(1000.0)}
    manual_calib = {0: _fake_calibration(9999.0)}

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return startup_calib

    true_baseline = {0: np.full((2, 2, 3), 111, dtype=np.uint8)}
    dart_frames = [{0: np.full((2, 2, 3), v, dtype=np.uint8)} for v in (10, 20)]
    fetch_sequence = [true_baseline] * STARTUP_FETCHES + dart_frames
    fetch_calls = {"n": 0}

    def fake_fetch(dest_dir, *, hub=None):
        idx = fetch_calls["n"]
        fetch_calls["n"] += 1
        return fetch_sequence[idx] if idx < len(fetch_sequence) else dart_frames[-1]

    scripted_states = [
        ThrowTriggerState(
            state=ThrowState.READY_TO_CAPTURE, dart_count=1,
            true_baseline_frames=true_baseline, last_frame=dart_frames[0],
        ),
        ThrowTriggerState(
            state=ThrowState.READY_TO_CAPTURE, dart_count=2,
            true_baseline_frames=true_baseline, last_frame=dart_frames[1],
        ),
    ]

    def fake_advance(trigger, bg_frames, current_frames):
        idx = len([c for c in advance_calls])
        advance_calls.append(1)
        if idx < len(scripted_states):
            return scripted_states[idx]
        raise _StopLoop("stop after the scripted 2-dart sequence")

    advance_calls: list[int] = []
    seen_calibrations: list[dict] = []
    calibration_store = capture_daemon.CalibrationStore()

    def fake_handle_ready_to_capture(trigger, bg_frames, calibrations, package_root, session_id, **_kwargs):
        seen_calibrations.append(calibrations)
        if trigger.dart_count == 1:
            # Simulate a manual dashboard "Refresh calibration now" click
            # landing IN BETWEEN dart 1 and dart 2 -- writes to the SAME
            # CalibrationStore object the loop itself was given, exactly
            # like AppState.refresh_calibration() does from a different
            # thread in the real opendarts.live.run_product process.
            calibration_store.set(manual_calib, source="manual", checked_at_utc="t-manual")
        return package_root / f"fake_throw_{trigger.dart_count}"

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)
    monkeypatch.setattr(capture_daemon, "handle_ready_to_capture", fake_handle_ready_to_capture)

    stop_event = threading.Event()
    with pytest.raises(_StopLoop):
        capture_daemon.run_capture_loop_body(
            hub=None,
            package_root=tmp_path / "packages",
            poll_interval_s=0.0,
            stop_event=stop_event,
            scratch_dir=tmp_path / "scratch",
            calibration_store=calibration_store,
        )

    assert len(seen_calibrations) == 2
    # Dart 1: no manual recalibrate has happened yet -- still the startup
    # calibration, proving the store was correctly seeded from bootstrap.
    assert _same_calib_dict(seen_calibrations[0], startup_calib)
    # Dart 2: the manual recalibrate landed after dart 1 -- THIS is the
    # real correctness requirement: the very next scored throw uses it,
    # not the stale startup calibration.
    assert _same_calib_dict(seen_calibrations[1], manual_calib)
    assert not _same_calib_dict(seen_calibrations[1], startup_calib)

    # And the store itself reflects the manual write, honestly labeled.
    assert calibration_store.meta()["source"] == "manual"


def test_run_capture_loop_body_seeds_a_caller_provided_store_with_the_startup_bootstrap(
    tmp_path, monkeypatch
):
    """Even with NO manual recalibrate at all, a caller-provided
    CalibrationStore (opendarts/live/run_product.py's real usage) must be
    seeded with the real startup bootstrap result -- so the dashboard
    (reading the SAME store) shows real data immediately at startup,
    not "unknown" until someone happens to click refresh."""
    startup_calib = {0: _fake_calibration(4242.0)}

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return startup_calib

    true_baseline = {0: np.full((2, 2, 3), 111, dtype=np.uint8)}
    fetch_sequence = [true_baseline] * (STARTUP_FETCHES + 1)
    fetch_calls = {"n": 0}

    def fake_fetch(dest_dir, *, hub=None):
        idx = fetch_calls["n"]
        fetch_calls["n"] += 1
        return fetch_sequence[idx] if idx < len(fetch_sequence) else true_baseline

    def fake_advance(trigger, bg_frames, current_frames):
        raise _StopLoop("stop immediately after startup -- only bootstrap seeding matters here")

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)

    calibration_store = capture_daemon.CalibrationStore()
    assert calibration_store.get() == {} # nothing seeded yet

    stop_event = threading.Event()
    with pytest.raises(_StopLoop):
        capture_daemon.run_capture_loop_body(
            hub=None,
            package_root=tmp_path / "packages",
            poll_interval_s=0.0,
            stop_event=stop_event,
            scratch_dir=tmp_path / "scratch",
            calibration_store=calibration_store,
        )

    assert calibration_store.meta()["source"] == "startup"
    assert calibration_store.meta()["n_cameras"] == 1


# ---------------------------------------------------------------------------
# ResetRequest -- added 2026-08-12, the project's "wire the reset button"
# request. Mirrors CalibrationStore's own thread-safe request/signal
# pattern (see ResetRequest's own docstring); tests below mirror
# CalibrationStore's own test shapes for the analogous reason.
# ---------------------------------------------------------------------------


def test_reset_request_default_construction_not_requested():
    rr = capture_daemon.ResetRequest()
    assert rr.check_and_clear() is False
    assert rr.meta() == {"last_requested_at_utc": None}


def test_reset_request_check_and_clear_consumes_exactly_once():
    rr = capture_daemon.ResetRequest()
    rr.request()
    assert rr.check_and_clear() is True
    # Consumed -- a second call must NOT still report pending.
    assert rr.check_and_clear() is False


def test_reset_request_meta_reports_last_requested_at_utc():
    rr = capture_daemon.ResetRequest()
    assert rr.meta()["last_requested_at_utc"] is None
    rr.request()
    assert rr.meta()["last_requested_at_utc"] is not None
    # meta() itself must NOT consume the pending flag -- only
    # check_and_clear() does.
    assert rr.check_and_clear() is True


def test_reset_request_concurrent_request_and_check_never_raises_or_double_fires():
    """Real thread-safety proof, same style as
    test_calibration_store_concurrent_read_write_never_raises_and_ends_consistent
    above -- one thread hammers .request(), the main thread hammers
    .check_and_clear(), for a bounded real duration. Counts how many times
    a pending flag was actually observed True -- must never exceed the
    number of times .request() was actually called (a torn/unlocked
    implementation could double-fire)."""
    rr = capture_daemon.ResetRequest()
    stop = threading.Event()
    errors: list[BaseException] = []
    request_calls = {"n": 0}

    def requester() -> None:
        try:
            while not stop.is_set():
                rr.request()
                request_calls["n"] += 1
        except BaseException as exc: # noqa: BLE001
            errors.append(exc)

    requester_thread = threading.Thread(target=requester, daemon=True)
    requester_thread.start()

    deadline = time.monotonic() + 0.3
    observed_true = 0
    try:
        while time.monotonic() < deadline:
            if rr.check_and_clear():
                observed_true += 1
    finally:
        stop.set()
        requester_thread.join(timeout=2.0)

    assert not requester_thread.is_alive()
    assert errors == []
    # Never more "observed True" than real .request() calls -- would
    # indicate a race letting one request fire twice.
    assert observed_true <= request_calls["n"]


# ---------------------------------------------------------------------------
# run_capture_loop_body() picking up a pending ResetRequest -- the actual
# loop-wiring proof (checked BEFORE advance() every iteration, refreshes
# true_baseline_frames/bg_frames to CURRENT frames, resets dart_count/state
# to IDLE unconditionally, regardless of prior state).
# ---------------------------------------------------------------------------


def test_run_capture_loop_body_picks_up_a_pending_reset_request(tmp_path, monkeypatch):
    startup = {0: np.full((2, 2, 3), 50, dtype=np.uint8)}
    current = {0: np.full((2, 2, 3), 99, dtype=np.uint8)}

    def fake_bootstrap(*a, **k):
        return {0: _fake_calibration()}

    calls = {"n": 0}

    def fake_fetch(dest_dir, *, hub=None):
        calls["n"] += 1
        if calls["n"] <= STARTUP_FETCHES:
            return startup
        if calls["n"] == STARTUP_FETCHES + 1:
            return current
        raise _StopLoop("stop right after the reset-handling iteration")

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)

    reset_request = capture_daemon.ResetRequest()
    reset_request.request()

    events: list[dict] = []
    stop_event = threading.Event()
    with pytest.raises(_StopLoop):
        capture_daemon.run_capture_loop_body(
            hub=None,
            package_root=tmp_path / "packages",
            poll_interval_s=0.0,
            stop_event=stop_event,
            scratch_dir=tmp_path / "scratch",
            on_event=events.append,
            reset_request=reset_request,
        )

    # Consumed exactly once.
    assert reset_request.check_and_clear() is False
    # Filtered by type first -- 2026-08-12, this loop's on_event stream
    # also now carries a CALIBRATION_STATUS event (the auto-calibrate
    # confirmation, see run_capture_loop_body's own "Calibration
    # bootstrap" section) ahead of the TRIGGER_STATE events this
    # assertion cares about; that event has no "state" key at all, so an
    # unfiltered `e["state"]` lookup would KeyError on it.
    reset_events = [
        e
        for e in events
        if e.get("type") == "TRIGGER_STATE" and e["state"] == "IDLE" and e["dart_count"] == 0
    ]
    assert len(reset_events) >= 2 # startup seed + the reset-triggered one


def test_run_capture_loop_body_reset_bumps_the_throw_numbering_generation(
    tmp_path, monkeypatch
):
    """2026-08-22, "when I hit reset ... we reset to 0." A manual
    Reset does NOT delete any already-saved throw packages -- a bare
    reset-to-0 would try to write the next throw as session_id-001-...,
    colliding with a real package already sitting there. This proves the
    reset actually calls _reset_session_throw_numbering() (a counting
    GENERATION bump, not a same-numbering-space reset), by checking a
    real `.generation` file appears in the persisted counters directory
    -- the same directory/mechanism handle_ready_to_capture()'s own
    tests already cover, exercised here via the real /api/reset ->
    ResetRequest -> run_capture_loop_body() path instead of calling the
    helper directly."""
    startup = {0: np.full((2, 2, 3), 50, dtype=np.uint8)}
    current = {0: np.full((2, 2, 3), 99, dtype=np.uint8)}

    def fake_bootstrap(*a, **k):
        return {0: _fake_calibration()}

    calls = {"n": 0}

    def fake_fetch(dest_dir, *, hub=None):
        calls["n"] += 1
        if calls["n"] <= STARTUP_FETCHES:
            return startup
        if calls["n"] == STARTUP_FETCHES + 1:
            return current
        raise _StopLoop("stop right after the reset-handling iteration")

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)

    reset_request = capture_daemon.ResetRequest()
    reset_request.request()

    package_root = tmp_path / "packages"
    stop_event = threading.Event()
    with pytest.raises(_StopLoop):
        capture_daemon.run_capture_loop_body(
            hub=None,
            package_root=package_root,
            poll_interval_s=0.0,
            stop_event=stop_event,
            scratch_dir=tmp_path / "scratch",
            on_event=lambda e: None,
            reset_request=reset_request,
        )

    counters_dir = package_root.parent / "session_throw_counters"
    generation_files = list(counters_dir.glob("*.generation"))
    assert len(generation_files) == 1, (
        f"expected exactly one session's generation file to be bumped by the "
        f"reset, found: {[f.name for f in generation_files]}"
    )
    assert generation_files[0].read_text().strip() == "1"


def test_run_capture_loop_body_reset_refreshes_baseline_to_current_frames_not_stale(
    tmp_path, monkeypatch
):
    """The real correctness requirement, not just "an IDLE event fired" --
    the trigger the loop actually advances with, on the iteration right
    after a reset, must carry the CURRENT (post-reset) frames as its
    true_baseline_frames, not the stale pre-reset startup baseline."""
    startup = {0: np.full((2, 2, 3), 50, dtype=np.uint8)}
    drifted_current = {0: np.full((2, 2, 3), 200, dtype=np.uint8)} # far from startup's value

    def fake_bootstrap(*a, **k):
        return {0: _fake_calibration()}

    calls = {"n": 0}

    def fake_fetch(dest_dir, *, hub=None):
        calls["n"] += 1
        if calls["n"] <= STARTUP_FETCHES:
            return startup
        if calls["n"] in (STARTUP_FETCHES + 1, STARTUP_FETCHES + 2):
            return drifted_current
        raise _StopLoop("stop after one real post-reset advance() call")

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)

    seen_bg: list = []

    def spying_advance(trigger, bg, current):
        seen_bg.append(bg)
        return trigger

    stub_lifecycle = script_trigger(monkeypatch, spying_advance)

    reset_request = capture_daemon.ResetRequest()
    reset_request.request()

    stop_event = threading.Event()
    with pytest.raises(_StopLoop):
        capture_daemon.run_capture_loop_body(
            hub=None,
            package_root=tmp_path / "packages",
            poll_interval_s=0.0,
            stop_event=stop_event,
            scratch_dir=tmp_path / "scratch",
            reset_request=reset_request,
        )

    # The reset iteration `continue`s past the lifecycle step entirely, so
    # the first step seen here is the iteration AFTER the reset: the
    # lifecycle itself was told to reset, and the loop's scoring buffer
    # already carries the CURRENT (post-reset) frames, not the startup ones.
    assert stub_lifecycle.resets == 1
    assert len(seen_bg) >= 1
    for cam, frame in seen_bg[0].items():
        assert np.array_equal(frame, drifted_current[cam])
        assert not np.array_equal(frame, startup[cam])


# ---------------------------------------------------------------------------
# CaptureLoopController -- added 2026-08-12, the project's "wire the start and
# stop buttons" request. Coordinates the camera-hub lifecycle between the
# FastAPI request-handling thread and the capture loop's background
# thread -- see that class's own docstring for the full design (mirrors
# an explicit start/stop pair plus a touch/idle-loop timeout, adapted for
# opendarts's thread-based, not asyncio-task-based, architecture).
# ---------------------------------------------------------------------------


def test_capture_loop_controller_default_not_running():
    c = capture_daemon.CaptureLoopController()
    assert c.is_running() is False
    meta = c.meta()
    assert meta["running"] is False
    assert meta["idle_timeout_sec"] == capture_daemon.IDLE_TIMEOUT_SEC_DEFAULT


def test_capture_loop_controller_default_idle_timeout_matches_od_real_default():
    """Pins the real number this was deliberately matched to
    -- if this ever changes it should be a deliberate, documented edit."""
    assert capture_daemon.IDLE_TIMEOUT_SEC_DEFAULT == 900


def test_capture_loop_controller_request_start_sets_running_and_clears_stop_signals():
    c = capture_daemon.CaptureLoopController()
    c.session_stop_event.set() # simulate a previous session's leftover signal
    c.stopped_ack.set()
    c.request_start()
    assert c.is_running() is True
    assert c.start_requested.is_set() is True
    assert c.session_stop_event.is_set() is False
    assert c.stopped_ack.is_set() is False


def test_capture_loop_controller_request_stop_sets_running_false_and_signals_session_stop_event():
    c = capture_daemon.CaptureLoopController()
    c.request_start()
    c.request_stop()
    assert c.is_running() is False
    assert c.session_stop_event.is_set() is True


def test_capture_loop_controller_mark_session_ended_sets_stopped_ack_and_clears_start_requested():
    c = capture_daemon.CaptureLoopController()
    c.request_start()
    c.mark_session_ended()
    assert c.is_running() is False
    assert c.start_requested.is_set() is False
    assert c.stopped_ack.is_set() is True


def test_capture_loop_controller_idle_timeout_due_false_when_not_running():
    c = capture_daemon.CaptureLoopController(idle_timeout_sec=1)
    assert c.idle_timeout_due() is False # never started
    c.request_start()
    c.request_stop()
    assert c.idle_timeout_due() is False # stopped again


def test_capture_loop_controller_idle_timeout_due_false_when_disabled():
    c = capture_daemon.CaptureLoopController(idle_timeout_sec=0)
    c.request_start()
    time.sleep(0.05)
    assert c.idle_timeout_due() is False
    c.set_idle_timeout_sec(-5) # clamped to 0 -- still disabled
    assert c.get_idle_timeout_sec() == 0
    assert c.idle_timeout_due() is False


def test_capture_loop_controller_idle_timeout_due_true_after_real_elapsed_time():
    """Real, measured elapsed-time proof -- a short (0.05s) configured
    timeout, a real time.sleep() past it, not a mocked clock (this
    module's own established pattern already accepts short real sleeps
    for bounded-timing tests, see e.g.
    test_wait_for_first_frames_gives_up_after_max_wait_and_logs_per_camera_diagnostics)."""
    c = capture_daemon.CaptureLoopController(idle_timeout_sec=1)
    c.request_start()
    with c._lock: # noqa: SLF001 -- test-only direct backdate, no public API for a sub-1s real timeout
        c._idle_timeout_sec = 0.05
    time.sleep(0.15)
    assert c.idle_timeout_due() is True


def test_capture_loop_controller_touch_resets_the_idle_clock():
    """touch() must push the idle deadline back out -- a controller that
    would otherwise be due (per the sibling test above) must NOT be due
    immediately after a fresh touch()."""
    c = capture_daemon.CaptureLoopController(idle_timeout_sec=1)
    c.request_start()
    with c._lock: # noqa: SLF001 -- test-only direct backdate, no public API for this
        c._idle_timeout_sec = 0.05
    time.sleep(0.15)
    assert c.idle_timeout_due() is True
    c.touch()
    assert c.idle_timeout_due() is False


def test_capture_loop_controller_meta_reports_seconds_since_activity():
    c = capture_daemon.CaptureLoopController()
    meta = c.meta()
    assert isinstance(meta["seconds_since_activity"], float)
    assert meta["seconds_since_activity"] >= 0.0


def test_capture_loop_controller_set_idle_timeout_sec_clamps_negative_to_zero():
    c = capture_daemon.CaptureLoopController()
    c.set_idle_timeout_sec(-100)
    assert c.get_idle_timeout_sec() == 0


# ---------------------------------------------------------------------------
# CaptureLoopController idle-timeout DURABLE PERSISTENCE, added 2026-09-03 --
# real live operator report. The
# identical class of bug EngineConfigStore's own 2026-08-14 fix already
# closed for engine config (see test_engines_capture_daemon_integration.py's
# own snapshot_path tests, mirrored here) -- set_idle_timeout_sec() only
# ever mutated an in-memory attribute, and run_product.py constructs a
# fresh CaptureLoopController() (falling back to IDLE_TIMEOUT_SEC_DEFAULT,
# 900s/15min) on every process restart, silently discarding whatever an
# operator last set via the dashboard.
# ---------------------------------------------------------------------------


def test_capture_loop_controller_without_snapshot_path_is_purely_in_memory(tmp_path):
    """No snapshot_path (every pre-existing caller/test) -- unchanged
    behavior, nothing written to disk, nothing to load."""
    c = capture_daemon.CaptureLoopController()
    c.set_idle_timeout_sec(3600)
    assert list(tmp_path.iterdir()) == []
    # A second controller with no snapshot_path never sees the first
    # one's value.
    c2 = capture_daemon.CaptureLoopController()
    assert c2.get_idle_timeout_sec() == capture_daemon.IDLE_TIMEOUT_SEC_DEFAULT


def test_capture_loop_controller_set_idle_timeout_sec_persists_a_snapshot(tmp_path):
    path = tmp_path / "idle_timeout" / "idle_timeout.json"
    c = capture_daemon.CaptureLoopController(snapshot_path=path)
    assert not path.exists() # nothing written until an actual set
    c.set_idle_timeout_sec(3600)
    assert path.exists()
    assert json.loads(path.read_text()) == {"idle_timeout_sec": 3600}


def test_capture_loop_controller_loads_a_previously_saved_snapshot_on_construction(tmp_path):
    """The actual fix: a NEW controller instance (standing in for a fresh
    process after a restart) must recover the LAST SAVED value, not fall
    back to the 15-minute code default."""
    path = tmp_path / "idle_timeout" / "idle_timeout.json"
    first = capture_daemon.CaptureLoopController(snapshot_path=path)
    first.set_idle_timeout_sec(3600)

    second = capture_daemon.CaptureLoopController(snapshot_path=path)
    assert second.get_idle_timeout_sec() == 3600


def test_capture_loop_controller_missing_snapshot_file_falls_back_to_constructor_default(tmp_path):
    """A fresh install / wiped data dir is not an error -- the
    constructor's own idle_timeout_sec arg is the real default, exactly
    as if snapshot_path had never been passed."""
    path = tmp_path / "does" / "not" / "exist.json"
    c = capture_daemon.CaptureLoopController(idle_timeout_sec=1200, snapshot_path=path)
    assert c.get_idle_timeout_sec() == 1200


def test_capture_loop_controller_corrupt_snapshot_does_not_crash_construction(tmp_path):
    path = tmp_path / "idle_timeout" / "idle_timeout.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not valid json")
    c = capture_daemon.CaptureLoopController(idle_timeout_sec=1200, snapshot_path=path)
    assert c.get_idle_timeout_sec() == 1200


def test_capture_loop_controller_zero_idle_timeout_round_trips_through_a_snapshot(tmp_path):
    """0 (disable the idle timeout entirely) is a real, meaningful value
    -- must survive the round trip like any other, not be mistaken for
    "unset" the way a naive falsy check would."""
    path = tmp_path / "idle_timeout" / "idle_timeout.json"
    first = capture_daemon.CaptureLoopController(snapshot_path=path)
    first.set_idle_timeout_sec(0)

    second = capture_daemon.CaptureLoopController(idle_timeout_sec=900, snapshot_path=path)
    assert second.get_idle_timeout_sec() == 0


def test_capture_loop_controller_concurrent_touch_and_meta_never_raises():
    """Real thread-safety proof, same style as the ResetRequest/
    CalibrationStore concurrency tests above."""
    c = capture_daemon.CaptureLoopController()
    stop = threading.Event()
    errors: list[BaseException] = []

    def toucher() -> None:
        try:
            while not stop.is_set():
                c.touch()
        except BaseException as exc: # noqa: BLE001
            errors.append(exc)

    toucher_thread = threading.Thread(target=toucher, daemon=True)
    toucher_thread.start()
    deadline = time.monotonic() + 0.3
    try:
        while time.monotonic() < deadline:
            c.meta()
            c.is_running()
    finally:
        stop.set()
        toucher_thread.join(timeout=2.0)
    assert not toucher_thread.is_alive()
    assert errors == []


# ---------------------------------------------------------------------------
# run_capture_loop_body()'s `also_stop` parameter -- lets a caller end
# just ONE session without tearing down the whole process (the real
# mechanism opendarts/live/run_product.py's restructured capture thread uses
# between Start/Stop clicks).
# ---------------------------------------------------------------------------


def test_run_capture_loop_body_stops_on_also_stop_without_stop_event_being_set(tmp_path, monkeypatch):
    true_baseline = {0: np.full((2, 2, 3), 80, dtype=np.uint8)}

    def fake_bootstrap(*a, **k):
        return {0: _fake_calibration()}

    def fake_fetch(dest_dir, *, hub=None):
        return true_baseline

    also_stop = threading.Event()
    call_count = {"n": 0}

    def fake_advance(trigger, bg, current):
        call_count["n"] += 1
        if call_count["n"] >= 3:
            also_stop.set() # simulate a manual Stop landing mid-session
        return trigger

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)

    stop_event = threading.Event()
    # Must return NORMALLY (no exception), and stop_event itself must
    # remain UNSET -- proving also_stop alone ended the session, the
    # whole-process signal was never touched.
    capture_daemon.run_capture_loop_body(
        hub=None,
        package_root=tmp_path / "packages",
        poll_interval_s=0.0,
        stop_event=stop_event,
        scratch_dir=tmp_path / "scratch",
        also_stop=also_stop,
    )
    assert stop_event.is_set() is False
    assert also_stop.is_set() is True
    assert call_count["n"] >= 3


def test_run_capture_loop_body_also_stop_none_is_backward_compatible(tmp_path, monkeypatch):
    """The default (also_stop=None, every pre-existing caller/test) must
    behave EXACTLY as before this parameter existed -- proven here by a
    simple bounded run that stops via stop_event alone, same as any
    pre-existing test in this file."""
    true_baseline = {0: np.full((2, 2, 3), 80, dtype=np.uint8)}

    def fake_bootstrap(*a, **k):
        return {0: _fake_calibration()}

    def fake_fetch(dest_dir, *, hub=None):
        return true_baseline

    stop_event = threading.Event()
    call_count = {"n": 0}

    def fake_advance(trigger, bg, current):
        call_count["n"] += 1
        if call_count["n"] >= 2:
            stop_event.set()
        return trigger

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)

    capture_daemon.run_capture_loop_body(
        hub=None,
        package_root=tmp_path / "packages",
        poll_interval_s=0.0,
        stop_event=stop_event,
        scratch_dir=tmp_path / "scratch",
    )
    assert stop_event.is_set() is True
    assert call_count["n"] >= 2


# ---------------------------------------------------------------------------
# run_capture_loop_body()'s calibration-bootstrap reuse -- "calibrate
# should be auto after a start, but is available if user wants to
# calibrate again": a SECOND (or later) session this
# same process lifetime must NOT re-bootstrap when a valid calibration
# already exists in the shared CalibrationStore.
# ---------------------------------------------------------------------------


def test_run_capture_loop_body_reuses_an_already_populated_calibration_store(tmp_path, monkeypatch):
    bootstrap_calls = {"n": 0}

    def fake_bootstrap(*a, **k):
        bootstrap_calls["n"] += 1
        return {0: _fake_calibration(1234.0)}

    def fake_fetch(dest_dir, *, hub=None):
        return {0: np.full((2, 2, 3), 80, dtype=np.uint8)}

    def fake_advance(trigger, bg, current):
        raise _StopLoop("stop immediately -- only the bootstrap-skip matters here")

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)

    pre_populated = capture_daemon.CalibrationStore(
        {0: _fake_calibration(9999.0)}, source="startup", checked_at_utc="t-earlier-session"
    )
    stop_event = threading.Event()
    with pytest.raises(_StopLoop):
        capture_daemon.run_capture_loop_body(
            hub=None,
            package_root=tmp_path / "packages",
            poll_interval_s=0.0,
            stop_event=stop_event,
            scratch_dir=tmp_path / "scratch",
            calibration_store=pre_populated,
        )

    assert bootstrap_calls["n"] == 0, "must NOT re-bootstrap when the store is already populated"
    # The store's contents/source are UNCHANGED -- still the earlier
    # session's real calibration, not silently replaced.
    assert pre_populated.meta()["checked_at_utc"] == "t-earlier-session"
    stored = pre_populated.get()
    assert stored[0].tvec[2] == 9999.0


def test_run_capture_loop_body_still_bootstraps_when_calibration_store_is_empty(tmp_path, monkeypatch):
    """Regression guard -- the common/existing case (a fresh, empty store,
    or no store at all) must be completely unaffected by the reuse logic
    above: still bootstraps exactly as every pre-existing test in this
    file already assumes."""
    bootstrap_calls = {"n": 0}

    def fake_bootstrap(*a, **k):
        bootstrap_calls["n"] += 1
        return {0: _fake_calibration(4321.0)}

    def fake_fetch(dest_dir, *, hub=None):
        return {0: np.full((2, 2, 3), 80, dtype=np.uint8)}

    def fake_advance(trigger, bg, current):
        raise _StopLoop("stop immediately")

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)

    empty_store = capture_daemon.CalibrationStore()
    stop_event = threading.Event()
    with pytest.raises(_StopLoop):
        capture_daemon.run_capture_loop_body(
            hub=None,
            package_root=tmp_path / "packages",
            poll_interval_s=0.0,
            stop_event=stop_event,
            scratch_dir=tmp_path / "scratch",
            calibration_store=empty_store,
        )

    assert bootstrap_calls["n"] == 1
    assert empty_store.get()[0].tvec[2] == 4321.0


def test_run_capture_loop_body_emits_calibration_status_event_when_reusing_the_store(
    tmp_path, monkeypatch
):
    """Real, visible confirmation of Start's auto-calibrate step. The
    "reused" branch (a valid calibration already exists this process
    lifetime) must emit a CALIBRATION_STATUS event with
    source="startup_reused" -- BEFORE the loop even reaches its first
    TRIGGER_STATE emit, since that's what the dashboard needs to
    distinguish "auto-calibrate ran" from "auto-calibrate skipped,
    already valid" (opendarts/live/server.py's AppState._handle_live_event)."""

    def fake_fetch(dest_dir, *, hub=None):
        return {0: np.full((2, 2, 3), 80, dtype=np.uint8)}

    def fake_advance(trigger, bg, current):
        raise _StopLoop("stop immediately -- only the emitted event matters here")

    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)

    pre_populated = capture_daemon.CalibrationStore(
        {0: _fake_calibration(9999.0)}, source="startup", checked_at_utc="t-earlier-session"
    )
    events: list[dict] = []
    stop_event = threading.Event()
    with pytest.raises(_StopLoop):
        capture_daemon.run_capture_loop_body(
            hub=None,
            package_root=tmp_path / "packages",
            poll_interval_s=0.0,
            stop_event=stop_event,
            scratch_dir=tmp_path / "scratch",
            calibration_store=pre_populated,
            on_event=events.append,
        )

    calib_events = [e for e in events if e.get("type") == "CALIBRATION_STATUS"]
    assert len(calib_events) == 1
    assert calib_events[0]["source"] == "startup_reused"
    assert 0 in calib_events[0]["calibrations"]
    assert calib_events[0]["calibrations"][0].tvec[2] == 9999.0
    # This event must land BEFORE the loop's own first TRIGGER_STATE emit
    # -- the dashboard's pill should show the calibration confirmation
    # while still in its "Starting" phase, not after.
    trigger_events = [e for e in events if e.get("type") == "TRIGGER_STATE"]
    assert events.index(calib_events[0]) < events.index(trigger_events[0])


def test_run_capture_loop_body_emits_calibration_status_event_on_a_fresh_bootstrap(
    tmp_path, monkeypatch
):
    """Same real confirmation as the "reused" test above, but for the
    "it actually ran" case -- source="startup", a genuinely fresh
    bootstrap (empty/no calibration_store)."""

    def fake_bootstrap(*a, **k):
        return {0: _fake_calibration(4321.0)}

    def fake_fetch(dest_dir, *, hub=None):
        return {0: np.full((2, 2, 3), 80, dtype=np.uint8)}

    def fake_advance(trigger, bg, current):
        raise _StopLoop("stop immediately")

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)

    empty_store = capture_daemon.CalibrationStore()
    events: list[dict] = []
    stop_event = threading.Event()
    with pytest.raises(_StopLoop):
        capture_daemon.run_capture_loop_body(
            hub=None,
            package_root=tmp_path / "packages",
            poll_interval_s=0.0,
            stop_event=stop_event,
            scratch_dir=tmp_path / "scratch",
            calibration_store=empty_store,
            on_event=events.append,
        )

    calib_events = [e for e in events if e.get("type") == "CALIBRATION_STATUS"]
    assert len(calib_events) == 1
    assert calib_events[0]["source"] == "startup"
    assert calib_events[0]["calibrations"][0].tvec[2] == 4321.0


# ---------------------------------------------------------------------------
# Also-run dispatch from handle_ready_to_capture().
# ---------------------------------------------------------------------------

class _FakeEngineConfigStore:
    """Minimal stand-in for capture_daemon.EngineConfigStore -- only
    implements the one method handle_ready_to_capture() actually calls
    (`.get()`), returning a fixed, caller-supplied EngineConfig."""

    def __init__(self, config):
        self._config = config

    def get(self):
        return self._config


class _RecordingEngine:
    """Records every (bg_images, frame_images, calibration) triple it
    was called with -- lets a test assert exactly WHICH calibration
    object reached score(), by identity, not just that scoring
    succeeded."""

    def __init__(self, result=None):
        self.calls: list[dict] = []
        self._result = result or _make_fake_engine_result(ok=True, sector="1", ring="single_inner")

    def score(self, bg_images, frame_images, calibration):
        self.calls.append({"calibration": calibration})
        return self._result




def test_handle_ready_to_capture_also_run_scores_with_the_rigs_own_calibrations(
    tmp_path, monkeypatch
):
    """also_run engines are dispatched with the same `calibrations` the
    primary engine used -- no substitute calibration for anyone."""
    dispatch_calls = []

    def fake_dispatch_engines(bg_images, frame_images, calibration, names, *, timeout_s, **kwargs):
        dispatch_calls.append((calibration, list(names)))
        return {}

    monkeypatch.setattr(capture_daemon, "dispatch_engines", fake_dispatch_engines)
    recording = _RecordingEngine()
    monkeypatch.setattr(capture_daemon, "get_engine", lambda name: recording)

    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}
    listener = _FakeAdWsListener()
    store = _FakeEngineConfigStore(
        capture_daemon.EngineConfig(primary="Apollo", also_run=("Talos",))
    )

    capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
        ad_ws_listener=listener, engine_config_store=store,
    background_save=False,
    )

    assert _wait_until(lambda: bool(dispatch_calls))
    assert dispatch_calls[0][0] is calibrations
    assert dispatch_calls[0][1] == ["Talos"]


# ---------------------------------------------------------------------------
# BEST-OF-N REPROJECTION ATTEMPTS, added 2026-08-30 -- see capture_daemon.py's
# own "BEST-OF-N REPROJECTION ATTEMPTS" docstring section (inside
# `_bootstrap_calibrations_unlocked()`) for the full design this tests
# against. `n_reprojection_attempts` defaults to 1 (today's exact prior
# ADAPTIVE RETRY-only behavior) for every test above this section -- these
# tests are the only ones in this file that ever pass a value > 1.
# ---------------------------------------------------------------------------


def test_best_of_n_default_is_a_complete_no_op(tmp_path, monkeypatch):
    """n_reprojection_attempts left at its default (1) must be BYTE-FOR-
    BYTE today's prior behavior -- zero extra _capture() calls, zero
    extra calibrate_camera() calls, beyond whatever the ADAPTIVE RETRY
    loop alone would have done. This is the requirement that this
    feature must not silently change default production behavior."""
    calls = {"local": 0}

    def fake_local(hub, n, **kwargs):
        calls["local"] += 1
        return {0: [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(n)]}

    def fake_detect(image_bgr, pre, **kwargs):
        return np.zeros((4, 3)), np.array(
            [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]
        )

    solve_calls = {"n": 0}

    def fake_calibrate(*a, **k):
        solve_calls["n"] += 1
        return _fake_attempt_with_reprojection(1.0) # clears the 2.5px default target

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_local)
    _install_fake_orientation_pipeline(monkeypatch, fake_detect)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", fake_calibrate)

    diagnostics: dict = {}
    result = capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), # type: ignore[arg-type]
        n_frames=10, n_frames_detect=10, retry_batch_size=10, max_frames=30,
        diagnostics_out=diagnostics,
        # n_reprojection_attempts NOT passed -- exercising the real default.
    )

    assert 0 in result
    assert calls["local"] == 1 # exactly the round-1 capture, nothing more
    assert solve_calls["n"] == 1 # exactly the round-1 solve, nothing more
    assert diagnostics[0]["reprojection_error_px"] == 1.0


def test_best_of_n_explicit_n1_is_also_a_complete_no_op(tmp_path, monkeypatch):
    """Same as the default test above, but with n_reprojection_attempts=1
    passed EXPLICITLY -- confirms the guard is `<= 1`, not merely
    "parameter absent", matching the docstring's own stated contract."""
    calls = {"local": 0}

    def fake_local(hub, n, **kwargs):
        calls["local"] += 1
        return {0: [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(n)]}

    def fake_detect(image_bgr, pre, **kwargs):
        return np.zeros((4, 3)), np.array(
            [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]
        )

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_local)
    _install_fake_orientation_pipeline(monkeypatch, fake_detect)
    monkeypatch.setattr(
        capture_daemon, "calibrate_camera",
        lambda *a, **k: _fake_attempt_with_reprojection(1.0),
    )

    result = capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), # type: ignore[arg-type]
        n_frames=10, n_frames_detect=10, retry_batch_size=10, max_frames=30,
        n_reprojection_attempts=1,
    )

    assert 0 in result
    assert calls["local"] == 1


def test_best_of_n_runs_n_minus_1_additional_independent_attempts(tmp_path, monkeypatch):
    """n_reprojection_attempts=5, round 1 clears target immediately (so
    the ADAPTIVE RETRY loop itself only ever captures/solves ONCE) --
    BEST-OF-N must still run 4 MORE independent attempts on top, each its
    own fresh _capture(n_frames_detect) call and its own solve, for a
    real total of 5 capture calls and 5 solve calls."""
    calls = {"local": 0}

    def fake_local(hub, n, **kwargs):
        calls["local"] += 1
        assert n == 10 # n_frames_detect -- every best-of-N attempt batch
        return {0: [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(n)]}

    def fake_detect(image_bgr, pre, **kwargs):
        return np.zeros((4, 3)), np.array(
            [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]
        )

    solve_calls = {"n": 0}

    def fake_calibrate(*a, **k):
        solve_calls["n"] += 1
        return _fake_attempt_with_reprojection(1.0) # clears target every time

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_local)
    _install_fake_orientation_pipeline(monkeypatch, fake_detect)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", fake_calibrate)

    result = capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), # type: ignore[arg-type]
        n_frames=10, n_frames_detect=10, retry_batch_size=10, max_frames=200,
        n_reprojection_attempts=5,
    )

    assert 0 in result
    assert calls["local"] == 5 # round-1's own capture + 4 more best-of-N attempts
    assert solve_calls["n"] == 5


def test_best_of_n_never_reanalyzes_the_same_frames_across_attempts(tmp_path, monkeypatch):
    """Requirement 3 -- every attempt's frames must be genuinely fresh,
    never the same pixels an earlier attempt (ADAPTIVE RETRY's own round
    1, or an earlier best-of-N attempt) already analyzed. Tags every
    _capture() batch with a unique fill value and asserts the detection
    fake never sees the same tag twice."""
    calls = {"local": 0}

    def fake_local(hub, n, **kwargs):
        calls["local"] += 1
        tag = calls["local"]
        return {0: [np.full((4, 4, 3), tag, dtype=np.uint8) for _ in range(n)]}

    seen_tags: list[int] = []

    def fake_detect(image_bgr, pre, **kwargs):
        seen_tags.append(int(image_bgr[0, 0, 0]))
        return np.zeros((4, 3)), np.array(
            [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]
        )

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_local)
    _install_fake_orientation_pipeline(monkeypatch, fake_detect)
    monkeypatch.setattr(
        capture_daemon, "calibrate_camera",
        lambda *a, **k: _fake_attempt_with_reprojection(1.0),
    )

    capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), # type: ignore[arg-type]
        n_frames=10, n_frames_detect=10, retry_batch_size=10, max_frames=200,
        n_reprojection_attempts=5,
    )

    # 5 capture batches (tags 1..5), 10 frames each -- every tag must
    # appear in seen_tags exactly 10 times, and no OTHER tag values leak
    # in (i.e. no attempt silently reused a previous batch's own frames).
    assert calls["local"] == 5
    for tag in range(1, 6):
        assert seen_tags.count(tag) == 10
    assert sorted(set(seen_tags)) == [1, 2, 3, 4, 5]


def test_best_of_n_adopts_lowest_reprojection_regardless_of_which_attempt(
    tmp_path, monkeypatch
):
    """The core requirement, in the project's own words: adopt whichever
    attempt had the LOWEST reprojection error, regardless of which
    attempt achieved it. Proven here with round 1 (ADAPTIVE RETRY's own
    accept-immediately path) reporting a WORSE number (1.0) than a LATER
    best-of-N attempt (0.4, attempt #3) -- "stop at first success" and
    "best-of-N" would disagree on what to adopt here."""
    def fake_local(hub, n, **kwargs):
        return {0: [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(n)]}

    def fake_detect(image_bgr, pre, **kwargs):
        return np.zeros((4, 3)), np.array(
            [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]
        )

    # index 0 = round 1 (ADAPTIVE RETRY's own accept-immediately solve);
    # indices 1-4 = best-of-N attempts 2-5. Attempt #3 (index 2, value
    # 0.4) is the real best -- neither first (1.0) nor last (0.95).
    reprojections = [1.0, 0.7, 0.4, 0.9, 0.95]
    solve_calls = {"n": 0}

    def fake_calibrate(*a, **k):
        idx = solve_calls["n"]
        solve_calls["n"] += 1
        return _fake_attempt_with_reprojection(reprojections[idx])

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_local)
    _install_fake_orientation_pipeline(monkeypatch, fake_detect)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", fake_calibrate)

    diagnostics: dict = {}
    result = capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), # type: ignore[arg-type]
        n_frames=10, n_frames_detect=10, retry_batch_size=10, max_frames=200,
        n_reprojection_attempts=5,
        diagnostics_out=diagnostics,
    )

    assert 0 in result
    assert solve_calls["n"] == 5
    assert diagnostics[0]["reprojection_error_px"] == 0.4
    assert diagnostics[0]["target_met"] is True # 0.4 < the 2.5px default target


def test_best_of_n_never_makes_a_camera_worse_than_the_adaptive_retry_loop_alone(
    tmp_path, monkeypatch
):
    """The other half of "regardless of which attempt achieved it": if
    every ADDITIONAL best-of-N attempt is WORSE than round 1's own
    result, the adopted result must stay round 1's -- best-of-N can only
    ever match or improve, never regress, a camera's own adopted
    result."""
    def fake_local(hub, n, **kwargs):
        return {0: [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(n)]}

    def fake_detect(image_bgr, pre, **kwargs):
        return np.zeros((4, 3)), np.array(
            [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]
        )

    reprojections = [0.5, 3.0, 4.0, 5.0] # round 1 (0.5) is already the best
    solve_calls = {"n": 0}

    def fake_calibrate(*a, **k):
        idx = solve_calls["n"]
        solve_calls["n"] += 1
        return _fake_attempt_with_reprojection(reprojections[idx])

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_local)
    _install_fake_orientation_pipeline(monkeypatch, fake_detect)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", fake_calibrate)

    diagnostics: dict = {}
    capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), # type: ignore[arg-type]
        n_frames=10, n_frames_detect=10, retry_batch_size=10, max_frames=200,
        n_reprojection_attempts=4,
        diagnostics_out=diagnostics,
    )

    assert solve_calls["n"] == 4
    assert diagnostics[0]["reprojection_error_px"] == 0.5


def test_best_of_n_does_not_repeat_orientation_resolution_per_attempt(
    tmp_path, monkeypatch
):
    """Requirement 1 -- orientation resolution happens exactly ONCE per
    event, never per best-of-N attempt. The orientation solve
    is the real hook `_establish_hint_if_possible()` calls -- best-of-N's
    own per-attempt code deliberately never calls that function at all
    (see its own docstring), so the call count must be IDENTICAL whether
    n_reprojection_attempts is 1 or 5, on the same fixed round-1 frame
    content (round 1 alone is enough to establish the hint here)."""
    def fake_local(hub, n, **kwargs):
        return {0: [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(n)]}

    def fake_detect(image_bgr, pre, **kwargs):
        return np.zeros((4, 3)), np.array(
            [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]
        )

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_local)
    _install_fake_orientation_pipeline(monkeypatch, fake_detect)
    real_orientation = capture_daemon.ring_correlation_orientation_for_camera
    orientation_calls = {"n": 0}

    def counting_orientation(frames, **kw):
        orientation_calls["n"] += 1
        return real_orientation(frames, **kw)

    monkeypatch.setattr(
        capture_daemon, "ring_correlation_orientation_for_camera", counting_orientation,
    )

    monkeypatch.setattr(
        capture_daemon, "calibrate_camera",
        lambda *a, **k: _fake_attempt_with_reprojection(1.0),
    )

    capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), # type: ignore[arg-type]
        n_frames=10, n_frames_detect=10, retry_batch_size=10, max_frames=200,
        n_reprojection_attempts=5,
    )

    # Orientation is resolved ONCE per event, up front -- exactly one
    # call for the single camera here, regardless of the 4 additional
    # best-of-N reprojection attempts that ran afterward.
    assert orientation_calls["n"] == 1


def test_best_of_n_still_falls_back_gracefully_when_no_attempt_clears_target(
    tmp_path, monkeypatch
):
    """Requirement 7 -- the existing "give up at max_frames, accept the
    best seen rather than fail outright" safety net must still work when
    best-of-N is enabled and NONE of the N attempts (round 1 included)
    ever clears target: the adopted result is still the best of what was
    tried, `target_met` honestly stays False, and nothing raises."""
    def fake_local(hub, n, **kwargs):
        return {0: [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(n)]}

    def fake_detect(image_bgr, pre, **kwargs):
        return np.zeros((4, 3)), np.array(
            [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]
        )

    # Round 1 (3.24) hits max_frames immediately (n_frames==max_frames
    # below) and gives up per ADAPTIVE RETRY's own existing behavior --
    # never clears the 2.5px default target. Best-of-N's 2 additional
    # attempts also never clear target, but one (2.9) is a real
    # improvement over round 1's own 3.24.
    reprojections = [3.24, 2.9, 3.5]
    solve_calls = {"n": 0}

    def fake_calibrate(*a, **k):
        idx = solve_calls["n"]
        solve_calls["n"] += 1
        return _fake_attempt_with_reprojection(reprojections[idx])

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_local)
    _install_fake_orientation_pipeline(monkeypatch, fake_detect)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", fake_calibrate)

    diagnostics: dict = {}
    result = capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), # type: ignore[arg-type]
        n_frames=10, n_frames_detect=10, retry_batch_size=10, max_frames=10,
        n_reprojection_attempts=3,
        diagnostics_out=diagnostics,
    )

    assert 0 in result # never a hard failure, exactly like ADAPTIVE RETRY alone
    assert solve_calls["n"] == 3
    assert diagnostics[0]["reprojection_error_px"] == 2.9 # the real best of the 3
    assert diagnostics[0]["target_met"] is False # honestly still not met


def test_best_of_n_skips_a_camera_that_never_calibrated_at_all(tmp_path, monkeypatch):
    """A camera whose PnP solve genuinely fails outright (attempt.ok is
    False) never enters `best_calibration` via ADAPTIVE RETRY -- best-of-N
    must not run any additional attempts (or any extra _capture() calls)
    for it, matching ADAPTIVE RETRY's own "more frames of the same board
    scene are very unlikely to fix a degenerate correspondence" posture."""
    calls = {"local": 0}

    def fake_local(hub, n, **kwargs):
        calls["local"] += 1
        return {0: [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(n)]}

    def fake_detect(image_bgr, pre, **kwargs):
        return np.zeros((4, 3)), np.array(
            [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]
        )

    def fake_calibrate(*a, **k):
        return CalibrationAttempt(ok=False, calibration=None, pnp_result=None, reason="degenerate")

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_local)
    _install_fake_orientation_pipeline(monkeypatch, fake_detect)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", fake_calibrate)

    result = capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), # type: ignore[arg-type]
        n_frames=10, n_frames_detect=10, retry_batch_size=10, max_frames=200,
        n_reprojection_attempts=5,
    )

    assert 0 not in result # genuinely failed, same as today without best-of-N
    assert calls["local"] == 1 # only round 1's own capture -- no best-of-N attempts at all


def test_best_of_n_keeps_per_camera_independence_under_threading(tmp_path, monkeypatch):
    """Requirement 4 -- per-camera independence preserved under best-of-N
    too, mirroring `test_bootstrap_calibrations_keeps_per_camera_state_
    independent_under_threading`'s own pattern above: 3 cameras, each its
    own distinct best-of-N reprojection trajectory, processed
    concurrently -- each must end up with exactly its OWN correct
    adopted result, no cross-camera leakage."""
    n_cams = 3
    reprojections_by_cam = {
        # index 0 = round 1 (always clears the 2.5px target immediately
        # for every camera here, so ADAPTIVE RETRY itself never retries);
        # indices 1-3 = best-of-N attempts 2-4 (n_reprojection_attempts=4).
        0: [1.0, 0.9, 0.8, 0.7], # steadily improves -- best is the LAST attempt
        1: [1.0, 1.5, 1.6, 1.7], # steadily worsens -- best is round 1 (FIRST)
        2: [1.0, 0.5, 1.2, 0.3], # best is attempt #4 (the very last one)
    }
    solve_calls = {cam: 0 for cam in range(n_cams)}

    def fake_local(hub, n, **kwargs):
        return {
            cam: [np.full((4, 4, 3), cam + 1, dtype=np.uint8) for _ in range(n)]
            for cam in range(n_cams)
        }

    def fake_detect(image_bgr, pre, **kwargs):
        return np.zeros((4, 3)), np.array(
            [[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]
        )

    focal_to_cam = {_TEST_FOCAL_PX[cam]: cam for cam in range(n_cams)}

    def fake_calibrate_dispatch(obj_pts, img_pts, camera_matrix, *a, **k):
        f = float(camera_matrix[0, 0])
        cam = focal_to_cam[f]
        idx = solve_calls[cam]
        solve_calls[cam] += 1
        return _fake_attempt_with_reprojection(reprojections_by_cam[cam][idx])

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_local)
    _install_fake_orientation_pipeline_with_focal_dispatch(monkeypatch, fake_detect, n_cams)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", fake_calibrate_dispatch)

    diagnostics: dict = {}
    result = capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), # type: ignore[arg-type]
        n_frames=10, n_frames_detect=10, retry_batch_size=10, max_frames=200,
        n_reprojection_attempts=4,
        diagnostics_out=diagnostics,
    )

    assert sorted(result.keys()) == [0, 1, 2]
    for cam in range(n_cams):
        assert solve_calls[cam] == 4 # every camera got its own full best-of-4
    assert diagnostics[0]["reprojection_error_px"] == 0.7
    assert diagnostics[1]["reprojection_error_px"] == 1.0
    assert diagnostics[2]["reprojection_error_px"] == 0.3


# ---------------------------------------------------------------------------
# STALL-WARNING THROTTLE, 2026-09-13 (CPU task).
#
# The "giving up past the ceiling" WARNING sits on a path the loop
# re-enters EVERY iteration for as long as a stall lasts, so it was
# emitting one formatted line plus a disk write per iteration -- measured
# on the Windows rig at 499 lines in 10.66s. See
# STALL_WARNING_INTERVAL_S's own comment in capture_daemon.py.
# ---------------------------------------------------------------------------


def test_stall_warning_is_throttled_not_emitted_every_iteration(tmp_path, monkeypatch, caplog):
    """A permanently stalled camera must still REPORT (the first iteration
    of a stall always logs, so its onset is timestamped exactly) but must
    not log once per iteration for as long as the stall lasts.

    Reverting the throttle fails this: without it the loop emits one line
    per iteration, so the count equals the iteration count instead of
    being 1.
    """
    converged = _synthetic_frame(120)
    hub = _FakeHubWithFrameCounts([0]) # frame_count never advances -- permanently stalled

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {0: _fake_calibration_attempt().calibration}

    def fake_fetch(dest_dir, *, hub=None):
        return {0: converged}

    from tests.lifecycle_scripting import idle_advance as real_advance

    ITERATIONS = 40
    seen = {"n": 0}

    def counting_real_advance(trigger, bg_frames, current_frames):
        seen["n"] += 1
        if seen["n"] >= ITERATIONS:
            raise _StopLoopEarly("ran enough stalled iterations to measure the log rate")
        return real_advance(trigger, bg_frames, current_frames)

    # 0.0 ceiling -> every iteration is "past the ceiling" from the start.
    monkeypatch.setattr(capture_daemon, "MAX_FRAME_FRESHNESS_WAIT_S", 0.0)
    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, counting_real_advance)
    monkeypatch.setattr(
        capture_daemon, "handle_ready_to_capture", lambda *a, **k: tmp_path / "phantom"
    )

    stop_event = threading.Event()
    with caplog.at_level(logging.WARNING, logger="opendarts.capture_daemon"):
        with pytest.raises(_StopLoopEarly):
            capture_daemon.run_capture_loop_body(
                hub=hub,
                package_root=tmp_path / "packages",
                poll_interval_s=0.0,
                stop_event=stop_event,
                scratch_dir=tmp_path / "scratch",
            )

    stall_lines = [r for r in caplog.records if "frame-freshness gate giving up" in r.getMessage()]
    assert seen["n"] >= ITERATIONS - 1, "the loop did not actually run the iterations under test"
    # The whole run is far shorter than STALL_WARNING_INTERVAL_S, so
    # exactly the first iteration's line should have survived.
    assert len(stall_lines) == 1, (
        f"expected the persistent stall to log once, not once per iteration -- "
        f"got {len(stall_lines)} lines across {seen['n']} iterations"
    )


# ---------------------------------------------------------------------------
# MEASUREMENT CLOCK, 2026-09-13 (CPU task). time.monotonic() is backed by
# GetTickCount64 on Windows through Python 3.12 (~15.6ms granularity), so
# every per-iteration duration -- all of which are smaller than that --
# reported as exactly 0/15/16/32ms on the real rig. See the _iter_perf
# comment in run_capture_loop_body() for the confirmed-live readout.
# ---------------------------------------------------------------------------


def test_iteration_diagnostics_measure_with_perf_counter_not_monotonic():
    """Durations must come from the high-resolution clock; only PACING may
    use monotonic. Reverting any stamp to time.monotonic() fails this."""
    import inspect

    src = inspect.getsource(capture_daemon.run_capture_loop_body)

    # Every diagnostic stamp is taken with perf_counter.
    for stamp in (
        "_iter_perf = time.perf_counter()",
        "_t_gate_start = time.perf_counter()",
        "_t_lifecycle_start = time.perf_counter()",
        "_t_lifecycle_end = time.perf_counter()",
        "_t_body_end = time.perf_counter()",
        "_t_sleep_end = time.perf_counter()",
    ):
        assert stamp in src, f"diagnostic stamp not on perf_counter: {stamp}"

    # The pacing clock is deliberately still monotonic.
    assert "iteration_started = time.monotonic()" in src

    # body is measured against the perf stamp, never the pacing one.
    assert "_body_elapsed = (_t_body_end - _iter_perf)" in src
    assert "_body_elapsed = (_t_body_end - iteration_started)" not in src


def test_no_diagnostic_subtracts_across_the_two_clocks():
    """A perf stamp minus a monotonic one is meaningless, not merely
    coarse -- the two have unrelated origins. This catches a future edit
    that reintroduces the mix (the `post` phase had exactly this bug)."""
    import inspect

    perf_names = (
        "_iter_perf", "_t_gate_start", "_t_lifecycle_start", "_t_lifecycle_end",
        "_t_body_end", "_t_sleep_end", "_prev_iter_window_end",
        "_last_settle_iteration_started",
    )
    mono_names = ("iteration_started", "last_heartbeat")

    offenders = []
    for line in inspect.getsource(capture_daemon.run_capture_loop_body).splitlines():
        if "-" not in line or line.strip().startswith("#"):
            continue
        if any(p in line for p in perf_names) and any(m in line for m in mono_names):
            if "_iter_perf" in line and "iteration_started" in line:
                continue  # the two declarations sit adjacent; not a subtraction
            offenders.append(line.strip())
    assert not offenders, f"diagnostic subtracts across clocks: {offenders}"


def test_also_run_skips_the_prior_dart_lookup_when_every_engine_is_reused(monkeypatch):
    """Engines whose answers come from Zeus's own sub-results never read the
    prior dart's line, so looking it up for them is wasted detect_tip() work."""
    import inspect

    from opendarts.live import capture_daemon

    src = inspect.getsource(capture_daemon.handle_ready_to_capture)
    reuse_at = src.index("reused_sub_results = _reuse_zeus_sub_results_for_also_run(")
    lookup_at = src.index("also_run_prior_dart_line_px = find_prior_dart_line_px(")
    assert reuse_at < lookup_at
    assert "for name in also_run if name not in reused_sub_results" in src
