"""Tests for capture_diagnostics.json -- 2026-08-16, "persist real
diagnostics" task (see docs/DESIGN.md). Real incident: two throws in
one recorded session both showed ad_ground_truth.json's
match_reason "ws_no_buffered_events" and both had by far the largest
inter-throw capture-timing gaps in that whole session; diagnosing why
took real after-the-fact archaeology because nothing durable recorded
the settle timeline or the AD WS listener's buffer state at capture
time. Covers:
  - opendarts.capture.throw_package.save_capture_diagnostics()/
    load_capture_diagnostics() -- the on-disk file itself, and
    load_throw_package()'s backward-compat handling.
  - opendarts.live.capture_daemon._build_capture_diagnostics() -- the real
    dict-building logic (settle duration, straggler camera, ambiguous-
    settle outcome, AD WS buffer snapshot).
  - handle_ready_to_capture() actually writing capture_diagnostics.json
    synchronously as part of a real throw capture, and surviving a
    diagnostics-build failure without affecting the saved package.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pytest

import opendarts.live.capture_daemon as capture_daemon
from opendarts.capture.throw_package import (
    load_capture_diagnostics,
    load_throw_package,
    save_capture_diagnostics,
    save_throw_package,
)
from opendarts.capture.trigger_state import ThrowState, ThrowTriggerState
from opendarts.pipeline import CameraCalibration, ScoreResult

from tests.test_capture_daemon import _FakeAdWsListener, _fake_calibration_attempt


def _fake_score_result() -> ScoreResult:
    return ScoreResult(
        ok=True, sector="20", ring="treble", board_xy_mm=(1.0, 2.0),
        n_cameras_used=1, reason=None, max_ray_disagreement_mm=0.5,
        triangulation=None,
    )


def _throw_trigger_ready(frame: dict[int, np.ndarray]) -> ThrowTriggerState:
    return ThrowTriggerState(state=ThrowState.READY_TO_CAPTURE, last_frame=frame)


# ---------------------------------------------------------------------------
# throw_package.py -- save/load round trip, backward compat.
# ---------------------------------------------------------------------------


def _saved_package_dir(tmp_path):
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calib = {0: CameraCalibration(
        camera_matrix=np.eye(3), dist_coeffs=np.zeros(5),
        rvec=np.zeros(3), tvec=np.array([0.0, 0.0, 1000.0]),
        pnp_result=None, landmark_spread_ok=True,
    )}
    dest_dir = tmp_path / "pkg"
    save_throw_package(dest_dir, "sess1", bg, frame, calib, _fake_score_result())
    return dest_dir


def test_save_and_load_capture_diagnostics_round_trip(tmp_path):
    dest_dir = _saved_package_dir(tmp_path)
    diagnostics = {
        "schema_version": 1,
        "settle": {
            "settle_duration_s": 1.234,
            "straggler_camera": 1,
            "per_camera_settle_offset_s": {"0": 0.1, "1": 1.234},
            "ambiguous_settle_fired": True,
            "ambiguous_settle_outcome": "grew",
        },
        "ad_ws_buffer_at_capture": {"connected": True, "buffered_events": []},
    }
    path = save_capture_diagnostics(dest_dir, diagnostics)
    assert path == dest_dir / "capture_diagnostics.json"

    loaded = load_capture_diagnostics(dest_dir)
    assert loaded == diagnostics


def test_load_capture_diagnostics_returns_none_when_absent(tmp_path):
    dest_dir = _saved_package_dir(tmp_path)
    assert load_capture_diagnostics(dest_dir) is None


def test_load_capture_diagnostics_returns_none_on_corrupt_json(tmp_path):
    dest_dir = _saved_package_dir(tmp_path)
    (dest_dir / "capture_diagnostics.json").write_text("{not valid json")
    assert load_capture_diagnostics(dest_dir) is None


def test_save_capture_diagnostics_requires_an_existing_package_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        save_capture_diagnostics(tmp_path / "does_not_exist", {"a": 1})


def test_load_throw_package_populates_capture_diagnostics_field(tmp_path):
    dest_dir = _saved_package_dir(tmp_path)
    diagnostics = {"schema_version": 1, "settle": {}, "ad_ws_buffer_at_capture": None}
    save_capture_diagnostics(dest_dir, diagnostics)

    pkg = load_throw_package(dest_dir)
    assert pkg.capture_diagnostics == diagnostics


def test_load_throw_package_capture_diagnostics_is_none_for_a_package_saved_before_this_field_existed(tmp_path):
    """Backward compat -- the whole point of the "absent, not malformed"
    convention every other optional package file already follows."""
    dest_dir = _saved_package_dir(tmp_path)
    pkg = load_throw_package(dest_dir)
    assert pkg.capture_diagnostics is None


# ---------------------------------------------------------------------------
# capture_daemon._build_capture_diagnostics() -- the real logic.
# ---------------------------------------------------------------------------


def test_build_capture_diagnostics_with_no_settle_timeline_reports_honest_nones():
    """A trigger with no settle_started_monotonic (e.g. constructed
    directly in READY_TO_CAPTURE, the exact shape most existing
    handle_ready_to_capture() tests' _throw_trigger_ready() helper
    builds) must report None, not a fabricated 0.0 duration."""
    trigger = ThrowTriggerState(state=ThrowState.READY_TO_CAPTURE)
    diagnostics = capture_daemon._build_capture_diagnostics(
        trigger, ad_ws_listener=None, captured_at_monotonic=time.monotonic()
    )
    assert diagnostics["schema_version"] == 2
    assert diagnostics["settle"]["settle_duration_s"] is None
    assert diagnostics["settle"]["straggler_camera"] is None
    assert diagnostics["settle"]["per_camera_settle_offset_s"] == {}
    assert diagnostics["settle"]["ambiguous_settle_fired"] is False
    assert diagnostics["settle"]["ambiguous_settle_outcome"] is None
    assert diagnostics["ad_ws_buffer_at_capture"] is None


def test_build_capture_diagnostics_reports_real_settle_duration_and_straggler():
    started = time.monotonic() - 2.0  # pretend the episode started 2s ago
    trigger = ThrowTriggerState(
        state=ThrowState.READY_TO_CAPTURE,
        settle_started_monotonic=started,
        camera_settled_at_monotonic={0: started + 0.5, 1: started + 1.8},
    )
    diagnostics = capture_daemon._build_capture_diagnostics(
        trigger, ad_ws_listener=None, captured_at_monotonic=time.monotonic()
    )
    settle = diagnostics["settle"]
    assert settle["settle_duration_s"] == pytest.approx(2.0, abs=0.05)
    assert settle["straggler_camera"] == 1, "camera 1 settled last -- it is the straggler"
    assert settle["per_camera_settle_offset_s"]["0"] == pytest.approx(0.5, abs=0.01)
    assert settle["per_camera_settle_offset_s"]["1"] == pytest.approx(1.8, abs=0.01)
    # legacy classifier keys are kept for schema stability, always inert
    assert settle["ambiguous_settle_fired"] is False
    assert settle["ambiguous_settle_outcome"] is None


def test_build_capture_diagnostics_uses_the_passed_timestamp_not_a_fresh_one(monkeypatch):
    """Real, live bug fix, 2026-09-01 -- flagged by the Orchestrator's own
    corpus measurement: real throws showed a suspicious, uniform ~500-700ms
    settle_duration_s cluster, nowhere close to the SAME transition's own
    synchronous throw_trigger log line (e.g. 0.06s for the incident that
    surfaced this). Root cause: this function used to call time.monotonic()
    itself, but runs inside the background save thread as of
    background_save=True -- AFTER save_throw_package()'s own disk I/O, so
    "now" was polluted by unrelated scheduling delay. Proven directly here:
    monkeypatch time.monotonic() to return a value far in the FUTURE
    (simulating exactly that background-thread delay) and confirm
    settle_duration_s still reflects the real, EARLY captured_at_monotonic
    value passed in, not the polluted "current" time."""
    real_monotonic = time.monotonic
    started = real_monotonic() - 0.06  # the real, true settle duration: 0.06s
    real_capture_moment = real_monotonic()  # captured synchronously, like _t_handle_start

    monkeypatch.setattr(
        capture_daemon.time, "monotonic", lambda: real_monotonic() + 10.0  # background delay
    )

    trigger = ThrowTriggerState(
        state=ThrowState.READY_TO_CAPTURE,
        settle_started_monotonic=started,
        camera_settled_at_monotonic={0: started + 0.06},
    )
    diagnostics = capture_daemon._build_capture_diagnostics(
        trigger, ad_ws_listener=None, captured_at_monotonic=real_capture_moment
    )
    assert diagnostics["settle"]["settle_duration_s"] == pytest.approx(0.06, abs=0.02), (
        "settle_duration_s must reflect the real, synchronously-captured timestamp -- "
        "not whatever time.monotonic() happens to return when this function actually runs"
    )


def test_build_capture_diagnostics_ambiguous_settle_not_fired_when_outcome_is_none():
    trigger = ThrowTriggerState(state=ThrowState.READY_TO_CAPTURE)
    diagnostics = capture_daemon._build_capture_diagnostics(
        trigger, ad_ws_listener=None, captured_at_monotonic=time.monotonic()
    )
    assert diagnostics["settle"]["ambiguous_settle_fired"] is False


def test_build_capture_diagnostics_embeds_the_ad_ws_listener_snapshot_when_given():
    trigger = ThrowTriggerState(state=ThrowState.READY_TO_CAPTURE)
    listener = _FakeAdWsListener()
    diagnostics = capture_daemon._build_capture_diagnostics(
        trigger, ad_ws_listener=listener, captured_at_monotonic=time.monotonic()
    )
    assert diagnostics["ad_ws_buffer_at_capture"] is not None
    assert diagnostics["ad_ws_buffer_at_capture"]["connected"] is False
    assert diagnostics["ad_ws_buffer_at_capture"]["buffered_events"] == []


def test_build_capture_diagnostics_is_json_serializable():
    started = time.monotonic()
    trigger = ThrowTriggerState(
        state=ThrowState.READY_TO_CAPTURE,
        settle_started_monotonic=started,
        camera_settled_at_monotonic={0: started, 2: started + 0.01},
    )
    diagnostics = capture_daemon._build_capture_diagnostics(
        trigger, ad_ws_listener=_FakeAdWsListener(), captured_at_monotonic=time.monotonic()
    )
    json.dumps(diagnostics)  # must not raise


# ---------------------------------------------------------------------------
# handle_ready_to_capture() -- real end-to-end wiring.
# ---------------------------------------------------------------------------


def test_handle_ready_to_capture_writes_capture_diagnostics_json(tmp_path, monkeypatch):
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}
    listener = _FakeAdWsListener()

    dest_dir = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
        ad_ws_listener=listener,
    background_save=False,
    )

    diag_path = dest_dir / "capture_diagnostics.json"
    assert diag_path.exists(), "capture_diagnostics.json must be written synchronously"
    diagnostics = json.loads(diag_path.read_text())
    assert diagnostics["schema_version"] == 2
    assert "settle" in diagnostics
    assert diagnostics["ad_ws_buffer_at_capture"] is not None

    # Also loadable back through the normal package API.
    pkg = load_throw_package(dest_dir)
    assert pkg.capture_diagnostics == diagnostics


def test_handle_ready_to_capture_capture_diagnostics_ad_ws_buffer_is_none_with_no_listener(tmp_path):
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}

    dest_dir = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
    background_save=False,
    )
    diagnostics = json.loads((dest_dir / "capture_diagnostics.json").read_text())
    assert diagnostics["ad_ws_buffer_at_capture"] is None


def test_handle_ready_to_capture_reports_real_settle_timeline_fields(tmp_path):
    """End-to-end with a REAL settle timeline on the trigger (not just
    bare READY_TO_CAPTURE) -- exactly what a real capture loop iteration
    hands handle_ready_to_capture() after driving advance() through a
    full settle episode."""
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8), 1: np.full((4, 4, 3), 200, dtype=np.uint8)}
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8), 1: np.full((4, 4, 3), 50, dtype=np.uint8)}
    calibrations = {
        0: _fake_calibration_attempt().calibration,
        1: _fake_calibration_attempt().calibration,
    }
    started = time.monotonic() - 3.5
    trigger = ThrowTriggerState(
        state=ThrowState.READY_TO_CAPTURE,
        last_frame=frame,
        settle_started_monotonic=started,
        camera_settled_at_monotonic={0: started + 1.0, 1: started + 3.5},
    )

    dest_dir = capture_daemon.handle_ready_to_capture(
        trigger, bg, calibrations, tmp_path / "packages", "sess1",
    background_save=False,
    )
    diagnostics = json.loads((dest_dir / "capture_diagnostics.json").read_text())
    settle = diagnostics["settle"]
    assert settle["settle_duration_s"] == pytest.approx(3.5, abs=0.2)
    assert settle["straggler_camera"] == 1
    assert settle["per_camera_settle_offset_s"]["1"] == pytest.approx(3.5, abs=0.05)


def test_handle_ready_to_capture_survives_a_crashing_diagnostics_snapshot(tmp_path, caplog):
    """Same "enrichment must never affect capture reliability" discipline
    as the AD ground-truth attach and also-run engine dispatch -- a
    listener whose diagnostics_snapshot() raises must not prevent the
    throw package itself from saving."""
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}

    class _CrashingListener(_FakeAdWsListener):
        def diagnostics_snapshot(self):
            raise RuntimeError("boom")

    dest_dir = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
        ad_ws_listener=_CrashingListener(),
    background_save=False,
    )
    assert (dest_dir / "meta.json").exists()
    assert (dest_dir / "result.json").exists()
    assert not (dest_dir / "capture_diagnostics.json").exists()
