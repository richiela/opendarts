"""Tests for opendarts/capture/calibration_package.py -- the calibration
package format (REPLAY applied to calibration bootstrap itself, see that
module's own docstring and docs/DESIGN.md's "Replay is the source of truth") plus
its live wiring into opendarts.live.capture_daemon.bootstrap_calibrations()/
opendarts.capture.throw_package.save_throw_package().

Layout, mirroring this project's other calibration-wiring test files:
  1. Module-level unit tests for calibration_package.py itself -- FFV1
     round-trip byte-exactness (synthetic frames, deterministic, no
     corpus needed), save/load round trip, graceful per-camera
     encode-failure degrade, and the rig-side cleanup function.
  2. CalibrationStore package_id plumbing.
  3. bootstrap_calibrations() wiring -- opt-in via
     calibration_package_root, background-vs-blocking save, zero
     behavior/memory change for every caller that doesn't opt in.
  4. handle_ready_to_capture()/save_throw_package() -- calibration_
     package_id reaches meta.json.
  5. One real, unmocked end-to-end pass proving the actual solved
     calibration is bit-identical with vs without package saving, over
     real archived camera frames (skipped when data/archive/ isn't on
     this machine -- gitignored, 1.9GB, same convention as
     tests/test_capture_daemon_oriented_wiring.py).
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.conftest import stub_confident_orientation

import opendarts.live.capture_daemon as capture_daemon
from tests.lifecycle_scripting import script_trigger
from opendarts.calibration.focal_length import FocalLengthResult
from opendarts.calibration.oriented_landmarks import PreOrientationLandmarks
from opendarts.capture import calibration_package as calib_pkg
from opendarts.capture.throw_package import load_throw_package, save_throw_package
from opendarts.pipeline import CalibrationAttempt, CameraCalibration, PnpResult, ScoreResult
from opendarts.capture.trigger_state import ThrowState, ThrowTriggerState

def _pick_archive_session() -> Path | None:
    """Picks whichever real session `data/archive/clean/` currently has,
    rather than hardcoding one date -- docs/DESIGN.md is explicit that this
    corpus is a living, evolving set resets/re-curates over time
    (see its "data/archive/clean/ is a living, curated corpus" standing
    guardrail), so pinning a specific session name here would silently
    stop testing anything real the moment that one session ages out,
    exactly the staleness trap tests/test_capture_daemon_oriented_
    wiring.py's own hardcoded `ARCHIVE_SESSION` already fell into (its
    pinned session no longer exists in the corpus at the time this
    file was written). Picks the first session (by name) that actually
    has at least one real throw directory with a calibration.json --
    every session in this corpus qualifies today, this is just honest
    about not assuming."""
    clean_root = Path("data/archive/clean")
    if not clean_root.is_dir():
        return None
    for session_dir in sorted(clean_root.iterdir()):
        if not session_dir.is_dir():
            continue
        if any(session_dir.glob("*/calibration.json")):
            return session_dir
    return None


ARCHIVE_SESSION = _pick_archive_session()
HAS_REAL_DATA = ARCHIVE_SESSION is not None


def _in_git_checkout() -> bool:
    """Whether `_code_version()` can possibly return a SHA here.

    A `git archive` export -- which is exactly what GitHub serves as the
    "Download source code (tar.gz)" link on a release -- carries no `.git`
    directory, so `git rev-parse HEAD` has nothing to answer from and
    `_code_version()` correctly returns None. The two tests below assert a
    REAL sha, which is a statement about the environment rather than about
    the code, so they skip rather than fail there.

    This is not papering over a defect. `_code_version()` degrading to None
    is its documented contract, it is covered by
    `test_code_version_degrades_to_none_never_raises_when_git_fails`, and
    every consumer already handles it -- `opendarts.live.build_info` renders
    it as an explicit "unknown" in the dashboard's About this rig table for
    precisely this case. What was broken was a test asserting a git
    checkout in an export that by definition is not one.
    """
    try:
        return calib_pkg._code_version() is not None
    except Exception: # noqa: BLE001
        return False


IN_GIT_CHECKOUT = _in_git_checkout()
_NEEDS_GIT = "no .git here (source export, not a clone) -- code_version cannot resolve a SHA"


def _wait_until(predicate, timeout_s: float = 5.0, poll_s: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return predicate()


def _fake_calibration(cam_offset: float = 0.0) -> CameraCalibration:
    return CameraCalibration(
        camera_matrix=np.eye(3),
        dist_coeffs=np.zeros(5),
        rvec=np.zeros(3),
        tvec=np.array([0.0, 0.0, 1000.0 + cam_offset]),
        pnp_result=PnpResult(
            ok=True, rvec=np.zeros(3), tvec=np.array([0.0, 0.0, 1000.0]),
            reprojection_error_px=1.23,
        ),
        landmark_spread_ok=True,
    )


def _synthetic_frames(n: int, w: int = 32, h: int = 24, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8) for _ in range(n)]


# ---------------------------------------------------------------------------
# 1. Module-level unit tests
# ---------------------------------------------------------------------------


def test_new_calibration_package_id_matches_project_session_id_convention():
    """`entropy=` pinned so the timestamp portion is asserted exactly
    (the project's session-id convention) -- the random-suffix DEFAULT
    behavior (no `entropy=` given) is covered separately below."""
    from datetime import datetime, timezone

    pkg_id = calib_pkg.new_calibration_package_id(
        now=datetime(2026, 8, 20, 15, 30, 45, tzinfo=timezone.utc), entropy="deadbeef"
    )
    assert pkg_id == "calib_20260820-153045-deadbeef"


def test_new_calibration_package_id_default_entropy_makes_concurrent_calls_collision_resistant():
    """Real bug fixed 2026-08-20 (verifier pass #2): the timestamp alone
    only has 1-second resolution, and nothing serializes overlapping
    calibration events -- two `new_calibration_package_id()` calls in the
    same real second (e.g. a double-clicked "Refresh calibration now")
    must not produce the same id, or they'd target the same on-disk
    directory. Calls it a realistic number of times (1000) with the SAME
    `now` to prove this isn't luck."""
    from datetime import datetime, timezone

    now = datetime(2026, 8, 20, 15, 30, 45, tzinfo=timezone.utc)
    ids = {calib_pkg.new_calibration_package_id(now=now) for _ in range(1000)}
    assert len(ids) == 1000, "default entropy produced a real collision across 1000 calls"
    for pkg_id in ids:
        assert pkg_id.startswith("calib_20260820-153045-")


def test_ffv1_round_trip_is_byte_exact_synthetic_frames(tmp_path):
    """The core storage-format claim, verified directly against THIS
    implementation (not just cited from an earlier session's finding) --
    real cv2.VideoWriter FFV1 encode, real FFV1 decode, real byte
    comparison. No external ffmpeg binary involved."""
    frames = _synthetic_frames(6, w=48, h=32, seed=1)
    out = tmp_path / "cam0_raw.mkv"
    calib_pkg._encode_raw_video(frames, out)
    assert out.exists()

    decoded = calib_pkg._decode_raw_video(out, n_frames=6, frame_width=48, frame_height=32)
    assert len(decoded) == 6
    for original, back in zip(frames, decoded):
        assert np.array_equal(original, back)


def test_ffv1_round_trip_is_byte_exact_real_corpus_frames():
    """Same claim, against real camera frames instead of synthetic noise
    -- the actual use case (a board photo, not random pixels, which a
    lossy codec would compress very differently)."""
    if not HAS_REAL_DATA:
        pytest.skip("data/archive/ not present (gitignored, 1.9GB)")
    packages = sorted(p.parent for p in ARCHIVE_SESSION.glob("*/result.json"))[:10]
    frames = [cv2.imread(str(p / "cam0_bg.png")) for p in packages]
    frames = [f for f in frames if f is not None]
    assert frames, "no real cam0_bg.png frames found in the archive session"

    with_tmp = Path("tmp/test_calibration_package_ffv1_real.mkv")
    with_tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        calib_pkg._encode_raw_video(frames, with_tmp)
        h, w = frames[0].shape[:2]
        decoded = calib_pkg._decode_raw_video(with_tmp, n_frames=len(frames), frame_width=w, frame_height=h)
        for original, back in zip(frames, decoded):
            assert np.array_equal(original, back)
    finally:
        with_tmp.unlink(missing_ok=True)


def test_encode_raises_value_error_on_empty_frame_list(tmp_path):
    with pytest.raises(ValueError, match="no frames"):
        calib_pkg._encode_raw_video([], tmp_path / "out.mkv")


def test_encode_raises_value_error_on_frame_shape_mismatch(tmp_path):
    frames = _synthetic_frames(2, w=10, h=10) + _synthetic_frames(1, w=20, h=20)
    with pytest.raises(ValueError, match="shape mismatch"):
        calib_pkg._encode_raw_video(frames, tmp_path / "out.mkv")


def test_save_and_load_calibration_package_round_trips_derived_values(tmp_path):
    raw_frames_by_cam = {
        0: _synthetic_frames(4, seed=1),
        1: _synthetic_frames(3, seed=2),
    }
    calibrations = {0: _fake_calibration(0.0), 1: _fake_calibration(5.0)}
    diagnostics = {
        0: {"reprojection_error_px": 0.9, "n_frames_used": 4, "target_met": True},
        1: {"reprojection_error_px": 3.1, "n_frames_used": 3, "target_met": False},
    }

    package_dir = calib_pkg.save_calibration_package(
        tmp_path, "calib_20260820-000000", raw_frames_by_cam, calibrations, diagnostics
    )
    assert package_dir == tmp_path / "calib_20260820-000000"
    assert (package_dir / "meta.json").exists()
    assert (package_dir / "derived_calibration.json").exists()
    assert (package_dir / "cam0_raw.mkv").exists()
    assert (package_dir / "cam1_raw.mkv").exists()

    loaded = calib_pkg.load_calibration_package(package_dir)
    assert loaded.package_id == "calib_20260820-000000"
    assert sorted(loaded.calibrations) == [0, 1]
    for cam in (0, 1):
        assert np.array_equal(loaded.calibrations[cam].camera_matrix, calibrations[cam].camera_matrix)
        assert np.array_equal(loaded.calibrations[cam].tvec, calibrations[cam].tvec)
        assert loaded.diagnostics[cam] == diagnostics[cam]
        for original, back in zip(raw_frames_by_cam[cam], loaded.raw_frames[cam]):
            assert np.array_equal(original, back)


def test_load_calibration_package_can_skip_raw_frame_decode(tmp_path):
    raw_frames_by_cam = {0: _synthetic_frames(3)}
    calibrations = {0: _fake_calibration()}
    package_dir = calib_pkg.save_calibration_package(
        tmp_path, "calib_x", raw_frames_by_cam, calibrations
    )
    loaded = calib_pkg.load_calibration_package(package_dir, decode_raw_frames=False)
    assert loaded.raw_frames[0] is None
    # Derived values are unaffected by skipping the decode.
    assert np.array_equal(loaded.calibrations[0].camera_matrix, calibrations[0].camera_matrix)


def test_save_calibration_package_one_bad_camera_does_not_lose_others(monkeypatch, tmp_path):
    """The one documented exception to 'fails loudly': a per-camera raw
    video encode failure must not lose the derived calibration record or
    any OTHER camera's raw video -- it must be recorded honestly as
    null/None, not silently dropped or raised out of the whole save."""
    real_encode = calib_pkg._encode_raw_video

    def flaky_encode(frames, out_path):
        if "cam1" in out_path.name:
            raise RuntimeError("simulated encode failure for cam1 only")
        return real_encode(frames, out_path)

    monkeypatch.setattr(calib_pkg, "_encode_raw_video", flaky_encode)
    raw_frames_by_cam = {0: _synthetic_frames(2), 1: _synthetic_frames(2)}
    calibrations = {0: _fake_calibration(), 1: _fake_calibration(1.0)}

    package_dir = calib_pkg.save_calibration_package(
        tmp_path, "calib_partial", raw_frames_by_cam, calibrations
    )
    assert (package_dir / "cam0_raw.mkv").exists()
    assert not (package_dir / "cam1_raw.mkv").exists()
    # Both cameras' derived calibration values are still present.
    derived = json.loads((package_dir / "derived_calibration.json").read_text())
    assert sorted(derived["cameras"]) == ["0", "1"]


def test_save_calibration_package_skips_camera_with_no_frames(tmp_path):
    package_dir = calib_pkg.save_calibration_package(
        tmp_path, "calib_empty_cam", {0: []}, {0: _fake_calibration()}
    )
    meta = json.loads((package_dir / "meta.json").read_text())
    # THE V2 PACKAGE SCHEMA (2026-08-27): meta.json's per-camera
    # entries gained a `storage` field (see calibration_package.py's
    # own dated comment) -- updated to
    # assert the new, correct shape rather than silently leaving this
    # test asserting the old one.
    assert meta["cameras"]["0"] == {
        "raw_video": None, "storage": None, "n_frames": 0, "error": None,
    }
    assert not (package_dir / "cam0_raw.mkv").exists()


def test_save_calibration_package_records_timing_fields(tmp_path):
    """2026-08-21: "add timings for calibration... both to the
    packages and to the logging." Real per-camera calibration_
    duration_s lands on each camera's own derived_calibration.json
    entry; the whole event's total_duration_s (identical across every
    camera's diagnostics -- per_cam_diagnostics has no separate
    top-level slot) gets pulled up to this package's own top level
    rather than staying duplicated per camera."""
    diagnostics = {
        0: {
            "reprojection_error_px": 0.8, "n_frames_used": 24, "target_met": True,
            "calibration_duration_s": 12.345, "calibration_total_duration_s": 41.2,
        },
        1: {
            "reprojection_error_px": 1.2, "n_frames_used": 30, "target_met": True,
            "calibration_duration_s": 41.2, "calibration_total_duration_s": 41.2,
        },
    }
    package_dir = calib_pkg.save_calibration_package(
        tmp_path, "calib_timing", {0: [], 1: []},
        {0: _fake_calibration(), 1: _fake_calibration()}, diagnostics,
    )
    derived = json.loads((package_dir / "derived_calibration.json").read_text())
    assert derived["total_duration_s"] == 41.2
    assert derived["cameras"]["0"]["calibration_duration_s"] == 12.345
    assert derived["cameras"]["1"]["calibration_duration_s"] == 41.2


def test_save_calibration_package_timing_absent_when_no_diagnostics(tmp_path):
    """The existing "diagnostics-empty package still writes validly"
    guarantee extends to the new timing fields -- total_duration_s
    degrades to None (not a raise) rather than assuming some camera's
    entry always has it."""
    package_dir = calib_pkg.save_calibration_package(
        tmp_path, "calib_no_diag", {0: []}, {0: _fake_calibration()},
    )
    derived = json.loads((package_dir / "derived_calibration.json").read_text())
    assert derived["total_duration_s"] is None
    assert "calibration_duration_s" not in derived["cameras"]["0"]


def test_save_calibration_package_records_section_timing_fields(tmp_path):
    """same day: "time each section first... then we can look
    for optimization or parallelization." The four shared/once-per-event
    sections (capture, motion-thresholds, ring-boundary, board-color)
    get pulled up to the package's own top level, same treatment as
    total_duration_s; the two genuinely-per-camera sections
    (detect/solve) stay on each camera's own entry."""
    diagnostics = {
        0: {
            "calibration_duration_s": 41.51, "calibration_total_duration_s": 49.31,
            "detect_duration_s": 30.0, "solve_duration_s": 0.02,
            "capture_duration_s": 10.0, "motion_threshold_duration_s": 0.5,
            "ring_boundary_offset_duration_s": 4.2, "board_color_duration_s": 3.1,
        },
        1: {
            "calibration_duration_s": 49.31, "calibration_total_duration_s": 49.31,
            "detect_duration_s": 38.0, "solve_duration_s": 0.03,
            "capture_duration_s": 10.0, "motion_threshold_duration_s": 0.5,
            "ring_boundary_offset_duration_s": 4.2, "board_color_duration_s": 3.1,
        },
    }
    package_dir = calib_pkg.save_calibration_package(
        tmp_path, "calib_section_timing", {0: [], 1: []},
        {0: _fake_calibration(), 1: _fake_calibration()}, diagnostics,
    )
    derived = json.loads((package_dir / "derived_calibration.json").read_text())
    assert derived["capture_duration_s"] == 10.0
    assert derived["motion_threshold_duration_s"] == 0.5
    assert derived["ring_boundary_offset_duration_s"] == 4.2
    assert derived["board_color_duration_s"] == 3.1
    assert derived["cameras"]["0"]["detect_duration_s"] == 30.0
    assert derived["cameras"]["0"]["solve_duration_s"] == 0.02
    assert derived["cameras"]["1"]["detect_duration_s"] == 38.0
    # Section timing fields must NOT leak onto the wrong camera.
    assert "detect_duration_s" not in derived # top-level, not shared-in like capture
    assert "capture_duration_s" not in derived["cameras"]["0"] # per-camera, not duplicated back down


# ---------------------------------------------------------------------------
# Rig-side cleanup
# ---------------------------------------------------------------------------


def _write_throw_meta(throw_dir: Path, calibration_package_id: str | None) -> None:
    throw_dir.mkdir(parents=True, exist_ok=True)
    meta = {"session": throw_dir.parent.name, "cameras": [0]}
    if calibration_package_id is not None:
        meta["calibration_package_id"] = calibration_package_id
    (throw_dir / "meta.json").write_text(json.dumps(meta))


def test_find_referenced_calibration_package_ids_scans_every_throw(tmp_path):
    root = tmp_path / "packages"
    _write_throw_meta(root / "sess1" / "throw_a", "calib_1")
    _write_throw_meta(root / "sess1" / "throw_b", "calib_2")
    _write_throw_meta(root / "sess2" / "throw_c", "calib_2") # duplicate ref, still one entry
    _write_throw_meta(root / "sess2" / "throw_d", None) # no calibration_package_id at all

    referenced = calib_pkg.find_referenced_calibration_package_ids(root)
    assert referenced == {"calib_1", "calib_2"}


def test_find_referenced_calibration_package_ids_skips_corrupt_meta(tmp_path):
    root = tmp_path / "packages"
    _write_throw_meta(root / "sess1" / "throw_a", "calib_1")
    corrupt_dir = root / "sess1" / "throw_bad"
    corrupt_dir.mkdir(parents=True)
    (corrupt_dir / "meta.json").write_text("{not valid json")

    referenced = calib_pkg.find_referenced_calibration_package_ids(root)
    assert referenced == {"calib_1"} # the corrupt throw is skipped, not fatal


def test_find_referenced_calibration_package_ids_missing_root_returns_empty(tmp_path):
    assert calib_pkg.find_referenced_calibration_package_ids(tmp_path / "nope") == set()


def _make_fake_calibration_package(root: Path, pkg_id: str) -> Path:
    d = root / pkg_id
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"package_id": pkg_id}))
    (d / "derived_calibration.json").write_text(json.dumps({"cameras": {}}))
    return d


def test_cleanup_keeps_active_and_referenced_deletes_orphaned(tmp_path):
    calib_root = tmp_path / "calibration_packages"
    throw_root = tmp_path / "packages"
    _make_fake_calibration_package(calib_root, "calib_referenced")
    _make_fake_calibration_package(calib_root, "calib_active_unreferenced")
    _make_fake_calibration_package(calib_root, "calib_orphaned")
    _write_throw_meta(throw_root / "sess1" / "throw_a", "calib_referenced")

    result = calib_pkg.cleanup_orphaned_calibration_packages(
        calib_root, throw_root, active_package_id="calib_active_unreferenced"
    )

    assert result == {
        "kept": ["calib_active_unreferenced", "calib_referenced"],
        "deleted": ["calib_orphaned"],
    }
    assert (calib_root / "calib_referenced").exists()
    assert (calib_root / "calib_active_unreferenced").exists()
    assert not (calib_root / "calib_orphaned").exists()


def test_cleanup_dry_run_reports_without_touching_disk(tmp_path):
    calib_root = tmp_path / "calibration_packages"
    throw_root = tmp_path / "packages"
    _make_fake_calibration_package(calib_root, "calib_orphaned")

    result = calib_pkg.cleanup_orphaned_calibration_packages(
        calib_root, throw_root, dry_run=True
    )
    assert result == {"kept": [], "deleted": ["calib_orphaned"]}
    assert (calib_root / "calib_orphaned").exists() # NOT actually deleted


def test_cleanup_with_no_active_and_nothing_referenced_deletes_everything(tmp_path):
    calib_root = tmp_path / "calibration_packages"
    throw_root = tmp_path / "packages"
    _make_fake_calibration_package(calib_root, "calib_a")
    _make_fake_calibration_package(calib_root, "calib_b")

    result = calib_pkg.cleanup_orphaned_calibration_packages(calib_root, throw_root)
    assert result == {"kept": [], "deleted": ["calib_a", "calib_b"]}
    assert not calib_root.exists() or not any(calib_root.iterdir())


def test_cleanup_missing_calibration_root_is_a_noop(tmp_path):
    result = calib_pkg.cleanup_orphaned_calibration_packages(
        tmp_path / "nonexistent", tmp_path / "packages"
    )
    assert result == {"kept": [], "deleted": []}


def test_cleanup_never_deletes_a_throw_package(tmp_path):
    """Sanity/safety check: cleanup only ever touches
    calibration_package_root -- throw_package_root is read-only input."""
    calib_root = tmp_path / "calibration_packages"
    throw_root = tmp_path / "packages"
    _write_throw_meta(throw_root / "sess1" / "throw_a", "calib_x")
    _make_fake_calibration_package(calib_root, "calib_x")

    calib_pkg.cleanup_orphaned_calibration_packages(calib_root, throw_root)
    assert (throw_root / "sess1" / "throw_a" / "meta.json").exists()


# ---------------------------------------------------------------------------
# Concurrent-save race guard (_IN_FLIGHT_PACKAGE_IDS) -- see module
# docstring's "CONCURRENT SAVES" section and save_calibration_package_
# background()'s own docstring for the real scenario this closes: two
# overlapping live calibration events (e.g. a double-clicked "Refresh
# calibration now") whose cleanup passes could otherwise race against
# each other's still-encoding, not-yet-referenced package.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_in_flight_state():
    """Every test in this module gets a clean `_IN_FLIGHT_PACKAGE_IDS`
    (module-level, process-global by design -- see module docstring) both
    before and after, so a test that fails mid-way (or one that directly
    pokes the set) can never leak state into an unrelated test."""
    calib_pkg._IN_FLIGHT_PACKAGE_IDS.clear()
    yield
    calib_pkg._IN_FLIGHT_PACKAGE_IDS.clear()


def test_cleanup_keeps_in_flight_package_even_when_not_active_or_referenced(tmp_path):
    """The direct unit-level proof of the guard itself: a package that is
    NEITHER active NOR referenced by any throw would normally be deleted
    -- but not while it's registered as in-flight."""
    calib_root = tmp_path / "calibration_packages"
    throw_root = tmp_path / "packages"
    _make_fake_calibration_package(calib_root, "calib_mid_encode")

    calib_pkg._IN_FLIGHT_PACKAGE_IDS["calib_mid_encode"] += 1
    result = calib_pkg.cleanup_orphaned_calibration_packages(
        calib_root, throw_root, active_package_id="calib_unrelated_active"
    )

    assert result == {"kept": ["calib_mid_encode"], "deleted": []}
    assert (calib_root / "calib_mid_encode").exists()


def test_save_calibration_package_background_registers_in_flight_before_thread_starts(tmp_path):
    """`package_id` must already be in `_IN_FLIGHT_PACKAGE_IDS` by the
    time `save_calibration_package_background()` RETURNS (i.e. registered
    synchronously, before the background thread even starts running) --
    otherwise a concurrent cleanup call could slip in during the real
    window this guard exists to close. Uses an Event to hold the
    background thread inside its encode step so the assertion happens
    deterministically, not via a timing guess."""
    started = threading.Event()
    release = threading.Event()

    def blocking_encode(frames, out_path):
        started.set()
        release.wait(timeout=5)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"fake")

    import unittest.mock as mock

    with mock.patch.object(calib_pkg, "_encode_raw_video", blocking_encode):
        thread = calib_pkg.save_calibration_package_background(
            tmp_path / "calibration_packages", "calib_slow",
            {0: _synthetic_frames(1)}, {0: _fake_calibration()},
        )
        assert "calib_slow" in calib_pkg._IN_FLIGHT_PACKAGE_IDS
        assert started.wait(timeout=5), "background thread never reached the encode step"
        assert "calib_slow" in calib_pkg._IN_FLIGHT_PACKAGE_IDS, (
            "must still be in-flight while its own encode is running"
        )
        release.set()
        thread.join(timeout=10)

    assert "calib_slow" not in calib_pkg._IN_FLIGHT_PACKAGE_IDS, (
        "must be cleared once the save (and any cleanup it triggered) has finished"
    )


def test_save_calibration_package_background_releases_its_frames_and_trims_when_done(tmp_path):
    """The background save is the last holder of a calibration's raw frame
    pool (~1 GB at peak on the rig). Once it has finished, the frames must
    be unreachable even while the Thread object itself is still referenced,
    and the freed heap handed back -- see opendarts.live.heap_trim."""
    import gc
    import unittest.mock as mock
    import weakref

    frames = _synthetic_frames(3)
    refs = [weakref.ref(f) for f in frames]
    trims: list[tuple[str, int]] = []

    def recording_trim(reason):
        # What matters is what is still alive AT the trim -- a trim that
        # runs while the closure still holds the pool frees nothing.
        gc.collect()
        trims.append((reason, sum(r() is not None for r in refs)))

    with mock.patch.object(calib_pkg, "release_freed_heap", recording_trim):
        thread = calib_pkg.save_calibration_package_background(
            tmp_path / "calibration_packages", "calib_release",
            {0: frames}, {0: _fake_calibration()},
        )
        del frames
        thread.join(timeout=30)
    gc.collect()

    assert not thread.is_alive()
    assert (tmp_path / "calibration_packages" / "calib_release").exists()
    assert [r for r in refs if r() is not None] == []
    assert trims == [("calibration package save", 0)]


def test_concurrent_background_saves_do_not_delete_each_others_in_flight_package(tmp_path):
    """End-to-end reproduction of the real race the guard was built for:
    package A's background save is still encoding (slow, held open via
    an Event) when package B's OWN background save completes and runs
    ITS post-save cleanup pass (active_package_id="calib_b", which knows
    nothing about A). Before the fix, A -- neither active-for-B's-cleanup
    nor yet referenced by any throw -- would be deleted out from under
    its own still-running encode. After the fix, A survives because it is
    registered as in-flight, and is only deleted by a LATER cleanup pass
    once it has actually finished (and remains unreferenced/inactive)."""
    calib_root = tmp_path / "calibration_packages"
    throw_root = tmp_path / "packages" # no throws reference anything

    a_started = threading.Event()
    a_release = threading.Event()
    real_encode = calib_pkg._encode_raw_video

    def encode_side_effect(frames, out_path):
        if "calib_a" in str(out_path):
            a_started.set()
            a_release.wait(timeout=5)
        return real_encode(frames, out_path)

    import unittest.mock as mock

    with mock.patch.object(calib_pkg, "_encode_raw_video", side_effect=encode_side_effect):
        thread_a = calib_pkg.save_calibration_package_background(
            calib_root, "calib_a", {0: _synthetic_frames(1)}, {0: _fake_calibration()},
        )
        assert a_started.wait(timeout=5), "package A's background thread never started encoding"

        # Package B's own save + its own post-save cleanup pass, run
        # fully synchronously here (blocking=True is what a live-wiring
        # test would use) while A is still mid-encode. This calls the
        # exact same cleanup_orphaned_calibration_packages() a real
        # overlapping "Refresh calibration now" click would trigger.
        thread_b = calib_pkg.save_calibration_package_background(
            calib_root, "calib_b", {0: _synthetic_frames(1)}, {0: _fake_calibration()},
            throw_package_root=throw_root,
        )
        thread_b.join(timeout=10)
        assert not thread_b.is_alive()

        # The real assertion: A's package directory must still exist --
        # B's cleanup pass ran while A was in-flight and must have kept it.
        assert (calib_root / "calib_a").exists(), (
            "package A was deleted by a concurrent cleanup pass while its own "
            "background save was still encoding -- the race guard did not hold"
        )

        a_release.set()
        thread_a.join(timeout=10)

    assert not thread_a.is_alive()
    assert "calib_a" not in calib_pkg._IN_FLIGHT_PACKAGE_IDS
    assert "calib_b" not in calib_pkg._IN_FLIGHT_PACKAGE_IDS

    # Now that both saves have actually finished, a-later cleanup pass
    # (nothing active, nothing referenced) is free to delete both --
    # proving the guard is a WINDOW, not a permanent keep.
    result = calib_pkg.cleanup_orphaned_calibration_packages(calib_root, throw_root)
    assert result == {"kept": [], "deleted": ["calib_a", "calib_b"]}


def test_concurrent_saves_sharing_the_same_package_id_do_not_lose_protection(tmp_path):
    """Real bug found by a verifier pass (2026-08-20, second look), fixed
    here -- direct reproduction of the exact scenario it demonstrated:
    TWO independent `save_calibration_package_background()` calls sharing
    the SAME `package_id` (the same-second timestamp-collision case
    `new_calibration_package_id()`'s new entropy suffix now makes
    astronomically unlikely, but this test proves the in-flight guard
    itself is also correct as defense in depth, not just "usually doesn't
    happen because ids differ"). Call 1 finishes fast; call 2 is held
    mid-encode via a real Event. Before the Counter fix, call 1's
    `finally` block would `.discard()` the shared id from a plain `set`,
    stripping protection from call 2 while it was still writing -- a
    concurrent cleanup pass would then delete the package out from under
    call 2's still-running encode. With the Counter fix, call 1 only
    decrements; call 2's own registration keeps the count above zero
    until call 2 itself finishes."""
    calib_root = tmp_path / "calibration_packages"
    throw_root = tmp_path / "packages"
    shared_id = "calib_same_20260820-000000"

    call2_started = threading.Event()
    call2_release = threading.Event()
    real_encode = calib_pkg._encode_raw_video
    call_count = {"n": 0}

    def encode_side_effect(frames, out_path):
        call_count["n"] += 1
        if call_count["n"] == 2:
            # The second physical encode to run is "call 2" -- hold it
            # open so call 1 (whichever finishes its own encode first)
            # can complete and run its `finally` block while call 2 is
            # still genuinely mid-write.
            call2_started.set()
            call2_release.wait(timeout=5)
        return real_encode(frames, out_path)

    import unittest.mock as mock

    with mock.patch.object(calib_pkg, "_encode_raw_video", side_effect=encode_side_effect):
        thread1 = calib_pkg.save_calibration_package_background(
            calib_root, shared_id, {0: _synthetic_frames(1)}, {0: _fake_calibration()},
        )
        thread2 = calib_pkg.save_calibration_package_background(
            calib_root, shared_id, {1: _synthetic_frames(1)}, {1: _fake_calibration(1.0)},
        )
        assert call2_started.wait(timeout=5), "second concurrent encode never started"
        # Give whichever thread reached its encode first a real chance to
        # fully finish (including its own `finally` discard) while the
        # other is still deliberately held open.
        _wait_until(
            lambda: sum(t.is_alive() for t in (thread1, thread2)) <= 1, timeout_s=5
        )

        # The real assertion: the shared id must still be genuinely
        # in-flight (count > 0) -- one call finishing must not have wiped
        # out the other call's own still-active registration.
        assert shared_id in calib_pkg._IN_FLIGHT_PACKAGE_IDS, (
            "one concurrent save finishing early stripped in-flight protection "
            "from the other still-running save sharing the same package_id"
        )
        # And a cleanup pass run right now (simulating a third overlapping
        # event's own post-save cleanup) must still keep it.
        result = calib_pkg.cleanup_orphaned_calibration_packages(
            calib_root, throw_root, active_package_id="calib_unrelated_third_event"
        )
        assert shared_id in result["kept"]
        assert (calib_root / shared_id).exists()

        call2_release.set()
        thread1.join(timeout=10)
        thread2.join(timeout=10)

    assert shared_id not in calib_pkg._IN_FLIGHT_PACKAGE_IDS


# ---------------------------------------------------------------------------
# save_calibration_package_background
# ---------------------------------------------------------------------------


def test_save_calibration_package_background_runs_and_completes(tmp_path):
    raw_frames_by_cam = {0: _synthetic_frames(2)}
    calibrations = {0: _fake_calibration()}
    thread = calib_pkg.save_calibration_package_background(
        tmp_path / "calibration_packages", "calib_bg", raw_frames_by_cam, calibrations
    )
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert (tmp_path / "calibration_packages" / "calib_bg" / "meta.json").exists()


def test_save_calibration_package_background_never_raises_on_failure(tmp_path, monkeypatch, caplog):
    """The whole point of the background wrapper: a save failure must be
    logged, never propagate anywhere (there is no caller left to catch
    it by the time this runs)."""
    def boom(*a, **k):
        raise RuntimeError("simulated disk failure")

    monkeypatch.setattr(calib_pkg, "save_calibration_package", boom)
    import logging

    with caplog.at_level(logging.ERROR):
        thread = calib_pkg.save_calibration_package_background(
            tmp_path, "calib_boom", {0: _synthetic_frames(1)}, {0: _fake_calibration()}
        )
        thread.join(timeout=5)
    assert not thread.is_alive()
    assert "background save failed" in caplog.text


def test_save_calibration_package_background_runs_cleanup_when_throw_root_given(tmp_path):
    calib_root = tmp_path / "calibration_packages"
    throw_root = tmp_path / "packages"
    _make_fake_calibration_package(calib_root, "calib_old_orphan")

    thread = calib_pkg.save_calibration_package_background(
        calib_root, "calib_new", {0: _synthetic_frames(2)}, {0: _fake_calibration()},
        throw_package_root=throw_root,
    )
    thread.join(timeout=10)
    assert (calib_root / "calib_new").exists() # the just-saved (active) one survives
    assert not (calib_root / "calib_old_orphan").exists() # orphan cleaned up


# ---------------------------------------------------------------------------
# 2. CalibrationStore package_id plumbing
# ---------------------------------------------------------------------------


def test_calibration_store_get_package_id_defaults_to_none():
    store = capture_daemon.CalibrationStore()
    assert store.get_package_id() is None
    assert store.meta()["calibration_package_id"] is None


def test_calibration_store_set_stores_and_returns_package_id():
    store = capture_daemon.CalibrationStore()
    store.set({0: _fake_calibration()}, source="startup", checked_at_utc="t0", package_id="calib_abc")
    assert store.get_package_id() == "calib_abc"
    assert store.meta()["calibration_package_id"] == "calib_abc"


def test_calibration_store_set_without_package_id_clears_it():
    store = capture_daemon.CalibrationStore(package_id="calib_old")
    assert store.get_package_id() == "calib_old"
    store.set({0: _fake_calibration()}, source="manual", checked_at_utc="t1")
    assert store.get_package_id() is None


def test_calibration_store_constructor_accepts_package_id():
    store = capture_daemon.CalibrationStore({0: _fake_calibration()}, package_id="calib_init")
    assert store.get_package_id() == "calib_init"


def test_calibration_store_get_with_package_id_returns_both_atomically():
    store = capture_daemon.CalibrationStore(
        {0: _fake_calibration()}, source="startup", checked_at_utc="t0", package_id="calib_a",
    )
    calibs, package_id = store.get_with_package_id()
    assert sorted(calibs) == [0]
    assert np.array_equal(calibs[0].camera_matrix, _fake_calibration().camera_matrix)
    assert package_id == "calib_a"


def test_calibration_store_get_with_package_id_never_mixes_two_different_set_calls():
    """The real bug two separate .get()/.get_package_id() calls could hit:
    a concurrent .set() landing between them. Simulated by swapping in a
    lock wrapper (a real threading.Lock instance cannot have its own
    .acquire attribute monkeypatched -- it's a C-level type, confirmed
    directly) that, on its first use, spawns a real .set() from a second
    thread WHILE still holding the real underlying lock -- proving the
    second thread genuinely blocks on the same critical section, not just
    asserting the outcome. get_with_package_id() must still return values
    from exactly ONE .set() call, never a mix of an old dict and a new
    package_id (or vice versa)."""

    class _SpyingLock:
        def __init__(self, real_lock, on_first_acquire):
            self._real_lock = real_lock
            self._on_first_acquire = on_first_acquire
            self._fired = False

        def __enter__(self):
            self._real_lock.acquire()
            if not self._fired:
                self._fired = True
                self._on_first_acquire()
            return self

        def __exit__(self, *exc_info):
            self._real_lock.release()
            return False

    store = capture_daemon.CalibrationStore(
        {0: _fake_calibration(1.0)}, source="startup", checked_at_utc="t0", package_id="calib_old",
    )
    real_lock = store._lock
    setter_holder: list[threading.Thread] = []

    def _spawn_concurrent_set():
        # Deliberately NOT joined here -- this runs from INSIDE
        # get_with_package_id()'s own critical section (the real lock is
        # still held by this same thread at this point), so joining here
        # would deadlock: the setter thread's own store.set() call blocks
        # on that same real lock until this thread's __exit__ releases
        # it, which can't happen while this thread is stuck waiting on
        # join(). Start it, give it a moment to reach (and correctly
        # block on) that real acquire, and let the caller join it AFTER
        # the outer critical section has exited.
        setter = threading.Thread(
            target=lambda: store.set(
                {0: _fake_calibration(2.0)}, source="manual", checked_at_utc="t1",
                package_id="calib_new",
            )
        )
        setter.start()
        setter_holder.append(setter)
        time.sleep(0.05) # give the setter a real chance to reach the blocking acquire

    store._lock = _SpyingLock(real_lock, _spawn_concurrent_set)
    try:
        calibs, package_id = store.get_with_package_id()
    finally:
        store._lock = real_lock
        if setter_holder:
            setter_holder[0].join(timeout=2)

    # Whichever call "won" the race for the lock, the two returned values
    # must describe the SAME .set() -- old+old or new+new, never mixed.
    is_old = package_id == "calib_old" and calibs[0].tvec[2] == 1001.0
    is_new = package_id == "calib_new" and calibs[0].tvec[2] == 1002.0
    assert is_old or is_new, (calibs[0].tvec, package_id)


# ---------------------------------------------------------------------------
# 2b. CalibrationStore snapshot_path persistence (2026-08-26) -- "make it
# so calibration is saved and reused... it should only be recalculated
# at the users request." See CalibrationStore's own docstring's
# "Durable across process restarts" section for the full design.
# `tests/conftest.py`'s autouse `_reset_live_derived_module_globals`
# fixture already resets board.py/board_color.py's live-derived
# globals before AND after every test in this file, so tests below
# that call set_ring_boundary_offsets()/set_board_color_thresholds()
# directly (to simulate "this event's bootstrap already applied
# them") don't need their own cleanup.
# ---------------------------------------------------------------------------


def test_calibration_store_snapshot_path_missing_file_falls_through_to_defaults(tmp_path):
    """No file at snapshot_path at all -- the common case (first-ever run
    on a fresh machine) -- must behave EXACTLY like no snapshot_path was
    given: empty store, constructor defaults, no exception."""
    store = capture_daemon.CalibrationStore(snapshot_path=tmp_path / "current_calibration.json")
    assert store.get() == {}
    assert store.meta()["source"] == "startup" # the constructor default, unchanged
    assert store.meta()["checked_at_utc"] is None


def test_calibration_store_set_persists_a_snapshot_that_a_new_store_loads(tmp_path):
    """The real round trip this whole feature exists for: .set() writes a
    snapshot; a BRAND NEW CalibrationStore (simulating a fresh process)
    constructed with the same snapshot_path loads it back at construction
    -- with NO calibrations passed to its own constructor -- and reports
    source="persisted"."""
    from opendarts.geometry.board import set_ring_boundary_offsets, get_ring_boundary_offsets
    from opendarts.geometry.board_color import set_board_color_thresholds, get_board_color_thresholds

    snapshot_path = tmp_path / "calibration_packages" / "current_calibration.json"
    # Simulates what bootstrap_calibrations() already did for this event,
    # BEFORE calibration_store.set() is ever called in both real call
    # sites (run_capture_loop_body, AppState.refresh_calibration) --
    # .set()'s own snapshot write reads these back live, per its own
    # docstring.
    set_ring_boundary_offsets(treble_inner_offset_mm=1.7, double_inner_offset_mm=1.4)
    set_board_color_thresholds(brightness_threshold=120.0, chroma_threshold=30.0)

    writer = capture_daemon.CalibrationStore(snapshot_path=snapshot_path)
    assert writer.get() == {} # nothing on disk yet
    writer.set(
        {0: _fake_calibration(42.0), 1: _fake_calibration(43.0)},
        source="startup",
        checked_at_utc="2026-08-26T12:00:00+00:00",
        package_id="calib_20260826-120000-abcd1234",
    )
    assert snapshot_path.exists()

    # Reset the live globals to defaults BEFORE constructing the "new
    # process" store below -- proves the reader (not something left over
    # from the writer's own set() call above) is what re-applies them.
    set_ring_boundary_offsets(None, None)
    set_board_color_thresholds(None, None)
    assert get_ring_boundary_offsets() == (None, None)
    assert get_board_color_thresholds() == (None, None)

    reader = capture_daemon.CalibrationStore(snapshot_path=snapshot_path)
    loaded = reader.get()
    assert sorted(loaded) == [0, 1]
    assert loaded[0].tvec[2] == 1042.0 # _fake_calibration(42.0)'s own tvec[2]
    assert loaded[1].tvec[2] == 1043.0
    meta = reader.meta()
    assert meta["source"] == "persisted"
    assert meta["checked_at_utc"] == "2026-08-26T12:00:00+00:00"
    assert meta["calibration_package_id"] == "calib_20260826-120000-abcd1234"
    assert meta["n_cameras"] == 2

    # The ring-boundary-offset/board-color globals were re-applied at
    # LOAD time from the persisted snapshot -- not left at the defaults
    # they were reset to just above.
    assert get_ring_boundary_offsets() == (1.7, 1.4)
    assert get_board_color_thresholds() == (120.0, 30.0)


def test_calibration_store_snapshot_path_corrupt_json_falls_through_to_defaults(tmp_path, caplog):
    snapshot_path = tmp_path / "current_calibration.json"
    snapshot_path.write_text("{not valid json")
    import logging

    with caplog.at_level(logging.WARNING):
        store = capture_daemon.CalibrationStore(snapshot_path=snapshot_path)
    assert store.get() == {}
    assert store.meta()["source"] == "startup"
    assert "failed to read/parse" in caplog.text


def test_calibration_store_snapshot_path_wrong_schema_falls_through_to_defaults(tmp_path):
    snapshot_path = tmp_path / "current_calibration.json"
    snapshot_path.write_text(json.dumps({"schema": "some-old-schema-v0", "cameras": {}}))
    store = capture_daemon.CalibrationStore(snapshot_path=snapshot_path)
    assert store.get() == {}
    assert store.meta()["source"] == "startup"


def test_calibration_store_snapshot_path_no_cameras_falls_through_to_defaults(tmp_path):
    snapshot_path = tmp_path / "current_calibration.json"
    snapshot_path.write_text(json.dumps({
        "schema": capture_daemon.CURRENT_CALIBRATION_SCHEMA,
        "checked_at_utc": "t0",
        "cameras": {},
    }))
    store = capture_daemon.CalibrationStore(snapshot_path=snapshot_path)
    assert store.get() == {}


def test_calibration_store_snapshot_path_missing_checked_at_utc_falls_through_to_defaults(tmp_path):
    """A schema-valid file missing the one field this whole feature exists
    to surface (checked_at_utc, for the dashboard's staleness display)
    must not silently load with a fabricated timestamp -- absent is safer
    than invented, matches this project's "never fabricate a value"
    convention elsewhere."""
    snapshot_path = tmp_path / "current_calibration.json"
    snapshot_path.write_text(json.dumps({
        "schema": capture_daemon.CURRENT_CALIBRATION_SCHEMA,
        "cameras": {"0": calib_pkg.calibration_to_dict(_fake_calibration())},
    }))
    store = capture_daemon.CalibrationStore(snapshot_path=snapshot_path)
    assert store.get() == {}


def test_calibration_store_set_without_snapshot_path_never_touches_disk(tmp_path):
    """Every existing caller/test that doesn't pass snapshot_path (the
    default, None) must be completely unaffected -- purely in-memory,
    same as before this feature existed."""
    store = capture_daemon.CalibrationStore()
    store.set({0: _fake_calibration()}, source="manual", checked_at_utc="t1")
    assert list(tmp_path.iterdir()) == [] # nothing written anywhere


class _StopLoop(Exception):
    """Local to this file -- mirrors tests/test_capture_daemon.py's own
    identically-named helper (not imported from there; that module has
    its own private copy too, deliberately not shared, same pattern)."""


def test_run_capture_loop_body_reuses_a_persisted_calibration_without_bootstrapping(
    tmp_path, monkeypatch
):
    """The real end-to-end point of this feature: a CalibrationStore
    constructed with snapshot_path pointing at a REAL prior .set()'s
    output -- simulating a fresh process after a restart, exactly like
    opendarts.live.run_product.py's own wiring -- must make
    run_capture_loop_body() skip bootstrap_calibrations() entirely on the
    very FIRST Start of this new process, the same as the already-tested
    same-process-lifetime reuse case above, via the SAME existing "already
    holds at least one camera's calibration" check (no new branch)."""
    snapshot_path = tmp_path / "calibration_packages" / "current_calibration.json"
    writer = capture_daemon.CalibrationStore(snapshot_path=snapshot_path)
    writer.set(
        {0: _fake_calibration(5555.0)},
        source="startup", checked_at_utc="t-prior-process", package_id="calib_prior",
    )

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

    # A brand-new store, mirroring opendarts.live.run_product.py's own
    # construction -- NOT the `writer` object above (that would just be
    # testing in-memory reuse, already covered elsewhere). This is a
    # SEPARATE CalibrationStore instance that only shares the snapshot
    # file on disk, the real shape of "a fresh process after a restart."
    fresh_process_store = capture_daemon.CalibrationStore(snapshot_path=snapshot_path)
    assert fresh_process_store.meta()["source"] == "persisted"

    stop_event = threading.Event()
    with pytest.raises(_StopLoop):
        capture_daemon.run_capture_loop_body(
            hub=None,
            package_root=tmp_path / "packages",
            poll_interval_s=0.0,
            stop_event=stop_event,
            scratch_dir=tmp_path / "scratch",
            calibration_store=fresh_process_store,
        )

    assert bootstrap_calls["n"] == 0, "must NOT bootstrap when a valid persisted calibration exists"
    assert fresh_process_store.get()[0].tvec[2] == 6555.0 # _fake_calibration(5555.0)'s own tvec[2]


# ---------------------------------------------------------------------------
# 3. bootstrap_calibrations() wiring
# ---------------------------------------------------------------------------


def _frames(n_frames: int, cams=(0,)) -> dict:
    return {c: [np.zeros((10, 10, 3), np.uint8) for _ in range(n_frames)] for c in cams}


# Shared fake for bootstrap_calibrations()'s live-derived orientation-hint
# pipeline (2026-08-20 restructuring -- see capture_daemon.py's own
# "LIVE-DERIVED ORIENTATION HINT" docstring section, and
# tests/test_capture_daemon.py's own identical helper/comment). The old
# single `correspond_landmarks_oriented(image_bgr, cam, **kwargs)` mock
# point no longer exists on capture_daemon -- the real call path is now a
# two-pass split (pre-orientation landmarks -> session-level hint
# aggregation -> finishing correspondence). None of these calibration-
# package tests exercise real orientation math; every piece here except
# `correspond` is a harmless, always-confident stand-in.
_FAKE_PRE_OK = PreOrientationLandmarks(
    ok=True, reason="ok", ellipse=None, seed_ellipse=None, bull_px=(1.0, 1.0),
    normalised_bull_radius=0.0, profile=None, phase_deg=0.0,
    spoke_score=1.0, phase_confidence=5.0, notes=[],
)


def _install_fake_orientation_pipeline(monkeypatch, correspond):
    # As of 2026-08-29, `ORIENTATION_METHOD` defaults to `"ring_
    # correlation"` (DOHS3) and runs BEFORE the digit-count fakes below
    # ever matter -- this file's own calibration-package tests exercise
    # package save/load/replay plumbing, not orientation math, so force
    # the selector back to the digit-count method (same as
    # tests/test_capture_daemon.py's own identical fix).
    stub_confident_orientation(monkeypatch, capture_daemon)
    monkeypatch.setattr(
        capture_daemon, "locate_pre_orientation_landmarks",
        lambda frame, **kw: (frame, _FAKE_PRE_OK),
    )
    monkeypatch.setattr(capture_daemon, "correspond_landmarks_from_pre_orientation", correspond)
    # LIVE-DERIVED FOCAL LENGTH, 2026-08-26 -- same "harmless, always-
    # confident stand-in" treatment as every other piece here (see
    # tests/test_capture_daemon.py's own identical helper/comment): none
    # of these calibration-package tests exercise real focal-length math,
    # and a resolved focal length is now required before any
    # calibrate_camera() call can happen at all.
    monkeypatch.setattr(
        capture_daemon, "derive_focal_length_from_oriented_results",
        lambda results, principal_point, min_frames=1: FocalLengthResult(
            ok=True, focal_length_px=900.0, reason="test-fake",
            n_points_used=20, n_frames_used=max(1, len(results)),
        ),
    )


def _mock_calibration_plumbing(monkeypatch):
    obj = np.zeros((4, 3))
    px = np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])

    def fake_correspond(image_bgr, pre, **kwargs):
        return obj, px

    def fake_calibrate(*a, **k):
        return CalibrationAttempt(
            ok=True, calibration=_fake_calibration(), pnp_result=PnpResult(
                ok=True, rvec=np.zeros(3), tvec=np.zeros(3), reprojection_error_px=0.5,
            ), reason="",
        )

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", lambda hub, n, **k: _frames(n))
    _install_fake_orientation_pipeline(monkeypatch, fake_correspond)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", fake_calibrate)


def test_bootstrap_calibrations_package_root_none_is_zero_behavior_change(tmp_path, monkeypatch):
    """The default (calibration_package_root=None) -- every existing
    caller/test -- must not create ANY calibration-package directory or
    touch calibration_package_out."""
    _mock_calibration_plumbing(monkeypatch)
    out: dict = {}
    result = capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), n_frames=2,
        calibration_package_out=out,
    )
    assert 0 in result
    assert out == {} # never touched -- package saving never ran


def test_bootstrap_calibrations_blocking_package_save_writes_a_real_package(tmp_path, monkeypatch):
    _mock_calibration_plumbing(monkeypatch)
    pkg_root = tmp_path / "calibration_packages"
    out: dict = {}
    result = capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), n_frames=3,
        calibration_package_root=pkg_root,
        calibration_package_out=out,
        calibration_package_blocking=True,
    )
    assert 0 in result
    assert out["package_id"].startswith("calib_")
    assert out["package_dir"] == pkg_root / out["package_id"]
    assert (out["package_dir"] / "meta.json").exists()
    assert (out["package_dir"] / "derived_calibration.json").exists()
    assert (out["package_dir"] / "cam0_raw.mkv").exists()

    loaded = calib_pkg.load_calibration_package(out["package_dir"])
    assert len(loaded.raw_frames[0]) == 3 # the exact n_frames captured this event


def test_bootstrap_calibrations_background_package_save_eventually_appears(tmp_path, monkeypatch):
    """Real live wiring uses the non-blocking path -- must not delay the
    function's own return, but the package must show up shortly after on
    its own background thread."""
    _mock_calibration_plumbing(monkeypatch)
    pkg_root = tmp_path / "calibration_packages"
    out: dict = {}

    t0 = time.monotonic()
    result = capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), n_frames=2,
        calibration_package_root=pkg_root,
        calibration_package_out=out,
        # calibration_package_blocking defaults to False -- the real live default
    )
    elapsed = time.monotonic() - t0

    assert 0 in result
    package_id = out["package_id"]
    assert package_id # generated synchronously, available immediately
    assert elapsed < 5.0, "bootstrap_calibrations() must not block on package saving"

    assert _wait_until(lambda: (pkg_root / package_id / "meta.json").exists(), timeout_s=10), (
        "calibration package never appeared -- background save did not run"
    )


def test_bootstrap_calibrations_package_save_failure_does_not_break_calibration(
    tmp_path, monkeypatch, caplog
):
    """The real live-path constraint: package saving must NEVER cascade
    into a broken calibration result, even when it fails outright."""
    _mock_calibration_plumbing(monkeypatch)

    def boom(*a, **k):
        raise RuntimeError("simulated package-save failure")

    monkeypatch.setattr(capture_daemon, "save_calibration_package", boom)

    import logging
    with caplog.at_level(logging.ERROR):
        result = capture_daemon.bootstrap_calibrations(
            tmp_path, hub=object(), n_frames=2,
            calibration_package_root=tmp_path / "calibration_packages",
            calibration_package_blocking=True,
        )
    assert 0 in result # the real calibration result is completely unaffected
    assert isinstance(result[0], CameraCalibration)


def test_bootstrap_calibrations_does_not_accumulate_raw_frames_when_opted_out(tmp_path, monkeypatch):
    """Memory-conscious guard: raw_frames_accum must stay empty (not just
    unused) unless a caller actually opts into calibration-package
    saving -- verified via a spy on how many frames the mocked detector
    actually sees vs. what a package would need, indirectly through the
    fact that no package directory is ever created and calibration_
    package_out (if given without a root) is never populated."""
    _mock_calibration_plumbing(monkeypatch)
    out: dict = {}
    capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), n_frames=5,
        calibration_package_out=out, # given, but calibration_package_root is NOT
    )
    assert out == {}
    assert not (tmp_path / "calibration_packages").exists()


# ---------------------------------------------------------------------------
# 4. handle_ready_to_capture()/save_throw_package() -- calibration_package_id
# ---------------------------------------------------------------------------


def _throw_trigger_ready(frame: dict) -> ThrowTriggerState:
    trigger = ThrowTriggerState()
    trigger.state = ThrowState.READY_TO_CAPTURE
    trigger.last_frame = frame
    trigger.dart_count = 1
    return trigger


def test_handle_ready_to_capture_stamps_calibration_package_id_onto_meta(tmp_path):
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration()}

    dest_dir = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
        calibration_package_id="calib_20260820-000000",
    background_save=False,
    )
    meta = json.loads((dest_dir / "meta.json").read_text())
    assert meta["calibration_package_id"] == "calib_20260820-000000"


def test_handle_ready_to_capture_omits_calibration_package_id_when_none(tmp_path):
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration()}

    dest_dir = capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
    background_save=False,
    )
    meta = json.loads((dest_dir / "meta.json").read_text())
    assert "calibration_package_id" not in meta


def _fake_score_result() -> ScoreResult:
    return ScoreResult(
        ok=True, sector="20", ring="treble", board_xy_mm=(1.0, 2.0),
        triangulation=None, n_cameras_used=3,
    )


def test_save_throw_package_round_trips_calibration_package_id(tmp_path):
    bg = {0: np.zeros((4, 4, 3), np.uint8)}
    frame = {0: np.full((4, 4, 3), 5, np.uint8)}
    calibrations = {0: _fake_calibration()}

    dest = save_throw_package(
        tmp_path / "throw1", "sess1", bg, frame, calibrations, _fake_score_result(),
        calibration_package_id="calib_20260820-153045",
    )
    pkg = load_throw_package(dest)
    assert pkg.calibration_package_id == "calib_20260820-153045"
    # calibration.json itself is UNCHANGED -- still the full CameraCalibration,
    # not replaced by the reference (see save_throw_package's own docstring).
    assert (dest / "calibration.json").exists()
    assert pkg.calibrations is not None


def test_save_throw_package_without_calibration_package_id_loads_as_none(tmp_path):
    bg = {0: np.zeros((4, 4, 3), np.uint8)}
    frame = {0: np.full((4, 4, 3), 5, np.uint8)}
    calibrations = {0: _fake_calibration()}

    dest = save_throw_package(
        tmp_path / "throw1", "sess1", bg, frame, calibrations, _fake_score_result(),
    )
    pkg = load_throw_package(dest)
    assert pkg.calibration_package_id is None
    meta = json.loads((dest / "meta.json").read_text())
    assert "calibration_package_id" not in meta # omitted, not written as null


def test_load_throw_package_backward_compatible_with_no_calibration_package_id_key(tmp_path):
    """A package saved before this field existed (no key at all in
    meta.json) must still load fine."""
    bg = {0: np.zeros((4, 4, 3), np.uint8)}
    frame = {0: np.full((4, 4, 3), 5, np.uint8)}
    calibrations = {0: _fake_calibration()}
    dest = save_throw_package(
        tmp_path / "throw1", "sess1", bg, frame, calibrations, _fake_score_result(),
    )
    meta_path = dest / "meta.json"
    meta = json.loads(meta_path.read_text())
    assert "calibration_package_id" not in meta # pre-existing-package simulation
    pkg = load_throw_package(dest)
    assert pkg.calibration_package_id is None


def test_load_throw_package_accepts_opendarts_frame_cameras_key(tmp_path):
    """Real gap found 2026-08-21/22 (cross-session engine cross-validation
    work): OpenDarts (a strict port of this project's own package format)
    names the same field `frame_cameras`, not `cameras` -- a historical
    naming accident across the two independently-built projects. This
    loader must accept an OpenDarts-shaped package, not just this
    project's own convention."""
    bg = {0: np.zeros((4, 4, 3), np.uint8)}
    frame = {0: np.full((4, 4, 3), 5, np.uint8)}
    calibrations = {0: _fake_calibration()}
    dest = save_throw_package(
        tmp_path / "throw1", "sess1", bg, frame, calibrations, _fake_score_result(),
    )
    meta_path = dest / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["frame_cameras"] = meta.pop("cameras") # simulate an OpenDarts-shaped meta.json
    meta_path.write_text(json.dumps(meta))

    pkg = load_throw_package(dest)
    assert list(pkg.bg_frames.keys()) == [0]
    assert list(pkg.dart_frames.keys()) == [0]


def test_load_throw_package_prefers_cameras_when_both_keys_present(tmp_path):
    """If a package somehow has both keys, this project's own `cameras`
    convention wins -- but only when those PNGs actually exist. An empty
    or conflicting `frame_cameras` must not steal a complete `cameras`
    package (the original 2026-08-21/22 preference).
    """
    bg = {0: np.zeros((4, 4, 3), np.uint8)}
    frame = {0: np.full((4, 4, 3), 5, np.uint8)}
    calibrations = {0: _fake_calibration()}
    dest = save_throw_package(
        tmp_path / "throw1", "sess1", bg, frame, calibrations, _fake_score_result(),
    )
    meta_path = dest / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["frame_cameras"] = [] # a conflicting, wrong value under the fallback key
    meta_path.write_text(json.dumps(meta))

    pkg = load_throw_package(dest)
    assert list(pkg.bg_frames.keys()) == [0] # used "cameras", not the empty "frame_cameras"


def test_load_throw_package_falls_back_to_frame_cameras_when_a_cameras_png_is_missing(
    tmp_path,
):
    """Some writers put cameras=[0,1,2] even when a camera dropped this throw;
    the PNGs that exist are `frame_cameras` (the recorded S1 throw).
    Preferring `cameras` then raising IOError is wrong for a valid 2-cam
    package. Retry with `frame_cameras` only when a `cameras` PNG is
    actually missing.
    """
    bg = {0: np.zeros((4, 4, 3), np.uint8)}
    frame = {0: np.full((4, 4, 3), 5, np.uint8)}
    calibrations = {0: _fake_calibration()}
    dest = save_throw_package(
        tmp_path / "throw1", "sess1", bg, frame, calibrations, _fake_score_result(),
    )
    meta_path = dest / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["cameras"] = [0, 1]
    meta["frame_cameras"] = [0]
    meta_path.write_text(json.dumps(meta))

    pkg = load_throw_package(dest)
    assert list(pkg.bg_frames.keys()) == [0]
    assert list(pkg.dart_frames.keys()) == [0]


def test_load_throw_package_raises_loud_with_neither_camera_key(tmp_path):
    """Neither `cameras` nor `frame_cameras` present is a real corruption/
    incompatibility case -- must raise loud (a clear KeyError), never
    silently proceed with an empty frame set.

    the v2 package schema (2026-08-26): save_throw_package() now ALWAYS
    writes `frame_cameras` too (previously a key opendarts never
    populated) -- deleting only `cameras` is no longer enough to
    reproduce "neither key present" against a package this project's own
    writer produced, so this fixture now strips both, matching what a
    genuinely pre-existing/corrupted package would actually look
    like."""
    bg = {0: np.zeros((4, 4, 3), np.uint8)}
    frame = {0: np.full((4, 4, 3), 5, np.uint8)}
    calibrations = {0: _fake_calibration()}
    dest = save_throw_package(
        tmp_path / "throw1", "sess1", bg, frame, calibrations, _fake_score_result(),
    )
    meta_path = dest / "meta.json"
    meta = json.loads(meta_path.read_text())
    del meta["cameras"]
    del meta["frame_cameras"]
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(KeyError, match="neither 'cameras' nor 'frame_cameras'"):
        load_throw_package(dest)


# ---------------------------------------------------------------------------
# 5. Real end-to-end: solved calibration is bit-identical with vs without
# package saving, over real archived frames.
# ---------------------------------------------------------------------------


class _Status:
    def __init__(self):
        self.frame_count = 0


class _ReplayHub:
    """Mirrors tests/test_capture_daemon_oriented_wiring.py's own
    ReplayHub -- feeds real archived bg frames through the exact same
    hub contract (configs/status[i].frame_count/grab_all()) bootstrap_
    calibrations() uses live, nothing inside the function under test is
    patched."""

    def __init__(self, frames):
        self.frames = frames
        self.configs = [object()] * (max(frames) + 1)
        self.status = {i: _Status() for i in range(len(self.configs))}
        self.round = 0

    def grab_all(self):
        out = {}
        for cam, fr in self.frames.items():
            out[cam] = fr[min(self.round, len(fr) - 1)]
            self.status[cam].frame_count += 1
        self.round += 1
        return out


@pytest.mark.slow
@pytest.mark.skipif(not HAS_REAL_DATA, reason="data/archive/ not present (gitignored, 1.9GB)")
def test_real_calibration_result_bit_identical_with_and_without_package_saving(tmp_path):
    """The real, load-bearing claim for this feature: adding calibration-
    package saving alongside bootstrap_calibrations() must not change the
    solved calibration by so much as one bit, on real archived frames,
    through the real unmocked function."""
    packages = sorted(p.parent for p in ARCHIVE_SESSION.glob("*/result.json"))
    assert packages, ARCHIVE_SESSION
    frames_by_cam = {cam: [] for cam in (0, 1, 2)}
    for package in packages:
        for cam in (0, 1, 2):
            image = cv2.imread(str(package / f"cam{cam}_bg.png"))
            if image is not None:
                frames_by_cam[cam].append(image)
    n_frames = max(len(v) for v in frames_by_cam.values())

    baseline = capture_daemon.bootstrap_calibrations(
        tmp_path / "baseline", hub=_ReplayHub(frames_by_cam), n_frames=n_frames,
    )
    pkg_out: dict = {}
    with_pkg = capture_daemon.bootstrap_calibrations(
        tmp_path / "withpkg", hub=_ReplayHub(frames_by_cam), n_frames=n_frames,
        calibration_package_root=tmp_path / "calibration_packages",
        calibration_package_out=pkg_out,
        calibration_package_blocking=True,
    )

    assert sorted(baseline) == sorted(with_pkg) == [0, 1, 2]
    for cam in baseline:
        a, b = baseline[cam], with_pkg[cam]
        assert np.array_equal(a.camera_matrix, b.camera_matrix), f"cam{cam} camera_matrix differs"
        assert np.array_equal(a.dist_coeffs, b.dist_coeffs), f"cam{cam} dist_coeffs differs"
        assert np.array_equal(a.rvec, b.rvec), f"cam{cam} rvec differs"
        assert np.array_equal(a.tvec, b.tvec), f"cam{cam} tvec differs"
        assert a.landmark_spread_ok == b.landmark_spread_ok

    # And the saved package's own raw frames decode back byte-exact
    # against the SAME real archived frames that were fed in.
    loaded = calib_pkg.load_calibration_package(pkg_out["package_dir"])
    for cam, raw in loaded.raw_frames.items():
        original_cycle = frames_by_cam[cam]
        for i, f in enumerate(raw):
            expected = original_cycle[min(i, len(original_cycle) - 1)]
            assert np.array_equal(f, expected), f"cam{cam} frame {i} not byte-exact"


# ---------------------------------------------------------------------------
# 6. REPLAY PERSISTENCE for the live-derived ring-boundary-offset/
# board-color values (2026-08-22) -- a real, confirmed bug:
# bootstrap_calibrations() applied both as in-memory global state for
# LIVE scoring but never persisted them anywhere a later, fresh-process
# REPLAY could find them, so replaying an old package silently fell
# back to the regulation-default constants instead of reproducing what
# actually scored the throw live. See docs/DESIGN.md's "Replay is the
# source of truth".
# ---------------------------------------------------------------------------

COLOR_DIAGNOSTICS_ACCEPTED = {
    "brightness_threshold": 140.5,
    "brightness_threshold_confidence": "high",
    "chroma_threshold": 30.2,
    "chroma_threshold_confidence": "high",
}

# ---------------------------------------------------------------------------
# FULL-BOUNDARY STORAGE GAP FIX, 2026-08-26. Before this, `RING_DIAGNOSTICS_ACCEPTED` above
# was the WHOLE of what a calibration package ever recorded about the
# ring-boundary-offset measurement: the two INNER boundaries' offset_mm/
# confidence/n_samples_used only -- never treble_outer/double_outer, never
# any boundary's mad_mm/n_samples_rejected/n_angles_attempted/per_camera
# breakdown. This fixture is realistic (not degenerate all-None): three
# cameras' worth of per_camera data on EVERY boundary, both inner AND
# outer boundaries populated, matching the exact shape
# `opendarts.calibration.ring_boundary_offset.boundary_measurement_to_payload()`
# produces from a real `BoundaryMeasurement` -- the same shape the offline
# session-level `ring_boundary_offset.json` has always written per
# boundary (see that module's own `result_to_payload()`).
# ---------------------------------------------------------------------------

REALISTIC_RING_BOUNDARY_MEASUREMENT = {
    "treble_inner": {
        "regulation_radius_mm": 99.0,
        "measured_radius_mm": 97.8979,
        "offset_mm": 1.1021048558186521,
        "mad_mm": 0.18,
        "n_samples_used": 40,
        "n_samples_rejected": 3,
        "n_angles_attempted": 20,
        "per_camera": {
            "0": {"radius_mm": 97.91, "n_profiles": 14, "mode": "bed_adjacent_peak"},
            "1": {"radius_mm": 97.85, "n_profiles": 13, "mode": "bed_adjacent_peak"},
            "2": {"radius_mm": 97.93, "n_profiles": 13, "mode": "bed_adjacent_peak"},
        },
        "confidence": 0.93,
    },
    "treble_outer": {
        "regulation_radius_mm": 107.0,
        "measured_radius_mm": 106.42,
        "offset_mm": 0.58,
        "mad_mm": 0.22,
        "n_samples_used": 38,
        "n_samples_rejected": 5,
        "n_angles_attempted": 20,
        "per_camera": {
            "0": {"radius_mm": 106.5, "n_profiles": 13, "mode": "bed_adjacent_peak"},
            "1": {"radius_mm": 106.38, "n_profiles": 12, "mode": "bed_adjacent_peak"},
            "2": {"radius_mm": 106.39, "n_profiles": 13, "mode": "bed_adjacent_peak"},
        },
        "confidence": 0.89,
    },
    "double_inner": {
        "regulation_radius_mm": 162.0,
        "measured_radius_mm": 160.13,
        "offset_mm": 1.87,
        "mad_mm": 0.15,
        "n_samples_used": 42,
        "n_samples_rejected": 2,
        "n_angles_attempted": 20,
        "per_camera": {
            "0": {"radius_mm": 160.09, "n_profiles": 14, "mode": "bed_adjacent_peak"},
            "1": {"radius_mm": 160.18, "n_profiles": 14, "mode": "bed_adjacent_peak"},
            "2": {"radius_mm": 160.12, "n_profiles": 14, "mode": "bed_adjacent_peak"},
        },
        "confidence": 0.88,
    },
    "double_outer": {
        "regulation_radius_mm": 170.0,
        "measured_radius_mm": 169.24,
        "offset_mm": 0.76,
        "mad_mm": 0.31,
        "n_samples_used": 35,
        "n_samples_rejected": 8,
        "n_angles_attempted": 20,
        "per_camera": {
            "0": {"radius_mm": 169.3, "n_profiles": 12, "mode": "median_crossing"},
            "1": {"radius_mm": 169.15, "n_profiles": 11, "mode": "median_crossing"},
            "2": {"radius_mm": 169.28, "n_profiles": 12, "mode": "median_crossing"},
        },
        "confidence": 0.81,
    },
}


# ---------------------------------------------------------------------------
# THE V2 PACKAGE SCHEMA (2026-08-27) -- NEW nested-payload
# diagnostics-input fixtures, matching EXACTLY what
# `opendarts.calibration.ring_boundary_offset.result_to_payload()`/
# `opendarts.geometry.board_color_calibration.result_to_payload()` really
# produce (the same shape `bootstrap_calibrations()` now feeds into
# `save_calibration_package()` as `diagnostics[cam]["ring_boundary_
# offset_payload"]`/`["board_color_calibration_payload"]`). Real writer-
# facing tests below use these, not the OLD flat fixtures above.
# ---------------------------------------------------------------------------

RING_BOUNDARY_OFFSET_PAYLOAD_ACCEPTED = {
    "schema": "ring-boundary-offset-v2",
    "solved_by": (
        "opendarts.calibration.ring_boundary_offset "
        "(live, from this calibration event's own captured frames)"
    ),
    "calibration_source": "live calibration burst (this event's own solved pose)",
    "source_images": [],
    "parameters": {
        "radial_step_mm": 0.1,
        "profile_smooth_samples": 5,
        "min_contrast": 40.0,
        "min_radial_scale_px_per_mm": 0.35,
        "in_sector_angle_offsets_deg": [-4.0, 0.0, 4.0],
        "transition_lo": 0.08,
        "transition_hi": 0.92,
        "median_crossing_sustain_samples": 5,
        "peak_significance_fraction": 0.5,
        "peak_contiguity_stop_fraction": 0.2,
        "min_profiles_per_camera": 10,
    },
    "boundaries": REALISTIC_RING_BOUNDARY_MEASUREMENT,
    "median_profiles": [],
}

BOARD_COLOR_CALIBRATION_PAYLOAD_ACCEPTED = {
    "schema": "board-color-calibration-v1",
    "solved_by": (
        "opendarts.geometry.board_color_calibration "
        "(live, from this calibration event's own captured frames)"
    ),
    "brightness_threshold_black_cream": {
        "value": 140.5, "low_group_stat": 60.0, "high_group_stat": 221.0,
        "low_group_n": 40, "high_group_n": 40, "gap": 161.0, "confidence": "high",
    },
    "chroma_threshold": {
        "value": 30.2, "low_group_stat": 12.0, "high_group_stat": 48.4,
        "low_group_n": 40, "high_group_n": 40, "gap": 36.4, "confidence": "high",
    },
    "patch_radius_px": 6,
    "n_samples": 328,
    "n_packages": 1,
    "n_packages_attempted": 1,
    "accuracy_single_camera": 0.98,
    "accuracy_majority_vote": 1.0,
    "warnings": [],
}


def test_boundary_measurement_to_payload_matches_fixture_shape():
    """`boundary_measurement_to_payload()` (the real function
    `bootstrap_calibrations()` now calls) must produce EXACTLY the field
    set the fixture above assumes -- a real `BoundaryMeasurement`
    dataclass in, the fixture's own shape out, so the fixture above isn't
    silently drifting from what the real code actually produces."""
    from opendarts.calibration.ring_boundary_offset import (
        BoundaryMeasurement,
        boundary_measurement_to_payload,
    )

    m = BoundaryMeasurement(
        boundary="treble_inner",
        regulation_radius_mm=99.0,
        measured_radius_mm=97.8979,
        offset_mm=1.1021048558186521,
        mad_mm=0.18,
        n_samples_used=40,
        n_samples_rejected=3,
        n_angles_attempted=20,
        per_camera={
            0: {"radius_mm": 97.91, "n_profiles": 14, "mode": "bed_adjacent_peak"},
        },
        confidence=0.93,
    )
    payload = boundary_measurement_to_payload(m)
    assert payload == {
        "regulation_radius_mm": 99.0,
        "measured_radius_mm": 97.8979,
        "offset_mm": 1.1021048558186521,
        "mad_mm": 0.18,
        "n_samples_used": 40,
        "n_samples_rejected": 3,
        "n_angles_attempted": 20,
        "per_camera": {"0": {"radius_mm": 97.91, "n_profiles": 14, "mode": "bed_adjacent_peak"}},
        "confidence": 0.93,
    }


def test_save_calibration_package_records_full_ring_boundary_measurement(tmp_path):
    """The original 2026-08-26 fix under test (still real after the
    2026-08-27 reshape, just relocated): ALL 4 boundaries' full data (not
    just the 2 inner boundaries' offset/confidence/n_samples_used) must
    round-trip through save_calibration_package() into
    derived_calibration.json, including treble_outer/double_outer (never
    stored before the 2026-08-26 fix) and every boundary's mad_mm/
    n_samples_rejected/n_angles_attempted/per_camera breakdown -- now
    living under the nested `ring_boundary_offset.boundaries` key (see
    calibration_package.py's own "REPLACE FOR NEW WRITES" comment) rather
    than the old flat `ring_boundary_measurement` key, which a NEW save no
    longer writes at all."""
    diagnostics = {
        0: {
            "ring_boundary_offset_accepted": True,
            "ring_boundary_offset_payload": RING_BOUNDARY_OFFSET_PAYLOAD_ACCEPTED,
            "board_color_calibration_payload": BOARD_COLOR_CALIBRATION_PAYLOAD_ACCEPTED,
        },
        1: {
            "ring_boundary_offset_accepted": True,
            "ring_boundary_offset_payload": RING_BOUNDARY_OFFSET_PAYLOAD_ACCEPTED,
            "board_color_calibration_payload": BOARD_COLOR_CALIBRATION_PAYLOAD_ACCEPTED,
        },
    }
    package_dir = calib_pkg.save_calibration_package(
        tmp_path, "calib_full_ring", {0: [], 1: []},
        {0: _fake_calibration(), 1: _fake_calibration()}, diagnostics,
    )
    derived = json.loads((package_dir / "derived_calibration.json").read_text())

    # The one flat legacy key kept (see calibration_package.py's own
    # comment for why): still there, unchanged.
    assert derived["ring_boundary_offset_accepted"] is True
    # Every OTHER old flat key is now GONE from a fresh save -- a real
    # regression guard for the reshape itself, not just a positive check.
    for old_flat_key in (
        "treble_inner_offset_mm", "treble_inner_confidence", "treble_inner_n_samples_used",
        "double_inner_offset_mm", "double_inner_confidence", "double_inner_n_samples_used",
        "ring_boundary_measurement",
        "brightness_threshold", "brightness_threshold_confidence",
        "chroma_threshold", "chroma_threshold_confidence",
    ):
        assert old_flat_key not in derived, f"{old_flat_key} should no longer be written"

    # The new nested key: all 4 boundaries, byte-for-byte.
    measurement = derived["ring_boundary_offset"]["boundaries"]
    assert set(measurement.keys()) == {
        "treble_inner", "treble_outer", "double_inner", "double_outer",
    }
    assert measurement == REALISTIC_RING_BOUNDARY_MEASUREMENT
    # Specifically confirm the previously-discarded fields survive:
    # treble_outer/double_outer at all, and mad_mm/n_samples_rejected/
    # n_angles_attempted/per_camera on every boundary (not just the two
    # inner ones).
    assert measurement["treble_outer"]["offset_mm"] == 0.58
    assert measurement["double_outer"]["offset_mm"] == 0.76
    for name in ("treble_inner", "treble_outer", "double_inner", "double_outer"):
        b = measurement[name]
        assert b["mad_mm"] is not None
        assert b["n_samples_rejected"] > 0
        assert b["n_angles_attempted"] == 20
        assert set(b["per_camera"].keys()) == {"0", "1", "2"}
        for cam_entry in b["per_camera"].values():
            assert "radius_mm" in cam_entry and "n_profiles" in cam_entry and "mode" in cam_entry

    # Also has its own solved_by/parameters/code_version -- the fields
    # gap #1's own fix added.
    assert derived["ring_boundary_offset"]["parameters"]["radial_step_mm"] == 0.1
    assert "code_version" in derived["ring_boundary_offset"]
    assert derived["board_color_calibration"]["brightness_threshold_black_cream"]["value"] == 140.5

    # Package-wide, not duplicated per-camera.
    assert "ring_boundary_measurement" not in derived["cameras"]["0"]
    assert "ring_boundary_offset" not in derived["cameras"]["0"]


def test_load_calibration_package_derived_values_round_trips_full_ring_boundary_measurement(
    tmp_path,
):
    calib_pkg.save_calibration_package(
        tmp_path, "calib_full_ring_roundtrip", {0: []}, {0: _fake_calibration()},
        {0: {
            "ring_boundary_offset_accepted": True,
            "ring_boundary_offset_payload": RING_BOUNDARY_OFFSET_PAYLOAD_ACCEPTED,
        }},
    )
    values = calib_pkg.load_calibration_package_derived_values(
        "calib_full_ring_roundtrip", search_roots=(tmp_path,),
    )
    assert values is not None
    assert values["ring_boundary_offset"]["boundaries"] == REALISTIC_RING_BOUNDARY_MEASUREMENT


def test_load_calibration_package_derived_values_old_format_package_missing_new_field(tmp_path):
    """REPLAY tolerance (docs/DESIGN.md's "Replay is the source of truth"): a package
    written BEFORE this fix has no `ring_boundary_measurement` key at all
    (not even null) in its on-disk derived_calibration.json -- loading it
    back must not raise, and every OLD field must still read exactly as
    before."""
    pkg_dir = tmp_path / "calib_old_format"
    pkg_dir.mkdir()
    (pkg_dir / "meta.json").write_text(json.dumps({
        "schema": calib_pkg.CALIBRATION_PACKAGE_SCHEMA,
        "package_id": "calib_old_format",
        "created_at_utc": "2026-08-20T00:00:00+00:00",
        "codec": "ffv1", "pix_fmt": "bgr24",
        "cameras": {},
    }))
    old_format_derived = {
        "schema": calib_pkg.CALIBRATION_PACKAGE_DERIVED_SCHEMA,
        "total_duration_s": 12.3,
        "treble_inner_offset_mm": 1.1021048558186521,
        "treble_inner_confidence": 0.93,
        "treble_inner_n_samples_used": 40,
        "double_inner_offset_mm": 1.87,
        "double_inner_confidence": 0.88,
        "double_inner_n_samples_used": 42,
        "ring_boundary_offset_accepted": True,
        # deliberately NO "ring_boundary_measurement" key -- this is what
        # every real package saved before 2026-08-26 actually looks like
        # on disk.
        "brightness_threshold": 140.5,
        "brightness_threshold_confidence": "high",
        "chroma_threshold": 30.2,
        "chroma_threshold_confidence": "high",
        "cameras": {"0": {**calib_pkg.calibration_to_dict(_fake_calibration())}},
    }
    (pkg_dir / "derived_calibration.json").write_text(json.dumps(old_format_derived))

    values = calib_pkg.load_calibration_package_derived_values(
        "calib_old_format", search_roots=(tmp_path,),
    )
    assert values is not None
    assert values["treble_inner_offset_mm"] == 1.1021048558186521
    assert values.get("ring_boundary_measurement") is None # absent, not an error
    # And the NEW nested key (added 2026-08-27) is likewise simply absent
    # on a genuinely old-format file -- never an error, never a
    # fabricated empty block.
    assert values.get("ring_boundary_offset") is None
    assert values.get("board_color_calibration") is None

    loaded = calib_pkg.load_calibration_package(pkg_dir, decode_raw_frames=False)
    assert loaded.package_id == "calib_old_format"


def test_save_calibration_package_records_ring_and_color_derived_values(tmp_path):
    """The actual measured VALUES (not just their timing) get pulled up
    to derived_calibration.json's own top level (`ring_boundary_offset_
    accepted`) or its own nested blocks (`ring_boundary_offset`/
    `board_color_calibration`), package-wide -- same "pulled up, not
    duplicated per camera" treatment as section_timing."""
    diagnostics = {
        0: {
            "ring_boundary_offset_accepted": True,
            "ring_boundary_offset_payload": RING_BOUNDARY_OFFSET_PAYLOAD_ACCEPTED,
            "board_color_calibration_payload": BOARD_COLOR_CALIBRATION_PAYLOAD_ACCEPTED,
        },
        1: {
            "ring_boundary_offset_accepted": True,
            "ring_boundary_offset_payload": RING_BOUNDARY_OFFSET_PAYLOAD_ACCEPTED,
            "board_color_calibration_payload": BOARD_COLOR_CALIBRATION_PAYLOAD_ACCEPTED,
        },
    }
    package_dir = calib_pkg.save_calibration_package(
        tmp_path, "calib_ring_color", {0: [], 1: []},
        {0: _fake_calibration(), 1: _fake_calibration()}, diagnostics,
    )
    derived = json.loads((package_dir / "derived_calibration.json").read_text())
    assert derived["ring_boundary_offset_accepted"] is True
    boundaries = derived["ring_boundary_offset"]["boundaries"]
    assert boundaries["treble_inner"]["offset_mm"] == 1.1021048558186521
    assert boundaries["treble_inner"]["confidence"] == 0.93
    assert boundaries["treble_inner"]["n_samples_used"] == 40
    assert boundaries["double_inner"]["offset_mm"] == 1.87
    assert boundaries["double_inner"]["confidence"] == 0.88
    assert boundaries["double_inner"]["n_samples_used"] == 42
    color = derived["board_color_calibration"]
    assert color["brightness_threshold_black_cream"]["value"] == 140.5
    assert color["brightness_threshold_black_cream"]["confidence"] == "high"
    assert color["chroma_threshold"]["value"] == 30.2
    assert color["chroma_threshold"]["confidence"] == "high"
    # Package-wide, not duplicated per-camera.
    assert "ring_boundary_offset" not in derived["cameras"]["0"]
    assert "board_color_calibration" not in derived["cameras"]["0"]


def test_save_calibration_package_ring_and_color_absent_when_no_diagnostics(tmp_path):
    """Same "diagnostics-empty package still writes validly" guarantee as
    the timing fields -- absent (None/missing), never a fabricated
    value."""
    package_dir = calib_pkg.save_calibration_package(
        tmp_path, "calib_no_ring_color", {0: []}, {0: _fake_calibration()},
    )
    derived = json.loads((package_dir / "derived_calibration.json").read_text())
    assert derived["ring_boundary_offset_accepted"] is None
    # The nested blocks are omitted entirely (not present-but-null) when
    # the measurement never ran this event -- the real shape is no key
    # at all, not `"ring_boundary_offset": null`.
    assert "ring_boundary_offset" not in derived
    assert "board_color_calibration" not in derived
    # And every old flat key is simply gone, not present-and-None.
    for old_flat_key in (
        "treble_inner_offset_mm", "double_inner_offset_mm",
        "brightness_threshold", "chroma_threshold",
    ):
        assert old_flat_key not in derived


def test_save_calibration_package_ring_and_color_rejected_values_stay_none(tmp_path):
    """A boundary/threshold REJECTED live (confidence too low / no
    measurement) must persist `offset_mm`/`value` as an absent value, not
    a fabricated number -- even though its confidence/n_samples_used ARE
    recorded (real diagnostic information, distinct from "was it
    adopted"). Matches the real live shape: `bootstrap_calibrations()`
    still populates the full `ring_boundary_offset_payload`/
    `board_color_calibration_payload` whenever the underlying measurement
    RAN (even if its result was too weak to adopt) -- only
    `ring_boundary_offset_accepted` (and each threshold's own
    `confidence`) records the accept/reject decision."""
    rejected_ring_payload = {
        **RING_BOUNDARY_OFFSET_PAYLOAD_ACCEPTED,
        "boundaries": {
            **REALISTIC_RING_BOUNDARY_MEASUREMENT,
            "treble_inner": {
                **REALISTIC_RING_BOUNDARY_MEASUREMENT["treble_inner"],
                "measured_radius_mm": None, "offset_mm": None, "confidence": 0.1,
            },
            "double_inner": {
                **REALISTIC_RING_BOUNDARY_MEASUREMENT["double_inner"],
                "measured_radius_mm": None, "offset_mm": None, "confidence": 0.05,
            },
        },
    }
    rejected_color_payload = {
        **BOARD_COLOR_CALIBRATION_PAYLOAD_ACCEPTED,
        "brightness_threshold_black_cream": {
            **BOARD_COLOR_CALIBRATION_PAYLOAD_ACCEPTED["brightness_threshold_black_cream"],
            "value": None, "confidence": "low",
        },
        "chroma_threshold": {
            **BOARD_COLOR_CALIBRATION_PAYLOAD_ACCEPTED["chroma_threshold"],
            "value": None, "confidence": "insufficient_data",
        },
    }
    diagnostics = {
        0: {
            "ring_boundary_offset_accepted": False,
            "ring_boundary_offset_payload": rejected_ring_payload,
            "board_color_calibration_payload": rejected_color_payload,
        },
    }
    package_dir = calib_pkg.save_calibration_package(
        tmp_path, "calib_rejected", {0: []}, {0: _fake_calibration()}, diagnostics,
    )
    derived = json.loads((package_dir / "derived_calibration.json").read_text())
    assert derived["ring_boundary_offset_accepted"] is False
    boundaries = derived["ring_boundary_offset"]["boundaries"]
    assert boundaries["treble_inner"]["offset_mm"] is None
    assert boundaries["treble_inner"]["confidence"] == 0.1 # diagnostic kept even though rejected
    color = derived["board_color_calibration"]
    assert color["brightness_threshold_black_cream"]["value"] is None
    assert color["brightness_threshold_black_cream"]["confidence"] == "low"

    # And this rejected package's own reader-facing helpers correctly
    # resolve to "nothing to apply" -- the real point of "rejected."
    from opendarts.capture.throw_package import (
        _package_board_color_threshold_value,
        _package_ring_boundary_offsets,
    )
    values = calib_pkg.load_calibration_package_derived_values(
        "calib_rejected", search_roots=(tmp_path,),
    )
    assert _package_ring_boundary_offsets(values) == (None, None)
    assert _package_board_color_threshold_value(
        values, "brightness_threshold", "brightness_threshold_black_cream",
    ) is None


def test_find_calibration_package_dir_prefers_live_root_then_archived_root(tmp_path):
    live_root = tmp_path / "live"
    archived_root = tmp_path / "archived"
    calib_pkg.save_calibration_package(live_root, "calib_both", {0: []}, {0: _fake_calibration()})
    calib_pkg.save_calibration_package(archived_root, "calib_both", {0: []}, {0: _fake_calibration()})
    calib_pkg.save_calibration_package(
        archived_root, "calib_archived_only", {0: []}, {0: _fake_calibration()},
    )

    found_both = calib_pkg.find_calibration_package_dir(
        "calib_both", search_roots=(live_root, archived_root),
    )
    assert found_both == live_root / "calib_both" # live wins when present in both

    found_archived_only = calib_pkg.find_calibration_package_dir(
        "calib_archived_only", search_roots=(live_root, archived_root),
    )
    assert found_archived_only == archived_root / "calib_archived_only"

    assert calib_pkg.find_calibration_package_dir(
        "calib_nowhere", search_roots=(live_root, archived_root),
    ) is None


def test_load_calibration_package_derived_values_returns_none_when_not_found(tmp_path):
    assert calib_pkg.load_calibration_package_derived_values(
        "calib_missing", search_roots=(tmp_path / "nope",),
    ) is None


def test_load_calibration_package_derived_values_returns_none_on_corrupt_json(tmp_path):
    pkg_dir = tmp_path / "calib_corrupt"
    pkg_dir.mkdir()
    (pkg_dir / "derived_calibration.json").write_text("{not valid json")
    assert calib_pkg.load_calibration_package_derived_values(
        "calib_corrupt", search_roots=(tmp_path,),
    ) is None


def test_load_calibration_package_derived_values_round_trips_real_values(tmp_path):
    calib_pkg.save_calibration_package(
        tmp_path, "calib_roundtrip", {0: []}, {0: _fake_calibration()},
        {0: {
            "ring_boundary_offset_accepted": True,
            "ring_boundary_offset_payload": RING_BOUNDARY_OFFSET_PAYLOAD_ACCEPTED,
        }},
    )
    values = calib_pkg.load_calibration_package_derived_values(
        "calib_roundtrip", search_roots=(tmp_path,),
    )
    assert values is not None
    assert values["ring_boundary_offset"]["boundaries"]["treble_inner"]["offset_mm"] == (
        1.1021048558186521
    )


# ---------------------------------------------------------------------------
# load_throw_package() REPLAY wiring -- the real end-to-end fix.
# ---------------------------------------------------------------------------


def _write_throw_with_calibration_package_id(
    tmp_path: Path, calibration_package_id: str | None,
) -> Path:
    bg = {0: np.zeros((4, 4, 3), np.uint8)}
    frame = {0: np.full((4, 4, 3), 5, np.uint8)}
    calibrations = {0: _fake_calibration()}
    return save_throw_package(
        tmp_path / "session1" / "throw1", "session1", bg, frame, calibrations,
        _fake_score_result(), calibration_package_id=calibration_package_id,
    )


def test_load_throw_package_never_applies_live_measured_ring_boundary_offset_from_calibration_package(
    tmp_path, monkeypatch, caplog,
):
    """STOP APPLYING, 2026-09-02 (see docs/DESIGN.md's dated entry for
    the full decisive-test data).
    Renamed from `..._applies_live_measured_ring_boundary_offset_...`,
    which asserted the OLD (now-wrong) behavior. A throw whose
    calibration package recorded a real live-measured offset must now
    score against the hardcoded INNER_RING_SCORING_OFFSET_MM default for
    BOTH inner wires, regardless of what was measured -- the measured
    value is still real and still recorded in the package (unchanged;
    see the calibration_package.py round-trip tests elsewhere in this
    file), and `load_throw_package()` still logs it (asserted here via
    `caplog`) so it's visible for research, but it must never reach the
    scoring radii."""
    import logging as _logging

    import opendarts.geometry.board as board_module

    default_treble = board_module.TREBLE_INNER_SCORING_RADIUS_MM
    default_double = board_module.DOUBLE_INNER_SCORING_RADIUS_MM

    live_root = tmp_path / "live_calib"
    monkeypatch.setattr(calib_pkg, "DEFAULT_CALIBRATION_PACKAGE_ROOT", live_root)
    monkeypatch.setattr(
        calib_pkg, "DEFAULT_ARCHIVED_CALIBRATION_PACKAGE_ROOT", tmp_path / "archived_calib_unused",
    )
    calib_pkg.save_calibration_package(
        live_root, "calib_live_ring", {0: []}, {0: _fake_calibration()},
        {0: {
            "ring_boundary_offset_accepted": True,
            "ring_boundary_offset_payload": RING_BOUNDARY_OFFSET_PAYLOAD_ACCEPTED,
        }},
    )
    dest = _write_throw_with_calibration_package_id(tmp_path, "calib_live_ring")

    with caplog.at_level(_logging.DEBUG, logger="opendarts.capture.throw_package"):
        load_throw_package(dest)

    # Scoring radii stayed at the hardcoded default -- unchanged by the
    # real, present, confident measurement this package carries.
    assert board_module.TREBLE_INNER_SCORING_RADIUS_MM == pytest.approx(default_treble)
    assert board_module.DOUBLE_INNER_SCORING_RADIUS_MM == pytest.approx(default_double)
    # But the measurement is still found and logged for visibility --
    # confirms the calibration-package-level lookup itself is intact,
    # just no longer wired to `set_ring_boundary_offsets()`.
    assert any(
        "1.1021048558186521" in rec.message and "not applied" in rec.message.lower()
        for rec in caplog.records
    )


def test_load_throw_package_ring_boundary_offset_rejected_in_package_still_uses_default(
    tmp_path, monkeypatch,
):
    """A calibration package with a REJECTED (None) ring-boundary offset
    -- and even a real session-level file present alongside it -- must
    STILL score against the hardcoded default. Renamed from
    `..._falls_back_to_session_level`: that fallback mechanism itself is
    gone (2026-09-02, see docs/DESIGN.md's dated entry) -- neither the
    package-level nor the session-level measurement is ever applied any
    more, so there is nothing left to "fall back" between; both are
    equally inert for scoring purposes."""
    import opendarts.geometry.board as board_module

    default_treble = board_module.TREBLE_INNER_SCORING_RADIUS_MM
    default_double = board_module.DOUBLE_INNER_SCORING_RADIUS_MM

    live_root = tmp_path / "live_calib"
    monkeypatch.setattr(calib_pkg, "DEFAULT_CALIBRATION_PACKAGE_ROOT", live_root)
    monkeypatch.setattr(
        calib_pkg, "DEFAULT_ARCHIVED_CALIBRATION_PACKAGE_ROOT", tmp_path / "archived_calib_unused",
    )
    rejected_diagnostics = {
        "treble_inner_offset_mm": None, "double_inner_offset_mm": None,
        "ring_boundary_offset_accepted": False,
    }
    calib_pkg.save_calibration_package(
        live_root, "calib_rejected_ring", {0: []}, {0: _fake_calibration()},
        {0: rejected_diagnostics},
    )
    dest = _write_throw_with_calibration_package_id(tmp_path, "calib_rejected_ring")

    session_dir = dest.parent
    session_offset_payload = {
        "schema": "ring-boundary-offset-v2",
        "boundaries": {
            "treble_inner": {"offset_mm": 2.5},
            "double_inner": {"offset_mm": 2.9},
        },
    }
    (session_dir / "ring_boundary_offset.json").write_text(json.dumps(session_offset_payload))

    load_throw_package(dest)

    assert board_module.TREBLE_INNER_SCORING_RADIUS_MM == pytest.approx(default_treble)
    assert board_module.DOUBLE_INNER_SCORING_RADIUS_MM == pytest.approx(default_double)


def test_load_throw_package_ring_boundary_offset_both_absent_falls_back_to_regulation_default(
    tmp_path, monkeypatch,
):
    import opendarts.geometry.board as board_module

    default_treble = board_module.TREBLE_INNER_SCORING_RADIUS_MM
    default_double = board_module.DOUBLE_INNER_SCORING_RADIUS_MM

    monkeypatch.setattr(calib_pkg, "DEFAULT_CALIBRATION_PACKAGE_ROOT", tmp_path / "live_unused")
    monkeypatch.setattr(
        calib_pkg, "DEFAULT_ARCHIVED_CALIBRATION_PACKAGE_ROOT", tmp_path / "archived_unused",
    )
    dest = _write_throw_with_calibration_package_id(tmp_path, "calib_does_not_exist")

    load_throw_package(dest)

    assert board_module.TREBLE_INNER_SCORING_RADIUS_MM == pytest.approx(default_treble)
    assert board_module.DOUBLE_INNER_SCORING_RADIUS_MM == pytest.approx(default_double)


def test_load_throw_package_calibration_package_id_absent_skips_lookup_and_uses_default(
    tmp_path, monkeypatch,
):
    """No `calibration_package_id` at all (the real, common case for
    every package saved before 2026-08-20 -- confirmed 125/185 real
    packages across this project's full archive) must skip the
    calibration-package-level lookup ENTIRELY (never call it, never
    error). Renamed from `..._and_uses_session_level`: the session-level
    file is no longer consulted at all for ring-boundary application
    (2026-09-02, see docs/DESIGN.md's dated entry) -- the throw always scores
    against the hardcoded default regardless of what either file says."""
    import opendarts.geometry.board as board_module

    default_treble = board_module.TREBLE_INNER_SCORING_RADIUS_MM
    default_double = board_module.DOUBLE_INNER_SCORING_RADIUS_MM

    calls: list[str] = []
    real_loader = calib_pkg.load_calibration_package_derived_values

    def _spy(package_id, **kwargs):
        calls.append(package_id)
        return real_loader(package_id, **kwargs)

    monkeypatch.setattr(calib_pkg, "load_calibration_package_derived_values", _spy)

    dest = _write_throw_with_calibration_package_id(tmp_path, None)
    meta_path = dest / "meta.json"
    meta = json.loads(meta_path.read_text())
    assert "calibration_package_id" not in meta

    session_dir = dest.parent
    session_offset_payload = {
        "schema": "ring-boundary-offset-v2",
        "boundaries": {
            "treble_inner": {"offset_mm": 3.1},
            "double_inner": {"offset_mm": 3.3},
        },
    }
    (session_dir / "ring_boundary_offset.json").write_text(json.dumps(session_offset_payload))

    load_throw_package(dest)

    assert calls == [] # the calibration-package-level lookup was never invoked
    assert board_module.TREBLE_INNER_SCORING_RADIUS_MM == pytest.approx(default_treble)
    assert board_module.DOUBLE_INNER_SCORING_RADIUS_MM == pytest.approx(default_double)


def test_load_throw_package_applies_live_measured_board_color_thresholds_from_calibration_package(
    tmp_path, monkeypatch,
):
    import opendarts.geometry.board_color as board_color_module

    live_root = tmp_path / "live_calib"
    monkeypatch.setattr(calib_pkg, "DEFAULT_CALIBRATION_PACKAGE_ROOT", live_root)
    monkeypatch.setattr(
        calib_pkg, "DEFAULT_ARCHIVED_CALIBRATION_PACKAGE_ROOT", tmp_path / "archived_calib_unused",
    )
    calib_pkg.save_calibration_package(
        live_root, "calib_live_color", {0: []}, {0: _fake_calibration()},
        {0: {"board_color_calibration_payload": BOARD_COLOR_CALIBRATION_PAYLOAD_ACCEPTED}},
    )
    dest = _write_throw_with_calibration_package_id(tmp_path, "calib_live_color")

    load_throw_package(dest)

    assert board_color_module.BRIGHTNESS_THRESHOLD_BLACK_CREAM == pytest.approx(140.5)
    assert board_color_module.CHROMA_THRESHOLD == pytest.approx(30.2)


def test_load_throw_package_board_color_per_threshold_falls_back_independently(
    tmp_path, monkeypatch,
):
    """Brightness accepted live, chroma rejected -- chroma alone must
    fall through to the session-level file (if any) while brightness
    keeps its package-level value; this is the existing per-threshold
    discipline `load_throw_package()` already applies to the session-
    level file, extended one layer up."""
    import opendarts.geometry.board_color as board_color_module

    live_root = tmp_path / "live_calib"
    monkeypatch.setattr(calib_pkg, "DEFAULT_CALIBRATION_PACKAGE_ROOT", live_root)
    monkeypatch.setattr(
        calib_pkg, "DEFAULT_ARCHIVED_CALIBRATION_PACKAGE_ROOT", tmp_path / "archived_calib_unused",
    )
    mixed_payload = {
        **BOARD_COLOR_CALIBRATION_PAYLOAD_ACCEPTED,
        "brightness_threshold_black_cream": {
            **BOARD_COLOR_CALIBRATION_PAYLOAD_ACCEPTED["brightness_threshold_black_cream"],
            "value": 133.0, "confidence": "high",
        },
        "chroma_threshold": {
            **BOARD_COLOR_CALIBRATION_PAYLOAD_ACCEPTED["chroma_threshold"],
            "value": None, "confidence": "low",
        },
    }
    calib_pkg.save_calibration_package(
        live_root, "calib_mixed_color", {0: []}, {0: _fake_calibration()},
        {0: {"board_color_calibration_payload": mixed_payload}},
    )
    dest = _write_throw_with_calibration_package_id(tmp_path, "calib_mixed_color")

    session_dir = dest.parent
    session_color_payload = {
        "schema": "board-color-calibration-v1",
        "brightness_threshold_black_cream": {"value": 999.0, "confidence": "high"},
        "chroma_threshold": {"value": 44.4, "confidence": "high"},
    }
    (session_dir / "board_color_calibration.json").write_text(json.dumps(session_color_payload))

    load_throw_package(dest)

    # brightness: package-level value wins (999.0 from the session file is NOT used)
    assert board_color_module.BRIGHTNESS_THRESHOLD_BLACK_CREAM == pytest.approx(133.0)
    # chroma: package-level was rejected -> falls through to the session-level value
    assert board_color_module.CHROMA_THRESHOLD == pytest.approx(44.4)


def test_load_throw_package_board_color_both_absent_falls_back_to_regulation_default(
    tmp_path, monkeypatch,
):
    import opendarts.geometry.board_color as board_color_module

    default_brightness = board_color_module.BRIGHTNESS_THRESHOLD_BLACK_CREAM
    default_chroma = board_color_module.CHROMA_THRESHOLD

    monkeypatch.setattr(calib_pkg, "DEFAULT_CALIBRATION_PACKAGE_ROOT", tmp_path / "live_unused")
    monkeypatch.setattr(
        calib_pkg, "DEFAULT_ARCHIVED_CALIBRATION_PACKAGE_ROOT", tmp_path / "archived_unused",
    )
    dest = _write_throw_with_calibration_package_id(tmp_path, "calib_does_not_exist")

    load_throw_package(dest)

    assert board_color_module.BRIGHTNESS_THRESHOLD_BLACK_CREAM == pytest.approx(default_brightness)
    assert board_color_module.CHROMA_THRESHOLD == pytest.approx(default_chroma)


def test_load_throw_package_finds_calibration_package_in_archived_root_when_absent_from_live_root(
    tmp_path, monkeypatch, caplog,
):
    """A pulled/archived throw's calibration package lives under
    `data/archive/calibration_packages/`, not the live
    `data/calibration_packages/` root (where an off-rig pull
    archives it) -- the lookup must still find it (proven
    via the diagnostic log line, since 2026-09-02 the found value is no
    longer applied to scoring -- see docs/DESIGN.md's dated entry and the
    other tests in this file renamed the same way)."""
    import logging as _logging

    import opendarts.geometry.board as board_module

    default_treble = board_module.TREBLE_INNER_SCORING_RADIUS_MM

    live_root = tmp_path / "live_calib_empty"
    archived_root = tmp_path / "archived_calib"
    monkeypatch.setattr(calib_pkg, "DEFAULT_CALIBRATION_PACKAGE_ROOT", live_root)
    monkeypatch.setattr(calib_pkg, "DEFAULT_ARCHIVED_CALIBRATION_PACKAGE_ROOT", archived_root)
    calib_pkg.save_calibration_package(
        archived_root, "calib_archived_ring", {0: []}, {0: _fake_calibration()},
        {0: {
            "ring_boundary_offset_accepted": True,
            "ring_boundary_offset_payload": RING_BOUNDARY_OFFSET_PAYLOAD_ACCEPTED,
        }},
    )
    dest = _write_throw_with_calibration_package_id(tmp_path, "calib_archived_ring")

    with caplog.at_level(_logging.DEBUG, logger="opendarts.capture.throw_package"):
        load_throw_package(dest)

    # Never applied to scoring...
    assert board_module.TREBLE_INNER_SCORING_RADIUS_MM == pytest.approx(default_treble)
    # ...but the archived-root lookup itself still genuinely found it.
    assert any("1.1021048558186521" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# 7. THE V2 PACKAGE SCHEMA (2026-08-27) -- code_version, the
# reader's key-presence dispatch helpers, and the meta.json per-camera
# `storage` field.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not IN_GIT_CHECKOUT, reason=_NEEDS_GIT)
def test_code_version_returns_a_real_full_sha_in_this_repo():
    """This worktree IS a real git checkout -- `_code_version()` must
    return a real, non-empty FULL (40-char) SHA, not None and not the
    old short (7-char) form, confirming the subprocess call is actually
    reachable/correct from this module's own real working directory (not
    just that it degrades gracefully). Renamed from
    `test_code_version_returns_a_real_short_sha_in_this_repo`
    (2026-08-27 follow-up round: `--short` dropped so `code_version` is
    the full 40-char SHA)."""
    calib_pkg._code_version.cache_clear()
    sha = calib_pkg._code_version()
    assert sha is not None
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha)


def test_code_version_degrades_to_none_never_raises_when_git_fails(monkeypatch):
    """Best-effort, per its own docstring: any failure (git missing, not
    a checkout, timeout, ...) must degrade to None, never raise, and
    never break a calibration save over it."""
    calib_pkg._code_version.cache_clear()

    def _raise(*a, **k):
        raise FileNotFoundError("git not found")
    monkeypatch.setattr(calib_pkg.subprocess, "run", _raise)
    assert calib_pkg._code_version() is None
    calib_pkg._code_version.cache_clear()


def test_code_version_degrades_to_none_on_nonzero_returncode(monkeypatch):
    """A `git` binary that exists but fails (e.g. not a git checkout,
    `rev-parse` exits non-zero) must also degrade to None, not raise and
    not return a garbage string."""
    calib_pkg._code_version.cache_clear()

    class _FakeProc:
        returncode = 128
        stdout = ""

    monkeypatch.setattr(calib_pkg.subprocess, "run", lambda *a, **k: _FakeProc())
    assert calib_pkg._code_version() is None
    calib_pkg._code_version.cache_clear()


def test_code_version_is_cached_across_calls(monkeypatch):
    """Cached for the lifetime of one process -- a second call must not
    re-invoke the subprocess (the running code's own commit cannot
    change mid-process)."""
    calib_pkg._code_version.cache_clear()
    calls = []

    class _FakeProc:
        returncode = 0
        stdout = "abc1234\n"

    def _fake_run(*a, **k):
        calls.append(1)
        return _FakeProc()
    monkeypatch.setattr(calib_pkg.subprocess, "run", _fake_run)
    assert calib_pkg._code_version() == "abc1234"
    assert calib_pkg._code_version() == "abc1234"
    assert len(calls) == 1
    calib_pkg._code_version.cache_clear()


@pytest.mark.skipif(not IN_GIT_CHECKOUT, reason=_NEEDS_GIT)
def test_save_calibration_package_writes_real_code_version_into_all_three_solved_blocks(tmp_path):
    """`code_version` lands at the package's own top level AND inside
    both nested `ring_boundary_offset`/`board_color_calibration` blocks
    (identical value -- one calibration event always runs under one code
    state)."""
    diagnostics = {
        0: {
            "ring_boundary_offset_accepted": True,
            "ring_boundary_offset_payload": RING_BOUNDARY_OFFSET_PAYLOAD_ACCEPTED,
            "board_color_calibration_payload": BOARD_COLOR_CALIBRATION_PAYLOAD_ACCEPTED,
        },
    }
    package_dir = calib_pkg.save_calibration_package(
        tmp_path, "calib_code_version", {0: []}, {0: _fake_calibration()}, diagnostics,
    )
    derived = json.loads((package_dir / "derived_calibration.json").read_text())
    assert derived["code_version"] is not None
    assert derived["ring_boundary_offset"]["code_version"] == derived["code_version"]
    assert derived["board_color_calibration"]["code_version"] == derived["code_version"]


def test_save_calibration_package_meta_json_per_camera_storage_ffv1_on_success(tmp_path):
    """QA's own second meta.json pass (2026-08-27) -- `cameras.N.storage`
    must be `"ffv1"` when the raw-video encode actually succeeded,
    the per-camera `storage` field (opendarts only ever writes
    `"ffv1"` -- there is no PNG-per-frame fallback path here, see the
    module's own docstring)."""
    frames = [np.zeros((4, 4, 3), np.uint8) for _ in range(2)]
    package_dir = calib_pkg.save_calibration_package(
        tmp_path, "calib_storage_ok", {0: frames}, {0: _fake_calibration()},
    )
    meta = json.loads((package_dir / "meta.json").read_text())
    assert meta["cameras"]["0"]["storage"] == "ffv1"
    assert meta["cameras"]["0"]["raw_video"] == "cam0_raw.mkv"


def test_save_calibration_package_meta_json_per_camera_storage_none_on_encode_failure(
    tmp_path, monkeypatch,
):
    """The other real value: `None` when the encode failed for this
    camera -- matches the existing `raw_video: null` case, now paired
    with an equally-null `storage`."""
    def _raise(*a, **k):
        raise RuntimeError("simulated FFV1 encode failure")
    monkeypatch.setattr(calib_pkg, "_encode_raw_video", _raise)
    frames = [np.zeros((4, 4, 3), np.uint8) for _ in range(2)]
    package_dir = calib_pkg.save_calibration_package(
        tmp_path, "calib_storage_fail", {0: frames}, {0: _fake_calibration()},
    )
    meta = json.loads((package_dir / "meta.json").read_text())
    assert meta["cameras"]["0"]["storage"] is None
    assert meta["cameras"]["0"]["raw_video"] is None


def test_package_ring_boundary_offsets_dispatches_old_flat_format(tmp_path):
    from opendarts.capture.throw_package import _package_ring_boundary_offsets

    old_format = {"treble_inner_offset_mm": 2.5, "double_inner_offset_mm": 2.9}
    assert _package_ring_boundary_offsets(old_format) == (2.5, 2.9)


def test_package_ring_boundary_offsets_dispatches_new_nested_format(tmp_path):
    from opendarts.capture.throw_package import _package_ring_boundary_offsets

    new_format = {
        "ring_boundary_offset": {
            "boundaries": {
                "treble_inner": {"offset_mm": 1.1},
                "double_inner": {"offset_mm": 1.9},
            },
        },
    }
    assert _package_ring_boundary_offsets(new_format) == (1.1, 1.9)


def test_package_ring_boundary_offsets_none_when_pkg_derived_is_none():
    from opendarts.capture.throw_package import _package_ring_boundary_offsets

    assert _package_ring_boundary_offsets(None) == (None, None)


def test_package_ring_boundary_offsets_none_when_neither_shape_present():
    from opendarts.capture.throw_package import _package_ring_boundary_offsets

    assert _package_ring_boundary_offsets({"schema": "calibration-package-derived-v1"}) == (
        None, None,
    )


def test_package_board_color_threshold_value_dispatches_old_flat_format():
    from opendarts.capture.throw_package import _package_board_color_threshold_value

    old_format = {"brightness_threshold": 133.0}
    assert _package_board_color_threshold_value(
        old_format, "brightness_threshold", "brightness_threshold_black_cream",
    ) == 133.0
    # Old flat format's REJECTED (None) value is also returned as-is --
    # already the final resolved value, no further confidence check.
    old_rejected = {"brightness_threshold": None}
    assert _package_board_color_threshold_value(
        old_rejected, "brightness_threshold", "brightness_threshold_black_cream",
    ) is None


def test_package_board_color_threshold_value_dispatches_new_nested_format():
    from opendarts.capture.throw_package import _package_board_color_threshold_value

    new_format = {
        "board_color_calibration": {
            "brightness_threshold_black_cream": {"value": 140.0, "confidence": "high"},
        },
    }
    assert _package_board_color_threshold_value(
        new_format, "brightness_threshold", "brightness_threshold_black_cream",
    ) == 140.0

    new_format_low_confidence = {
        "board_color_calibration": {
            "brightness_threshold_black_cream": {"value": 140.0, "confidence": "low"},
        },
    }
    assert _package_board_color_threshold_value(
        new_format_low_confidence, "brightness_threshold", "brightness_threshold_black_cream",
    ) is None


# ---------------------------------------------------------------------------
# 8. V2 PACKAGE SCHEMA, CALIBRATION FOLLOW-UP (2026-08-27) -- the 4 real
# gaps QA found on a fresh pull (one recorded calibration package)
# against the real cross-repo parity checker: the `frame_indices_used`
# rename (gap #1), the new `raw_extra_frame_indices` field (gap #2),
# 7 previously-computed-but-never-persisted k1/cx/distortion
# provenance fields (gap #3), and the `code_version` short->full SHA
# fix (gap #4, covered above in section 7's `_code_version()` tests).
# ---------------------------------------------------------------------------


def test_save_calibration_package_writes_frame_indices_used_renamed_field(tmp_path):
    """Gap #1 -- pure rename. The whitelist must write the NEW key name
    (`frame_indices_used`) and must NOT write the old one
    (`frames_used_indices`) -- a rename, not a second synonym key."""
    diagnostics = {0: {"frame_indices_used": [2, 3, 4]}}
    package_dir = calib_pkg.save_calibration_package(
        tmp_path, "calib_frame_indices_used", {0: []}, {0: _fake_calibration()}, diagnostics,
    )
    derived = json.loads((package_dir / "derived_calibration.json").read_text())
    cam0 = derived["cameras"]["0"]
    assert cam0["frame_indices_used"] == [2, 3, 4]
    assert "frames_used_indices" not in cam0


def test_save_calibration_package_writes_raw_extra_frame_indices(tmp_path):
    """Gap #2 -- the new field, both with a real non-empty complement
    (cam0) and the real empty-list case, nothing left over (cam1)."""
    diagnostics = {
        0: {
            "frame_indices_used": [5, 6, 7, 8, 9],
            "raw_extra_frame_indices": [0, 1, 2, 3, 4],
        },
        1: {
            "frame_indices_used": [0, 1, 2],
            "raw_extra_frame_indices": [],
        },
    }
    package_dir = calib_pkg.save_calibration_package(
        tmp_path, "calib_raw_extra_frame_indices", {0: [], 1: []},
        {0: _fake_calibration(), 1: _fake_calibration(1.0)}, diagnostics,
    )
    derived = json.loads((package_dir / "derived_calibration.json").read_text())
    assert derived["cameras"]["0"]["raw_extra_frame_indices"] == [0, 1, 2, 3, 4]
    assert derived["cameras"]["1"]["raw_extra_frame_indices"] == []


def test_save_calibration_package_writes_k1_cx_distortion_provenance_fields(tmp_path):
    """Gap #3 -- the 7 real k1/cx/distortion fields
    `bootstrap_calibrations()`'s own `per_cam_diagnostics` has computed
    since the 2026-08-26 k1/cx work landed, but this whitelist never let
    through before this round. cam0 gets real, fully-resolved values
    (every tier succeeded); cam1 gets the real "k1-only succeeded, +cx
    tier never did" degraded shape (k1 is always non-None per
    `capture_daemon.py`'s own convention, the cx-tier fields None) --
    both must pass through exactly as given, never fabricated or
    dropped."""
    diagnostics = {
        0: {
            "k1": -0.220,
            "distortion_source": "live",
            "cx_px": -44.5,
            "cx_focal_length_px": 834.6,
            "cx_k1": -0.221,
            "joint_focal_length_px": 838.0,
            "principal_point_source": "live",
        },
        1: {
            "k1": 0.0,
            "distortion_source": "unavailable",
            "cx_px": None,
            "cx_focal_length_px": None,
            "cx_k1": None,
            "joint_focal_length_px": None,
            "principal_point_source": None,
        },
    }
    package_dir = calib_pkg.save_calibration_package(
        tmp_path, "calib_k1_cx_distortion", {0: [], 1: []},
        {0: _fake_calibration(), 1: _fake_calibration(1.0)}, diagnostics,
    )
    derived = json.loads((package_dir / "derived_calibration.json").read_text())
    cam0 = derived["cameras"]["0"]
    assert cam0["k1"] == pytest.approx(-0.220)
    assert cam0["distortion_source"] == "live"
    assert cam0["cx_px"] == pytest.approx(-44.5)
    assert cam0["cx_focal_length_px"] == pytest.approx(834.6)
    assert cam0["cx_k1"] == pytest.approx(-0.221)
    assert cam0["joint_focal_length_px"] == pytest.approx(838.0)
    assert cam0["principal_point_source"] == "live"

    cam1 = derived["cameras"]["1"]
    assert cam1["k1"] == 0.0
    assert cam1["distortion_source"] == "unavailable"
    assert cam1["cx_px"] is None
    assert cam1["cx_focal_length_px"] is None
    assert cam1["cx_k1"] is None
    assert cam1["joint_focal_length_px"] is None
    assert cam1["principal_point_source"] is None


def test_save_calibration_package_omits_new_provenance_fields_when_diagnostics_empty(tmp_path):
    """A caller that never collected diagnostics (this module's own
    documented `diagnostics=None`/`{}` case) writes a valid package with
    none of the new gap #1/#2/#3 fields present at all -- absent, never
    a fabricated null, matching every other optional diagnostics field's
    own convention in this whitelist."""
    package_dir = calib_pkg.save_calibration_package(
        tmp_path, "calib_no_diagnostics_v2f_followup", {0: []}, {0: _fake_calibration()}, None,
    )
    derived = json.loads((package_dir / "derived_calibration.json").read_text())
    cam0 = derived["cameras"]["0"]
    for key in (
        "k1", "distortion_source", "cx_px", "cx_focal_length_px",
        "cx_k1", "joint_focal_length_px", "principal_point_source",
        "frame_indices_used", "raw_extra_frame_indices",
    ):
        assert key not in cam0


def test_bootstrap_to_saved_package_end_to_end_carries_k1_cx_and_frame_indices(
    tmp_path, monkeypatch,
):
    """End-to-end proof that gap #3's 7 fields (and gaps #1/#2's renamed
    frame-selection fields) are reachable through the REAL production
    call sequence, not just a hand-built diagnostics dict fed straight
    to `save_calibration_package()`: run the real (synthetic-frame)
    `bootstrap_calibrations()` -- the same function that produces
    `established_k1`/`established_cx`/etc. as local variables and
    threads them into `per_cam_diagnostics` -- then feed its own
    `diagnostics_out` straight into `save_calibration_package()`, exactly
    as `bootstrap_calibrations()`'s live background-save wiring does. If
    a future refactor ever breaks the thread from those locals into
    `per_cam_diagnostics`, or the whitelist ever drops one of these keys
    again, this test (not just the hand-built unit tests above) catches
    it. This is also the "the write path must be reachable by a future
    replay-solve caller too, not hardwired to the live bootstrap call"
    property flagged by the coordinator: it proves the k1/cx fields flow
    through `bootstrap_calibrations()`'s own generic
    `diagnostics_out`/`save_calibration_package()` seam, the SAME seam
    any future caller (live or replay) that runs this project's real
    solver functions and produces this same diagnostics shape would use
    -- nothing here depends on live-only capture-loop state that a
    frames-already-decoded replay caller wouldn't also have by the time
    it reaches this exact point.
    """
    import tests.test_capture_daemon_oriented_wiring as odw

    n_frames = 3

    def fake_correspond(image_bgr, pre, *, orientation_hint_deg=None, results_out=None, **kw):
        if results_out is not None:
            results_out.append(odw._ok_result())
        return odw._OBJ, odw._PX

    # This test's own helper (`odw._install_fake_pipeline`) lives in
    # test_capture_daemon_oriented_wiring.py, whose own autouse fixture
    # (forcing ORIENTATION_METHOD back to the digit-count method) does
    # NOT apply here -- autouse fixtures are scoped to where a test is
    # COLLECTED from, not where an imported helper function happens to
    # live. This test exercises the digit-count method's own real
    # diagnostics shape (`odw._ok_result()` et al), so force the
    # selector explicitly.
    monkeypatch.setattr(
        capture_daemon, "_capture_calibration_frames_local",
        lambda hub, n, **k: odw._frames(n, cams=(0,)),
    )
    odw._install_fake_pipeline(monkeypatch, correspond=fake_correspond)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", lambda *a, **k: odw._fake_attempt())

    diagnostics: dict = {}
    calibrations = capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), n_frames=n_frames, # type: ignore[arg-type]
        diagnostics_out=diagnostics,
    )
    # Real production shape check before trusting the rest of this test:
    # `bootstrap_calibrations()` must have actually populated these keys
    # itself (k1 defaults to 0.0, never absent, per its own convention).
    assert "k1" in diagnostics[0]
    assert "frame_indices_used" in diagnostics[0]
    assert "raw_extra_frame_indices" in diagnostics[0]

    package_dir = calib_pkg.save_calibration_package(
        tmp_path / "pkg_out", "calib_e2e_k1_cx", {0: []}, calibrations, diagnostics,
    )
    derived = json.loads((package_dir / "derived_calibration.json").read_text())
    cam0 = derived["cameras"]["0"]
    # k1 is always written (0.0 default, never dropped by the whitelist).
    assert cam0["k1"] == diagnostics[0]["k1"]
    assert cam0["distortion_source"] == diagnostics[0]["distortion_source"]
    assert cam0["frame_indices_used"] == diagnostics[0]["frame_indices_used"]
    assert cam0["raw_extra_frame_indices"] == diagnostics[0]["raw_extra_frame_indices"]
    # cx-tier fields: whatever this synthetic run actually resolved
    # (None if that tier never ran/succeeded on synthetic data) must
    # match diagnostics_out exactly -- no silent drop either way.
    for key in ("cx_px", "cx_focal_length_px", "cx_k1", "joint_focal_length_px",
                "principal_point_source"):
        assert cam0[key] == diagnostics[0][key]
