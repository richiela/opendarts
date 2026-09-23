"""Integration tests for the RING-CORRELATION (DOHS3) orientation method
wired live into `opendarts.live.capture_daemon.bootstrap_calibrations()`,
2026-08-29 -- see that module's own "RING-CORRELATION ORIENTATION
(DOHS3)" section for the full design this exercises. Deliberately a
exercising the retry/inference/refusal shape (Mode A/B split,
rig-consensus "inference" step, retry-with-fresh-frames-up-to-10-attempts
contract) against `opendarts.calibration.ring_correlation_orientation.
ring_correlation_orientation_for_camera()`. It was once a line-for-line
mirror of an equivalent file covering a second orientation method; that
method and its tests were removed, so this file now stands alone --
worth noticing, not something to silently "improve" independently.

`ORIENTATION_METHOD` already defaults to `"ring_correlation"` as of this
task -- this file's own autouse fixture below
sets it explicitly anyway, matching every sibling wiring-test file's own
"don't rely on the module-level default, state the method under test
explicitly" convention, so this file keeps testing exactly what it says
it tests even if a future task changes the default again.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from tests.conftest import stub_confident_orientation

import opendarts.live.capture_daemon as capture_daemon
from opendarts.calibration.focal_length import FocalLengthResult
from opendarts.calibration.oriented_landmarks import OrientedLandmarkResult, PreOrientationLandmarks
from opendarts.calibration.rig_ring_geometry import RingGeometry, save_ring_geometry
from opendarts.calibration.ring_correlation_orientation import RingCorrelationOrientationResult
from opendarts.pipeline import CalibrationAttempt, CameraCalibration


@pytest.fixture(autouse=True)
def _use_ring_correlation_orientation_method(monkeypatch):
    stub_confident_orientation(monkeypatch, capture_daemon)


_OBJ = np.zeros((4, 3))
_PX = np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])

# Real gap structure (spec section 2's own measured table, same values
# test_capture_daemon_rig_consensus.py/test_capture_daemon_aggregate_
# mark_wiring.py both already use).
_GAPS = [103.33, 107.78, 148.89]
_TRUE_HINTS = {0: 87.2, 1: (87.2 + 103.33) % 360.0, 2: (87.2 + 103.33 + 107.78) % 360.0}


def _tagged_frames(n_frames: int, cams: tuple[int, ...]) -> dict[int, list[np.ndarray]]:
    out = {}
    for c in cams:
        frame = np.zeros((10, 10, 3), np.uint8)
        frame[0, 0, 0] = c
        out[c] = [frame.copy() for _ in range(n_frames)]
    return out


def _pre_for_cam(cam: int) -> PreOrientationLandmarks:
    return PreOrientationLandmarks(
        ok=True, reason="ok", ellipse=None, seed_ellipse=None, bull_px=(float(cam), 0.0),
        normalised_bull_radius=0.0, profile=None, phase_deg=0.0,
        spoke_score=1.0, phase_confidence=5.0, notes=[],
    )


def _fake_locate_pre_orientation_landmarks(frame, **kw):
    cam = int(frame[0, 0, 0])
    return frame, _pre_for_cam(cam)


def _ok_result(ambiguous: bool = False) -> OrientedLandmarkResult:
    return OrientedLandmarkResult(
        ok=True, reason="ok", quad_px=_PX.copy(), object_points_mm=_OBJ.copy(),
        phase_confidence=2.5, colour_margin=0.9, orientation_ambiguous=ambiguous,
    )


def _fake_attempt() -> CalibrationAttempt:
    calib = CameraCalibration(
        camera_matrix=np.eye(3), dist_coeffs=np.zeros(5), rvec=np.zeros(3),
        tvec=np.array([0.0, 0.0, 1000.0]), pnp_result=None, landmark_spread_ok=True,
    )
    return CalibrationAttempt(ok=True, calibration=calib, pnp_result=None, reason="")


def _install_common_fakes(monkeypatch):
    """Every non-orientation-decision fake `_detect_batch()`/
    `_process_camera_round()` needs regardless of which orientation
    method is active -- `_detect_batch()` still unconditionally runs the
    digit-count method's own evidence-gathering stages even when
    ORIENTATION_METHOD="ring_correlation" (see that module's own comment
    on this), so these must still be faked even though their RESULTS are
    unused for the orientation decision itself in these tests."""
    monkeypatch.setattr(
        capture_daemon, "locate_pre_orientation_landmarks",
        _fake_locate_pre_orientation_landmarks,
    )

    def fake_correspond(image_bgr, pre, *, orientation_hint_deg=None, results_out=None, **kw):
        if results_out is not None:
            results_out.append(_ok_result())
        if orientation_hint_deg is None:
            return None
        return _OBJ, _PX

    monkeypatch.setattr(capture_daemon, "correspond_landmarks_from_pre_orientation", fake_correspond)
    monkeypatch.setattr(
        capture_daemon, "derive_focal_length_from_oriented_results",
        lambda results, principal_point, min_frames=1: FocalLengthResult(
            ok=True, focal_length_px=900.0, reason="test-fake",
            n_points_used=20, n_frames_used=max(1, len(results)),
        ),
    )
    monkeypatch.setattr(capture_daemon, "calibrate_camera", lambda *a, **k: _fake_attempt())


def _seed_geometry(tmp_path):
    geometry = RingGeometry(
        gaps_deg=list(_GAPS), n_events=21, first_learned_utc="t0", last_updated_utc="t20",
        spread_deg=[0.5, 0.5, 0.5], recent_samples=[list(_GAPS)] * 5,
    )
    save_ring_geometry(tmp_path, geometry)
    return geometry


def _ring_correlation_result(hint_deg, *, ok=True, pass_fraction=1.0) -> RingCorrelationOrientationResult:
    return RingCorrelationOrientationResult(
        ok=ok, hint_deg=hint_deg if ok else None, pass_fraction=pass_fraction,
        n_frames=1, n_passed=1 if ok else 0, n_agreeing=1 if ok else 0,
        majority_hint_deg=hint_deg, per_frame=[],
    )


# ---------------------------------------------------------------------
# Mode B (no ring geometry) -- every camera must resolve LIVE.
# ---------------------------------------------------------------------


def test_mode_b_all_cameras_pass_immediately(tmp_path, monkeypatch):
    def fake_capture(hub, n, **k):
        return _tagged_frames(n, cams=(0, 1, 2))

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_capture)
    _install_common_fakes(monkeypatch)

    def fake_ring_correlation(frames, *, min_pass_fraction=1.0, agreement_tolerance_deg=None):
        cam = int(frames[0][0, 0, 0])
        return _ring_correlation_result(_TRUE_HINTS[cam])

    monkeypatch.setattr(capture_daemon, "ring_correlation_orientation_for_camera", fake_ring_correlation)

    calib_root = tmp_path / "calib_root"
    diagnostics: dict = {}
    result = capture_daemon.bootstrap_calibrations(
        tmp_path / "snap", hub=object(), n_frames=10, n_frames_detect=10,
        max_frames=10, retry_batch_size=10, # type: ignore[arg-type]
        calibration_package_root=calib_root, diagnostics_out=diagnostics,
    )

    assert sorted(result) == [0, 1, 2]
    for cam in (0, 1, 2):
        assert diagnostics[cam]["orientation_hint_source"] == "ring_correlation_live"
        assert diagnostics[cam]["orientation_hint_deg"] == pytest.approx(_TRUE_HINTS[cam], abs=1e-6)

    # Ring geometry LEARNED (spec R3, ring-correlation equivalent) --
    # every camera resolved via genuine live derivation this event.
    geometry_file = calib_root / "ring_geometry_fallback.json"
    assert geometry_file.exists()


def test_mode_b_refuses_after_max_retries(tmp_path, monkeypatch):
    def fake_capture(hub, n, **k):
        return _tagged_frames(n, cams=(0, 1, 2))

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_capture)
    _install_common_fakes(monkeypatch)

    def fake_ring_correlation(frames, *, min_pass_fraction=1.0, agreement_tolerance_deg=None):
        cam = int(frames[0][0, 0, 0])
        if cam == 2:
            return _ring_correlation_result(None, ok=False, pass_fraction=0.0)
        return _ring_correlation_result(_TRUE_HINTS[cam])

    monkeypatch.setattr(capture_daemon, "ring_correlation_orientation_for_camera", fake_ring_correlation)

    with pytest.raises(capture_daemon.OrientationConsensusRefusedError) as exc_info:
        capture_daemon.bootstrap_calibrations(
            tmp_path / "snap", hub=object(), n_frames=10, n_frames_detect=10,
            max_frames=10, retry_batch_size=10, # type: ignore[arg-type]
            calibration_package_root=None,
        )
    assert "cam(s) [2]" in str(exc_info.value) or "[2]" in str(exc_info.value)
    assert "ring-correlation orientation method" in str(exc_info.value)


def test_mode_b_retries_with_fresh_frames_until_success(tmp_path, monkeypatch):
    """cam0 fails its first 2 attempts, then passes on the 3rd -- with a
    small n_frames=2 pool, this FORCES the fallback fresh-`_capture()`
    path (the pool is exhausted after 2 attempts) -- confirms both the
    retry-with-fresh-frames mechanic and the pool-then-fallback design."""
    capture_calls = {"n": 0}

    def fake_capture(hub, n, **k):
        capture_calls["n"] += 1
        return _tagged_frames(n, cams=(0, 1, 2))

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_capture)
    _install_common_fakes(monkeypatch)

    cam0_attempts = {"n": 0}

    def fake_ring_correlation(frames, *, min_pass_fraction=1.0, agreement_tolerance_deg=None):
        cam = int(frames[0][0, 0, 0])
        if cam == 0:
            cam0_attempts["n"] += 1
            if cam0_attempts["n"] < 3:
                return _ring_correlation_result(None, ok=False, pass_fraction=0.0)
        return _ring_correlation_result(_TRUE_HINTS[cam])

    monkeypatch.setattr(capture_daemon, "ring_correlation_orientation_for_camera", fake_ring_correlation)

    diagnostics: dict = {}
    result = capture_daemon.bootstrap_calibrations(
        tmp_path / "snap", hub=object(), n_frames=2, n_frames_detect=2,
        max_frames=2, retry_batch_size=2, # type: ignore[arg-type]
        calibration_package_root=None, diagnostics_out=diagnostics,
    )

    assert sorted(result) == [0, 1, 2]
    assert diagnostics[0]["orientation_hint_source"] == "ring_correlation_live"
    assert cam0_attempts["n"] == 3
    # 1 initial capture (n_frames=2) + at least 1 fresh fallback capture
    # once cam0's own 2-frame pool was exhausted on its 3rd attempt.
    assert capture_calls["n"] >= 2


# ---------------------------------------------------------------------
# Mode A (ring geometry known) -- inference (rig-consensus) reused.
# ---------------------------------------------------------------------


def test_mode_a_fills_subfloor_camera_via_rig_consensus_single_attempt(tmp_path, monkeypatch):
    calib_root = tmp_path / "calib_root"
    _seed_geometry(calib_root)

    capture_calls = {"n": 0}

    def fake_capture(hub, n, **k):
        capture_calls["n"] += 1
        return _tagged_frames(n, cams=(0, 1, 2))

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_capture)
    _install_common_fakes(monkeypatch)

    def fake_ring_correlation(frames, *, min_pass_fraction=1.0, agreement_tolerance_deg=None):
        cam = int(frames[0][0, 0, 0])
        if cam == 2:
            # Never individually passes, but its own best (sub-floor)
            # candidate matches the truth -- used as the rig-consensus
            # tie-break, per spec R2.2 step 3.
            return _ring_correlation_result(_TRUE_HINTS[2], ok=False, pass_fraction=0.0)
        return _ring_correlation_result(_TRUE_HINTS[cam])

    monkeypatch.setattr(capture_daemon, "ring_correlation_orientation_for_camera", fake_ring_correlation)

    diagnostics: dict = {}
    result = capture_daemon.bootstrap_calibrations(
        tmp_path / "snap", hub=object(), n_frames=10, n_frames_detect=10,
        max_frames=10, retry_batch_size=10, # type: ignore[arg-type]
        calibration_package_root=calib_root, diagnostics_out=diagnostics,
    )

    assert sorted(result) == [0, 1, 2]
    assert diagnostics[0]["orientation_hint_source"] == "ring_correlation_live"
    assert diagnostics[1]["orientation_hint_source"] == "ring_correlation_live"
    assert diagnostics[2]["orientation_hint_source"] == "ring_correlation_rig_consensus"
    assert diagnostics[2]["orientation_hint_deg"] == pytest.approx(_TRUE_HINTS[2], abs=1.0)
    # Resolved on the FIRST attempt -- one shared capture call total
    # (the initial n_frames pool already had enough for every camera's
    # first window).
    assert capture_calls["n"] == 1


def test_mode_a_d2_rejects_confident_alias_and_refuses_ambiguous_refill(tmp_path, monkeypatch, caplog):
    """UPDATED 2026-08-31 per spec section 9's post-ship mirror-ambiguity
    tie-break fix (`opendarts.calibration.rig_ring_geometry`'s own top
    docstring, "POST-SHIP FIX" section): a camera whose own live
    derivation CLEARS the confidence floor but is a real +162deg alias
    is still rejected by the D2 cross-check and NEVER accepted as
    'live' -- but for this 3-camera rig, cam2's own (now D2-rejected)
    value was the ONLY signal available to fill it back in via
    rig-consensus, and that resolution is structurally always tied with
    nothing else to break it. The correct behaviour is now a loud
    refusal (`OrientationConsensusRefusedError`), not a silent guess --
    this test used to assert a successful (but not provably correct)
    refill before the post-ship fix landed."""
    calib_root = tmp_path / "calib_root"
    _seed_geometry(calib_root)

    def fake_capture(hub, n, **k):
        return _tagged_frames(n, cams=(0, 1, 2))

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_capture)
    _install_common_fakes(monkeypatch)

    aliased_hint = (_TRUE_HINTS[2] + 162.0) % 360.0

    def fake_ring_correlation(frames, *, min_pass_fraction=1.0, agreement_tolerance_deg=None):
        cam = int(frames[0][0, 0, 0])
        hint = aliased_hint if cam == 2 else _TRUE_HINTS[cam]
        return _ring_correlation_result(hint) # every camera confidently "passes"

    monkeypatch.setattr(capture_daemon, "ring_correlation_orientation_for_camera", fake_ring_correlation)

    diagnostics: dict = {}
    with caplog.at_level(logging.WARNING), pytest.raises(capture_daemon.OrientationConsensusRefusedError) as exc_info:
        capture_daemon.bootstrap_calibrations(
            tmp_path / "snap", hub=object(), n_frames=10, n_frames_detect=10,
            max_frames=10, retry_batch_size=10, # type: ignore[arg-type]
            calibration_package_root=calib_root, diagnostics_out=diagnostics,
        )
    assert "[2]" in str(exc_info.value) or "cam2" in str(exc_info.value)
    # The retry loop's own WARNING logs must show the alias was actually
    # caught by D2 cross-check (and correctly excluded from the fill-in
    # tie-break) on every attempt, not just that a refusal happened for
    # some unrelated reason.
    assert any("D2-rejected as inconsistent" in rec.message for rec in caplog.records)
    assert any("mirror ambiguity" in rec.message for rec in caplog.records)


def test_no_regression_all_cameras_confident_stays_live_with_geometry_known(tmp_path, monkeypatch):
    calib_root = tmp_path / "calib_root"
    _seed_geometry(calib_root)

    def fake_capture(hub, n, **k):
        return _tagged_frames(n, cams=(0, 1, 2))

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_capture)
    _install_common_fakes(monkeypatch)

    def fake_ring_correlation(frames, *, min_pass_fraction=1.0, agreement_tolerance_deg=None):
        cam = int(frames[0][0, 0, 0])
        return _ring_correlation_result(_TRUE_HINTS[cam])

    monkeypatch.setattr(capture_daemon, "ring_correlation_orientation_for_camera", fake_ring_correlation)

    diagnostics: dict = {}
    result = capture_daemon.bootstrap_calibrations(
        tmp_path / "snap", hub=object(), n_frames=10, n_frames_detect=10,
        max_frames=10, retry_batch_size=10, # type: ignore[arg-type]
        calibration_package_root=calib_root, diagnostics_out=diagnostics,
    )

    assert sorted(result) == [0, 1, 2]
    for cam in (0, 1, 2):
        assert diagnostics[cam]["orientation_hint_source"] == "ring_correlation_live"


def test_ambiguous_frames_after_established_hint_report_the_real_source(
    tmp_path, monkeypatch, caplog
):
    """2026-09-03 regression guard (cam0-calibration-fails-until-restart
    defect investigation, see docs/DESIGN.md). `_report_round()`'s own
    ambiguous-lock guidance used to recognise only the bare `"live"`/
    `"rig_consensus"` strings -- the digit-count method's (DOHS1)
    convention, and also what the OLDER, DOHS1-coupled Mode A wrote.
    Ring-correlation (DOHS3, this file's own method) writes
    `"ring_correlation_live"`/`"ring_correlation_rig_consensus"`
    instead, which never matched -- so under the live DOHS3 default this
    warning ALWAYS claimed "NO established orientation hint at all" even
    when a real, confident, correctly-established hint genuinely
    existed. Confirmed via direct instrumentation against a real failing
    calibration package before this fix landed (see docs/DESIGN.md). Purely a
    diagnostic-message fix -- the calibration RESULT is identical either
    way (this test's own `assert 0 in result` line is unchanged by the
    fix, matching this file's DOHS1 sibling,
    `test_partially_ambiguous_camera_still_calibrates_from_the_good_
    frames` in `tests/test_capture_daemon_oriented_wiring.py`), only the
    WARNING text changes."""
    # _min_good_frames_cumulative()'s ABSOLUTE floor is 5 (2026-09-03 --
    # see that function's own docstring and docs/DESIGN.md's DEFECT 2 entry
    # for why the earlier strict-majority-of-the-total rule was
    # replaced); n_frames=10 with 7 clean / 3 ambiguous clears it with
    # room to spare while still genuinely demonstrating partial
    # tolerance, matching the digit-count sibling test.
    n_frames = 10
    state = {"n": 0}

    def fake_capture(hub, n, **k):
        return _tagged_frames(n, cams=(0,))

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_capture)
    _install_common_fakes(monkeypatch)

    def fake_correspond(image_bgr, pre, *, orientation_hint_deg=None, results_out=None, **kw):
        i = state["n"]
        state["n"] += 1
        ambiguous = i >= 7
        if results_out is not None:
            results_out.append(_ok_result(ambiguous=ambiguous))
        if ambiguous or orientation_hint_deg is None:
            return None
        return _OBJ, _PX

    monkeypatch.setattr(capture_daemon, "correspond_landmarks_from_pre_orientation", fake_correspond)

    def fake_ring_correlation(frames, *, min_pass_fraction=1.0, agreement_tolerance_deg=None):
        return _ring_correlation_result(_TRUE_HINTS[0])

    monkeypatch.setattr(capture_daemon, "ring_correlation_orientation_for_camera", fake_ring_correlation)

    with caplog.at_level(logging.WARNING):
        result = capture_daemon.bootstrap_calibrations(
            tmp_path, hub=object(), n_frames=n_frames, n_frames_detect=n_frames,
            max_frames=n_frames, retry_batch_size=n_frames, # type: ignore[arg-type]
        )

    assert 0 in result # 7 of 10 clean clears the absolute floor of 5, unchanged by this fix
    text = caplog.text
    assert "3/10 calibration frame(s) REJECTED" in text
    # The real fix: the guidance names the ACTUAL source (ring-correlation
    # live derivation), not the old string-mismatch fallback message.
    assert "derived LIVE from its own captured frames" in text
    assert "opendarts.calibration.ring_correlation_orientation" in text
    assert "NO established orientation hint at all" not in text


# ---------------------------------------------------------------------
# Selector -- switching to each of the other two methods must still work.
# ---------------------------------------------------------------------






def test_invalid_orientation_method_raises_loudly(tmp_path, monkeypatch):
    """An unrecognised `ORIENTATION_METHOD` value must refuse loudly, not
    silently fall through to one of the three real methods -- matches
    this whole file's/project's own "fail visibly, never guess" bar."""

    def fake_capture(hub, n, **k):
        return _tagged_frames(n, cams=(0, 1, 2))

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_capture)
    _install_common_fakes(monkeypatch)
    monkeypatch.setattr(capture_daemon, "ORIENTATION_METHOD", "not-a-real-method")

    with pytest.raises(ValueError, match="not-a-real-method"):
        capture_daemon.bootstrap_calibrations(
            tmp_path / "snap", hub=object(), n_frames=10, n_frames_detect=10,
            max_frames=10, retry_batch_size=10, # type: ignore[arg-type]
            calibration_package_root=None,
        )
