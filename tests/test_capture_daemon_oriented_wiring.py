"""Tests for the live calibration path being wired to
`opendarts.calibration.oriented_landmarks` (2026-08-13), and, since
2026-08-20, to a LIVE-DERIVED per-camera orientation hint
(`opendarts.calibration.ring_correlation_orientation`)
in place of a hardcoded per-camera orientation-hint constant --
see `capture_daemon.bootstrap_calibrations()`'s own "LIVE-DERIVED
ORIENTATION HINT" docstring section for the full design (why the two-pass
split, the fallback rule, the diagnostics fields).

`capture_daemon.bootstrap_calibrations()` -- the ONE function both real
calibration call sites go through (startup auto-calibrate and the
dashboard's "Refresh calibration now" button) -- used to source its 4
landmark pixels from `sector_correspondence.detect_and_correspond()`,
then from `oriented_landmarks.correspond_landmarks_oriented()` with a
hardcoded per-camera hint, and now from the same detector fed a hint
derived from the frames it is itself capturing.

Two tiers:
  1. Wiring/logic, fully synthetic -- which detector functions are called
     each round, that a live-derived hint is preferred by default, that
     the hardcoded constant is used ONLY as a last resort after
     `max_frames` with no live hint available, that a camera with
     genuinely no hint at all (no live evidence AND no fallback constant)
     is skipped, and that an ambiguous orientation lock is rejected AND
     reported rather than silently averaged into a calibration that
     could be a quarter-turn wrong.
  2. One real end-to-end pass through the actual function over a real
     archived session's frames (skipped when `data/archive/` isn't on
     this machine -- it is gitignored, 1.9GB).
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from opendarts.calibration.ring_correlation_orientation import (
    RingCorrelationOrientationResult,
)

import opendarts.live.capture_daemon as capture_daemon
from opendarts.calibration.focal_length import FocalLengthResult
from opendarts.calibration.oriented_landmarks import (
    OrientedLandmarkResult,
    PreOrientationLandmarks,
)
from opendarts.pipeline import CalibrationAttempt, CameraCalibration

CORPUS_ROOT = Path("data/archive/clean")
ARCHIVE_SESSIONS = (
    sorted(p for p in CORPUS_ROOT.glob("*") if p.is_dir())
    if CORPUS_ROOT.is_dir() else []
)
HAS_REAL_DATA = bool(ARCHIVE_SESSIONS)

_OBJ = np.zeros((4, 3))
_PX = np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])

_FAKE_PRE_OK = PreOrientationLandmarks(
    ok=True, reason="ok", ellipse=None, seed_ellipse=None, bull_px=(1.0, 1.0),
    normalised_bull_radius=0.0, profile=None, phase_deg=0.0,
    spoke_score=1.0, phase_confidence=5.0, notes=[],
)
_FAKE_PRE_FAIL = PreOrientationLandmarks(
    ok=False, reason="seed ellipse failed: synthetic frame", ellipse=None,
    seed_ellipse=None, bull_px=None, normalised_bull_radius=0.0, profile=None,
    phase_deg=0.0, spoke_score=0.0, phase_confidence=0.0, notes=[],
)


def _confident_session_result(frames):
    """A stand-in per-camera orientation solve that SUCCEEDS -- the "live
    derivation just works" case most of these tests want by default. The
    refusal tests below override this with the never-confident one."""
    n = len(list(frames))
    if not n:
        return RingCorrelationOrientationResult(
            ok=False, hint_deg=None, pass_fraction=0.0, n_frames=0, n_passed=0,
            n_agreeing=0, majority_hint_deg=None, per_frame=[], reason="no frames",
        )
    return RingCorrelationOrientationResult(
        ok=True, hint_deg=222.0, pass_fraction=1.0, n_frames=n, n_passed=n,
        n_agreeing=n, majority_hint_deg=222.0, per_frame=[], reason="ok",
    )


def _never_confident_session_result(frames):
    """A stand-in that NEVER produces a hint, however many frames
    accumulate -- the "live derivation genuinely fails" case. This is what
    makes the refusal tests below meaningful; do not replace it with a
    confident stub."""
    n = len(list(frames))
    return RingCorrelationOrientationResult(
        ok=False, hint_deg=None, pass_fraction=0.0, n_frames=n, n_passed=0,
        n_agreeing=0, majority_hint_deg=None, per_frame=[],
        reason="no confident roll",
    )


def _install_fake_pipeline(
    monkeypatch,
    *,
    correspond,
    pre=_FAKE_PRE_OK,
    session_result_fn=_confident_session_result,
):
    """Same shared-fake shape `tests/test_capture_daemon.py` uses (kept
    independently here rather than imported -- this file is meant to be
    readable on its own as the wiring-contract doc)."""
    monkeypatch.setattr(
        capture_daemon, "locate_pre_orientation_landmarks",
        lambda frame, **kw: (frame, pre),
    )
    monkeypatch.setattr(capture_daemon, "correspond_landmarks_from_pre_orientation", correspond)
    monkeypatch.setattr(
        capture_daemon, "ring_correlation_orientation_for_camera",
        lambda frames, **kw: session_result_fn(frames),
    )
    # LIVE-DERIVED FOCAL LENGTH, 2026-08-26 -- same "harmless, always-
    # confident stand-in" treatment as every other piece here: none of
    # these wiring tests exercise real focal-length math (that's
    # tests/test_focal_length.py's own job), and a resolved focal length
    # is now required before any calibrate_camera() call can happen at
    # all -- see opendarts.live.capture_daemon._resolve_focal_length_px()'s
    # own docstring. No test in this file distinguishes cameras by focal
    # length (unlike 2 tests in tests/test_capture_daemon.py that do --
    # see that file's own _install_fake_orientation_pipeline_with_focal_
    # dispatch()), so one fixed value for every camera is sufficient here.
    monkeypatch.setattr(
        capture_daemon, "derive_focal_length_from_oriented_results",
        lambda results, principal_point, min_frames=1: FocalLengthResult(
            ok=True, focal_length_px=900.0, reason="test-fake",
            n_points_used=20, n_frames_used=max(1, len(results)),
        ),
    )


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


def _frames(n_frames: int, cams=(0,)) -> dict:
    return {c: [np.zeros((10, 10, 3), np.uint8) for _ in range(n_frames)] for c in cams}


# --- tier 1: wiring -----------------------------------------------------


def test_bootstrap_uses_the_live_derived_hint_by_default(tmp_path, monkeypatch):
    """The wiring itself: every captured frame goes through
    `correspond_landmarks_from_pre_orientation`, and by default (live
    derivation succeeding) it is handed the LIVE-DERIVED hint --
    the orientation solve's own session hint -- not the
    hardcoded rig constant."""
    n_frames = 3
    seen: list[dict] = []

    def fake_correspond(image_bgr, pre, *, orientation_hint_deg=None, results_out=None, **kw):
        seen.append({"hint": orientation_hint_deg})
        if results_out is not None:
            results_out.append(_ok_result())
        return _OBJ, _PX

    monkeypatch.setattr(
        capture_daemon, "_capture_calibration_frames_local",
        lambda hub, n, **k: _frames(n, cams=(0, 1, 2)),
    )
    _install_fake_pipeline(monkeypatch, correspond=fake_correspond)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", lambda *a, **k: _fake_attempt())

    diagnostics: dict = {}
    result = capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), n_frames=n_frames, # type: ignore[arg-type]
        diagnostics_out=diagnostics,
    )

    assert sorted(result) == [0, 1, 2]
    assert len(seen) == 3 * n_frames
    for call in seen:
        assert call["hint"] == 222.0 # _confident_session_result's own hint
    for cam in (0, 1, 2):
        assert diagnostics[cam]["orientation_hint_source"] == "ring_correlation_live"
        assert diagnostics[cam]["orientation_hint_deg"] == 222.0


def test_no_hardcoded_fallback_refuses_the_whole_bootstrap(tmp_path, monkeypatch, caplog):
    """Ring-consensus orientation rule R1 (2026-08-29):
    `MEASURED_CAMERA_ORIENTATION_HINTS_DEG` is deleted from every live
    code path. A camera whose live derivation never clears the
    confidence floor, with no ring geometry available (Mode B -- no
    `calibration_package_root` here, so `load_ring_geometry()` returns
    `None`), makes the WHOLE bootstrap REFUSE
    (`OrientationConsensusRefusedError`) rather than silently
    substituting a hardcoded value -- a deliberate design decision
    do something or refuse to calibrate.\""""
    seen_hints: list[float | None] = []

    def fake_correspond(image_bgr, pre, *, orientation_hint_deg=None, results_out=None, **kw):
        seen_hints.append(orientation_hint_deg)
        if results_out is not None:
            results_out.append(_ok_result())
        return _OBJ, _PX

    cam = 0
    monkeypatch.setattr(
        capture_daemon, "_capture_calibration_frames_local",
        lambda hub, n, **k: _frames(n, cams=(cam,)),
    )
    _install_fake_pipeline(
        monkeypatch, correspond=fake_correspond,
        session_result_fn=_never_confident_session_result,
    )
    monkeypatch.setattr(capture_daemon, "calibrate_camera", lambda *a, **k: _fake_attempt())

    with caplog.at_level(logging.WARNING):
        with pytest.raises(capture_daemon.OrientationConsensusRefusedError) as exc_info:
            capture_daemon.bootstrap_calibrations(
                tmp_path, hub=object(), n_frames=5, max_frames=5, # type: ignore[arg-type]
            )

    # Every call this camera ever made used no hint at all -- never a
    # hardcoded substitution.
    assert all(h is None for h in seen_hints)
    msg = str(exc_info.value)
    assert f"cam{cam}" in msg # names the camera
    assert "REFUSED" in msg
    assert "REFUSED" in msg and "cam(s) [0]" in msg




def test_ambiguous_orientation_is_reported_not_swallowed(tmp_path, monkeypatch, caplog):
    """An ambiguous lock means the returned quad may belong to a rotation
    of the board a quarter turn away -- a calibration that is INVERTED,
    not merely noisy. It must be rejected as a frame AND named in the
    log, distinctly from an ordinary detection failure."""
    n_frames = 4

    def fake_correspond(image_bgr, pre, *, orientation_hint_deg=None, results_out=None, **kw):
        if results_out is not None:
            results_out.append(_ok_result(ambiguous=True))
        return None # what the real function does for an ambiguous lock

    monkeypatch.setattr(
        capture_daemon, "_capture_calibration_frames_local",
        lambda hub, n, **k: _frames(n, cams=(0,)),
    )
    _install_fake_pipeline(monkeypatch, correspond=fake_correspond)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", lambda *a, **k: _fake_attempt())

    with caplog.at_level(logging.WARNING):
        result = capture_daemon.bootstrap_calibrations(
            tmp_path, hub=object(), n_frames=n_frames # type: ignore[arg-type]
        )

    assert result == {}
    text = caplog.text
    assert "ambiguous orientation lock" in text
    assert f"{n_frames}/{n_frames}" in text # every frame, counted honestly
    assert "check those specific frames" in text # says what to actually check
    assert "orientation ambiguous" in text # named in the per-reason breakdown


def test_partially_ambiguous_camera_still_calibrates_from_the_good_frames(
    tmp_path, monkeypatch, caplog
):
    """Ambiguity is per frame, not per camera: enough clean frames to
    clear `_min_good_frames_cumulative()`'s ABSOLUTE floor (5,
    2026-09-03 -- see that function's own docstring and docs/DESIGN.md's
    DEFECT 2 entry for why the earlier strict-majority-of-the-total rule
    was replaced) must still produce a calibration, with the rejected
    ones reported."""
    n_frames = 10 # _min_good_frames_cumulative(10) == 5 (the absolute floor)
    state = {"n": 0}

    def fake_correspond(image_bgr, pre, *, orientation_hint_deg=None, results_out=None, **kw):
        i = state["n"]
        state["n"] += 1
        ambiguous = i >= 7
        if results_out is not None:
            results_out.append(_ok_result(ambiguous=ambiguous))
        return None if ambiguous else (_OBJ, _PX)

    monkeypatch.setattr(
        capture_daemon, "_capture_calibration_frames_local",
        lambda hub, n, **k: _frames(n, cams=(0,)),
    )
    _install_fake_pipeline(monkeypatch, correspond=fake_correspond)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", lambda *a, **k: _fake_attempt())

    with caplog.at_level(logging.WARNING):
        result = capture_daemon.bootstrap_calibrations(
            tmp_path, hub=object(), # type: ignore[arg-type]
            n_frames=n_frames, n_frames_detect=n_frames,
        )

    assert 0 in result # 7 of 10 clean clears the absolute floor of 5
    assert "3/10 calibration frame(s) REJECTED" in caplog.text


def test_a_camera_whose_pre_orientation_stage_always_fails_refuses(tmp_path, monkeypatch):
    """The other real failure mode this two-pass split introduces: a
    camera whose frames never even clear seed-ellipse/bull/phase-lock
    detection contributes NO orientation evidence at all (pre.ok=False
    every frame) -- the orientation solve never sees a single
    frame, so this camera can never establish a live hint, and (no ring
    geometry available here -- Mode B) the whole bootstrap REFUSES
    (spec R1), rather than the old pre-2026-08-29 "silently return {}".

    Note `correspond_landmarks_from_pre_orientation` IS still called for
    a `pre.ok=False` frame (same "handle it the way find_oriented_landmarks()'s
    own early return would" contract that function's own docstring
    states) -- it just does no real work and returns None immediately,
    which is exactly what this fake reproduces rather than asserting it
    is never reached at all."""
    def fake_correspond(image_bgr, pre, *, results_out=None, **kw):
        assert not pre.ok # every frame here really is a failed pre-orientation frame
        if results_out is not None:
            results_out.append(OrientedLandmarkResult(ok=False, reason=pre.reason))
        return None

    unknown_cam = 5
    monkeypatch.setattr(
        capture_daemon, "_capture_calibration_frames_local",
        lambda hub, n, **k: _frames(n, cams=(unknown_cam,)),
    )
    _install_fake_pipeline(
        monkeypatch, correspond=fake_correspond, pre=_FAKE_PRE_FAIL,
        session_result_fn=_never_confident_session_result,
    )
    monkeypatch.setattr(capture_daemon, "calibrate_camera", lambda *a, **k: _fake_attempt())

    with pytest.raises(capture_daemon.OrientationConsensusRefusedError):
        capture_daemon.bootstrap_calibrations(
            tmp_path, hub=object(), n_frames=2, max_frames=2, # type: ignore[arg-type]
        )


# --- tier 2: one real end-to-end pass through the real function ---------


@pytest.mark.slow
@pytest.mark.skipif(not HAS_REAL_DATA, reason="data/archive/ not present (gitignored, 1.9GB)")
def test_real_archived_session_calibrates_all_three_cameras(tmp_path):
    """The REAL `bootstrap_calibrations()`, unmocked end to end (including
    the real live-derived orientation hint -- nothing about the
    orientation pipeline is patched here), over one real archived
    session's real calibration frames.

    Frames reach it through the production hub contract
    (`configs`/`status[i].frame_count`/`grab_all()`), which is the entire
    interface `_capture_calibration_frames_local()` uses -- nothing
    inside the function under test is patched.
    """
    import cv2

    # PINNED, not ARCHIVE_SESSIONS[0] -- fixing a verifier-found issue
    # (2026-08-20): indexing the first alphabetically-sorted session
    # silently changes WHICH session this test measures every time
    # `data/archive/clean/`'s own contents change (a new pull, a corpus
    # reset -- see docs/DESIGN.md's own "living, curated corpus" guardrail).
    # This session is real, present in the corpus as of this fix, and its
    # own real measured numbers (see `reprojection_error_px` assertion
    # below) are what `CALIBRATION_REPROJECTION_ERROR_TOLERANCE_PX` is
    # picked from -- pinning the name keeps that measurement meaningful
    # even after the corpus grows or gets pruned.
    session_dir = Path("data/archive/clean/20260813-164658")
    if not session_dir.is_dir():
        pytest.skip(f"pinned session {session_dir} not present in this corpus")
    packages = sorted(p.parent for p in session_dir.glob("throw_*/result.json"))
    if not packages:
        packages = sorted(p.parent for p in session_dir.glob("*/result.json"))
    assert packages, session_dir
    # Capped, not the whole session -- this is a real end-to-end SMOKE
    # test (does the real, unmocked call path work at all, cameras
    # included), not the authoritative accuracy measurement -- that is
    # the full 13-session corpus measurement's job, run separately and
    # never part of the pytest smoke suite. 40 packages is comfortably more than the number
    # the orientation solve needed to reach 100% roll-match on the real
    # corpus.
    packages = packages[:40]
    frames_by_cam = {cam: [] for cam in (0, 1, 2)}
    for package in packages:
        for cam in (0, 1, 2):
            image = cv2.imread(str(package / f"cam{cam}_bg.png"))
            if image is not None:
                frames_by_cam[cam].append(image)

    class _Status:
        def __init__(self):
            self.frame_count = 0

    class ReplayHub:
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

    n_frames = max(len(v) for v in frames_by_cam.values())
    diagnostics: dict = {}
    calibrations = capture_daemon.bootstrap_calibrations(
        tmp_path, hub=ReplayHub(frames_by_cam), n_frames=n_frames,
        max_frames=n_frames, # a capped smoke test -- no real retry budget needed here
        diagnostics_out=diagnostics,
    )

    assert sorted(calibrations) == [0, 1, 2]
    for cam, calib in calibrations.items():
        assert calib.landmark_spread_ok
        assert calib.pnp_result is not None
        # REAL MEASURED tolerance, not a round guess (docs/DESIGN.md's "measure
        # the real number before writing a tolerance" rule) -- fixing a
        # verifier-found issue (2026-08-20): a prior version of this test
        # loosened this from 6.0 to 10.0px with no new measurement behind
        # it. Actual measured `reprojection_error_px` for this now-PINNED
        # session's 3 cameras, with the current (HIGH-1/HIGH-2/MEDIUM-3/
        # MEDIUM-4-fixed) `bootstrap_calibrations()`, this exact 40-package
        # replay (throwaway, not
        # shipped): cam0=1.015px, cam1=1.715px, cam2=0.781px -- max
        # 1.715px, FAR tighter than even the old 6.0px bound. 3.0px keeps
        # real margin above that measured max (loose enough not to be
        # flaky on a sub-pixel detector change) while staying nowhere
        # near the "tens of px" signature of a genuinely broken
        # correspondence (a quarter-turn-wrong orientation lock, a wrong
        # camera intrinsic) this test actually needs to catch.
        assert calib.pnp_result.reprojection_error_px < 3.0
        # Cameras sit ~400-450mm from the board on this rig; a quarter-turn
        # -wrong correspondence does not produce a pose in this range.
        assert 300.0 < float(np.linalg.norm(calib.tvec)) < 600.0
        # The real point of this task: the hint that got this camera here
        # was LIVE-DERIVED, not the hardcoded constant.
        assert diagnostics[cam]["orientation_hint_source"] == "ring_correlation_live"


# ---------------------------------------------------------------------
# LIVE-DERIVED RING-BOUNDARY OFFSET + BOARD-COLOR THRESHOLDS (2026-08-21)
# -- closing the gap flagged live the same night: these two derivations
# existed, tested, but nothing ever triggered or applied them, so every
# throw kept scoring against the generic hardcoded defaults. See
# bootstrap_calibrations()'s own "LIVE-DERIVED RING-BOUNDARY OFFSET +
# BOARD-COLOR THRESHOLDS" docstring section for the full design.
# ---------------------------------------------------------------------


def _install_confident_boundary_and_color_mocks(monkeypatch, *, boundary_confident=True, color_confident=True):
    """Mocks the underlying pure derivation functions bootstrap_
    calibrations() now calls directly, at their real source modules
    (they're imported LOCALLY inside the function body, so patching
    capture_daemon's own namespace would miss them)."""
    import opendarts.calibration.ring_boundary_offset as rbo
    import opendarts.geometry.board_color_calibration as bcc
    from opendarts.calibration.ring_boundary_offset import BoundaryMeasurement, RingBoundaryOffsetResult
    from opendarts.geometry.board_color_calibration import BoardColorCalibrationResult, ThresholdDerivation

    def fake_measure_ring(calibrations, bg_frames_per_camera):
        def boundary(name, reg_radius, measured):
            return BoundaryMeasurement(
                boundary=name, regulation_radius_mm=reg_radius,
                measured_radius_mm=measured,
                offset_mm=(reg_radius - measured) if measured is not None else None,
                mad_mm=0.3 if measured is not None else None,
                n_samples_used=20 if measured is not None else 0,
                n_samples_rejected=0, n_angles_attempted=20,
                per_camera={}, confidence=0.9 if measured is not None else 0.0,
            )
        treble_measured = 105.3 if boundary_confident else None
        double_measured = 168.8 if boundary_confident else None
        return RingBoundaryOffsetResult(
            boundaries={
                "treble_inner": boundary("treble_inner", 107.0, treble_measured),
                "double_inner": boundary("double_inner", 170.0, double_measured),
            },
        )
    # raising=False: the measurement moved to dev/ (2026-09-17) and is
    # no longer an attribute of this module at all. The stub stays as
    # the tripwire for it coming back -- if it ever does, this patches
    # it and the assertions below still mean what they say.
    monkeypatch.setattr(rbo, "measure_ring_boundary_offsets", fake_measure_ring, raising=False)

    def fake_collect_color_samples(package_id, calibrations, bg_images, patch_radius=None,
                                   points=None, projected_px_by_camera=None):
        return ["fake_sample"] # non-empty is all bootstrap's own code checks the length of

    def fake_derive_thresholds(samples, patch_radius=None):
        conf = "high" if color_confident else "insufficient_data"
        return BoardColorCalibrationResult(
            brightness_threshold=ThresholdDerivation(
                value=140.0 if color_confident else None, low_group_stat=100.0, high_group_stat=180.0,
                low_group_n=10, high_group_n=10, gap=80.0, confidence=conf,
            ),
            chroma_threshold=ThresholdDerivation(
                value=40.0 if color_confident else None, low_group_stat=10.0, high_group_stat=60.0,
                low_group_n=10, high_group_n=10, gap=50.0, confidence=conf,
            ),
            patch_radius=8, n_samples=len(samples), n_packages=1, n_packages_attempted=1,
            accuracy_single_camera=None, accuracy_majority_vote=None,
        )
    monkeypatch.setattr(bcc, "collect_color_samples", fake_collect_color_samples)
    monkeypatch.setattr(bcc, "derive_thresholds", fake_derive_thresholds)


def test_bootstrap_records_live_derived_ring_boundary_but_never_applies_it_while_applying_color(
    tmp_path, monkeypatch,
):
    """STOP APPLYING, 2026-09-02 (see docs/DESIGN.md's dated entry):
    renamed from `..._applies_live_
    derived_ring_boundary_and_color_when_confident`, which asserted the
    OLD (now-wrong) behavior that a confident ring-boundary measurement
    got applied to the live scoring globals. Board-color is untouched by
    this change and still gets applied exactly as before -- only the
    ring-boundary half of this test's own name/assertions changed. The
    scoring radii must stay at their hardcoded INNER_RING_SCORING_
    OFFSET_MM default regardless of how confident this event's own
    measurement was; the measured value itself is still recorded (see
    the nested-payload tests below) for diagnostics."""
    import opendarts.geometry.board as board
    import opendarts.geometry.board_color as board_color

    n_frames = 3
    default_treble = board.TREBLE_INNER_SCORING_RADIUS_MM
    default_double = board.DOUBLE_INNER_SCORING_RADIUS_MM

    def fake_correspond(image_bgr, pre, *, orientation_hint_deg=None, results_out=None, **kw):
        if results_out is not None:
            results_out.append(_ok_result())
        return _OBJ, _PX

    monkeypatch.setattr(
        capture_daemon, "_capture_calibration_frames_local",
        lambda hub, n, **k: _frames(n, cams=(0, 1, 2)),
    )
    _install_fake_pipeline(monkeypatch, correspond=fake_correspond)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", lambda *a, **k: _fake_attempt())
    _install_confident_boundary_and_color_mocks(monkeypatch)

    capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), n_frames=n_frames, # type: ignore[arg-type]
    )

    # treble_inner: 107.0 - 105.3 = 1.7mm offset; double_inner: 170.0 - 168.8 = 1.2mm
    # -- a real, confident measurement -- but it must NEVER reach the
    # scoring radii, which stay at their hardcoded default.
    assert board.TREBLE_INNER_SCORING_RADIUS_MM == default_treble
    assert board.DOUBLE_INNER_SCORING_RADIUS_MM == default_double
    assert board.TREBLE_INNER_SCORING_RADIUS_MM == board.TREBLE_INNER_RADIUS_MM - board.INNER_RING_SCORING_OFFSET_MM
    assert board.DOUBLE_INNER_SCORING_RADIUS_MM == board.DOUBLE_INNER_RADIUS_MM - board.INNER_RING_SCORING_OFFSET_MM
    # Board-color is a separate, unaffected derivation -- still applied.
    assert board_color.BRIGHTNESS_THRESHOLD_BLACK_CREAM == 140.0
    assert board_color.CHROMA_THRESHOLD == 40.0


def test_bootstrap_keeps_hardcoded_defaults_when_measurement_not_confident(tmp_path, monkeypatch):
    """The other half of the safety contract, same lesson as tonight's
    PIXEL_DIFF_THRESHOLD incident: a measurement that didn't actually
    produce a real value (ring-boundary) or didn't clear the confidence
    gate (color) must NOT get adopted -- the hardcoded defaults stay in
    effect, exactly as if this wiring didn't exist."""
    import opendarts.geometry.board as board
    import opendarts.geometry.board_color as board_color

    n_frames = 3

    def fake_correspond(image_bgr, pre, *, orientation_hint_deg=None, results_out=None, **kw):
        if results_out is not None:
            results_out.append(_ok_result())
        return _OBJ, _PX

    monkeypatch.setattr(
        capture_daemon, "_capture_calibration_frames_local",
        lambda hub, n, **k: _frames(n, cams=(0, 1, 2)),
    )
    _install_fake_pipeline(monkeypatch, correspond=fake_correspond)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", lambda *a, **k: _fake_attempt())
    _install_confident_boundary_and_color_mocks(monkeypatch, boundary_confident=False, color_confident=False)

    capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), n_frames=n_frames, # type: ignore[arg-type]
    )

    assert board.TREBLE_INNER_SCORING_RADIUS_MM == board.TREBLE_INNER_RADIUS_MM - board.INNER_RING_SCORING_OFFSET_MM
    assert board.DOUBLE_INNER_SCORING_RADIUS_MM == board.DOUBLE_INNER_RADIUS_MM - board.INNER_RING_SCORING_OFFSET_MM
    assert board_color.BRIGHTNESS_THRESHOLD_BLACK_CREAM == 129.0
    assert board_color.CHROMA_THRESHOLD == 35.0


def test_bootstrap_survives_ring_boundary_and_color_derivation_raising(tmp_path, monkeypatch):
    """Never break calibration itself over this -- same posture as the
    motion-threshold wiring right above it in bootstrap_calibrations()."""
    import opendarts.calibration.ring_boundary_offset as rbo
    import opendarts.geometry.board_color_calibration as bcc
    import opendarts.geometry.board as board
    import opendarts.geometry.board_color as board_color

    n_frames = 3

    def fake_correspond(image_bgr, pre, *, orientation_hint_deg=None, results_out=None, **kw):
        if results_out is not None:
            results_out.append(_ok_result())
        return _OBJ, _PX

    monkeypatch.setattr(
        capture_daemon, "_capture_calibration_frames_local",
        lambda hub, n, **k: _frames(n, cams=(0, 1, 2)),
    )
    _install_fake_pipeline(monkeypatch, correspond=fake_correspond)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", lambda *a, **k: _fake_attempt())

    def _raise(*a, **k):
        raise RuntimeError("synthetic derivation failure")
    # raising=False: the measurement moved to dev/ (2026-09-17) and is
    # no longer an attribute of this module at all. The stub stays as
    # the tripwire for it coming back -- if it ever does, this patches
    # it and the assertions below still mean what they say.
    monkeypatch.setattr(rbo, "measure_ring_boundary_offsets", _raise, raising=False)
    monkeypatch.setattr(bcc, "collect_color_samples", _raise)

    result = capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), n_frames=n_frames, # type: ignore[arg-type]
    )

    assert sorted(result) == [0, 1, 2] # calibration itself still succeeded
    assert board.TREBLE_INNER_SCORING_RADIUS_MM == board.TREBLE_INNER_RADIUS_MM - board.INNER_RING_SCORING_OFFSET_MM
    assert board_color.BRIGHTNESS_THRESHOLD_BLACK_CREAM == 129.0


# ---------------------------------------------------------------------
# THE V2 PACKAGE SCHEMA (2026-08-27) -- the FULL nested
# `ring_boundary_offset_payload`/`board_color_calibration_payload`
# diagnostics fields (the canonical shape), plus the new frame-selection
# provenance (`n_raw_extra_frames_used`/`frame_indices_used`/
# `raw_extra_frame_indices` -- the latter two renamed/added in a same-day
# follow-up round, see `bootstrap_calibrations()`'s own dated comments
# immediately above their computation). See `bootstrap_calibrations()`'s
# own dated comments for the full design.
# ---------------------------------------------------------------------


def test_bootstrap_records_full_nested_ring_and_color_payloads_when_confident(
    tmp_path, monkeypatch,
):
    """The real, canonically-shaped payloads (not just the flat legacy
    scalars) must land in `diagnostics_out`, with real `schema`/
    `solved_by`/`parameters`/`boundaries` content -- not a stub."""
    n_frames = 3

    def fake_correspond(image_bgr, pre, *, orientation_hint_deg=None, results_out=None, **kw):
        if results_out is not None:
            results_out.append(_ok_result())
        return _OBJ, _PX

    monkeypatch.setattr(
        capture_daemon, "_capture_calibration_frames_local",
        lambda hub, n, **k: _frames(n, cams=(0, 1, 2)),
    )
    _install_fake_pipeline(monkeypatch, correspond=fake_correspond)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", lambda *a, **k: _fake_attempt())
    _install_confident_boundary_and_color_mocks(monkeypatch)

    diagnostics: dict = {}
    capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), n_frames=n_frames, # type: ignore[arg-type]
        diagnostics_out=diagnostics,
    )

    for cam in (0, 1, 2):
        # RING PAYLOAD REMOVED FROM THE LIVE PATH, 2026-09-13. It was a
        # third of every calibration (10.6-11.5s of 31s on the Windows
        # rig) producing a number that `set_ring_boundary_offsets(None,
        # None)` then unconditionally refused to apply -- research data
        # a shipping product was paying for on every Start. The key
        # survives, reported as absent, so a package reader can tell
        # "not measured" from "field did not exist yet".
        assert diagnostics[cam]["ring_boundary_offset_payload"] is None
        assert diagnostics[cam]["ring_boundary_measurement"] is None
        assert diagnostics[cam]["ring_boundary_offset_accepted"] is False
        assert diagnostics[cam]["ring_boundary_offset_duration_s"] == 0.0

        # Board colour is UNAFFECTED and still derived live -- it is
        # applied to scoring, which is exactly the distinction that made
        # the ring measurement removable and this one not.
        color_payload = diagnostics[cam]["board_color_calibration_payload"]
        assert color_payload is not None
        assert color_payload["schema"] == "board-color-calibration-v1"
        assert color_payload["brightness_threshold_black_cream"]["value"] == 140.0
        assert color_payload["chroma_threshold"]["value"] == 40.0

    # Package-wide (identical across every camera's own diagnostics
    # entry), same duplication convention as calibration_total_duration_s.
    assert (diagnostics[0]["board_color_calibration_payload"]
            == diagnostics[1]["board_color_calibration_payload"])


def test_the_live_path_does_not_measure_ring_boundary_offsets_at_all(
    tmp_path, monkeypatch,
):
    """Replaces two tests deleted 2026-09-13 -- one that pinned the ring
    payload's `source_images`, one that pinned it being recorded even
    when rejected. Both described behaviour that no longer exists: the
    live ring-boundary-offset measurement was removed because it cost a
    third of every calibration (10.6-11.5s of 31s on the Windows rig) to
    produce a number `set_ring_boundary_offsets(None, None)` then
    unconditionally refused to apply.

    This is the guard that matters now: it must not come back by
    accident. Reinstating the call is a ~35% regression in calibration
    time that no other test would notice, because nothing downstream
    consumes the result.
    """
    import opendarts.calibration.ring_boundary_offset as rbo

    called = {"n": 0}

    def _explode(*a, **k):
        called["n"] += 1
        raise AssertionError(
            "the live path measured ring boundary offsets -- removed "
            "2026-09-13, and nothing applies the result"
        )

    # raising=False: the measurement moved to dev/ (2026-09-17) and is
    # no longer an attribute of this module at all. The stub stays as
    # the tripwire for it coming back -- if it ever does, this patches
    # it and the assertions below still mean what they say.
    monkeypatch.setattr(rbo, "measure_ring_boundary_offsets", _explode, raising=False)

    def fake_correspond(image_bgr, pre, *, orientation_hint_deg=None, results_out=None, **kw):
        if results_out is not None:
            results_out.append(_ok_result())
        return _OBJ, _PX

    monkeypatch.setattr(
        capture_daemon, "_capture_calibration_frames_local",
        lambda hub, n, **k: _frames(n, cams=(0, 1, 2)),
    )
    _install_fake_pipeline(monkeypatch, correspond=fake_correspond)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", lambda *a, **k: _fake_attempt())

    diagnostics: dict = {}
    capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), n_frames=4,  # type: ignore[arg-type]
        diagnostics_out=diagnostics,
    )

    assert called["n"] == 0
    for cam in (0, 1, 2):
        assert diagnostics[cam]["ring_boundary_offset_payload"] is None
    # And the phase is gone from the wall-clock accounting entirely,
    # rather than lingering as a permanent 0.00s line.
    assert "ring_boundary_offset" not in diagnostics[0]["phase_wall_s"]


def test_bootstrap_frame_selection_indices_are_a_real_contiguous_range_when_all_detect(
    tmp_path, monkeypatch,
):
    """V2 package schema, frame-selection provenance: when every captured frame is
    detected and corresponds successfully (no raw-extra frames, no
    rejections -- the common/simple case this synthetic harness
    produces), `frame_indices_used` must be the real, verified
    contiguous range [0, n_frames_used), not a placeholder,
    `raw_extra_frame_indices` must be the real, verified EMPTY list (no
    raw-pool frame is left over), and `n_raw_extra_frames_used` must be
    0 -- n_frames_detect's default comfortably exceeds n_frames=3 here,
    so nothing lands in the raw-extra pool."""
    n_frames = 3

    def fake_correspond(image_bgr, pre, *, orientation_hint_deg=None, results_out=None, **kw):
        if results_out is not None:
            results_out.append(_ok_result())
        return _OBJ, _PX

    monkeypatch.setattr(
        capture_daemon, "_capture_calibration_frames_local",
        lambda hub, n, **k: _frames(n, cams=(0, 1, 2)),
    )
    _install_fake_pipeline(monkeypatch, correspond=fake_correspond)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", lambda *a, **k: _fake_attempt())

    diagnostics: dict = {}
    capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), n_frames=n_frames, # type: ignore[arg-type]
        diagnostics_out=diagnostics,
    )

    for cam in (0, 1, 2):
        d = diagnostics[cam]
        assert d["frame_indices_used"] == list(range(d["n_frames_used"]))
        assert d["raw_extra_frame_indices"] == []
        assert d["n_raw_extra_frames_used"] == 0
        assert d["n_frames_raw_pool"] == d["n_frames_used"]


def test_bootstrap_frame_selection_indices_correctly_exclude_raw_extra_frames(
    tmp_path, monkeypatch,
):
    """DECOUPLED CAPTURE-VS-DETECT TARGET (n_frames > n_frames_detect):
    the raw-extra frames beyond `n_frames_detect` are captured but never
    run through detection at all -- `frame_indices_used` must correctly
    identify only the DETECTED subset's raw-pool positions, not simply
    `range(n_frames_used)`, and `raw_extra_frame_indices` must be its
    exact complement within the raw pool.
    `bootstrap_calibrations()`'s own pool-seeding code (see its "Seed the
    pool with this camera's own raw-only extra frames" comment) seeds
    the raw-extra frames into `pre_orientation_pool[cam]` BEFORE round
    1's own detected frames get appended, so with
    n_frames=15/n_frames_detect=10 the 5 raw-extra frames occupy pool
    positions [0, 5) and the 10 detected frames occupy [5, 15) --
    verified here as the real, current behavior (this module's own code
    comment says "order doesn't matter for CORRECTNESS," not "the order
    is unspecified/random" -- it's a deterministic consequence of this
    seed-before-round-1 sequencing, safe to assert precisely)."""
    n_frames = 15
    n_frames_detect = 10

    def fake_correspond(image_bgr, pre, *, orientation_hint_deg=None, results_out=None, **kw):
        if results_out is not None:
            results_out.append(_ok_result())
        return _OBJ, _PX

    monkeypatch.setattr(
        capture_daemon, "_capture_calibration_frames_local",
        lambda hub, n, **k: _frames(n, cams=(0,)),
    )
    _install_fake_pipeline(monkeypatch, correspond=fake_correspond)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", lambda *a, **k: _fake_attempt())

    diagnostics: dict = {}
    capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), # type: ignore[arg-type]
        n_frames=n_frames, n_frames_detect=n_frames_detect,
        diagnostics_out=diagnostics,
    )

    d = diagnostics[0]
    assert d["n_frames_raw_pool"] == 15
    assert d["n_frames_used"] == 10
    assert d["n_raw_extra_frames_used"] == 5
    assert d["frame_indices_used"] == list(range(5, 15))
    assert d["raw_extra_frame_indices"] == list(range(0, 5))
    # Every used index really does fall inside the raw pool this
    # package's own cam0_raw.mkv/PNGs would encode (bounds sanity, not
    # just the exact-order assertion above).
    assert all(0 <= i < d["n_frames_raw_pool"] for i in d["frame_indices_used"])
    assert len(set(d["frame_indices_used"])) == len(d["frame_indices_used"]) # no duplicates
    # The two lists are a real partition of the full raw pool -- disjoint
    # and together cover every index exactly once.
    assert set(d["frame_indices_used"]) & set(d["raw_extra_frame_indices"]) == set()
    assert set(d["frame_indices_used"]) | set(d["raw_extra_frame_indices"]) == set(
        range(d["n_frames_raw_pool"])
    )


# ---------------------------------------------------------------------
# TIMING (2026-08-21)
# to the packages and to the logging... how long it takes to calibrate
# (and for each camera)."
# ---------------------------------------------------------------------


def test_bootstrap_reports_real_total_and_per_camera_timing(tmp_path, monkeypatch):
    """Real, measured durations (not zero, not identical placeholders)
    land in diagnostics_out for every camera -- calibration_duration_s
    per camera, calibration_total_duration_s (the whole call) on every
    camera's own entry since diagnostics_out is strictly per-camera."""
    n_frames = 3

    def fake_correspond(image_bgr, pre, *, orientation_hint_deg=None, results_out=None, **kw):
        if results_out is not None:
            results_out.append(_ok_result())
        return _OBJ, _PX

    monkeypatch.setattr(
        capture_daemon, "_capture_calibration_frames_local",
        lambda hub, n, **k: _frames(n, cams=(0, 1, 2)),
    )
    _install_fake_pipeline(monkeypatch, correspond=fake_correspond)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", lambda *a, **k: _fake_attempt())

    diagnostics: dict = {}
    capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), n_frames=n_frames, # type: ignore[arg-type]
        diagnostics_out=diagnostics,
    )

    assert sorted(diagnostics) == [0, 1, 2]
    totals = set()
    for cam in (0, 1, 2):
        duration = diagnostics[cam]["calibration_duration_s"]
        total = diagnostics[cam]["calibration_total_duration_s"]
        assert isinstance(duration, float) and duration >= 0.0
        assert isinstance(total, float) and total >= 0.0
        # Every camera's own duration cannot exceed the whole call's total.
        assert duration <= total
        totals.add(total)
    # calibration_total_duration_s is the SAME value on every camera's
    # entry (one shared total, no separate top-level diagnostics slot).
    assert len(totals) == 1


def test_bootstrap_logs_timing_summary(tmp_path, monkeypatch, caplog):
    """The logging half of the same request -- one clear line naming
    the total and every camera's own duration, not just silently
    available in diagnostics_out for a caller that happens to ask."""
    n_frames = 3

    def fake_correspond(image_bgr, pre, *, orientation_hint_deg=None, results_out=None, **kw):
        if results_out is not None:
            results_out.append(_ok_result())
        return _OBJ, _PX

    monkeypatch.setattr(
        capture_daemon, "_capture_calibration_frames_local",
        lambda hub, n, **k: _frames(n, cams=(0, 1, 2)),
    )
    _install_fake_pipeline(monkeypatch, correspond=fake_correspond)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", lambda *a, **k: _fake_attempt())

    with caplog.at_level(logging.INFO, logger="opendarts.capture_daemon"):
        capture_daemon.bootstrap_calibrations(
            tmp_path, hub=object(), n_frames=n_frames, # type: ignore[arg-type]
        )

    timing_records = [r for r in caplog.records if "bootstrap_calibrations: TOTAL" in r.message]
    assert len(timing_records) == 1
    msg = timing_records[0].message
    assert "cam0=" in msg and "cam1=" in msg and "cam2=" in msg


def test_bootstrap_warns_on_rejected_board_color_threshold(tmp_path, monkeypatch, caplog):
    """Parity fix, 2026-08-21 (found by the OpenDarts port, ported back
    here): ring-boundary-offset already warns by name when a boundary
    is rejected; board-color used to only log the INFO measurement
    line regardless of whether a threshold was actually adopted --
    silent on rejection. Now warns per-threshold, same posture."""
    n_frames = 3

    def fake_correspond(image_bgr, pre, *, orientation_hint_deg=None, results_out=None, **kw):
        if results_out is not None:
            results_out.append(_ok_result())
        return _OBJ, _PX

    monkeypatch.setattr(
        capture_daemon, "_capture_calibration_frames_local",
        lambda hub, n, **k: _frames(n, cams=(0, 1, 2)),
    )
    _install_fake_pipeline(monkeypatch, correspond=fake_correspond)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", lambda *a, **k: _fake_attempt())
    _install_confident_boundary_and_color_mocks(monkeypatch, boundary_confident=True, color_confident=False)

    with caplog.at_level(logging.WARNING, logger="opendarts.capture_daemon"):
        capture_daemon.bootstrap_calibrations(
            tmp_path, hub=object(), n_frames=n_frames, # type: ignore[arg-type]
        )

    messages = [r.message for r in caplog.records]
    assert any("brightness threshold rejected" in m for m in messages)
    assert any("chroma threshold rejected" in m for m in messages)


def test_bootstrap_reports_section_timing_breakdown(tmp_path, monkeypatch, caplog):
    """The real answer to "time each section first":
    capture / per-camera detect / per-camera solve / the three
    post-loop live-derived-value sections all land in diagnostics_out
    AND in one clear log line, so a slow calibration can actually be
    diagnosed instead of just knowing the grand total."""
    n_frames = 3

    def fake_correspond(image_bgr, pre, *, orientation_hint_deg=None, results_out=None, **kw):
        if results_out is not None:
            results_out.append(_ok_result())
        return _OBJ, _PX

    monkeypatch.setattr(
        capture_daemon, "_capture_calibration_frames_local",
        lambda hub, n, **k: _frames(n, cams=(0, 1, 2)),
    )
    _install_fake_pipeline(monkeypatch, correspond=fake_correspond)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", lambda *a, **k: _fake_attempt())
    _install_confident_boundary_and_color_mocks(monkeypatch)

    diagnostics: dict = {}
    with caplog.at_level(logging.INFO, logger="opendarts.capture_daemon"):
        capture_daemon.bootstrap_calibrations(
            tmp_path, hub=object(), n_frames=n_frames, # type: ignore[arg-type]
            diagnostics_out=diagnostics,
        )

    for cam in (0, 1, 2):
        d = diagnostics[cam]
        for key in (
            "detect_duration_s", "solve_duration_s", "capture_duration_s",
            "motion_threshold_duration_s", "board_color_duration_s",
        ):
            assert key in d, f"cam{cam} missing {key}"
            assert isinstance(d[key], float) and d[key] >= 0.0
        # Every section's own share must fit inside this camera's total.
        assert d["detect_duration_s"] <= d["calibration_duration_s"]
        assert d["solve_duration_s"] <= d["calibration_duration_s"]

    section_records = [r for r in caplog.records if "SECTION TIMING" in r.message]
    assert len(section_records) == 1
    msg = section_records[0].message
    # The SECTION line is now the WITHIN-ROUNDS breakdown only -- its
    # detect/solve figures are sums across cameras running in parallel and
    # so legitimately exceed the span containing them, which is exactly
    # why they are kept out of the wall-clock accounting below.
    for term in ("capture=", "detect(sum)=", "solve(sum)="):
        assert term in msg, f"missing {term!r} in: {msg}"

    # PHASE WALL CLOCK: disjoint sequential spans that must ACCOUNT FOR
    # THE WHOLE RUN. Six buckets used to be reported with no statement of
    # coverage, and measured on the real rigs they silently left 27% of
    # one run and 22% of another unattributed -- the largest missing piece
    # being the orientation phase, which had no timer at all.
    phase_records = [r for r in caplog.records if "PHASE WALL CLOCK" in r.message]
    assert len(phase_records) == 1
    phase_msg = phase_records[0].message
    for term in (
        "setup+capture=", "orientation=", "rounds=", "best_of_n=",
        "motion_thresholds=", "board_color=", "unaccounted=",
    ):
        assert term in phase_msg, f"missing {term!r} in: {phase_msg}"
    # Removed 2026-09-13 along with the measurement itself -- a phase
    # that no longer runs must not keep a permanent 0.00s line, which
    # would read as "measured, took no time".
    assert "ring_boundary_offset=" not in phase_msg

    for cam in (0, 1, 2):
        phases = diagnostics[cam]["phase_wall_s"]
        assert "unaccounted" in phases
        total = diagnostics[cam]["calibration_total_duration_s"]
        # The whole point: the parts add up to the total, by construction.
        assert abs(sum(phases.values()) - total) < 0.05, (
            f"phases {phases} do not account for total {total}"
        )
        # And the remainder is genuinely small -- a phase quietly growing
        # its own untimed section should show up here, not be absorbed.
        assert abs(phases["unaccounted"]) <= max(0.25, 0.05 * total), (
            f"unaccounted {phases['unaccounted']}s of {total}s -- something "
            "real is running outside every measured span"
        )


# ---------------------------------------------------------------------
# DECOUPLED CAPTURE-VS-DETECT TARGET (2026-08-21) -- real perf scoping
# task. `CALIBRATION_N_FRAMES` (raw capture target, still ~50 by
# default) and `CALIBRATION_N_FRAMES_DETECT` (the NEW, separate target
# round 1's detect/solve/retry logic actually chases, default 10) used
# to be the same number -- every camera paid full landmark-detection
# cost for all 50 captured frames even though real A/B testing this
# session (99 real archived bg frames, replayed against all 99 real
# throws) proved N=10 detected frames gives identical scored segments
# to N=50, while `measure_ring_boundary_offsets()`'s treble_inner
# boundary specifically still needs the bigger ~50-frame RAW pool (it
# never touches detected landmarks at all, just raw pixels + the
# already-solved calibration -- same for board-color). See
# `CALIBRATION_N_FRAMES_DETECT`'s own dated module comment in
# capture_daemon.py for the full real-evidence writeup.
# ---------------------------------------------------------------------


def test_bootstrap_detect_stage_only_processes_n_frames_detect_not_the_full_raw_capture(
    tmp_path, monkeypatch
):
    """The actual point of this whole change: round 1's expensive
    landmark detection (`locate_pre_orientation_landmarks()` -- the
    pre-orientation stage, the real per-frame cost this task exists to
    cut) must only run on `n_frames_detect` frames, even though
    `n_frames` (the raw capture target) is captured much bigger. Also
    proves the raw pool itself still ends up the FULL `n_frames` size
    -- the un-detected frames are not discarded, just not detected."""
    n_frames = 50
    n_frames_detect = 10

    pre_stage_calls = {"n": 0}

    def counting_locate(frame, **kw):
        pre_stage_calls["n"] += 1
        return frame, _FAKE_PRE_OK

    def fake_correspond(image_bgr, pre, *, orientation_hint_deg=None, results_out=None, **kw):
        if results_out is not None:
            results_out.append(_ok_result())
        return _OBJ, _PX

    monkeypatch.setattr(
        capture_daemon, "_capture_calibration_frames_local",
        lambda hub, n, **k: _frames(n, cams=(0,)),
    )
    _install_fake_pipeline(monkeypatch, correspond=fake_correspond)
    # Override the pre-stage AFTER _install_fake_pipeline (which also
    # sets it) with the counting wrapper -- this test's own point of
    # control, same "last monkeypatch.setattr wins" pattern every other
    # test in this file uses for its one overridden piece.
    monkeypatch.setattr(capture_daemon, "locate_pre_orientation_landmarks", counting_locate)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", lambda *a, **k: _fake_attempt())

    diagnostics: dict = {}
    result = capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), n_frames=n_frames, n_frames_detect=n_frames_detect, # type: ignore[arg-type]
        diagnostics_out=diagnostics,
    )

    assert 0 in result
    # Only 10 frames were ever run through the expensive detector...
    assert pre_stage_calls["n"] == n_frames_detect
    assert diagnostics[0]["n_frames_used"] == n_frames_detect
    # ...but the full 50-frame raw pool is still there for the two
    # post-loop sections that don't need detection at all.
    assert diagnostics[0]["n_frames_raw_pool"] == n_frames


def test_board_color_receives_the_full_raw_pool_not_just_detected_frames(
    tmp_path, monkeypatch
):
    """The other half of the decoupling: ring-boundary-offset and
    board-color read raw pixels only (never the `_pre` half of
    `pre_orientation_pool`'s tuples -- confirmed by reading their real
    consumption code before making this change), so they must NOT be
    starved down to `n_frames_detect` -- they need to see the FULL
    `n_frames` raw pool (detected frames PLUS the raw-only "extra"
    ones), same as before this task's change."""
    import opendarts.calibration.ring_boundary_offset as rbo
    import opendarts.geometry.board_color_calibration as bcc
    from opendarts.calibration.ring_boundary_offset import (
        BoundaryMeasurement,
        RingBoundaryOffsetResult,
    )
    from opendarts.geometry.board_color_calibration import (
        BoardColorCalibrationResult,
        ThresholdDerivation,
    )

    n_frames = 50
    n_frames_detect = 10

    def fake_correspond(image_bgr, pre, *, orientation_hint_deg=None, results_out=None, **kw):
        if results_out is not None:
            results_out.append(_ok_result())
        return _OBJ, _PX

    monkeypatch.setattr(
        capture_daemon, "_capture_calibration_frames_local",
        lambda hub, n, **k: _frames(n, cams=(0, 1, 2)),
    )
    _install_fake_pipeline(monkeypatch, correspond=fake_correspond)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", lambda *a, **k: _fake_attempt())

    seen_ring_pool_sizes: dict = {}

    def fake_measure_ring(calibrations, bg_frames_per_camera):
        for cam, frames in bg_frames_per_camera.items():
            seen_ring_pool_sizes[cam] = len(frames)

        def boundary(name, reg_radius, measured):
            return BoundaryMeasurement(
                boundary=name, regulation_radius_mm=reg_radius,
                measured_radius_mm=measured,
                offset_mm=(reg_radius - measured) if measured is not None else None,
                mad_mm=0.3, n_samples_used=20, n_samples_rejected=0,
                n_angles_attempted=20, per_camera={}, confidence=0.9,
            )
        return RingBoundaryOffsetResult(boundaries={
            "treble_inner": boundary("treble_inner", 107.0, 105.3),
            "double_inner": boundary("double_inner", 170.0, 168.8),
        })
    # raising=False: the measurement moved to dev/ (2026-09-17) and is
    # no longer an attribute of this module at all. The stub stays as
    # the tripwire for it coming back -- if it ever does, this patches
    # it and the assertions below still mean what they say.
    monkeypatch.setattr(rbo, "measure_ring_boundary_offsets", fake_measure_ring, raising=False)

    color_call_count = {"n": 0}

    def fake_collect_color_samples(package_id, calibrations, bg_images, patch_radius=None,
                                   points=None, projected_px_by_camera=None):
        color_call_count["n"] += 1
        return ["fake_sample"]

    def fake_derive_thresholds(samples, patch_radius=None):
        return BoardColorCalibrationResult(
            brightness_threshold=ThresholdDerivation(
                value=140.0, low_group_stat=100.0, high_group_stat=180.0,
                low_group_n=10, high_group_n=10, gap=80.0, confidence="high",
            ),
            chroma_threshold=ThresholdDerivation(
                value=40.0, low_group_stat=10.0, high_group_stat=60.0,
                low_group_n=10, high_group_n=10, gap=50.0, confidence="high",
            ),
            patch_radius=8, n_samples=len(samples), n_packages=1,
            n_packages_attempted=1, accuracy_single_camera=None,
            accuracy_majority_vote=None,
        )
    monkeypatch.setattr(bcc, "collect_color_samples", fake_collect_color_samples)
    monkeypatch.setattr(bcc, "derive_thresholds", fake_derive_thresholds)

    capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), n_frames=n_frames, n_frames_detect=n_frames_detect, # type: ignore[arg-type]
    )

    # Ring-boundary-offset is no longer measured live at all (2026-09-13
    # -- see bootstrap_calibrations()'s own removal comment), so it reads
    # no pool. Asserted rather than dropped: this test's REAL subject is
    # that a full-pool consumer gets the full pool, and board-color is
    # now the only live one -- if the ring measurement ever comes back,
    # this line is where it has to be re-decided.
    assert seen_ring_pool_sizes == {}
    # collect_color_samples() is called once per raw pool frame per
    # camera (see bootstrap_calibrations()'s own board-color loop) --
    # 3 cameras x 50 raw frames, not 3 x 10. THIS is the decoupling claim
    # the test exists for, and it is unchanged.
    assert color_call_count["n"] == n_frames * 3


def test_bootstrap_calibration_package_and_raw_pool_are_consistently_normalized(
    tmp_path, monkeypatch
):
    """Two real, live-active bugs a verifier pass found 2026-08-21 in the
    DECOUPLED CAPTURE-VS-DETECT TARGET change, both fixed here, both
    regression-guarded by this one test:

    Bug 1: `locate_pre_orientation_landmarks()` (oriented_landmarks.py)
    returns its frame AFTER running it through
    `normalise_illuminant()` (grey-world white balance, so an absolute
    HSV threshold downstream means the same thing regardless of the
    camera's white-balance drift) -- that is what a DETECTED pool
    entry's `pb` half already is. The raw-only "extra" entries (the
    `n_frames - n_frames_detect` frames captured but deliberately never
    detected) must get the SAME normalization when seeded into the pool
    -- otherwise ring-boundary-offset/board-color (an ABSOLUTE HSV
    computation) silently consume a raw/normalized mix.

    Bug 2 (REPLAY, docs/DESIGN.md's "Replay is the source of truth"): a saved calibration
    package's raw frame COUNT must match `n_frames_raw_pool` (the FULL
    raw pool the live derivations actually consumed), not just the
    detected subset -- otherwise the package can't reproduce what was
    actually computed.

    Deliberately uses the REAL `locate_pre_orientation_landmarks` (NOT
    the identity-function mock `_install_fake_pipeline()`/every other
    test in this file uses) -- that mock is exactly what would erase
    Bug 1's inconsistent-normalization symptom, per the verifier's own
    finding."""
    from opendarts.calibration.oriented_landmarks import normalise_illuminant
    from opendarts.capture.calibration_package import load_calibration_package

    # This test is about normalisation consistency, not orientation:
    # it deliberately uses the REAL landmark locator, so the orientation
    # solve must be stubbed or it refuses on synthetic frames.
    monkeypatch.setattr(
        capture_daemon, "ring_correlation_orientation_for_camera",
        lambda frames, **kw: _confident_session_result(frames),
    )

    n_frames = 20
    n_frames_detect = 5

    # A real, non-degenerate color (distinct per-channel means) so
    # normalise_illuminant() applies a genuine, non-trivial gain -- a
    # uniform frame like this also reliably fails real seed-ellipse
    # detection (no double-ring edges to find), so every "detected"
    # frame deterministically takes locate_pre_orientation_landmarks()'s
    # own early ok=False return path (still white-balance-normalized,
    # per that function's own docstring/code -- the normalization
    # happens before the seed-ellipse search, not after).
    def _frame():
        return np.full((64, 64, 3), (50, 100, 150), dtype=np.uint8)

    def fake_local(hub, n, **k):
        return {0: [_frame() for _ in range(n)]}

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_local)
    # locate_pre_orientation_landmarks is deliberately left REAL here.

    def fake_correspond(image_bgr, pre, *, orientation_hint_deg=None, results_out=None, **kw):
        if results_out is not None:
            results_out.append(_ok_result())
        return _OBJ, _PX
    monkeypatch.setattr(capture_daemon, "correspond_landmarks_from_pre_orientation", fake_correspond)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", lambda *a, **k: _fake_attempt())
    # LIVE-DERIVED FOCAL LENGTH, 2026-08-26 -- this test bypasses
    # _install_fake_pipeline() entirely (deliberately uses the REAL
    # locate_pre_orientation_landmarks, see the docstring above), so its
    # focal-length fake isn't installed either -- same "harmless,
    # always-confident stand-in" needed here directly.
    monkeypatch.setattr(
        capture_daemon, "derive_focal_length_from_oriented_results",
        lambda results, principal_point, min_frames=1: FocalLengthResult(
            ok=True, focal_length_px=900.0, reason="test-fake",
            n_points_used=20, n_frames_used=max(1, len(results)),
        ),
    )

    diagnostics: dict = {}
    package_out: dict = {}
    capture_daemon.bootstrap_calibrations(
        tmp_path, hub=object(), n_frames=n_frames, n_frames_detect=n_frames_detect, # type: ignore[arg-type]
        diagnostics_out=diagnostics,
        calibration_package_root=tmp_path / "pkgs",
        calibration_package_out=package_out,
        calibration_package_blocking=True,
    )

    assert 0 in diagnostics
    assert diagnostics[0]["n_frames_used"] == n_frames_detect # only 5 detected
    assert diagnostics[0]["n_frames_raw_pool"] == n_frames # but 20 in the raw pool

    loaded = load_calibration_package(package_out["package_dir"])
    assert loaded.raw_frames[0] is not None
    # Bug 2: the saved package's raw frame count matches the FULL raw
    # pool (n_frames_raw_pool), not just the n_frames_detect subset that
    # went through real detection.
    assert len(loaded.raw_frames[0]) == n_frames

    # Bug 1: every saved raw frame -- detected AND raw-extra alike -- is
    # byte-identical to normalise_illuminant() applied directly to the
    # original captured frame. If a raw-extra frame had been seeded
    # un-normalized, this would fail for most of the 20 (only 5 would
    # match, the ones that went through real detection).
    expected = normalise_illuminant(_frame())
    for saved_frame in loaded.raw_frames[0]:
        assert np.array_equal(saved_frame, expected)


def test_best_of_n_does_not_null_out_the_frame_selection_record(tmp_path, monkeypatch):
    """The regression this exists for, found on the rigs 2026-09-12.

    BEST-OF-N REPROJECTION ATTEMPTS (2026-08-30) appends its attempt
    frames to `pre_orientation_pool[cam]` but keeps their detections in a
    LOCAL list, never in `accumulated_detections[cam]`. The old
    provenance code re-derived the mapping between the two by scanning
    the pool for `pre is not None` and then asserting the two lengths
    matched -- so from that day on the check failed on EVERY camera of
    EVERY calibration (a fixed `(n_reprojection_attempts - 1) *
    n_frames_detect` gap: exactly 20 at the shipped 5 and 5), logged a
    scary warning, and wrote `frame_indices_used: null` and
    `raw_extra_frame_indices: null` into every calibration package.

    The warning was real; what was inconsistent was the check itself. The
    mapping is now recorded where it is created, so there is no invariant
    left to violate -- and the frame-selection record actually survives.
    """
    import logging

    n_frames = 12
    n_frames_detect = 4

    def fake_correspond(image_bgr, pre, *, orientation_hint_deg=None, results_out=None, **kw):
        if results_out is not None:
            results_out.append(_ok_result())
        return _OBJ, _PX

    monkeypatch.setattr(
        capture_daemon, "_capture_calibration_frames_local",
        lambda hub, n, **k: _frames(n, cams=(0,)),
    )
    _install_fake_pipeline(monkeypatch, correspond=fake_correspond)
    monkeypatch.setattr(capture_daemon, "calibrate_camera", lambda *a, **k: _fake_attempt())

    diagnostics: dict = {}
    caplog_records: list = []

    class _Collect(logging.Handler):
        def emit(self, record):
            caplog_records.append(record.getMessage())

    logger = logging.getLogger("opendarts.capture_daemon")
    handler = _Collect()
    logger.addHandler(handler)
    try:
        capture_daemon.bootstrap_calibrations(
            tmp_path, hub=object(), # type: ignore[arg-type]
            n_frames=n_frames, n_frames_detect=n_frames_detect,
            n_reprojection_attempts=5, # the shipped value -- what the rigs run
            diagnostics_out=diagnostics,
        )
    finally:
        logger.removeHandler(handler)

    d = diagnostics[0]
    assert d["frame_indices_used"] is not None, (
        "the frame-selection record was dropped -- this is the bug: every "
        "package written since best-of-N landed recorded null here"
    )
    assert d["raw_extra_frame_indices"] is not None
    assert not any("invariant check failed" in m for m in caplog_records), (
        "the invariant warning fired -- it fired on every real calibration "
        "on both rigs, on every camera, and was never actionable"
    )

    # The record must be true, not merely present: every index names a
    # real pool slot, and used/extra partition the pool exactly once.
    pool_n = d["n_frames_raw_pool"]
    used = d["frame_indices_used"]
    extra = d["raw_extra_frame_indices"]
    assert all(0 <= i < pool_n for i in used), f"{used} outside a {pool_n}-frame pool"
    assert sorted(used + extra) == list(range(pool_n)), (
        "used and extra must partition the raw pool exactly once"
    )
    assert len(set(used)) == len(used), "duplicate indices in the used record"
