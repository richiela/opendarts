"""Tests for `opendarts.live.capture_daemon._check_cross_camera_ellipse_
aspect_consistency()` -- the cross-camera ellipse-aspect DEFENSE-IN-
DEPTH tripwire, 2026-09-03. See `capture_daemon.
ELLIPSE_ASPECT_OUTLIER_DEVIATION_THRESHOLD`'s own module-level comment
block for the full design/threshold derivation, and `opendarts.calibration.
landmark_detection._mask_for_detection()`'s own docstring for the real
primary fix this tripwire is a safety net for (a two-pass, ROI-
restricted adaptive colour segmentation -- NOT this tripwire itself,
which cannot and does not attempt to fix a bad seed ellipse, only to
detect and hold one back from contaminating a calibration).

Fully synthetic, multi-camera, camera-DIFFERENTIATED fakes, following
the SAME pattern `tests/test_capture_daemon_rig_consensus.py`'s own
module docstring already establishes (and explains why it's needed --
none of `tests/test_capture_daemon_oriented_wiring.py`'s existing
single-camera-per-test fakes can exercise a genuinely multi-camera
cross-check at all): each fake frame carries its own camera index AND
an "outlier this round" flag as distinct pixel values (`frame[0,0,0]`,
`frame[0,0,1]`), threaded through a fake `locate_pre_orientation_
landmarks()` into a real `Ellipse` with a controlled aspect ratio, so
the SAME synthetic bootstrap call can differentiate "this camera's seed
ellipse is fine" from "this camera's seed ellipse is an outlier" on a
per-round, per-camera basis.

Forces `ORIENTATION_METHOD_DIGIT_COUNT` (the simplest orientation
method -- hint established live, in-loop, via `aggregate_session_
orientation()`, no separate pre-phase) and never seeds `ring_geometry_
fallback.json`, so Mode A (`_resolve_mode_a_orientation_after_round_
one()`) never fires -- this file tests the tripwire in isolation from
that entirely separate mechanism, matching this project's own "one
concern per test file" convention.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from tests.conftest import stub_confident_orientation

import opendarts.live.capture_daemon as capture_daemon
from opendarts.calibration.focal_length import FocalLengthResult
from opendarts.calibration.landmark_detection import Ellipse
from opendarts.calibration.oriented_landmarks import PreOrientationLandmarks
from opendarts.pipeline import CalibrationAttempt, CameraCalibration

_OBJ = np.zeros((4, 3))
_PX = np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]])

# Healthy aspect ~2.0 (matches this rig's own real measured healthy
# range, see ELLIPSE_ASPECT_OUTLIER_DEVIATION_THRESHOLD's own comment
# block); "outlier" aspect ~2.5 -- a real, measured magnitude this
# rig's own pre-ROI-fix cam0 failures showed even in their PARTIALLY
# corrected state (a rejected earlier fix's own measured range, kept
# here only as a realistic magnitude, not a live constant).
_HEALTHY_ELLIPSE = Ellipse(cx=100.0, cy=100.0, major_axis_px=200.0, minor_axis_px=100.0, angle_deg=0.0)
_OUTLIER_ELLIPSE = Ellipse(cx=100.0, cy=100.0, major_axis_px=250.0, minor_axis_px=100.0, angle_deg=0.0)
# A small, genuinely-healthy per-camera spread -- real foreshortening,
# not noise, per this rig's own real measured healthy ceiling (~0.03-
# 0.05) -- deliberately UNDER the threshold.
_HEALTHY_ELLIPSE_2 = Ellipse(cx=100.0, cy=100.0, major_axis_px=204.0, minor_axis_px=100.0, angle_deg=0.0)


@pytest.fixture(autouse=True)
def _use_digit_count_orientation_method(monkeypatch):
    stub_confident_orientation(monkeypatch, capture_daemon)


def _tagged_frames(n_frames: int, cams: tuple[int, ...], *, outlier_cams: frozenset[int] = frozenset()) -> dict[int, list[np.ndarray]]:
    out = {}
    for c in cams:
        frame = np.zeros((10, 10, 3), np.uint8)
        frame[0, 0, 0] = c
        frame[0, 0, 1] = 1 if c in outlier_cams else 0
        out[c] = [frame.copy() for _ in range(n_frames)]
    return out


def _pre_for_frame(frame) -> PreOrientationLandmarks:
    cam = int(frame[0, 0, 0])
    outlier = bool(frame[0, 0, 1])
    ellipse = _OUTLIER_ELLIPSE if outlier else _HEALTHY_ELLIPSE
    return PreOrientationLandmarks(
        ok=True, reason="ok", ellipse=ellipse, seed_ellipse=ellipse, bull_px=(float(cam), 0.0),
        normalised_bull_radius=0.0, profile=None, phase_deg=0.0,
        spoke_score=1.0, phase_confidence=5.0, notes=[],
    )


def _fake_locate_pre_orientation_landmarks(frame, **kw):
    return frame, _pre_for_frame(frame)


def _fake_attempt() -> CalibrationAttempt:
    calib = CameraCalibration(
        camera_matrix=np.eye(3), dist_coeffs=np.zeros(5), rvec=np.zeros(3),
        tvec=np.array([0.0, 0.0, 1000.0]), pnp_result=None, landmark_spread_ok=True,
    )
    return CalibrationAttempt(ok=True, calibration=calib, pnp_result=None, reason="")


def _install_common_fakes(monkeypatch):
    monkeypatch.setattr(
        capture_daemon, "locate_pre_orientation_landmarks",
        _fake_locate_pre_orientation_landmarks,
    )

    def fake_correspond(image_bgr, pre, *, orientation_hint_deg=None, results_out=None, **kw):
        from opendarts.calibration.oriented_landmarks import OrientedLandmarkResult
        if results_out is not None:
            results_out.append(OrientedLandmarkResult(
                ok=True, reason="ok", quad_px=_PX.copy(), object_points_mm=_OBJ.copy(),
                phase_confidence=2.5, colour_margin=0.9, orientation_ambiguous=False,
            ))
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


def test_two_agreeing_cameras_never_flagged(tmp_path, monkeypatch):
    """Baseline sanity: cam0/cam1 both healthy (aspect ~2.0), never an
    outlier, never nulled -- both solve in round 1, no warning."""
    def fake_capture(hub, n, **k):
        return _tagged_frames(n, cams=(0, 1))

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_capture)
    _install_common_fakes(monkeypatch)

    diagnostics: dict = {}
    result = capture_daemon.bootstrap_calibrations(
        tmp_path / "snap", hub=object(), n_frames=5, n_frames_detect=5,
        max_frames=5, retry_batch_size=5,  # type: ignore[arg-type]
        calibration_package_root=None, diagnostics_out=diagnostics,
    )
    assert sorted(result) == [0, 1]


def test_small_healthy_spread_below_threshold_does_not_fire(tmp_path, monkeypatch, caplog):
    """A genuinely small (real, sub-threshold) per-camera spread must
    NOT trigger the tripwire -- foreshortening, not a defect."""
    def fake_capture(hub, n, **k):
        cams = (0, 1)
        out = {}
        for c, ellipse in zip(cams, (_HEALTHY_ELLIPSE, _HEALTHY_ELLIPSE_2)):
            frame = np.zeros((10, 10, 3), np.uint8)
            frame[0, 0, 0] = c
            out[c] = [frame.copy() for _ in range(n)]
        return out

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_capture)
    _install_common_fakes(monkeypatch)

    # Route cam1's frames to the second, near-but-not-quite-identical
    # healthy ellipse via a per-camera override on top of the shared fake.
    def _pre_for_frame_two_healthy(frame):
        cam = int(frame[0, 0, 0])
        ellipse = _HEALTHY_ELLIPSE if cam == 0 else _HEALTHY_ELLIPSE_2
        return PreOrientationLandmarks(
            ok=True, reason="ok", ellipse=ellipse, seed_ellipse=ellipse, bull_px=(float(cam), 0.0),
            normalised_bull_radius=0.0, profile=None, phase_deg=0.0,
            spoke_score=1.0, phase_confidence=5.0, notes=[],
        )

    monkeypatch.setattr(
        capture_daemon, "locate_pre_orientation_landmarks",
        lambda frame, **kw: (frame, _pre_for_frame_two_healthy(frame)),
    )

    with caplog.at_level(logging.WARNING, logger="opendarts.live.capture_daemon"):
        result = capture_daemon.bootstrap_calibrations(
            tmp_path / "snap", hub=object(), n_frames=5, n_frames_detect=5,
            max_frames=5, retry_batch_size=5,  # type: ignore[arg-type]
            calibration_package_root=None, diagnostics_out={},
        )
    assert sorted(result) == [0, 1]
    assert not any("seed-ellipse aspect ratio" in r.message for r in caplog.records)


def test_fewer_than_two_confident_cameras_is_a_noop(tmp_path, monkeypatch, caplog):
    """A single-camera event has no sibling to compare against -- the
    tripwire must not fire and must not block that camera's own
    progress (spec's own explicit <2-camera edge case)."""
    def fake_capture(hub, n, **k):
        return _tagged_frames(n, cams=(0,), outlier_cams=frozenset({0}))

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_capture)
    _install_common_fakes(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="opendarts.live.capture_daemon"):
        result = capture_daemon.bootstrap_calibrations(
            tmp_path / "snap", hub=object(), n_frames=5, n_frames_detect=5,
            max_frames=5, retry_batch_size=5,  # type: ignore[arg-type]
            calibration_package_root=None, diagnostics_out={},
        )
    # Solves normally -- an "outlier" ellipse with no sibling to compare
    # against is never even evaluated by this check.
    assert sorted(result) == [0]
    assert not any("seed-ellipse aspect ratio" in r.message for r in caplog.records)


def test_outlier_camera_flagged_and_this_rounds_detections_marked_not_usable(tmp_path, monkeypatch, caplog):
    """The core mechanism: cam2's own seed ellipse is an outlier vs its
    2 healthy siblings this round -- a real WARNING must name it, and
    its round-1 detections must genuinely be marked not-usable, proven
    directly: `_try_solve()`'s own pre-existing "not enough usable
    detections" warning fires immediately afterward with 0/2 valid (not
    2/2, which is what round 1's own unmodified processing actually
    produced before this tripwire intervened)."""
    call_count = {"n": 0}

    def fake_capture(hub, n, **k):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _tagged_frames(n, cams=(0, 1, 2), outlier_cams=frozenset({2}))
        # Round 2+: cam2 stops being an outlier (would only be reached if
        # this test's own round-1 assertion below is wrong).
        return _tagged_frames(n, cams=(2,), outlier_cams=frozenset())

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_capture)
    _install_common_fakes(monkeypatch)

    diagnostics: dict = {}
    with caplog.at_level(logging.WARNING, logger="opendarts.live.capture_daemon"):
        result = capture_daemon.bootstrap_calibrations(
            tmp_path / "snap", hub=object(), n_frames=2, n_frames_detect=2,
            max_frames=2, retry_batch_size=2,  # type: ignore[arg-type]
            calibration_package_root=None, diagnostics_out=diagnostics,
        )

    warnings = [r for r in caplog.records if "seed-ellipse aspect ratio" in r.message]
    assert len(warnings) == 1, [r.message for r in caplog.records]
    assert "cam2" in warnings[0].message

    # cam0/cam1 solved normally in round 1 (never flagged, never nulled).
    assert 0 in result
    assert 1 in result

    # The real, provable mechanism proof: after nulling, `_try_solve()`
    # genuinely re-ran against the now-corrected (empty) pool and
    # genuinely failed to average (0 valid of 2 needed) -- NOT that it
    # silently kept using the outlier detections. This is the second
    # WARNING, emitted by the pre-existing `_try_solve()` machinery this
    # tripwire feeds into, unmodified by this task.
    give_up_warnings = [
        r for r in caplog.records
        if "giving up, frame cap reached" in r.message and "cam2" in r.message
    ]
    assert len(give_up_warnings) == 1, [r.message for r in caplog.records]
    assert "only 0/2 total captured frame(s)" in give_up_warnings[0].message

    # Honest, real, stated residual limitation of this test's own setup
    # (NOT a gap this tripwire introduces): with max_frames=2 and no
    # further real data ever offered (this test's own fake_capture has
    # nothing left to give cam2 once round 1's data is nulled), the
    # pre-existing "best-of-N reprojection attempts" persistence
    # (`best_calibration[cam]`, unmodified by this task) still returns
    # whatever it computed BEFORE this tripwire ever got a chance to
    # intervene -- round 1's own initial, now-known-contaminated attempt.
    # This tripwire delays acceptance of a KNOWN-bad round's own
    # in-progress work; it cannot conjure clean data a camera never gets
    # another chance to provide. See `test_outlier_camera_recovers_
    # once_next_rounds_detections_are_healthy` below for the case where
    # real data DOES let it self-correct.
    assert 2 in result


def test_outlier_camera_recovers_once_next_rounds_detections_are_healthy(tmp_path, monkeypatch, caplog):
    """The real 'delay, never discard' proof: cam2 is an outlier ONLY in
    round 1 -- once round 2 supplies genuinely healthy detections (which
    now dominate its accumulated pool), it solves successfully, and the
    tripwire does NOT fire again on round 2's own healthy data."""
    call_count = {"n": 0}

    def fake_capture(hub, n, **k):
        call_count["n"] += 1
        if call_count["n"] == 1:
            # Round 1 (the initial `_capture(n_frames)` call): all 3
            # cameras, cam2 tagged as this round's outlier.
            return _tagged_frames(n, cams=(0, 1, 2), outlier_cams=frozenset({2}))
        # Round 2+ (`new_batch = _capture(retry_batch_size)`): only cam2
        # is still `remaining` by then (cam0/cam1 already solved in
        # round 1) -- healthy this time.
        return _tagged_frames(n, cams=(2,), outlier_cams=frozenset())

    monkeypatch.setattr(capture_daemon, "_capture_calibration_frames_local", fake_capture)
    _install_common_fakes(monkeypatch)

    diagnostics: dict = {}
    with caplog.at_level(logging.WARNING, logger="opendarts.live.capture_daemon"):
        result = capture_daemon.bootstrap_calibrations(
            tmp_path / "snap", hub=object(), n_frames=2, n_frames_detect=2,
            max_frames=10, retry_batch_size=5,  # type: ignore[arg-type]
            calibration_package_root=None, diagnostics_out=diagnostics,
        )

    warnings = [r for r in caplog.records if "seed-ellipse aspect ratio" in r.message]
    assert len(warnings) == 1, [r.message for r in caplog.records]
    assert "cam2" in warnings[0].message

    # All three cameras eventually solve -- cam2's own recovery is the
    # real point of this test.
    assert sorted(result) == [0, 1, 2]
    # cam2's own n_frames_used reflects round 2's healthy contribution
    # dominating its accumulated pool (2 nulled round-1 + 5 healthy
    # round-2 = 7 total captured, 5 valid, clears
    # _min_calibration_frames_required(7) == 4) -- not round 1's own
    # (contaminated, never-counted) 2 frames alone.
    assert diagnostics[2]["n_frames_used"] == 7


# ---------------------------------------------------------------------
# STABILITY CORROBORATION (2026-09-13). The cross-camera deviation alone
# is not evidence of a bad ellipse -- it assumes every camera sees the
# ring from a similar angle. On the Windows rig's wheel one camera sits
# materially more face-on (1.616 against its siblings' 1.675-1.679, a
# deviation of 0.058 straddling the 0.05 gate) and its fit was verified
# CORRECT against the actual ring. What separates that from a
# fragmenting mask is stability: measured over 8 frames per camera on
# the real rig, every camera including the outlier held a within-camera
# spread of 0.0049-0.0081.
# ---------------------------------------------------------------------

def _ellipse_with_aspect(aspect: float) -> Ellipse:
    return Ellipse(cx=100.0, cy=100.0, major_axis_px=100.0 * aspect,
                   minor_axis_px=100.0, angle_deg=0.0)


def _install_aspect_script(monkeypatch, per_cam_aspects: dict):
    """`per_cam_aspects[cam]` is the list of aspects that camera's frames
    report, cycled -- so a camera can be made self-consistent or jittery
    independently of how far it sits from its siblings."""
    def fake_locate(frame, **kw):
        cam = int(frame[0, 0, 0])
        idx = int(frame[0, 0, 2])
        seq = per_cam_aspects[cam]
        ellipse = _ellipse_with_aspect(seq[idx % len(seq)])
        return frame, PreOrientationLandmarks(
            ok=True, reason="ok", ellipse=ellipse, seed_ellipse=ellipse,
            bull_px=(float(cam), 0.0), normalised_bull_radius=0.0, profile=None,
            phase_deg=0.0, spoke_score=1.0, phase_confidence=5.0, notes=[],
        )
    monkeypatch.setattr(capture_daemon, "locate_pre_orientation_landmarks", fake_locate)


def _indexed_frames(n_frames: int, cams: tuple) -> dict:
    out = {}
    for c in cams:
        frames = []
        for i in range(n_frames):
            f = np.zeros((10, 10, 3), np.uint8)
            f[0, 0, 0] = c
            f[0, 0, 2] = i
            frames.append(f)
        out[c] = frames
    return out


def _run(tmp_path, monkeypatch, caplog, per_cam_aspects, n=4):
    monkeypatch.setattr(
        capture_daemon, "_capture_calibration_frames_local",
        lambda hub, k, **kw: _indexed_frames(k, cams=(0, 1, 2)),
    )
    _install_common_fakes(monkeypatch)
    _install_aspect_script(monkeypatch, per_cam_aspects)
    diagnostics: dict = {}
    # INFO on the module's OWN logger name -- the "persistently different
    # view" line is INFO, and the sibling tests' logger= argument names a
    # logger this module does not actually use, so it only ever caught
    # WARNINGs by propagation.
    with caplog.at_level(logging.INFO, logger="opendarts.capture_daemon"):
        capture_daemon.bootstrap_calibrations(
            tmp_path / "snap", hub=object(), n_frames=n, n_frames_detect=n,
            max_frames=n, retry_batch_size=n,  # type: ignore[arg-type]
            calibration_package_root=None, diagnostics_out=diagnostics,
        )
    return caplog.records


def test_a_self_consistent_camera_that_differs_from_its_siblings_is_kept(
    tmp_path, monkeypatch, caplog,
):
    """The real rig case. cam1 reads 1.616 against 1.675/1.679 -- over
    the 0.05 gate -- but every one of its own frames agrees. That is a
    camera placed differently, and its frames must NOT be thrown away:
    doing so cost ~16s per calibration and discarded 5 checked frames in
    favour of 25 that the check never even ran on (the retry round has
    only one camera left, below MIN_CAMERAS_FOR_ELLIPSE_ASPECT_CHECK)."""
    records = _run(tmp_path, monkeypatch, caplog, {
        0: [1.675], 1: [1.616], 2: [1.679],
    })
    assert not [r for r in records if "is an outlier" in r.message], (
        "a self-consistent camera must not be treated as a fragmenting mask"
    )
    kept = [r for r in records if "persistently different VIEW" in r.message]
    assert len(kept) >= 1, [r.message for r in records]
    assert "cam1" in kept[0].message
    # It must still be REPORTED -- silence would hide a genuinely
    # misaligned camera on a rig meant to be symmetric.
    assert "placement" in kept[0].message


def test_an_unstable_camera_in_the_same_band_is_still_flagged(
    tmp_path, monkeypatch, caplog,
):
    """Same cross-camera deviation, but cam1's own frames scatter well
    past the measured healthy ceiling -- the signature of a mask that
    fragments frame by frame, which is exactly what the tripwire is
    for."""
    records = _run(tmp_path, monkeypatch, caplog, {
        0: [1.675], 1: [1.600, 1.630, 1.600, 1.630], 2: [1.679],
    })
    flagged = [r for r in records if "is an outlier" in r.message]
    assert len(flagged) == 1, [r.message for r in records]
    assert "cam1" in flagged[0].message


def test_a_gross_deviation_fires_even_when_perfectly_stable(
    tmp_path, monkeypatch, caplog,
):
    """The safety valve on the stability rule. The healthy side of
    ELLIPSE_ASPECT_INSTABILITY_THRESHOLD is measured; the failing side is
    not, and a mask could in principle fragment consistently. Above the
    original investigation's own quoted failure floor, magnitude alone
    is enough."""
    records = _run(tmp_path, monkeypatch, caplog, {
        0: [2.000], 1: [2.000], 2: [2.500],  # deviation 0.5, rock stable
    })
    flagged = [r for r in records if "is an outlier" in r.message]
    assert len(flagged) == 1, [r.message for r in records]
    assert "cam2" in flagged[0].message
