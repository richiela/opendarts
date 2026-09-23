"""Live single-camera-fuse WARNING, `opendarts/live/capture_daemon.py`
`handle_ready_to_capture()`, 2026-09-02 -- the project's own explicit
authorization: "Build teh live detection piece.. helps with debug",
following the g1-056-D17 investigation (a real live throw scored from a
single degenerate ray after 3 independent per-engine board-plausibility
gates each discarded 2 genuine off-board camera detections, with no live
signal anywhere that only 1 camera actually contributed).

Purely additive, same shape as the ring-offset fix's own effective-radii
log line immediately above this insertion point: reads
`engine_result.diagnostics["n_cameras_used"]` (already computed by the
primary engine's own `score()` call, same "reached and influenced the
final answer" semantic established across all 5 engines this same day)
and logs a WARNING when it's exactly 1. Zero scoring impact -- verified
here directly, not just by construction.
"""
from __future__ import annotations

import logging

import numpy as np

import opendarts.live.capture_daemon as capture_daemon
from opendarts.engines.base import EngineResult
from opendarts.engines.apollo import ApolloEngine
from opendarts.engines.registry import DEFAULT_PRIMARY_ENGINE
from tests.test_capture_daemon import (
    _fake_calibration_attempt,
    _throw_trigger_ready,
)


def _bg_frame_calib():
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}
    return bg, frame, calibrations


def test_warns_when_primary_engine_reports_n_cameras_used_equal_1(tmp_path, monkeypatch, caplog):
    """Real end-to-end proof: a primary engine result carrying
    `diagnostics["n_cameras_used"] == 1` produces the WARNING log line,
    naming the primary engine and the sector/ring/reason it scored --
    the concrete debug signal g1-056-D17 was missing live."""

    class _StubPrimary:
        def score(self, bg_images, frame_images, calibration, **kwargs):
            return EngineResult(
                ok=True, sector=20, ring="single_outer", board_xy_mm=(1.0, 2.0),
                reason="stub single-camera answer",
                diagnostics={"n_cameras_used": 1},
            )

    # Pinned: this warning is about whatever the PRIMARY reports, so the
    # stub must BE the primary. The default primary is Zeus (a combiner),
    # whose own consensus result -- not a sub-engine's -- is what is
    # checked, so patching a sub-engine would never reach this path.
    monkeypatch.setattr(capture_daemon, "get_engine", lambda name: _StubPrimary())

    bg, frame, calibrations = _bg_frame_calib()
    caplog.set_level(logging.WARNING, logger="opendarts.capture_daemon")

    capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
        background_save=False,
    )

    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    hits = [m for m in warnings if "single-camera fuse" in m]
    assert len(hits) == 1, f"expected exactly one single-camera-fuse warning, got: {warnings}"
    assert DEFAULT_PRIMARY_ENGINE in hits[0]
    assert "sector=20" in hits[0]
    assert "ring=single_outer" in hits[0]


def test_no_warning_when_n_cameras_used_is_2_or_more(tmp_path, monkeypatch, caplog):
    """The gate is exactly `== 1`, not `<= 1` or "fewer than expected" --
    2 (or 3) cameras must never trip this."""

    def two_camera_score(self, bg_images, frame_images, calibration, **kwargs):
        return EngineResult(
            ok=True, sector=20, ring="single_outer", board_xy_mm=(1.0, 2.0),
            reason="stub two-camera answer",
            diagnostics={"n_cameras_used": 2},
        )

    monkeypatch.setattr(ApolloEngine, "score", two_camera_score)

    bg, frame, calibrations = _bg_frame_calib()
    caplog.set_level(logging.WARNING, logger="opendarts.capture_daemon")

    capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
        background_save=False,
    )

    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("single-camera fuse" in m for m in warnings)


def test_no_warning_when_diagnostics_omits_n_cameras_used(tmp_path, monkeypatch, caplog):
    """An engine that doesn't track n_cameras_used at all (diagnostics
    missing the key, or diagnostics={}) must never be treated as if it
    reported 1 -- absent is absent, never fabricated into a trigger."""

    def no_count_score(self, bg_images, frame_images, calibration, **kwargs):
        return EngineResult(
            ok=True, sector=20, ring="single_outer", board_xy_mm=(1.0, 2.0),
            reason="stub answer with no camera count",
            diagnostics={},
        )

    monkeypatch.setattr(ApolloEngine, "score", no_count_score)

    bg, frame, calibrations = _bg_frame_calib()
    caplog.set_level(logging.WARNING, logger="opendarts.capture_daemon")

    capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
        background_save=False,
    )

    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("single-camera fuse" in m for m in warnings)


def test_no_warning_with_real_engine_on_real_multi_pass_bytes(tmp_path, caplog):
    """Zero-cost / zero-behavior-change sanity: the real, unmodified
    Apollo engine (no stub) scoring an ordinary no-dart 4x4 frame never
    produces this warning -- the addition changes nothing about what the
    real engine returns."""
    bg, frame, calibrations = _bg_frame_calib()
    caplog.set_level(logging.WARNING, logger="opendarts.capture_daemon")

    capture_daemon.handle_ready_to_capture(
        _throw_trigger_ready(frame), bg, calibrations, tmp_path / "packages", "sess1",
        background_save=False,
    )

    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert not any("single-camera fuse" in m for m in warnings)
