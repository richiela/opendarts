"""PER-ITERATION COST BREAKDOWN -- the diagnostic log line
opendarts.live.capture_daemon.run_capture_loop_body() emits per iteration
while opendarts.live.diagnostics_gate is on:

    iteration diagnostic: it= state= fetch= gate= lifecycle= post= body= sleep=

`lifecycle=` brackets the whole decision step (observe + adapter);
`post=` is present only on the iteration that transitions state; IDLE
iterations are sampled every _ITERATION_DIAG_SAMPLE_EVERY_N_ITERATIONS.
The trigger is scripted through the loop's decision seam so each
scenario is deterministic.
"""
from __future__ import annotations

import logging
import threading

import numpy as np
import pytest

import opendarts.live.capture_daemon as capture_daemon
from opendarts.capture.trigger_state import ThrowState, ThrowTriggerState
from opendarts.live import diagnostics_gate
from tests.lifecycle_scripting import script_trigger
from tests.test_capture_daemon import _fake_calibration_attempt, _StopLoopEarly


class _FakeHubWithFrameCounts:
    class _Status:
        def __init__(self):
            self.frame_count = 0
            self.last_read_at_monotonic = None

    def __init__(self, cams):
        self.status = {cam: self._Status() for cam in cams}


_FIELDS = ("it=", "state=", "fetch=", "gate=", "lifecycle=", "post=", "body=", "sleep=")


def _frame(fill: int) -> np.ndarray:
    return np.full((8, 8, 3), fill, dtype=np.uint8)


def _run_scripted(monkeypatch, tmp_path, hub, states, poll_interval_s=0.0):
    """Real loop, scripted trigger: one state per iteration, then stop."""

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {cam: _fake_calibration_attempt().calibration for cam in hub.status}

    def fake_fetch(dest_dir, *, hub=None):
        for status in hub.status.values():
            status.frame_count += 1
        return {cam: _frame(80) for cam in hub.status}

    calls = {"n": 0}

    def scripted(trigger, bg_frames, current_frames):
        idx = calls["n"]
        calls["n"] += 1
        if idx >= len(states):
            raise _StopLoopEarly("scripted sequence exhausted")
        return states[idx]

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, scripted)
    monkeypatch.setattr(
        capture_daemon, "handle_ready_to_capture", lambda *a, **k: tmp_path / "phantom"
    )

    with pytest.raises(_StopLoopEarly):
        capture_daemon.run_capture_loop_body(
            hub=hub,
            package_root=tmp_path / "packages",
            poll_interval_s=poll_interval_s,
            stop_event=threading.Event(),
            scratch_dir=tmp_path / "scratch",
        )


def _idle(n):
    return [ThrowTriggerState(state=ThrowState.IDLE) for _ in range(n)]


def _settle_episode():
    """idle, idle, hand, settling x3, commit, idle, idle"""
    return (
        _idle(2)
        + [ThrowTriggerState(state=ThrowState.MOTION_DETECTED)]
        + [ThrowTriggerState(state=ThrowState.SETTLING) for _ in range(3)]
        + [ThrowTriggerState(state=ThrowState.READY_TO_CAPTURE, dart_count=1, last_frame={0: _frame(20)})]
        + [ThrowTriggerState(state=ThrowState.IDLE, dart_count=1) for _ in range(2)]
    )


def _diag_lines(caplog):
    return [r.message for r in caplog.records if r.message.startswith("iteration diagnostic:")]


def test_iteration_diagnostic_gated_off_by_default_no_lines_at_all(tmp_path, monkeypatch, caplog):
    assert diagnostics_gate.enabled() is False
    hub = _FakeHubWithFrameCounts([0, 1])
    with caplog.at_level(logging.DEBUG, logger="opendarts.capture_daemon"):
        _run_scripted(monkeypatch, tmp_path, hub, _settle_episode())
    assert _diag_lines(caplog) == []


def test_iteration_diagnostic_emits_on_every_non_idle_iteration_with_the_full_field_set(
    tmp_path, monkeypatch, caplog
):
    diagnostics_gate.set_enabled(True)
    hub = _FakeHubWithFrameCounts([0, 1])
    with caplog.at_level(logging.DEBUG, logger="opendarts.capture_daemon"):
        _run_scripted(monkeypatch, tmp_path, hub, _settle_episode())

    lines = _diag_lines(caplog)
    assert lines
    settle_lines = [m for m in lines if "state=MOTION_DETECTED" in m or "state=SETTLING" in m]
    # state= is the state AT ENTRY: the hand iteration enters IDLE, the
    # three settling iterations enter MOTION_DETECTED/SETTLING and the
    # commit iteration enters SETTLING -- four non-IDLE entries in all.
    assert len(settle_lines) == 4, settle_lines
    for line in lines:
        for field in _FIELDS:
            assert field in line, f"missing {field!r} in {line}"
        assert "lifecycle=n/a" not in line, line
        # legacy sub-timings are gone for good
        for gone in ("advance=", "motion=", "settled=", "ambig=", "settle_window="):
            assert gone not in line, line


def test_iteration_diagnostic_idle_sampling_every_n_iterations(tmp_path, monkeypatch, caplog):
    diagnostics_gate.set_enabled(True)
    monkeypatch.setattr(capture_daemon, "_ITERATION_DIAG_SAMPLE_EVERY_N_ITERATIONS", 3)
    hub = _FakeHubWithFrameCounts([0])
    with caplog.at_level(logging.DEBUG, logger="opendarts.capture_daemon"):
        _run_scripted(monkeypatch, tmp_path, hub, _idle(9))

    idle_lines = [m for m in _diag_lines(caplog) if "state=IDLE" in m]
    # iterations 3, 6, 9 (1-indexed, `iteration % 3 == 0`)
    assert len(idle_lines) == 3, idle_lines
    for line in idle_lines:
        assert "post=n/a" in line


def test_iteration_diagnostic_post_present_only_on_transition_iterations(tmp_path, monkeypatch, caplog):
    diagnostics_gate.set_enabled(True)
    hub = _FakeHubWithFrameCounts([0, 1])
    with caplog.at_level(logging.DEBUG, logger="opendarts.capture_daemon"):
        _run_scripted(monkeypatch, tmp_path, hub, _settle_episode())

    lines = _diag_lines(caplog)
    post_present = [m for m in lines if "post=n/a" not in m]
    assert post_present
    transition_iterations = set()
    for r in caplog.records:
        if r.message.startswith("trigger state:") and "->" in r.message and "still" not in r.message:
            it = int(r.message.split("(iteration ")[1].split(")")[0].split(",")[0])
            transition_iterations.add(it)
    for line in post_present:
        it = int(line.split("it=")[1].split(" ")[0])
        assert it in transition_iterations, line
    for line in lines:
        it = int(line.split("it=")[1].split(" ")[0])
        if it not in transition_iterations:
            assert "post=n/a" in line, line


def test_iteration_diagnostic_self_consistency_check_does_not_crash_or_false_positive(
    tmp_path, monkeypatch, caplog
):
    diagnostics_gate.set_enabled(True)
    hub = _FakeHubWithFrameCounts([0])
    with caplog.at_level(logging.DEBUG, logger="opendarts.capture_daemon"):
        _run_scripted(monkeypatch, tmp_path, hub, _idle(8))
    recon_lines = [r.message for r in caplog.records if "self-consistency check" in r.message]
    assert recon_lines == []
