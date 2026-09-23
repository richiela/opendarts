"""Part 2 of the Zeus-latency follow-up task, 2026-09-06/07: "background
the PRIMARY engine's score call, not just the save." Real, measured
problem this closes: before this task, `run_capture_loop_body()` called
`handle_ready_to_capture()` (primary-engine `score()` + session/throw-
number naming + `THROW_DETECTED` emit) SYNCHRONOUSLY -- only the package
SAVE (2026-09-01) was backgrounded. While that synchronous portion ran,
the loop could not fetch a fresh frame or refresh `motion_bg_frames`,
working against the same-day two-buffer-split absorption window
(`POST_CAPTURE_REFRACTORY_WINDOW_S`).

Mirrors `tests/test_engines_capture_daemon_integration.py::
test_run_capture_loop_body_iteration_timing_unaffected_by_slow_also_
run_engine`'s own already-established pattern exactly -- that test
proves an also-run engine never blocks the loop (already true before
this task); THIS test proves the genuinely NEW claim: the PRIMARY
engine's own `score()` call no longer blocks the loop either, once
`_dispatch_handle_ready_to_capture_in_background()` (see that function's
own docstring in opendarts/live/capture_daemon.py) is wired in."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import opendarts.live.capture_daemon as capture_daemon
from tests.lifecycle_scripting import script_trigger
from opendarts.engines.base import EngineResult
from tests.test_capture_daemon import (
    STARTUP_FETCHES,
    ThrowState,
    ThrowTriggerState,
    _fake_calibration_attempt,
    _StopLoop,
)


def test_run_capture_loop_body_iteration_timing_unaffected_by_slow_primary_engine(
    tmp_path, monkeypatch
):
    """The real proof this task exists for: with a REAL (not
    monkeypatched) handle_ready_to_capture() wired to a deliberately slow
    PRIMARY engine (the configured PRIMARY engine's own score()
    sleeps 3s -- resolved from the registry, not hardcoded), running one READY_TO_CAPTURE
    through the REAL run_capture_loop_body() must complete in well under
    3s -- the trigger/settle state machine's own iteration/state
    progression is not waiting on the primary engine's own scoring,
    regardless of how slow it is. Before Part 2, this was false: the
    primary engine ran synchronously on the loop's own thread."""
    HANG_SECONDS = 3.0
    class _SlowPrimary:
        """A stand-in PRIMARY engine that does nothing but take a long
        time. Deliberately not a real engine: this test is about the
        dispatch seam, not about scoring, and a real engine would add its
        own (variable, much larger) cost on top of the sleep."""

        def score(self, bg_images, frame_images, calibration, **kwargs):
            time.sleep(HANG_SECONDS)
            return EngineResult(
                ok=False, sector=None, ring=None, board_xy_mm=None,
                reason="slow stub", diagnostics={},
            )

    monkeypatch.setattr(capture_daemon, "get_engine", lambda name: _SlowPrimary())

    true_baseline = {0: np.full((2, 2, 3), 111, dtype=np.uint8)}
    dart_frame = {0: np.full((2, 2, 3), 20, dtype=np.uint8)}

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {0: _fake_calibration_attempt().calibration}

    fetch_sequence = [true_baseline] * STARTUP_FETCHES + [dart_frame]
    fetch_calls = {"n": 0}

    def fake_fetch(dest_dir, *, hub=None):
        idx = fetch_calls["n"]
        fetch_calls["n"] += 1
        return fetch_sequence[idx] if idx < len(fetch_sequence) else dart_frame

    scripted_states = [
        ThrowTriggerState(
            state=ThrowState.READY_TO_CAPTURE, dart_count=1,
            true_baseline_frames=true_baseline, last_frame=dart_frame,
        ),
    ]
    advance_calls = {"n": 0}

    def fake_advance(trigger, bg_frames, current_frames):
        idx = advance_calls["n"]
        advance_calls["n"] += 1
        if idx < len(scripted_states):
            return scripted_states[idx]
        raise _StopLoop("stop after the one scripted READY_TO_CAPTURE")

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)
    # handle_ready_to_capture is DELIBERATELY left real (not monkeypatched)
    # -- this is the whole point of this test, matching the also-run
    # engine test's own established convention.

    stop_event = threading.Event()
    started = time.monotonic()
    with pytest.raises(_StopLoop):
        capture_daemon.run_capture_loop_body(
            hub=None,
            package_root=tmp_path / "packages",
            poll_interval_s=0.0,
            stop_event=stop_event,
            scratch_dir=tmp_path / "scratch",
            engine_config_store=capture_daemon.EngineConfigStore(also_run=()),
        )
    elapsed = time.monotonic() - started

    assert elapsed < 1.5, (
        f"run_capture_loop_body() took {elapsed:.2f}s to process one READY_TO_CAPTURE "
        f"and raise -- a {HANG_SECONDS}s PRIMARY engine must NOT be on the loop's own "
        f"critical path once Part 2's background dispatch is wired in; this is the "
        f"exact regression this test exists to catch"
    )

    # Real, bounded wait for the background thread's own score() call to
    # actually finish -- proves the slow score DID genuinely run (not
    # skipped/short-circuited), just off the loop's own critical path.
    deadline = time.monotonic() + HANG_SECONDS + 2.0
    packages_dir = tmp_path / "packages"
    while time.monotonic() < deadline:
        if packages_dir.exists() and any(packages_dir.rglob("result.json")):
            break
        time.sleep(0.05)
    result_files = list(packages_dir.rglob("result.json")) if packages_dir.exists() else []
    assert result_files, "the backgrounded primary engine call never actually completed/saved"


def test_falsifies_cleanly_without_the_background_dispatch(tmp_path, monkeypatch):
    """Direct falsification: force the loop back to calling
    handle_ready_to_capture() SYNCHRONOUSLY even when background_save is
    True (bypassing _dispatch_handle_ready_to_capture_in_background()
    entirely, simulating the pre-Part-2 code path) and confirm the exact
    same scenario above now DOES block for the full HANG_SECONDS -- this
    is what proves the test above is measuring something real, not
    passing by construction regardless of the fix."""
    HANG_SECONDS = 1.0  # smaller than above, to keep this test fast
    class _SlowPrimary:
        """A stand-in PRIMARY engine that does nothing but take a long
        time. Deliberately not a real engine: this test is about the
        dispatch seam, not about scoring, and a real engine would add its
        own (variable, much larger) cost on top of the sleep."""

        def score(self, bg_images, frame_images, calibration, **kwargs):
            time.sleep(HANG_SECONDS)
            return EngineResult(
                ok=False, sector=None, ring=None, board_xy_mm=None,
                reason="slow stub", diagnostics={},
            )

    monkeypatch.setattr(capture_daemon, "get_engine", lambda name: _SlowPrimary())

    # Force the SAME synchronous call path this loop used before Part 2,
    # regardless of background_save's value -- a real function, not a
    # bypass of handle_ready_to_capture() itself, so the primary engine's
    # own score() call still genuinely blocks whichever thread calls it.
    real_handle_ready_to_capture = capture_daemon.handle_ready_to_capture

    def sync_dispatch(*args, **kwargs):
        real_handle_ready_to_capture(*args, **kwargs)

    monkeypatch.setattr(
        capture_daemon, "_dispatch_handle_ready_to_capture_in_background", sync_dispatch
    )
    # The loop dispatches via a real threading.Thread whose target is
    # _dispatch_handle_ready_to_capture_in_background -- forcing THAT
    # function itself to run synchronously isn't enough on its own,
    # since it would still run on the spawned thread, not the loop's
    # own. Patch threading.Thread.start() used by this specific call
    # site to run the target inline instead of on a real thread.
    import opendarts.live.capture_daemon as cd

    class _InlineThread:
        def __init__(self, target=None, args=(), kwargs=None, name=None, daemon=None):
            self._target = target
            self._args = args
            self._kwargs = kwargs or {}

        def start(self):
            self._target(*self._args, **self._kwargs)

    monkeypatch.setattr(cd.threading, "Thread", _InlineThread)

    true_baseline = {0: np.full((2, 2, 3), 111, dtype=np.uint8)}
    dart_frame = {0: np.full((2, 2, 3), 20, dtype=np.uint8)}

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {0: _fake_calibration_attempt().calibration}

    fetch_sequence = [true_baseline] * STARTUP_FETCHES + [dart_frame]
    fetch_calls = {"n": 0}

    def fake_fetch(dest_dir, *, hub=None):
        idx = fetch_calls["n"]
        fetch_calls["n"] += 1
        return fetch_sequence[idx] if idx < len(fetch_sequence) else dart_frame

    scripted_states = [
        ThrowTriggerState(
            state=ThrowState.READY_TO_CAPTURE, dart_count=1,
            true_baseline_frames=true_baseline, last_frame=dart_frame,
        ),
    ]
    advance_calls = {"n": 0}

    def fake_advance(trigger, bg_frames, current_frames):
        idx = advance_calls["n"]
        advance_calls["n"] += 1
        if idx < len(scripted_states):
            return scripted_states[idx]
        raise _StopLoop("stop after the one scripted READY_TO_CAPTURE")

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    monkeypatch.setattr(capture_daemon, "fetch_current_frames", fake_fetch)
    script_trigger(monkeypatch, fake_advance)

    started = time.monotonic()
    with pytest.raises(_StopLoop):
        capture_daemon.run_capture_loop_body(
            hub=None,
            package_root=tmp_path / "packages",
            poll_interval_s=0.0,
            stop_event=threading.Event(),
            scratch_dir=tmp_path / "scratch",
            engine_config_store=capture_daemon.EngineConfigStore(also_run=()),
        )
    elapsed = time.monotonic() - started

    assert elapsed >= HANG_SECONDS, (
        f"expected the forced-synchronous path to block for at least {HANG_SECONDS}s "
        f"(reproducing the pre-Part-2 behavior), but it returned in {elapsed:.2f}s -- "
        f"this falsification is supposed to fail without the real background dispatch"
    )
