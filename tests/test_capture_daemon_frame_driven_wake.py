"""Tests for opendarts.live.capture_daemon's frame-driven wake mechanism
(`_wait_for_next_frame()`/`FrameWakeState`) -- 2026-09-05, -
approved frame-driven-capture task ("I want to go to frame driven
first... that way we process all frames and dont just poll and miss
frames"). See that function's own docstring, and
opendarts.live.local_capture.LocalCameraHub's module docstring ("FRAME-
DRIVEN WAKE PRIMITIVE" section), for the full design.

Covers, directly and in isolation from the rest of run_capture_loop_
body(): the no-hub degrade-safely path, drop-to-latest
accounting (single drop and sustained-overrun escalation), stop_event
promptness (both "already set" and "set mid-wait" cases, with real
timing bounds), and an end-to-end run of the real loop against a real
LocalCameraHub frame-generation path.
"""
from __future__ import annotations

import logging
import threading
import time

import pytest

import opendarts.live.capture_daemon as capture_daemon
from tests.lifecycle_scripting import script_trigger
from opendarts.live.capture_daemon import (
    SUSTAINED_FRAME_DROP_WARNING_ITERATIONS,
    FrameWakeState,
    _wait_for_next_frame,
)
from opendarts.live.local_capture import CameraConfig, LocalCameraHub

from tests.test_capture_daemon import _fake_calibration_attempt, _StopLoopEarly
from tests.test_local_capture import make_fake_capture_factory


class _FakeGenerationHub:
    """A minimal, non-threaded stand-in for LocalCameraHub's own
    frame-generation contract -- lets these tests control exactly what
    generation `wait_for_new_frame()` returns without needing a real
    pump thread for every scenario. `advance_by` controls how many
    generations a single call jumps (simulating drop-to-latest when > 1),
    `never_advance=True` simulates a genuinely stalled/dead pump."""

    def __init__(self, start_generation=0, advance_by=1, never_advance=False):
        self._generation = start_generation
        self.advance_by = advance_by
        self.never_advance = never_advance
        self.calls: list[tuple[int, float]] = []

    def wait_for_new_frame(self, last_generation: int, timeout: float) -> int:
        self.calls.append((last_generation, timeout))
        if self.never_advance:
            time.sleep(min(timeout, 0.05)) # honor a real, bounded wait
            return self._generation
        self._generation += self.advance_by
        return self._generation


def _mk_stop_event(already_set: bool = False) -> threading.Event:
    ev = threading.Event()
    if already_set:
        ev.set()
    return ev


# ---------------------------------------------------------------------
# Fallback path (no hub / hub without the primitive)
# ---------------------------------------------------------------------




def test_fallback_to_old_wait_when_hub_is_none(monkeypatch):
    calls = []
    monkeypatch.setattr(
        capture_daemon,
        "_wait_for_next_iteration",
        lambda *a: calls.append(a),
    )
    state = FrameWakeState(last_frame_generation=0)
    result = _wait_for_next_frame(
        None, state, _mk_stop_event(), iteration_started=0.0, poll_interval_s=0.05, iteration=1,
    )
    assert len(calls) == 1
    assert result is state


def test_fallback_to_old_wait_when_hub_lacks_the_primitive(monkeypatch):
    """A bare test-double hub built before this task (no wait_for_new_
    frame attribute at all) must degrade safely, matching this module's
    own existing frame-freshness-gate hasattr() convention."""
    calls = []
    monkeypatch.setattr(
        capture_daemon,
        "_wait_for_next_iteration",
        lambda *a: calls.append(a),
    )

    class _BareHub:
        pass

    state = FrameWakeState(last_frame_generation=0)
    result = _wait_for_next_frame(
        _BareHub(), state, _mk_stop_event(), iteration_started=0.0,
        poll_interval_s=0.05, iteration=1,
    )
    assert len(calls) == 1
    assert result is state


# ---------------------------------------------------------------------
# Real frame-generation-driven wake, no drop
# ---------------------------------------------------------------------


def test_wakes_on_the_new_generation_with_no_drop(monkeypatch):
    hub = _FakeGenerationHub(start_generation=10, advance_by=1)
    state = FrameWakeState(last_frame_generation=10, consecutive_drop_iterations=3, total_frames_dropped=7)

    result = _wait_for_next_frame(
        hub, state, _mk_stop_event(), iteration_started=0.0, poll_interval_s=0.05, iteration=1,
    )

    assert result.last_frame_generation == 11
    # A clean (no-drop) wake resets the consecutive counter but keeps
    # the lifetime total untouched -- it's cumulative, never reset.
    assert result.consecutive_drop_iterations == 0
    assert result.total_frames_dropped == 7


# ---------------------------------------------------------------------
# Drop-to-latest accounting
# ---------------------------------------------------------------------


def test_single_dropped_generation_is_counted_and_logged(monkeypatch, caplog):
    # DEBUG, not WARNING, as of 2026-09-12. A single dropped cycle is not a
    # fault: it happens whenever an iteration's work outlasts a frame
    # period, which scoring a dart (94-172ms) and calibrating both do by
    # arithmetic. It is still reported for anyone digging -- it just no
    # longer interrupts. The SUSTAINED escalation is the real signal and is
    # still asserted as a WARNING below.
    """advance_by=3 simulates the pump completing 3 cycles while this
    loop's own previous iteration was busy -- exactly 2 should be
    reported dropped (3 - 1, the one the caller DOES consume)."""
    hub = _FakeGenerationHub(start_generation=0, advance_by=3)
    state = FrameWakeState(last_frame_generation=0)

    with caplog.at_level(logging.DEBUG, logger="opendarts.capture_daemon"):
        result = _wait_for_next_frame(
            hub, state, _mk_stop_event(), iteration_started=0.0,
            poll_interval_s=0.05, iteration=42,
        )

    assert result.last_frame_generation == 3
    assert result.total_frames_dropped == 2
    assert result.consecutive_drop_iterations == 1
    drop_lines = [r.message for r in caplog.records if "dropped" in r.message]
    assert drop_lines, "the drop should still be reported, at debug"
    assert all(r.levelno < logging.WARNING for r in caplog.records
               if "dropped" in r.message and "SUSTAINED" not in r.message), (
        "a single dropped cycle must not be a warning -- it is normal during "
        "scoring and calibration, and warning about it trains the eye to "
        "ignore warnings"
    )
    assert "iteration 42" in drop_lines[0]
    assert "SUSTAINED" not in drop_lines[0] # a single drop is not yet sustained


def test_sustained_overrun_escalates_to_a_louder_warning(caplog):
    """SUSTAINED_FRAME_DROP_WARNING_ITERATIONS consecutive drop-carrying
    calls must escalate to a distinctly-worded warning naming the
    overrun as SUSTAINED, per this task's own explicit requirement that
    silently falling behind is the worst failure mode here."""
    hub = _FakeGenerationHub(start_generation=0, advance_by=2) # always drops 1
    state = FrameWakeState(last_frame_generation=0)

    with caplog.at_level(logging.WARNING, logger="opendarts.capture_daemon"):
        for i in range(1, SUSTAINED_FRAME_DROP_WARNING_ITERATIONS + 1):
            state = _wait_for_next_frame(
                hub, state, _mk_stop_event(), iteration_started=0.0,
                poll_interval_s=0.05, iteration=i,
            )

    assert state.consecutive_drop_iterations == SUSTAINED_FRAME_DROP_WARNING_ITERATIONS
    sustained_lines = [r.message for r in caplog.records if "SUSTAINED" in r.message]
    assert len(sustained_lines) == 1, (
        f"expected exactly one SUSTAINED warning at iteration "
        f"{SUSTAINED_FRAME_DROP_WARNING_ITERATIONS}, got: {sustained_lines}"
    )
    assert f"iteration {SUSTAINED_FRAME_DROP_WARNING_ITERATIONS}" in sustained_lines[0]
    # At WARNING level, the escalation should be the ONLY thing visible.
    # The per-iteration drop line is DEBUG as of 2026-09-12: a single
    # dropped cycle is normal during scoring and calibration, and warning
    # about it trained the eye to ignore warnings. The escalation is the
    # real signal -- consecutive overruns mean the loop is persistently
    # slower than the pump -- so it must still cut through.
    ordinary_drop_lines = [
        r.message for r in caplog.records
        if "dropped" in r.message and "SUSTAINED" not in r.message
    ]
    assert not ordinary_drop_lines, (
        "a single dropped cycle must not surface at WARNING; only the "
        f"sustained escalation should. Got: {ordinary_drop_lines}"
    )


def test_consecutive_drop_counter_resets_after_a_clean_iteration():
    """A single clean (no-drop) iteration in between must reset the
    consecutive-overrun counter to 0 -- "sustained" means CONSECUTIVE,
    not merely a running lifetime tally."""
    dropping_hub = _FakeGenerationHub(start_generation=0, advance_by=2)
    state = FrameWakeState(last_frame_generation=0)

    for i in range(1, 4):
        state = _wait_for_next_frame(
            dropping_hub, state, _mk_stop_event(), iteration_started=0.0,
            poll_interval_s=0.05, iteration=i,
        )
    assert state.consecutive_drop_iterations == 3

    # One clean call (advance_by=1, no drop) using a fresh hub anchored
    # at the same generation the dropping hub last reached.
    clean_hub = _FakeGenerationHub(start_generation=state.last_frame_generation, advance_by=1)
    state = _wait_for_next_frame(
        clean_hub, state, _mk_stop_event(), iteration_started=0.0,
        poll_interval_s=0.05, iteration=4,
    )
    assert state.consecutive_drop_iterations == 0
    # advance_by=2 drops exactly 1 generation per call (2 - 1) -- 3 calls
    # -> 3 total dropped, unaffected by the consecutive-counter reset
    # (lifetime total is cumulative, never reset).
    assert state.total_frames_dropped == 3


# ---------------------------------------------------------------------
# stop_event promptness -- real, timed proofs
# ---------------------------------------------------------------------


def test_returns_immediately_when_stop_event_already_set():
    hub = _FakeGenerationHub(never_advance=True)
    state = FrameWakeState(last_frame_generation=0)

    t0 = time.monotonic()
    result = _wait_for_next_frame(
        hub, state, _mk_stop_event(already_set=True), iteration_started=0.0,
        poll_interval_s=0.05, iteration=1,
    )
    elapsed = time.monotonic() - t0

    assert result is state # nothing new observed
    assert elapsed < 0.02, f"took {elapsed:.4f}s to notice an already-set stop_event"
    assert hub.calls == [] # never even attempted a wait


def test_stop_event_set_mid_wait_interrupts_within_a_bounded_real_time():
    """The real, disclosed tradeoff this design makes: stop_event set
    CONCURRENTLY, while a real never-advancing hub is being waited on,
    must still interrupt within a small, bounded real time -- not
    instantly (a Condition-based wait cannot observe an unrelated Event
    directly, see _wait_for_next_frame()'s own docstring), but nowhere
    close to `poll_interval_s`'s own old ~50ms-per-slice ceiling, let
    alone unboundedly."""
    hub = _FakeGenerationHub(never_advance=True)
    state = FrameWakeState(last_frame_generation=0)
    stop_event = threading.Event()

    def flip_stop_soon():
        time.sleep(0.03)
        stop_event.set()

    t = threading.Thread(target=flip_stop_soon, daemon=True)
    t0 = time.monotonic()
    t.start()
    result = _wait_for_next_frame(
        hub, state, stop_event, iteration_started=0.0, poll_interval_s=0.5, iteration=1,
    )
    elapsed = time.monotonic() - t0
    t.join(timeout=1.0)

    assert result is state
    # Real bound: the flip happens at ~30ms, plus at most one
    # _FRAME_WAIT_STOP_EVENT_RECHECK_S slice (~10ms) before it's
    # noticed -- generous margin against real scheduling jitter.
    assert elapsed < 0.3, (
        f"took {elapsed:.3f}s to notice stop_event set mid-wait -- expected well under "
        f"the old poll_interval_s=0.5s ceiling this replaces"
    )


def test_never_blocks_indefinitely_even_if_stop_event_is_never_set():
    """A genuinely stalled hub (never advances) with a stop_event that
    is NEVER set must still return -- proving there is no path to an
    indefinite block, only a bounded one. Runs the wait loop for a
    short, deliberately bounded real duration by using a hub that starts
    advancing after a few calls (simulating "the camera eventually
    recovers"), so this test itself completes quickly rather than
    genuinely blocking forever if the fix were broken."""

    class _RecoveringHub:
        def __init__(self):
            self.n_calls = 0

        def wait_for_new_frame(self, last_generation, timeout):
            self.n_calls += 1
            if self.n_calls < 3:
                time.sleep(min(timeout, 0.02))
                return last_generation # still stalled
            return last_generation + 1 # "recovers"

    hub = _RecoveringHub()
    state = FrameWakeState(last_frame_generation=0)
    stop_event = threading.Event() # never set

    t0 = time.monotonic()
    result = _wait_for_next_frame(
        hub, state, stop_event, iteration_started=0.0, poll_interval_s=0.02, iteration=1,
    )
    elapsed = time.monotonic() - t0

    assert result.last_frame_generation == 1
    assert hub.n_calls == 3
    assert elapsed < 1.0 # genuinely bounded, not stuck


# ---------------------------------------------------------------------
# Real, TIMED end-to-end proof of Requirement 1's own math
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Real end-to-end proof: run_capture_loop_body() actually USES the new
# frame-driven path with a genuine LocalCameraHub, not just the isolated
# _wait_for_next_frame() unit above.
# ---------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stop_pump_threads_after_every_test():
    """Same autouse cleanup tests/test_local_capture.py's own fixture of
    the same name provides -- only this file's own real-hub test below
    actually needs it, but applying it to the whole module is harmless."""
    created: list[LocalCameraHub] = []
    original_init = LocalCameraHub.__init__

    def _tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created.append(self)

    LocalCameraHub.__init__ = _tracking_init # type: ignore[method-assign]
    try:
        yield
    finally:
        LocalCameraHub.__init__ = original_init # type: ignore[method-assign]
        for hub in created:
            hub.close_all()


def test_run_capture_loop_body_end_to_end_uses_the_real_frame_generation_path(
    tmp_path, monkeypatch
):
    """Every OTHER test in this project's existing suite that drives
    run_capture_loop_body() end to end uses a bare test-double hub with
    no `wait_for_new_frame` -- which means they all exercise ONLY the
    fallback path (byte-identical old behavior), never the new
    frame-driven one. This test uses a REAL LocalCameraHub (mocked
    cv2.VideoCapture, real pump thread) so the loop's own 3 call sites
    genuinely reach `hub.wait_for_new_frame()` -- proven by a real call
    counter, not inferred from the absence of a crash."""
    import cv2

    monkeypatch.setattr(cv2, "VideoCapture", make_fake_capture_factory(frame_ok=True))
    hub = LocalCameraHub(configs=[CameraConfig(device=0)])
    ok_flags = hub.open_all()
    assert all(ok_flags)

    real_wait_for_new_frame = hub.wait_for_new_frame
    wait_calls = {"n": 0}

    def counting_wait_for_new_frame(last_generation, timeout):
        wait_calls["n"] += 1
        return real_wait_for_new_frame(last_generation, timeout)

    monkeypatch.setattr(hub, "wait_for_new_frame", counting_wait_for_new_frame)

    def fake_bootstrap(snapshot_dir, *, hub=None, **_kwargs):
        return {0: _fake_calibration_attempt().calibration}

    calls = {"n": 0}

    def counting_advance(trigger, bg_frames, current_frames):
        calls["n"] += 1
        if calls["n"] > 5:
            raise _StopLoopEarly("ran enough real iterations for this test")
        return trigger

    monkeypatch.setattr(capture_daemon, "bootstrap_calibrations", fake_bootstrap)
    script_trigger(monkeypatch, counting_advance)
    monkeypatch.setattr(
        capture_daemon, "handle_ready_to_capture", lambda *a, **k: tmp_path / "phantom"
    )

    stop_event = threading.Event()
    with pytest.raises(_StopLoopEarly):
        capture_daemon.run_capture_loop_body(
            hub=hub,
            package_root=tmp_path / "packages",
            poll_interval_s=0.05,
            stop_event=stop_event,
            scratch_dir=tmp_path / "scratch",
        )

    assert calls["n"] > 5 # the real advance() ran multiple real iterations
    assert wait_calls["n"] > 0, (
        "run_capture_loop_body() never called hub.wait_for_new_frame() -- "
        "the frame-driven path was not actually reached end to end"
    )
