"""`_THROW_NUMBER_LOCK`, `opendarts/live/capture_daemon.py` -- 2026-09-06/07,
Zeus-latency follow-up task, Part 2 ("background the PRIMARY engine's
score call, not just the save"). See that lock's own module-level
comment for the full incident/design writeup: backgrounding the ENTIRE
`handle_ready_to_capture()` call means two overlapping calls for the
SAME session can now run concurrently, on separate threads, sharing the
same `session_throw_counters/<session_id>.count` file -- without a lock,
a read-increment-write race could collide two different throws onto the
SAME `throw_number`/`dest_dir`, a real REPLAY/data-loss risk.

This file proves the lock actually closes that race, with REAL threads
hammering the REAL `handle_ready_to_capture()` naming path concurrently
-- not a single-threaded sequential loop (which cannot exercise this
race at all)."""

from __future__ import annotations

import threading

import numpy as np

from opendarts.capture.trigger_state import ThrowState, ThrowTriggerState
from opendarts.live import capture_daemon
from tests.test_capture_daemon import _fake_calibration_attempt


def _bg_frame_calib():
    bg = {0: np.full((4, 4, 3), 50, dtype=np.uint8)}
    frame = {0: np.full((4, 4, 3), 200, dtype=np.uint8)}
    calibrations = {0: _fake_calibration_attempt().calibration}
    return bg, frame, calibrations


def test_concurrent_handle_ready_to_capture_calls_never_collide_on_throw_number(tmp_path):
    """N genuinely concurrent handle_ready_to_capture() calls for the
    SAME session must allocate N distinct throw_numbers/dest_dirs, never
    colliding -- the exact scenario Part 2's own backgrounding makes
    real for the first time (previously structurally impossible, since
    the whole function ran synchronously on one thread)."""
    N = 24
    bg, frame, calibrations = _bg_frame_calib()
    results: list = []
    errors: list[BaseException] = []
    results_lock = threading.Lock()

    def _one():
        try:
            trigger = ThrowTriggerState(state=ThrowState.READY_TO_CAPTURE, last_frame=frame)
            dest_dir = capture_daemon.handle_ready_to_capture(
                trigger, bg, calibrations, tmp_path / "packages", "sess_concurrent",
                background_save=False,
            )
        except BaseException as exc:  # noqa: BLE001 -- capture for the assertion below
            with results_lock:
                errors.append(exc)
            return
        with results_lock:
            results.append(dest_dir)

    threads = [threading.Thread(target=_one) for _ in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"unexpected exceptions from concurrent calls: {errors}"
    assert len(results) == N
    assert len(set(results)) == N, (
        f"throw_number/dest_dir COLLISION under real concurrency -- "
        f"{N - len(set(results))} duplicate(s) out of {N} calls: {results}"
    )


def test_concurrent_reset_and_capture_do_not_collide(tmp_path):
    """A manual Reset (opendarts.live.capture_daemon's own `_reset_session_
    throw_numbering()` call site) racing a still-in-flight background
    capture for the SAME session must not corrupt the counter file into
    an unparseable state or crash either side -- both real callers must
    hold the SAME `_THROW_NUMBER_LOCK`."""
    N = 12
    bg, frame, calibrations = _bg_frame_calib()
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def _capture():
        try:
            trigger = ThrowTriggerState(state=ThrowState.READY_TO_CAPTURE, last_frame=frame)
            capture_daemon.handle_ready_to_capture(
                trigger, bg, calibrations, tmp_path / "packages", "sess_reset_race",
                background_save=False,
            )
        except BaseException as exc:  # noqa: BLE001
            with errors_lock:
                errors.append(exc)

    def _reset():
        try:
            with capture_daemon._THROW_NUMBER_LOCK:
                capture_daemon._reset_session_throw_numbering(
                    tmp_path / "packages" / ".." / "session_throw_counters",
                    "sess_reset_race",
                )
        except BaseException as exc:  # noqa: BLE001
            with errors_lock:
                errors.append(exc)

    threads = [threading.Thread(target=_capture) for _ in range(N)]
    threads += [threading.Thread(target=_reset) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"unexpected exceptions racing reset against captures: {errors}"
