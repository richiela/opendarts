"""opendarts.live.local_capture FRAME-SET SYNC, HUB INTEGRATION.

The old drain-and-refuse design (one dedicated pump thread reading every
camera in lockstep, draining a queued camera, then refusing to judge a set
whose capture times spread too far) is gone -- see that module's own
"READING AND CHOOSING SPLIT APART" docstring section and
opendarts.live.set_assembly's module docstring for the full redesign and
the incident that drove it (2026-10-07: one camera served frames 4 behind
for a whole session; the drain/refuse patch that followed then blinded a
rig for ~6 minutes when its own two thresholds disagreed about what
"behind" meant).

The Assembler's own choosing rule -- the late/lagging-camera case, the
stalled-camera case, jitter, history depth -- is unit-tested directly,
with hand-built arrival streams and no threads at all, in
tests/test_set_assembly.py (owned by the module's author). These tests are
the layer ABOVE that: does the real CameraHub actually wire a per-camera
reader thread to the Assembler with real `cap.get(CAP_PROP_POS_MSEC)` and
the Linux-only driver-clock gate, does `_pump_once()` really publish what
the Assembler chooses, and does `close_all()` really retire every reader
thread before any capture is released.
"""
from __future__ import annotations

import sys
import threading
import time

import cv2
import numpy as np
import pytest

from opendarts.live import local_capture as lc
from opendarts.live.local_capture import CameraConfig, CameraHub

#: Nominal rate used throughout -- the same 30fps every other hub test uses.
PERIOD_MS = 1000.0 / 30


@pytest.fixture(autouse=True)
def _linux_driver_clock_and_hub_cleanup(monkeypatch):
    """Every test here wants the Linux-only driver-clock path engaged (see
    local_capture._note_capture_time's own docstring) -- patched at the
    module level `local_capture.sys` resolves to, same seam the module
    itself reads, rather than guessing at a more specific hook. Also
    tracks every hub built during the test and closes it afterward, same
    purpose as tests/test_local_capture.py's own fixture of this name
    (duplicated, not imported -- pytest fixtures are file-scoped)."""
    monkeypatch.setattr(lc.sys, "platform", "linux", raising=False)
    created: list[CameraHub] = []
    original_init = CameraHub.__init__

    def _tracking_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        created.append(self)

    CameraHub.__init__ = _tracking_init  # type: ignore[method-assign]
    try:
        yield
    finally:
        CameraHub.__init__ = original_init  # type: ignore[method-assign]
        for hub in created:
            hub.close_all()


class _DriverTimedCapture:
    """A cv2.VideoCapture stand-in that reports a real, usable
    CAP_PROP_POS_MSEC on the same clock as time.perf_counter() -- what a
    real V4L2 driver gives on Linux (set_assembly's "THE CLOCK" section).

    `lag_ms` models a steady driver-queue backlog: this camera's captures
    really happened `lag_ms` before they are read, exactly the Oct 7
    incident's shape (one camera served frames N periods behind for a
    whole session) -- not a one-off hiccup. `fails` makes every read from
    then on come back empty, modelling a camera that has stopped
    delivering at all.
    """

    def __init__(self, device, backend=None, *, lag_ms: float = 0.0,
                period_ms: float = PERIOD_MS) -> None:
        self.device = device
        self._opened = True
        self.lag_ms = lag_ms
        self.period_ms = period_ms
        self._ts: "float | None" = None
        self.fails = False
        self.reads = 0

    def isOpened(self) -> bool:  # noqa: N802
        return self._opened

    def release(self) -> None:
        self._opened = False

    def set(self, prop, value) -> bool:
        return True

    def get(self, prop) -> float:
        if prop == cv2.CAP_PROP_POS_MSEC:
            return self._ts or 0.0
        return 0.0

    def read(self):
        time.sleep(self.period_ms / 1000.0)
        self.reads += 1
        if self.fails:
            return False, None
        self._ts = time.perf_counter() * 1000.0 - self.lag_ms
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        frame[:, :, 0] = (int(self.device) + 1) % 255
        return True, frame


def _wait_until(predicate, timeout_s: float = 5.0, interval_s: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


def test_a_camera_two_periods_late_is_matched_not_refused(monkeypatch):
    """The Oct 7 / Oct 11 shape, at the hub level: cam2's driver stamps say
    it is two whole periods behind the other two. The hub must keep
    publishing matched sets that include it -- reaching back into its
    short history, per set_assembly's own rule -- never refuse a set or
    leave it permanently out."""
    caps = {
        0: _DriverTimedCapture(0),
        1: _DriverTimedCapture(1),
        2: _DriverTimedCapture(2, lag_ms=2 * PERIOD_MS),
    }
    monkeypatch.setattr(cv2, "VideoCapture", lambda d, b: caps[d])
    hub = CameraHub(configs=[CameraConfig(device=i) for i in (0, 1, 2)])
    hub.open_all()

    assert _wait_until(lambda: hub.frame_generation() > 60)

    # Sample the hub's own reported sync repeatedly, over real cycles --
    # not just once, so a single lucky set cannot hide a real refusal.
    samples = []
    for _ in range(20):
        samples.append(hub.frame_set_sync())
        time.sleep(0.02)

    judged = [s for s in samples if s["clock"] == "driver"]
    assert judged, "no set was ever judged on the driver clock"
    assert all(s["in_sync"] for s in judged), (
        "a late-but-steady camera must be matched, never refused: "
        f"{[s for s in judged if not s['in_sync']]}"
    )
    assert all(2 not in s["stalled"] for s in judged), "cam2 was left out of a set"
    # A margin over one period, not a tight one: this drives real reader
    # threads under real OS scheduling (tests/test_set_assembly.py proves
    # the Assembler's own rule tightly, with no threads and no jitter at
    # all) -- comfortably enough to catch a real refusal (which would be
    # several periods, or simply never judged) without flaking on a
    # loaded CI box.
    assert all(s["spread_ms"] < 2 * PERIOD_MS for s in judged), (
        f"set spread exceeded two periods: {[s['spread_ms'] for s in judged]}"
    )


def test_a_stalled_camera_is_left_out_while_sets_continue(monkeypatch):
    """One camera stops delivering outright. Sets must keep coming from
    the other two (never refused, never frozen waiting on the dead one),
    and the stalled camera's own frame_count -- the capture loop's real
    per-camera freshness gate -- must stop advancing."""
    caps = {
        0: _DriverTimedCapture(0),
        1: _DriverTimedCapture(1),
        2: _DriverTimedCapture(2),
    }
    monkeypatch.setattr(cv2, "VideoCapture", lambda d, b: caps[d])
    hub = CameraHub(configs=[CameraConfig(device=i) for i in (0, 1, 2)])
    hub.open_all()

    assert _wait_until(lambda: hub.status[2].frame_count >= 3)
    caps[2].fails = True
    frozen_count = hub.status[2].frame_count

    # The other two must keep making sets well past cam2's own stall
    # threshold (STALL_PERIODS of its period, floored at STALL_MIN_MS).
    assert _wait_until(lambda: hub.status[0].frame_count >= frozen_count + 20, timeout_s=5.0)
    assert _wait_until(lambda: hub.frame_set_sync()["stalled"] == [2], timeout_s=2.0)
    assert hub.status[2].frame_count == frozen_count, (
        "a stalled camera's frame_count must not advance -- that is the "
        "capture loop's own per-camera freshness gate"
    )
    assert hub.status[0].frame_count > 0 and hub.status[1].frame_count > 0


def test_close_all_joins_every_reader_thread(monkeypatch):
    """close_all() must retire every per-camera reader thread it started
    -- no `local-cam-reader-*` thread may still be alive afterward, the
    same "retire the readers before the captures they hold" guarantee the
    pump thread has always had, now extended to these."""
    caps = {i: _DriverTimedCapture(i) for i in (0, 1, 2)}
    monkeypatch.setattr(cv2, "VideoCapture", lambda d, b: caps[d])
    hub = CameraHub(configs=[CameraConfig(device=i) for i in (0, 1, 2)])
    hub.open_all()

    assert _wait_until(lambda: all(t.is_alive() for t in hub._reader_threads.values()))
    reader_names = {t.name for t in hub._reader_threads.values()}
    assert reader_names == {"local-cam-reader-0", "local-cam-reader-1", "local-cam-reader-2"}

    hub.close_all()

    assert not hub._reader_threads
    alive_readers = [
        t for t in threading.enumerate() if t.name.startswith("local-cam-reader-")
    ]
    assert not alive_readers, f"reader thread(s) still alive after close_all(): {alive_readers}"


def test_frame_set_sync_reports_no_sync_judgement_before_any_set() -> None:
    """A hub that never opened (or opened zero cameras) has nothing to
    judge yet -- in_sync defaults True (nothing to refuse), never a
    crash or a fabricated spread."""
    hub = CameraHub(configs=[CameraConfig(device=0, enabled=False)])
    report = hub.frame_set_sync()
    assert report == {
        "in_sync": True, "spread_ms": None, "clock": None,
        "stalled": [], "repeats_total": 0, "dropped": {},
    }


def test_driver_clock_is_linux_only(monkeypatch):
    """Off Linux, a usable CAP_PROP_POS_MSEC must still be ignored for the
    Assembler's clock choice -- Windows' own MF reader returns 0 here and
    macOS's AVFoundation synthesises a value not on our clock at all (see
    set_assembly's "THE CLOCK" section); only the platform check, not the
    value's shape, tells those apart from Linux's real V4L2 stamps."""
    monkeypatch.setattr(lc.sys, "platform", "darwin", raising=False)
    caps = {i: _DriverTimedCapture(i) for i in (0, 1, 2)}
    monkeypatch.setattr(cv2, "VideoCapture", lambda d, b: caps[d])
    hub = CameraHub(configs=[CameraConfig(device=i) for i in (0, 1, 2)])
    hub.open_all()

    assert _wait_until(lambda: hub.frame_generation() > 30)
    report = hub.frame_set_sync()
    assert report["clock"] == "arrival", (
        "a non-Linux platform must never feed driver capture times to the "
        f"Assembler, even when the fake backend reports a usable one: {report}"
    )
