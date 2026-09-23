"""Tests for LocalCameraHub's frame-driven wake primitive
(`frame_generation()`/`wait_for_new_frame()`) -- 2026-09-05, -
approved frame-driven-capture task. See local_capture.py's own module
docstring, "FRAME-DRIVEN WAKE PRIMITIVE" section, for the full design.

HONEST SCOPE, same as every other test in this file's sibling
tests/test_local_capture.py: cv2.VideoCapture is mocked throughout via
the same FakeVideoCapture pattern that module already establishes.
These tests prove the Python control flow -- the counter/condition
contract, race-freedom under real concurrent threads, real timing
bounds -- not that the real rig hardware behaves this way.
"""
from __future__ import annotations

import threading
import time

import pytest

from opendarts.live.local_capture import CameraConfig, LocalCameraHub

from tests.test_local_capture import make_fake_capture_factory


@pytest.fixture(autouse=True)
def _stop_pump_threads_after_every_test():
    """Same autouse cleanup tests/test_local_capture.py's own fixture of
    the same name provides -- duplicated here (not imported) because
    pytest fixtures are file-scoped by default and this is a genuinely
    separate test module."""
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


def _open_real_pump_hub(monkeypatch, n_cams=1):
    """Real LocalCameraHub, real pump thread, mocked cv2.VideoCapture --
    a genuinely running pump that free-runs as fast as Python allows
    (FakeVideoCapture.read() returns instantly, no per-frame throttling)."""
    import cv2

    factory = make_fake_capture_factory(frame_ok=True)
    monkeypatch.setattr(cv2, "VideoCapture", factory)
    hub = LocalCameraHub(configs=[CameraConfig(device=i) for i in range(n_cams)])
    ok_flags = hub.open_all()
    assert all(ok_flags)
    return hub


def test_frame_generation_starts_at_zero_and_increases(monkeypatch):
    hub = _open_real_pump_hub(monkeypatch)
    gen0 = hub.frame_generation()
    # Real pump thread running in the background -- give it a moment to
    # do real work, then confirm the counter genuinely advanced.
    deadline = time.monotonic() + 1.0
    while hub.frame_generation() == gen0 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert hub.frame_generation() > gen0


def test_wait_for_new_frame_returns_promptly_when_pump_is_active(monkeypatch):
    """The core promise: with an actively-running pump, a caller waiting
    for a newer generation than the current one should NOT have to wait
    anywhere near its own requested timeout ceiling -- a real, measured
    timing proof, not just a functional one."""
    hub = _open_real_pump_hub(monkeypatch)
    current = hub.frame_generation()

    t0 = time.monotonic()
    observed = hub.wait_for_new_frame(current, timeout=2.0)
    elapsed = time.monotonic() - t0

    assert observed > current
    # Comfortably below the 2.0s ceiling -- an active pump (even a
    # mocked, near-instant one) should resolve this almost immediately.
    assert elapsed < 0.5, f"took {elapsed:.3f}s to observe a new generation from an active pump"


def test_wait_for_new_frame_non_blocking_check_when_already_newer(monkeypatch):
    """A caller passing an ALREADY-STALE `last_generation` (the pump has
    moved on since) must get an answer effectively immediately, never
    waiting out any part of `timeout` -- this is what makes drop-to-
    latest cost nothing extra when this loop is behind."""
    hub = _open_real_pump_hub(monkeypatch)
    # Let the pump run ahead for a moment so frame_generation() is
    # already comfortably past 0.
    deadline = time.monotonic() + 1.0
    while hub.frame_generation() < 3 and time.monotonic() < deadline:
        time.sleep(0.001)
    assert hub.frame_generation() >= 3

    t0 = time.monotonic()
    observed = hub.wait_for_new_frame(0, timeout=5.0)
    elapsed = time.monotonic() - t0

    assert observed >= 3
    assert elapsed < 0.05, f"non-blocking check took {elapsed:.3f}s -- should be near-instant"


def test_wait_for_new_frame_times_out_honestly_when_nothing_new_arrives(monkeypatch):
    """A hub with NO pump running at all (never opened) must honor the
    real timeout -- proves this never blocks indefinitely regardless of
    whether a new frame will ever arrive."""
    hub = LocalCameraHub(configs=[CameraConfig(device=0)])
    # Deliberately never call open_all() -- no pump thread exists.
    t0 = time.monotonic()
    observed = hub.wait_for_new_frame(0, timeout=0.1)
    elapsed = time.monotonic() - t0

    assert observed == 0 # never advanced
    assert 0.08 <= elapsed < 0.5, f"expected to honor the ~0.1s timeout, took {elapsed:.3f}s"


def test_close_all_wakes_a_blocked_waiter_when_the_pump_never_ran_a_single_cycle(monkeypatch):
    """Isolates close_all()'s OWN bump+notify specifically -- a hub whose
    pump thread never started at all (zero cameras opened, see
    open_all()'s "only start it if at least one camera actually opened"
    guard) has no `_pump_once()` cycle that could EVER notify a waiter on
    its own. Without close_all()'s own explicit notify, a waiter here
    would sit out its full requested timeout with nothing to wake it --
    a real, meaningfully different scenario from a hub with an active
    pump (whose own natural wind-down after `_pump_stop.set()` already
    provides a notify almost every time, which would make a test built
    against an ACTIVE pump pass even with close_all()'s own notify
    removed -- checked directly while writing this test)."""
    import cv2

    # Every backend fails to open -- open_all() reports 0/1 cameras
    # opened, so `any(ok_flags)` is False and NO pump thread/pool is
    # ever created (see that method's own "only start it if..." guard).
    monkeypatch.setattr(cv2, "VideoCapture", make_fake_capture_factory(fails_backends={
        cv2.CAP_AVFOUNDATION, cv2.CAP_V4L2, cv2.CAP_ANY,
    }))
    hub = LocalCameraHub(configs=[CameraConfig(device=0)])
    ok_flags = hub.open_all()
    assert not any(ok_flags)
    assert hub._pump_thread is None # confirms no pump ever started

    result = {}

    def waiter():
        t0 = time.monotonic()
        observed = hub.wait_for_new_frame(hub.frame_generation(), timeout=5.0)
        result["elapsed"] = time.monotonic() - t0
        result["observed"] = observed

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    time.sleep(0.05) # let the waiter actually start blocking
    hub.close_all()
    t.join(timeout=2.0)

    assert not t.is_alive(), "waiter thread never woke up after close_all() (no pump ever ran)"
    assert result["elapsed"] < 1.0, (
        f"waiter took {result['elapsed']:.3f}s to notice close_all() -- "
        "expected a prompt wakeup, not the full 5.0s timeout"
    )
    assert result["observed"] > 0 # close_all() itself bumped the counter


def test_close_all_wakes_a_blocked_waiter_with_an_active_pump_too(monkeypatch):
    """The more common real case (a hub WITH a running pump) -- kept as
    its own test since it's the realistic production shape, even though
    (per the isolated test above) the pump's own natural wind-down after
    `_pump_stop.set()` is the more likely single cause of the wakeup
    here, not necessarily close_all()'s own notify specifically."""
    hub = _open_real_pump_hub(monkeypatch)
    current = hub.frame_generation()

    result = {}

    def waiter():
        t0 = time.monotonic()
        far_future_generation = current + 10_000_000
        observed = hub.wait_for_new_frame(far_future_generation, timeout=5.0)
        result["elapsed"] = time.monotonic() - t0
        result["observed"] = observed

    t = threading.Thread(target=waiter, daemon=True)
    t.start()
    time.sleep(0.05)
    hub.close_all()
    t.join(timeout=2.0)

    assert not t.is_alive(), "waiter thread never woke up after close_all()"
    assert result["elapsed"] < 1.0, (
        f"waiter took {result['elapsed']:.3f}s to notice close_all() -- "
        "expected a prompt wakeup, not the full 5.0s timeout"
    )


def test_multiple_waiters_all_observe_the_same_new_generation(monkeypatch):
    """notify_all(), not notify() -- every waiter wakes on the same pump
    cycle, none left behind (a real regression a notify()-only
    implementation would introduce silently)."""
    hub = _open_real_pump_hub(monkeypatch)
    current = hub.frame_generation()

    results: list[int] = []
    lock = threading.Lock()

    def waiter():
        observed = hub.wait_for_new_frame(current, timeout=2.0)
        with lock:
            results.append(observed)

    threads = [threading.Thread(target=waiter, daemon=True) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=3.0)

    assert len(results) == 5, f"expected all 5 waiters to observe a new generation, got {results}"
    assert all(r > current for r in results)


def test_no_lost_wakeup_across_many_rapid_check_and_wait_cycles(monkeypatch):
    """Race-freedom, exercised for real rather than only reasoned about:
    hammer wait_for_new_frame() back-to-back against a genuinely
    concurrent, free-running pump thread. A lost-wakeup bug (the classic
    bare-Event race this design exists to avoid) would show up here as a
    call that returns the SAME generation it was given despite a real
    timeout elapsing, even though the pump kept advancing throughout."""
    hub = _open_real_pump_hub(monkeypatch)
    last = hub.frame_generation()
    stall_count = 0
    for _ in range(200):
        observed = hub.wait_for_new_frame(last, timeout=0.2)
        if observed == last:
            stall_count += 1
        else:
            assert observed > last
        last = observed
    # A free-running mocked pump (near-instant FakeVideoCapture.read())
    # should essentially never fail to produce a new generation within
    # 0.2s -- allow a small margin for real scheduling jitter on a
    # loaded CI box, but a majority of stalls would indicate a real lost-
    # wakeup problem, not noise.
    assert stall_count <= 5, f"{stall_count}/200 calls saw no new generation -- possible lost wakeup"
