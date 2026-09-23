"""The hub tap: that CameraHub's pump really fills a FrameRing, at the same
point it calls the frame sink, with the same clocks it writes onto
CameraStatus.

THREE CAMERAS, because the tap is where "per frame" and "per set" stop
being the same number. A ring bumps once per SET and retains three
FRAMES, and a one-camera test cannot tell those apart -- this project has
already shipped a units bug that passed for exactly that reason.

REAL DEFAULTING IN THE FAKE. `configs=[]` means "the default three
devices" to the real CameraHub, not "no cameras" -- a phantom-camera bug
passed once because the fakes disagreed with the real hub about that. The
hubs here are built the way a real caller builds them.
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import pytest

from opendarts.capture.frame_ring import FrameRing
from opendarts.live import local_capture as local_capture_module
from opendarts.live.local_capture import CameraConfig, CameraHub


@pytest.fixture(autouse=True)
def _close_every_hub():
    """Same purpose as tests/test_local_capture.py's own autouse fixture:
    open_all() starts a real daemon pump thread that free-runs against a
    mock, and a leaked one is a real source of timing flakiness later."""
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


class _CountingCapture:
    """A cv2.VideoCapture stand-in whose frames identify their camera AND
    their read number, so an assertion can prove a retained frame is the
    one this camera produced at that moment rather than merely "an array
    of the right shape"."""

    def __init__(self, device, backend, **_kwargs) -> None:
        self.device = device
        self.reads = 0
        self._lock = threading.Lock()

    def isOpened(self) -> bool:  # noqa: N802 -- cv2's own spelling
        return True

    def release(self) -> None:
        return None

    def set(self, prop, value) -> bool:
        return True

    def get(self, prop) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 6.0
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 4.0
        return 0.0

    def read(self):
        with self._lock:
            self.reads += 1
            n = self.reads
        # Real cameras pace the pump; this one would free-run at whatever
        # Python allows and make every test a race, so it costs a
        # millisecond per read the way a 1000fps camera would.
        time.sleep(0.001)
        arr = np.zeros((4, 6, 3), dtype=np.uint8)
        arr[:, :, 0] = int(self.device) + 1
        arr[:, :, 1] = n % 251
        return True, arr


@pytest.fixture
def three_cameras(monkeypatch):
    monkeypatch.setattr(local_capture_module.cv2, "VideoCapture", _CountingCapture)
    # These tests prove ROUTING -- which slot a frame lands in -- by tagging
    # single pixel values in a 4x6 frame. A local camera's frames now go
    # through a q50 JPEG round trip (SYNTHETIC JPEG), which smears exactly
    # those tags, so it is switched off here. What it does to a frame is
    # tested where it lives, in test_local_capture.py.
    monkeypatch.setattr(local_capture_module, "_synthesise_jpeg", lambda frame: None)


def _wait_until(predicate, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def test_the_pump_fills_the_ring_with_every_slot_and_real_pixels(three_cameras):
    ring = FrameRing(10.0)
    hub = CameraHub(configs=[], frame_ring=ring)
    assert len(hub.configs) == 3           # real defaulting: [] means three
    hub.open_all()
    assert _wait_until(lambda: ring.stats()["sets"] >= 5)
    hub.close_all()

    sets = ring.snapshot().sets
    assert sets
    for fs in sets:
        assert sorted(fs.frames) == [0, 1, 2]
        for slot, arr in fs.frames.items():
            # CONTENT. Channel 0 carries the device index, so a frame
            # filed under the wrong slot fails here rather than passing a
            # shape check.
            assert arr[0, 0, 0] == slot + 1
    # Frames are three per set, not one -- the units the ring reports.
    stats = ring.stats()
    assert stats["frames"] == stats["sets"] * 3


def test_the_ring_sees_the_same_generation_the_hub_published(three_cameras):
    ring = FrameRing(10.0)
    hub = CameraHub(configs=[], frame_ring=ring)
    hub.open_all()
    assert _wait_until(lambda: ring.stats()["sets"] >= 10)
    hub.close_all()

    generations = [fs.generation for fs in ring.snapshot().sets]
    # One bump per CYCLE, strictly increasing, no repeats -- the counter
    # the capture loop's own drop accounting reads.
    assert generations == sorted(set(generations))
    assert all(b - a == 1 for a, b in zip(generations, generations[1:]))


def test_ring_stamps_match_the_camera_status_stamps_from_the_same_cycle(three_cameras):
    """The pump samples wall and monotonic ONCE per cycle and uses that
    pair for both CameraStatus and the ring. Two independent time calls
    would describe two instants and make the ring's ordering disagree with
    the status fields -- which nothing else in the suite would catch."""
    from concurrent.futures import ThreadPoolExecutor

    ring = FrameRing(10.0)
    hub = CameraHub(configs=[], frame_ring=ring)
    # Drive the pump BY HAND rather than starting the thread, so the
    # comparison is exact instead of racing a cycle that ran in between.
    hub._caps = {i: _CountingCapture(i, None) for i in (0, 1, 2)}
    hub._pool = ThreadPoolExecutor(max_workers=3)
    try:
        # Twice, not once: the second cycle is where a stamp reused from
        # the first would show up.
        hub._pump_once()
        hub._pump_once()
    finally:
        hub._pool.shutdown(wait=True)
        hub._pool = None

    sets = ring.snapshot().sets
    assert len(sets) == 2
    for i in (0, 1, 2):
        assert hub.status[i].last_read_at == sets[-1].wall_s
        assert hub.status[i].last_read_at_monotonic == sets[-1].monotonic_s
    assert sets[0].wall_s != sets[1].wall_s
    # And the two clocks really are different clocks: wall is epoch
    # (1.7e9-ish), monotonic is uptime. A build that stamped both from one
    # call would show them equal, and every duration computed across them
    # would be wrong by decades without raising.
    assert sets[-1].wall_s > 1_600_000_000
    assert abs(sets[-1].wall_s - sets[-1].monotonic_s) > 1_000_000


def test_the_sink_and_the_ring_get_the_same_set_and_neither_disturbs_the_other(
    three_cameras,
):
    ring = FrameRing(10.0)
    seen: list[dict] = []
    hub = CameraHub(configs=[], frame_ring=ring, frame_sink=seen.append)
    hub.open_all()
    assert _wait_until(lambda: len(seen) >= 5 and ring.stats()["sets"] >= 5)
    hub.close_all()

    assert all(sorted(published) == [0, 1, 2] for published in seen)
    # The sink is handed the same arrays the ring retained -- one dict is
    # built per cycle for both, so a frame cannot reach one and not the
    # other.
    ring_by_generation = {fs.generation: fs.frames for fs in ring.snapshot().sets}
    assert ring_by_generation
    matched = 0
    for frames in ring_by_generation.values():
        for published in seen:
            if all(published.get(s) is frames[s] for s in (0, 1, 2)):
                matched += 1
                break
    assert matched > 0


def test_no_ring_attached_leaves_the_pump_behaviour_completely_alone(three_cameras):
    """The tap must cost nothing when nobody asked for it -- this is the
    capture path every scored dart goes through."""
    seen: list[dict] = []
    hub = CameraHub(configs=[], frame_sink=seen.append)
    hub.open_all()
    assert _wait_until(lambda: len(seen) >= 5)
    frames = hub.grab_all()
    hub.close_all()
    assert sorted(frames) == [0, 1, 2]
    assert hub.frame_ring is None
    assert hub.frame_ring_errors == 0


def test_a_ring_that_raises_is_logged_once_and_never_kills_the_pump(
    three_cameras, caplog,
):
    """Called MANY times on purpose: a log-spam bug in this project passed
    because no test called the function twice."""

    class _Exploding(FrameRing):
        def __init__(self) -> None:
            super().__init__(10.0)
            self.calls = 0

        def append(self, *args, **kwargs):  # type: ignore[override]
            self.calls += 1
            raise RuntimeError("boom")

    ring = _Exploding()
    hub = CameraHub(configs=[], frame_ring=ring)
    with caplog.at_level("ERROR"):
        hub.open_all()
        assert _wait_until(lambda: ring.calls >= 5)
        frames = hub.grab_all()
        hub.close_all()

    assert sorted(frames) == [0, 1, 2]          # the pump kept working
    assert hub.frame_ring_errors >= 5           # every failure counted
    assert sum("frame ring raised" in r.message for r in caplog.records) == 1


def test_a_ring_can_be_attached_and_detached_while_the_pump_runs(three_cameras):
    """Attaching costs gigabytes, so detaching must actually stop the
    spend without a restart -- otherwise the setting is a lie."""
    ring = FrameRing(10.0)
    hub = CameraHub(configs=[])
    hub.open_all()
    assert _wait_until(lambda: hub.frame_generation() > 3)
    assert ring.stats()["sets"] == 0            # nothing before attaching

    hub.set_frame_ring(ring)
    assert hub.frame_ring is ring
    assert _wait_until(lambda: ring.stats()["sets"] >= 5)

    hub.set_frame_ring(None)
    settled = ring.stats()["sets"]
    time.sleep(0.15)                            # several pump cycles
    hub.close_all()
    assert ring.stats()["sets"] == settled
    # Detaching must NOT destroy what was already retained -- someone may
    # be mid-capture on it.
    assert settled > 0
    assert ring.snapshot().sets


def test_a_disabled_ring_attached_to_a_live_hub_retains_nothing(three_cameras):
    ring = FrameRing(0.0)
    hub = CameraHub(configs=[], frame_ring=ring)
    hub.open_all()
    assert _wait_until(lambda: hub.frame_generation() > 5)
    hub.close_all()
    assert ring.stats()["sets"] == 0
    assert hub.frame_ring_errors == 0


def test_a_paused_ring_stops_growing_while_the_pump_keeps_delivering(three_cameras):
    ring = FrameRing(10.0)
    hub = CameraHub(configs=[], frame_ring=ring)
    hub.open_all()
    assert _wait_until(lambda: ring.stats()["sets"] >= 5)
    ring.pause()
    frozen = ring.stats()["sets"]
    generation = hub.frame_generation()
    assert _wait_until(lambda: hub.frame_generation() > generation + 5)
    assert ring.stats()["sets"] == frozen
    assert ring.stats()["dropped_while_paused"] > 0
    ring.resume()
    assert _wait_until(lambda: ring.stats()["sets"] > frozen)
    hub.close_all()
